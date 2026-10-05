from __future__ import annotations
import asyncio, time
import pandas as pd
from config import settings
from db import add_trade, delete_position, get_positions, set_position, stats, set_risk, get_risk
from kraken_client import KrakenTrader
from ml_model import predict, train_model

class KrakenBot:
    def __init__(self):
        self.kraken=KrakenTrader(settings); self.models={}; self.running=settings.autonomous
        self.last_train={}; self.last_scan=None; self.error=None; self.signals=[]
        self.cooldown_until=0
        if get_risk('paper_balance') is None: set_risk('paper_balance',settings.paper_start_balance)

    @staticmethod
    def _df(rows): return pd.DataFrame(rows,columns=['timestamp','open','high','low','close','volume'])
    @staticmethod
    def _position(symbol): return next((p for p in get_positions() if p['symbol']==symbol),None)

    def _can_trade(self):
        s=stats()
        if s['last_24h_pnl'] <= -settings.daily_loss_limit_usd: return False,'24h loss limit reached'
        if s['trades_24h'] >= settings.max_trades_per_day: return False,'24h trade limit reached'
        if s['consecutive_losses'] >= settings.max_consecutive_losses: return False,'loss circuit breaker active'
        if time.time() < self.cooldown_until: return False,'cooldown active'
        return True,''

    async def scan_symbol(self,symbol):
        rows=await asyncio.to_thread(self.kraken.fetch_ohlcv,symbol,settings.timeframe,settings.candles)
        df=self._df(rows); now=time.time(); state=self.models.get(symbol)
        if state is None or now-self.last_train.get(symbol,0)>=settings.train_every_seconds:
            cost=settings.round_trip_cost_pct/100+settings.slippage_buffer_pct/100
            state=await asyncio.to_thread(train_model,df,settings.forecast_bars,cost)
            self.models[symbol]=state; self.last_train[symbol]=now
        prediction=predict(state,df,settings.forecast_bars)
        if not prediction: return None
        ticker=await asyncio.to_thread(self.kraken.fetch_ticker,symbol)
        bid=float(ticker.get('bid') or ticker.get('last') or 0); ask=float(ticker.get('ask') or ticker.get('last') or 0); last=float(ticker.get('last') or 0)
        if min(bid,ask,last)<=0:return None
        spread=max(0,ask/bid-1); p=prediction['probability_up']; expected=prediction['expected_move']
        net=(2*p-1)*expected; cost=settings.round_trip_cost_pct/100+settings.slippage_buffer_pct/100+spread; score=net-cost
        return {'symbol':symbol,'price':last,'bid':bid,'ask':ask,'spread':spread,'probability_up':p,'expected_move':expected,'net_edge':net,'cost_estimate':cost,'score':score,'accuracy':state.accuracy,'samples':state.samples,'trained_at':state.trained_at}

    async def scan(self):
        results=[]
        for symbol in settings.symbols:
            try:
                r=await self.scan_symbol(symbol)
                if r: results.append(r)
            except Exception as e: print(f'SCAN ERROR {symbol}: {type(e).__name__}: {e}')
        results.sort(key=lambda x:x['score'],reverse=True); self.signals=results; self.last_scan=time.time(); return results

    async def manage_positions(self):
        for p in get_positions():
            try:
                ticker=await asyncio.to_thread(self.kraken.fetch_ticker,p['symbol']); price=float(ticker.get('bid') or ticker.get('last') or 0)
                if price<=0: continue
                age=(time.time()-p['opened_ts'])/60; reason=None
                if price<=p['stop_price']: reason='stop_loss'
                elif price>=p['target_price']: reason='take_profit'
                elif age>=settings.max_hold_minutes: reason='time_exit'
                if reason:
                    result=await asyncio.to_thread(self.kraken.market_sell,p['symbol'],p['amount']); exit_price=float(result.get('price') or price)
                    pnl=(exit_price-p['entry_price'])*p['amount']; mode='DRY_RUN' if settings.dry_run else 'LIVE'
                    add_trade({'symbol':p['symbol'],'side':'SELL','price':exit_price,'amount':p['amount'],'notional':exit_price*p['amount'],'pnl':pnl,'status':'CLOSED','mode':mode,'reason':reason})
                    if settings.dry_run:
                        bal=float(get_risk('paper_balance',settings.paper_start_balance)); set_risk('paper_balance',bal+pnl)
                    delete_position(p['symbol']); self.cooldown_until=time.time()+settings.cooldown_minutes*60
                    print(f'EXIT {p["symbol"]} {reason} pnl={pnl:.2f}')
            except Exception as e: self.error=f'POSITION ERROR: {type(e).__name__}: {e}'; print(self.error)

    async def maybe_enter_best(self):
        if not self.signals:return
        allowed,reason=self._can_trade()
        if not allowed:return
        best=self.signals[0]
        if self._position(best['symbol']) or best['accuracy']<settings.min_training_accuracy or best['probability_up']<settings.min_probability or best['expected_move']<settings.min_expected_move or best['score']<=0:return
        quote=settings.max_trade_usd
        if settings.dry_run:
            quote=min(quote,float(get_risk('paper_balance',settings.paper_start_balance))*settings.max_position_pct)
        else:
            try:
                free=await asyncio.to_thread(self.kraken.free_quote,'USD'); quote=min(quote,free*settings.max_position_pct)
            except Exception: return
        if quote<5:return
        result=await asyncio.to_thread(self.kraken.market_buy,best['symbol'],quote); entry=float(result.get('price') or best['ask']); amount=float(result.get('amount') or quote/entry); notional=entry*amount
        set_position({'symbol':best['symbol'],'entry_price':entry,'amount':amount,'notional':notional,'opened_ts':time.time(),'stop_price':entry*(1-settings.stop_loss_pct),'target_price':entry*(1+settings.take_profit_pct)})
        add_trade({'symbol':best['symbol'],'side':'BUY','price':entry,'amount':amount,'notional':notional,'status':'OPEN','mode':'DRY_RUN' if settings.dry_run else 'LIVE','reason':f'p={best["probability_up"]:.3f} score={best["score"]:.5f}'})
        print(f'ENTRY {best["symbol"]} price={entry:.4f} amount={amount:.8f}')

    async def run(self):
        while True:
            if self.running:
                try:
                    await self.manage_positions(); await self.scan(); await self.maybe_enter_best(); self.error=None
                except Exception as e: self.error=f'BOT ERROR: {type(e).__name__}: {e}'; print(self.error)
            await asyncio.sleep(max(5,settings.scan_seconds))

bot=KrakenBot()
