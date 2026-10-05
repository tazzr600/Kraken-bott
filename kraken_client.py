from __future__ import annotations

import time
import ccxt

class KrakenTrader:
    def __init__(self, settings):
        self.s = settings
        self.exchange = ccxt.kraken({
            "apiKey": settings.kraken_api_key,
            "secret": settings.kraken_api_secret,
            "enableRateLimit": True,
            "timeout": 20000,
        })
        self.markets_loaded = False

    def load_markets(self):
        if not self.markets_loaded:
            self.exchange.load_markets()
            self.markets_loaded = True

    def fetch_ohlcv(self, symbol, timeframe, limit):
        self.load_markets()
        return self.exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)

    def fetch_ticker(self, symbol):
        self.load_markets()
        return self.exchange.fetch_ticker(symbol)

    def fetch_order_book(self, symbol, limit=10):
        self.load_markets()
        return self.exchange.fetch_order_book(symbol, limit=limit)

    def free_quote(self, quote="USD"):
        self.load_markets()
        balance = self.exchange.fetch_balance()
        return float((balance.get("free") or {}).get(quote, 0) or 0)

    def market_buy(self, symbol, quote_usd):
        self.load_markets()
        ticker = self.fetch_ticker(symbol)
        ask = float(ticker.get("ask") or ticker.get("last") or 0)
        if ask <= 0:
            raise RuntimeError("No valid ask price")

        amount = quote_usd / ask
        amount = float(self.exchange.amount_to_precision(symbol, amount))
        if amount <= 0:
            raise RuntimeError("Order amount rounded to zero")

        if self.s.dry_run or not self.s.live_trading:
            return {
                "id": f"DRY-BUY-{int(time.time()*1000)}",
                "status": "DRY_RUN",
                "symbol": symbol,
                "side": "buy",
                "amount": amount,
                "price": ask,
                "cost": amount * ask,
            }

        return self.exchange.create_market_buy_order(symbol, amount)

    def market_sell(self, symbol, amount):
        self.load_markets()
        amount = float(self.exchange.amount_to_precision(symbol, amount))
        if amount <= 0:
            raise RuntimeError("Sell amount rounded to zero")

        ticker = self.fetch_ticker(symbol)
        bid = float(ticker.get("bid") or ticker.get("last") or 0)

        if self.s.dry_run or not self.s.live_trading:
            return {
                "id": f"DRY-SELL-{int(time.time()*1000)}",
                "status": "DRY_RUN",
                "symbol": symbol,
                "side": "sell",
                "amount": amount,
                "price": bid,
                "cost": amount * bid,
            }

        return self.exchange.create_market_sell_order(symbol, amount)
