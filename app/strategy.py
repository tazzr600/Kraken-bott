import math,time
from decimal import Decimal
from .market_data import GeoBlockedError
from .models import Opportunity,Position,Signal

def normal_cdf(x):
    return .5*(1+math.erf(x/math.sqrt(2)))

class FairValueModel:
    def __init__(self,s): self.s=s
    def estimate(self,m,u,yes,no):
        remaining=max(m.end_ts-time.time(),1)
        distance=math.log(max(u.spot,1e-9)/max(u.start_price,1e-9))
        momentum=.5*u.momentum_15s+.3*u.momentum_30s+.2*u.momentum_60s
        speed=math.copysign(min(abs(u.speed)*120,.03),distance or 1)
        drift=self.s.price_distance_weight*distance+self.s.momentum_weight*momentum+self.s.speed_weight*speed+self.s.basis_weight*u.basis
        sigma=min(max(u.volatility*math.sqrt(remaining/60),self.s.volatility_floor),.20)
        up=min(max(normal_cdf(drift/max(sigma,1e-6)),.01),.99)
        down=1-up
        eu=up-float(yes.ask or 1)-float(self.s.fee_buffer)-float(self.s.slippage_buffer)
        ed=down-float(no.ask or 1)-float(self.s.fee_buffer)-float(self.s.slippage_buffer)
        if eu>=ed and eu>0: direction,selected="UP",eu
        elif ed>0: direction,selected="DOWN",ed
        else: direction,selected="NEUTRAL",max(eu,ed)
        confidence=min(1,abs(up-.5)*2)
        reason=f"FV {up:.3f}/{down:.3f} | distance {distance*100:.3f}% | momentum {momentum*100:.3f}% | vol {u.volatility*100:.3f}% | basis {u.basis*100:.3f}%"
        return Signal(up,down,direction,confidence,eu,ed,selected,abs(drift)+sigma,u.volatility,reason)

class Engine:
    def __init__(self,data,db,executor,s,log):
        self.data=data;self.db=db;self.executor=executor;self.s=s;self.log=log
        self.model=FairValueModel(s);self.failures=0;self.last_trade=0;self.last_signal={}

    def position(self,mid):
        p=self.db.position(mid)
        return Position(mid,Decimal(str(p["yes_shares"])),Decimal(str(p["no_shares"])),Decimal(str(p["yes_cost"])),Decimal(str(p["no_cost"])),Decimal(str(p["paired"])))

    def decide(self,m,s,p):
        previous=self.last_signal.get(m.id)
        self.last_signal[m.id]=s.direction
        if not p.yes_shares and not p.no_shares:
            if s.direction in {"UP","DOWN"} and s.selected_edge>=float(self.s.min_entry_edge):
                return "ENTRY",s.direction,s.selected_edge
            return "HOLD","UP",0
        if s.direction=="NEUTRAL": return "HOLD","UP",0
        if s.direction=="UP":
            if (previous=="DOWN" or p.unpaired_no>self.s.max_unpaired_shares) and p.no_shares<p.yes_shares:
                return "HEDGE","DOWN",max(s.edge_down,0)
            if s.edge_up>=float(self.s.min_add_edge): return "ADD","UP",s.edge_up
        if s.direction=="DOWN":
            if (previous=="UP" or p.unpaired_yes>self.s.max_unpaired_shares) and p.yes_shares<p.no_shares:
                return "HEDGE","UP",max(s.edge_up,0)
            if s.edge_down>=float(self.s.min_add_edge): return "ADD","DOWN",s.edge_down
        return "HOLD",s.direction,0

    def size(self,m,p,outcome,ask):
        if not ask:return Decimal("0")
        spend=min(self.s.target_trade_spend,self.s.max_trade_spend,self.s.max_daily_spend-self.db.daily_spend(),self.s.max_market_spend-p.yes_cost-p.no_cost)
        if spend<=0:return Decimal("0")
        book=self.data.book(m.yes_token if outcome=="UP" else m.no_token)
        shares=min(spend/ask,book.ask_size)
        minimum=max(m.min_size,Decimal("1"))
        if shares<minimum:return Decimal("0")
        return (shares/minimum).to_integral_value()*minimum

    def scan(self):
        if not self.data.clob_status()["available"]:
            return []
        opportunities=[]
        for market in self.data.discover():
            try:
                yes,no=self.data.books(market)
                if not yes.ask or not no.ask:continue
                if yes.ask_size<self.s.min_book_size or no.ask_size<self.s.min_book_size:continue
                underlying=self.data.underlying(market)
                signal=self.model.estimate(market,underlying,yes,no)
                p=self.position(market.id)
                action,outcome,edge=self.decide(market,signal,p)
                if action=="HOLD":continue
                ask=yes.ask if outcome=="UP" else no.ask
                size=self.size(market,p,outcome,ask)
                if size<=0:continue
                pair_cost=p.yes_vwap+p.no_vwap if p.yes_shares and p.no_shares else None
                metadata={"edge":edge,"pair_cost":float(pair_cost) if pair_cost is not None else None}
                opportunities.append(Opportunity(market,underlying,yes,no,signal,action,outcome,size,ask,size*ask,metadata))
            except GeoBlockedError:
                return []
            except Exception as exc:
                self.log("WARN","SCAN_ERROR",f"{market.slug}: {exc}",market.id)
        return sorted(opportunities,key=lambda x:x.metadata["edge"],reverse=True)

    def execute(self,o):
        if self.failures>=self.s.max_consecutive_failures:return False
        if time.time()-self.last_trade<self.s.cooldown_seconds:return False
        token=o.market.yes_token if o.outcome=="UP" else o.market.no_token
        response=self.executor.buy(token,o.shares,o.price,o.market.tick_size)
        filled=self.executor.filled_shares(response,o.shares)
        if filled<=0:
            self.failures+=1
            self.log("WARN","NO_FILL",f"{o.action} {o.outcome} produced no paper fill",o.market.id)
            return False
        spend=filled*o.price
        oid=self.executor.response_id(response)
        self.db.order(ts=int(time.time()),order_id=oid,market_id=o.market.id,token_id=token,outcome=o.outcome,shares=float(filled),price=float(o.price),spend=float(spend),status="FILLED",paper=1,raw=self.executor.raw(response))
        self.db.fill(ts=int(time.time()),order_id=oid,market_id=o.market.id,token_id=token,outcome=o.outcome,shares=float(filled),price=float(o.price),spend=float(spend),paper=1)
        self.failures=0;self.last_trade=time.time()
        p=self.position(o.market.id)
        paired_cost=p.yes_vwap+p.no_vwap if p.yes_shares and p.no_shares else Decimal("0")
        self.log("INFO",o.action,f"{filled:.2f} {o.outcome} @ {o.price:.4f} | model_edge={o.metadata['edge']:.4f} | paired_cost={paired_cost:.4f} | {o.signal.reason}",o.market.id)
        return True

    def cycle(self):
        opportunities=self.scan()
        if opportunities:self.execute(opportunities[0])
        return opportunities
