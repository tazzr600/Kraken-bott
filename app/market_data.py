import json, math, time, urllib.parse, urllib.request
from decimal import Decimal
from .models import Book, Market, Underlying
GAMMA="https://gamma-api.polymarket.com";CLOB="https://clob.polymarket.com";BINANCE="https://api.binance.com";FUTURES="https://fapi.binance.com"
class Http:
    def get_json(self,url,timeout=5):
        req=urllib.request.Request(url,headers={"User-Agent":"polymarket-paper-engine/2.0","Accept":"application/json"})
        with urllib.request.urlopen(req,timeout=timeout) as r:return json.loads(r.read().decode())
def parse_json(v,d):
    if isinstance(v,str):
        try:return json.loads(v)
        except:return d
    return v if v is not None else d
def ts(v):
    if not v:return 0
    try:
        from datetime import datetime
        return int(datetime.fromisoformat(str(v).replace("Z","+00:00")).timestamp())
    except:return 0
def slug_ts(s):
    try:return int(str(s).rsplit("-",1)[-1])
    except:return 0
class PolymarketData:
    def __init__(self,s,log):
        self.s=s;self.log=log;self.http=Http();self.market_cache=[];self.market_cache_ts=0;self.book_cache={};self.book_cache_ts={};self.underlying_cache={};self.underlying_cache_ts={}
    def discover(self):
        now=int(time.time())
        if self.market_cache and now-self.market_cache_ts<self.s.market_refresh_seconds:return self.market_cache
        found=[]
        for series in self.s.series:
            try:
                q=urllib.parse.urlencode({"series_slug":series,"closed":"false","limit":500,"order":"endDate","ascending":"true"})
                events=self.http.get_json(f"{GAMMA}/events?{q}")
                for e in events or []:
                    es=ts(e.get("eventStartTime")) or ts(e.get("startTime")) or ts(e.get("startDate")) or slug_ts(e.get("slug",""));ee=ts(e.get("endDate")) or ts(e.get("endTime"))
                    for raw in e.get("markets",[]) or []:
                        if raw.get("closed") or raw.get("active") is False or raw.get("acceptingOrders") is False:continue
                        if float(raw.get("liquidityNum") or raw.get("liquidity") or 0)<float(self.s.min_liquidity):continue
                        slug=str(raw.get("slug") or e.get("slug") or "");start=ts(raw.get("eventStartTime")) or es or ts(raw.get("startDate")) or slug_ts(slug);end=ts(raw.get("endDate")) or ee
                        if not(start and end and start<=now<end):continue
                        tokens=parse_json(raw.get("clobTokenIds"),[]);labels=[str(x).lower() for x in parse_json(raw.get("outcomes"),["Up","Down"])]
                        if len(tokens)<2:continue
                        yi=labels.index("up") if "up" in labels else 0;ni=labels.index("down") if "down" in labels else 1
                        found.append(Market(str(raw.get("id") or raw.get("conditionId") or slug),str(raw.get("conditionId") or ""),slug,str(raw.get("question") or e.get("title") or ""),series,start,end,str(tokens[yi]),str(tokens[ni]),Decimal(str(raw.get("orderMinSize") or "5")),Decimal(str(raw.get("orderPriceMinTickSize") or ".01")),Decimal(str(raw.get("liquidityNum") or raw.get("liquidity") or 0))))
            except Exception as exc:self.log("WARN","DISCOVERY_ERROR",f"{series}: {exc}")
        self.market_cache=found[:self.s.max_markets];self.market_cache_ts=now;return self.market_cache
    def book(self,token):
        now=time.time()
        if token in self.book_cache and now-self.book_cache_ts.get(token,0)<.65:return self.book_cache[token]
        raw=self.http.get_json(f"{CLOB}/book?{urllib.parse.urlencode({'token_id':token})}")
        asks=sorted([(Decimal(str(x["price"])),Decimal(str(x["size"]))) for x in raw.get("asks",[])],key=lambda x:x[0]);bids=sorted([(Decimal(str(x["price"])),Decimal(str(x["size"]))) for x in raw.get("bids",[])],key=lambda x:x[0],reverse=True)
        b=Book(token, bids[0][0] if bids else None,bids[0][1] if bids else Decimal("0"),asks[0][0] if asks else None,asks[0][1] if asks else Decimal("0"),((asks[0][0]+bids[0][0])/2) if asks and bids else None,Decimal(str(raw.get("last_trade_price"))) if raw.get("last_trade_price") else None)
        self.book_cache[token]=b;self.book_cache_ts[token]=now;return b
    def books(self,m):return self.book(m.yes_token),self.book(m.no_token)
    def symbol(self,series):
        s=series.lower();return "BTCUSDT" if s.startswith("btc") else "ETHUSDT" if s.startswith("eth") else "SOLUSDT" if s.startswith("sol") else None
    def ticker(self,base,symbol,futures=False):
        path="/fapi/v1/ticker/bookTicker" if futures else "/api/v3/ticker/bookTicker";raw=self.http.get_json(f"{base}{path}?{urllib.parse.urlencode({'symbol':symbol})}");a=float(raw.get("askPrice",0));b=float(raw.get("bidPrice",0));return (a+b)/2 if a and b else 0
    def klines(self,symbol,start=None,limit=60):
        p={"symbol":symbol,"interval":"1m","limit":limit}
        if start:p["startTime"]=int(start)
        return self.http.get_json(f"{BINANCE}/api/v3/klines?{urllib.parse.urlencode(p)}")
    def underlying(self,m):
        symbol=self.symbol(m.series)
        if not symbol:raise RuntimeError("unsupported underlying")
        now=time.time()
        if symbol in self.underlying_cache and now-self.underlying_cache_ts.get(symbol,0)<.8:return self.underlying_cache[symbol]
        spot=self.ticker(BINANCE,symbol);perp=self.ticker(FUTURES,symbol,True);basis=perp/spot-1 if spot and perp else 0
        start_rows=self.klines(symbol,m.start_ts*1000,2);start=float(start_rows[0][1]) if start_rows else spot
        rows=self.klines(symbol,limit=60);closes=[float(x[4]) for x in rows if len(x)>4];rs=[math.log(closes[i]/closes[i-1]) for i in range(1,len(closes)) if closes[i-1]>0]
        if len(rs)>2:
            mean=sum(rs)/len(rs);vol=math.sqrt(sum((x-mean)**2 for x in rs)/max(1,len(rs)-1))
        else:vol=self.s.volatility_floor
        vol=min(max(vol,self.s.volatility_floor),self.s.volatility_cap);cur=math.log(spot/start) if spot and start else 0;m60=math.log(closes[-1]/closes[-2]) if len(closes)>1 else cur;m30=(m60+math.log(closes[-2]/closes[-3]))/2 if len(closes)>2 else m60;speed=abs(cur)/max(time.time()-m.start_ts,1)
        u=Underlying(symbol,spot,perp,basis,start,vol,cur,m30,m60,speed,len(closes));self.underlying_cache[symbol]=u;self.underlying_cache_ts[symbol]=now;return u
