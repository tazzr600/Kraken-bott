import logging,threading,time
from decimal import Decimal
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from .config import settings
from .db import Database
from .execution import PaperExecutor
from .market_data import PolymarketData
from .strategy import Engine

logging.basicConfig(level=logging.INFO,format="%(asctime)s %(levelname)s %(message)s")
app=FastAPI(title="Polymarket HFT Pair Engine")
db=Database(settings.database_path,settings.paper_start_balance)
stop_event=threading.Event();worker_thread=None;lock=threading.Lock();engine=None
state={"running":False,"mode":"PAPER","started_at":None,"last_scan":None,"last_error":None,"markets_discovered":0,"opportunities":0,"last_action":None,"cycle":0}

def log(level,event,message,market_id=None):
    db.event(level,event,message,market_id)
    getattr(logging,level.lower(),logging.info)(f"{event}: {message}")

def mark_to_market():
    cash=settings.paper_start_balance-db.daily_spend()
    value=Decimal("0")
    if engine:
        for p in db.positions():
            try:
                m=next((x for x in engine.data.discover() if x.id==p["market_id"]),None)
                if not m: continue
                y,n=engine.data.books(m)
                paired=Decimal(str(p["paired"]))
                up=Decimal(str(max(p["yes_shares"]-p["no_shares"],0)))
                down=Decimal(str(max(p["no_shares"]-p["yes_shares"],0)))
                value += paired
                value += up*(y.bid or Decimal("0"))
                value += down*(n.bid or Decimal("0"))
            except Exception:
                continue
    equity=cash+value
    return equity, cash, equity-settings.paper_start_balance

def worker():
    global engine
    try:
        engine=Engine(PolymarketData(settings,log),db,PaperExecutor(settings),settings,log)
        with lock:state["running"]=True;state["started_at"]=time.time()
        log("INFO","STARTED","PAPER mode")
        while not stop_event.is_set():
            began=time.time()
            try:
                opportunities=engine.cycle();markets=engine.data.discover()
                with lock:
                    state["cycle"]+=1;state["last_scan"]=time.time();state["markets_discovered"]=len(markets)
                    state["opportunities"]=len(opportunities)
                    state["last_action"]=f"{opportunities[0].action}:{opportunities[0].outcome}:{opportunities[0].market.slug}" if opportunities else "HOLD"
                    state["last_error"]=None
                equity,cash,pnl=mark_to_market()
                db.record_equity(equity,cash,pnl)
            except Exception as exc:
                with lock:state["last_error"]=repr(exc)
                log("ERROR","LOOP_ERROR",repr(exc))
            stop_event.wait(max(settings.scan_interval-(time.time()-began),0.05))
    except Exception as exc:
        with lock:state["last_error"]=repr(exc)
        log("ERROR","START_ERROR",repr(exc))
    finally:
        with lock:state["running"]=False
        log("INFO","STOPPED","worker stopped")

def start_worker():
    global worker_thread
    with lock:
        if worker_thread and worker_thread.is_alive():return False
        stop_event.clear();worker_thread=threading.Thread(target=worker,name="polymarket-engine",daemon=True);worker_thread.start();return True

@app.on_event("startup")
def startup():
    if settings.auto_start:start_worker()

@app.on_event("shutdown")
def shutdown():stop_event.set()

@app.get("/health")
def health():
    with lock:return {"ok":True,**state}

@app.get("/api/status")
def status():
    with lock:s=dict(state)
    return {**s,"daily_spend":float(db.daily_spend()),"trades_today":db.trades_today(),"max_daily_spend":float(settings.max_daily_spend),"target_trade_spend":float(settings.target_trade_spend),"series":list(settings.series)}

@app.get("/api/markets")
def markets():
    if engine is None:return []
    rows=[]
    for market in engine.data.discover():
        try:
            yes,no=engine.data.books(market);u=engine.data.underlying(market);signal=engine.model.estimate(market,u,yes,no);p=engine.position(market.id)
            rows.append({"id":market.id,"slug":market.slug,"title":market.title,"series":market.series,"spot":u.spot,"start_price":u.start_price,"perp":u.perp,"basis":u.basis,"volatility":u.volatility,"yes":{"bid":float(yes.bid or 0),"ask":float(yes.ask or 0),"size":float(yes.ask_size)},"no":{"bid":float(no.bid or 0),"ask":float(no.ask or 0),"size":float(no.ask_size)},"fair_up":signal.fair_up,"fair_down":signal.fair_down,"edge_up":signal.edge_up,"edge_down":signal.edge_down,"direction":signal.direction,"confidence":signal.confidence,"action":engine.decide(market,signal,p)[0],"yes_shares":float(p.yes_shares),"no_shares":float(p.no_shares),"paired":float(min(p.yes_shares,p.no_shares)),"pair_cost":float(p.yes_vwap+p.no_vwap) if p.yes_shares and p.no_shares else None,"reason":signal.reason})
        except Exception as exc:rows.append({"id":market.id,"slug":market.slug,"error":str(exc)})
    return rows

@app.get("/api/positions")
def positions():return db.positions()
@app.get("/api/orders")
def orders():return db.recent_orders()
@app.get("/api/events")
def events():return db.recent_events()
@app.get("/api/equity")
def equity():return db.equity_series()
@app.post("/api/control/start")
def control_start():return {"ok":start_worker(),"running":True}
@app.post("/api/control/stop")
def control_stop():stop_event.set();return {"ok":True,"running":False}
@app.get("/",response_class=HTMLResponse)
def dashboard():
    with open("app/templates/dashboard.html","r",encoding="utf-8") as f:return HTMLResponse(f.read())
