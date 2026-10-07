import logging
import threading
import time
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from .config import settings
from .db import Database
from .polymarket import PM
from .strategy import Engine

logging.basicConfig(level=logging.INFO)
app = FastAPI(title="Polymarket Pair Bot")
db = Database(settings.database_path)
stop = threading.Event()
state = {
    "running": False,
    "mode": "LIVE" if settings.live_trading else "PAPER",
    "opportunities": 0,
    "last_scan": None,
    "last_error": None,
}

def log(level, event, message, market_id=None):
    db.event(level, event, message, market_id)
    getattr(logging, level.lower(), logging.info)(f"{event}: {message}")

def worker():
    api = None
    try:
        api = PM()
        engine = Engine(api, db, log)
        state["running"] = True
        log("INFO", "STARTED", f"Mode={state['mode']}")
        while not stop.is_set():
            try:
                ops = engine.scan()
                state["opportunities"] = len(ops)
                state["last_scan"] = time.time()
                state["last_error"] = None
                if ops:
                    engine.execute(ops[0])
            except Exception as e:
                state["last_error"] = repr(e)
                log("ERROR", "LOOP_ERROR", repr(e))
            stop.wait(settings.scan_interval)
    except Exception as e:
        state["last_error"] = repr(e)
        log("ERROR", "START_ERROR", repr(e))
    finally:
        state["running"] = False
        if api:
            api.close()

@app.on_event("startup")
def startup():
    threading.Thread(target=worker, daemon=True).start()

@app.on_event("shutdown")
def shutdown():
    stop.set()

@app.get("/health")
def health():
    return {"ok": True, **state}

@app.get("/api/status")
def status():
    return {
        **state,
        "daily_spend": str(db.daily_spend()),
        "min_net_edge": str(settings.min_net_edge),
    }

@app.get("/api/inventory")
def inventory():
    return db.inventory()

@app.get("/api/events")
def events():
    return db.events()

@app.get("/api/stop")
def stop_bot():
    stop.set()
    return JSONResponse({"ok": True})

@app.get("/", response_class=HTMLResponse)
def dashboard():
    html = """
    <!doctype html><html><head><meta name="viewport" content="width=device-width,initial-scale=1">
    <title>Polymarket Pair Bot</title>
    <style>
    body{background:#07090d;color:#e8edf5;font-family:system-ui;margin:0}
    .wrap{max-width:1100px;margin:auto;padding:24px}
    .grid{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:20px 0}
    .card{background:#0e131b;border:1px solid #202b3a;border-radius:12px;padding:16px}
    .value{font-size:25px;font-weight:700;margin-top:6px}
    table{width:100%;border-collapse:collapse}td,th{padding:8px;border-bottom:1px solid #202b3a;text-align:left}
    pre{white-space:pre-wrap;color:#aeb8c8}
    @media(max-width:800px){.grid{grid-template-columns:repeat(2,1fr)}}
    </style></head><body><div class="wrap">
    <h1>POLYMARKET PAIR BOT</h1><p>Two-sided YES + NO inventory engine</p>
    <div class="grid"><div class="card">MODE<div id="mode" class="value">—</div></div>
    <div class="card">STATUS<div id="run" class="value">—</div></div>
    <div class="card">OPPORTUNITIES<div id="opp" class="value">—</div></div>
    <div class="card">DAILY SPEND<div id="spend" class="value">—</div></div></div>
    <div class="card"><h3>Inventory</h3><div id="inv">Loading…</div></div><br>
    <div class="card"><h3>Events</h3><pre id="events">Loading…</pre></div></div>
    <script>
    async function refresh(){
      let s=await (await fetch('/api/status')).json();
      mode.textContent=s.mode;run.textContent=s.running?'RUNNING':'STOPPED';opp.textContent=s.opportunities;spend.textContent='$'+s.daily_spend;
      let i=await (await fetch('/api/inventory')).json();
      inv.innerHTML=i.length?'<table><tr><th>Market</th><th>YES</th><th>NO</th><th>YES Cost</th><th>NO Cost</th></tr>'+
      i.map(x=>'<tr><td>'+x.market_id+'</td><td>'+Number(x.yes).toFixed(2)+'</td><td>'+Number(x.no).toFixed(2)+'</td><td>$'+Number(x.yes_cost).toFixed(2)+'</td><td>$'+Number(x.no_cost).toFixed(2)+'</td></tr>').join('')+'</table>':'No inventory';
      let e=await (await fetch('/api/events')).json();events.textContent=e.map(x=>new Date(x.ts*1000).toLocaleTimeString()+' ['+x.level+'] '+x.event+': '+x.message).join('\n');
    } refresh();setInterval(refresh,3000);
    </script></body></html>
    """
    return HTMLResponse(html)
