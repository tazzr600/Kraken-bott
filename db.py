import sqlite3
import time

DB = "jev_crypto.sqlite3"

def db():
    c = sqlite3.connect(DB, timeout=10)
    c.row_factory = sqlite3.Row
    return c

def init_db():
    c = db()
    c.execute("""CREATE TABLE IF NOT EXISTS trades(
        id INTEGER PRIMARY KEY, ts REAL, symbol TEXT, side TEXT, price REAL,
        amount REAL, notional REAL, pnl REAL DEFAULT 0, status TEXT, mode TEXT, reason TEXT)""")
    c.execute("""CREATE TABLE IF NOT EXISTS positions(
        symbol TEXT PRIMARY KEY, entry_price REAL, amount REAL, notional REAL,
        opened_ts REAL, stop_price REAL, target_price REAL)""")
    c.execute("""CREATE TABLE IF NOT EXISTS risk_state(
        key TEXT PRIMARY KEY, value TEXT)""")
    c.execute("INSERT OR IGNORE INTO risk_state(key,value) VALUES('paper_balance','1000')")
    c.commit(); c.close()

def add_trade(x):
    c=db(); c.execute("INSERT INTO trades(ts,symbol,side,price,amount,notional,pnl,status,mode,reason) VALUES(?,?,?,?,?,?,?,?,?,?)",
        (time.time(),x.get('symbol'),x.get('side'),x.get('price'),x.get('amount'),x.get('notional'),x.get('pnl',0),x.get('status'),x.get('mode'),x.get('reason',''))); c.commit(); c.close()

def set_position(x):
    c=db(); c.execute("INSERT OR REPLACE INTO positions(symbol,entry_price,amount,notional,opened_ts,stop_price,target_price) VALUES(?,?,?,?,?,?,?)",
        (x['symbol'],x['entry_price'],x['amount'],x['notional'],x['opened_ts'],x['stop_price'],x['target_price'])); c.commit(); c.close()

def get_positions():
    c=db(); rows=c.execute('SELECT * FROM positions').fetchall(); c.close(); return [dict(r) for r in rows]

def delete_position(symbol):
    c=db(); c.execute('DELETE FROM positions WHERE symbol=?',(symbol,)); c.commit(); c.close()

def set_risk(key,value):
    c=db(); c.execute('INSERT OR REPLACE INTO risk_state(key,value) VALUES(?,?)',(key,str(value))); c.commit(); c.close()

def get_risk(key,default=None):
    c=db(); r=c.execute('SELECT value FROM risk_state WHERE key=?',(key,)).fetchone(); c.close(); return default if not r else r['value']

def stats():
    c=db()
    row=c.execute("SELECT COALESCE(SUM(pnl),0) pnl,COUNT(*) trades,COALESCE(SUM(CASE WHEN pnl>0 THEN 1 ELSE 0 END),0) wins FROM trades WHERE status='CLOSED'").fetchone()
    today=c.execute("SELECT COALESCE(SUM(pnl),0) pnl,COUNT(*) n FROM trades WHERE status='CLOSED' AND ts>=?",(time.time()-86400,)).fetchone()
    losses=c.execute("SELECT pnl FROM trades WHERE status='CLOSED' ORDER BY ts DESC LIMIT 10").fetchall(); c.close()
    n=int(row['trades'] or 0); consecutive=0
    for r in losses:
        if float(r['pnl'] or 0)<0: consecutive+=1
        else: break
    return {'pnl':float(row['pnl'] or 0),'trades':n,'wins':int(row['wins'] or 0),'win_rate':float(row['wins'])/n if n else 0,'last_24h_pnl':float(today['pnl'] or 0),'trades_24h':int(today['n'] or 0),'consecutive_losses':consecutive,'paper_balance':float(get_risk('paper_balance',1000))}
