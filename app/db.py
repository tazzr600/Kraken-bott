import sqlite3
import threading
import time
from decimal import Decimal

class Database:
    def __init__(self, path, paper_start):
        self.c=sqlite3.connect(path,check_same_thread=False)
        self.c.row_factory=sqlite3.Row
        self.lock=threading.RLock()
        with self.lock:
            self.c.executescript("""
            CREATE TABLE IF NOT EXISTS events(id INTEGER PRIMARY KEY,ts INTEGER,level TEXT,event TEXT,market_id TEXT,message TEXT,data TEXT);
            CREATE TABLE IF NOT EXISTS orders(id INTEGER PRIMARY KEY,ts INTEGER,order_id TEXT,market_id TEXT,token_id TEXT,outcome TEXT,shares REAL,price REAL,spend REAL,status TEXT,paper INTEGER,raw TEXT);
            CREATE TABLE IF NOT EXISTS fills(id INTEGER PRIMARY KEY,ts INTEGER,order_id TEXT,market_id TEXT,token_id TEXT,outcome TEXT,shares REAL,price REAL,spend REAL,paper INTEGER);
            CREATE TABLE IF NOT EXISTS equity(id INTEGER PRIMARY KEY,ts INTEGER,equity REAL,cash REAL,pnl REAL);
            """)
            self.c.commit()

    def event(self,level,event,message,market_id=None,data=None):
        with self.lock:
            self.c.execute("INSERT INTO events(ts,level,event,market_id,message,data) VALUES(?,?,?,?,?,?)",(int(time.time()),level,event,market_id,message,data))
            self.c.commit()

    def order(self,**x):
        with self.lock:
            self.c.execute("INSERT INTO orders(ts,order_id,market_id,token_id,outcome,shares,price,spend,status,paper,raw) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                [x.get(k) for k in ("ts","order_id","market_id","token_id","outcome","shares","price","spend","status","paper","raw")])
            self.c.commit()

    def fill(self,**x):
        with self.lock:
            self.c.execute("INSERT INTO fills(ts,order_id,market_id,token_id,outcome,shares,price,spend,paper) VALUES(?,?,?,?,?,?,?,?,?)",
                [x.get(k) for k in ("ts","order_id","market_id","token_id","outcome","shares","price","spend","paper")])
            self.c.commit()

    def positions(self):
        with self.lock:
            rows=self.c.execute("""
                SELECT market_id,
                SUM(CASE WHEN outcome='UP' THEN shares ELSE 0 END) y,
                SUM(CASE WHEN outcome='DOWN' THEN shares ELSE 0 END) n,
                SUM(CASE WHEN outcome='UP' THEN spend ELSE 0 END) yc,
                SUM(CASE WHEN outcome='DOWN' THEN spend ELSE 0 END) nc
                FROM fills GROUP BY market_id
            """).fetchall()
        result=[]
        for r in rows:
            y=Decimal(str(r["y"] or 0));n=Decimal(str(r["n"] or 0))
            yv=Decimal(str(r["yc"] or 0))/y if y else Decimal("0")
            nv=Decimal(str(r["nc"] or 0))/n if n else Decimal("0")
            result.append({"market_id":r["market_id"],"yes_shares":float(y),"no_shares":float(n),
                "yes_cost":float(r["yc"] or 0),"no_cost":float(r["nc"] or 0),
                "unpaired":float(abs(y-n)),"paired":float(min(y,n)),
                "paired_cost":float(yv+nv) if y and n else 0.0})
        return result

    def position(self,market_id):
        return next((x for x in self.positions() if x["market_id"]==market_id),
            {"market_id":market_id,"yes_shares":0,"no_shares":0,"yes_cost":0,"no_cost":0,"unpaired":0,"paired":0,"paired_cost":0})

    def daily_spend(self):
        l=time.localtime()
        start=int(time.mktime((l.tm_year,l.tm_mon,l.tm_mday,0,0,0,0,0,-1)))
        with self.lock:
            value=self.c.execute("SELECT COALESCE(SUM(spend),0) FROM fills WHERE ts>=?",(start,)).fetchone()[0]
        return Decimal(str(value or 0))

    def trades_today(self):
        l=time.localtime()
        start=int(time.mktime((l.tm_year,l.tm_mon,l.tm_mday,0,0,0,0,0,-1)))
        with self.lock:
            return int(self.c.execute("SELECT COUNT(*) FROM fills WHERE ts>=?",(start,)).fetchone()[0])

    def recent_events(self,n=80):
        with self.lock:return [dict(x) for x in self.c.execute("SELECT * FROM events ORDER BY id DESC LIMIT ?",(n,)).fetchall()]

    def recent_orders(self,n=100):
        with self.lock:return [dict(x) for x in self.c.execute("SELECT * FROM orders ORDER BY id DESC LIMIT ?",(n,)).fetchall()]

    def equity_series(self,n=240):
        with self.lock:return [dict(x) for x in self.c.execute("SELECT ts,equity,cash,pnl FROM equity ORDER BY id DESC LIMIT ?",(n,)).fetchall()][::-1]

    def record_equity(self,equity,cash,pnl):
        with self.lock:
            self.c.execute("INSERT INTO equity(ts,equity,cash,pnl) VALUES(?,?,?,?,?)".replace("?,?,?,?,?","?,?,?,?"),
                (int(time.time()),float(equity),float(cash),float(pnl)))
            self.c.commit()
