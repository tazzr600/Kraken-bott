import sqlite3
import threading
import time
from decimal import Decimal

class Database:
    def __init__(self, path):
        self.c = sqlite3.connect(path, check_same_thread=False)
        self.c.row_factory = sqlite3.Row
        self.lock = threading.Lock()
        with self.lock:
            self.c.executescript("""
            CREATE TABLE IF NOT EXISTS events(
                id INTEGER PRIMARY KEY, ts INTEGER, level TEXT,
                event TEXT, market_id TEXT, message TEXT
            );
            CREATE TABLE IF NOT EXISTS orders(
                id INTEGER PRIMARY KEY, ts INTEGER, order_id TEXT,
                market_id TEXT, asset_id TEXT, side TEXT, shares REAL,
                price REAL, status TEXT, paper INTEGER, raw TEXT
            );
            CREATE TABLE IF NOT EXISTS fills(
                id INTEGER PRIMARY KEY, ts INTEGER, order_id TEXT,
                market_id TEXT, asset_id TEXT, outcome TEXT, shares REAL,
                price REAL, spend REAL, paper INTEGER
            );
            """)
            self.c.commit()

    def event(self, *x):
        with self.lock:
            self.c.execute(
                "INSERT INTO events(ts,level,event,market_id,message) VALUES(?,?,?,?,?)",
                (int(time.time()), *x),
            )
            self.c.commit()

    def order(self, **x):
        with self.lock:
            self.c.execute(
                "INSERT INTO orders(ts,order_id,market_id,asset_id,side,shares,price,status,paper,raw) VALUES(?,?,?,?,?,?,?,?,?,?)",
                [x.get(k) for k in ("ts","order_id","market_id","asset_id","side","shares","price","status","paper","raw")],
            )
            self.c.commit()

    def fill(self, **x):
        with self.lock:
            self.c.execute(
                "INSERT INTO fills(ts,order_id,market_id,asset_id,outcome,shares,price,spend,paper) VALUES(?,?,?,?,?,?,?,?,?)",
                [x.get(k) for k in ("ts","order_id","market_id","asset_id","outcome","shares","price","spend","paper")],
            )
            self.c.commit()

    def daily_spend(self):
        start = int(time.time()) - int(time.time()) % 86400
        with self.lock:
            return Decimal(str(
                self.c.execute(
                    "SELECT COALESCE(SUM(spend),0) FROM fills WHERE ts>=?",
                    (start,),
                ).fetchone()[0]
            ))

    def inventory(self):
        with self.lock:
            rows = self.c.execute("""
                SELECT market_id, MAX(market_id) market,
                SUM(CASE WHEN outcome='YES' THEN shares ELSE 0 END) yes,
                SUM(CASE WHEN outcome='NO' THEN shares ELSE 0 END) no,
                SUM(CASE WHEN outcome='YES' THEN spend ELSE 0 END) yes_cost,
                SUM(CASE WHEN outcome='NO' THEN spend ELSE 0 END) no_cost
                FROM fills GROUP BY market_id
            """).fetchall()
            return [dict(r) for r in rows]

    def events(self, n=50):
        with self.lock:
            return [dict(r) for r in self.c.execute(
                "SELECT * FROM events ORDER BY id DESC LIMIT ?", (n,)
            ).fetchall()]
