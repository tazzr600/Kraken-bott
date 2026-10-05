import os
import sqlite3
import time

# Use Railway's persistent volume if DB_PATH is provided.
# Otherwise use the project directory.
DB = os.getenv("DB_PATH", "jev_crypto.sqlite3")


def db():
    conn = sqlite3.connect(DB, timeout=30)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = db()

    try:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts REAL NOT NULL,
                symbol TEXT,
                side TEXT,
                price REAL,
                amount REAL,
                notional REAL,
                pnl REAL DEFAULT 0,
                status TEXT,
                mode TEXT,
                reason TEXT
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS positions (
                symbol TEXT PRIMARY KEY,
                entry_price REAL,
                amount REAL,
                notional REAL,
                opened_ts REAL,
                stop_price REAL,
                target_price REAL
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS risk_state (
                key TEXT PRIMARY KEY,
                value TEXT
            )
        """)

        conn.execute("""
            INSERT OR IGNORE INTO risk_state
            (key, value)
            VALUES ('paper_balance', '1000')
        """)

        conn.commit()

    finally:
        conn.close()


def add_trade(x):
    conn = db()

    try:
        conn.execute("""
            INSERT INTO trades (
                ts,
                symbol,
                side,
                price,
                amount,
                notional,
                pnl,
                status,
                mode,
                reason
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            time.time(),
            x.get("symbol"),
            x.get("side"),
            x.get("price"),
            x.get("amount"),
            x.get("notional"),
            x.get("pnl", 0),
            x.get("status"),
            x.get("mode"),
            x.get("reason", "")
        ))

        conn.commit()

    finally:
        conn.close()


def set_position(x):
    conn = db()

    try:
        conn.execute("""
            INSERT OR REPLACE INTO positions (
                symbol,
                entry_price,
                amount,
                notional,
                opened_ts,
                stop_price,
                target_price
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            x["symbol"],
            x["entry_price"],
            x["amount"],
            x["notional"],
            x["opened_ts"],
            x["stop_price"],
            x["target_price"]
        ))

        conn.commit()

    finally:
        conn.close()


def get_positions():
    conn = db()

    try:
        rows = conn.execute(
            "SELECT * FROM positions"
        ).fetchall()

        return [dict(row) for row in rows]

    finally:
        conn.close()


def delete_position(symbol):
    conn = db()

    try:
        conn.execute(
            "DELETE FROM positions WHERE symbol=?",
            (symbol,)
        )

        conn.commit()

    finally:
        conn.close()


def set_risk(key, value):
    conn = db()

    try:
        conn.execute("""
            INSERT OR REPLACE INTO risk_state
            (key, value)
            VALUES (?, ?)
        """, (
            key,
            str(value)
        ))

        conn.commit()

    finally:
        conn.close()


def get_risk(key, default=None):
    conn = db()

    try:
        # Extra protection: if a fresh database somehow appears,
        # create the table before querying it.
        conn.execute("""
            CREATE TABLE IF NOT EXISTS risk_state (
                key TEXT PRIMARY KEY,
                value TEXT
            )
        """)

        conn.commit()

        row = conn.execute("""
            SELECT value
            FROM risk_state
            WHERE key=?
        """, (key,)).fetchone()

        if row is None:
            return default

        return row["value"]

    finally:
        conn.close()


def stats():
    conn = db()

    try:
        row = conn.execute("""
            SELECT
                COALESCE(SUM(pnl), 0) AS pnl,
                COUNT(*) AS trades,
                COALESCE(
                    SUM(
                        CASE
                            WHEN pnl > 0 THEN 1
                            ELSE 0
                        END
                    ),
                    0
                ) AS wins
            FROM trades
            WHERE status='CLOSED'
        """).fetchone()

        today = conn.execute("""
            SELECT
                COALESCE(SUM(pnl), 0) AS pnl,
                COUNT(*) AS n
            FROM trades
            WHERE status='CLOSED'
            AND ts >= ?
        """, (
            time.time() - 86400,
        )).fetchone()

        recent = conn.execute("""
            SELECT pnl
            FROM trades
            WHERE status='CLOSED'
            ORDER BY ts DESC
            LIMIT 10
        """).fetchall()

    finally:
        conn.close()

    total_trades = int(row["trades"] or 0)
    wins = int(row["wins"] or 0)

    consecutive_losses = 0

    for trade in recent:
        if float(trade["pnl"] or 0) < 0:
            consecutive_losses += 1
        else:
            break

    paper_balance = float(
        get_risk(
            "paper_balance",
            1000
        )
    )

    return {
        "pnl": float(row["pnl"] or 0),
        "trades": total_trades,
        "wins": wins,
        "win_rate": (
            wins / total_trades
            if total_trades
            else 0
        ),
        "last_24h_pnl": float(
            today["pnl"] or 0
        ),
        "trades_24h": int(
            today["n"] or 0
        ),
        "consecutive_losses": consecutive_losses,
        "paper_balance": paper_balance
    }


# ============================================================
# INITIALIZE DATABASE IMMEDIATELY
# ============================================================

init_db()
