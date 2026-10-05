import os
import sqlite3
import time

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

        defaults = {
            "paper_balance": "1000",
            "paper_start_balance": "1000",
            "paper_invested": "0",
            "paper_realized_pnl": "0",
        }

        for key, value in defaults.items():
            conn.execute(
                """
                INSERT OR IGNORE INTO risk_state
                (key, value)
                VALUES (?, ?)
                """,
                (key, value),
            )

        conn.commit()

    finally:
        conn.close()


def add_trade(data):
    conn = db()

    try:
        conn.execute(
            """
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
            """,
            (
                time.time(),
                data.get("symbol"),
                data.get("side"),
                data.get("price"),
                data.get("amount"),
                data.get("notional"),
                data.get("pnl", 0),
                data.get("status"),
                data.get("mode"),
                data.get("reason", ""),
            ),
        )

        conn.commit()

    finally:
        conn.close()


def set_position(data):
    conn = db()

    try:
        conn.execute(
            """
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
            """,
            (
                data["symbol"],
                data["entry_price"],
                data["amount"],
                data["notional"],
                data["opened_ts"],
                data["stop_price"],
                data["target_price"],
            ),
        )

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
            (symbol,),
        )

        conn.commit()

    finally:
        conn.close()


def set_risk(key, value):
    conn = db()

    try:
        conn.execute(
            """
            INSERT OR REPLACE INTO risk_state
            (key, value)
            VALUES (?, ?)
            """,
            (key, str(value)),
        )

        conn.commit()

    finally:
        conn.close()


def get_risk(key, default=None):
    conn = db()

    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS risk_state (
                key TEXT PRIMARY KEY,
                value TEXT
            )
            """
        )

        conn.commit()

        row = conn.execute(
            """
            SELECT value
            FROM risk_state
            WHERE key=?
            """,
            (key,),
        ).fetchone()

        if row is None:
            return default

        return row["value"]

    finally:
        conn.close()


def stats():
    conn = db()

    try:
        row = conn.execute(
            """
            SELECT
                COALESCE(SUM(pnl), 0) AS pnl,
                COUNT(*) AS trades,
                COALESCE(
                    SUM(
                        CASE
                            WHEN pnl > 0
                            THEN 1
                            ELSE 0
                        END
                    ),
                    0
                ) AS wins
            FROM trades
            WHERE status='CLOSED'
            """
        ).fetchone()

        today = conn.execute(
            """
            SELECT
                COALESCE(SUM(pnl), 0) AS pnl,
                COUNT(*) AS n
            FROM trades
            WHERE status='CLOSED'
            AND ts >= ?
            """,
            (time.time() - 86400,),
        ).fetchone()

        recent = conn.execute(
            """
            SELECT pnl
            FROM trades
            WHERE status='CLOSED'
            ORDER BY ts DESC
            LIMIT 10
            """
        ).fetchall()

        open_positions = conn.execute(
            """
            SELECT
                COALESCE(SUM(notional), 0)
                AS invested
            FROM positions
            """
        ).fetchone()

    finally:
        conn.close()

    total_trades = int(
        row["trades"] or 0
    )

    wins = int(
        row["wins"] or 0
    )

    consecutive_losses = 0

    for trade in recent:
        if float(
            trade["pnl"] or 0
        ) < 0:
            consecutive_losses += 1
        else:
            break

    start_balance = float(
        get_risk(
            "paper_start_balance",
            1000,
        )
    )

    paper_balance = float(
        get_risk(
            "paper_balance",
            start_balance,
        )
    )

    realized_pnl = float(
        get_risk(
            "paper_realized_pnl",
            0,
        )
    )

    invested = float(
        open_positions["invested"] or 0
    )

    equity = (
        paper_balance
        + invested
    )

    return {
        "pnl": float(
            row["pnl"] or 0
        ),
        "trades": total_trades,
        "wins": wins,
        "losses": total_trades - wins,
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
        "consecutive_losses":
            consecutive_losses,
        "paper_start_balance":
            start_balance,
        "paper_balance":
            paper_balance,
        "paper_invested":
            invested,
        "paper_equity":
            equity,
        "realized_pnl":
            realized_pnl,
        "return_pct": (
            (
                equity - start_balance
            ) / start_balance
            if start_balance
            else 0
        ),
    }


init_db()


init_db()
