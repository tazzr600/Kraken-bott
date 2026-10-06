import os
import sqlite3
import time


DB = os.getenv(
    "DB_PATH",
    "jev_crypto.sqlite3",
)


def db():

    connection = sqlite3.connect(
        DB,
        timeout=30,
    )

    connection.row_factory = (
        sqlite3.Row
    )

    return connection


# ============================================================
# INITIALIZE
# ============================================================

def init_db():

    connection = db()

    try:

        connection.execute("""
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

        connection.execute("""
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

        connection.execute("""
            CREATE TABLE IF NOT EXISTS risk_state (
                key TEXT PRIMARY KEY,
                value TEXT
            )
        """)

        connection.execute("""
            CREATE TABLE IF NOT EXISTS equity_snapshots (
                ts REAL PRIMARY KEY,
                equity REAL,
                balance REAL,
                invested REAL,
                realized_pnl REAL,
                unrealized_pnl REAL,
                return_pct REAL
            )
        """)

        defaults = {

            "paper_balance":
                "1000",

            "paper_start_balance":
                "1000",

            "paper_invested":
                "0",

            "paper_realized_pnl":
                "0",

        }

        for key, value in defaults.items():

            connection.execute(
                """
                INSERT OR IGNORE INTO
                risk_state
                (key, value)
                VALUES (?, ?)
                """,
                (
                    key,
                    value,
                ),
            )

        connection.commit()

    finally:

        connection.close()


# ============================================================
# TRADES
# ============================================================

def add_trade(data):

    connection = db()

    try:

        connection.execute(
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
            VALUES (
                ?, ?, ?, ?, ?, ?,
                ?, ?, ?, ?
            )
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

        connection.commit()

    finally:

        connection.close()


# ============================================================
# POSITIONS
# ============================================================

def set_position(data):

    connection = db()

    try:

        connection.execute(
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

        connection.commit()

    finally:

        connection.close()


def get_positions():

    connection = db()

    try:

        rows = connection.execute(
            "SELECT * FROM positions"
        ).fetchall()

        return [
            dict(row)
            for row in rows
        ]

    finally:

        connection.close()


def delete_position(symbol):

    connection = db()

    try:

        connection.execute(
            """
            DELETE FROM positions
            WHERE symbol=?
            """,
            (symbol,),
        )

        connection.commit()

    finally:

        connection.close()


# ============================================================
# RISK
# ============================================================

def set_risk(
    key,
    value,
):

    connection = db()

    try:

        connection.execute(
            """
            INSERT OR REPLACE INTO
            risk_state
            (key, value)
            VALUES (?, ?)
            """,
            (
                key,
                str(value),
            ),
        )

        connection.commit()

    finally:

        connection.close()


def get_risk(
    key,
    default=None,
):

    connection = db()

    try:

        row = connection.execute(
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

        connection.close()


# ============================================================
# STATS
# ============================================================

def stats():

    connection = db()

    try:

        row = connection.execute(
            """
            SELECT
                COALESCE(SUM(pnl), 0)
                AS pnl,

                COUNT(*)
                AS trades,

                COALESCE(
                    SUM(
                        CASE
                            WHEN pnl > 0
                            THEN 1
                            ELSE 0
                        END
                    ),
                    0
                )
                AS wins

            FROM trades

            WHERE status='CLOSED'
            """
        ).fetchone()

        day = connection.execute(
            """
            SELECT
                COALESCE(SUM(pnl), 0)
                AS pnl,

                COUNT(*)
                AS n

            FROM trades

            WHERE status='CLOSED'

            AND ts >= ?
            """,
            (
                time.time() - 86400,
            ),
        ).fetchone()

        recent = connection.execute(
            """
            SELECT pnl
            FROM trades
            WHERE status='CLOSED'
            ORDER BY ts DESC
            LIMIT 20
            """
        ).fetchall()

        invested_row = connection.execute(
            """
            SELECT
                COALESCE(
                    SUM(notional),
                    0
                )
                AS invested

            FROM positions
            """
        ).fetchone()

    finally:

        connection.close()

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
        invested_row[
            "invested"
        ]
        or 0
    )

    equity = (
        paper_balance
        + invested
    )

    return {

        "pnl":
            float(
                row["pnl"] or 0
            ),

        "trades":
            total_trades,

        "wins":
            wins,

        "losses":
            total_trades - wins,

        "win_rate":
            (
                wins / total_trades
                if total_trades
                else 0
            ),

        "last_24h_pnl":
            float(
                day["pnl"] or 0
            ),

        "trades_24h":
            int(
                day["n"] or 0
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

        "return_pct":
            (
                (
                    equity
                    - start_balance
                )
                / start_balance
                if start_balance
                else 0
            ),

    }


# ============================================================
# EQUITY SNAPSHOTS
# ============================================================

def record_equity_snapshot(
    prices=None,
):

    prices = prices or {}

    positions = get_positions()

    unrealized = 0.0

    for position in positions:

        symbol = position[
            "symbol"
        ]

        price = float(
            prices.get(
                symbol,
                position[
                    "entry_price"
                ],
            )
        )

        unrealized += (
            price
            -
            float(
                position[
                    "entry_price"
                ]
            )
        ) * float(
            position["amount"]
        )

    current = stats()

    balance = float(
        current["paper_balance"]
    )

    invested = float(
        current["paper_invested"]
    )

    equity = (
        balance
        + invested
        + unrealized
    )

    start = float(
        current[
            "paper_start_balance"
        ]
    )

    return_pct = (
        (
            equity - start
        )
        / start
        if start
        else 0
    )

    connection = db()

    try:

        connection.execute(
            """
            INSERT INTO
            equity_snapshots
            (
                ts,
                equity,
                balance,
                invested,
                realized_pnl,
                unrealized_pnl,
                return_pct
            )
            VALUES (
                ?, ?, ?, ?,
                ?, ?, ?
            )
            """,
            (
                time.time(),
                equity,
                balance,
                invested,
                float(
                    current[
                        "realized_pnl"
                    ]
                ),
                unrealized,
                return_pct,
            ),
        )

        connection.commit()

    finally:

        connection.close()


def equity_history(
    limit=500,
):

    connection = db()

    try:

        rows = connection.execute(
            """
            SELECT *
            FROM equity_snapshots
            ORDER BY ts DESC
            LIMIT ?
            """,
            (
                int(limit),
            ),
        ).fetchall()

        rows = list(
            reversed(rows)
        )

        return [
            dict(row)
            for row in rows
        ]

    finally:

        connection.close()


# ============================================================
# INITIALIZE
# ============================================================

init_db()
