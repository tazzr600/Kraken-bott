from __future__ import annotations

import os
import sqlite3
import time
from contextlib import contextmanager
from typing import Any, Dict, List, Optional


# ============================================================
# DATABASE CONFIG
# ============================================================

DB = os.getenv(
    "DB_PATH",
    "jev_crypto.sqlite3",
)


# ============================================================
# CONNECTION
# ============================================================

def db() -> sqlite3.Connection:
    """
    Create a SQLite connection.

    WAL mode + busy timeout help prevent database-lock problems
    when the FastAPI dashboard and bot worker access SQLite at
    the same time.
    """

    connection = sqlite3.connect(
        DB,
        timeout=30,
        isolation_level=None,
        check_same_thread=False,
    )

    connection.row_factory = sqlite3.Row

    # SQLite reliability settings
    connection.execute("PRAGMA busy_timeout = 30000")
    connection.execute("PRAGMA journal_mode = WAL")
    connection.execute("PRAGMA synchronous = NORMAL")
    connection.execute("PRAGMA foreign_keys = ON")

    return connection


@contextmanager
def transaction():
    """
    Safe transaction helper.
    """

    connection = db()

    try:
        connection.execute("BEGIN")
        yield connection
        connection.execute("COMMIT")

    except Exception:
        try:
            connection.execute("ROLLBACK")
        except Exception:
            pass

        raise

    finally:
        connection.close()


# ============================================================
# HELPERS
# ============================================================

def _column_exists(
    connection: sqlite3.Connection,
    table: str,
    column: str,
) -> bool:

    rows = connection.execute(
        f"PRAGMA table_info({table})"
    ).fetchall()

    return any(
        row["name"] == column
        for row in rows
    )


def _ensure_column(
    connection: sqlite3.Connection,
    table: str,
    column: str,
    definition: str,
) -> None:

    if not _column_exists(
        connection,
        table,
        column,
    ):

        connection.execute(
            f"""
            ALTER TABLE {table}
            ADD COLUMN {column} {definition}
            """
        )


def _safe_float(
    value: Any,
    default: float = 0.0,
) -> float:

    try:
        if value is None:
            return default

        return float(value)

    except (
        TypeError,
        ValueError,
    ):

        return default


def _safe_int(
    value: Any,
    default: int = 0,
) -> int:

    try:
        if value is None:
            return default

        return int(value)

    except (
        TypeError,
        ValueError,
    ):

        return default


# ============================================================
# INITIALIZE DATABASE
# ============================================================

def init_db() -> None:

    connection = db()

    try:

        # ----------------------------------------------------
        # TRADES
        # ----------------------------------------------------

        connection.execute(
            """
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
            """
        )

        # ----------------------------------------------------
        # POSITIONS
        # ----------------------------------------------------

        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS positions (
                symbol TEXT PRIMARY KEY,

                entry_price REAL,

                amount REAL,

                notional REAL,

                opened_ts REAL,

                stop_price REAL,

                target_price REAL
            )
            """
        )

        # ----------------------------------------------------
        # RISK STATE
        # ----------------------------------------------------

        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS risk_state (
                key TEXT PRIMARY KEY,

                value TEXT
            )
            """
        )

        # ----------------------------------------------------
        # EQUITY SNAPSHOTS
        # ----------------------------------------------------

        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS equity_snapshots (
                ts REAL PRIMARY KEY,

                equity REAL,

                balance REAL,

                invested REAL,

                realized_pnl REAL,

                unrealized_pnl REAL,

                return_pct REAL
            )
            """
        )

        # ----------------------------------------------------
        # MIGRATE OLD DATABASES
        # ----------------------------------------------------

        # Trades
        _ensure_column(
            connection,
            "trades",
            "reason",
            "TEXT",
        )

        _ensure_column(
            connection,
            "trades",
            "mode",
            "TEXT",
        )

        _ensure_column(
            connection,
            "trades",
            "status",
            "TEXT",
        )

        # Positions
        _ensure_column(
            connection,
            "positions",
            "stop_price",
            "REAL",
        )

        _ensure_column(
            connection,
            "positions",
            "target_price",
            "REAL",
        )

        # ----------------------------------------------------
        # INDEXES
        # ----------------------------------------------------

        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS
            idx_trades_ts
            ON trades(ts)
            """
        )

        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS
            idx_trades_status
            ON trades(status)
            """
        )

        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS
            idx_trades_symbol
            ON trades(symbol)
            """
        )

        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS
            idx_equity_ts
            ON equity_snapshots(ts)
            """
        )

        # ----------------------------------------------------
        # DEFAULT PAPER ACCOUNT
        # ----------------------------------------------------

        defaults = {

            "paper_balance": "1000",

            "paper_start_balance": "1000",

            "paper_invested": "0",

            "paper_realized_pnl": "0",

            "trades_today": "0",

            "consecutive_losses": "0",

            "last_trade_ts": "0",

            "daily_start_equity": "1000",

            "daily_start_date": "",

        }

        for key, value in defaults.items():

            connection.execute(
                """
                INSERT OR IGNORE INTO
                risk_state
                (
                    key,
                    value
                )
                VALUES (?, ?)
                """,
                (
                    key,
                    value,
                ),
            )

        # ----------------------------------------------------
        # COMMIT
        # ----------------------------------------------------

        connection.commit()

    finally:

        connection.close()


# ============================================================
# TRADES
# ============================================================

def add_trade(
    data: Dict[str, Any],
) -> int:
    """
    Record a trade.

    Returns:
        SQLite trade ID.
    """

    connection = db()

    try:

        cursor = connection.execute(
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
                _safe_float(
                    data.get(
                        "ts",
                        time.time(),
                    ),
                    time.time(),
                ),

                data.get("symbol"),

                data.get("side"),

                _safe_float(
                    data.get("price")
                ),

                _safe_float(
                    data.get("amount")
                ),

                _safe_float(
                    data.get("notional")
                ),

                _safe_float(
                    data.get("pnl")
                ),

                data.get("status"),

                data.get("mode"),

                data.get(
                    "reason",
                    "",
                ),
            ),
        )

        connection.commit()

        return int(
            cursor.lastrowid
        )

    finally:

        connection.close()


def get_trades(
    limit: int = 100,
    symbol: Optional[str] = None,
) -> List[Dict[str, Any]]:

    connection = db()

    try:

        limit = max(
            1,
            min(
                int(limit),
                5000,
            ),
        )

        if symbol:

            rows = connection.execute(
                """
                SELECT *
                FROM trades
                WHERE symbol = ?
                ORDER BY ts DESC
                LIMIT ?
                """,
                (
                    symbol,
                    limit,
                ),
            ).fetchall()

        else:

            rows = connection.execute(
                """
                SELECT *
                FROM trades
                ORDER BY ts DESC
                LIMIT ?
                """,
                (
                    limit,
                ),
            ).fetchall()

        return [
            dict(row)
            for row in rows
        ]

    finally:

        connection.close()


# ============================================================
# POSITIONS
# ============================================================

def set_position(
    data: Dict[str, Any],
) -> None:

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
            VALUES (
                ?, ?, ?, ?, ?, ?, ?
            )
            """,
            (
                data["symbol"],

                _safe_float(
                    data.get(
                        "entry_price"
                    )
                ),

                _safe_float(
                    data.get(
                        "amount"
                    )
                ),

                _safe_float(
                    data.get(
                        "notional"
                    )
                ),

                _safe_float(
                    data.get(
                        "opened_ts",
                        time.time(),
                    ),
                    time.time(),
                ),

                _safe_float(
                    data.get(
                        "stop_price"
                    )
                ),

                _safe_float(
                    data.get(
                        "target_price"
                    )
                ),
            ),
        )

        connection.commit()

    finally:

        connection.close()


def get_position(
    symbol: Optional[str] = None,
) -> Optional[Dict[str, Any]]:

    connection = db()

    try:

        if symbol:

            row = connection.execute(
                """
                SELECT *
                FROM positions
                WHERE symbol = ?
                LIMIT 1
                """,
                (
                    symbol,
                ),
            ).fetchone()

        else:

            row = connection.execute(
                """
                SELECT *
                FROM positions
                ORDER BY opened_ts ASC
                LIMIT 1
                """
            ).fetchone()

        if row is None:
            return None

        return dict(row)

    finally:

        connection.close()


def get_positions() -> List[Dict[str, Any]]:

    connection = db()

    try:

        rows = connection.execute(
            """
            SELECT *
            FROM positions
            ORDER BY opened_ts ASC
            """
        ).fetchall()

        return [
            dict(row)
            for row in rows
        ]

    finally:

        connection.close()


def delete_position(
    symbol: str,
) -> None:

    connection = db()

    try:

        connection.execute(
            """
            DELETE FROM positions
            WHERE symbol = ?
            """,
            (
                symbol,
            ),
        )

        connection.commit()

    finally:

        connection.close()


def clear_positions() -> None:

    connection = db()

    try:

        connection.execute(
            "DELETE FROM positions"
        )

        connection.commit()

    finally:

        connection.close()


# ============================================================
# RISK STATE
# ============================================================

def set_risk(
    key: str,
    value: Any,
) -> None:

    connection = db()

    try:

        connection.execute(
            """
            INSERT OR REPLACE INTO
            risk_state
            (
                key,
                value
            )
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
    key: str,
    default: Any = None,
) -> Any:

    connection = db()

    try:

        row = connection.execute(
            """
            SELECT value
            FROM risk_state
            WHERE key = ?
            """,
            (
                key,
            ),
        ).fetchone()

        if row is None:
            return default

        return row["value"]

    finally:

        connection.close()


def get_risk_float(
    key: str,
    default: float = 0.0,
) -> float:

    return _safe_float(
        get_risk(
            key,
            default,
        ),
        default,
    )


def get_risk_int(
    key: str,
    default: int = 0,
) -> int:

    return _safe_int(
        get_risk(
            key,
            default,
        ),
        default,
    )


# ============================================================
# PAPER ACCOUNT
# ============================================================

def get_paper_balance() -> float:

    return get_risk_float(
        "paper_balance",
        1000,
    )


def set_paper_balance(
    value: float,
) -> None:

    set_risk(
        "paper_balance",
        max(
            0.0,
            float(value),
        ),
    )


def get_paper_start_balance() -> float:

    return get_risk_float(
        "paper_start_balance",
        1000,
    )


def get_paper_invested() -> float:

    # Prefer the actual open positions.
    positions = get_positions()

    if positions:

        return sum(
            _safe_float(
                position.get(
                    "notional"
                )
            )
            for position in positions
        )

    # Fall back to stored state.
    return get_risk_float(
        "paper_invested",
        0,
    )


def set_paper_invested(
    value: float,
) -> None:

    set_risk(
        "paper_invested",
        max(
            0.0,
            float(value),
        ),
    )


def get_realized_pnl() -> float:

    return get_risk_float(
        "paper_realized_pnl",
        0,
    )


def set_realized_pnl(
    value: float,
) -> None:

    set_risk(
        "paper_realized_pnl",
        float(value),
    )


def add_realized_pnl(
    pnl: float,
) -> float:

    current = get_realized_pnl()

    new_value = (
        current
        + float(pnl)
    )

    set_realized_pnl(
        new_value
    )

    return new_value


# ============================================================
# TRADE/RISK COUNTERS
# ============================================================

def get_trades_today() -> int:

    return get_risk_int(
        "trades_today",
        0,
    )


def set_trades_today(
    value: int,
) -> None:

    set_risk(
        "trades_today",
        max(
            0,
            int(value),
        ),
    )


def increment_trades_today() -> int:

    value = (
        get_trades_today()
        + 1
    )

    set_trades_today(
        value
    )

    return value


def get_consecutive_losses() -> int:

    return get_risk_int(
        "consecutive_losses",
        0,
    )


def set_consecutive_losses(
    value: int,
) -> None:

    set_risk(
        "consecutive_losses",
        max(
            0,
            int(value),
        ),
    )


def register_closed_trade(
    pnl: float,
) -> None:

    """
    Update risk counters after a completed trade.
    """

    increment_trades_today()

    set_risk(
        "last_trade_ts",
        time.time(),
    )

    if pnl < 0:

        set_consecutive_losses(
            get_consecutive_losses()
            + 1
        )

    else:

        set_consecutive_losses(
            0
        )


# ============================================================
# STATS
# ============================================================

def stats() -> Dict[str, Any]:

    connection = db()

    try:

        # ----------------------------------------------------
        # ALL CLOSED TRADES
        # ----------------------------------------------------

        row = connection.execute(
            """
            SELECT

                COALESCE(
                    SUM(pnl),
                    0
                ) AS pnl,

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
                ) AS wins,

                COALESCE(
                    SUM(
                        CASE
                            WHEN pnl < 0
                            THEN 1
                            ELSE 0
                        END
                    ),
                    0
                ) AS losses

            FROM trades

            WHERE UPPER(
                COALESCE(status, '')
            ) = 'CLOSED'
            """
        ).fetchone()

        # ----------------------------------------------------
        # CURRENT CALENDAR DAY
        # ----------------------------------------------------
        #
        # Daily risk is based on completed SELL trades since
        # local midnight. BUY entries are not completed trades.

        local_now = time.localtime()
        day_start = time.mktime((
            local_now.tm_year,
            local_now.tm_mon,
            local_now.tm_mday,
            0, 0, 0,
            local_now.tm_wday,
            local_now.tm_yday,
            local_now.tm_isdst,
        ))

        day = connection.execute(
            """
            SELECT
                COALESCE(SUM(pnl), 0) AS pnl,
                COUNT(*) AS n
            FROM trades
            WHERE UPPER(COALESCE(status, '')) = 'CLOSED'
              AND UPPER(COALESCE(side, '')) = 'SELL'
              AND ts >= ?
            """,
            (day_start,),
        ).fetchone()

        # Preserve the legacy rolling-24h dashboard metrics.
        rolling = connection.execute(
            """
            SELECT
                COALESCE(SUM(pnl), 0) AS pnl,
                COUNT(*) AS n
            FROM trades
            WHERE UPPER(COALESCE(status, '')) = 'CLOSED'
              AND UPPER(COALESCE(side, '')) = 'SELL'
              AND ts >= ?
            """,
            (time.time() - 86400,),
        ).fetchone()

        # ----------------------------------------------------
        # RECENT CLOSED TRADES
        # ----------------------------------------------------

        recent = connection.execute(
            """
            SELECT
                pnl
            FROM trades

            WHERE UPPER(
                COALESCE(status, '')
            ) = 'CLOSED'

            ORDER BY ts DESC

            LIMIT 100
            """
        ).fetchall()

    finally:

        connection.close()

    # --------------------------------------------------------
    # COUNTS
    # --------------------------------------------------------

    total_trades = _safe_int(
        row["trades"]
    )

    wins = _safe_int(
        row["wins"]
    )

    losses = _safe_int(
        row["losses"]
    )

    # --------------------------------------------------------
    # CONSECUTIVE LOSSES
    # --------------------------------------------------------

    consecutive_losses = 0

    for trade in recent:

        pnl = _safe_float(
            trade["pnl"]
        )

        if pnl < 0:

            consecutive_losses += 1

        else:

            break

    # --------------------------------------------------------
    # PAPER ACCOUNT
    # --------------------------------------------------------

    start_balance = (
        get_paper_start_balance()
    )

    paper_balance = (
        get_paper_balance()
    )

    realized_pnl = (
        get_realized_pnl()
    )

    positions = get_positions()

    invested = sum(
        _safe_float(
            position.get(
                "notional"
            )
        )
        for position in positions
    )

    # Keep stored invested value synchronized.
    stored_invested = (
        get_paper_invested()
    )

    if abs(
        stored_invested
        - invested
    ) > 0.0000001:

        set_paper_invested(
            invested
        )

    # --------------------------------------------------------
    # EQUITY
    #
    # Cash balance + capital currently invested.
    # Unrealized P&L is NOT added here because the open
    # position's notional is already included.
    # --------------------------------------------------------

    equity = (
        paper_balance
        + invested
    )

    return_pct = (
        (
            equity
            - start_balance
        )
        / start_balance
        if start_balance > 0
        else 0.0
    )

    return {

        "pnl":
            _safe_float(
                row["pnl"]
            ),

        "trades":
            total_trades,

        "wins":
            wins,

        "losses":
            losses,

        "win_rate":
            (
                wins / total_trades
                if total_trades > 0
                else 0.0
            ),

        "daily_pnl":
            _safe_float(
                day["pnl"]
            ),

        "trades_today":
            _safe_int(
                day["n"]
            ),

        # Retain legacy fields for dashboard compatibility.
        "last_24h_pnl":
            _safe_float(
                rolling["pnl"]
            ),

        "trades_24h":
            _safe_int(
                rolling["n"]
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
            return_pct,

    }


# ============================================================
# EQUITY SNAPSHOTS
# ============================================================

def record_equity_snapshot(
    prices: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:

    prices = prices or {}

    positions = get_positions()

    unrealized = 0.0

    invested = 0.0

    # --------------------------------------------------------
    # CALCULATE OPEN POSITION VALUE
    # --------------------------------------------------------

    for position in positions:

        symbol = position[
            "symbol"
        ]

        entry_price = _safe_float(
            position.get(
                "entry_price"
            )
        )

        amount = _safe_float(
            position.get(
                "amount"
            )
        )

        notional = _safe_float(
            position.get(
                "notional"
            )
        )

        invested += notional

        price = _safe_float(
            prices.get(
                symbol,
                entry_price,
            ),
            entry_price,
        )

        unrealized += (
            price
            - entry_price
        ) * amount

    current = stats()

    balance = _safe_float(
        current[
            "paper_balance"
        ]
    )

    start = _safe_float(
        current[
            "paper_start_balance"
        ]
    )

    # --------------------------------------------------------
    # MARK-TO-MARKET EQUITY
    #
    # Cash + current market value of positions.
    #
    # Current market value =
    # entry notional + unrealized P&L
    # --------------------------------------------------------

    equity = (
        balance
        + invested
        + unrealized
    )

    return_pct = (
        (
            equity
            - start
        )
        / start
        if start > 0
        else 0.0
    )

    timestamp = time.time()

    connection = db()

    try:

        connection.execute(
            """
            INSERT INTO equity_snapshots (
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
                timestamp,

                equity,

                balance,

                invested,

                _safe_float(
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

    return {

        "ts":
            timestamp,

        "equity":
            equity,

        "balance":
            balance,

        "invested":
            invested,

        "realized_pnl":
            _safe_float(
                current[
                    "realized_pnl"
                ]
            ),

        "unrealized_pnl":
            unrealized,

        "return_pct":
            return_pct,

    }


# ============================================================
# EQUITY HISTORY
# ============================================================

def equity_history(
    limit: int = 500,
) -> List[Dict[str, Any]]:

    connection = db()

    try:

        limit = max(
            1,
            min(
                int(limit),
                10000,
            ),
        )

        rows = connection.execute(
            """
            SELECT *
            FROM equity_snapshots
            ORDER BY ts DESC
            LIMIT ?
            """,
            (
                limit,
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
# DAILY RISK
# ============================================================

def reset_daily_state_if_needed() -> None:
    """
    Reset daily counters when a new local calendar day begins.

    This does NOT reset the paper account.
    """

    today = time.strftime(
        "%Y-%m-%d",
        time.localtime(),
    )

    saved_date = get_risk(
        "daily_start_date",
        "",
    )

    if saved_date == today:
        return

    current_equity = (
        stats()[
            "paper_equity"
        ]
    )

    set_risk(
        "daily_start_date",
        today,
    )

    set_risk(
        "daily_start_equity",
        current_equity,
    )

    set_risk(
        "trades_today",
        0,
    )

    set_risk(
        "consecutive_losses",
        0,
    )


def daily_start_equity() -> float:

    return get_risk_float(
        "daily_start_equity",
        get_paper_start_balance(),
    )


def daily_pnl() -> float:

    return (
        stats()["paper_equity"]
        - daily_start_equity()
    )


# ============================================================
# DATABASE HEALTH
# ============================================================

def database_health() -> Dict[str, Any]:

    connection = db()

    try:

        connection.execute(
            "SELECT 1"
        )

        trade_count = connection.execute(
            "SELECT COUNT(*) AS n FROM trades"
        ).fetchone()["n"]

        position_count = connection.execute(
            "SELECT COUNT(*) AS n FROM positions"
        ).fetchone()["n"]

        return {

            "ok":
                True,

            "database":
                DB,

            "trades":
                int(trade_count),

            "positions":
                int(position_count),

        }

    except Exception as exc:

        return {

            "ok":
                False,

            "database":
                DB,

            "error":
                str(exc),

        }

    finally:

        connection.close()


# ============================================================
# INITIALIZE
# ============================================================

init_db()
