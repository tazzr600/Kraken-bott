from __future__ import annotations

import threading
import time
import traceback
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse

import db
from bot import KrakenBot
from config import settings


# ============================================================
# APP
# ============================================================

app = FastAPI(
    title="KRAKEN BOT",
    version="1.0.0",
)


# ============================================================
# GLOBAL BOT
# ============================================================

bot = KrakenBot()

bot_thread: threading.Thread | None = None


# ============================================================
# SAFE HELPERS
# ============================================================

def safe_call(function, default=None):
    """
    Run a function without allowing one component failure
    to crash an API endpoint.
    """
    try:
        return function()
    except Exception as exc:
        print(
            f"SAFE API ERROR: "
            f"{type(exc).__name__}: {exc}"
        )
        return default


def safe_dict(value):
    if isinstance(value, dict):
        return value
    return {}


def safe_list(value):
    if isinstance(value, list):
        return value
    return []


def get_bot_running() -> bool:
    return bool(getattr(bot, "running", False))


def get_mode() -> str:
    return (
        "LIVE"
        if settings.live_trading and not settings.dry_run
        else "PAPER"
    )


# ============================================================
# DATABASE
# ============================================================

@app.on_event("startup")
def startup():

    print("=" * 70)
    print("KRAKEN BOT STARTING")
    print("=" * 70)

    # Database
    try:
        db.init_db()
        print("DATABASE: OK")
    except Exception as exc:
        print(
            "DATABASE ERROR:",
            type(exc).__name__,
            str(exc),
        )

    # Environment
    print(
        "KRAKEN CREDENTIALS CONFIGURED:",
        bool(
            settings.kraken_api_key
            and settings.kraken_api_secret
        ),
    )

    print(
        "LIVE TRADING:",
        settings.live_trading,
    )

    print(
        "DRY RUN:",
        settings.dry_run,
    )

    print(
        "AUTONOMOUS:",
        settings.autonomous,
    )

    # Kraken connection
    try:

        connection = bot.kraken.test_connection()

        print(
            "KRAKEN CONNECTION:",
            "OK" if connection.get("connected") else "FAILED",
        )

    except Exception as exc:

        print(
            "KRAKEN CONNECTION ERROR:",
            type(exc).__name__,
            str(exc),
        )

    # Kraken authentication
    try:

        authentication = bot.kraken.test_authentication()

        print(
            "KRAKEN AUTHENTICATION:",
            "OK"
            if authentication.get("authenticated")
            else "FAILED",
        )

    except Exception as exc:

        print(
            "KRAKEN AUTHENTICATION ERROR:",
            type(exc).__name__,
            str(exc),
        )

    print(
        "KRAKEN LIVE ORDERS:",
        bot.kraken.live_orders_enabled,
    )

    print("=" * 70)

    # Automatically start bot
    if settings.autonomous:

        start_bot_thread()

        print(
            "AUTONOMOUS BOT: STARTED"
        )

    else:

        print(
            "AUTONOMOUS BOT: DISABLED"
        )


# ============================================================
# HEALTH
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "ok",
        "service": "kraken-bot",
        "timestamp": time.time(),
    }


# ============================================================
# STATUS
# ============================================================

@app.get("/api/status")
def api_status():

    """
    Main dashboard endpoint.

    IMPORTANT:
    Every subsystem is protected so a scanner/database/model
    error does not turn the entire endpoint into HTTP 500.
    """

    errors: list[str] = []


    # --------------------------------------------------------
    # Kraken
    # --------------------------------------------------------

    try:

        kraken_status = bot.kraken.connection_status()

    except Exception as exc:

        kraken_status = {
            "connected": False,
            "authenticated": False,
            "live_orders_enabled": False,
            "credentials_configured": bool(
                settings.kraken_api_key
                and settings.kraken_api_secret
            ),
            "error": str(exc),
            "error_type": type(exc).__name__,
        }

        errors.append(
            f"kraken: {type(exc).__name__}: {exc}"
        )


    # --------------------------------------------------------
    # Scanner
    # --------------------------------------------------------

    try:

        scanner_status = bot.scanner.status()

    except Exception as exc:

        scanner_status = {
            "markets_loaded": 0,
            "tickers_received": 0,
            "liquid_markets": 0,
            "markets_sent_to_ml": 0,
            "last_refresh": 0,
            "last_error": str(exc),
            "allowed_quotes":
                settings.allowed_quote_list,
            "min_volume":
                settings.min_quote_volume_usd,
            "max_spread_pct":
                settings.max_spread_pct,
        }

        errors.append(
            f"scanner: {type(exc).__name__}: {exc}"
        )


    # --------------------------------------------------------
    # Database stats
    # --------------------------------------------------------

    try:

        statistics = db.stats()

        if not isinstance(statistics, dict):
            statistics = {}

    except Exception as exc:

        statistics = {}

        errors.append(
            f"database: {type(exc).__name__}: {exc}"
        )


    # --------------------------------------------------------
    # Equity
    # --------------------------------------------------------

    try:

        equity = db.equity_history(240)

        if not isinstance(equity, list):
            equity = []

    except Exception as exc:

        equity = []

        errors.append(
            f"equity: {type(exc).__name__}: {exc}"
        )


    # --------------------------------------------------------
    # Signals
    # --------------------------------------------------------

    signals = []

    try:

        signals = getattr(
            bot,
            "last_signals",
            [],
        )

        if signals is None:
            signals = []

        if not isinstance(signals, list):
            signals = list(signals)

    except Exception as exc:

        errors.append(
            f"signals: {type(exc).__name__}: {exc}"
        )

        signals = []


    # --------------------------------------------------------
    # Positions
    # --------------------------------------------------------

    positions = []

    try:

        # Try common database function names.
        for function_name in (
            "open_positions",
            "get_positions",
            "positions",
            "list_positions",
        ):

            function = getattr(
                db,
                function_name,
                None,
            )

            if callable(function):

                result = function()

                if isinstance(result, list):
                    positions = result

                elif isinstance(result, dict):
                    positions = list(
                        result.values()
                    )

                break

    except Exception as exc:

        errors.append(
            f"positions: {type(exc).__name__}: {exc}"
        )

        positions = []


    # --------------------------------------------------------
    # Normalize scanner names for dashboard
    # --------------------------------------------------------

    # The new scanner uses liquid_markets.
    # The older dashboard used markets_discovered.
    scanner_status["markets_discovered"] = (
        scanner_status.get("liquid_markets", 0)
    )

    scanner_status["max_scan_symbols"] = (
        scanner_status.get(
            "markets_sent_to_ml",
            settings.max_scan_symbols,
        )
    )


    # --------------------------------------------------------
    # Last scan
    # --------------------------------------------------------

    last_scan = getattr(
        bot,
        "last_scan",
        None,
    )

    if last_scan is None:

        last_scan = getattr(
            bot,
            "last_scan_time",
            None,
        )


    # --------------------------------------------------------
    # Response
    # --------------------------------------------------------

    response = {
        "ok": True,

        "service": "kraken-bot",

        "timestamp": time.time(),

        "running":
            get_bot_running(),

        "mode":
            get_mode(),

        "autonomous":
            settings.autonomous,

        "error":
            "; ".join(errors) if errors else None,

        "kraken":
            kraken_status,

        "scanner":
            scanner_status,

        "stats":
            statistics,

        "signals":
            signals,

        "positions":
            positions,

        "equity":
            equity,

        "last_scan":
            last_scan,
    }

    return JSONResponse(
        status_code=200,
        content=response,
    )


# ============================================================
# EQUITY
# ============================================================

@app.get("/api/equity")
def api_equity(limit: int = 240):

    try:

        limit = max(
            1,
            min(limit, 5000),
        )

        history = db.equity_history(limit)

        if not isinstance(history, list):
            history = []

        return history

    except Exception as exc:

        print(
            "EQUITY API ERROR:",
            type(exc).__name__,
            str(exc),
        )

        return []


# ============================================================
# PERFORMANCE
# ============================================================

@app.get("/api/performance")
def api_performance():

    try:

        result = db.stats()

        if not isinstance(result, dict):
            result = {}

        return result

    except Exception as exc:

        print(
            "PERFORMANCE API ERROR:",
            type(exc).__name__,
            str(exc),
        )

        return {
            "paper_equity":
                settings.paper_start_balance,

            "paper_balance":
                settings.paper_start_balance,

            "paper_start_balance":
                settings.paper_start_balance,

            "realized_pnl": 0.0,

            "unrealized_pnl": 0.0,

            "return_pct": 0.0,

            "win_rate": 0.0,

            "trades": 0,

            "wins": 0,

            "losses": 0,

            "profit_factor": 0.0,

            "last_24h_pnl": 0.0,

            "last_24h_trades": 0,

            "consecutive_losses": 0,

            "drawdown_pct": 0.0,
        }


# ============================================================
# KRAKEN TEST
# ============================================================

@app.get("/api/kraken/test")
def api_kraken_test():

    connection = safe_call(
        bot.kraken.test_connection,
        {
            "connected": False,
            "error": "Connection test failed",
        },
    )

    authentication = safe_call(
        bot.kraken.test_authentication,
        {
            "authenticated": False,
            "error": "Authentication test failed",
        },
    )

    return {
        "connection": connection,
        "authentication": authentication,
        "status": safe_call(
            bot.kraken.connection_status,
            {},
        ),
    }


# ============================================================
# START BOT
# ============================================================

def start_bot_thread():

    global bot_thread

    if get_bot_running():
        return False

    try:

        bot.running = True

    except Exception:

        pass


    def runner():

        try:

            print(
                "BOT THREAD: STARTING"
            )

            bot.run()

        except Exception as exc:

            print(
                "BOT THREAD CRASH:",
                type(exc).__name__,
                str(exc),
            )

            traceback.print_exc()

        finally:

            try:
                bot.running = False
            except Exception:
                pass

            print(
                "BOT THREAD: STOPPED"
            )


    bot_thread = threading.Thread(
        target=runner,
        daemon=True,
        name="kraken-bot",
    )

    bot_thread.start()

    return True


@app.post("/api/start")
def api_start():

    started = start_bot_thread()

    return {
        "ok": True,
        "started": started,
        "running": get_bot_running(),
        "mode": get_mode(),
    }


# ============================================================
# STOP BOT
# ============================================================

@app.post("/api/stop")
def api_stop():

    try:

        stop_method = getattr(
            bot,
            "stop",
            None,
        )

        if callable(stop_method):

            stop_method()

        else:

            bot.running = False

    except Exception as exc:

        print(
            "BOT STOP ERROR:",
            type(exc).__name__,
            str(exc),
        )

        try:
            bot.running = False
        except Exception:
            pass


    return {
        "ok": True,
        "running": get_bot_running(),
        "message": "Bot stopped",
    }


# ============================================================
# FULL MARKET SCAN
# ============================================================

@app.post("/api/scan")
def api_scan():

    try:

        result = bot.scan()

        if result is None:
            result = []

        if not isinstance(result, list):
            result = list(result)

        # Keep dashboard cache updated.
        try:
            bot.last_signals = result
        except Exception:
            pass

        return {
            "ok": True,
            "signals": result,
            "markets": result,
            "count": len(result),
            "timestamp": time.time(),
        }

    except Exception as exc:

        print(
            "SCAN API ERROR:",
            type(exc).__name__,
            str(exc),
        )

        traceback.print_exc()

        return JSONResponse(
            status_code=200,
            content={
                "ok": False,
                "signals": [],
                "markets": [],
                "count": 0,
                "error":
                    f"{type(exc).__name__}: {exc}",
            },
        )


# ============================================================
# SIGNALS
# ============================================================

@app.get("/api/signals")
def api_signals():

    try:

        signals = getattr(
            bot,
            "last_signals",
            [],
        )

        if signals is None:
            signals = []

        if not isinstance(signals, list):
            signals = list(signals)

        return signals

    except Exception as exc:

        print(
            "SIGNALS API ERROR:",
            type(exc).__name__,
            str(exc),
        )

        return []


# ============================================================
# POSITIONS
# ============================================================

@app.get("/api/positions")
def api_positions():

    try:

        for function_name in (
            "open_positions",
            "get_positions",
            "positions",
            "list_positions",
        ):

            function = getattr(
                db,
                function_name,
                None,
            )

            if callable(function):

                result = function()

                if isinstance(result, list):
                    return result

                if isinstance(result, dict):
                    return list(
                        result.values()
                    )

    except Exception as exc:

        print(
            "POSITIONS API ERROR:",
            type(exc).__name__,
            str(exc),
        )

    return []


# ============================================================
# SCANNER
# ============================================================

@app.get("/api/scanner")
def api_scanner():

    try:

        return bot.scanner.status()

    except Exception as exc:

        return {
            "markets_loaded": 0,
            "tickers_received": 0,
            "liquid_markets": 0,
            "markets_sent_to_ml": 0,
            "markets_discovered": 0,
            "max_scan_symbols":
                settings.max_scan_symbols,
            "last_refresh": 0,
            "last_error":
                f"{type(exc).__name__}: {exc}",
            "allowed_quotes":
                settings.allowed_quote_list,
            "min_volume":
                settings.min_quote_volume_usd,
            "max_spread_pct":
                settings.max_spread_pct,
        }


# ============================================================
# ROOT DASHBOARD
# ============================================================

@app.get("/")
def dashboard():

    index_file = (
        Path(__file__).resolve().parent
        / "index.html"
    )

    if not index_file.exists():

        return JSONResponse(
            status_code=500,
            content={
                "error":
                    "index.html not found"
            },
        )

    return FileResponse(
        index_file,
        media_type="text/html",
    )


# ============================================================
# GLOBAL ERROR HANDLER
# ============================================================

@app.exception_handler(Exception)
async def global_exception_handler(
    request,
    exc: Exception,
):

    print(
        "UNHANDLED API ERROR:",
        request.url.path,
        type(exc).__name__,
        str(exc),
    )

    traceback.print_exc()

    return JSONResponse(
        status_code=500,
        content={
            "ok": False,
            "error":
                f"{type(exc).__name__}: {exc}",
            "path":
                request.url.path,
        },
    )
