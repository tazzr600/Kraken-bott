from __future__ import annotations

import asyncio
import math
import threading
import time
import traceback
from typing import Any

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

import db
from bot import KrakenBot
from config import settings


# ============================================================
# DATABASE
# ============================================================

db.init_db()


# ============================================================
# APP
# ============================================================

app = FastAPI(
    title="KRAKEN BOT",
    version="2.1",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# BOT
# ============================================================

bot = KrakenBot()

bot_thread: threading.Thread | None = None
worker_error: str | None = None
worker_started_at: float | None = None

worker_lock = threading.Lock()


# ============================================================
# JSON SAFETY
# ============================================================

def json_safe(value: Any):

    if value is None:
        return None

    if isinstance(value, (str, bool, int)):
        return value

    if isinstance(value, float):

        if not math.isfinite(value):
            return None

        return value

    if isinstance(value, dict):

        return {
            str(key): json_safe(val)
            for key, val in value.items()
        }

    if isinstance(value, (list, tuple, set)):

        return [
            json_safe(item)
            for item in value
        ]

    if hasattr(value, "item"):

        try:
            return json_safe(value.item())
        except Exception:
            pass

    if hasattr(value, "_asdict"):

        try:
            return json_safe(value._asdict())
        except Exception:
            pass

    if hasattr(value, "__dict__"):

        try:
            return json_safe(vars(value))
        except Exception:
            pass

    return str(value)


# ============================================================
# WORKER
# ============================================================

def worker_alive() -> bool:

    return bool(
        bot_thread
        and bot_thread.is_alive()
    )


def start_bot_thread():

    global bot_thread
    global worker_error
    global worker_started_at

    with worker_lock:

        if worker_alive():

            bot.running = True

            return {
                "started": False,
                "already_running": True,
                "worker_alive": True,
            }

        worker_error = None

        bot.running = True

        worker_started_at = time.time()

        def runner():

            global worker_error

            try:

                print("=" * 60)
                print("BOT THREAD: STARTING")
                print("=" * 60)

                asyncio.run(
                    bot.run()
                )

            except Exception as exc:

                worker_error = (
                    f"{type(exc).__name__}: {exc}"
                )

                print(
                    "BOT THREAD CRASH:",
                    worker_error,
                )

                traceback.print_exc()

            finally:

                bot.running = False

                print(
                    "BOT THREAD: STOPPED"
                )

        bot_thread = threading.Thread(
            target=runner,
            name="kraken-bot",
            daemon=True,
        )

        bot_thread.start()

        time.sleep(0.25)

        return {
            "started": worker_alive(),
            "already_running": False,
            "worker_alive": worker_alive(),
        }


# ============================================================
# STARTUP
# ============================================================

@app.on_event("startup")
def startup():

    print("=" * 60)
    print("KRAKEN BOT STARTING")
    print("=" * 60)

    # --------------------------------------------------------
    # DATABASE
    # --------------------------------------------------------

    try:

        db.init_db()

        print("DATABASE: OK")

    except Exception as exc:

        print(
            "DATABASE ERROR:",
            type(exc).__name__,
            exc,
        )

    # --------------------------------------------------------
    # KRAKEN CONNECTION
    # --------------------------------------------------------

    try:

        connection = bot.kraken.test_connection()

        print(
            "KRAKEN CONNECTION:",
            connection,
        )

    except Exception as exc:

        print(
            "KRAKEN CONNECTION ERROR:",
            type(exc).__name__,
            exc,
        )

    # --------------------------------------------------------
    # KRAKEN AUTH
    # --------------------------------------------------------

    try:

        authentication = (
            bot.kraken.test_authentication()
        )

        print(
            "KRAKEN AUTHENTICATION:",
            authentication,
        )

        if (
            settings.autonomous
            and isinstance(authentication, dict)
            and authentication.get("authenticated") is True
        ):

            result = start_bot_thread()

            print(
                "AUTONOMOUS BOT START RESULT:",
                result,
            )

    except Exception as exc:

        print(
            "KRAKEN AUTH ERROR:",
            type(exc).__name__,
            exc,
        )

    print(
        "PAPER MODE:",
        settings.dry_run,
    )

    print(
        "LIVE ORDERS:",
        bot.kraken.live_orders_enabled,
    )

    print("=" * 60)


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


@app.get("/api/health")
def api_health():

    return JSONResponse(
        content=json_safe({
            "ok": True,
            "running": bool(bot.running),
            "worker_alive": worker_alive(),
            "worker_error": worker_error,
        })
    )


# ============================================================
# STATUS
# ============================================================

@app.get("/api/status")
def status():

    # --------------------------------------------------------
    # DATABASE
    # --------------------------------------------------------

    try:

        statistics = db.stats()

        if not isinstance(statistics, dict):
            statistics = {}

    except Exception as exc:

        statistics = {
            "paper_equity":
                settings.paper_start_balance,
            "paper_balance":
                settings.paper_start_balance,
            "realized_pnl": 0,
            "return_pct": 0,
            "win_rate": 0,
            "trades": 0,
            "wins": 0,
            "losses": 0,
            "profit_factor": 0,
            "unrealized_pnl": 0,
            "last_24h_pnl": 0,
            "last_24h_trades": 0,
            "drawdown_pct": 0,
            "consecutive_losses": 0,
            "error": str(exc),
        }

    # --------------------------------------------------------
    # KRAKEN
    # --------------------------------------------------------

    try:

        kraken_status = bot.kraken.connection_status()

        if not isinstance(kraken_status, dict):
            kraken_status = {}

    except Exception as exc:

        kraken_status = {
            "connected": False,
            "authenticated": False,
            "live_orders_enabled": False,
            "error": str(exc),
        }

    # --------------------------------------------------------
    # SCANNER
    # --------------------------------------------------------

    try:

        scanner = bot.scanner.status()

        if not isinstance(scanner, dict):
            scanner = {}

    except Exception as exc:

        scanner = {
            "markets_loaded": 0,
            "tickers_received": 0,
            "liquid_markets": 0,
            "markets_sent_to_ml": 0,
            "last_error": str(exc),
        }

    # --------------------------------------------------------
    # EQUITY
    # --------------------------------------------------------

    try:

        equity = db.equity_history(240)

        if not isinstance(equity, list):
            equity = []

    except Exception:

        equity = []

    # --------------------------------------------------------
    # POSITIONS
    # --------------------------------------------------------

    try:

        positions = db.get_positions()

        if not isinstance(positions, list):
            positions = []

    except Exception:

        positions = []

    # --------------------------------------------------------
    # SIGNALS
    # --------------------------------------------------------

    signals = getattr(
        bot,
        "signals",
        [],
    )

    if not isinstance(signals, list):
        signals = []

    # --------------------------------------------------------
    # UNIFIED STATUS
    # --------------------------------------------------------

    running = bool(bot.running)
    alive = worker_alive()

    response = {

        "ok": True,

        # FRONTEND COMPATIBILITY
        "running": running,

        "worker_alive": alive,

        "worker_error": worker_error,

        # ALSO PROVIDE BOT OBJECT
        "bot": {
            "running": running,
            "worker_alive": alive,
            "worker_error": worker_error,
            "started_at": worker_started_at,
        },

        # MODE
        "mode": (
            "LIVE"
            if bot.kraken.live_orders_enabled
            else "PAPER"
        ),

        "autonomous": bool(
            settings.autonomous
        ),

        # KRAKEN
        "kraken": kraken_status,

        # SCANNER
        "scanner": scanner,

        # PERFORMANCE
        "stats": statistics,

        # SIGNALS
        "signals": signals,

        "last_signals": signals,

        # POSITIONS
        "positions": positions,

        # EQUITY
        "equity": equity,

        # SCANNER TIMESTAMP
        "last_scan": getattr(
            bot,
            "last_scan",
            None,
        ),

    }

    return JSONResponse(
        content=json_safe(response)
    )


# ============================================================
# START
# ============================================================

@app.post("/api/start")
def start():

    try:

        print("=" * 60)
        print("START BOT REQUEST")
        print("=" * 60)

        # Already running
        if worker_alive():

            bot.running = True

            return JSONResponse(
                content=json_safe({
                    "ok": True,
                    "started": False,
                    "already_running": True,
                    "running": True,
                    "worker_alive": True,
                    "message": "Bot is already running.",
                })
            )

        # Test Kraken authentication
        authentication = (
            bot.kraken.test_authentication()
        )

        if not isinstance(
            authentication,
            dict
        ):

            authentication = {
                "authenticated": False,
                "error": "Invalid authentication response",
            }

        if not authentication.get(
            "authenticated"
        ):

            return JSONResponse(
                content=json_safe({
                    "ok": False,
                    "started": False,
                    "running": False,
                    "worker_alive": False,
                    "error":
                        authentication.get(
                            "error",
                            "Kraken authentication failed.",
                        ),
                })
            )

        result = start_bot_thread()

        running = bool(bot.running)
        alive = worker_alive()

        return JSONResponse(
            content=json_safe({
                "ok": True,
                **result,
                "running": running,
                "worker_alive": alive,
                "message": (
                    "Bot started."
                    if alive
                    else "Bot failed to start."
                ),
            })
        )

    except Exception as exc:

        traceback.print_exc()

        bot.running = False

        return JSONResponse(
            content=json_safe({
                "ok": False,
                "started": False,
                "running": False,
                "worker_alive": False,
                "error":
                    f"{type(exc).__name__}: {exc}",
            })
        )


# ============================================================
# STOP
# ============================================================

@app.post("/api/stop")
def stop():

    try:

        bot.running = False

        return JSONResponse(
            content=json_safe({
                "ok": True,
                "running": False,
                "worker_alive": worker_alive(),
                "message": "Bot stopped",
            })
        )

    except Exception as exc:

        return JSONResponse(
            content=json_safe({
                "ok": False,
                "running": False,
                "error": str(exc),
            })
        )


# ============================================================
# MANUAL SCAN
# ============================================================

@app.post("/api/scan")
def scan():

    try:

        result = asyncio.run(
            bot.scan()
        )

        if result is None:
            result = []

        return JSONResponse(
            content=json_safe({
                "ok": True,
                "signals": result,
                "markets": result,
                "count": len(result),
            })
        )

    except Exception as exc:

        traceback.print_exc()

        return JSONResponse(
            content=json_safe({
                "ok": False,
                "signals": [],
                "markets": [],
                "count": 0,
                "error":
                    f"{type(exc).__name__}: {exc}",
            })
        )


# ============================================================
# SIGNALS
# ============================================================

@app.get("/api/signals")
def signals():

    return JSONResponse(
        content=json_safe({
            "ok": True,
            "signals":
                getattr(
                    bot,
                    "signals",
                    [],
                ),
        })
    )


# ============================================================
# POSITIONS
# ============================================================

@app.get("/api/positions")
def positions():

    try:

        data = db.get_positions()

        if not isinstance(data, list):
            data = []

        return JSONResponse(
            content=json_safe({
                "ok": True,
                "positions": data,
            })
        )

    except Exception as exc:

        return JSONResponse(
            content=json_safe({
                "ok": False,
                "positions": [],
                "error": str(exc),
            })
        )


# ============================================================
# EQUITY
# ============================================================

@app.get("/api/equity")
def equity():

    try:

        data = db.equity_history(500)

        if not isinstance(data, list):
            data = []

        return JSONResponse(
            content=json_safe({
                "ok": True,
                "equity": data,
            })
        )

    except Exception as exc:

        return JSONResponse(
            content=json_safe({
                "ok": False,
                "equity": [],
                "error": str(exc),
            })
        )


# ============================================================
# PERFORMANCE
# ============================================================

@app.get("/api/performance")
def performance():

    try:

        statistics = db.stats()

        if not isinstance(statistics, dict):
            statistics = {}

        return JSONResponse(
            content=json_safe({
                "ok": True,

                # Keep nested version
                "stats": statistics,

                # Also expose flat fields
                # for dashboard compatibility
                **statistics,
            })
        )

    except Exception as exc:

        return JSONResponse(
            content=json_safe({
                "ok": False,
                "stats": {},
                "error": str(exc),
            })
        )


# ============================================================
# SCANNER
# ============================================================

@app.get("/api/scanner")
def scanner():

    try:

        data = bot.scanner.status()

        if not isinstance(data, dict):
            data = {}

        return JSONResponse(
            content=json_safe({
                "ok": True,
                "scanner": data,
            })
        )

    except Exception as exc:

        return JSONResponse(
            content=json_safe({
                "ok": False,
                "scanner": {},
                "error": str(exc),
            })
        )


# ============================================================
# KRAKEN TEST
# ============================================================

@app.get("/api/kraken/test")
def kraken_test():

    try:

        connection = (
            bot.kraken.test_connection()
        )

    except Exception as exc:

        connection = {
            "connected": False,
            "error": str(exc),
        }

    try:

        authentication = (
            bot.kraken.test_authentication()
        )

    except Exception as exc:

        authentication = {
            "authenticated": False,
            "error": str(exc),
        }

    try:

        kraken_status = (
            bot.kraken.connection_status()
        )

    except Exception as exc:

        kraken_status = {
            "connected": False,
            "authenticated": False,
            "error": str(exc),
        }

    return JSONResponse(
        content=json_safe({

            "ok": True,

            "connection":
                connection,

            "authentication":
                authentication,

            "status":
                kraken_status,

        })
    )


# ============================================================
# DEBUG
# ============================================================

@app.get("/api/debug")
def debug():

    try:
        kraken_status = (
            bot.kraken.connection_status()
        )
    except Exception as exc:
        kraken_status = {
            "error": str(exc)
        }

    return JSONResponse(
        content=json_safe({

            "ok": True,

            "running":
                bool(bot.running),

            "worker_alive":
                worker_alive(),

            "worker_error":
                worker_error,

            "bot_error":
                getattr(
                    bot,
                    "error",
                    None,
                ),

            "last_scan":
                getattr(
                    bot,
                    "last_scan",
                    None,
                ),

            "kraken":
                kraken_status,

        })
    )


# ============================================================
# DASHBOARD
# ============================================================

@app.get("/")
def dashboard():

    return FileResponse(
        "index.html"
    )


# ============================================================
# GLOBAL ERROR HANDLER
# ============================================================

@app.exception_handler(Exception)
async def global_errors(
    request,
    exc,
):

    print(
        "GLOBAL ERROR:",
        type(exc).__name__,
        exc,
    )

    traceback.print_exc()

    return JSONResponse(
        status_code=200,
        content=json_safe({
            "ok": False,
            "error":
                f"{type(exc).__name__}: {exc}",
        }),
    )
