from __future__ import annotations

import math
import threading
import time
import traceback
from typing import Any

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from config import settings
import db
from bot import KrakenBot


app = FastAPI(title="KRAKEN BOT")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# GLOBAL BOT STATE
# ============================================================

bot = KrakenBot()

worker_thread: threading.Thread | None = None
worker_error: str | None = None
worker_started_at: float | None = None
worker_lock = threading.Lock()


# ============================================================
# JSON SAFETY
# ============================================================

def json_safe(value: Any):
    """
    Convert numpy / pandas / dataclasses / SQLite values
    into normal JSON-safe Python values.
    """

    if value is None:
        return None

    if isinstance(value, bool):
        return value

    if isinstance(value, int):
        return value

    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value

    if isinstance(value, str):
        return value

    if isinstance(value, dict):
        return {
            str(k): json_safe(v)
            for k, v in value.items()
        }

    if isinstance(value, (list, tuple, set)):
        return [
            json_safe(v)
            for v in value
        ]

    # numpy scalar
    if hasattr(value, "item"):
        try:
            return json_safe(value.item())
        except Exception:
            pass

    # dataclass / normal object
    if hasattr(value, "__dict__"):
        try:
            return json_safe(vars(value))
        except Exception:
            pass

    return str(value)


def safe_call(function, default=None):
    try:
        return function()
    except Exception as exc:
        return default


# ============================================================
# BOT WORKER
# ============================================================

def bot_worker():
    """
    Dedicated autonomous trading worker.

    The bot itself controls its running state.
    """

    global worker_error

    worker_error = None

    try:
        print("======================================")
        print("KRAKEN BOT WORKER STARTING")
        print("======================================")

        bot.running = True

        print("BOT RUNNING FLAG:", bot.running)
        print("MODE:", "LIVE" if getattr(bot, "live_trading", False) else "PAPER")

        # KrakenBot.run() contains the autonomous loop.
        bot.run()

        print("======================================")
        print("KRAKEN BOT WORKER EXITED")
        print("======================================")

    except Exception as exc:
        worker_error = f"{type(exc).__name__}: {exc}"

        print("======================================")
        print("KRAKEN BOT WORKER CRASHED")
        print(worker_error)
        print("======================================")

        traceback.print_exc()

        try:
            bot.running = False
        except Exception:
            pass


def start_worker():
    global worker_thread
    global worker_started_at
    global worker_error

    with worker_lock:

        # Already running
        if worker_thread is not None and worker_thread.is_alive():
            try:
                bot.running = True
            except Exception:
                pass

            return {
                "started": False,
                "already_running": True,
                "message": "Bot is already running",
            }

        worker_error = None

        try:
            bot.running = True
        except Exception:
            pass

        worker_started_at = time.time()

        worker_thread = threading.Thread(
            target=bot_worker,
            name="kraken-bot-worker",
            daemon=True,
        )

        worker_thread.start()

        # Give the thread a moment to initialize.
        time.sleep(0.25)

        alive = worker_thread.is_alive()

        return {
            "started": alive,
            "already_running": False,
            "message": (
                "Bot worker started"
                if alive
                else "Bot worker exited immediately"
            ),
        }


def stop_worker():
    global worker_thread

    try:
        bot.running = False
    except Exception:
        pass

    return {
        "stopped": True,
        "message": "Bot stop signal sent",
    }


# ============================================================
# STARTUP
# ============================================================

@app.on_event("startup")
def startup_event():

    print("======================================")
    print("KRAKEN BOT SERVER STARTING")
    print("======================================")

    try:
        db.init_db()
        print("DATABASE: OK")
    except Exception as exc:
        print("DATABASE ERROR:", exc)

    # Test public Kraken connection
    try:
        result = bot.kraken.test_connection()
        print("KRAKEN CONNECTION:", result)
    except Exception as exc:
        print("KRAKEN CONNECTION ERROR:", exc)

    # Test private Kraken authentication
    try:
        result = bot.kraken.test_authentication()
        print("KRAKEN AUTHENTICATION:", result)
    except Exception as exc:
        print("KRAKEN AUTH ERROR:", exc)

    print("KRAKEN CREDENTIALS CONFIGURED:",
          bot.kraken.credentials_configured())

    print("LIVE ORDERS:",
          bot.kraken.live_orders_enabled)

    print("SERVER READY")
    print("======================================")


# ============================================================
# HEALTH
# ============================================================

@app.get("/health")
def health():

    return {
        "ok": True,
        "service": "kraken-bot",
        "time": time.time(),
    }


@app.get("/api/health")
def api_health():

    alive = (
        worker_thread is not None
        and worker_thread.is_alive()
    )

    return json_safe({
        "ok": True,
        "bot_running": bool(getattr(bot, "running", False)),
        "worker_alive": alive,
        "worker_error": worker_error,
    })


# ============================================================
# STATUS
# ============================================================

@app.get("/api/status")
def api_status():

    try:
        stats = safe_call(
            db.stats,
            {
                "paper_equity": 1000.0,
                "paper_balance": 1000.0,
                "realized_pnl": 0.0,
                "return_pct": 0.0,
                "win_rate": 0.0,
            },
        )

        scanner_status = safe_call(
            bot.scanner.status,
            {},
        )

        kraken_status = safe_call(
            bot.kraken.connection_status,
            {},
        )

        equity = safe_call(
            lambda: db.equity_history(500),
            [],
        )

        signals = getattr(
            bot,
            "last_signals",
            [],
        )

        positions = safe_call(
            db.open_positions,
            [],
        )

        running = bool(
            getattr(bot, "running", False)
        )

        worker_alive = bool(
            worker_thread is not None
            and worker_thread.is_alive()
        )

        last_scan = getattr(
            bot,
            "last_scan",
            None,
        )

        response = {
            "ok": True,

            "running": running,

            "worker_alive": worker_alive,

            "worker_error": worker_error,

            "worker_started_at": worker_started_at,

            "mode": (
                "LIVE"
                if bot.kraken.live_orders_enabled
                else "PAPER"
            ),

            "kraken": kraken_status,

            "scanner": scanner_status,

            "stats": stats,

            "equity": equity,

            "signals": signals,

            "positions": positions,

            "last_scan": last_scan,

        }

        return JSONResponse(
            content=json_safe(response)
        )

    except Exception as exc:

        print("STATUS ERROR:", exc)
        traceback.print_exc()

        # IMPORTANT:
        # Never let /api/status crash the dashboard.

        return JSONResponse(
            status_code=200,
            content=json_safe({
                "ok": False,
                "running": bool(
                    getattr(bot, "running", False)
                ),
                "worker_alive": bool(
                    worker_thread
                    and worker_thread.is_alive()
                ),
                "mode": "LIVE"
                if bot.kraken.live_orders_enabled
                else "PAPER",
                "error": f"{type(exc).__name__}: {exc}",
                "stats": {
                    "paper_equity": 1000.0,
                    "paper_balance": 1000.0,
                    "realized_pnl": 0.0,
                    "return_pct": 0.0,
                    "win_rate": 0.0,
                },
                "scanner": {},
                "signals": [],
                "positions": [],
                "equity": [],
            })
        )


# ============================================================
# START BOT
# ============================================================

@app.post("/api/start")
def api_start():

    print("======================================")
    print("START BOT REQUEST RECEIVED")
    print("======================================")

    try:

        # Verify Kraken authentication first.
        auth = bot.kraken.test_authentication()

        print("START AUTH CHECK:", auth)

        if not auth.get("authenticated", False):

            return JSONResponse(
                status_code=200,
                content=json_safe({
                    "ok": False,
                    "started": False,
                    "message": "Kraken authentication failed",
                    "error": auth.get("error"),
                }),
            )

        result = start_worker()

        print("START RESULT:", result)

        return JSONResponse(
            status_code=200,
            content=json_safe({
                "ok": True,
                **result,
                "running": bool(
                    getattr(bot, "running", False)
                ),
                "worker_alive": bool(
                    worker_thread
                    and worker_thread.is_alive()
                ),
            }),
        )

    except Exception as exc:

        print("START BOT ERROR:", exc)
        traceback.print_exc()

        try:
            bot.running = False
        except Exception:
            pass

        return JSONResponse(
            status_code=200,
            content=json_safe({
                "ok": False,
                "started": False,
                "running": False,
                "worker_alive": False,
                "error": f"{type(exc).__name__}: {exc}",
            }),
        )


# ============================================================
# STOP BOT
# ============================================================

@app.post("/api/stop")
def api_stop():

    print("STOP BOT REQUEST RECEIVED")

    try:

        result = stop_worker()

        return JSONResponse(
            status_code=200,
            content=json_safe({
                "ok": True,
                **result,
                "running": bool(
                    getattr(bot, "running", False)
                ),
            }),
        )

    except Exception as exc:

        print("STOP ERROR:", exc)

        return JSONResponse(
            status_code=200,
            content=json_safe({
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
            }),
        )


# ============================================================
# FULL MARKET SCAN
# ============================================================

@app.post("/api/scan")
def api_scan():

    try:

        # Force a fresh market universe.
        try:
            bot.scanner.refresh(force=True)
        except TypeError:
            bot.scanner.refresh()

        signals = bot.scan()

        return JSONResponse(
            status_code=200,
            content=json_safe({
                "ok": True,
                "signals": signals,
                "markets": signals,
            }),
        )

    except Exception as exc:

        print("SCAN ERROR:", exc)
        traceback.print_exc()

        return JSONResponse(
            status_code=200,
            content=json_safe({
                "ok": False,
                "signals": [],
                "markets": [],
                "error": f"{type(exc).__name__}: {exc}",
            }),
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

        return JSONResponse(
            content=json_safe({
                "ok": True,
                "signals": signals,
            })
        )

    except Exception as exc:

        return JSONResponse(
            status_code=200,
            content={
                "ok": False,
                "signals": [],
                "error": f"{type(exc).__name__}: {exc}",
            },
        )


# ============================================================
# POSITIONS
# ============================================================

@app.get("/api/positions")
def api_positions():

    try:

        positions = db.open_positions()

        return JSONResponse(
            content=json_safe({
                "ok": True,
                "positions": positions,
            })
        )

    except Exception as exc:

        return JSONResponse(
            status_code=200,
            content={
                "ok": False,
                "positions": [],
                "error": f"{type(exc).__name__}: {exc}",
            },
        )


# ============================================================
# EQUITY
# ============================================================

@app.get("/api/equity")
def api_equity():

    try:

        history = db.equity_history(500)

        return JSONResponse(
            content=json_safe({
                "ok": True,
                "equity": history,
            })
        )

    except Exception as exc:

        return JSONResponse(
            status_code=200,
            content={
                "ok": False,
                "equity": [],
                "error": f"{type(exc).__name__}: {exc}",
            },
        )


# ============================================================
# PERFORMANCE
# ============================================================

@app.get("/api/performance")
def api_performance():

    try:

        stats = db.stats()

        return JSONResponse(
            content=json_safe({
                "ok": True,
                "stats": stats,
            })
        )

    except Exception as exc:

        return JSONResponse(
            status_code=200,
            content={
                "ok": False,
                "stats": {},
                "error": f"{type(exc).__name__}: {exc}",
            },
        )


# ============================================================
# SCANNER
# ============================================================

@app.get("/api/scanner")
def api_scanner():

    try:

        status = bot.scanner.status()

        return JSONResponse(
            content=json_safe({
                "ok": True,
                "scanner": status,
            })
        )

    except Exception as exc:

        return JSONResponse(
            status_code=200,
            content={
                "ok": False,
                "scanner": {},
                "error": f"{type(exc).__name__}: {exc}",
            },
        )


# ============================================================
# KRAKEN TEST
# ============================================================

@app.get("/api/kraken/test")
def api_kraken_test():

    try:

        connection = bot.kraken.test_connection()
        authentication = bot.kraken.test_authentication()

        return JSONResponse(
            content=json_safe({
                "ok": True,
                "connection": connection,
                "authentication": authentication,
            })
        )

    except Exception as exc:

        return JSONResponse(
            status_code=200,
            content={
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
            },
        )


# ============================================================
# DEBUG
# ============================================================

@app.get("/api/debug")
def api_debug():

    return JSONResponse(
        content=json_safe({
            "ok": True,

            "bot_running": bool(
                getattr(bot, "running", False)
            ),

            "worker_alive": bool(
                worker_thread
                and worker_thread.is_alive()
            ),

            "worker_error": worker_error,

            "worker_started_at": worker_started_at,

            "kraken_authenticated": bool(
                getattr(
                    bot.kraken,
                    "authenticated",
                    False
                )
            ),

            "kraken_connected": bool(
                getattr(
                    bot.kraken,
                    "connected",
                    False
                )
            ),

            "live_orders": bool(
                bot.kraken.live_orders_enabled
            ),

            "last_error": getattr(
                bot.kraken,
                "last_error",
                None,
            ),
        })
    )


# ============================================================
# DASHBOARD
# ============================================================

@app.get("/")
def dashboard():

    return FileResponse("index.html")


# ============================================================
# GLOBAL ERROR HANDLER
# ============================================================

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):

    print("GLOBAL API ERROR:", exc)
    traceback.print_exc()

    return JSONResponse(
        status_code=200,
        content=json_safe({
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
        }),
    )
