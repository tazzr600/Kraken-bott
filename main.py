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
# DATABASE FIRST
# ============================================================

db.init_db()


# ============================================================
# APP
# ============================================================

app = FastAPI(
    title="KRAKEN BOT",
    version="2.0",
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

bot_thread = None
worker_error = None
worker_started_at = None

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
            return json_safe(
                value.item()
            )
        except Exception:
            pass

    if hasattr(value, "_asdict"):

        try:
            return json_safe(
                value._asdict()
            )
        except Exception:
            pass

    if hasattr(value, "__dict__"):

        try:
            return json_safe(
                vars(value)
            )
        except Exception:
            pass

    return str(value)


# ============================================================
# WORKER
# ============================================================

def worker_alive():

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

                # IMPORTANT:
                # bot.run() is async.
                asyncio.run(
                    bot.run()
                )

            except Exception as exc:

                worker_error = (
                    f"{type(exc).__name__}: "
                    f"{exc}"
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
        }


# ============================================================
# STARTUP
# ============================================================

@app.on_event("startup")
def startup():

    print("=" * 60)
    print("KRAKEN BOT STARTING")
    print("=" * 60)

    try:

        db.init_db()

        print(
            "DATABASE: OK"
        )

    except Exception as exc:

        print(
            "DATABASE ERROR:",
            exc,
        )

    # --------------------------------------------------------
    # KRAKEN CONNECTION
    # --------------------------------------------------------

    try:

        connection = (
            bot.kraken.test_connection()
        )

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

        # Automatically start when authenticated.
        if (
            settings.autonomous
            and authentication.get(
                "authenticated"
            )
        ):

            start_bot_thread()

            print(
                "AUTONOMOUS BOT: STARTED"
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

    return json_safe({
        "ok": True,
        "running": bot.running,
        "worker_alive": worker_alive(),
        "worker_error": worker_error,
    })


# ============================================================
# STATUS
# ============================================================

@app.get("/api/status")
def status():

    try:

        statistics = db.stats()

    except Exception as exc:

        statistics = {
            "paper_equity":
                settings.paper_start_balance,
            "paper_balance":
                settings.paper_start_balance,
            "realized_pnl": 0,
            "return_pct": 0,
            "win_rate": 0,
            "error": str(exc),
        }

    try:

        scanner = bot.scanner.status()

    except Exception as exc:

        scanner = {
            "markets_loaded": 0,
            "tickers_received": 0,
            "liquid_markets": 0,
            "markets_sent_to_ml": 0,
            "last_error":
                str(exc),
        }

    try:

        equity = db.equity_history(240)

    except Exception:

        equity = []

    try:

        positions = db.get_positions()

    except Exception:

        positions = []

    signals = getattr(
        bot,
        "signals",
        [],
    )

    response = {

        "ok": True,

        "running":
            bool(bot.running),

        "worker_alive":
            worker_alive(),

        "worker_error":
            worker_error,

        "mode":
            (
                "LIVE"
                if bot.kraken.live_orders_enabled
                else "PAPER"
            ),

        "autonomous":
            settings.autonomous,

        "kraken":
            bot.kraken.connection_status(),

        "scanner":
            scanner,

        "stats":
            statistics,

        "signals":
            signals,

        "last_signals":
            signals,

        "positions":
            positions,

        "equity":
            equity,

        "last_scan":
            bot.last_scan,

    }

    return JSONResponse(
        content=json_safe(
            response
        )
    )


# ============================================================
# START
# ============================================================

@app.post("/api/start")
def start():

    try:

        print(
            "START BOT REQUEST"
        )

        authentication = (
            bot.kraken.test_authentication()
        )

        if not authentication.get(
            "authenticated"
        ):

            return JSONResponse(
                content=json_safe({
                    "ok": False,
                    "started": False,
                    "error":
                        authentication.get(
                            "error",
                            "Kraken authentication failed",
                        ),
                })
            )

        result = start_bot_thread()

        return JSONResponse(
            content=json_safe({
                "ok": True,
                **result,
                "running":
                    bot.running,
                "worker_alive":
                    worker_alive(),
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

    bot.running = False

    return {
        "ok": True,
        "running": False,
        "message": "Bot stopped",
    }


# ============================================================
# MANUAL SCAN
# ============================================================

@app.post("/api/scan")
def scan():

    try:

        result = asyncio.run(
            bot.scan()
        )

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

    return JSONResponse(
        content=json_safe({
            "ok": True,
            "positions":
                db.get_positions(),
        })
    )


# ============================================================
# EQUITY
# ============================================================

@app.get("/api/equity")
def equity():

    return JSONResponse(
        content=json_safe({
            "ok": True,
            "equity":
                db.equity_history(500),
        })
    )


# ============================================================
# PERFORMANCE
# ============================================================

@app.get("/api/performance")
def performance():

    return JSONResponse(
        content=json_safe({
            "ok": True,
            "stats":
                db.stats(),
        })
    )


# ============================================================
# SCANNER
# ============================================================

@app.get("/api/scanner")
def scanner():

    return JSONResponse(
        content=json_safe({
            "ok": True,
            "scanner":
                bot.scanner.status(),
        })
    )


# ============================================================
# KRAKEN TEST
# ============================================================

@app.get("/api/kraken/test")
def kraken_test():

    return JSONResponse(
        content=json_safe({

            "connection":
                bot.kraken.test_connection(),

            "authentication":
                bot.kraken.test_authentication(),

            "status":
                bot.kraken.connection_status(),

        })
    )


# ============================================================
# DEBUG
# ============================================================

@app.get("/api/debug")
def debug():

    return json_safe({

        "ok": True,

        "running":
            bot.running,

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
            bot.last_scan,

        "kraken":
            bot.kraken.connection_status(),

    })


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
