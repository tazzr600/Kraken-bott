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
    version="3.0",
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
# ACCOUNT / MODE HELPERS
# ============================================================

def current_mode():

    try:

        return bot.kraken.mode

    except Exception:

        return (
            "LIVE"
            if bot.kraken.live_orders_enabled
            else "PAPER"
        )


def paper_balance():

    try:

        statistics = db.stats()

        if isinstance(statistics, dict):

            equity = statistics.get(
                "paper_equity"
            )

            balance = statistics.get(
                "paper_balance"
            )

            return {
                "equity": float(
                    equity
                    if equity is not None
                    else settings.paper_start_balance
                ),
                "cash": float(
                    balance
                    if balance is not None
                    else settings.paper_start_balance
                ),
            }

    except Exception:

        pass

    return {
        "equity":
            float(settings.paper_start_balance),

        "cash":
            float(settings.paper_start_balance),
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
    # ALWAYS BOOT PAPER
    # --------------------------------------------------------
    #
    # This is intentional.
    #
    # A Railway restart/redeploy must never automatically
    # turn real-money trading on.
    #
    try:

        bot.kraken.set_mode("PAPER")

        print(
            "TRADING MODE: PAPER"
        )

    except Exception as exc:

        print(
            "MODE INITIALIZATION ERROR:",
            type(exc).__name__,
            exc,
        )

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

        # ----------------------------------------------------
        # AUTONOMOUS START
        # ----------------------------------------------------
        #
        # Starts the engine in PAPER mode.
        #
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
        "INITIAL MODE:",
        current_mode(),
    )

    print(
        "LIVE ORDERS ENABLED:",
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
            "running":
                bool(bot.running),
            "worker_alive":
                worker_alive(),
            "worker_error":
                worker_error,
            "mode":
                current_mode(),
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

        kraken_status = (
            bot.kraken.connection_status()
        )

        if not isinstance(
            kraken_status,
            dict,
        ):

            kraken_status = {}

    except Exception as exc:

        kraken_status = {
            "connected": False,
            "authenticated": False,
            "live_orders_enabled": False,
            "mode": "PAPER",
            "error": str(exc),
        }

    # --------------------------------------------------------
    # SCANNER
    # --------------------------------------------------------

    try:

        scanner = bot.scanner.status()

        if not isinstance(
            scanner,
            dict,
        ):

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

        if not isinstance(
            equity,
            list,
        ):

            equity = []

    except Exception:

        equity = []

    # --------------------------------------------------------
    # POSITIONS
    # --------------------------------------------------------

    try:

        positions = db.get_positions()

        if not isinstance(
            positions,
            list,
        ):

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

    if not isinstance(
        signals,
        list,
    ):

        signals = []

    # --------------------------------------------------------
    # PAPER ACCOUNT
    # --------------------------------------------------------

    paper = paper_balance()

    # --------------------------------------------------------
    # UNIFIED STATUS
    # --------------------------------------------------------

    running = bool(
        bot.running
    )

    alive = worker_alive()

    response = {

        "ok": True,

        "running":
            running,

        "worker_alive":
            alive,

        "worker_error":
            worker_error,

        "bot": {

            "running":
                running,

            "worker_alive":
                alive,

            "worker_error":
                worker_error,

            "started_at":
                worker_started_at,
        },

        # ----------------------------------------------------
        # TRADING MODE
        # ----------------------------------------------------

        "mode":
            current_mode(),

        "paper":
            current_mode() == "PAPER",

        "live":
            current_mode() == "LIVE",

        "live_orders_enabled":
            bot.kraken.live_orders_enabled,

        "autonomous":
            bool(settings.autonomous),

        # ----------------------------------------------------
        # KRAKEN
        # ----------------------------------------------------

        "kraken":
            kraken_status,

        # ----------------------------------------------------
        # ACCOUNT
        # ----------------------------------------------------

        "account": {

            "mode":
                current_mode(),

            "paper_equity":
                paper["equity"],

            "paper_cash":
                paper["cash"],

            "live_orders_enabled":
                bot.kraken.live_orders_enabled,
        },

        # ----------------------------------------------------
        # SCANNER
        # ----------------------------------------------------

        "scanner":
            scanner,

        # ----------------------------------------------------
        # PERFORMANCE
        # ----------------------------------------------------

        "stats":
            statistics,

        # ----------------------------------------------------
        # SIGNALS
        # ----------------------------------------------------

        "signals":
            signals,

        "last_signals":
            signals,

        # ----------------------------------------------------
        # POSITIONS
        # ----------------------------------------------------

        "positions":
            positions,

        # ----------------------------------------------------
        # EQUITY
        # ----------------------------------------------------

        "equity":
            equity,

        # ----------------------------------------------------
        # LAST SCAN
        # ----------------------------------------------------

        "last_scan":
            getattr(
                bot,
                "last_scan",
                None,
            ),
    }

    return JSONResponse(
        content=json_safe(response)
    )


# ============================================================
# TRADING MODE
# ============================================================

@app.get("/api/trading-mode")
def get_trading_mode():

    return JSONResponse(
        content=json_safe({
            "ok": True,

            "mode":
                current_mode(),

            "paper":
                current_mode() == "PAPER",

            "live":
                current_mode() == "LIVE",

            "live_orders_enabled":
                bot.kraken.live_orders_enabled,

            "authenticated":
                bot.kraken.authenticated,

            "configuration": {

                "live_trading":
                    bool(
                        getattr(
                            settings,
                            "live_trading",
                            False,
                        )
                    ),

                "dry_run":
                    bool(
                        getattr(
                            settings,
                            "dry_run",
                            True,
                        )
                    ),
            },
        })
    )


@app.post("/api/trading-mode")
def set_trading_mode(payload: dict[str, Any]):

    try:

        requested = str(
            payload.get(
                "mode",
                "",
            )
        ).strip().upper()

        if requested not in {
            "PAPER",
            "LIVE",
        }:

            return JSONResponse(
                status_code=400,
                content={
                    "ok": False,
                    "error":
                        "Mode must be PAPER or LIVE.",
                },
            )

        # ----------------------------------------------------
        # PAPER
        # ----------------------------------------------------

        if requested == "PAPER":

            result = (
                bot.kraken.set_mode("PAPER")
            )

            print(
                "TRADING MODE CHANGED: PAPER"
            )

            return JSONResponse(
                content=json_safe({
                    "ok": True,
                    **result,
                    "message":
                        "Trading mode changed to PAPER. "
                        "No real orders will be submitted.",
                })
            )

        # ----------------------------------------------------
        # LIVE
        # ----------------------------------------------------
        #
        # Require explicit confirmation.
        #

        confirmation = str(
            payload.get(
                "confirmation",
                "",
            )
        ).strip().upper()

        if confirmation != "ENABLE LIVE":

            return JSONResponse(
                status_code=400,
                content={
                    "ok": False,
                    "error":
                        "LIVE mode requires confirmation "
                        "'ENABLE LIVE'.",
                },
            )

        # ----------------------------------------------------
        # Authenticate again immediately before LIVE
        # ----------------------------------------------------

        authentication = (
            bot.kraken.test_authentication()
        )

        if not authentication.get(
            "authenticated"
        ):

            return JSONResponse(
                status_code=401,
                content=json_safe({
                    "ok": False,
                    "error":
                        authentication.get(
                            "error",
                            "Kraken authentication failed.",
                        ),
                })
            )

        # ----------------------------------------------------
        # Verify real USD balance
        # ----------------------------------------------------

        account = (
            bot.kraken.account_summary(
                "USD"
            )
        )

        if account.get("error"):

            return JSONResponse(
                status_code=400,
                content=json_safe({
                    "ok": False,
                    "error":
                        account["error"],
                })
            )

        # ----------------------------------------------------
        # Enable LIVE
        # ----------------------------------------------------

        result = (
            bot.kraken.set_mode("LIVE")
        )

        print("=" * 60)
        print("LIVE TRADING ENABLED")
        print(
            "AVAILABLE USD:",
            account.get("free"),
        )
        print(
            "TOTAL USD:",
            account.get("total"),
        )
        print("=" * 60)

        return JSONResponse(
            content=json_safe({
                "ok": True,

                **result,

                "account":
                    account,

                "message":
                    "LIVE trading enabled.",
            })
        )

    except Exception as exc:

        traceback.print_exc()

        return JSONResponse(
            status_code=400,
            content=json_safe({
                "ok": False,
                "error":
                    f"{type(exc).__name__}: {exc}",
            })
        )


# ============================================================
# ACCOUNT BALANCE
# ============================================================

@app.get("/api/balance")
def balance():

    try:

        mode = current_mode()

        # ----------------------------------------------------
        # PAPER
        # ----------------------------------------------------

        if mode == "PAPER":

            paper = paper_balance()

            return JSONResponse(
                content=json_safe({
                    "ok": True,

                    "mode":
                        "PAPER",

                    "currency":
                        "USD",

                    "equity":
                        paper["equity"],

                    "cash":
                        paper["cash"],

                    "free":
                        paper["cash"],

                    "total":
                        paper["equity"],

                    "live":
                        False,

                    "source":
                        "paper_account",
                })
            )

        # ----------------------------------------------------
        # LIVE
        # ----------------------------------------------------

        account = (
            bot.kraken.account_summary(
                "USD"
            )
        )

        if account.get("error"):

            return JSONResponse(
                status_code=503,
                content=json_safe({
                    "ok": False,
                    "mode":
                        "LIVE",
                    "error":
                        account["error"],
                })
            )

        return JSONResponse(
            content=json_safe({
                "ok": True,

                "mode":
                    "LIVE",

                "currency":
                    "USD",

                "equity":
                    account.get("total", 0),

                "cash":
                    account.get("free", 0),

                "free":
                    account.get("free", 0),

                "used":
                    account.get("used", 0),

                "total":
                    account.get("total", 0),

                "live":
                    True,

                "source":
                    "kraken",
            })
        )

    except Exception as exc:

        return JSONResponse(
            status_code=500,
            content=json_safe({
                "ok": False,
                "error":
                    f"{type(exc).__name__}: {exc}",
            })
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

        # ----------------------------------------------------
        # Already running
        # ----------------------------------------------------

        if worker_alive():

            bot.running = True

            return JSONResponse(
                content=json_safe({
                    "ok": True,
                    "started": False,
                    "already_running": True,
                    "running": True,
                    "worker_alive": True,
                    "mode":
                        current_mode(),
                    "message":
                        "Bot is already running.",
                })
            )

        # ----------------------------------------------------
        # Authenticate
        # ----------------------------------------------------

        authentication = (
            bot.kraken.test_authentication()
        )

        if not isinstance(
            authentication,
            dict,
        ):

            authentication = {
                "authenticated": False,
                "error":
                    "Invalid authentication response",
            }

        if not authentication.get(
            "authenticated"
        ):

            return JSONResponse(
                status_code=401,
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

        # ----------------------------------------------------
        # Start existing bot
        # ----------------------------------------------------

        result = start_bot_thread()

        running = bool(
            bot.running
        )

        alive = worker_alive()

        return JSONResponse(
            content=json_safe({
                "ok": True,

                **result,

                "running":
                    running,

                "worker_alive":
                    alive,

                "mode":
                    current_mode(),

                "message": (
                    "Bot started."
                    if alive
                    else
                    "Bot failed to start."
                ),
            })
        )

    except Exception as exc:

        traceback.print_exc()

        bot.running = False

        return JSONResponse(
            status_code=500,
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

        print(
            "BOT STOP REQUESTED"
        )

        return JSONResponse(
            content=json_safe({
                "ok": True,

                "running":
                    False,

                "worker_alive":
                    worker_alive(),

                "mode":
                    current_mode(),

                "message":
                    "Bot stopped. No new trades will be opened.",
            })
        )

    except Exception as exc:

        return JSONResponse(
            status_code=500,
            content=json_safe({
                "ok": False,
                "running": False,
                "error":
                    str(exc),
            })
        )


# ============================================================
# MANUAL SCAN
# ============================================================

@app.post("/api/scan")
def scan():

    try:

        # ----------------------------------------------------
        # IMPORTANT:
        #
        # Do NOT run asyncio.run(bot.scan()) here.
        #
        # The bot already has a running async worker.
        # Queue the scan instead of blocking the HTTP request.
        # ----------------------------------------------------

        if not bot.running:

            return JSONResponse(
                status_code=409,
                content={
                    "ok": False,
                    "error":
                        "Bot is not running. "
                        "Start the bot first.",
                },
            )

        # New bot implementation
        if hasattr(
            bot,
            "request_scan",
        ):

            result = bot.request_scan()

            return JSONResponse(
                status_code=202,
                content=json_safe({
                    "ok": True,

                    "queued":
                        bool(
                            result.get(
                                "accepted",
                                False,
                            )
                        )
                        if isinstance(
                            result,
                            dict,
                        )
                        else True,

                    "already_scanning":
                        bool(
                            result.get(
                                "already_scanning",
                                False,
                            )
                        )
                        if isinstance(
                            result,
                            dict,
                        )
                        else False,

                    "message":
                        "Full-market scan queued. "
                        "The running bot will execute it.",
                })
            )

        # ----------------------------------------------------
        # Compatibility fallback
        # ----------------------------------------------------

        return JSONResponse(
            status_code=503,
            content={
                "ok": False,
                "error":
                    "Bot scan queue is unavailable. "
                    "Update bot.py first.",
            },
        )

    except Exception as exc:

        traceback.print_exc()

        return JSONResponse(
            status_code=500,
            content=json_safe({
                "ok": False,
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

        if not isinstance(
            data,
            list,
        ):

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
                "error":
                    str(exc),
            })
        )


# ============================================================
# EQUITY
# ============================================================

@app.get("/api/equity")
def equity():

    try:

        data = db.equity_history(500)

        if not isinstance(
            data,
            list,
        ):

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
                "error":
                    str(exc),
            })
        )


# ============================================================
# PERFORMANCE
# ============================================================

@app.get("/api/performance")
def performance():

    try:

        statistics = db.stats()

        if not isinstance(
            statistics,
            dict,
        ):

            statistics = {}

        return JSONResponse(
            content=json_safe({
                "ok": True,

                "stats":
                    statistics,

                **statistics,
            })
        )

    except Exception as exc:

        return JSONResponse(
            content=json_safe({
                "ok": False,
                "stats": {},
                "error":
                    str(exc),
            })
        )


# ============================================================
# SCANNER
# ============================================================

@app.get("/api/scanner")
def scanner():

    try:

        data = bot.scanner.status()

        if not isinstance(
            data,
            dict,
        ):

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
                "error":
                    str(exc),
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

            "mode":
                current_mode(),
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
            "error":
                str(exc),
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

            "mode":
                current_mode(),

            "live_orders_enabled":
                bot.kraken.live_orders_enabled,

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
