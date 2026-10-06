from __future__ import annotations

import asyncio
import math
import threading
import time
import traceback

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse

from bot import KrakenBot
from config import settings
import db


# ============================================================
# APP
# ============================================================

app = FastAPI(
    title="Kraken Day Trader",
    version="1.0.1",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# DATABASE
# ============================================================

try:
    db.init_db()
except Exception as exc:
    print("Initial database setup warning:", exc)


# ============================================================
# BOT STATE
# ============================================================

_bot_lock = threading.RLock()
_worker_lock = threading.Lock()

bot: KrakenBot | None = None

_worker_thread: threading.Thread | None = None
_worker_error: str | None = None
_worker_started_at: float | None = None
_worker_finished_at: float | None = None


# ============================================================
# BOT CREATION
# ============================================================

def create_bot() -> KrakenBot:
    """
    Always create a fresh bot instance.

    This is important because a previous bot instance may have
    an internal stop event or failed state that prevents a
    subsequent Start command from working.
    """

    new_bot = KrakenBot()

    try:
        new_bot.kraken.set_mode("PAPER")
    except Exception as exc:
        print("Unable to initialize bot in PAPER mode:", exc)

    return new_bot


def get_bot() -> KrakenBot:
    global bot

    with _bot_lock:

        if bot is None:
            bot = create_bot()

        return bot


# Create initial instance.
bot = create_bot()


# ============================================================
# HELPERS
# ============================================================

def current_mode():
    try:
        return get_bot().kraken.mode
    except Exception:
        return "PAPER"


def paper_balance():

    current_bot = get_bot()

    try:

        if hasattr(db, "paper_balance"):

            data = db.paper_balance()

            return {
                "equity": float(
                    data.get("equity", 0)
                ),

                "cash": float(
                    data.get("cash", 0)
                ),
            }

    except Exception:
        pass

    try:

        history = db.equity_history(1)

        if history:

            last = history[-1]

            equity = float(
                last.get("equity")
                or last.get("value")
                or 0
            )

            return {
                "equity": equity,
                "cash": equity,
            }

    except Exception:
        pass

    starting_balance = float(
        getattr(
            settings,
            "paper_start_balance",
            1000,
        )
    )

    return {
        "equity": starting_balance,
        "cash": starting_balance,
    }


def safe_float(value, default=0.0):

    try:

        result = float(value)

        if math.isfinite(result):
            return result

    except Exception:
        pass

    return float(default)


def worker_alive():

    thread = _worker_thread

    return bool(
        thread is not None
        and thread.is_alive()
    )


# ============================================================
# WORKER
# ============================================================

def _run_bot_worker(worker_bot: KrakenBot):

    global _worker_error
    global _worker_finished_at

    try:

        print(
            "=================================================="
        )
        print("KRAKEN BOT WORKER STARTING")
        print(
            "Mode:",
            worker_bot.kraken.mode,
        )
        print(
            "Authenticated:",
            worker_bot.kraken.authenticated,
        )
        print(
            "Live orders:",
            worker_bot.kraken.live_orders_enabled,
        )
        print(
            "=================================================="
        )

        asyncio.run(
            worker_bot.run()
        )

        # If run() returns normally, record it.
        # This is useful because a trading worker normally
        # should remain alive until explicitly stopped.

        _worker_error = (
            "Bot worker exited normally. "
            "bot.run() returned."
        )

        print(
            "WARNING: bot.run() returned and the worker stopped."
        )

    except Exception as exc:

        _worker_error = (
            f"{type(exc).__name__}: {exc}"
        )

        print(
            "=================================================="
        )
        print("KRAKEN BOT WORKER CRASHED")
        print(
            f"{type(exc).__name__}: {exc}"
        )
        print(
            "=================================================="
        )

        traceback.print_exc()

    finally:

        _worker_finished_at = time.time()

        print(
            "Kraken bot worker finished."
        )


def start_bot_thread():

    global bot
    global _worker_thread
    global _worker_error
    global _worker_started_at
    global _worker_finished_at

    with _worker_lock:

        # ----------------------------------------------------
        # Already running
        # ----------------------------------------------------

        if (
            _worker_thread is not None
            and _worker_thread.is_alive()
        ):

            return {
                "started": False,
                "running": True,
                "message": "Bot is already running.",
            }

        # ----------------------------------------------------
        # Create a completely fresh bot instance.
        # ----------------------------------------------------

        with _bot_lock:

            old_bot = bot

            # Preserve the current requested mode.
            try:
                previous_mode = old_bot.kraken.mode
            except Exception:
                previous_mode = "PAPER"

            bot = create_bot()

            new_bot = bot

        # ----------------------------------------------------
        # Authenticate the fresh bot.
        # ----------------------------------------------------

        try:

            authentication = (
                new_bot.kraken.test_authentication()
            )

            print(
                "Start authentication:",
                authentication,
            )

        except Exception as exc:

            _worker_error = (
                f"{type(exc).__name__}: "
                f"Authentication failed: {exc}"
            )

            return {
                "started": False,
                "running": False,
                "error": _worker_error,
            }

        # ----------------------------------------------------
        # Restore LIVE mode only when it was already enabled.
        # ----------------------------------------------------

        if previous_mode == "LIVE":

            try:

                new_bot.kraken.set_mode("LIVE")

                print(
                    "Restored LIVE trading mode."
                )

            except Exception as exc:

                print(
                    "Could not restore LIVE mode:",
                    exc,
                )

                # Safety first: remain PAPER.
                try:
                    new_bot.kraken.set_mode(
                        "PAPER"
                    )
                except Exception:
                    pass

        else:

            try:
                new_bot.kraken.set_mode(
                    "PAPER"
                )
            except Exception:
                pass

        # ----------------------------------------------------
        # Clear previous worker state.
        # ----------------------------------------------------

        _worker_error = None
        _worker_finished_at = None
        _worker_started_at = time.time()

        # ----------------------------------------------------
        # Create worker.
        # ----------------------------------------------------

        thread = threading.Thread(
            target=_run_bot_worker,
            args=(new_bot,),
            name="kraken-bot-worker",
            daemon=True,
        )

        _worker_thread = thread

        thread.start()

    # --------------------------------------------------------
    # Give the thread a short opportunity to start.
    # This catches immediate crashes instead of reporting
    # "started" when it actually died instantly.
    # --------------------------------------------------------

    deadline = time.time() + 1.5

    while time.time() < deadline:

        if worker_alive():

            return {
                "started": True,
                "running": True,
                "message": "Bot started.",
            }

        if _worker_error:

            return {
                "started": False,
                "running": False,
                "error": _worker_error,
            }

        time.sleep(0.05)

    return {
        "started": worker_alive(),
        "running": worker_alive(),
        "message": (
            "Bot started."
            if worker_alive()
            else "Bot worker stopped immediately."
        ),
        "worker_error": _worker_error,
    }


# ============================================================
# STARTUP
# ============================================================

@app.on_event("startup")
async def startup_event():

    global bot

    print(
        "=================================================="
    )
    print("KRAKEN BOT STARTUP")
    print(
        "=================================================="
    )

    # --------------------------------------------------------
    # Database
    # --------------------------------------------------------

    try:

        db.init_db()

        print(
            "Database: READY"
        )

    except Exception as exc:

        print(
            "Database initialization warning:",
            exc,
        )

    # --------------------------------------------------------
    # Fresh bot
    # --------------------------------------------------------

    with _bot_lock:

        bot = create_bot()

        current_bot = bot

    # --------------------------------------------------------
    # Force PAPER after deployment/restart.
    # --------------------------------------------------------

    try:

        current_bot.kraken.set_mode(
            "PAPER"
        )

        print(
            "Trading mode: PAPER"
        )

    except Exception as exc:

        print(
            "Unable to force PAPER mode:",
            exc,
        )

    # --------------------------------------------------------
    # Kraken connection
    # --------------------------------------------------------

    try:

        connection = (
            current_bot.kraken.test_connection()
        )

        print(
            "Kraken connection:",
            connection,
        )

    except Exception as exc:

        print(
            "Kraken connection test failed:",
            exc,
        )

    # --------------------------------------------------------
    # Kraken authentication
    # --------------------------------------------------------

    try:

        authentication = (
            current_bot.kraken.test_authentication()
        )

        print(
            "Kraken authentication:",
            authentication,
        )

    except Exception as exc:

        print(
            "Kraken authentication test failed:",
            exc,
        )

    # --------------------------------------------------------
    # Autonomous mode
    # --------------------------------------------------------

    if (
        getattr(
            settings,
            "autonomous",
            False,
        )
        and current_bot.kraken.authenticated
    ):

        result = start_bot_thread()

        print(
            "Autonomous bot startup:",
            result,
        )

    else:

        print(
            "Autonomous bot startup skipped."
        )

        print(
            "autonomous=",
            getattr(
                settings,
                "autonomous",
                False,
            ),
        )

        print(
            "authenticated=",
            current_bot.kraken.authenticated,
        )

    print(
        "=================================================="
    )


# ============================================================
# GLOBAL ERROR HANDLER
# ============================================================

@app.exception_handler(Exception)
async def global_exception_handler(
    request: Request,
    exc: Exception,
):

    traceback.print_exc()

    return JSONResponse(
        status_code=200,
        content={
            "ok": False,
            "error": (
                f"{type(exc).__name__}: {exc}"
            ),
        },
    )


# ============================================================
# STATUS
# ============================================================

@app.get("/api/status")
async def api_status():

    current_bot = get_bot()

    mode = current_mode()

    try:
        scanner = current_bot.scanner_status()
    except Exception:
        scanner = {}

    try:
        stats = current_bot.stats()
    except Exception:
        stats = {}

    try:
        signals = current_bot.get_signals()
    except Exception:
        signals = []

    try:
        positions = current_bot.get_positions()
    except Exception:
        positions = []

    try:
        equity_history = db.equity_history(500)
    except Exception:
        equity_history = []

    paper = paper_balance()

    return {
        "ok": True,

        "running": worker_alive(),

        "worker_alive": worker_alive(),

        "worker_error": _worker_error,

        "worker_started_at": _worker_started_at,

        "worker_finished_at": _worker_finished_at,

        "mode": mode,

        "paper": mode == "PAPER",

        "live": mode == "LIVE",

        "live_orders_enabled": bool(
            current_bot.kraken.live_orders_enabled
        ),

        "autonomous": bool(
            getattr(
                settings,
                "autonomous",
                False,
            )
        ),

        "bot": {
            "running": worker_alive(),
            "mode": mode,
            "paper": mode == "PAPER",
            "live": mode == "LIVE",
            "live_orders_enabled": bool(
                current_bot.kraken.live_orders_enabled
            ),
        },

        "kraken": (
            current_bot.kraken.connection_status()
        ),

        "account": {
            "paper_equity": safe_float(
                paper["equity"]
            ),

            "paper_cash": safe_float(
                paper["cash"]
            ),

            "live": mode == "LIVE",
        },

        "scanner": scanner,

        "stats": stats,

        "signals": signals,

        "positions": positions,

        "equity": equity_history,

        "last_scan": getattr(
            current_bot,
            "last_scan",
            None,
        ),
    }


# ============================================================
# TRADING MODE - GET
# ============================================================

@app.get("/api/trading-mode")
async def get_trading_mode():

    current_bot = get_bot()

    mode = current_mode()

    return {
        "ok": True,

        "mode": mode,

        "paper": mode == "PAPER",

        "live": mode == "LIVE",

        "live_orders_enabled": bool(
            current_bot.kraken.live_orders_enabled
        ),

        "authenticated": bool(
            current_bot.kraken.authenticated
        ),

        "config": {
            "live_trading": bool(
                getattr(
                    settings,
                    "live_trading",
                    False,
                )
            ),

            "dry_run": bool(
                getattr(
                    settings,
                    "dry_run",
                    True,
                )
            ),

            "autonomous": bool(
                getattr(
                    settings,
                    "autonomous",
                    False,
                )
            ),
        },

        "kraken": (
            current_bot.kraken.connection_status()
        ),
    }


# ============================================================
# TRADING MODE - POST
# ============================================================

@app.post("/api/trading-mode")
async def set_trading_mode(
    request: Request,
):

    current_bot = get_bot()

    body = await request.json()

    requested_mode = str(
        body.get("mode", "")
    ).upper().strip()

    # --------------------------------------------------------
    # PAPER
    # --------------------------------------------------------

    if requested_mode == "PAPER":

        current_bot.kraken.set_mode(
            "PAPER"
        )

        return {
            "ok": True,
            "mode": "PAPER",
            "paper": True,
            "live": False,
            "live_orders_enabled": False,
            "message": "PAPER trading enabled.",
        }

    # --------------------------------------------------------
    # LIVE
    # --------------------------------------------------------

    if requested_mode == "LIVE":

        confirmation = str(
            body.get(
                "confirmation",
                "",
            )
        ).strip()

        if confirmation != "ENABLE LIVE":

            return JSONResponse(
                status_code=400,
                content={
                    "ok": False,
                    "error": (
                        'Live trading requires confirmation '
                        '"ENABLE LIVE".'
                    ),
                },
            )

        if not bool(
            getattr(
                settings,
                "live_trading",
                False,
            )
        ):

            raise RuntimeError(
                "LIVE trading is disabled by configuration. "
                "Set LIVE_TRADING=true."
            )

        if bool(
            getattr(
                settings,
                "dry_run",
                True,
            )
        ):

            raise RuntimeError(
                "LIVE trading is blocked because DRY_RUN=true. "
                "Set DRY_RUN=false."
            )

        authentication = (
            current_bot.kraken.test_authentication()
        )

        if not authentication.get(
            "authenticated",
            False,
        ):

            raise RuntimeError(
                authentication.get(
                    "error",
                    "Kraken authentication failed.",
                )
            )

        account = (
            current_bot.kraken.account_summary(
                "USD"
            )
        )

        if account.get("error"):

            raise RuntimeError(
                account["error"]
            )

        current_bot.kraken.set_mode(
            "LIVE"
        )

        portfolio_value = safe_float(
            account.get(
                "portfolio_value_usd",
                account.get("total", 0),
            )
        )

        available_usd = safe_float(
            account.get(
                "usd_free",
                account.get("free", 0),
            )
        )

        return {
            "ok": True,

            "mode": "LIVE",

            "paper": False,

            "live": True,

            "live_orders_enabled": bool(
                current_bot.kraken.live_orders_enabled
            ),

            "authenticated": True,

            "message": (
                "LIVE trading enabled. "
                "The bot is authorized to submit "
                "real Kraken orders."
            ),

            "account": {
                "available_usd": available_usd,

                "free": available_usd,

                "usd_total": safe_float(
                    account.get(
                        "usd_total",
                        account.get(
                            "total",
                            0,
                        ),
                    )
                ),

                "usdg_total": safe_float(
                    account.get(
                        "usdg_total",
                        0,
                    )
                ),

                "portfolio_value_usd": portfolio_value,

                "total": portfolio_value,

                "assets": account.get(
                    "portfolio_assets",
                    [],
                ),
            },
        }

    return JSONResponse(
        status_code=400,
        content={
            "ok": False,
            "error": (
                "Invalid trading mode. "
                "Use PAPER or LIVE."
            ),
        },
    )


# ============================================================
# BALANCE
# ============================================================

@app.get("/api/balance")
async def api_balance():

    current_bot = get_bot()

    mode = current_mode()

    # --------------------------------------------------------
    # PAPER
    # --------------------------------------------------------

    if mode == "PAPER":

        paper = paper_balance()

        equity = safe_float(
            paper.get(
                "equity",
                0,
            )
        )

        cash = safe_float(
            paper.get(
                "cash",
                0,
            )
        )

        return {
            "ok": True,

            "mode": "PAPER",

            "source": "paper",

            "equity": equity,

            "cash": cash,

            "free": cash,

            "total": equity,

            "portfolio_value_usd": equity,

            "usd_free": cash,

            "usd_total": equity,

            "usdg_total": 0.0,
        }

    # --------------------------------------------------------
    # LIVE
    # --------------------------------------------------------

    try:

        account = (
            current_bot.kraken.account_summary(
                "USD"
            )
        )

        if account.get("error"):

            return {
                "ok": False,
                "mode": "LIVE",
                "source": "kraken",
                "error": account["error"],
            }

        available_usd = safe_float(
            account.get(
                "usd_free",
                account.get("free", 0),
            )
        )

        usd_total = safe_float(
            account.get(
                "usd_total",
                account.get("total", 0),
            )
        )

        usdg_total = safe_float(
            account.get(
                "usdg_total",
                0,
            )
        )

        portfolio_value = safe_float(
            account.get(
                "portfolio_value_usd",
                account.get(
                    "total",
                    0,
                ),
            )
        )

        return {
            "ok": True,

            "mode": "LIVE",

            "source": "kraken",

            "equity": portfolio_value,

            "total": portfolio_value,

            "portfolio_value_usd": portfolio_value,

            "cash": available_usd,

            "free": available_usd,

            "usd_free": available_usd,

            "usd_total": usd_total,

            "usdg_total": usdg_total,

            "portfolio_assets": account.get(
                "portfolio_assets",
                [],
            ),

            "live_orders_enabled": bool(
                current_bot.kraken.live_orders_enabled
            ),
        }

    except Exception as exc:

        return {
            "ok": False,

            "mode": "LIVE",

            "source": "kraken",

            "error": (
                f"{type(exc).__name__}: {exc}"
            ),
        }


# ============================================================
# START
# ============================================================

@app.post("/api/start")
async def api_start():

    result = start_bot_thread()

    return {
        "ok": bool(
            result.get(
                "started",
                False,
            )
            or result.get(
                "running",
                False,
            )
        ),

        **result,
    }


# ============================================================
# STOP
# ============================================================

@app.post("/api/stop")
async def api_stop():

    current_bot = get_bot()

    try:

        if hasattr(
            current_bot,
            "stop",
        ):

            result = (
                current_bot.stop()
            )

        elif hasattr(
            current_bot,
            "request_stop",
        ):

            result = (
                current_bot.request_stop()
            )

        else:

            result = None

    except Exception as exc:

        return {
            "ok": False,
            "error": (
                f"{type(exc).__name__}: {exc}"
            ),
        }

    return {
        "ok": True,

        "running": worker_alive(),

        "result": result,

        "message": "Bot stop requested.",
    }


# ============================================================
# SCAN
# ============================================================

@app.post("/api/scan")
async def api_scan():

    current_bot = get_bot()

    try:

        if not worker_alive():

            return JSONResponse(
                status_code=503,
                content={
                    "ok": False,
                    "error": (
                        "Bot worker is offline. "
                        "Start the bot first.",
                    ),
                    "worker_error": _worker_error,
                },
            )

        if hasattr(
            current_bot,
            "request_scan",
        ):

            result = (
                current_bot.request_scan()
            )

            return JSONResponse(
                status_code=202,
                content={
                    "ok": True,
                    "queued": True,
                    "already_scanning": (
                        result is False
                    ),
                    "message": (
                        "Full-market scan queued."
                    ),
                },
            )

        return JSONResponse(
            status_code=503,
            content={
                "ok": False,
                "error": (
                    "Scanner queue is unavailable."
                ),
            },
        )

    except Exception as exc:

        return {
            "ok": False,
            "error": (
                f"{type(exc).__name__}: {exc}"
            ),
        }


# ============================================================
# SIGNALS
# ============================================================

@app.get("/api/signals")
async def api_signals():

    current_bot = get_bot()

    try:

        signals = (
            current_bot.get_signals()
        )

    except Exception as exc:

        return {
            "ok": False,
            "signals": [],
            "error": (
                f"{type(exc).__name__}: {exc}"
            ),
        }

    return {
        "ok": True,
        "signals": signals,
    }


# ============================================================
# POSITIONS
# ============================================================

@app.get("/api/positions")
async def api_positions():

    current_bot = get_bot()

    try:

        positions = (
            current_bot.get_positions()
        )

    except Exception as exc:

        return {
            "ok": False,
            "positions": [],
            "error": (
                f"{type(exc).__name__}: {exc}"
            ),
        }

    return {
        "ok": True,
        "positions": positions,
    }


# ============================================================
# EQUITY
# ============================================================

@app.get("/api/equity")
async def api_equity():

    try:

        history = db.equity_history(500)

    except Exception as exc:

        return {
            "ok": False,
            "history": [],
            "error": (
                f"{type(exc).__name__}: {exc}"
            ),
        }

    return {
        "ok": True,
        "history": history,
    }


# ============================================================
# PERFORMANCE
# ============================================================

@app.get("/api/performance")
async def api_performance():

    current_bot = get_bot()

    try:

        stats = (
            current_bot.stats()
        )

    except Exception as exc:

        return {
            "ok": False,
            "error": (
                f"{type(exc).__name__}: {exc}"
            ),
        }

    return {
        "ok": True,
        "stats": stats,
    }


# ============================================================
# SCANNER STATUS
# ============================================================

@app.get("/api/scanner")
async def api_scanner():

    current_bot = get_bot()

    try:

        scanner = (
            current_bot.scanner_status()
        )

    except Exception as exc:

        return {
            "ok": False,
            "scanner": {},
            "error": (
                f"{type(exc).__name__}: {exc}"
            ),
        }

    return {
        "ok": True,
        "scanner": scanner,
    }


# ============================================================
# KRAKEN TEST
# ============================================================

@app.post("/api/kraken/test")
async def api_kraken_test():

    current_bot = get_bot()

    connection = (
        current_bot.kraken.test_connection()
    )

    authentication = (
        current_bot.kraken.test_authentication()
    )

    return {
        "ok": bool(
            connection.get("ok")
            and authentication.get("ok")
        ),

        "connection": connection,

        "authentication": authentication,

        "kraken": (
            current_bot.kraken.connection_status()
        ),
    }


# ============================================================
# DEBUG
# ============================================================

@app.get("/api/debug")
async def api_debug():

    current_bot = get_bot()

    return {
        "ok": True,

        "worker_alive": worker_alive(),

        "worker_error": _worker_error,

        "worker_started_at": (
            _worker_started_at
        ),

        "worker_finished_at": (
            _worker_finished_at
        ),

        "mode": current_mode(),

        "kraken": (
            current_bot.kraken.connection_status()
        ),

        "config": {
            "live_trading": bool(
                getattr(
                    settings,
                    "live_trading",
                    False,
                )
            ),

            "dry_run": bool(
                getattr(
                    settings,
                    "dry_run",
                    True,
                )
            ),

            "autonomous": bool(
                getattr(
                    settings,
                    "autonomous",
                    False,
                )
            ),

            "max_trade_usd": safe_float(
                getattr(
                    settings,
                    "max_trade_usd",
                    0,
                )
            ),

            "max_position_pct": safe_float(
                getattr(
                    settings,
                    "max_position_pct",
                    0,
                )
            ),

            "min_expected_move": safe_float(
                getattr(
                    settings,
                    "min_expected_move",
                    0,
                )
            ),
        },
    }


# ============================================================
# DASHBOARD
# ============================================================

@app.get(
    "/",
    response_class=HTMLResponse,
)
async def dashboard():

    try:

        with open(
            "index.html",
            "r",
            encoding="utf-8",
        ) as file:

            return HTMLResponse(
                file.read()
            )

    except FileNotFoundError:

        return HTMLResponse(
            """
            <html>
                <body
                    style="
                        background:#0b0b0f;
                        color:white;
                        font-family:Arial;
                        padding:40px;
                    "
                >
                    <h1>Kraken Bot</h1>
                    <p>index.html was not found.</p>
                </body>
            </html>
            """,
            status_code=200,
        )


# ============================================================
# HEALTH
# ============================================================

@app.get("/health")
async def health():

    return {
        "ok": True,

        "status": (
            "online"
            if worker_alive()
            else "bot_stopped"
        ),

        "worker_alive": worker_alive(),

        "worker_error": _worker_error,

        "mode": current_mode(),
    }
