from __future__ import annotations

import asyncio
import math
import threading
import time
import traceback

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from bot import KrakenBot
from config import settings
import db


# ============================================================
# APP
# ============================================================

app = FastAPI(
    title="Kraken Day Trader",
    version="1.0.0",
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

db.init_db()


# ============================================================
# BOT
# ============================================================

bot = KrakenBot()

_worker_thread = None
_worker_error = None
_worker_lock = threading.Lock()


# ============================================================
# HELPERS
# ============================================================

def current_mode():
    try:
        return bot.kraken.mode
    except Exception:
        return "PAPER"


def paper_balance():
    """
    Read the current paper account balance.

    Falls back safely if the database implementation does not
    expose the expected helper.
    """

    try:
        if hasattr(db, "paper_balance"):
            data = db.paper_balance()

            return {
                "equity": float(data.get("equity", 0)),
                "cash": float(data.get("cash", 0)),
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

    return {
        "equity": float(
            getattr(
                settings,
                "paper_start_balance",
                1000,
            )
        ),
        "cash": float(
            getattr(
                settings,
                "paper_start_balance",
                1000,
            )
        ),
    }


def safe_float(value, default=0.0):
    try:
        result = float(value)

        if math.isfinite(result):
            return result

    except Exception:
        pass

    return float(default)


# ============================================================
# BOT WORKER
# ============================================================

def start_bot_thread():
    global _worker_thread
    global _worker_error

    with _worker_lock:
        if (
            _worker_thread is not None
            and _worker_thread.is_alive()
        ):
            return False

        _worker_error = None

        def runner():
            global _worker_error

            try:
                asyncio.run(bot.run())

            except Exception as exc:
                _worker_error = (
                    f"{type(exc).__name__}: {exc}"
                )

                traceback.print_exc()

        _worker_thread = threading.Thread(
            target=runner,
            name="kraken-bot-worker",
            daemon=True,
        )

        _worker_thread.start()

        return True


def worker_alive():
    return bool(
        _worker_thread is not None
        and _worker_thread.is_alive()
    )


# ============================================================
# STARTUP
# ============================================================

@app.on_event("startup")
async def startup_event():

    # --------------------------------------------------------
    # ALWAYS START IN PAPER MODE
    # --------------------------------------------------------

    try:
        bot.kraken.set_mode("PAPER")
    except Exception as exc:
        print(
            "Unable to force PAPER mode at startup:",
            exc,
        )

    # --------------------------------------------------------
    # DATABASE
    # --------------------------------------------------------

    try:
        db.init_db()
    except Exception as exc:
        print(
            "Database initialization warning:",
            exc,
        )

    # --------------------------------------------------------
    # KRAKEN CONNECTION
    # --------------------------------------------------------

    try:
        connection = bot.kraken.test_connection()

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
    # KRAKEN AUTHENTICATION
    # --------------------------------------------------------

    try:
        authentication = (
            bot.kraken.test_authentication()
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
    # AUTONOMOUS BOT
    # --------------------------------------------------------

    if (
        getattr(settings, "autonomous", False)
        and bot.kraken.authenticated
    ):
        started = start_bot_thread()

        print(
            "Autonomous bot startup:",
            "STARTED" if started else "ALREADY RUNNING",
        )

    else:
        print(
            "Autonomous bot startup skipped.",
            "autonomous=",
            getattr(settings, "autonomous", False),
            "authenticated=",
            bot.kraken.authenticated,
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
            "error": f"{type(exc).__name__}: {exc}",
        },
    )


# ============================================================
# STATUS
# ============================================================

@app.get("/api/status")
async def api_status():

    mode = current_mode()

    try:
        scanner = bot.scanner_status()
    except Exception:
        scanner = {}

    try:
        stats = bot.stats()
    except Exception:
        stats = {}

    try:
        signals = bot.get_signals()
    except Exception:
        signals = []

    try:
        positions = bot.get_positions()
    except Exception:
        positions = []

    try:
        equity_history = db.equity_history(500)
    except Exception:
        equity_history = []

    return {
        "ok": True,

        "running": worker_alive(),

        "worker_alive": worker_alive(),

        "worker_error": _worker_error,

        "mode": mode,

        "paper": mode == "PAPER",

        "live": mode == "LIVE",

        "live_orders_enabled": bool(
            bot.kraken.live_orders_enabled
        ),

        "autonomous": bool(
            getattr(settings, "autonomous", False)
        ),

        "bot": {
            "running": worker_alive(),
            "mode": mode,
            "paper": mode == "PAPER",
            "live": mode == "LIVE",
            "live_orders_enabled": bool(
                bot.kraken.live_orders_enabled
            ),
        },

        "kraken": bot.kraken.connection_status(),

        "account": {
            "paper_equity": safe_float(
                paper_balance()["equity"]
            ),
            "paper_cash": safe_float(
                paper_balance()["cash"]
            ),
            "live": mode == "LIVE",
        },

        "scanner": scanner,

        "stats": stats,

        "signals": signals,

        "positions": positions,

        "equity": equity_history,

        "last_scan": getattr(
            bot,
            "last_scan",
            None,
        ),
    }


# ============================================================
# TRADING MODE - GET
# ============================================================

@app.get("/api/trading-mode")
async def get_trading_mode():

    mode = current_mode()

    return {
        "ok": True,

        "mode": mode,

        "paper": mode == "PAPER",

        "live": mode == "LIVE",

        "live_orders_enabled": bool(
            bot.kraken.live_orders_enabled
        ),

        "authenticated": bool(
            bot.kraken.authenticated
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

        "kraken": bot.kraken.connection_status(),
    }


# ============================================================
# TRADING MODE - POST
# ============================================================

@app.post("/api/trading-mode")
async def set_trading_mode(request: Request):

    body = await request.json()

    requested_mode = str(
        body.get("mode", "")
    ).upper().strip()

    # --------------------------------------------------------
    # PAPER
    # --------------------------------------------------------

    if requested_mode == "PAPER":

        bot.kraken.set_mode("PAPER")

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
            body.get("confirmation", "")
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

        # ----------------------------------------------------
        # Check configuration
        # ----------------------------------------------------

        if not bool(
            getattr(
                settings,
                "live_trading",
                False,
            )
        ):
            raise RuntimeError(
                "LIVE trading is disabled by configuration. "
                "Set LIVE_TRADING=true before enabling LIVE mode."
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
                "Set DRY_RUN=false before enabling LIVE mode."
            )

        # ----------------------------------------------------
        # Re-authenticate before enabling live orders
        # ----------------------------------------------------

        authentication = (
            bot.kraken.test_authentication()
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

        # ----------------------------------------------------
        # Get actual account summary
        # ----------------------------------------------------

        account = bot.kraken.account_summary(
            "USD"
        )

        if account.get("error"):
            raise RuntimeError(
                account["error"]
            )

        # ----------------------------------------------------
        # NOW enable LIVE runtime mode
        # ----------------------------------------------------

        bot.kraken.set_mode("LIVE")

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
                bot.kraken.live_orders_enabled
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
                        account.get("total", 0),
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

    # --------------------------------------------------------
    # Invalid mode
    # --------------------------------------------------------

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

    mode = current_mode()

    # --------------------------------------------------------
    # PAPER
    # --------------------------------------------------------

    if mode == "PAPER":

        paper = paper_balance()

        equity = safe_float(
            paper.get("equity", 0)
        )

        cash = safe_float(
            paper.get("cash", 0)
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

        account = bot.kraken.account_summary(
            "USD"
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
                account.get("total", 0),
            )
        )

        return {
            "ok": True,

            "mode": "LIVE",

            "source": "kraken",

            # Full portfolio value.
            "equity": portfolio_value,

            "total": portfolio_value,

            "portfolio_value_usd": portfolio_value,

            # Actual spendable USD.
            "cash": available_usd,

            "free": available_usd,

            "usd_free": available_usd,

            # Individual balances.
            "usd_total": usd_total,

            "usdg_total": usdg_total,

            "portfolio_assets": account.get(
                "portfolio_assets",
                [],
            ),

            "live_orders_enabled": bool(
                bot.kraken.live_orders_enabled
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

    started = start_bot_thread()

    return {
        "ok": True,

        "started": started,

        "running": worker_alive(),

        "mode": current_mode(),

        "message": (
            "Bot started."
            if started
            else "Bot is already running."
        ),
    }


# ============================================================
# STOP
# ============================================================

@app.post("/api/stop")
async def api_stop():

    try:
        if hasattr(bot, "stop"):
            result = bot.stop()
        elif hasattr(bot, "request_stop"):
            result = bot.request_stop()
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

    try:

        if not worker_alive():
            return JSONResponse(
                status_code=503,
                content={
                    "ok": False,
                    "error": (
                        "Bot worker is offline. "
                        "Start the bot first."
                    ),
                },
            )

        # Preferred method.
        if hasattr(bot, "request_scan"):

            result = bot.request_scan()

            return JSONResponse(
                status_code=202,
                content={
                    "ok": True,
                    "queued": True,
                    "already_scanning": bool(
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

    try:
        signals = bot.get_signals()

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

    try:
        positions = bot.get_positions()

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

    try:
        stats = bot.stats()

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

    try:
        scanner = bot.scanner_status()

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

    connection = bot.kraken.test_connection()

    authentication = bot.kraken.test_authentication()

    return {
        "ok": bool(
            connection.get("ok")
            and authentication.get("ok")
        ),

        "connection": connection,

        "authentication": authentication,

        "kraken": bot.kraken.connection_status(),
    }


# ============================================================
# DEBUG
# ============================================================

@app.get("/api/debug")
async def api_debug():

    return {
        "ok": True,

        "worker_alive": worker_alive(),

        "worker_error": _worker_error,

        "mode": current_mode(),

        "kraken": bot.kraken.connection_status(),

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

@app.get("/", response_class=HTMLResponse)
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
# HEALTH CHECK
# ============================================================

@app.get("/health")
async def health():

    return {
        "ok": True,
        "status": "online",
        "worker_alive": worker_alive(),
        "mode": current_mode(),
    }
