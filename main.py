from __future__ import annotations

import asyncio
import logging
import threading
import time
import traceback
from typing import Any

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from bot import KrakenBot
from config import settings
import db


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

logger = logging.getLogger("kraken-day-trader")


# ============================================================
# APP
# ============================================================

app = FastAPI(
    title="Kraken Day Trader",
    version="3.0.0",
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# GLOBAL STATE
# ============================================================

bot_lock = threading.RLock()
worker_lock = threading.Lock()

current_bot: KrakenBot | None = None
worker_thread: threading.Thread | None = None

_worker_running = False
_worker_error: str | None = None
_worker_started_at: float = 0.0
_worker_stopped_at: float = 0.0

_last_start_attempt = 0.0


# ============================================================
# SAFE HELPERS
# ============================================================

def now_ts() -> float:
    return time.time()


def json_safe(value: Any) -> Any:
    """
    Convert common bot/database values into JSON-safe objects.
    """
    if value is None:
        return None

    if isinstance(value, (str, int, float, bool)):
        return value

    if isinstance(value, dict):
        return {
            str(k): json_safe(v)
            for k, v in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]

    try:
        return float(value)
    except Exception:
        return str(value)


def call_method(
    obj: Any,
    method_name: str,
    default: Any = None,
    *args,
    **kwargs,
):
    """
    Safely call a method if it exists.

    This keeps the API alive even if an optional dashboard
    method is unavailable.
    """
    try:
        method = getattr(obj, method_name, None)

        if not callable(method):
            return default

        return method(*args, **kwargs)

    except Exception:
        logger.exception(
            "Error calling %s.%s",
            type(obj).__name__,
            method_name,
        )
        return default


def bot_is_running(bot: Any) -> bool:
    if bot is None:
        return False

    try:
        return bool(getattr(bot, "running", False))
    except Exception:
        return False


def worker_alive() -> bool:
    global worker_thread

    return bool(
        worker_thread is not None
        and worker_thread.is_alive()
    )


# ============================================================
# BOT CREATION
# ============================================================

def create_bot() -> KrakenBot:
    """
    Always create a fresh bot in PAPER mode.

    IMPORTANT:
    We intentionally do NOT restore LIVE mode after a
    Railway restart/deployment.
    """

    bot = KrakenBot()

    # Force safe startup mode whenever possible.
    try:
        if hasattr(bot, "force_paper_mode"):
            bot.force_paper_mode()
        elif hasattr(bot, "trader"):
            trader = getattr(bot, "trader", None)

            if trader is not None:
                if hasattr(trader, "force_paper_mode"):
                    trader.force_paper_mode()
                elif hasattr(trader, "set_mode"):
                    trader.set_mode("PAPER")
    except Exception:
        logger.exception("Unable to force PAPER mode during bot creation.")

    return bot


# ============================================================
# GLOBAL BOT ACCESS
# ============================================================

def get_bot() -> KrakenBot:
    global current_bot

    with bot_lock:
        if current_bot is None:
            current_bot = create_bot()

        return current_bot


# ============================================================
# MODE HELPERS
# ============================================================

def get_trader(bot: Any):
    """
    Locate the Kraken trader/client regardless of whether the
    bot exposes it as trader, client, or kraken.
    """

    for attr in (
        "trader",
        "client",
        "kraken",
        "exchange",
    ):
        try:
            obj = getattr(bot, attr, None)

            if obj is not None:
                return obj

        except Exception:
            pass

    return None


def get_runtime_mode(bot: Any) -> str:
    trader = get_trader(bot)

    if trader is not None:
        try:
            status = trader.status()

            if isinstance(status, dict):
                mode = status.get("mode")

                if mode:
                    return str(mode).upper()

        except Exception:
            pass

        try:
            mode = getattr(trader, "mode", None)

            if mode:
                return str(mode).upper()

        except Exception:
            pass

    try:
        mode = getattr(bot, "mode", None)

        if mode:
            return str(mode).upper()

    except Exception:
        pass

    return "PAPER"


def force_paper(bot: Any) -> bool:
    """
    Hard safety function.

    Railway restarts must never silently resume LIVE trading.
    """

    try:
        trader = get_trader(bot)

        if trader is not None:

            if hasattr(trader, "force_paper_mode"):
                trader.force_paper_mode()
                return True

            if hasattr(trader, "set_mode"):
                result = trader.set_mode("PAPER")

                if result is False:
                    return False

                return True

        if hasattr(bot, "force_paper_mode"):
            bot.force_paper_mode()
            return True

        if hasattr(bot, "set_mode"):
            result = bot.set_mode("PAPER")

            if result is False:
                return False

            return True

    except Exception:
        logger.exception("Failed forcing PAPER mode.")

    return False


def set_live_mode(bot: Any) -> tuple[bool, str]:
    """
    LIVE mode requires explicit dashboard confirmation and
    all configuration safety gates.
    """

    if not bool(getattr(settings, "live_trading", False)):
        return (
            False,
            "LIVE_TRADING is disabled in Railway environment.",
        )

    if bool(getattr(settings, "dry_run", True)):
        return (
            False,
            "DRY_RUN is enabled. Disable DRY_RUN before live trading.",
        )

    trader = get_trader(bot)

    if trader is None:
        return False, "Kraken trader is unavailable."

    try:
        authenticated = bool(
            getattr(trader, "authenticated", False)
        )

        if not authenticated:
            try:
                trader.test_authentication()
                authenticated = bool(
                    getattr(trader, "authenticated", False)
                )
            except Exception:
                authenticated = False

        if not authenticated:
            return False, "Kraken authentication failed."

        if hasattr(trader, "set_mode"):
            result = trader.set_mode("LIVE")

            if result is False:
                return False, "Trader rejected LIVE mode."

            return True, "LIVE mode enabled."

        if hasattr(bot, "set_mode"):
            result = bot.set_mode("LIVE")

            if result is False:
                return False, "Bot rejected LIVE mode."

            return True, "LIVE mode enabled."

        return False, "LIVE mode control is unavailable."

    except Exception as exc:
        logger.exception("Failed enabling LIVE mode.")
        return False, str(exc)


# ============================================================
# AUTHENTICATION
# ============================================================

def test_kraken_connection(bot: Any) -> dict:
    trader = get_trader(bot)

    if trader is None:
        return {
            "ok": False,
            "connected": False,
            "authenticated": False,
            "error": "Kraken trader unavailable.",
        }

    result: dict[str, Any] = {
        "ok": False,
        "connected": False,
        "authenticated": False,
        "mode": get_runtime_mode(bot),
    }

    try:
        if hasattr(trader, "test_connection"):
            connection = trader.test_connection()

            if isinstance(connection, dict):
                result.update(connection)
            else:
                result["connected"] = bool(connection)

        else:
            result["connected"] = True

    except Exception as exc:
        result["connection_error"] = str(exc)

    try:
        if hasattr(trader, "test_authentication"):
            authentication = trader.test_authentication()

            if isinstance(authentication, dict):
                result.update(authentication)
            else:
                result["authenticated"] = bool(authentication)
        else:
            result["authenticated"] = bool(
                getattr(trader, "authenticated", False)
            )

    except Exception as exc:
        result["authentication_error"] = str(exc)

    result["ok"] = bool(
        result.get("connected", False)
        and result.get("authenticated", False)
    )

    result["mode"] = get_runtime_mode(bot)

    return json_safe(result)


# ============================================================
# WORKER
# ============================================================

def _run_bot_worker(bot: KrakenBot) -> None:
    global _worker_running
    global _worker_error
    global _worker_stopped_at

    logger.info("====================================================")
    logger.info("KRAKEN BOT WORKER STARTING")
    logger.info("====================================================")

    _worker_running = True
    _worker_error = None

    try:
        logger.info(
            "Runtime mode: %s",
            get_runtime_mode(bot),
        )

        trader = get_trader(bot)

        if trader is not None:
            logger.info(
                "Authenticated: %s",
                getattr(trader, "authenticated", False),
            )

            logger.info(
                "Live orders enabled: %s",
                getattr(trader, "live_orders_enabled", False),
            )

        # Bot.run() is responsible for:
        #
        # - continuous market scanning
        # - signal generation
        # - entering positions
        # - managing the current position
        # - rotation
        # - emergency stop
        # - risk limits
        #
        asyncio.run(bot.run())

        logger.warning(
            "Kraken bot worker exited normally."
        )

        _worker_error = (
            "Bot worker exited normally. "
            "Start the bot again if this was unexpected."
        )

    except Exception as exc:
        _worker_error = (
            f"{type(exc).__name__}: {exc}"
        )

        logger.error(
            "KRAKEN BOT WORKER CRASHED: %s",
            exc,
        )

        logger.error(
            traceback.format_exc()
        )

    finally:
        _worker_running = False
        _worker_stopped_at = now_ts()

        try:
            bot.running = False
        except Exception:
            pass

        logger.info(
            "Kraken bot worker stopped."
        )


def start_bot_thread() -> tuple[bool, str]:
    """
    Start exactly one bot worker.

    Every fresh application start begins PAPER.
    """

    global current_bot
    global worker_thread
    global _worker_running
    global _worker_error
    global _worker_started_at
    global _last_start_attempt

    with worker_lock:

        if worker_alive() or _worker_running:
            return False, "Bot is already running."

        # Prevent accidental double-click/start spam.
        current_time = now_ts()

        if current_time - _last_start_attempt < 2:
            return False, "Start request already processing."

        _last_start_attempt = current_time

        with bot_lock:

            # Create a fresh bot every time the worker is started.
            current_bot = create_bot()

            bot = current_bot

            # ------------------------------------------------
            # HARD SAFETY: ALWAYS PAPER ON NEW WORKER
            # ------------------------------------------------

            if not force_paper(bot):
                logger.warning(
                    "Could not explicitly force PAPER mode, "
                    "but worker will not enable LIVE automatically."
                )

            logger.info(
                "Worker startup mode forced to %s",
                get_runtime_mode(bot),
            )

            # ------------------------------------------------
            # KRAKEN AUTH
            # ------------------------------------------------

            try:
                trader = get_trader(bot)

                if trader is not None:
                    if hasattr(trader, "test_connection"):
                        trader.test_connection()

                    if hasattr(trader, "test_authentication"):
                        trader.test_authentication()

            except Exception as exc:
                logger.error(
                    "Kraken authentication failed: %s",
                    exc,
                )

                _worker_error = (
                    f"Kraken authentication failed: {exc}"
                )

                return False, _worker_error

            # ------------------------------------------------
            # CHECK AUTHENTICATION
            # ------------------------------------------------

            trader = get_trader(bot)

            authenticated = bool(
                getattr(
                    trader,
                    "authenticated",
                    False,
                )
            ) if trader is not None else False

            if not authenticated:
                _worker_error = (
                    "Kraken authentication failed. "
                    "Check KRAKEN_API_KEY and KRAKEN_API_SECRET."
                )

                logger.error(_worker_error)

                return False, _worker_error

            # ------------------------------------------------
            # CREATE WORKER
            # ------------------------------------------------

            _worker_error = None
            _worker_started_at = now_ts()

            worker_thread = threading.Thread(
                target=_run_bot_worker,
                args=(bot,),
                name="kraken-day-trader-worker",
                daemon=True,
            )

            worker_thread.start()

            # ------------------------------------------------
            # VERIFY THREAD STARTED
            # ------------------------------------------------

            time.sleep(0.25)

            if not worker_thread.is_alive():
                return (
                    False,
                    _worker_error
                    or "Bot worker stopped immediately.",
                )

            return (
                True,
                "Kraken bot started successfully in PAPER mode.",
            )


def stop_bot_thread() -> tuple[bool, str]:
    """
    Request a clean bot shutdown.
    """

    global current_bot

    with worker_lock:

        bot = current_bot

        if bot is None:
            return False, "Bot is not running."

        if not worker_alive() and not _worker_running:
            try:
                bot.running = False
            except Exception:
                pass

            return False, "Bot is already stopped."

        logger.info("Stopping Kraken bot...")

        try:
            if hasattr(bot, "request_stop"):
                bot.request_stop()
            elif hasattr(bot, "stop"):
                bot.stop()
            else:
                bot.running = False

        except Exception:
            logger.exception(
                "Error requesting bot shutdown."
            )

            try:
                bot.running = False
            except Exception:
                pass

        # Give the async worker a moment to exit.
        thread = worker_thread

        if thread is not None and thread.is_alive():
            thread.join(timeout=5)

        if thread is not None and thread.is_alive():
            logger.warning(
                "Bot worker did not exit within shutdown timeout."
            )

            return (
                False,
                "Stop requested, but worker is still shutting down.",
            )

        return True, "Kraken bot stopped."


# ============================================================
# BOT DATA
# ============================================================

def get_bot_stats(bot: Any) -> dict:
    stats = call_method(
        bot,
        "stats",
        default={},
    )

    if not isinstance(stats, dict):
        stats = {}

    # Add DB stats as a fallback/current source.
    try:
        db_stats = db.stats()

        if isinstance(db_stats, dict):
            for key, value in db_stats.items():
                stats.setdefault(key, value)

    except Exception:
        logger.exception("Unable to retrieve DB stats.")

    return json_safe(stats)


def get_bot_signals(bot: Any) -> list:
    signals = call_method(
        bot,
        "get_signals",
        default=[],
    )

    if signals is None:
        return []

    if isinstance(signals, dict):
        return [json_safe(signals)]

    if isinstance(signals, (list, tuple)):
        return [
            json_safe(signal)
            for signal in signals
        ]

    return []


def get_bot_positions(bot: Any) -> list:
    positions = call_method(
        bot,
        "get_positions",
        default=None,
    )

    if positions is None:
        positions = call_method(
            bot,
            "positions",
            default=[],
        )

    if positions is None:
        positions = []

    if isinstance(positions, dict):
        return [json_safe(positions)]

    if isinstance(positions, (list, tuple)):
        return [
            json_safe(position)
            for position in positions
        ]

    return []


def get_scanner_status(bot: Any) -> dict:
    status = call_method(
        bot,
        "scanner_status",
        default={},
    )

    if not isinstance(status, dict):
        status = {}

    return json_safe(status)


# ============================================================
# PAPER ACCOUNT
# ============================================================

def get_paper_account() -> dict:
    """
    Uses the current db.py accounting model.

    Equity = cash + invested + unrealized.

    We intentionally do NOT calculate:
        balance + position_notional + pnl

    because that double counts the position.
    """

    try:
        stats = db.stats()

        start_balance = float(
            stats.get(
                "paper_start_balance",
                getattr(
                    settings,
                    "paper_start_balance",
                    1000.0,
                ),
            )
        )

        balance = float(
            stats.get(
                "paper_balance",
                start_balance,
            )
        )

        invested = float(
            stats.get(
                "paper_invested",
                0.0,
            )
        )

        realized_pnl = float(
            stats.get(
                "realized_pnl",
                0.0,
            )
        )

        paper_equity = float(
            stats.get(
                "paper_equity",
                balance + invested,
            )
        )

        return {
            "start_balance": start_balance,
            "balance": balance,
            "cash": balance,
            "invested": invested,
            "equity": paper_equity,
            "realized_pnl": realized_pnl,
            "return_pct": (
                ((paper_equity / start_balance) - 1) * 100
                if start_balance > 0
                else 0.0
            ),
        }

    except Exception as exc:
        logger.exception(
            "Unable to calculate paper account."
        )

        start_balance = float(
            getattr(
                settings,
                "paper_start_balance",
                1000.0,
            )
        )

        return {
            "start_balance": start_balance,
            "balance": start_balance,
            "cash": start_balance,
            "invested": 0.0,
            "equity": start_balance,
            "realized_pnl": 0.0,
            "return_pct": 0.0,
            "error": str(exc),
        }


# ============================================================
# API: ROOT
# ============================================================

@app.get("/", response_class=HTMLResponse)
async def root():
    """
    Serve dashboard if static frontend exists.
    """

    try:
        with open(
            "index.html",
            "r",
            encoding="utf-8",
        ) as file:
            return HTMLResponse(
                content=file.read()
            )

    except FileNotFoundError:
        return HTMLResponse(
            """
            <!DOCTYPE html>
            <html>
            <head>
                <title>Kraken Day Trader</title>
            </head>
            <body>
                <h1>Kraken Day Trader</h1>
                <p>API is online.</p>
                <p>Dashboard index.html was not found.</p>
            </body>
            </html>
            """,
            status_code=200,
        )


# ============================================================
# API: HEALTH
# ============================================================

@app.get("/health")
async def health():
    return {
        "ok": True,
        "service": "kraken-day-trader",
        "timestamp": now_ts(),
        "worker_running": worker_alive(),
    }


# ============================================================
# API: STATUS
# ============================================================

@app.get("/api/status")
async def api_status():
    bot = get_bot()

    trader = get_trader(bot)

    trader_status = {}

    if trader is not None:
        try:
            if hasattr(trader, "status"):
                result = trader.status()

                if isinstance(result, dict):
                    trader_status = result
        except Exception:
            logger.exception(
                "Unable to retrieve trader status."
            )

    stats = get_bot_stats(bot)
    signals = get_bot_signals(bot)
    positions = get_bot_positions(bot)
    scanner = get_scanner_status(bot)
    paper = get_paper_account()

    mode = get_runtime_mode(bot)

    return json_safe({
        "ok": True,

        "service": "kraken-day-trader",

        "running": bot_is_running(bot)
        or worker_alive(),

        "worker_running": worker_alive(),

        "worker_error": _worker_error,

        "worker_started_at": _worker_started_at,

        "worker_stopped_at": _worker_stopped_at,

        "mode": mode,

        "safe_mode": mode != "LIVE",

        "paper_mode": mode == "PAPER",

        "live_mode": mode == "LIVE",

        "authenticated": bool(
            getattr(
                trader,
                "authenticated",
                False,
            )
        ) if trader is not None else False,

        "live_orders_enabled": bool(
            getattr(
                trader,
                "live_orders_enabled",
                False,
            )
        ) if trader is not None else False,

        "trader": trader_status,

        "stats": stats,

        "paper": paper,

        "positions": positions,

        "position": (
            positions[0]
            if positions
            else None
        ),

        "signals": signals[:20],

        "scanner": scanner,

        "config": {
            "autonomous": bool(
                getattr(
                    settings,
                    "autonomous",
                    True,
                )
            ),

            "timeframe": getattr(
                settings,
                "timeframe",
                "5m",
            ),

            "scan_seconds": getattr(
                settings,
                "scan_seconds",
                30,
            ),

            "max_trade_usd": getattr(
                settings,
                "max_trade_usd",
                50,
            ),

            "max_position_pct": getattr(
                settings,
                "max_position_pct",
                0.05,
            ),

            "stop_loss_pct": getattr(
                settings,
                "stop_loss_pct",
                0.008,
            ),

            "daily_loss_limit_usd": getattr(
                settings,
                "daily_loss_limit_usd",
                25,
            ),

            "max_trades_per_day": getattr(
                settings,
                "max_trades_per_day",
                10,
            ),

            "max_consecutive_losses": getattr(
                settings,
                "max_consecutive_losses",
                3,
            ),
        },

        "timestamp": now_ts(),
    })


# ============================================================
# API: START
# ============================================================

@app.post("/api/start")
async def api_start():
    success, message = start_bot_thread()

    bot = get_bot()

    return json_safe({
        "ok": success,
        "message": message,
        "running": worker_alive(),
        "mode": get_runtime_mode(bot),
        "worker_error": _worker_error,
    })


# ============================================================
# API: STOP
# ============================================================

@app.post("/api/stop")
async def api_stop():
    success, message = stop_bot_thread()

    bot = get_bot()

    return json_safe({
        "ok": success,
        "message": message,
        "running": worker_alive(),
        "mode": get_runtime_mode(bot),
        "worker_error": _worker_error,
    })


# ============================================================
# API: TRADING MODE
# ============================================================

@app.get("/api/trading-mode")
async def api_get_trading_mode():
    bot = get_bot()

    trader = get_trader(bot)

    return json_safe({
        "ok": True,

        "mode": get_runtime_mode(bot),

        "authenticated": bool(
            getattr(
                trader,
                "authenticated",
                False,
            )
        ) if trader is not None else False,

        "live_trading_configured": bool(
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

        "live_orders_enabled": bool(
            getattr(
                trader,
                "live_orders_enabled",
                False,
            )
        ) if trader is not None else False,

        "timestamp": now_ts(),
    })


@app.post("/api/trading-mode")
async def api_set_trading_mode(request: Request):
    """
    Expected:
        {"mode":"PAPER"}

    or:

        {"mode":"LIVE","confirmation":"ENABLE LIVE"}
    """

    bot = get_bot()

    try:
        body = await request.json()
    except Exception:
        body = {}

    requested_mode = str(
        body.get("mode", "")
    ).upper().strip()

    # --------------------------------------------------------
    # PAPER
    # --------------------------------------------------------

    if requested_mode == "PAPER":

        success = force_paper(bot)

        if not success:
            return JSONResponse(
                status_code=500,
                content={
                    "ok": False,
                    "message": "Unable to switch to PAPER mode.",
                },
            )

        return json_safe({
            "ok": True,
            "message": "PAPER mode enabled.",
            "mode": "PAPER",
            "live_orders_enabled": False,
        })

    # --------------------------------------------------------
    # LIVE
    # --------------------------------------------------------

    if requested_mode == "LIVE":

        confirmation = str(
            body.get(
                "confirmation",
                "",
            )
        ).strip().upper()

        if confirmation != "ENABLE LIVE":
            return JSONResponse(
                status_code=400,
                content={
                    "ok": False,
                    "message": (
                        'LIVE mode requires confirmation '
                        '"ENABLE LIVE".'
                    ),
                },
            )

        success, message = set_live_mode(bot)

        if not success:
            return JSONResponse(
                status_code=400,
                content={
                    "ok": False,
                    "message": message,
                },
            )

        return json_safe({
            "ok": True,
            "message": message,
            "mode": "LIVE",
            "live_orders_enabled": True,
        })

    return JSONResponse(
        status_code=400,
        content={
            "ok": False,
            "message": (
                'Invalid mode. Use "PAPER" or "LIVE".'
            ),
        },
    )


# ============================================================
# API: BALANCE
# ============================================================

@app.get("/api/balance")
async def api_balance():
    bot = get_bot()

    paper = get_paper_account()

    trader = get_trader(bot)

    live_account = None

    if trader is not None:

        try:
            if hasattr(trader, "account_summary"):
                live_account = trader.account_summary()

        except Exception:
            logger.exception(
                "Unable to retrieve account summary."
            )

    return json_safe({
        "ok": True,

        "mode": get_runtime_mode(bot),

        "paper": paper,

        "live": live_account,

        "timestamp": now_ts(),
    })


# ============================================================
# API: POSITIONS
# ============================================================

@app.get("/api/positions")
async def api_positions():
    bot = get_bot()

    positions = get_bot_positions(bot)

    # Fallback directly to DB.
    if not positions:
        try:
            positions = db.get_positions()
        except Exception:
            positions = []

    return json_safe({
        "ok": True,
        "positions": positions,
        "count": len(positions),
    })


# ============================================================
# API: SIGNALS
# ============================================================

@app.get("/api/signals")
async def api_signals():
    bot = get_bot()

    signals = get_bot_signals(bot)

    return json_safe({
        "ok": True,
        "signals": signals,
        "count": len(signals),
        "timestamp": now_ts(),
    })


# ============================================================
# API: SCANNER
# ============================================================

@app.get("/api/scanner")
async def api_scanner():
    bot = get_bot()

    scanner = get_scanner_status(bot)

    signals = get_bot_signals(bot)

    return json_safe({
        "ok": True,

        "scanner": scanner,

        "signals": signals[:20],

        "timestamp": now_ts(),
    })


# ============================================================
# API: MANUAL SCAN
# ============================================================

@app.post("/api/scan")
async def api_scan():
    bot = get_bot()

    """
    Manual scan endpoint.

    If the bot has request_scan(), use it.
    Otherwise fall back to the bot's scan() method.
    """

    try:

        if hasattr(bot, "request_scan"):

            result = bot.request_scan()

            return json_safe({
                "ok": True,
                "result": result,
                "signals": get_bot_signals(bot),
                "timestamp": now_ts(),
            })

        if hasattr(bot, "scan"):

            result = bot.scan()

            return json_safe({
                "ok": True,
                "result": result,
                "signals": get_bot_signals(bot),
                "timestamp": now_ts(),
            })

        return JSONResponse(
            status_code=501,
            content={
                "ok": False,
                "message": (
                    "Manual scan is not available "
                    "in the current bot implementation."
                ),
            },
        )

    except Exception as exc:

        logger.exception(
            "Manual scan failed."
        )

        return JSONResponse(
            status_code=500,
            content={
                "ok": False,
                "message": str(exc),
            },
        )


# ============================================================
# API: EQUITY
# ============================================================

@app.get("/api/equity")
async def api_equity():
    bot = get_bot()

    history = []

    try:
        history = db.equity_history(
            limit=500
        )
    except Exception:
        logger.exception(
            "Unable to retrieve equity history."
        )

    paper = get_paper_account()

    return json_safe({
        "ok": True,

        "mode": get_runtime_mode(bot),

        "current": paper,

        "history": history,

        "timestamp": now_ts(),
    })


# ============================================================
# API: PERFORMANCE
# ============================================================

@app.get("/api/performance")
async def api_performance():
    bot = get_bot()

    stats = get_bot_stats(bot)

    history = []

    try:
        history = db.equity_history(
            limit=500
        )
    except Exception:
        pass

    return json_safe({
        "ok": True,

        "mode": get_runtime_mode(bot),

        "stats": stats,

        "equity_history": history,

        "timestamp": now_ts(),
    })


# ============================================================
# API: KRAKEN TEST
# ============================================================

@app.get("/api/kraken/test")
async def api_kraken_test():
    bot = get_bot()

    result = test_kraken_connection(bot)

    return json_safe(result)


# ============================================================
# API: DEBUG
# ============================================================

@app.get("/api/debug")
async def api_debug():
    bot = get_bot()

    trader = get_trader(bot)

    return json_safe({
        "ok": True,

        "timestamp": now_ts(),

        "worker": {
            "running": _worker_running,
            "alive": worker_alive(),
            "started_at": _worker_started_at,
            "stopped_at": _worker_stopped_at,
            "error": _worker_error,
        },

        "bot": {
            "exists": bot is not None,
            "running": bot_is_running(bot),
            "mode": get_runtime_mode(bot),
            "type": (
                type(bot).__name__
                if bot is not None
                else None
            ),
        },

        "trader": {
            "exists": trader is not None,

            "type": (
                type(trader).__name__
                if trader is not None
                else None
            ),

            "authenticated": bool(
                getattr(
                    trader,
                    "authenticated",
                    False,
                )
            ) if trader is not None else False,

            "live_orders_enabled": bool(
                getattr(
                    trader,
                    "live_orders_enabled",
                    False,
                )
            ) if trader is not None else False,
        },

        "database": (
            db.database_health()
            if hasattr(
                db,
                "database_health",
            )
            else {}
        ),

        "paper": get_paper_account(),

        "settings": {
            "autonomous": bool(
                getattr(
                    settings,
                    "autonomous",
                    True,
                )
            ),

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

            "timeframe": getattr(
                settings,
                "timeframe",
                "5m",
            ),

            "candles": getattr(
                settings,
                "candles",
                720,
            ),

            "scan_seconds": getattr(
                settings,
                "scan_seconds",
                30,
            ),

            "market_refresh_seconds": getattr(
                settings,
                "market_refresh_seconds",
                1800,
            ),

            "max_scan_symbols": getattr(
                settings,
                "max_scan_symbols",
                20,
            ),

            "min_probability": getattr(
                settings,
                "min_probability",
                0.60,
            ),

            "min_expected_move": getattr(
                settings,
                "min_expected_move",
                0.004,
            ),

            "min_training_accuracy": getattr(
                settings,
                "min_training_accuracy",
                0.52,
            ),

            "max_trade_usd": getattr(
                settings,
                "max_trade_usd",
                50,
            ),

            "max_position_pct": getattr(
                settings,
                "max_position_pct",
                0.05,
            ),

            "stop_loss_pct": getattr(
                settings,
                "stop_loss_pct",
                0.008,
            ),

            "daily_loss_limit_usd": getattr(
                settings,
                "daily_loss_limit_usd",
                25,
            ),

            "max_trades_per_day": getattr(
                settings,
                "max_trades_per_day",
                10,
            ),

            "max_consecutive_losses": getattr(
                settings,
                "max_consecutive_losses",
                3,
            ),

            "cooldown_minutes": getattr(
                settings,
                "cooldown_minutes",
                15,
            ),
        },
    })


# ============================================================
# GLOBAL ERROR HANDLER
# ============================================================

@app.exception_handler(Exception)
async def global_exception_handler(
    request: Request,
    exc: Exception,
):
    logger.error(
        "Unhandled API exception: %s",
        exc,
    )

    logger.error(
        traceback.format_exc()
    )

    return JSONResponse(
        status_code=500,
        content={
            "ok": False,
            "error": str(exc),
            "path": str(
                request.url.path
            ),
        },
    )


# ============================================================
# STARTUP
# ============================================================

@app.on_event("startup")
async def startup_event():
    global current_bot

    logger.info("====================================================")
    logger.info("KRAKEN DAY TRADER STARTING")
    logger.info("====================================================")

    # --------------------------------------------------------
    # DATABASE
    # --------------------------------------------------------

    try:
        db.init_db()
        logger.info("Database initialized.")
    except Exception:
        logger.exception(
            "Database initialization failed."
        )

    # --------------------------------------------------------
    # FRESH BOT
    # --------------------------------------------------------

    with bot_lock:
        current_bot = create_bot()

        bot = current_bot

    # --------------------------------------------------------
    # ALWAYS PAPER AFTER RESTART
    # --------------------------------------------------------

    force_paper(bot)

    logger.info(
        "Startup trading mode: %s",
        get_runtime_mode(bot),
    )

    # --------------------------------------------------------
    # KRAKEN CONNECTION
    # --------------------------------------------------------

    try:
        result = test_kraken_connection(bot)

        logger.info(
            "Kraken connection: %s",
            result,
        )

    except Exception:
        logger.exception(
            "Kraken startup test failed."
        )

    # --------------------------------------------------------
    # AUTONOMOUS START
    # --------------------------------------------------------

    autonomous = bool(
        getattr(
            settings,
            "autonomous",
            True,
        )
    )

    trader = get_trader(bot)

    authenticated = bool(
        getattr(
            trader,
            "authenticated",
            False,
        )
    ) if trader is not None else False

    if autonomous and authenticated:

        logger.info(
            "AUTONOMOUS_MODE enabled."
        )

        success, message = start_bot_thread()

        if success:
            logger.info(
                message
            )
        else:
            logger.error(
                "Autonomous worker failed to start: %s",
                message,
            )

    elif autonomous:

        logger.warning(
            "AUTONOMOUS_MODE is enabled but Kraken "
            "authentication is unavailable. "
            "Worker will remain stopped."
        )

    else:

        logger.info(
            "AUTONOMOUS_MODE disabled. "
            "Bot is online but worker is stopped."
        )


# ============================================================
# SHUTDOWN
# ============================================================

@app.on_event("shutdown")
async def shutdown_event():
    logger.info(
        "Application shutting down..."
    )

    try:
        stop_bot_thread()
    except Exception:
        logger.exception(
            "Error stopping bot during shutdown."
        )

    logger.info(
        "Kraken Day Trader shutdown complete."
    )


# ============================================================
# OPTIONAL STATIC FILES
# ============================================================

# If you have a static directory, FastAPI can serve it.
#
# This is intentionally optional so Railway does not crash
# if the directory does not exist.

try:
    import os

    if os.path.isdir("static"):
        app.mount(
            "/static",
            StaticFiles(directory="static"),
            name="static",
        )

except Exception:
    logger.exception(
        "Unable to mount static directory."
    )
