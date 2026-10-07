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
    version="3.1.0",
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
_worker_started_at = 0.0
_worker_stopped_at = 0.0

_last_start_attempt = 0.0


# ============================================================
# HELPERS
# ============================================================

def now_ts() -> float:
    return time.time()


def json_safe(value: Any) -> Any:
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
        return [
            json_safe(v)
            for v in value
        ]

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
    try:
        method = getattr(
            obj,
            method_name,
            None,
        )

        if not callable(method):
            return default

        return method(
            *args,
            **kwargs,
        )

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
        return bool(
            getattr(
                bot,
                "running",
                False,
            )
        )
    except Exception:
        return False


def worker_alive() -> bool:
    return bool(
        worker_thread is not None
        and worker_thread.is_alive()
    )


# ============================================================
# BOT
# ============================================================

def create_bot() -> KrakenBot:
    """
    Every newly created bot starts in PAPER mode.

    LIVE is never restored automatically after Railway
    restarts/deployments.
    """

    bot = KrakenBot()

    try:
        force_paper(bot)
    except Exception:
        logger.exception(
            "Unable to force PAPER mode."
        )

    return bot


def get_bot() -> KrakenBot:
    global current_bot

    with bot_lock:

        if current_bot is None:
            current_bot = create_bot()

        return current_bot


# ============================================================
# TRADER
# ============================================================

def get_trader(bot: Any):
    for name in (
        "trader",
        "client",
        "kraken",
        "exchange",
    ):
        try:
            obj = getattr(
                bot,
                name,
                None,
            )

            if obj is not None:
                return obj

        except Exception:
            pass

    return None


# ============================================================
# MODE
# ============================================================

def get_runtime_mode(bot: Any) -> str:

    trader = get_trader(bot)

    if trader is not None:

        try:
            if hasattr(
                trader,
                "status",
            ):
                status = trader.status()

                if isinstance(
                    status,
                    dict,
                ):
                    mode = status.get(
                        "mode"
                    )

                    if mode:
                        return str(
                            mode
                        ).upper()

        except Exception:
            pass

        try:
            mode = getattr(
                trader,
                "mode",
                None,
            )

            if mode:
                return str(
                    mode
                ).upper()

        except Exception:
            pass

    try:
        mode = getattr(
            bot,
            "mode",
            None,
        )

        if mode:
            return str(
                mode
            ).upper()

    except Exception:
        pass

    return "PAPER"


def force_paper(bot: Any) -> bool:

    trader = get_trader(bot)

    try:

        if trader is not None:

            if hasattr(
                trader,
                "force_paper_mode",
            ):
                trader.force_paper_mode()
                return True

            if hasattr(
                trader,
                "set_mode",
            ):
                result = trader.set_mode(
                    "PAPER"
                )

                return result is not False

        if hasattr(
            bot,
            "force_paper_mode",
        ):
            bot.force_paper_mode()
            return True

        if hasattr(
            bot,
            "set_mode",
        ):
            result = bot.set_mode(
                "PAPER"
            )

            return result is not False

    except Exception:
        logger.exception(
            "Failed forcing PAPER mode."
        )

    return False


def set_live_mode(
    bot: Any,
) -> tuple[bool, str]:

    if not bool(
        getattr(
            settings,
            "live_trading",
            False,
        )
    ):
        return (
            False,
            "LIVE_TRADING is disabled.",
        )

    if bool(
        getattr(
            settings,
            "dry_run",
            True,
        )
    ):
        return (
            False,
            "DRY_RUN is enabled.",
        )

    trader = get_trader(bot)

    if trader is None:
        return (
            False,
            "Kraken trader unavailable.",
        )

    try:

        authenticated = bool(
            getattr(
                trader,
                "authenticated",
                False,
            )
        )

        if not authenticated:

            if hasattr(
                trader,
                "test_authentication",
            ):
                result = trader.test_authentication()

                if isinstance(
                    result,
                    dict,
                ):
                    authenticated = bool(
                        result.get(
                            "authenticated",
                            result.get(
                                "ok",
                                False,
                            ),
                        )
                    )
                else:
                    authenticated = bool(
                        result
                    )

        if not authenticated:
            return (
                False,
                "Kraken authentication failed.",
            )

        if hasattr(
            trader,
            "set_mode",
        ):
            result = trader.set_mode(
                "LIVE"
            )

            if result is False:
                return (
                    False,
                    "Trader rejected LIVE mode.",
                )

            return (
                True,
                "LIVE mode enabled.",
            )

        return (
            False,
            "LIVE mode is unavailable.",
        )

    except Exception as exc:

        logger.exception(
            "Failed enabling LIVE mode."
        )

        return (
            False,
            str(exc),
        )


# ============================================================
# KRAKEN CONNECTION
# ============================================================

def test_kraken_connection(
    bot: Any,
) -> dict:

    trader = get_trader(bot)

    result = {
        "ok": False,
        "connected": False,
        "authenticated": False,
        "live_orders_enabled": False,
        "mode": get_runtime_mode(bot),
    }

    if trader is None:

        result["error"] = (
            "Kraken trader unavailable."
        )

        return result

    # --------------------------------------------------------
    # CONNECTION
    # --------------------------------------------------------

    try:

        if hasattr(
            trader,
            "test_connection",
        ):

            connection = (
                trader.test_connection()
            )

            if isinstance(
                connection,
                dict,
            ):

                result.update(
                    connection
                )

                result["connected"] = bool(
                    connection.get(
                        "connected",
                        connection.get(
                            "ok",
                            False,
                        ),
                    )
                )

            else:

                result["connected"] = bool(
                    connection
                )

        else:

            # If the trader exists but has no
            # explicit test method, don't
            # automatically report disconnected.
            result["connected"] = True

    except Exception as exc:

        logger.error(
            "Kraken connection test failed: %s",
            exc,
        )

        result["connection_error"] = str(
            exc
        )

    # --------------------------------------------------------
    # AUTHENTICATION
    # --------------------------------------------------------
    #
    # PAPER mode only needs public market-data connectivity.
    # Do not require private API credentials just to run PAPER.
    # Authentication is checked only when LIVE orders are enabled.

    result["mode"] = get_runtime_mode(bot)

    if result["mode"] == "LIVE" or bool(
        getattr(trader, "live_orders_enabled", False)
    ):
        try:
            if hasattr(trader, "test_authentication"):
                authentication = trader.test_authentication()

                if isinstance(authentication, dict):
                    result.update(authentication)
                    result["authenticated"] = bool(
                        authentication.get(
                            "authenticated",
                            authentication.get("ok", False),
                        )
                    )
                else:
                    result["authenticated"] = bool(authentication)
            else:
                result["authenticated"] = bool(
                    getattr(trader, "authenticated", False)
                )
        except Exception as exc:
            logger.error(
                "Kraken authentication test failed: %s",
                exc,
            )
            result["authentication_error"] = str(exc)
    else:
        result["authenticated"] = bool(
            getattr(trader, "authenticated", False)
        )
        result["authentication_skipped"] = True

    # --------------------------------------------------------
    # LIVE ORDERS
    # --------------------------------------------------------

    result["live_orders_enabled"] = bool(
        getattr(
            trader,
            "live_orders_enabled",
            False,
        )
    )

    result["ok"] = bool(
        result["connected"]
        and (
            result["authenticated"]
            if result["mode"] == "LIVE"
            else True
        )
    )

    return json_safe(
        result
    )


# ============================================================
# PAPER ACCOUNT
# ============================================================

def get_paper_account() -> dict:

    try:

        stats = db.stats()

        start = float(
            stats.get(
                "paper_start_balance",
                getattr(
                    settings,
                    "paper_start_balance",
                    1000.0,
                ),
            )
        )

        cash = float(
            stats.get(
                "paper_balance",
                start,
            )
        )

        invested = float(
            stats.get(
                "paper_invested",
                0.0,
            )
        )

        equity = float(
            stats.get(
                "paper_equity",
                cash + invested,
            )
        )

        realized = float(
            stats.get(
                "realized_pnl",
                0.0,
            )
        )

        return {
            "start_balance": start,

            "balance": cash,

            "cash": cash,

            "free": cash,

            "paper_balance": cash,

            "paper_cash": cash,

            "invested": invested,

            "equity": equity,

            "paper_equity": equity,

            "total": equity,

            "portfolio_value_usd": equity,

            "realized_pnl": realized,

            "return_pct": (
                ((equity / start) - 1)
                * 100
                if start > 0
                else 0.0
            ),
        }

    except Exception as exc:

        logger.exception(
            "Paper account error."
        )

        start = float(
            getattr(
                settings,
                "paper_start_balance",
                1000.0,
            )
        )

        return {
            "start_balance": start,
            "balance": start,
            "cash": start,
            "free": start,
            "paper_balance": start,
            "paper_cash": start,
            "invested": 0.0,
            "equity": start,
            "paper_equity": start,
            "total": start,
            "portfolio_value_usd": start,
            "realized_pnl": 0.0,
            "return_pct": 0.0,
            "error": str(exc),
        }


# ============================================================
# BOT DATA
# ============================================================

def get_bot_stats(
    bot: Any,
) -> dict:

    stats = call_method(
        bot,
        "stats",
        {},
    )

    if not isinstance(
        stats,
        dict,
    ):
        stats = {}

    try:

        db_stats = db.stats()

        if isinstance(
            db_stats,
            dict,
        ):

            for key, value in db_stats.items():

                stats.setdefault(
                    key,
                    value,
                )

    except Exception:
        logger.exception(
            "Unable to retrieve DB stats."
        )

    paper = get_paper_account()

    # Guarantee dashboard fields.
    stats.setdefault(
        "paper_balance",
        paper["paper_balance"],
    )

    stats.setdefault(
        "paper_cash",
        paper["paper_cash"],
    )

    stats.setdefault(
        "paper_equity",
        paper["paper_equity"],
    )

    stats.setdefault(
        "equity",
        paper["equity"],
    )

    stats.setdefault(
        "cash",
        paper["cash"],
    )

    return json_safe(
        stats
    )


def get_bot_signals(
    bot: Any,
) -> list:

    signals = call_method(
        bot,
        "get_signals",
        [],
    )

    if isinstance(
        signals,
        dict,
    ):
        return [
            json_safe(signals)
        ]

    if isinstance(
        signals,
        (list, tuple),
    ):
        return [
            json_safe(x)
            for x in signals
        ]

    return []


def get_bot_positions(
    bot: Any,
) -> list:

    positions = call_method(
        bot,
        "get_positions",
        None,
    )

    if positions is None:

        positions = call_method(
            bot,
            "positions",
            [],
        )

    if isinstance(
        positions,
        dict,
    ):
        return [
            json_safe(positions)
        ]

    if isinstance(
        positions,
        (list, tuple),
    ):
        return [
            json_safe(x)
            for x in positions
        ]

    return []


def get_scanner_status(
    bot: Any,
) -> dict:

    result = call_method(
        bot,
        "scanner_status",
        {},
    )

    if not isinstance(
        result,
        dict,
    ):
        return {}

    return json_safe(
        result
    )


# ============================================================
# WORKER
# ============================================================

def _run_bot_worker(
    bot: KrakenBot,
):

    global _worker_running
    global _worker_error
    global _worker_stopped_at

    _worker_running = True
    _worker_error = None

    logger.info(
        "===================================================="
    )

    logger.info(
        "KRAKEN BOT WORKER STARTED"
    )

    logger.info(
        "Mode: %s",
        get_runtime_mode(bot),
    )

    logger.info(
        "===================================================="
    )

    try:

        asyncio.run(
            bot.run()
        )

        logger.warning(
            "Bot worker exited normally."
        )

        _worker_error = (
            "Bot worker exited normally."
        )

    except Exception as exc:

        _worker_error = (
            f"{type(exc).__name__}: {exc}"
        )

        logger.error(
            "BOT WORKER CRASHED: %s",
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
            "Bot worker stopped."
        )


def start_bot_thread() -> tuple[bool, str]:

    global current_bot
    global worker_thread
    global _worker_error
    global _worker_started_at
    global _last_start_attempt

    with worker_lock:

        if worker_alive() or _worker_running:

            return (
                False,
                "Bot is already running.",
            )

        current = now_ts()

        if current - _last_start_attempt < 2:

            return (
                False,
                "Start request already processing.",
            )

        _last_start_attempt = current

        with bot_lock:

            current_bot = create_bot()

            bot = current_bot

            # ALWAYS PAPER after restart/start.
            paper_forced = force_paper(bot)

            if not paper_forced or get_runtime_mode(bot) != "PAPER":
                _worker_error = (
                    "Safety check failed: bot could not be forced into PAPER mode."
                )
                logger.error(_worker_error)
                return (
                    False,
                    _worker_error,
                )

            logger.info(
                "Worker mode: %s",
                get_runtime_mode(bot),
            )

            # ------------------------------------------------
            # START
            # ------------------------------------------------
            #
            # PAPER mode must start independently of a one-time
            # Kraken connectivity check. Public API/DNS/rate-limit
            # issues can be transient, and bot.run() already handles
            # per-cycle API failures without killing the worker.
            #
            # Do not block the Start button on load_markets().
            # The scanner will retry on its normal cycle.

            # PAPER mode intentionally does not require private
            # API authentication. LIVE remains gated separately.

            # ------------------------------------------------
            # START
            # ------------------------------------------------

            _worker_error = None
            _worker_started_at = now_ts()

            # Mark the engine running before handing it to the
            # background worker. This makes the start operation
            # deterministic and avoids waiting on the first scan.
            try:
                if hasattr(bot, "start"):
                    bot.start()
                else:
                    bot.running = True
            except Exception as exc:
                _worker_error = (
                    f"Bot initialization failed: "
                    f"{type(exc).__name__}: {exc}"
                )
                logger.exception(
                    "Bot initialization failed."
                )
                return (
                    False,
                    _worker_error,
                )

            try:
                worker_thread = threading.Thread(
                    target=_run_bot_worker,
                    args=(bot,),
                    name="kraken-bot-worker",
                    daemon=True,
                )

                worker_thread.start()

            except Exception as exc:
                try:
                    bot.running = False
                except Exception:
                    pass

                _worker_error = (
                    f"Worker start failed: "
                    f"{type(exc).__name__}: {exc}"
                )
                logger.exception(
                    "Worker thread could not start."
                )
                return (
                    False,
                    _worker_error,
                )

            # Do not wait for the first market scan here.
            # bot.run() owns its own retry/error handling.
            return (
                True,
                "Kraken bot started successfully.",
            )


def stop_bot_thread() -> tuple[bool, str]:

    global current_bot

    with worker_lock:

        bot = current_bot

        if bot is None:

            return (
                False,
                "Bot is not running.",
            )

        if not worker_alive() and not _worker_running:

            try:
                bot.running = False
            except Exception:
                pass

            return (
                False,
                "Bot is already stopped.",
            )

        try:

            if hasattr(
                bot,
                "request_stop",
            ):
                bot.request_stop()

            elif hasattr(
                bot,
                "stop",
            ):
                bot.stop()

            else:
                bot.running = False

        except Exception:

            logger.exception(
                "Stop request failed."
            )

            try:
                bot.running = False
            except Exception:
                pass

        thread = worker_thread

        if (
            thread is not None
            and thread.is_alive()
        ):

            thread.join(
                timeout=5
            )

        if (
            thread is not None
            and thread.is_alive()
        ):

            return (
                False,
                "Stop requested; worker is still shutting down.",
            )

        return (
            True,
            "Kraken bot stopped.",
        )


# ============================================================
# ROOT
# ============================================================

@app.get(
    "/",
    response_class=HTMLResponse,
)
async def root():

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
            <!DOCTYPE html>
            <html>
            <head>
                <title>Kraken Bot</title>
            </head>
            <body>
                <h1>KRAKEN BOT</h1>
                <p>API online.</p>
            </body>
            </html>
            """
        )


# ============================================================
# HEALTH
# ============================================================

@app.get("/health")
async def health():

    bot = get_bot()

    running = (
        bot_is_running(bot)
        or worker_alive()
        or _worker_running
    )

    return {
        "ok": True,
        "service": "kraken-day-trader",
        "mode": get_runtime_mode(bot),
        "worker_running": worker_alive(),
        "bot_running": running,
        "worker_error": _worker_error,
        "timestamp": now_ts(),
    }


# ============================================================
# STATUS
# ============================================================

@app.get("/api/status")
async def api_status():

    bot = get_bot()

    # --------------------------------------------------------
    # KRAKEN
    # --------------------------------------------------------
    #
    # Status must be non-blocking. Do not call Kraken's network
    # API on every dashboard refresh. The worker updates this
    # connection state during normal operation.
    trader = get_trader(bot)

    kraken = call_method(
        trader,
        "connection_status",
        {
            "connected": bool(
                getattr(
                    trader,
                    "connected",
                    False,
                )
            ) if trader is not None else False,
            "authenticated": bool(
                getattr(
                    trader,
                    "authenticated",
                    False,
                )
            ) if trader is not None else False,
            "mode": get_runtime_mode(bot),
        },
    )

    if not isinstance(kraken, dict):
        kraken = {
            "connected": False,
            "authenticated": False,
            "mode": get_runtime_mode(bot),
        }

    kraken["mode"] = get_runtime_mode(bot)

    # --------------------------------------------------------
    # BOT
    # --------------------------------------------------------

    running = (
        bot_is_running(bot)
        or worker_alive()
        or _worker_running
    )

    # --------------------------------------------------------
    # STATS
    # --------------------------------------------------------

    stats = get_bot_stats(
        bot
    )

    paper = get_paper_account()

    # --------------------------------------------------------
    # POSITIONS
    # --------------------------------------------------------

    positions = get_bot_positions(
        bot
    )

    # --------------------------------------------------------
    # SIGNALS
    # --------------------------------------------------------

    signals = get_bot_signals(
        bot
    )

    # --------------------------------------------------------
    # SCANNER
    # --------------------------------------------------------

    scanner = get_scanner_status(
        bot
    )

    # --------------------------------------------------------
    # ACCOUNT COMPATIBILITY
    # --------------------------------------------------------

    account = {
        "paper_balance": paper[
            "paper_balance"
        ],

        "paper_cash": paper[
            "paper_cash"
        ],

        "paper_equity": paper[
            "paper_equity"
        ],

        "cash": paper[
            "cash"
        ],

        "equity": paper[
            "equity"
        ],

        "free": paper[
            "free"
        ],

        "total": paper[
            "total"
        ],

        "invested": paper[
            "invested"
        ],
    }

    # --------------------------------------------------------
    # BOT COMPATIBILITY
    # --------------------------------------------------------

    bot_status = {
        "running": running,
        "mode": get_runtime_mode(bot),
        "error": _worker_error,
    }

    return json_safe({

        "ok": True,

        # Existing frontend expects this.
        "running": running,

        "worker_alive": worker_alive(),

        "worker_running": _worker_running,

        "worker_error": _worker_error,

        # Existing frontend expects this object.
        "kraken": kraken,

        # Keep trader too for newer dashboard versions.
        "trader": kraken,

        # Existing frontend uses bot.running.
        "bot": bot_status,

        # Existing frontend uses account.paper_equity.
        "account": account,

        # Existing dashboard compatibility.
        "paper": paper,

        "mode": get_runtime_mode(
            bot
        ),

        "stats": stats,

        "positions": positions,

        "position": (
            positions[0]
            if positions
            else None
        ),

        "signals": signals[:20],

        "scanner": scanner,

        "equity": {
            "paper_equity": paper[
                "paper_equity"
            ],

            "paper_cash": paper[
                "paper_cash"
            ],

            "cash": paper[
                "cash"
            ],

            "equity": paper[
                "equity"
            ],
        },

        "timestamp": now_ts(),
    })


# ============================================================
# START
# ============================================================

@app.post("/api/start")
async def api_start():

    success, message = (
        start_bot_thread()
    )

    bot = get_bot()

    return json_safe({

        "ok": success,

        "message": message,

        "running": (
            worker_alive()
            or _worker_running
        ),

        "worker_alive": worker_alive(),

        "mode": get_runtime_mode(
            bot
        ),

        # Do not perform a network/API call here. The Start
        # endpoint must return immediately after launching the worker.
        "kraken": (
            call_method(
                get_trader(bot),
                "connection_status",
                {
                    "connected": False,
                    "authenticated": False,
                    "mode": get_runtime_mode(bot),
                },
            )
            or {
                "connected": False,
                "authenticated": False,
                "mode": get_runtime_mode(bot),
            }
        ),

        "error": (
            _worker_error
            or (
                None
                if success
                else message
            )
        ),
    })


# ============================================================
# STOP
# ============================================================

@app.post("/api/stop")
async def api_stop():

    success, message = (
        stop_bot_thread()
    )

    bot = get_bot()

    return json_safe({

        "ok": success,

        "message": message,

        "running": (
            worker_alive()
            or _worker_running
        ),

        "worker_alive": worker_alive(),

        "mode": get_runtime_mode(
            bot
        ),

        "error": _worker_error,
    })


# ============================================================
# TRADING MODE
# ============================================================

@app.get("/api/trading-mode")
async def api_get_trading_mode():

    bot = get_bot()

    trader = get_trader(
        bot
    )

    # Keep this endpoint fast and non-blocking. Do not perform
    # load_markets()/network I/O just to render the mode panel.
    kraken = call_method(
        trader,
        "connection_status",
        {
            "connected": False,
            "authenticated": False,
            "mode": get_runtime_mode(bot),
        },
    )

    if not isinstance(kraken, dict):
        kraken = {
            "connected": False,
            "authenticated": False,
            "mode": get_runtime_mode(bot),
        }

    return json_safe({

        "ok": True,

        "mode": get_runtime_mode(
            bot
        ),

        "connected": kraken[
            "connected"
        ],

        "authenticated": kraken[
            "authenticated"
        ],

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
        ) if trader else False,

        "timestamp": now_ts(),
    })


@app.post("/api/trading-mode")
async def api_set_trading_mode(
    request: Request,
):

    bot = get_bot()

    try:
        body = await request.json()
    except Exception:
        body = {}

    mode = str(
        body.get(
            "mode",
            "",
        )
    ).upper().strip()

    # --------------------------------------------------------
    # PAPER
    # --------------------------------------------------------

    if mode == "PAPER":

        if not force_paper(
            bot
        ):

            return JSONResponse(
                status_code=500,
                content={
                    "ok": False,
                    "message": (
                        "Unable to switch to PAPER."
                    ),
                },
            )

        return {
            "ok": True,
            "message": "PAPER TRADING enabled.",
            "mode": "PAPER",
            "live_orders_enabled": False,
        }

    # --------------------------------------------------------
    # LIVE
    # --------------------------------------------------------

    if mode == "LIVE":

        confirmation = str(
            body.get(
                "confirmation",
                "",
            )
        ).upper().strip()

        if confirmation != "ENABLE LIVE":

            return JSONResponse(
                status_code=400,
                content={
                    "ok": False,
                    "message": (
                        'LIVE trading requires '
                        '"ENABLE LIVE".'
                    ),
                },
            )

        success, message = (
            set_live_mode(
                bot
            )
        )

        if not success:

            return JSONResponse(
                status_code=400,
                content={
                    "ok": False,
                    "message": message,
                },
            )

        return {
            "ok": True,
            "message": message,
            "mode": "LIVE",
            "live_orders_enabled": True,
        }

    return JSONResponse(
        status_code=400,
        content={
            "ok": False,
            "message": (
                'Use mode "PAPER" or "LIVE".'
            ),
        },
    )


# ============================================================
# BALANCE
# ============================================================

@app.get("/api/balance")
async def api_balance():

    bot = get_bot()

    paper = get_paper_account()

    trader = get_trader(
        bot
    )

    live = {}

    # --------------------------------------------------------
    # PAPER
    # --------------------------------------------------------

    if get_runtime_mode(
        bot
    ) == "PAPER":

        return json_safe({

            "ok": True,

            "mode": "PAPER",

            # Existing dashboard fields.
            "free": paper[
                "free"
            ],

            "cash": paper[
                "cash"
            ],

            "total": paper[
                "total"
            ],

            "equity": paper[
                "equity"
            ],

            "portfolio_value_usd": paper[
                "portfolio_value_usd"
            ],

            "usd_free": paper[
                "free"
            ],

            "usd_total": paper[
                "total"
            ],

            "usdg_total": 0.0,

            "paper_balance": paper[
                "paper_balance"
            ],

            "paper_cash": paper[
                "paper_cash"
            ],

            "paper_equity": paper[
                "paper_equity"
            ],

            "invested": paper[
                "invested"
            ],

            "realized_pnl": paper[
                "realized_pnl"
            ],

            "start_balance": paper[
                "start_balance"
            ],
        })

    # --------------------------------------------------------
    # LIVE
    # --------------------------------------------------------

    if trader is not None:

        try:

            if hasattr(
                trader,
                "account_summary",
            ):

                result = (
                    trader.account_summary()
                )

                if isinstance(
                    result,
                    dict,
                ):
                    live = result

        except Exception:
            logger.exception(
                "Unable to retrieve live balance."
            )

    return json_safe({
        "ok": True,
        "mode": "LIVE",

        "free": live.get(
            "free",
            live.get(
                "usd_free",
                0.0,
            ),
        ),

        "cash": live.get(
            "cash",
            live.get(
                "free",
                0.0,
            ),
        ),

        "total": live.get(
            "total",
            live.get(
                "equity",
                0.0,
            ),
        ),

        "equity": live.get(
            "equity",
            live.get(
                "total",
                0.0,
            ),
        ),

        "portfolio_value_usd": live.get(
            "portfolio_value_usd",
            live.get(
                "equity",
                0.0,
            ),
        ),

        "usd_free": live.get(
            "usd_free",
            live.get(
                "free",
                0.0,
            ),
        ),

        "usd_total": live.get(
            "usd_total",
            live.get(
                "total",
                0.0,
            ),
        ),

        "usdg_total": live.get(
            "usdg_total",
            0.0,
        ),

        "live": live,
    })


# ============================================================
# POSITIONS
# ============================================================

@app.get("/api/positions")
async def api_positions():

    bot = get_bot()

    positions = get_bot_positions(
        bot
    )

    if not positions:

        try:
            positions = db.get_positions()
        except Exception:
            positions = []

    return json_safe({

        "ok": True,

        "positions": positions,

        "count": len(
            positions
        ),
    })


# ============================================================
# SIGNALS
# ============================================================

@app.get("/api/signals")
async def api_signals():

    bot = get_bot()

    signals = get_bot_signals(
        bot
    )

    return json_safe({

        "ok": True,

        "signals": signals,

        "count": len(
            signals
        ),

        "timestamp": now_ts(),
    })


# ============================================================
# SCANNER
# ============================================================

@app.get("/api/scanner")
async def api_scanner():

    bot = get_bot()

    scanner = get_scanner_status(
        bot
    )

    signals = get_bot_signals(
        bot
    )

    return json_safe({

        "ok": True,

        "scanner": scanner,

        "signals": signals[:20],

        "timestamp": now_ts(),
    })


# ============================================================
# MANUAL SCAN
# ============================================================

@app.post("/api/scan")
async def api_scan():

    bot = get_bot()

    try:

        if hasattr(
            bot,
            "request_scan",
        ):

            result = bot.request_scan()

            return json_safe({
                "ok": True,
                "result": result,
                "signals": get_bot_signals(
                    bot
                ),
            })

        if hasattr(
            bot,
            "scan",
        ):

            result = bot.scan()

            return json_safe({
                "ok": True,
                "result": result,
                "signals": get_bot_signals(
                    bot
                ),
            })

        return JSONResponse(
            status_code=501,
            content={
                "ok": False,
                "message": (
                    "Scanner unavailable."
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
# EQUITY
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

        "mode": get_runtime_mode(
            bot
        ),

        "current": paper,

        "paper_equity": paper[
            "paper_equity"
        ],

        "paper_cash": paper[
            "paper_cash"
        ],

        "history": history,
    })


# ============================================================
# PERFORMANCE
# ============================================================

@app.get("/api/performance")
async def api_performance():

    bot = get_bot()

    stats = get_bot_stats(
        bot
    )

    paper = get_paper_account()

    history = []

    try:
        history = db.equity_history(
            limit=500
        )
    except Exception:
        pass

    return json_safe({

        "ok": True,

        "mode": get_runtime_mode(
            bot
        ),

        "stats": stats,

        "paper_equity": paper[
            "paper_equity"
        ],

        "paper_cash": paper[
            "paper_cash"
        ],

        "equity": paper[
            "equity"
        ],

        "cash": paper[
            "cash"
        ],

        "equity_history": history,
    })


# ============================================================
# KRAKEN TEST
# ============================================================

@app.get("/api/kraken/test")
async def api_kraken_test():

    bot = get_bot()

    return json_safe(
        test_kraken_connection(
            bot
        )
    )


# ============================================================
# DEBUG
# ============================================================

@app.get("/api/debug")
async def api_debug():

    bot = get_bot()

    trader = get_trader(
        bot
    )

    kraken = test_kraken_connection(
        bot
    )

    return json_safe({

        "ok": True,

        "worker": {
            "running": _worker_running,
            "alive": worker_alive(),
            "started_at": _worker_started_at,
            "stopped_at": _worker_stopped_at,
            "error": _worker_error,
        },

        "bot": {
            "running": bot_is_running(
                bot
            ),
            "mode": get_runtime_mode(
                bot
            ),
        },

        "kraken": kraken,

        "trader": {
            "exists": trader is not None,

            "authenticated": bool(
                getattr(
                    trader,
                    "authenticated",
                    False,
                )
            ) if trader else False,

            "live_orders_enabled": bool(
                getattr(
                    trader,
                    "live_orders_enabled",
                    False,
                )
            ) if trader else False,
        },

        "paper": get_paper_account(),

        "database": (
            db.database_health()
            if hasattr(
                db,
                "database_health",
            )
            else {}
        ),
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
        "Unhandled API error: %s",
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

    logger.info(
        "===================================================="
    )

    logger.info(
        "KRAKEN DAY TRADER STARTING"
    )

    logger.info(
        "===================================================="
    )

    # --------------------------------------------------------
    # DATABASE
    # --------------------------------------------------------

    try:

        db.init_db()

        logger.info(
            "Database initialized."
        )

    except Exception:

        logger.exception(
            "Database initialization failed."
        )

    # --------------------------------------------------------
    # CREATE BOT
    # --------------------------------------------------------

    with bot_lock:

        current_bot = create_bot()

        bot = current_bot

    # --------------------------------------------------------
    # ALWAYS PAPER
    # --------------------------------------------------------

    force_paper(
        bot
    )

    logger.info(
        "Startup mode: %s",
        get_runtime_mode(
            bot
        ),
    )

    # --------------------------------------------------------
    # KRAKEN
    # --------------------------------------------------------

    try:

        kraken = test_kraken_connection(
            bot
        )

        logger.info(
            "Kraken status: %s",
            kraken,
        )

    except Exception:

        logger.exception(
            "Kraken startup test failed."
        )

        kraken = {
            "connected": False,
            "authenticated": False,
        }

    # --------------------------------------------------------
    # AUTONOMOUS
    # --------------------------------------------------------

    autonomous = bool(
        getattr(
            settings,
            "autonomous",
            True,
        )
    )

    if autonomous:

        # Start the autonomous PAPER engine even if the initial
        # connectivity probe is temporarily unavailable.
        # The bot loop is resilient to individual market/API
        # failures and will retry on the next scan cycle.
        success, message = (
            start_bot_thread()
        )

        if success:

            logger.info(
                message
            )

        else:

            logger.error(
                "Autonomous startup failed: %s",
                message,
            )

    else:

        logger.info(
            "Autonomous mode disabled."
        )


# ============================================================
# SHUTDOWN
# ============================================================

@app.on_event("shutdown")
async def shutdown_event():

    logger.info(
        "Shutting down Kraken bot..."
    )

    try:

        stop_bot_thread()

    except Exception:

        logger.exception(
            "Shutdown error."
        )

    logger.info(
        "Kraken bot shutdown complete."
    )


# ============================================================
# STATIC
# ============================================================

try:

    import os

    if os.path.isdir(
        "static"
    ):

        app.mount(
            "/static",
            StaticFiles(
                directory="static"
            ),
            name="static",
        )

except Exception:

    logger.exception(
        "Static directory mount failed."
    )
