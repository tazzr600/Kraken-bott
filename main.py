import asyncio

from fastapi import FastAPI
from fastapi.responses import HTMLResponse

from bot import bot
from config import settings
from db import (
    init_db,
    get_positions,
    stats,
)


app = FastAPI(
    title="KRAKEN BOT"
)


# --------------------------------------------------
# STARTUP
# --------------------------------------------------

@app.on_event("startup")
async def startup():

    init_db()

    print("=" * 60)
    print("KRAKEN BOT STARTING")
    print("=" * 60)

    # ----------------------------------------------
    # CHECK PUBLIC KRAKEN CONNECTION
    # ----------------------------------------------

    result = await asyncio.to_thread(
        bot.kraken.test_connection
    )

    if result["connected"]:

        print(
            "KRAKEN CONNECTION: OK"
        )

    else:

        print(
            "KRAKEN CONNECTION FAILED:",
            result["error"]
        )

    # ----------------------------------------------
    # CHECK PRIVATE KRAKEN AUTHENTICATION
    # ----------------------------------------------

    auth = await asyncio.to_thread(
        bot.kraken.test_authentication
    )

    if auth["authenticated"]:

        print(
            "KRAKEN AUTHENTICATION: OK"
        )

    else:

        print(
            "KRAKEN AUTHENTICATION FAILED:",
            auth["error"]
        )

        print(
            "KRAKEN ERROR TYPE:",
            auth.get("error_type")
        )

    print(
        "KRAKEN LIVE ORDERS:",
        bot.kraken.live_orders_enabled
    )

    print(
        "KRAKEN CREDENTIALS CONFIGURED:",
        bot.kraken.credentials_configured()
    )

    print("=" * 60)

    # ----------------------------------------------
    # START BOT
    # ----------------------------------------------

    asyncio.create_task(
        bot.run()
    )


# --------------------------------------------------
# HEALTH
# --------------------------------------------------

@app.get("/health")
async def health():

    kraken = (
        bot.kraken.connection_status()
    )

    return {

        "ok":
            True,

        "running":
            bot.running,

        "live":
            settings.live_trading,

        "dry_run":
            settings.dry_run,

        "exchange":
            "kraken",

        "kraken_connected":
            kraken["connected"],

        "kraken_authenticated":
            kraken["authenticated"],

        "kraken_error":
            kraken["error"],

        "kraken_error_type":
            kraken["error_type"],

        "api_configured":
            kraken["credentials_configured"],

        "live_orders_enabled":
            kraken["live_orders_enabled"],

        "symbols":
            settings.symbols,

        "ml_models":
            len(bot.models),

        "last_scan":
            bot.last_scan,

        "error":
            bot.error,
    }


# --------------------------------------------------
# STATUS
# --------------------------------------------------

@app.get("/api/status")
async def status():

    kraken = (
        bot.kraken.connection_status()
    )

    return {

        "running":
            bot.running,

        "mode":
            (
                "LIVE"
                if (
                    settings.live_trading
                    and not settings.dry_run
                )
                else "PAPER"
            ),

        "kraken":
            kraken,

        "signals":
            bot.signals,

        "positions":
            get_positions(),

        "stats":
            stats(),

        "error":
            bot.error,

        "last_scan":
            bot.last_scan,
    }


# --------------------------------------------------
# MANUAL KRAKEN AUTH TEST
# --------------------------------------------------

@app.post("/api/kraken/test")
async def test_kraken():

    connection = await asyncio.to_thread(
        bot.kraken.test_connection
    )

    authentication = await asyncio.to_thread(
        bot.kraken.test_authentication
    )

    return {

        "connection":
            connection,

        "authentication":
            authentication,

        "status":
            bot.kraken.connection_status(),
    }


# --------------------------------------------------
# START
# --------------------------------------------------

@app.post("/api/start")
async def start():

    result = await asyncio.to_thread(
        bot.kraken.test_authentication
    )

    if not result["authenticated"]:

        return {

            "running":
                False,

            "error":
                "Kraken authentication failed",

            "kraken":
                result,
        }

    bot.running = True

    return {

        "running":
            True,

        "kraken":
            result,
    }


# --------------------------------------------------
# STOP
# --------------------------------------------------

@app.post("/api/stop")
async def stop():

    bot.running = False

    return {
        "running": False
    }


# --------------------------------------------------
# MANUAL SCAN
# --------------------------------------------------

@app.post("/api/scan")
async def scan():

    return {
        "signals":
            await bot.scan()
    }


# --------------------------------------------------
# DASHBOARD
# --------------------------------------------------

@app.get(
    "/",
    response_class=HTMLResponse
)
async def home():

    with open(
        "index.html",
        "r",
        encoding="utf-8"
    ) as f:

        return f.read()
