import os
from dataclasses import dataclass
from decimal import Decimal
from dotenv import load_dotenv

load_dotenv()

def dec(n, d):
    try:
        return Decimal(os.getenv(n, d))
    except Exception:
        return Decimal(d)

def integer(n, d):
    try:
        return int(os.getenv(n, str(d)))
    except Exception:
        return d

def boolean(n, d=False):
    return os.getenv(n, str(d)).lower() in {"1", "true", "yes", "on"}

@dataclass(frozen=True)
class Settings:
    live_trading: bool = boolean("LIVE_TRADING", False)
    scan_interval: int = integer("SCAN_INTERVAL_SECONDS", 3)
    max_markets: int = integer("MAX_MARKETS", 200)
    min_liquidity: Decimal = dec("MIN_LIQUIDITY", "1000")
    min_net_edge: Decimal = dec("MIN_NET_EDGE", "0.015")
    fee_buffer: Decimal = dec("FEE_BUFFER", "0.005")
    slippage_buffer: Decimal = dec("SLIPPAGE_BUFFER", "0.003")
    min_pair_size: Decimal = dec("MIN_PAIR_SIZE", "5")
    max_pair_size: Decimal = dec("MAX_PAIR_SIZE", "50")
    max_pair_spend: Decimal = dec("MAX_PAIR_SPEND", "50")
    max_unpaired: Decimal = dec("MAX_UNPAIRED_SHARES", "10")
    max_daily_spend: Decimal = dec("MAX_DAILY_SPEND", "250")
    max_failures: int = integer("MAX_CONSECUTIVE_FAILURES", 3)
    cooldown: int = integer("COOLDOWN_SECONDS", 10)
    database_path: str = os.getenv("DATABASE_PATH", "polymarket_bot.db")
    keywords: tuple[str, ...] = tuple(
        x.strip().lower() for x in os.getenv("MARKET_KEYWORDS", "bitcoin,btc").split(",") if x.strip()
    )
    short_only: bool = boolean("ONLY_SHORT_DURATION", True)

settings = Settings()
