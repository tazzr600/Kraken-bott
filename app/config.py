import os
from dataclasses import dataclass
from decimal import Decimal
from dotenv import load_dotenv

load_dotenv()

def dec(name, default):
    try: return Decimal(os.getenv(name, default))
    except Exception: return Decimal(default)

def integer(name, default):
    try: return int(os.getenv(name, str(default)))
    except Exception: return default

def number(name, default):
    try: return float(os.getenv(name, str(default)))
    except Exception: return default

def boolean(name, default=False):
    return os.getenv(name, str(default)).strip().lower() in {"1","true","yes","on"}

def csv(name, default):
    return tuple(x.strip() for x in os.getenv(name, default).split(",") if x.strip())

@dataclass(frozen=True)
class Settings:
    live_trading: bool = False
    auto_start: bool = boolean("AUTO_START", True)
    scan_interval: float = number("SCAN_INTERVAL_SECONDS", 1.5)
    market_refresh_seconds: int = integer("MARKET_REFRESH_SECONDS", 20)
    series: tuple[str,...] = csv("POLYMARKET_SERIES","btc-up-or-down-5m,eth-up-or-down-5m,sol-up-or-down-5m")
    max_markets: int = integer("MAX_ACTIVE_MARKETS", 6)
    min_liquidity: Decimal = dec("MIN_LIQUIDITY","500")
    min_book_size: Decimal = dec("MIN_BOOK_SIZE","5")
    min_entry_edge: Decimal = dec("MIN_ENTRY_EDGE","0.025")
    min_add_edge: Decimal = dec("MIN_ADD_EDGE","0.018")
    min_hedge_edge: Decimal = dec("MIN_HEDGE_EDGE","0.005")
    fee_buffer: Decimal = dec("FEE_BUFFER","0.006")
    slippage_buffer: Decimal = dec("SLIPPAGE_BUFFER","0.004")
    target_trade_spend: Decimal = dec("TARGET_TRADE_SPEND","35.88")
    max_trade_spend: Decimal = dec("MAX_TRADE_SPEND","50")
    max_market_spend: Decimal = dec("MAX_MARKET_SPEND","100")
    max_unpaired_shares: Decimal = dec("MAX_UNPAIRED_SHARES","10")
    max_daily_spend: Decimal = dec("MAX_DAILY_SPEND","250")
    max_consecutive_failures: int = integer("MAX_CONSECUTIVE_FAILURES",3)
    cooldown_seconds: float = number("COOLDOWN_SECONDS",3)
    volatility_floor: float = number("VOLATILITY_FLOOR",0.00035)
    volatility_cap: float = number("VOLATILITY_CAP",0.02)
    momentum_weight: float = number("MOMENTUM_WEIGHT",0.35)
    speed_weight: float = number("SPEED_WEIGHT",0.20)
    basis_weight: float = number("BASIS_WEIGHT",0.10)
    price_distance_weight: float = number("PRICE_DISTANCE_WEIGHT",0.35)
    paper_start_balance: Decimal = dec("PAPER_START_BALANCE","1000")
    database_path: str = os.getenv("DATABASE_PATH","polymarket_bot.db")

settings = Settings()
