import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

def flag(name: str, default: bool = False) -> bool:
    return os.getenv(name, str(default)).lower() in {"1", "true", "yes", "on"}

@dataclass
class Settings:
    kraken_api_key: str = os.getenv("KRAKEN_API_KEY", "")
    kraken_api_secret: str = os.getenv("KRAKEN_API_SECRET", "")

    live_trading: bool = flag("LIVE_TRADING", False)
    dry_run: bool = flag("DRY_RUN", True)
    autonomous: bool = flag("AUTONOMOUS_MODE", True)

    symbols_raw: str = os.getenv("SYMBOLS", "BTC/USD,ETH/USD,SOL/USD")
    timeframe: str = os.getenv("TIMEFRAME", "5m")
    candles: int = int(os.getenv("CANDLES", "720"))
    scan_seconds: int = int(os.getenv("SCAN_SECONDS", "20"))

    train_every_seconds: int = int(os.getenv("TRAIN_EVERY_SECONDS", "900"))
    forecast_bars: int = int(os.getenv("FORECAST_BARS", "3"))
    min_probability: float = float(os.getenv("MIN_PROBABILITY", "0.60"))
    min_expected_move: float = float(os.getenv("MIN_EXPECTED_MOVE", "0.004"))
    min_training_accuracy: float = float(os.getenv("MIN_TRAINING_ACCURACY", "0.52"))

    max_trade_usd: float = float(os.getenv("MAX_TRADE_USD", "50"))
    max_position_pct: float = float(os.getenv("MAX_POSITION_PCT", "0.05"))
    stop_loss_pct: float = float(os.getenv("STOP_LOSS_PCT", "0.008"))
    take_profit_pct: float = float(os.getenv("TAKE_PROFIT_PCT", "0.012"))
    max_hold_minutes: int = int(os.getenv("MAX_HOLD_MINUTES", "90"))
    daily_loss_limit_usd: float = float(os.getenv("DAILY_LOSS_LIMIT_USD", "25"))
    max_trades_per_day: int = int(os.getenv("MAX_TRADES_PER_DAY", "10"))
    max_consecutive_losses: int = int(os.getenv("MAX_CONSECUTIVE_LOSSES", "3"))
    cooldown_minutes: int = int(os.getenv("COOLDOWN_MINUTES", "15"))
    paper_start_balance: float = float(os.getenv("PAPER_START_BALANCE", "1000"))

    # Starting estimate for cost filtering. Actual Kraken fees depend on tier/pair.
    round_trip_cost_pct: float = float(os.getenv("ROUND_TRIP_COST_PCT", "1.60"))
    slippage_buffer_pct: float = float(os.getenv("SLIPPAGE_BUFFER_PCT", "0.15"))

    @property
    def symbols(self):
        return [x.strip() for x in self.symbols_raw.split(",") if x.strip()]

settings = Settings()
