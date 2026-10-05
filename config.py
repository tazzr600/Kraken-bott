import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


def flag(name: str, default: bool = False) -> bool:
    return os.getenv(
        name,
        str(default)
    ).strip().lower() in {
        "1",
        "true",
        "yes",
        "on"
    }


def clean_env(name: str, default: str = "") -> str:
    """
    Removes accidental spaces/newlines when API
    credentials are pasted into Railway.
    """
    return os.getenv(
        name,
        default
    ).strip()


@dataclass
class Settings:

    # --------------------------------------------------
    # KRAKEN
    # --------------------------------------------------

    kraken_api_key: str = clean_env(
        "KRAKEN_API_KEY"
    )

    kraken_api_secret: str = clean_env(
        "KRAKEN_API_SECRET"
    )

    # --------------------------------------------------
    # TRADING SAFETY
    # --------------------------------------------------

    live_trading: bool = flag(
        "LIVE_TRADING",
        False
    )

    dry_run: bool = flag(
        "DRY_RUN",
        True
    )

    autonomous: bool = flag(
        "AUTONOMOUS_MODE",
        True
    )

    # --------------------------------------------------
    # MARKET
    # --------------------------------------------------

    symbols_raw: str = clean_env(
        "SYMBOLS",
        "BTC/USD,ETH/USD,SOL/USD"
    )

    timeframe: str = clean_env(
        "TIMEFRAME",
        "5m"
    )

    candles: int = int(
        clean_env(
            "CANDLES",
            "720"
        )
    )

    scan_seconds: int = int(
        clean_env(
            "SCAN_SECONDS",
            "20"
        )
    )

    # --------------------------------------------------
    # MACHINE LEARNING
    # --------------------------------------------------

    train_every_seconds: int = int(
        clean_env(
            "TRAIN_EVERY_SECONDS",
            "900"
        )
    )

    forecast_bars: int = int(
        clean_env(
            "FORECAST_BARS",
            "3"
        )
    )

    min_probability: float = float(
        clean_env(
            "MIN_PROBABILITY",
            "0.60"
        )
    )

    min_expected_move: float = float(
        clean_env(
            "MIN_EXPECTED_MOVE",
            "0.004"
        )
    )

    min_training_accuracy: float = float(
        clean_env(
            "MIN_TRAINING_ACCURACY",
            "0.52"
        )
    )

    # --------------------------------------------------
    # POSITION / RISK
    # --------------------------------------------------

    max_trade_usd: float = float(
        clean_env(
            "MAX_TRADE_USD",
            "50"
        )
    )

    max_position_pct: float = float(
        clean_env(
            "MAX_POSITION_PCT",
            "0.05"
        )
    )

    stop_loss_pct: float = float(
        clean_env(
            "STOP_LOSS_PCT",
            "0.008"
        )
    )

    take_profit_pct: float = float(
        clean_env(
            "TAKE_PROFIT_PCT",
            "0.012"
        )
    )

    max_hold_minutes: int = int(
        clean_env(
            "MAX_HOLD_MINUTES",
            "90"
        )
    )

    daily_loss_limit_usd: float = float(
        clean_env(
            "DAILY_LOSS_LIMIT_USD",
            "25"
        )
    )

    max_trades_per_day: int = int(
        clean_env(
            "MAX_TRADES_PER_DAY",
            "10"
        )
    )

    max_consecutive_losses: int = int(
        clean_env(
            "MAX_CONSECUTIVE_LOSSES",
            "3"
        )
    )

    cooldown_minutes: int = int(
        clean_env(
            "COOLDOWN_MINUTES",
            "15"
        )
    )

    paper_start_balance: float = float(
        clean_env(
            "PAPER_START_BALANCE",
            "1000"
        )
    )

    # --------------------------------------------------
    # COST ESTIMATES
    # --------------------------------------------------

    round_trip_cost_pct: float = float(
        clean_env(
            "ROUND_TRIP_COST_PCT",
            "1.60"
        )
    )

    slippage_buffer_pct: float = float(
        clean_env(
            "SLIPPAGE_BUFFER_PCT",
            "0.15"
        )
    )

    # --------------------------------------------------
    # SYMBOL LIST
    # --------------------------------------------------

    @property
    def symbols(self):

        return [
            x.strip()
            for x in self.symbols_raw.split(",")
            if x.strip()
        ]


settings = Settings()
