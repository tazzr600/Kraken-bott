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
        "on",
    }


def clean_env(name: str, default: str = "") -> str:
    return os.getenv(
        name,
        default
    ).strip()


@dataclass
class Settings:

    # =========================================================
    # KRAKEN
    # =========================================================

    kraken_api_key: str = clean_env(
        "KRAKEN_API_KEY"
    )

    kraken_api_secret: str = clean_env(
        "KRAKEN_API_SECRET"
    )

    # =========================================================
    # SAFETY
    # =========================================================

    # KEEP LIVE TRADING OFF WHILE TESTING
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

    # =========================================================
    # MARKET DATA
    # =========================================================

    # Leave blank.
    # The bot will automatically discover Kraken markets.
    symbols_raw: str = clean_env(
        "SYMBOLS",
        ""
    )

    timeframe: str = clean_env(
        "TIMEFRAME",
        "1m"
    )

    candles: int = int(
        clean_env(
            "CANDLES",
            "360"
        )
    )

    scan_seconds: int = int(
        clean_env(
            "SCAN_SECONDS",
            "15"
        )
    )

    # =========================================================
    # FULL KRAKEN MARKET SCANNER
    # =========================================================

    # Refresh the entire Kraken market universe every
    # 30 minutes by default.
    market_refresh_seconds: int = int(
        clean_env(
            "MARKET_REFRESH_SECONDS",
            "1800"
        )
    )

    # Number of liquid markets to actually run the expensive
    # OHLCV + machine-learning analysis on each scan.
    max_scan_symbols: int = int(
        clean_env(
            "MAX_SCAN_SYMBOLS",
            "12"
        )
    )

    # Minimum 24h quote volume required.
    min_quote_volume_usd: float = float(
        clean_env(
            "MIN_QUOTE_VOLUME_USD",
            "250000"
        )
    )

    # Maximum allowed bid/ask spread.
    max_spread_pct: float = float(
        clean_env(
            "MAX_SPREAD_PCT",
            "0.80"
        )
    )

    # Start with USD markets.
    #
    # You can later add:
    # USD,USDT,USDC
    #
    # Example:
    # ALLOWED_QUOTES=USD,USDT,USDC
    allowed_quotes: str = clean_env(
        "ALLOWED_QUOTES",
        "USD"
    )

    # =========================================================
    # MACHINE LEARNING
    # =========================================================

    train_every_seconds: int = int(
        clean_env(
            "TRAIN_EVERY_SECONDS",
            "900"
        )
    )

    forecast_bars: int = int(
        clean_env(
            "FORECAST_BARS",
            "5"
        )
    )

    min_probability: float = float(
        clean_env(
            "MIN_PROBABILITY",
            "0.54"
        )
    )

    min_expected_move: float = float(
        clean_env(
            "MIN_EXPECTED_MOVE",
            "0.001"
        )
    )

    min_training_accuracy: float = float(
        clean_env(
            "MIN_TRAINING_ACCURACY",
            "0.48"
        )
    )

    # Minimum expected reward relative to the emergency stop.
    # Kept below 1.0 so short-horizon PAPER entries are not
    # blocked by an overly strict risk/reward gate.
    min_reward_risk: float = float(
        clean_env(
            "MIN_REWARD_RISK",
            "0.75"
        )
    )

    # =========================================================
    # POSITION SIZING
    # =========================================================

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

    # =========================================================
    # RISK MANAGEMENT
    # =========================================================

    stop_loss_pct: float = float(
        clean_env(
            "STOP_LOSS_PCT",
            "0.008"
        )
    )

    take_profit_pct: float = float(
        clean_env(
            "TAKE_PROFIT_PCT",
            "0.008"
        )
    )

    max_hold_minutes: int = int(
        clean_env(
            "MAX_HOLD_MINUTES",
            "30"
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
            "20"
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
            "2"
        )
    )

    # =========================================================
    # PAPER ACCOUNT
    # =========================================================

    paper_start_balance: float = float(
        clean_env(
            "PAPER_START_BALANCE",
            "1000"
        )
    )

    # =========================================================
    # TRADING COST MODEL
    # =========================================================
    #
    # These are estimates for PAPER testing.
    # They are NOT guaranteed Kraken fees.
    #
    # We keep these configurable so we can tune the bot
    # to your actual Kraken fee tier later.

    round_trip_cost_pct: float = float(
        clean_env(
            "ROUND_TRIP_COST_PCT",
            "0.40"
        )
    )

    slippage_buffer_pct: float = float(
        clean_env(
            "SLIPPAGE_BUFFER_PCT",
            "0.10"
        )
    )

    # =========================================================
    # SYMBOL HELPERS
    # =========================================================

    @property
    def symbols(self):
        return [
            x.strip()
            for x in self.symbols_raw.split(",")
            if x.strip()
        ]

    @property
    def allowed_quote_list(self):
        return [
            x.strip().upper()
            for x in self.allowed_quotes.split(",")
            if x.strip()
        ]


settings = Settings()
