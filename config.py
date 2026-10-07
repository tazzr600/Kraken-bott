import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

def flag(name, default=False):
    return os.getenv(name, str(default)).strip().lower() in {"1","true","yes","on"}

def clean(name, default=""):
    return os.getenv(name, default).strip()

@dataclass
class Settings:
    kraken_api_key: str = clean("KRAKEN_API_KEY")
    kraken_api_secret: str = clean("KRAKEN_API_SECRET")

    live_trading: bool = flag("LIVE_TRADING", False)
    dry_run: bool = flag("DRY_RUN", True)
    autonomous: bool = flag("AUTONOMOUS_MODE", True)

    symbols_raw: str = clean("SYMBOLS", "")
    timeframe: str = clean("TIMEFRAME", "5m")
    candles: int = int(clean("CANDLES", "720"))
    scan_seconds: int = int(clean("SCAN_SECONDS", "15"))
    market_refresh_seconds: int = int(clean("MARKET_REFRESH_SECONDS", "1800"))
    max_scan_symbols: int = int(clean("MAX_SCAN_SYMBOLS", "20"))
    min_quote_volume_usd: float = float(clean("MIN_QUOTE_VOLUME_USD", "250000"))
    max_spread_pct: float = float(clean("MAX_SPREAD_PCT", "0.80"))
    allowed_quotes: str = clean("ALLOWED_QUOTES", "USD")
    trade_quotes: str = clean("TRADE_QUOTES", "USD")

    # Two-sided inventory engine
    min_market_spread_pct: float = float(clean("MIN_MARKET_SPREAD_PCT", "0.20"))
    min_net_edge_pct: float = float(clean("MIN_NET_EDGE_PCT", "0.20"))
    market_maker_buffer_pct: float = float(clean("MARKET_MAKER_BUFFER_PCT", "0.20"))
    inventory_skew_pct: float = float(clean("INVENTORY_SKEW_PCT", "0.35"))
    max_open_markets: int = int(clean("MAX_OPEN_MARKETS", "5"))

    # Position/risk
    max_trade_usd: float = float(clean("MAX_TRADE_USD", "25"))
    max_position_pct: float = float(clean("MAX_POSITION_PCT", "0.05"))
    daily_loss_limit_usd: float = float(clean("DAILY_LOSS_LIMIT_USD", "25"))
    max_trades_per_day: int = int(clean("MAX_TRADES_PER_DAY", "20"))
    max_consecutive_losses: int = int(clean("MAX_CONSECUTIVE_LOSSES", "3"))
    cooldown_minutes: int = int(clean("COOLDOWN_MINUTES", "2"))

    # Execution cost model
    round_trip_cost_pct: float = float(clean("ROUND_TRIP_COST_PCT", "0.40"))
    slippage_buffer_pct: float = float(clean("SLIPPAGE_BUFFER_PCT", "0.10"))

    # Retained compatibility settings
    stop_loss_pct: float = float(clean("STOP_LOSS_PCT", "0.008"))
    take_profit_pct: float = float(clean("TAKE_PROFIT_PCT", "0.012"))
    max_hold_minutes: int = int(clean("MAX_HOLD_MINUTES", "90"))
    train_every_seconds: int = int(clean("TRAIN_EVERY_SECONDS", "900"))
    forecast_bars: int = int(clean("FORECAST_BARS", "3"))
    min_probability: float = float(clean("MIN_PROBABILITY", "0.60"))
    min_expected_move: float = float(clean("MIN_EXPECTED_MOVE", "0.004"))
    min_training_accuracy: float = float(clean("MIN_TRAINING_ACCURACY", "0.52"))
    min_reward_risk: float = float(clean("MIN_REWARD_RISK", "1.25"))
    performance_gate_trades: int = int(clean("PERFORMANCE_GATE_TRADES", "20"))
    performance_gate_profit_factor: float = float(clean("PERFORMANCE_GATE_PROFIT_FACTOR", "1.05"))
    paper_adaptive_entry: bool = flag("PAPER_ADAPTIVE_ENTRY", True)
    paper_min_probability: float = float(clean("PAPER_MIN_PROBABILITY", "0.56"))
    paper_min_training_accuracy: float = float(clean("PAPER_MIN_TRAINING_ACCURACY", "0.52"))
    paper_min_reward_risk: float = float(clean("PAPER_MIN_REWARD_RISK", "1.25"))
    profit_lock_trigger_pct: float = float(clean("PROFIT_LOCK_TRIGGER_PCT", "2.50"))
    profit_lock_trigger_buffer_pct: float = float(clean("PROFIT_LOCK_TRIGGER_BUFFER_PCT", "0.25"))
    profit_lock_min_net_pct: float = float(clean("PROFIT_LOCK_MIN_NET_PCT", "0.25"))
    paper_start_balance: float = float(clean("PAPER_START_BALANCE", "1000"))

    @property
    def symbols(self):
        return [x.strip() for x in self.symbols_raw.split(",") if x.strip()]

    @property
    def allowed_quote_list(self):
        return [x.strip().upper() for x in self.allowed_quotes.split(",") if x.strip()]

    @property
    def trade_quote_list(self):
        return [x.strip().upper() for x in self.trade_quotes.split(",") if x.strip()]

settings = Settings()
