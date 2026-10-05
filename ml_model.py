from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    VotingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


FEATURES = [
    # Momentum
    "ret_1",
    "ret_3",
    "ret_6",
    "ret_12",
    "roc_20",

    # Trend
    "ema_fast_gap",
    "ema_slow_gap",
    "ema_cross",
    "trend_15",
    "trend_30",
    "adx_like",

    # Mean reversion
    "rsi",
    "bb_position",
    "bb_width",

    # Breakout / price action
    "breakout_up",
    "breakout_down",
    "range_pct",
    "close_location",

    # Volatility
    "atr_pct",
    "volatility",

    # Volume
    "volume_z",
    "volume_ratio",

    # MACD
    "macd",
    "macd_signal",
    "macd_hist",

    # VWAP
    "vwap_gap",

    # Market structure
    "high_20_gap",
    "low_20_gap",
]


def _safe_series(series, default=0.0):
    return pd.to_numeric(
        series,
        errors="coerce"
    ).replace(
        [np.inf, -np.inf],
        np.nan
    ).fillna(default)


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()

    for col in ["open", "high", "low", "close", "volume"]:
        x[col] = _safe_series(x[col])

    close = x["close"]
    high = x["high"]
    low = x["low"]
    volume = x["volume"]

    # ---------------------------------------------------------
    # RETURNS / MOMENTUM
    # ---------------------------------------------------------

    x["ret_1"] = close.pct_change(1)
    x["ret_3"] = close.pct_change(3)
    x["ret_6"] = close.pct_change(6)
    x["ret_12"] = close.pct_change(12)
    x["roc_20"] = close.pct_change(20)

    # ---------------------------------------------------------
    # EMA / TREND
    # ---------------------------------------------------------

    ema_fast = close.ewm(span=9, adjust=False).mean()
    ema_slow = close.ewm(span=21, adjust=False).mean()

    x["ema_fast_gap"] = close / ema_fast - 1
    x["ema_slow_gap"] = close / ema_slow - 1
    x["ema_cross"] = ema_fast / ema_slow - 1

    x["trend_15"] = close / close.shift(15) - 1
    x["trend_30"] = close / close.shift(30) - 1

    # ---------------------------------------------------------
    # RSI
    # ---------------------------------------------------------

    delta = close.diff()

    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()

    rs = gain / loss.replace(0, np.nan)

    x["rsi"] = 100 - (100 / (1 + rs))
    x["rsi"] = x["rsi"].fillna(50)

    # ---------------------------------------------------------
    # ATR
    # ---------------------------------------------------------

    previous_close = close.shift(1)

    tr = pd.concat(
        [
            high - low,
            (high - previous_close).abs(),
            (low - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    atr = tr.rolling(14).mean()

    x["atr_pct"] = atr / close.replace(0, np.nan)

    # ---------------------------------------------------------
    # VOLATILITY
    # ---------------------------------------------------------

    x["volatility"] = (
        close.pct_change()
        .rolling(20)
        .std()
    )

    # ---------------------------------------------------------
    # BOLLINGER BANDS
    # ---------------------------------------------------------

    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std()

    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std

    bb_width = (
        (bb_upper - bb_lower)
        / bb_mid.replace(0, np.nan)
    )

    bb_position = (
        (close - bb_lower)
        / (bb_upper - bb_lower).replace(0, np.nan)
    )

    x["bb_width"] = bb_width
    x["bb_position"] = bb_position

    # ---------------------------------------------------------
    # BREAKOUTS
    # ---------------------------------------------------------

    previous_high_20 = high.shift(1).rolling(20).max()
    previous_low_20 = low.shift(1).rolling(20).min()

    x["breakout_up"] = (
        close / previous_high_20.replace(0, np.nan) - 1
    )

    x["breakout_down"] = (
        close / previous_low_20.replace(0, np.nan) - 1
    )

    # ---------------------------------------------------------
    # CANDLE / PRICE ACTION
    # ---------------------------------------------------------

    candle_range = (
        high - low
    ).replace(0, np.nan)

    x["range_pct"] = candle_range / close.replace(0, np.nan)

    x["close_location"] = (
        (close - low) / candle_range
    )

    # ---------------------------------------------------------
    # ADX-LIKE TREND STRENGTH
    # ---------------------------------------------------------

    plus_move = high.diff()
    minus_move = -low.diff()

    plus_dm = plus_move.where(
        (plus_move > minus_move) &
        (plus_move > 0),
        0
    )

    minus_dm = minus_move.where(
        (minus_move > plus_move) &
        (minus_move > 0),
        0
    )

    atr_safe = atr.replace(0, np.nan)

    plus_di = (
        100 * plus_dm.rolling(14).mean()
        / atr_safe
    )

    minus_di = (
        100 * minus_dm.rolling(14).mean()
        / atr_safe
    )

    dx = (
        100
        * (plus_di - minus_di).abs()
        / (plus_di + minus_di).replace(0, np.nan)
    )

    x["adx_like"] = dx.rolling(14).mean()

    # ---------------------------------------------------------
    # VOLUME
    # ---------------------------------------------------------

    volume_mean = volume.rolling(20).mean()
    volume_std = volume.rolling(20).std()

    x["volume_z"] = (
        (volume - volume_mean)
        / volume_std.replace(0, np.nan)
    )

    x["volume_ratio"] = (
        volume
        / volume_mean.replace(0, np.nan)
    )

    # ---------------------------------------------------------
    # MACD
    # ---------------------------------------------------------

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()

    macd = ema12 - ema26
    macd_signal = macd.ewm(span=9, adjust=False).mean()

    x["macd"] = macd / close.replace(0, np.nan)
    x["macd_signal"] = (
        macd_signal / close.replace(0, np.nan)
    )
    x["macd_hist"] = (
        (macd - macd_signal)
        / close.replace(0, np.nan)
    )

    # ---------------------------------------------------------
    # VWAP
    # ---------------------------------------------------------

    typical_price = (
        high + low + close
    ) / 3

    cumulative_volume = volume.cumsum()

    vwap = (
        typical_price * volume
    ).cumsum() / cumulative_volume.replace(0, np.nan)

    x["vwap_gap"] = (
        close / vwap.replace(0, np.nan) - 1
    )

    # ---------------------------------------------------------
    # MARKET STRUCTURE
    # ---------------------------------------------------------

    rolling_high = high.rolling(20).max()
    rolling_low = low.rolling(20).min()

    x["high_20_gap"] = (
        close / rolling_high.replace(0, np.nan) - 1
    )

    x["low_20_gap"] = (
        close / rolling_low.replace(0, np.nan) - 1
    )

    # Clean everything
    x[FEATURES] = (
        x[FEATURES]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
    )

    return x


# =============================================================
# STRATEGY ENGINE
# =============================================================

def strategy_snapshot(df: pd.DataFrame) -> dict:
    f = make_features(df)

    latest = f.iloc[-1]

    votes = {}

    # Momentum
    momentum = (
        latest["ret_3"] > 0
        and latest["ret_6"] > 0
        and latest["roc_20"] > 0
    )

    votes["momentum"] = 1 if momentum else -1

    # Trend
    trend = (
        latest["ema_cross"] > 0
        and latest["trend_15"] > 0
        and latest["trend_30"] > 0
    )

    votes["trend"] = 1 if trend else -1

    # Mean reversion
    if latest["rsi"] < 35:
        mean_reversion = 1
    elif latest["rsi"] > 65:
        mean_reversion = -1
    else:
        mean_reversion = 0

    votes["mean_reversion"] = mean_reversion

    # Breakout
    if latest["breakout_up"] > 0:
        breakout = 1
    elif latest["breakout_down"] < 0:
        breakout = -1
    else:
        breakout = 0

    votes["breakout"] = breakout

    # VWAP
    votes["vwap"] = (
        1 if latest["vwap_gap"] > 0
        else -1
    )

    # MACD
    votes["macd"] = (
        1 if latest["macd_hist"] > 0
        else -1
    )

    # Volume confirmation
    if latest["volume_ratio"] > 1.2:
        votes["volume"] = (
            1
            if latest["close_location"] > 0.5
            else -1
        )
    else:
        votes["volume"] = 0

    values = list(votes.values())

    score = (
        sum(values) / len(values)
        if values
        else 0
    )

    agreement = (
        sum(1 for v in values if v > 0)
        / len(values)
        if values
        else 0
    )

    return {
        "votes": votes,
        "strategy_score": float(score),
        "strategy_agreement": float(agreement),
    }


# =============================================================
# MARKET REGIME
# =============================================================

def detect_regime(df: pd.DataFrame) -> str:
    f = make_features(df)
    latest = f.iloc[-1]

    trend_strength = abs(
        float(latest["trend_30"])
    )

    volatility = abs(
        float(latest["volatility"])
    )

    if trend_strength > 0.025:
        return "strong_trend"

    if volatility > 0.02:
        return "high_volatility"

    if trend_strength < 0.008:
        return "range"

    return "mixed"


# =============================================================
# TRAINING DATA
# =============================================================

def build_training_set(
    df: pd.DataFrame,
    forecast_bars: int = 3,
    cost: float = 0.0,
):
    f = make_features(df)

    future_return = (
        f["close"]
        .shift(-forecast_bars)
        / f["close"]
        - 1
    )

    data = f[FEATURES].copy()

    data["future_return"] = future_return

    data = data.replace(
        [np.inf, -np.inf],
        np.nan
    ).dropna()

    if len(data) < 100:
        raise ValueError(
            f"Not enough training samples: {len(data)}"
        )

    # IMPORTANT:
    # Train the classifier on direction.
    #
    # Profitability/cost is handled later by the trading
    # decision engine. This prevents the training set from
    # becoming 100% class 0 during quiet markets.
    data["target"] = (
        data["future_return"] > 0
    ).astype(int)

    X = data[FEATURES]
    y = data["target"]

    # Make absolutely sure both classes exist.
    if y.nunique() < 2:
        # Fall back to median split only when the market data
        # is extraordinarily one-sided.
        median_return = data["future_return"].median()

        y = (
            data["future_return"] > median_return
        ).astype(int)

    if y.nunique() < 2:
        raise ValueError(
            "Training data contains only one class."
        )

    return (
        X,
        y,
        data["future_return"]
    )


# =============================================================
# MODEL STATE
# =============================================================

@dataclass
class ModelState:
    model: object
    accuracy: float
    trained_at: float
    samples: int
    regime: str
    strategy_weights: dict = field(default_factory=dict)


# =============================================================
# TRAIN MODEL
# =============================================================

def train_model(
    df: pd.DataFrame,
    forecast_bars: int = 3,
    cost: float = 0.0,
) -> ModelState:

    X, y, future_returns = build_training_set(
        df,
        forecast_bars,
        cost,
    )

    split = int(len(X) * 0.80)

    if split < 50:
        raise ValueError(
            "Not enough data for training split."
        )

    X_train = X.iloc[:split]
    X_test = X.iloc[split:]

    y_train = y.iloc[:split]
    y_test = y.iloc[split:]

    # Make sure training itself has both classes.
    if y_train.nunique() < 2:
        raise ValueError(
            "Training window contains only one class."
        )

    # Make sure test has both classes when possible.
    # If the final test section happens to contain one
    # direction only, accuracy can still be calculated.
    models = [
        (
            "hist",
            HistGradientBoostingClassifier(
                max_iter=150,
                learning_rate=0.05,
                max_leaf_nodes=15,
                l2_regularization=1.0,
                random_state=42,
            ),
        ),

        (
            "rf",
            RandomForestClassifier(
                n_estimators=250,
                max_depth=8,
                min_samples_leaf=5,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
            ),
        ),

        (
            "logistic",
            Pipeline(
                [
                    (
                        "scale",
                        StandardScaler()
                    ),
                    (
                        "model",
                        LogisticRegression(
                            max_iter=1000,
                            class_weight="balanced",
                        ),
                    ),
                ]
            ),
        ),
    ]

    ensemble = VotingClassifier(
        estimators=models,
        voting="soft",
        weights=[2, 2, 1],
        flatten_transform=True,
    )

    ensemble.fit(
        X_train,
        y_train
    )

    predictions = ensemble.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    regime = detect_regime(df)

    snapshot = strategy_snapshot(df)

    return ModelState(
        model=ensemble,
        accuracy=float(accuracy),
        trained_at=time.time(),
        samples=len(X),
        regime=regime,
        strategy_weights={
            "strategy_score": snapshot["strategy_score"],
            "strategy_agreement": snapshot[
                "strategy_agreement"
            ],
        },
    )


# =============================================================
# PREDICTION
# =============================================================

def predict(
    state: ModelState,
    df: pd.DataFrame,
    forecast_bars: int = 3,
):
    if state is None:
        return None

    features = make_features(df)

    if len(features) < 2:
        return None

    latest = features[FEATURES].iloc[
        [-1]
    ]

    probabilities = state.model.predict_proba(
        latest
    )[0]

    classes = list(
        state.model.classes_
    )

    if 1 in classes:
        probability_up = float(
            probabilities[
                classes.index(1)
            ]
        )
    else:
        probability_up = 0.0

    # Estimate expected move from recent realized movement.
    recent_returns = (
        features["ret_1"]
        .tail(30)
        .dropna()
    )

    if len(recent_returns) == 0:
        expected_move = 0.0
    else:
        expected_move = float(
            recent_returns.std()
            * np.sqrt(
                max(1, forecast_bars)
            )
        )

    snapshot = strategy_snapshot(df)

    strategy_score = snapshot[
        "strategy_score"
    ]

    agreement = snapshot[
        "strategy_agreement"
    ]

    # Combine ML probability with strategy confirmation.
    ml_score = (
        probability_up - 0.5
    ) * 2

    combined_score = (
        ml_score * 0.70
        + strategy_score * 0.30
    )

    if combined_score > 0.10:
        direction = "LONG"
    elif combined_score < -0.10:
        direction = "SHORT"
    else:
        direction = "NEUTRAL"

    confidence = abs(
        combined_score
    )

    return {
        "probability_up": probability_up,

        "expected_move": expected_move,

        "direction": direction,

        "confidence": float(
            min(1.0, confidence)
        ),

        "strategy_score": float(
            strategy_score
        ),

        "strategy_agreement": float(
            agreement
        ),

        "regime": state.regime,

        "accuracy": state.accuracy,

        "samples": state.samples,

        "strategies": snapshot["votes"],
    }
