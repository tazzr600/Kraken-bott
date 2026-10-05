from __future__ import annotations

import math
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


# ============================================================
# MULTI-STRATEGY FEATURE ENGINE
# ============================================================

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


# ============================================================
# FEATURE CALCULATION
# ============================================================

def make_features(df: pd.DataFrame) -> pd.DataFrame:

    x = df.copy()

    close = x["close"].astype(float)
    high = x["high"].astype(float)
    low = x["low"].astype(float)
    volume = x["volume"].astype(float)

    # --------------------------------------------------------
    # MOMENTUM
    # --------------------------------------------------------

    x["ret_1"] = close.pct_change(1)
    x["ret_3"] = close.pct_change(3)
    x["ret_6"] = close.pct_change(6)
    x["ret_12"] = close.pct_change(12)
    x["roc_20"] = close.pct_change(20)

    # --------------------------------------------------------
    # EMA / TREND
    # --------------------------------------------------------

    ema_fast = close.ewm(
        span=8,
        adjust=False
    ).mean()

    ema_slow = close.ewm(
        span=21,
        adjust=False
    ).mean()

    ema_50 = close.ewm(
        span=50,
        adjust=False
    ).mean()

    x["ema_fast_gap"] = (
        close / ema_fast - 1
    )

    x["ema_slow_gap"] = (
        close / ema_slow - 1
    )

    x["ema_cross"] = (
        ema_fast / ema_slow - 1
    )

    x["trend_15"] = close.pct_change(15)
    x["trend_30"] = close.pct_change(30)

    # --------------------------------------------------------
    # TRUE RANGE / ATR
    # --------------------------------------------------------

    prev_close = close.shift(1)

    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    atr = tr.rolling(14).mean()

    # --------------------------------------------------------
    # ADX-LIKE TREND STRENGTH
    # --------------------------------------------------------

    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = up_move.where(
        (up_move > down_move) &
        (up_move > 0),
        0.0,
    )

    minus_dm = down_move.where(
        (down_move > up_move) &
        (down_move > 0),
        0.0,
    )

    plus_di = (
        100 *
        plus_dm.rolling(14).mean()
        / atr.replace(0, np.nan)
    )

    minus_di = (
        100 *
        minus_dm.rolling(14).mean()
        / atr.replace(0, np.nan)
    )

    x["adx_like"] = (
        (plus_di - minus_di).abs()
        /
        (plus_di + minus_di).replace(
            0,
            np.nan,
        )
    )

    # --------------------------------------------------------
    # RSI
    # --------------------------------------------------------

    delta = close.diff()

    gain = (
        delta.clip(lower=0)
        .rolling(14)
        .mean()
    )

    loss = (
        -delta.clip(upper=0)
        .rolling(14)
        .mean()
    )

    rs = gain / loss.replace(
        0,
        np.nan,
    )

    x["rsi"] = (
        100 -
        (
            100 /
            (1 + rs)
        )
    ).fillna(50)

    # --------------------------------------------------------
    # BOLLINGER BANDS
    # --------------------------------------------------------

    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std()

    bb_upper = bb_mid + (
        2 * bb_std
    )

    bb_lower = bb_mid - (
        2 * bb_std
    )

    band_width = (
        bb_upper - bb_lower
    ).replace(
        0,
        np.nan,
    )

    x["bb_position"] = (
        (close - bb_lower)
        / band_width
    )

    x["bb_width"] = (
        band_width
        / bb_mid.replace(
            0,
            np.nan,
        )
    )

    # --------------------------------------------------------
    # BREAKOUT
    # --------------------------------------------------------

    previous_high_20 = (
        high
        .shift(1)
        .rolling(20)
        .max()
    )

    previous_low_20 = (
        low
        .shift(1)
        .rolling(20)
        .min()
    )

    x["breakout_up"] = (
        close /
        previous_high_20 -
        1
    )

    x["breakout_down"] = (
        close /
        previous_low_20 -
        1
    )

    # --------------------------------------------------------
    # PRICE ACTION
    # --------------------------------------------------------

    x["range_pct"] = (
        (high - low)
        /
        close.replace(
            0,
            np.nan,
        )
    )

    x["close_location"] = (
        (close - low)
        /
        (high - low).replace(
            0,
            np.nan,
        )
    )

    # --------------------------------------------------------
    # VOLATILITY
    # --------------------------------------------------------

    x["atr_pct"] = (
        atr /
        close.replace(
            0,
            np.nan,
        )
    )

    x["volatility"] = (
        close
        .pct_change()
        .rolling(30)
        .std()
    )

    # --------------------------------------------------------
    # VOLUME
    # --------------------------------------------------------

    volume_mean = (
        volume
        .rolling(30)
        .mean()
    )

    volume_std = (
        volume
        .rolling(30)
        .std()
    )

    x["volume_z"] = (
        (volume - volume_mean)
        /
        volume_std.replace(
            0,
            np.nan,
        )
    )

    x["volume_ratio"] = (
        volume /
        volume_mean.replace(
            0,
            np.nan,
        )
    )

    # --------------------------------------------------------
    # MACD
    # --------------------------------------------------------

    ema_12 = close.ewm(
        span=12,
        adjust=False,
    ).mean()

    ema_26 = close.ewm(
        span=26,
        adjust=False,
    ).mean()

    macd_raw = (
        ema_12 -
        ema_26
    )

    macd_signal_raw = (
        macd_raw
        .ewm(
            span=9,
            adjust=False,
        )
        .mean()
    )

    x["macd"] = (
        macd_raw /
        close.replace(
            0,
            np.nan,
        )
    )

    x["macd_signal"] = (
        macd_signal_raw /
        close.replace(
            0,
            np.nan,
        )
    )

    x["macd_hist"] = (
        (macd_raw - macd_signal_raw)
        /
        close.replace(
            0,
            np.nan,
        )
    )

    # --------------------------------------------------------
    # VWAP
    # --------------------------------------------------------

    typical_price = (
        high +
        low +
        close
    ) / 3

    price_volume = (
        typical_price *
        volume
    )

    rolling_pv = (
        price_volume
        .rolling(30)
        .sum()
    )

    rolling_volume = (
        volume
        .rolling(30)
        .sum()
    )

    vwap = (
        rolling_pv /
        rolling_volume.replace(
            0,
            np.nan,
        )
    )

    x["vwap_gap"] = (
        close /
        vwap -
        1
    )

    # --------------------------------------------------------
    # MARKET STRUCTURE
    # --------------------------------------------------------

    x["high_20_gap"] = (
        close /
        previous_high_20 -
        1
    )

    x["low_20_gap"] = (
        close /
        previous_low_20 -
        1
    )

    return x.replace(
        [
            np.inf,
            -np.inf,
        ],
        np.nan,
    )


# ============================================================
# STRATEGY ENGINE
# ============================================================

def strategy_snapshot(
    features: pd.DataFrame,
) -> dict:

    r = features.iloc[-1]

    # --------------------------------------------------------
    # MOMENTUM STRATEGY
    # --------------------------------------------------------

    momentum = np.mean(
        [
            np.sign(
                float(
                    r["ret_3"] or 0
                )
            ),
            np.sign(
                float(
                    r["ret_6"] or 0
                )
            ),
            np.sign(
                float(
                    r["roc_20"] or 0
                )
            ),
        ]
    )

    # --------------------------------------------------------
    # TREND STRATEGY
    # --------------------------------------------------------

    trend = np.mean(
        [
            np.sign(
                float(
                    r["ema_cross"] or 0
                )
            ),
            np.sign(
                float(
                    r["trend_15"] or 0
                )
            ),
            np.sign(
                float(
                    r["trend_30"] or 0
                )
            ),
        ]
    )

    # --------------------------------------------------------
    # MEAN REVERSION
    # --------------------------------------------------------

    rsi = float(
        r["rsi"]
    )

    bb_position = float(
        r["bb_position"]
    )

    if (
        rsi < 35
        or bb_position < 0.15
    ):
        mean_reversion = 1.0

    elif (
        rsi > 65
        or bb_position > 0.85
    ):
        mean_reversion = -1.0

    else:
        mean_reversion = 0.0

    # --------------------------------------------------------
    # BREAKOUT
    # --------------------------------------------------------

    breakout = 0.0

    if (
        float(
            r["breakout_up"]
            or 0
        ) > 0
    ):
        breakout = 1.0

    elif (
        float(
            r["breakout_down"]
            or 0
        ) < 0
    ):
        breakout = -1.0

    # --------------------------------------------------------
    # VWAP
    # --------------------------------------------------------

    vwap = np.sign(
        float(
            r["vwap_gap"] or 0
        )
    )

    # --------------------------------------------------------
    # MACD
    # --------------------------------------------------------

    macd = np.sign(
        float(
            r["macd_hist"] or 0
        )
    )

    # --------------------------------------------------------
    # VOLUME
    # --------------------------------------------------------

    volume = np.sign(
        float(
            r["volume_z"] or 0
        )
    )

    strategy_values = [
        momentum,
        trend,
        mean_reversion,
        breakout,
        vwap,
        macd,
        volume,
    ]

    strategy_score = float(
        np.mean(strategy_values)
    )

    return {
        "momentum": float(momentum),
        "trend": float(trend),
        "mean_reversion": float(
            mean_reversion
        ),
        "breakout": float(
            breakout
        ),
        "vwap": float(vwap),
        "macd": float(macd),
        "volume": float(volume),
        "strategy_score": strategy_score,
    }


# ============================================================
# MARKET REGIME
# ============================================================

def detect_regime(
    features: pd.DataFrame,
) -> str:

    r = features.iloc[-1]

    trend = abs(
        float(
            r["trend_30"] or 0
        )
    )

    volatility = float(
        r["volatility"] or 0
    )

    adx = float(
        r["adx_like"] or 0
    )

    if not math.isfinite(
        volatility
    ):
        return "unknown"

    if (
        trend > 0.025
        and adx > 0.25
    ):
        return "strong_trend"

    if volatility > 0.02:
        return "high_volatility"

    if (
        trend < 0.008
        and volatility < 0.012
    ):
        return "range"

    return "mixed"


# ============================================================
# TRAINING DATA
# ============================================================

def build_training_set(
    df: pd.DataFrame,
    horizon: int,
    cost: float,
):

    features = make_features(df)

    future_return = (
        df["close"].shift(
            -horizon
        )
        /
        df["close"]
        - 1
    )

    # Positive class only when the expected move
    # exceeds estimated trading costs.
    target = (
        future_return > cost
    ).astype(int)

    data = features[
        FEATURES
    ].copy()

    data["target"] = target

    data = data.dropna()

    if len(data) < 250:
        return None, None

    return (
        data[FEATURES],
        data["target"],
    )


# ============================================================
# MODEL STATE
# ============================================================

@dataclass
class ModelState:

    model: object | None = None

    accuracy: float = 0.0

    trained_at: float = 0.0

    samples: int = 0

    regime: str = "unknown"

    strategy_weights: dict = field(
        default_factory=dict
    )


# ============================================================
# TRAIN MULTI-MODEL ENSEMBLE
# ============================================================

def train_model(
    df: pd.DataFrame,
    horizon: int,
    cost: float,
) -> ModelState:

    X, y = build_training_set(
        df,
        horizon,
        cost,
    )

    state = ModelState()

    if X is None:
        return state

    split = int(
        len(X) * 0.80
    )

    if (
        split < 150
        or len(X) - split < 30
    ):
        return state

    X_train = X.iloc[:split]
    X_test = X.iloc[split:]

    y_train = y.iloc[:split]
    y_test = y.iloc[split:]

    # --------------------------------------------------------
    # MODEL 1: GRADIENT BOOSTING
    # --------------------------------------------------------

    gradient = HistGradientBoostingClassifier(
        max_iter=220,
        learning_rate=0.045,
        max_leaf_nodes=15,
        l2_regularization=1.0,
        random_state=42,
    )

    # --------------------------------------------------------
    # MODEL 2: RANDOM FOREST
    # --------------------------------------------------------

    forest = RandomForestClassifier(
        n_estimators=250,
        max_depth=7,
        min_samples_leaf=5,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1,
    )

    # --------------------------------------------------------
    # MODEL 3: LOGISTIC REGRESSION
    # --------------------------------------------------------

    logistic = Pipeline(
        [
            (
                "scale",
                StandardScaler(),
            ),
            (
                "classifier",
                LogisticRegression(
                    C=0.35,
                    max_iter=1000,
                    class_weight="balanced",
                ),
            ),
        ]
    )

    # --------------------------------------------------------
    # SOFT-VOTING ENSEMBLE
    # --------------------------------------------------------

    model = VotingClassifier(
        estimators=[
            (
                "gradient",
                gradient,
            ),
            (
                "forest",
                forest,
            ),
            (
                "logistic",
                logistic,
            ),
        ],
        voting="soft",
        weights=[
            2,
            2,
            1,
        ],
    )

    model.fit(
        X_train,
        y_train,
    )

    predictions = model.predict(
        X_test
    )

    accuracy = accuracy_score(
        y_test,
        predictions,
    )

    state.model = model

    state.accuracy = float(
        accuracy
    )

    state.trained_at = time.time()

    state.samples = len(
        X_train
    )

    features = make_features(
        df
    )

    state.regime = detect_regime(
        features
    )

    return state


# ============================================================
# PREDICTION
# ============================================================

def predict(
    state: ModelState,
    df: pd.DataFrame,
    horizon: int,
):

    if state.model is None:
        return None

    features = make_features(
        df
    )

    row = features[
        FEATURES
    ].tail(1)

    if row.isna().any().any():
        return None

    probabilities = (
        state.model
        .predict_proba(row)[0]
    )

    classes = list(
        state.model.classes_
    )

    if 1 in classes:

        p_up = float(
            probabilities[
                classes.index(1)
            ]
        )

    else:

        p_up = 0.5

    volatility = float(
        features[
            "volatility"
        ]
        .tail(1)
        .iloc[0]
    )

    if not math.isfinite(
        volatility
    ):
        return None

    # Estimate expected movement.
    expected_move = max(
        volatility
        *
        math.sqrt(
            max(
                1,
                horizon
            )
        ),
        0.0,
    )

    # Strategy-level confirmation.
    strategies = strategy_snapshot(
        features
    )

    strategy_score = float(
        strategies[
            "strategy_score"
        ]
    )

    strategy_agreement = abs(
        strategy_score
    )

    # Market regime.
    regime = detect_regime(
        features
    )

    # --------------------------------------------------------
    # CONFIDENCE
    # --------------------------------------------------------

    model_confidence = (
        abs(
            p_up - 0.5
        ) * 2
    )

    confidence = (
        0.70 *
        model_confidence
        +
        0.30 *
        strategy_agreement
    )

    confidence = max(
        0.0,
        min(
            1.0,
            confidence
        )
    )

    # --------------------------------------------------------
    # FINAL DIRECTION
    # --------------------------------------------------------

    if (
        p_up >= 0.50
        and strategy_score >= 0
    ):
        direction = "LONG"

    elif (
        p_up < 0.50
        and strategy_score <= 0
    ):
        direction = "SHORT"

    else:
        direction = "NEUTRAL"

    return {
        # Existing bot.py expects these:
        "probability_up": p_up,
        "expected_move": expected_move,

        # New intelligence:
        "direction": direction,
        "confidence": float(
            confidence
        ),
        "strategy_score": strategy_score,
        "strategy_agreement": float(
            strategy_agreement
        ),
        "regime": regime,

        # Individual strategy information:
        "strategies": strategies,
    }
