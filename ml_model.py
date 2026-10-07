from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ============================================================
# MODEL STATE
# ============================================================

@dataclass
class ModelState:

    classifier: object | None = None

    feature_columns: list[str] = field(
        default_factory=list
    )

    accuracy: float = 0.0

    positive_precision: float = 0.0

    samples: int = 0

    trained_at: float = 0.0

    training_cost: float = 0.0

    baseline_probability: float = 0.5


# ============================================================
# HELPERS
# ============================================================

def _safe_div(a, b):

    if isinstance(b, pd.Series):

        b = b.replace(
            0,
            np.nan
        )

    return a / b


def _clip(value, low, high):

    try:

        return float(
            np.clip(
                value,
                low,
                high
            )
        )

    except Exception:

        return float(low)


def _last(features, name, default=0.0):

    try:

        value = features[name].iloc[-1]

        if pd.isna(value):

            return float(default)

        return float(value)

    except Exception:

        return float(default)


# ============================================================
# RSI
# ============================================================

def rsi(
    close,
    period=14
):

    delta = close.diff()

    gain = delta.clip(
        lower=0
    )

    loss = (
        -delta.clip(
            upper=0
        )
    )

    avg_gain = (
        gain
        .ewm(
            alpha=1 / period,
            adjust=False
        )
        .mean()
    )

    avg_loss = (
        loss
        .ewm(
            alpha=1 / period,
            adjust=False
        )
        .mean()
    )

    rs = _safe_div(
        avg_gain,
        avg_loss
    )

    result = (
        100
        -
        (
            100
            /
            (
                1 + rs
            )
        )
    )

    return result.fillna(50)


# ============================================================
# ATR
# ============================================================

def atr(
    df,
    period=14
):

    high = df["high"]

    low = df["low"]

    close = df["close"]

    previous_close = close.shift(1)

    tr = pd.concat(
        [
            high - low,

            (
                high
                -
                previous_close
            ).abs(),

            (
                low
                -
                previous_close
            ).abs(),
        ],
        axis=1,
    ).max(
        axis=1
    )

    return (
        tr
        .rolling(
            period
        )
        .mean()
    )


# ============================================================
# MACD
# ============================================================

def macd(close):

    fast = (
        close
        .ewm(
            span=12,
            adjust=False
        )
        .mean()
    )

    slow = (
        close
        .ewm(
            span=26,
            adjust=False
        )
        .mean()
    )

    line = fast - slow

    signal = (
        line
        .ewm(
            span=9,
            adjust=False
        )
        .mean()
    )

    histogram = line - signal

    return (
        line,
        signal,
        histogram
    )


# ============================================================
# VWAP
# ============================================================

def vwap(
    df,
    period=50
):

    typical = (
        df["high"]
        +
        df["low"]
        +
        df["close"]
    ) / 3.0

    volume_price = (
        typical
        *
        df["volume"]
    )

    numerator = (
        volume_price
        .rolling(
            period
        )
        .sum()
    )

    denominator = (
        df["volume"]
        .rolling(
            period
        )
        .sum()
    )

    return _safe_div(
        numerator,
        denominator
    )


# ============================================================
# FEATURE ENGINE
# ============================================================

def build_features(
    df: pd.DataFrame
):

    data = df.copy()

    required = [
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]

    for column in required:

        if column not in data:

            raise ValueError(
                f"Missing OHLCV column: {column}"
            )

        data[column] = pd.to_numeric(
            data[column],
            errors="coerce"
        )

    close = data["close"]

    high = data["high"]

    low = data["low"]

    open_price = data["open"]

    volume = data["volume"]

    # ========================================================
    # RETURNS
    # ========================================================

    data["return_1"] = close.pct_change(1)

    data["return_2"] = close.pct_change(2)

    data["return_3"] = close.pct_change(3)

    data["return_5"] = close.pct_change(5)

    data["return_10"] = close.pct_change(10)

    data["return_20"] = close.pct_change(20)

    # ========================================================
    # MOVING AVERAGES
    # ========================================================

    data["sma_10"] = (
        close
        .rolling(10)
        .mean()
    )

    data["sma_20"] = (
        close
        .rolling(20)
        .mean()
    )

    data["sma_50"] = (
        close
        .rolling(50)
        .mean()
    )

    data["ema_9"] = (
        close
        .ewm(
            span=9,
            adjust=False
        )
        .mean()
    )

    data["ema_21"] = (
        close
        .ewm(
            span=21,
            adjust=False
        )
        .mean()
    )

    data["ema_50"] = (
        close
        .ewm(
            span=50,
            adjust=False
        )
        .mean()
    )

    # ========================================================
    # TREND DISTANCE
    # ========================================================

    data["price_vs_sma20"] = (
        _safe_div(
            close,
            data["sma_20"]
        )
        - 1
    )

    data["price_vs_sma50"] = (
        _safe_div(
            close,
            data["sma_50"]
        )
        - 1
    )

    data["ema9_vs_ema21"] = (
        _safe_div(
            data["ema_9"],
            data["ema_21"]
        )
        - 1
    )

    data["ema21_vs_ema50"] = (
        _safe_div(
            data["ema_21"],
            data["ema_50"]
        )
        - 1
    )

    # ========================================================
    # RSI
    # ========================================================

    data["rsi_14"] = rsi(
        close,
        14
    )

    data["rsi_7"] = rsi(
        close,
        7
    )

    data["rsi_change"] = (
        data["rsi_14"]
        .diff()
    )

    # ========================================================
    # MACD
    # ========================================================

    (
        data["macd"],
        data["macd_signal"],
        data["macd_hist"],
    ) = macd(close)

    data["macd_hist_change"] = (
        data["macd_hist"]
        .diff()
    )

    # ========================================================
    # ATR / VOLATILITY
    # ========================================================

    data["atr"] = atr(
        data,
        14
    )

    data["atr_pct"] = (
        _safe_div(
            data["atr"],
            close
        )
    )

    data["volatility_5"] = (
        data["return_1"]
        .rolling(5)
        .std()
    )

    data["volatility_10"] = (
        data["return_1"]
        .rolling(10)
        .std()
    )

    data["volatility_20"] = (
        data["return_1"]
        .rolling(20)
        .std()
    )

    # ========================================================
    # BOLLINGER
    # ========================================================

    bb_mid = (
        close
        .rolling(20)
        .mean()
    )

    bb_std = (
        close
        .rolling(20)
        .std()
    )

    bb_upper = (
        bb_mid
        +
        2 * bb_std
    )

    bb_lower = (
        bb_mid
        -
        2 * bb_std
    )

    data["bb_width"] = (
        _safe_div(
            bb_upper - bb_lower,
            bb_mid
        )
    )

    data["bb_position"] = (
        _safe_div(
            close - bb_lower,
            bb_upper - bb_lower
        )
    )

    # ========================================================
    # VOLUME
    # ========================================================

    data["volume_sma20"] = (
        volume
        .rolling(20)
        .mean()
    )

    data["volume_ratio"] = (
        _safe_div(
            volume,
            data["volume_sma20"]
        )
    )

    data["volume_change"] = (
        volume
        .pct_change()
    )

    # ========================================================
    # VWAP
    # ========================================================

    data["vwap"] = vwap(
        data,
        50
    )

    data["price_vs_vwap"] = (
        _safe_div(
            close,
            data["vwap"]
        )
        - 1
    )

    # ========================================================
    # BREAKOUT / BREAKDOWN
    # ========================================================

    previous_high_10 = (
        high
        .shift(1)
        .rolling(10)
        .max()
    )

    previous_low_10 = (
        low
        .shift(1)
        .rolling(10)
        .min()
    )

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

    previous_high_50 = (
        high
        .shift(1)
        .rolling(50)
        .max()
    )

    previous_low_50 = (
        low
        .shift(1)
        .rolling(50)
        .min()
    )

    data["breakout_10"] = (
        _safe_div(
            close,
            previous_high_10
        )
        - 1
    )

    data["breakdown_10"] = (
        _safe_div(
            close,
            previous_low_10
        )
        - 1
    )

    data["breakout_20"] = (
        _safe_div(
            close,
            previous_high_20
        )
        - 1
    )

    data["breakdown_20"] = (
        _safe_div(
            close,
            previous_low_20
        )
        - 1
    )

    data["breakout_50"] = (
        _safe_div(
            close,
            previous_high_50
        )
        - 1
    )

    data["breakdown_50"] = (
        _safe_div(
            close,
            previous_low_50
        )
        - 1
    )

    # ========================================================
    # MARKET STRUCTURE
    # ========================================================

    data["higher_high"] = (
        (
            high
            >
            high.shift(1)
        )
        .astype(float)
    )

    data["higher_low"] = (
        (
            low
            >
            low.shift(1)
        )
        .astype(float)
    )

    data["lower_high"] = (
        (
            high
            <
            high.shift(1)
        )
        .astype(float)
    )

    data["lower_low"] = (
        (
            low
            <
            low.shift(1)
        )
        .astype(float)
    )

    data["structure_score"] = (
        data["higher_high"]
        +
        data["higher_low"]
        -
        data["lower_high"]
        -
        data["lower_low"]
    ) / 2.0

    # ========================================================
    # CANDLE STRUCTURE
    # ========================================================

    candle_range = (
        high - low
    ).replace(
        0,
        np.nan
    )

    body = (
        close - open_price
    )

    data["body_pct"] = (
        _safe_div(
            body,
            open_price
        )
    )

    data["body_to_range"] = (
        _safe_div(
            body.abs(),
            candle_range
        )
    )

    data["upper_wick"] = (
        high
        -
        pd.concat(
            [
                open_price,
                close
            ],
            axis=1,
        ).max(axis=1)
    )

    data["lower_wick"] = (
        pd.concat(
            [
                open_price,
                close
            ],
            axis=1,
        ).min(axis=1)
        -
        low
    )

    data["upper_wick_pct"] = (
        _safe_div(
            data["upper_wick"],
            candle_range
        )
    )

    data["lower_wick_pct"] = (
        _safe_div(
            data["lower_wick"],
            candle_range
        )
    )

    # ========================================================
    # CLEAN
    # ========================================================

    data = data.replace(
        [
            np.inf,
            -np.inf
        ],
        np.nan
    )

    return data


# ============================================================
# STRATEGY ENGINE
# ============================================================

def strategy_signals(
    features: pd.DataFrame
):

    x = features.iloc[-1]

    # ========================================================
    # MOMENTUM
    # ========================================================

    momentum = 0.0

    momentum += (
        np.tanh(
            float(
                x.get(
                    "return_3",
                    0
                )
                or 0
            )
            * 45
        )
        * 0.20
    )

    momentum += (
        np.tanh(
            float(
                x.get(
                    "return_5",
                    0
                )
                or 0
            )
            * 40
        )
        * 0.30
    )

    momentum += (
        np.tanh(
            float(
                x.get(
                    "return_10",
                    0
                )
                or 0
            )
            * 25
        )
        * 0.30
    )

    momentum += (
        np.tanh(
            float(
                x.get(
                    "return_20",
                    0
                )
                or 0
            )
            * 15
        )
        * 0.20
    )

    momentum = _clip(
        momentum,
        -1,
        1
    )

    # ========================================================
    # TREND
    # ========================================================

    trend = 0.0

    trend += (
        np.tanh(
            float(
                x.get(
                    "price_vs_sma20",
                    0
                )
                or 0
            )
            * 20
        )
        * 0.25
    )

    trend += (
        np.tanh(
            float(
                x.get(
                    "price_vs_sma50",
                    0
                )
                or 0
            )
            * 15
        )
        * 0.20
    )

    trend += (
        np.tanh(
            float(
                x.get(
                    "ema9_vs_ema21",
                    0
                )
                or 0
            )
            * 30
        )
        * 0.30
    )

    trend += (
        np.tanh(
            float(
                x.get(
                    "ema21_vs_ema50",
                    0
                )
                or 0
            )
            * 25
        )
        * 0.25
    )

    trend = _clip(
        trend,
        -1,
        1
    )

    # ========================================================
    # MEAN REVERSION
    # ========================================================

    rsi_value = float(
        x.get(
            "rsi_14",
            50
        )
        or 50
    )

    rsi_change = float(
        x.get(
            "rsi_change",
            0
        )
        or 0
    )

    bb_position = float(
        x.get(
            "bb_position",
            0.5
        )
        or 0.5
    )

    mean_reversion = 0.0

    if rsi_value < 28:

        mean_reversion += 0.80

    elif rsi_value < 38:

        mean_reversion += 0.35

    elif rsi_value > 72:

        mean_reversion -= 0.80

    elif rsi_value > 62:

        mean_reversion -= 0.35

    if bb_position < 0.08:

        mean_reversion += 0.40

    elif bb_position < 0.20:

        mean_reversion += 0.18

    elif bb_position > 0.92:

        mean_reversion -= 0.40

    elif bb_position > 0.80:

        mean_reversion -= 0.18

    # Oversold + improving RSI is better than oversold
    # while continuing to fall.

    if rsi_value < 45 and rsi_change > 0:

        mean_reversion += 0.15

    if rsi_value > 55 and rsi_change < 0:

        mean_reversion -= 0.15

    mean_reversion = _clip(
        mean_reversion,
        -1,
        1
    )

    # ========================================================
    # BREAKOUT
    # ========================================================

    breakout = 0.0

    b10 = float(
        x.get(
            "breakout_10",
            0
        )
        or 0
    )

    b20 = float(
        x.get(
            "breakout_20",
            0
        )
        or 0
    )

    b50 = float(
        x.get(
            "breakout_50",
            0
        )
        or 0
    )

    d10 = float(
        x.get(
            "breakdown_10",
            0
        )
        or 0
    )

    d20 = float(
        x.get(
            "breakdown_20",
            0
        )
        or 0
    )

    d50 = float(
        x.get(
            "breakdown_50",
            0
        )
        or 0
    )

    breakout += np.tanh(
        b10 * 100
    ) * 0.20

    breakout += np.tanh(
        b20 * 100
    ) * 0.30

    breakout += np.tanh(
        b50 * 100
    ) * 0.20

    breakout += np.tanh(
        d10 * 100
    ) * 0.10

    breakout += np.tanh(
        d20 * 100
    ) * 0.10

    breakout += np.tanh(
        d50 * 100
    ) * 0.10

    volume_ratio = float(
        x.get(
            "volume_ratio",
            1
        )
        or 1
    )

    if volume_ratio > 1.50:

        breakout *= 1.25

    elif volume_ratio < 0.70:

        breakout *= 0.60

    breakout = _clip(
        breakout,
        -1,
        1
    )

    # ========================================================
    # VWAP
    # ========================================================

    price_vs_vwap = float(
        x.get(
            "price_vs_vwap",
            0
        )
        or 0
    )

    vwap_score = np.tanh(
        price_vs_vwap * 30
    )

    # ========================================================
    # MACD
    # ========================================================

    close = float(
        features["close"].iloc[-1]
    )

    macd_hist = float(
        x.get(
            "macd_hist",
            0
        )
        or 0
    )

    macd_change = float(
        x.get(
            "macd_hist_change",
            0
        )
        or 0
    )

    scale = max(
        abs(close),
        1e-9
    )

    macd_score = (
        np.tanh(
            macd_hist
            /
            scale
            *
            1000
        )
        * 0.70
        +
        np.tanh(
            macd_change
            /
            scale
            *
            1000
        )
        * 0.30
    )

    macd_score = _clip(
        macd_score,
        -1,
        1
    )

    # ========================================================
    # VOLUME
    # ========================================================

    volume_score = np.tanh(
        (
            volume_ratio
            - 1.0
        )
        *
        1.5
    )

    volume_score = _clip(
        volume_score,
        -1,
        1
    )

    # ========================================================
    # MARKET STRUCTURE
    # ========================================================

    structure_score = _clip(
        float(
            x.get(
                "structure_score",
                0
            )
            or 0
        ),
        -1,
        1
    )

    # ========================================================
    # VOLATILITY
    # ========================================================

    volatility = float(
        x.get(
            "volatility_20",
            0
        )
        or 0
    )

    atr_pct = float(
        x.get(
            "atr_pct",
            0
        )
        or 0
    )

    volatility_score = np.tanh(
        volatility * 100
    )

    # Extremely low volatility = poor day-trade
    # environment.

    if atr_pct < 0.001:

        volatility_score *= 0.50

    # ========================================================
    # STRATEGIES
    # ========================================================

    strategies = {

        "momentum":
            momentum,

        "trend":
            trend,

        "mean_reversion":
            mean_reversion,

        "breakout":
            breakout,

        "vwap":
            _clip(
                vwap_score,
                -1,
                1
            ),

        "macd":
            macd_score,

        "volume":
            volume_score,

        "volatility":
            _clip(
                volatility_score,
                -1,
                1
            ),

        "market_structure":
            structure_score,
    }

    # ========================================================
    # WEIGHTS
    # ========================================================

    weights = {

        "momentum": 0.16,

        "trend": 0.18,

        "mean_reversion": 0.07,

        "breakout": 0.14,

        "vwap": 0.10,

        "macd": 0.12,

        "volume": 0.08,

        "volatility": 0.04,

        "market_structure": 0.11,
    }

    weighted_sum = 0.0

    total_weight = 0.0

    for name, weight in weights.items():

        weighted_sum += (
            strategies[name]
            *
            weight
        )

        total_weight += weight

    strategy_score = (
        weighted_sum
        /
        total_weight
        if total_weight
        else 0
    )

    strategy_score = _clip(
        strategy_score,
        -1,
        1
    )

    # ========================================================
    # AGREEMENT
    # ========================================================

    values = np.array(
        list(
            strategies.values()
        ),
        dtype=float
    )

    bullish = np.sum(
        values > 0.10
    )

    bearish = np.sum(
        values < -0.10
    )

    total = len(values)

    if strategy_score > 0:

        agreement = (
            bullish
            /
            total
            if total
            else 0
        )

    elif strategy_score < 0:

        agreement = (
            bearish
            /
            total
            if total
            else 0
        )

    else:

        agreement = 0.0

    agreement = _clip(
        agreement,
        0,
        1
    )

    # ========================================================
    # BEARISH REVERSAL SCORE
    # ========================================================

    bearish_reversal = 0.0

    # Price momentum turning down.

    if (
        float(
            x.get(
                "return_3",
                0
            )
            or 0
        )
        < 0
    ):

        bearish_reversal += 0.15

    if (
        float(
            x.get(
                "return_5",
                0
            )
            or 0
        )
        < 0
    ):

        bearish_reversal += 0.15

    # EMA deterioration.

    if (
        float(
            x.get(
                "ema9_vs_ema21",
                0
            )
            or 0
        )
        < 0
    ):

        bearish_reversal += 0.15

    if (
        float(
            x.get(
                "ema21_vs_ema50",
                0
            )
            or 0
        )
        < 0
    ):

        bearish_reversal += 0.15

    # MACD turning negative.

    if macd_hist < 0:

        bearish_reversal += 0.15

    if macd_change < 0:

        bearish_reversal += 0.10

    # Price below VWAP.

    if price_vs_vwap < 0:

        bearish_reversal += 0.10

    # Market structure.

    if structure_score < 0:

        bearish_reversal += 0.10

    bearish_reversal = _clip(
        bearish_reversal,
        0,
        1
    )

    # ========================================================
    # REGIME
    # ========================================================

    trend_value = float(
        x.get(
            "ema21_vs_ema50",
            0
        )
        or 0
    )

    if abs(trend_value) > 0.015:

        regime = (
            "TRENDING_UP"
            if trend_value > 0
            else "TRENDING_DOWN"
        )

    elif volatility > 0.02:

        regime = "HIGH_VOLATILITY"

    else:

        regime = "RANGE"

    return {

        "strategy_score":
            strategy_score,

        "strategy_agreement":
            agreement,

        "bearish_reversal":
            bearish_reversal,

        "strategies":
            {
                key: float(value)
                for key, value
                in strategies.items()
            },

        "regime":
            regime,
    }


# ============================================================
# TRAIN MODEL
# ============================================================

def train_model(
    df: pd.DataFrame,
    forecast_bars: int = 3,
    trading_cost: float = 0.004
):

    features = build_features(
        df
    )

    # ========================================================
    # TARGET
    # ========================================================

    future_return = (
        features["close"]
        .shift(
            -forecast_bars
        )
        /
        features["close"]
        - 1
    )

    # Positive class means the future move exceeded
    # the estimated round-trip cost.

    target = (
        future_return
        >
        trading_cost
    ).astype(int)

    # ========================================================
    # FEATURES
    # ========================================================

    feature_columns = [

        "return_1",
        "return_2",
        "return_3",
        "return_5",
        "return_10",
        "return_20",

        "price_vs_sma20",
        "price_vs_sma50",

        "ema9_vs_ema21",
        "ema21_vs_ema50",

        "rsi_14",
        "rsi_7",
        "rsi_change",

        "macd",
        "macd_signal",
        "macd_hist",
        "macd_hist_change",

        "atr_pct",

        "volatility_5",
        "volatility_10",
        "volatility_20",

        "bb_width",
        "bb_position",

        "volume_ratio",
        "volume_change",

        "price_vs_vwap",

        "breakout_10",
        "breakdown_10",
        "breakout_20",
        "breakdown_20",
        "breakout_50",
        "breakdown_50",

        "higher_high",
        "higher_low",
        "lower_high",
        "lower_low",

        "structure_score",

        "body_pct",
        "body_to_range",

        "upper_wick_pct",
        "lower_wick_pct",
    ]

    training = (
        features[
            feature_columns
        ]
        .copy()
    )

    valid = (
        training.notna()
        .all(axis=1)
        &
        future_return.notna()
    )

    training = training.loc[
        valid
    ]

    labels = target.loc[
        valid
    ]

    # ========================================================
    # SANITY CHECK
    # ========================================================

    if len(training) < 150:

        return ModelState(
            classifier=None,
            feature_columns=feature_columns,
            accuracy=0.0,
            samples=len(training),
            trained_at=time.time(),
            training_cost=trading_cost,
            baseline_probability=0.5,
        )

    # ========================================================
    # CHRONOLOGICAL SPLIT
    # ========================================================

    split = int(
        len(training)
        *
        0.80
    )

    split = max(
        100,
        min(
            split,
            len(training) - 30
        )
    )

    X_train = training.iloc[
        :split
    ]

    y_train = labels.iloc[
        :split
    ]

    X_test = training.iloc[
        split:
    ]

    y_test = labels.iloc[
        split:
    ]

    if y_train.nunique() < 2:

        baseline = float(
            y_train.mean()
        )

        return ModelState(
            classifier=None,
            feature_columns=feature_columns,
            accuracy=0.5,
            samples=len(training),
            trained_at=time.time(),
            training_cost=trading_cost,
            baseline_probability=baseline,
        )

    # ========================================================
    # MODEL 1
    # ========================================================

    gradient = HistGradientBoostingClassifier(

        max_iter=180,

        learning_rate=0.045,

        max_leaf_nodes=15,

        max_depth=5,

        min_samples_leaf=12,

        l2_regularization=0.15,

        random_state=42,
    )

    # ========================================================
    # MODEL 2
    # ========================================================

    forest = RandomForestClassifier(

        n_estimators=250,

        max_depth=8,

        min_samples_leaf=8,

        max_features="sqrt",

        class_weight="balanced_subsample",

        random_state=42,

        n_jobs=-1,
    )

    # ========================================================
    # MODEL 3
    # ========================================================

    logistic = Pipeline(
        [
            (
                "scale",
                StandardScaler()
            ),

            (
                "model",
                LogisticRegression(
                    C=0.35,
                    max_iter=1500,
                    class_weight="balanced",
                    random_state=42,
                )
            ),
        ]
    )

    # ========================================================
    # TRAIN
    # ========================================================

    gradient.fit(
        X_train,
        y_train
    )

    forest.fit(
        X_train,
        y_train
    )

    logistic.fit(
        X_train,
        y_train
    )

    # ========================================================
    # VALIDATION
    # ========================================================

    p1 = (
        gradient
        .predict_proba(
            X_test
        )[:, 1]
    )

    p2 = (
        forest
        .predict_proba(
            X_test
        )[:, 1]
    )

    p3 = (
        logistic
        .predict_proba(
            X_test
        )[:, 1]
    )

    probabilities = (
        p1 * 0.45
        +
        p2 * 0.35
        +
        p3 * 0.20
    )

    predictions = (
        probabilities
        >= 0.50
    ).astype(int)

    accuracy = float(
        (
            predictions
            ==
            y_test.to_numpy()
        ).mean()
    )

    # Positive-class precision is more useful for a long-only trader:
    # when the model says "qualified upside", how often was there actually
    # a move larger than the modeled trading cost?
    positive_mask = predictions == 1
    positive_precision = float(
        y_test.to_numpy()[positive_mask].mean()
    ) if positive_mask.any() else 0.0

    # ========================================================
    # ENSEMBLE
    # ========================================================

    class Ensemble:

        def __init__(
            self,
            gradient_model,
            forest_model,
            logistic_model,
        ):

            self.gradient = (
                gradient_model
            )

            self.forest = (
                forest_model
            )

            self.logistic = (
                logistic_model
            )

        def predict_probability(
            self,
            X
        ):

            a = (
                self.gradient
                .predict_proba(
                    X
                )[:, 1]
            )

            b = (
                self.forest
                .predict_proba(
                    X
                )[:, 1]
            )

            c = (
                self.logistic
                .predict_proba(
                    X
                )[:, 1]
            )

            return (
                a * 0.45
                +
                b * 0.35
                +
                c * 0.20
            )

    ensemble = Ensemble(
        gradient,
        forest,
        logistic
    )

    baseline_probability = float(
        y_train.mean()
    )

    return ModelState(

        classifier=ensemble,

        feature_columns=feature_columns,

        accuracy=accuracy,
        positive_precision=positive_precision,

        samples=len(training),

        trained_at=time.time(),

        training_cost=trading_cost,

        baseline_probability=baseline_probability,
    )


# ============================================================
# PREDICTION
# ============================================================

def predict(
    state: ModelState,
    df: pd.DataFrame,
    forecast_bars: int = 3
):

    if (
        state is None
        or
        state.classifier is None
    ):

        return None

    features = build_features(
        df
    )

    row = (
        features[
            state.feature_columns
        ]
        .iloc[
            -1:
        ]
        .copy()
    )

    if row.isna().any(
        axis=None
    ):

        return None

    # ========================================================
    # ML PROBABILITY
    # ========================================================

    probability = float(
        state.classifier
        .predict_probability(
            row
        )[0]
    )

    probability = _clip(
        probability,
        0.01,
        0.99
    )

    # ========================================================
    # STRATEGY ENGINE
    # ========================================================

    strategy = strategy_signals(
        features
    )

    strategy_score = float(
        strategy[
            "strategy_score"
        ]
    )

    agreement = float(
        strategy[
            "strategy_agreement"
        ]
    )

    bearish_reversal = float(
        strategy[
            "bearish_reversal"
        ]
    )

    # ========================================================
    # ML DIRECTION
    # ========================================================

    ml_direction_score = (
        probability
        -
        0.5
    ) * 2

    # ========================================================
    # COMBINED DIRECTION
    # ========================================================

    combined_direction = (
        ml_direction_score
        * 0.65
        +
        strategy_score
        * 0.35
    )

    # ========================================================
    # CONFIDENCE
    # ========================================================

    ml_confidence = abs(
        ml_direction_score
    )

    strategy_confidence = (
        abs(
            strategy_score
        )
        *
        agreement
    )

    confidence = (
        ml_confidence
        * 0.65
        +
        strategy_confidence
        * 0.35
    )

    # Penalize disagreement.

    if agreement < 0.35:

        confidence *= 0.75

    confidence = _clip(
        confidence,
        0,
        1
    )

    # ========================================================
    # MARKET VALUES
    # ========================================================

    close = _last(
        features,
        "close",
        0
    )

    atr_pct = _last(
        features,
        "atr_pct",
        0
    )

    volatility = _last(
        features,
        "volatility_20",
        0
    )

    momentum_5 = _last(
        features,
        "return_5",
        0
    )

    momentum_10 = _last(
        features,
        "return_10",
        0
    )

    trend = _last(
        features,
        "ema21_vs_ema50",
        0
    )

    # ========================================================
    # EXPECTED MOVE
    # ========================================================
    #
    # This is a short-term movement estimate, not a promise.
    #
    # It combines:
    #
    #   ATR
    #   realized volatility
    #   directional strength
    #   momentum
    #   trend
    #
    # The output is capped so a single abnormal candle
    # cannot create a ridiculous expected return.
    # ========================================================

    movement_base = max(
        atr_pct,
        volatility * 1.20,
        0.001
    )

    directional_strength = max(
        0,
        abs(
            combined_direction
        )
    )

    momentum_confirmation = (
        abs(momentum_5)
        +
        abs(momentum_10)
    ) / 2

    trend_confirmation = abs(
        trend
    )

    expected_move = (
        movement_base
        *
        (
            0.75
            +
            directional_strength
            * 1.25
        )
    )

    expected_move *= (
        1
        +
        min(
            momentum_confirmation * 8,
            0.40
        )
    )

    expected_move *= (
        1
        +
        min(
            trend_confirmation * 8,
            0.30
        )
    )

    # ========================================================
    # REGIME ADJUSTMENT
    # ========================================================

    regime = strategy[
        "regime"
    ]

    if regime == "RANGE":

        expected_move *= 0.80

    elif regime == "HIGH_VOLATILITY":

        expected_move *= 1.10

    expected_move = _clip(
        expected_move,
        0.0001,
        0.25
    )

    # ========================================================
    # EXPECTED NET MOVE
    # ========================================================

    # The model knows the approximate cost used during training.

    expected_net_move = (
        expected_move
        -
        state.training_cost
    )

    # ========================================================
    # DIRECTION
    # ========================================================

    if combined_direction > 0.08:

        direction = "LONG"

    elif combined_direction < -0.08:

        direction = "SHORT"

    else:

        direction = "NEUTRAL"

    # ========================================================
    # EDGE
    # ========================================================

    # Higher probability + larger expected net movement
    # + stronger strategy agreement = stronger setup.

    edge = (
        max(
            probability - 0.50,
            0
        )
        *
        2
        *
        max(
            expected_net_move,
            0
        )
        *
        (
            0.50
            +
            0.50 * agreement
        )
    )

    # Penalize bearish reversal risk on LONG setups.

    if direction == "LONG":

        edge *= (
            1
            -
            bearish_reversal * 0.50
        )

    edge = max(
        edge,
        0
    )

    # ========================================================
    # RETURN
    # ========================================================

    return {

        "probability_up":
            probability,

        "expected_move":
            expected_move,

        "expected_net_move":
            expected_net_move,

        "direction":
            direction,

        "confidence":
            confidence,

        "strategy_score":
            strategy_score,

        "strategy_agreement":
            agreement,

        "bearish_reversal":
            bearish_reversal,

        "combined_direction":
            combined_direction,

        "edge":
            edge,

        "regime":
            regime,

        "strategies":
            strategy[
                "strategies"
            ],

        "price":
            close,

        "model_accuracy":
            state.accuracy,

        "positive_precision":
            getattr(state, "positive_precision", 0.0),

        "samples":
            state.samples,
    }
