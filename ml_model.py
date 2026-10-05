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

    samples: int = 0

    trained_at: float = 0.0

    training_cost: float = 0.0

    baseline_probability: float = 0.5


# ============================================================
# BASIC HELPERS
# ============================================================

def _safe_div(
    a,
    b
):

    b = b.replace(
        0,
        np.nan
    )

    return a / b


def _clip(
    value,
    low,
    high
):

    return float(
        np.clip(
            value,
            low,
            high
        )
    )


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

    return result.fillna(
        50
    )


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

    previous_close = (
        close.shift(1)
    )

    tr = pd.concat(
        [
            high - low,

            (high - previous_close).abs(),

            (low - previous_close).abs(),
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

def macd(
    close
):

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

    line = (
        fast - slow
    )

    signal = (
        line
        .ewm(
            span=9,
            adjust=False
        )
        .mean()
    )

    histogram = (
        line - signal
    )

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

    # --------------------------------------------------------
    # RETURNS / MOMENTUM
    # --------------------------------------------------------

    data["return_1"] = (
        close.pct_change(1)
    )

    data["return_3"] = (
        close.pct_change(3)
    )

    data["return_5"] = (
        close.pct_change(5)
    )

    data["return_10"] = (
        close.pct_change(10)
    )

    data["return_20"] = (
        close.pct_change(20)
    )

    # --------------------------------------------------------
    # MOVING AVERAGES
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # TREND DISTANCES
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # RSI
    # --------------------------------------------------------

    data["rsi_14"] = rsi(
        close,
        14
    )

    data["rsi_7"] = rsi(
        close,
        7
    )

    # --------------------------------------------------------
    # MACD
    # --------------------------------------------------------

    (
        data["macd"],
        data["macd_signal"],
        data["macd_hist"],
    ) = macd(
        close
    )

    data["macd_hist_change"] = (
        data["macd_hist"]
        .diff()
    )

    # --------------------------------------------------------
    # ATR / VOLATILITY
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # BOLLINGER BANDS
    # --------------------------------------------------------

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

    data["bb_position"] = _safe_div(
        close - bb_lower,
        bb_upper - bb_lower
    )

    # --------------------------------------------------------
    # VOLUME
    # --------------------------------------------------------

    data["volume_sma20"] = (
        volume
        .rolling(20)
        .mean()
    )

    data["volume_ratio"] = _safe_div(
        volume,
        data["volume_sma20"]
    )

    data["volume_change"] = (
        volume
        .pct_change()
        .replace(
            [np.inf, -np.inf],
            np.nan
        )
    )

    # --------------------------------------------------------
    # VWAP
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # MARKET STRUCTURE
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # CANDLE STRUCTURE
    # --------------------------------------------------------

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
            axis=1
        ).max(
            axis=1
        )
    )

    data["lower_wick"] = (
        pd.concat(
            [
                open_price,
                close
            ],
            axis=1
        ).min(
            axis=1
        )
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

    # --------------------------------------------------------
    # CLEAN NUMBERS
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # MOMENTUM
    # --------------------------------------------------------

    momentum = 0.0

    momentum += np.tanh(
        float(
            x.get(
                "return_5",
                0
            )
            or 0
        )
        * 40
    ) * 0.35

    momentum += np.tanh(
        float(
            x.get(
                "return_10",
                0
            )
            or 0
        )
        * 25
    ) * 0.35

    momentum += np.tanh(
        float(
            x.get(
                "return_20",
                0
            )
            or 0
        )
        * 15
    ) * 0.30

    # --------------------------------------------------------
    # TREND
    # --------------------------------------------------------

    trend = 0.0

    trend += np.tanh(
        float(
            x.get(
                "price_vs_sma20",
                0
            )
            or 0
        )
        * 20
    ) * 0.25

    trend += np.tanh(
        float(
            x.get(
                "price_vs_sma50",
                0
            )
            or 0
        )
        * 15
    ) * 0.25

    trend += np.tanh(
        float(
            x.get(
                "ema9_vs_ema21",
                0
            )
            or 0
        )
        * 30
    ) * 0.25

    trend += np.tanh(
        float(
            x.get(
                "ema21_vs_ema50",
                0
            )
            or 0
        )
        * 25
    ) * 0.25

    # --------------------------------------------------------
    # MEAN REVERSION
    # --------------------------------------------------------

    rsi_value = float(
        x.get(
            "rsi_14",
            50
        )
        or 50
    )

    bb_position = float(
        x.get(
            "bb_position",
            0.5
        )
        or 0.5
    )

    mean_reversion = 0.0

    # Oversold = positive reversal opportunity.

    if rsi_value < 30:
        mean_reversion += 0.80

    elif rsi_value < 40:
        mean_reversion += 0.35

    elif rsi_value > 70:
        mean_reversion -= 0.80

    elif rsi_value > 60:
        mean_reversion -= 0.35

    if bb_position < 0.10:
        mean_reversion += 0.45

    elif bb_position < 0.25:
        mean_reversion += 0.20

    elif bb_position > 0.90:
        mean_reversion -= 0.45

    elif bb_position > 0.75:
        mean_reversion -= 0.20

    mean_reversion = _clip(
        mean_reversion,
        -1,
        1
    )

    # --------------------------------------------------------
    # BREAKOUT
    # --------------------------------------------------------

    breakout = 0.0

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

    breakdown20 = float(
        x.get(
            "breakdown_20",
            0
        )
        or 0
    )

    breakdown50 = float(
        x.get(
            "breakdown_50",
            0
        )
        or 0
    )

    breakout += np.tanh(
        b20 * 100
    ) * 0.30

    breakout += np.tanh(
        b50 * 100
    ) * 0.25

    breakout -= np.tanh(
        abs(breakdown20) * 100
    ) * 0.20

    breakout -= np.tanh(
        abs(breakdown50) * 100
    ) * 0.15

    # Volume confirmation.

    volume_ratio = float(
        x.get(
            "volume_ratio",
            1
        )
        or 1
    )

    if volume_ratio > 1.5:

        breakout *= 1.25

    elif volume_ratio < 0.7:

        breakout *= 0.65

    breakout = _clip(
        breakout,
        -1,
        1
    )

    # --------------------------------------------------------
    # VWAP
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # MACD
    # --------------------------------------------------------

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

    macd_score = (
        np.tanh(
            macd_hist
            /
            max(
                abs(
                    float(
                        features["close"].iloc[-1]
                    )
                ),
                1e-9
            )
            * 1000
        )
        * 0.70
        +
        np.tanh(
            macd_change
            /
            max(
                abs(
                    float(
                        features["close"].iloc[-1]
                    )
                ),
                1e-9
            )
            * 1000
        )
        * 0.30
    )

    macd_score = _clip(
        macd_score,
        -1,
        1
    )

    # --------------------------------------------------------
    # VOLUME
    # --------------------------------------------------------

    volume_score = np.tanh(
        (
            volume_ratio
            - 1.0
        )
        *
        1.5
    )

    # --------------------------------------------------------
    # MARKET STRUCTURE
    # --------------------------------------------------------

    structure = float(
        x.get(
            "structure_score",
            0
        )
        or 0
    )

    structure_score = _clip(
        structure,
        -1,
        1
    )

    # --------------------------------------------------------
    # VOLATILITY
    # --------------------------------------------------------

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

    # Very low volatility can mean there is not enough
    # movement to overcome fees/slippage.

    volatility_score = np.tanh(
        (
            volatility * 100
        )
    )

    if atr_pct < 0.001:

        volatility_score *= 0.50

    # --------------------------------------------------------
    # STRATEGY DICTIONARY
    # --------------------------------------------------------

    strategies = {

        "momentum":
            _clip(
                momentum,
                -1,
                1
            ),

        "trend":
            _clip(
                trend,
                -1,
                1
            ),

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
            _clip(
                volume_score,
                -1,
                1
            ),

        "volatility":
            _clip(
                volatility_score,
                -1,
                1
            ),

        "market_structure":
            structure_score,
    }

    # --------------------------------------------------------
    # STRATEGY WEIGHTS
    # --------------------------------------------------------

    weights = {

        "momentum": 0.14,

        "trend": 0.16,

        "mean_reversion": 0.08,

        "breakout": 0.14,

        "vwap": 0.10,

        "macd": 0.12,

        "volume": 0.08,

        "volatility": 0.06,

        "market_structure": 0.12,
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

    # --------------------------------------------------------
    # STRATEGY AGREEMENT
    # --------------------------------------------------------

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

    total = len(
        values
    )

    if total:

        agreement = (
            max(
                bullish,
                bearish
            )
            /
            total
        )

    else:

        agreement = 0.0

    # Agreement is directional rather than simply
    # "how many strategies fired."

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

    agreement = _clip(
        agreement,
        0,
        1
    )

    # --------------------------------------------------------
    # REGIME
    # --------------------------------------------------------

    trend_value = float(
        x.get(
            "ema21_vs_ema50",
            0
        )
        or 0
    )

    vol_value = float(
        x.get(
            "volatility_20",
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

    elif vol_value > 0.02:

        regime = "HIGH_VOLATILITY"

    else:

        regime = "RANGE"

    return {
        "strategy_score":
            strategy_score,

        "strategy_agreement":
            agreement,

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

    # --------------------------------------------------------
    # TARGET
    # --------------------------------------------------------

    future_return = (
        features["close"]
        .shift(
            -forecast_bars
        )
        /
        features["close"]
        - 1
    )

    # The model must predict a move that has a chance
    # to overcome estimated trading costs.

    target = (
        future_return
        >
        trading_cost
    ).astype(int)

    # --------------------------------------------------------
    # FEATURE LIST
    # --------------------------------------------------------

    feature_columns = [

        "return_1",
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

        "macd",
        "macd_signal",
        "macd_hist",
        "macd_hist_change",

        "atr_pct",

        "volatility_10",
        "volatility_20",

        "bb_width",
        "bb_position",

        "volume_ratio",
        "volume_change",

        "price_vs_vwap",

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

    # --------------------------------------------------------
    # CLEAN TRAINING DATA
    # --------------------------------------------------------

    training = (
        features[
            feature_columns
        ]
        .copy()
    )

    valid = (
        training.notna()
        .all(
            axis=1
        )
        &
        future_return.notna()
    )

    training = (
        training.loc[
            valid
        ]
    )

    labels = (
        target.loc[
            valid
        ]
    )

    # --------------------------------------------------------
    # SANITY CHECK
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # CHRONOLOGICAL TRAIN / VALIDATION SPLIT
    # --------------------------------------------------------

    split = int(
        len(training)
        *
        0.80
    )

    if split < 100:

        split = max(
            1,
            len(training) - 30
        )

    X_train = (
        training.iloc[
            :split
        ]
    )

    y_train = (
        labels.iloc[
            :split
        ]
    )

    X_test = (
        training.iloc[
            split:
        ]
    )

    y_test = (
        labels.iloc[
            split:
        ]
    )

    # --------------------------------------------------------
    # CLASS BALANCE CHECK
    # --------------------------------------------------------

    if (
        y_train.nunique()
        < 2
    ):

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
    # HIST GRADIENT BOOSTING
    # ========================================================

    gradient = (
        HistGradientBoostingClassifier(

            max_iter=180,

            learning_rate=0.045,

            max_leaf_nodes=15,

            max_depth=5,

            min_samples_leaf=12,

            l2_regularization=0.15,

            random_state=42,
        )
    )

    # ========================================================
    # MODEL 2
    # RANDOM FOREST
    # ========================================================

    forest = (
        RandomForestClassifier(

            n_estimators=250,

            max_depth=8,

            min_samples_leaf=8,

            max_features="sqrt",

            class_weight="balanced_subsample",

            random_state=42,

            n_jobs=-1,
        )
    )

    # ========================================================
    # MODEL 3
    # LOGISTIC REGRESSION
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

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # ENSEMBLE VALIDATION
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # WRAP ENSEMBLE
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # BASELINE
    # --------------------------------------------------------

    baseline_probability = float(
        y_train.mean()
    )

    # --------------------------------------------------------
    # RETURN MODEL STATE
    # --------------------------------------------------------

    return ModelState(

        classifier=
            ensemble,

        feature_columns=
            feature_columns,

        accuracy=
            accuracy,

        samples=
            len(training),

        trained_at=
            time.time(),

        training_cost=
            trading_cost,

        baseline_probability=
            baseline_probability,
    )


# ============================================================
# PREDICTION
# ============================================================

def predict(
    state: ModelState,
    df: pd.DataFrame,
    forecast_bars: int = 3
):

    # --------------------------------------------------------
    # NO MODEL
    # --------------------------------------------------------

    if (
        state is None
        or
        state.classifier is None
    ):

        return None

    # --------------------------------------------------------
    # FEATURES
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # ML PROBABILITY
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # STRATEGIES
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # ML + STRATEGY COMBINATION
    # --------------------------------------------------------

    ml_direction_score = (
        probability
        -
        0.5
    ) * 2

    combined_direction = (
        ml_direction_score
        *
        0.65
        +
        strategy_score
        *
        0.35
    )

    # --------------------------------------------------------
    # CONFIDENCE
    # --------------------------------------------------------

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
        *
        0.65
        +
        strategy_confidence
        *
        0.35
    )

    confidence = _clip(
        confidence,
        0,
        1
    )

    # --------------------------------------------------------
    # EXPECTED MOVE
    # --------------------------------------------------------

    close = float(
        features[
            "close"
        ].iloc[
            -1
        ]
    )

    atr_pct = float(
        features[
            "atr_pct"
        ].iloc[
            -1
        ]
        or 0
    )

    volatility = float(
        features[
            "volatility_20"
        ].iloc[
            -1
        ]
        or 0
    )

    momentum = float(
        features[
            "return_5"
        ].iloc[
            -1
        ]
        or 0
    )

    trend = float(
        features[
            "ema21_vs_ema50"
        ].iloc[
            -1
        ]
        or 0
    )

    # --------------------------------------------------------
    # ESTIMATE EXPECTED MOVEMENT
    # --------------------------------------------------------

    movement_base = max(
        atr_pct,
        volatility * 1.25,
        0.001
    )

    directional_strength = max(
        0.0,
        abs(
            combined_direction
        )
    )

    expected_move = (
        movement_base
        *
        (
            0.70
            +
            directional_strength
            *
            1.30
        )
    )

    # Momentum/trend confirmation.

    confirmation = (
        abs(momentum)
        +
        abs(trend)
    ) / 2.0

    expected_move *= (
        1.0
        +
        min(
            confirmation * 8,
            0.50
        )
    )

    expected_move = _clip(
        expected_move,
        0.0001,
        0.25
    )

    # --------------------------------------------------------
    # DIRECTION
    # --------------------------------------------------------

    if combined_direction > 0.08:

        direction = "LONG"

    elif combined_direction < -0.08:

        direction = "SHORT"

    else:

        direction = "NEUTRAL"

    # --------------------------------------------------------
    # RETURN
    # --------------------------------------------------------

    return {

        "probability_up":
            probability,

        "expected_move":
            expected_move,

        "direction":
            direction,

        "confidence":
            confidence,

        "strategy_score":
            strategy_score,

        "strategy_agreement":
            agreement,

        "regime":
            strategy[
                "regime"
            ],

        "strategies":
            strategy[
                "strategies"
            ],

        "combined_direction":
            combined_direction,

        "price":
            close,

        "model_accuracy":
            state.accuracy,

        "samples":
            state.samples,
    }
