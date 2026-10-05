from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

FEATURES = [
    "ret_1", "ret_3", "ret_6",
    "ema_fast_gap", "ema_slow_gap",
    "rsi", "atr_pct", "range_pct",
    "volume_z", "trend_15", "trend_30", "volatility",
]

def make_features(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    close = x["close"].astype(float)
    high = x["high"].astype(float)
    low = x["low"].astype(float)
    volume = x["volume"].astype(float)

    x["ret_1"] = close.pct_change(1)
    x["ret_3"] = close.pct_change(3)
    x["ret_6"] = close.pct_change(6)

    ema_fast = close.ewm(span=8, adjust=False).mean()
    ema_slow = close.ewm(span=21, adjust=False).mean()
    x["ema_fast_gap"] = close / ema_fast - 1
    x["ema_slow_gap"] = close / ema_slow - 1

    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    x["rsi"] = (100 - (100 / (1 + rs))).fillna(50)

    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low),
         (high - prev_close).abs(),
         (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    x["atr_pct"] = tr.rolling(14).mean() / close
    x["range_pct"] = (high - low) / close

    vol_mean = volume.rolling(30).mean()
    vol_std = volume.rolling(30).std().replace(0, np.nan)
    x["volume_z"] = (volume - vol_mean) / vol_std

    x["trend_15"] = close.pct_change(15)
    x["trend_30"] = close.pct_change(30)
    x["volatility"] = close.pct_change().rolling(30).std()

    return x.replace([np.inf, -np.inf], np.nan)

def build_training_set(df: pd.DataFrame, horizon: int, cost: float):
    x = make_features(df)
    future_return = df["close"].shift(-horizon) / df["close"] - 1

    # 1 means the future move exceeded estimated trading costs.
    y = (future_return > cost).astype(int)

    data = x[FEATURES].copy()
    data["target"] = y
    data = data.dropna()

    if len(data) < 250:
        return None, None

    return data[FEATURES], data["target"]

@dataclass
class ModelState:
    model: Pipeline | None = None
    accuracy: float = 0.0
    trained_at: float = 0.0
    samples: int = 0

def train_model(df: pd.DataFrame, horizon: int, cost: float) -> ModelState:
    X, y = build_training_set(df, horizon, cost)
    state = ModelState()

    if X is None:
        return state

    split = int(len(X) * 0.80)
    if split < 150 or len(X) - split < 30:
        return state

    # Time-ordered split: no random shuffle, reducing future leakage.
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    model = Pipeline([
        ("scale", StandardScaler()),
        ("clf", HistGradientBoostingClassifier(
            max_iter=180,
            learning_rate=0.055,
            max_leaf_nodes=15,
            l2_regularization=0.5,
            random_state=42,
        )),
    ])

    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    state.model = model
    state.accuracy = float(accuracy_score(y_test, pred))
    state.trained_at = time.time()
    state.samples = len(X_train)
    return state

def predict(state: ModelState, df: pd.DataFrame, horizon: int):
    if state.model is None:
        return None

    x = make_features(df)
    row = x[FEATURES].tail(1)

    if row.isna().any().any():
        return None

    p_up = float(state.model.predict_proba(row)[0][1])
    vol = float(x["volatility"].tail(1).iloc[0])

    if not math.isfinite(vol):
        return None

    expected_move = max(vol * math.sqrt(max(1, horizon)), 0.0)

    return {
        "probability_up": p_up,
        "expected_move": expected_move,
    }
