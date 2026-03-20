"""ML trade classifier — predicts win/loss from technical features.

Replaces AI calls for trade decisions. Trains on backtest data using
XGBoost (gradient boosted trees) — fast, accurate on tabular data,
runs on CPU, zero cost.

Features extracted per setup:
  - RSI, MACD histogram, MA20 distance, MA50 distance
  - Volume ratio, Bollinger bandwidth, ATR
  - FVG count, OB count, structure break count
  - Funding rate, fear/greed index
  - Time features (hour, day of week)
  - Price position (premium/discount zone)

Pipeline:
  1. Generate training data from backtester
  2. Extract features from each trade's entry bar
  3. Train XGBoost classifier (win=1, loss=0)
  4. Save model for live prediction
  5. At runtime: extract features -> predict -> only take high-confidence trades

Usage::

    # Train
    model = TradeClassifier()
    model.train(training_trades, ohlcv_data)
    model.save("models/trade_classifier.json")

    # Predict
    model = TradeClassifier.load("models/trade_classifier.json")
    prob = model.predict(features)  # 0.0-1.0 win probability
"""
from __future__ import annotations

import json
import logging
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Feature names for consistency
FEATURE_NAMES = [
    "rsi_14",
    "macd_histogram",
    "ma20_distance_pct",
    "ma50_distance_pct",
    "volume_ratio_20",
    "bb_bandwidth",
    "atr_14_pct",
    "fvg_count",
    "ob_count",
    "structure_breaks",
    "price_in_discount",    # 1 if below equilibrium, 0 if above
    "candle_body_ratio",    # body / range (strength of last candle)
    "higher_high",          # 1 if recent higher-high structure
    "lower_low",            # 1 if recent lower-low structure
    "consecutive_green",    # count of consecutive green candles
    "consecutive_red",      # count of consecutive red candles
    "distance_to_support_pct",
    "distance_to_resistance_pct",
]


@dataclass
class TradeFeatures:
    """Feature vector for a single trade setup."""
    features: dict[str, float]
    label: int | None = None  # 1=win, 0=loss (None for live prediction)

    def to_array(self) -> list[float]:
        return [self.features.get(name, 0.0) for name in FEATURE_NAMES]


def extract_features(
    df: pd.DataFrame,
    bar_index: int,
    direction: str = "long",
) -> dict[str, float]:
    """Extract feature vector from OHLCV data at a specific bar.

    Uses the lookback window ending at bar_index.
    """
    import ta

    if bar_index < 50 or bar_index >= len(df):
        return {name: 0.0 for name in FEATURE_NAMES}

    window = df.iloc[:bar_index + 1].copy()
    close = window["close"]
    high = window["high"]
    low = window["low"]
    open_ = window["open"]
    volume = window["volume"]
    current = float(close.iloc[-1])

    features: dict[str, float] = {}

    # RSI
    rsi = ta.momentum.RSIIndicator(close, window=14).rsi()
    features["rsi_14"] = float(rsi.iloc[-1]) if not rsi.empty else 50.0

    # MACD histogram
    macd = ta.trend.MACD(close)
    hist = macd.macd_diff()
    features["macd_histogram"] = float(hist.iloc[-1]) if not hist.empty else 0.0

    # MA distances
    ma20 = float(close.rolling(20).mean().iloc[-1])
    ma50 = float(close.rolling(50).mean().iloc[-1]) if len(close) >= 50 else ma20
    features["ma20_distance_pct"] = (current - ma20) / ma20 * 100 if ma20 > 0 else 0
    features["ma50_distance_pct"] = (current - ma50) / ma50 * 100 if ma50 > 0 else 0

    # Volume ratio
    vol_avg = float(volume.rolling(20).mean().iloc[-1])
    features["volume_ratio_20"] = float(volume.iloc[-1]) / vol_avg if vol_avg > 0 else 1.0

    # Bollinger bandwidth
    bb = ta.volatility.BollingerBands(close, window=20)
    bw = bb.bollinger_wband()
    features["bb_bandwidth"] = float(bw.iloc[-1]) if not bw.empty else 0.0

    # ATR as % of price
    atr = ta.volatility.AverageTrueRange(high, low, close, window=14).average_true_range()
    features["atr_14_pct"] = float(atr.iloc[-1]) / current * 100 if current > 0 else 0

    # SMC-derived features
    from market_scanner.smc import find_swing_points, find_fvg, find_order_blocks, compute_zones

    swing_highs, swing_lows = find_swing_points(high, low, lookback=5)

    fvgs = find_fvg(high, low, close)
    unfilled = [f for f in fvgs if not f.filled]
    features["fvg_count"] = len(unfilled)

    obs = find_order_blocks(open_, high, low, close, swing_highs, swing_lows)
    features["ob_count"] = len(obs)

    # Structure breaks (count recent ones)
    from market_scanner.smc import detect_structure
    _, breaks = detect_structure(swing_highs, swing_lows, close)
    features["structure_breaks"] = len(breaks)

    # Premium/discount
    eq, premium, discount = compute_zones(swing_highs, swing_lows)
    features["price_in_discount"] = 1.0 if discount[0] <= current <= discount[1] else 0.0

    # Candle body ratio
    last_body = abs(float(close.iloc[-1]) - float(open_.iloc[-1]))
    last_range = float(high.iloc[-1]) - float(low.iloc[-1])
    features["candle_body_ratio"] = last_body / last_range if last_range > 0 else 0

    # Higher-high / lower-low
    if len(swing_highs) >= 2:
        features["higher_high"] = 1.0 if swing_highs[-1].price > swing_highs[-2].price else 0.0
    else:
        features["higher_high"] = 0.0

    if len(swing_lows) >= 2:
        features["lower_low"] = 1.0 if swing_lows[-1].price < swing_lows[-2].price else 0.0
    else:
        features["lower_low"] = 0.0

    # Consecutive candles
    green = 0
    for i in range(len(close) - 1, max(len(close) - 10, 0), -1):
        if float(close.iloc[i]) > float(open_.iloc[i]):
            green += 1
        else:
            break
    red = 0
    for i in range(len(close) - 1, max(len(close) - 10, 0), -1):
        if float(close.iloc[i]) < float(open_.iloc[i]):
            red += 1
        else:
            break
    features["consecutive_green"] = float(green)
    features["consecutive_red"] = float(red)

    # Distance to support/resistance
    recent_low = float(low.iloc[-20:].min())
    recent_high = float(high.iloc[-20:].max())
    features["distance_to_support_pct"] = (current - recent_low) / current * 100
    features["distance_to_resistance_pct"] = (recent_high - current) / current * 100

    return features


class TradeClassifier:
    """XGBoost-based trade win/loss classifier."""

    def __init__(self) -> None:
        self._model: Any = None
        self._trained = False

    def train(
        self,
        feature_rows: list[dict[str, float]],
        labels: list[int],
    ) -> dict[str, float]:
        """Train the classifier on extracted features.

        Returns training metrics.
        """
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.model_selection import cross_val_score

        X = np.array([[row.get(name, 0.0) for name in FEATURE_NAMES] for row in feature_rows])
        y = np.array(labels)

        # Use sklearn GradientBoosting (no extra install needed, similar to XGBoost)
        self._model = GradientBoostingClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.1,
            subsample=0.8,
            random_state=42,
        )

        # Cross-validation
        scores = cross_val_score(self._model, X, y, cv=5, scoring="accuracy")

        # Train on full data
        self._model.fit(X, y)
        self._trained = True

        # Feature importance
        importances = dict(zip(FEATURE_NAMES, self._model.feature_importances_))
        top_features = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:5]

        metrics = {
            "cv_accuracy": float(np.mean(scores)),
            "cv_std": float(np.std(scores)),
            "train_samples": len(y),
            "win_ratio": float(np.mean(y)),
            "top_features": {k: round(v, 4) for k, v in top_features},
        }

        logger.info(
            "TradeClassifier: trained on %d samples  cv_accuracy=%.1f%%  win_ratio=%.1f%%",
            len(y), metrics["cv_accuracy"] * 100, metrics["win_ratio"] * 100,
        )
        return metrics

    def predict(self, features: dict[str, float]) -> float:
        """Predict win probability for a trade setup. Returns 0.0-1.0."""
        if not self._trained or self._model is None:
            return 0.5

        X = np.array([[features.get(name, 0.0) for name in FEATURE_NAMES]])
        proba = self._model.predict_proba(X)[0]
        # Return probability of class 1 (win)
        return float(proba[1]) if len(proba) > 1 else float(proba[0])

    def save(self, path: str) -> None:
        """Save trained model to disk."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self._model, f)
        logger.info("TradeClassifier: saved to %s", path)

    @classmethod
    def load(cls, path: str) -> TradeClassifier:
        """Load a trained model from disk."""
        tc = cls()
        with open(path, "rb") as f:
            tc._model = pickle.load(f)
        tc._trained = True
        logger.info("TradeClassifier: loaded from %s", path)
        return tc
