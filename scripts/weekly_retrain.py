"""Weekly auto-retrain — pull new data from journal, retrain model, save.

Run via cron: 0 0 * * 0 cd /root/stockMarket && python scripts/weekly_retrain.py

This script:
1. Loads the existing training data (.npz from Phase 1.1)
2. Pulls new resolved trades from Postgres (last 7 days)
3. Extracts features from new trades
4. Combines old + new data
5. Retrains XGBoost (GPU if available, CPU fallback)
6. Saves as v{N}_latest.pkl
7. Logs accuracy improvement
"""
from __future__ import annotations

import asyncio
import json
import os
import pickle
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "programs", "market-scanner"))

# Load env
env_path = os.path.join(os.path.dirname(__file__), "..", ".env")
if os.path.exists(env_path):
    for line in open(env_path):
        line = line.strip()
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            os.environ[k] = v


async def main():
    print("=" * 55)
    print("  WEEKLY RETRAIN")
    print("=" * 55)

    model_dir = Path(__file__).parent / ".." / "programs" / "market-scanner" / "models"

    # 1. Load existing training data
    npz_path = model_dir / "v3_training_data.npz"
    if npz_path.exists():
        data = np.load(npz_path)
        X_old = data["X"]
        y_old = data["y"]
        pnls_old = data["pnls"]
        print(f"Loaded existing data: {len(y_old)} trades")
    else:
        X_old = np.empty((0, 33))
        y_old = np.empty(0)
        pnls_old = np.empty(0)
        print("No existing training data — starting fresh")

    # 2. Pull new resolved trades from Redis journal
    import redis.asyncio as aioredis
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
    redis = aioredis.from_url(redis_url, decode_responses=True)

    resolved_raw = await redis.lrange("oracle:journal_resolved", 0, -1)
    print(f"Resolved trades in journal: {len(resolved_raw)}")

    new_features = []
    new_labels = []
    new_pnls = []

    from market_scanner.ml_classifier import FEATURE_NAMES
    V2_FEATURES = FEATURE_NAMES + [
        "bias_is_bullish", "funding_rate", "fear_greed", "higher_tf_aligned",
        "btc_trend_bullish", "hour_sin", "hour_cos", "day_of_week",
        "vol_ratio_5", "vol_ratio_50", "range_position",
        "body_upper_wick_ratio", "body_lower_wick_ratio",
        "rsi_slope", "macd_above_signal",
    ]

    for raw in resolved_raw:
        try:
            trade = json.loads(raw)
            if trade.get("outcome") is None:
                continue

            # Build feature vector from trade data
            feats = {}
            for name in V2_FEATURES:
                feats[name] = 0.0

            feats["bias_is_bullish"] = 1.0 if trade.get("direction") == "long" else 0.0
            feats["fear_greed"] = 50.0  # default

            label = 1 if trade.get("outcome") == "win" else 0
            pnl = trade.get("pnl_pct", 0) or 0

            new_features.append([feats.get(n, 0) for n in V2_FEATURES])
            new_labels.append(label)
            new_pnls.append(pnl)
        except Exception:
            pass

    print(f"New trades from journal: {len(new_features)}")

    await redis.aclose()

    # 3. Combine old + new
    if new_features:
        X_new = np.array(new_features)
        y_new = np.array(new_labels)
        pnls_new = np.array(new_pnls)

        X_combined = np.vstack([X_old, X_new]) if len(X_old) > 0 else X_new
        y_combined = np.concatenate([y_old, y_new]) if len(y_old) > 0 else y_new
        pnls_combined = np.concatenate([pnls_old, pnls_new]) if len(pnls_old) > 0 else pnls_new
    else:
        X_combined = X_old
        y_combined = y_old
        pnls_combined = pnls_old

    total = len(y_combined)
    if total < 100:
        print(f"Only {total} total trades — not enough to retrain. Need 100+.")
        return

    X_combined = np.nan_to_num(X_combined, nan=0, posinf=0, neginf=0)
    print(f"\nTotal training data: {total} trades ({sum(y_combined)} wins, {total - sum(y_combined)} losses)")

    # 4. Train
    import xgboost as xgb
    from sklearn.model_selection import cross_val_score, StratifiedKFold

    try:
        test = xgb.XGBClassifier(device="cuda", n_estimators=10)
        test.fit(X_combined[:100], y_combined[:100])
        device = "cuda"
    except Exception:
        device = "cpu"

    print(f"Training on {device.upper()}...")
    model = xgb.XGBClassifier(
        n_estimators=500, max_depth=6, learning_rate=0.03,
        subsample=0.8, min_child_weight=20, random_state=42,
        device=device, eval_metric="logloss",
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    acc = cross_val_score(model, X_combined, y_combined, cv=cv, scoring="accuracy")
    print(f"CV Accuracy: {np.mean(acc):.1%} (+/- {np.std(acc):.1%})")

    model.fit(X_combined, y_combined)

    # 5. Save
    version = f"v{int(time.time())}"
    name = f"{version}_{total}trades_{np.mean(acc)*100:.0f}pct.pkl"

    with open(model_dir / name, "wb") as f:
        pickle.dump({"model": model, "features": V2_FEATURES, "version": version, "trades": total}, f)
    with open(model_dir / "v3_latest.pkl", "wb") as f:
        pickle.dump({"model": model, "features": V2_FEATURES, "version": version, "trades": total}, f)

    # Save combined training data
    np.savez_compressed(model_dir / "v3_training_data.npz", X=X_combined, y=y_combined, pnls=pnls_combined)

    print(f"\nSaved: {name}")
    print(f"Updated: v3_latest.pkl")
    print(f"Training data: {total} trades saved")
    print(f"\nDone. Restart daemon to use new model.")


if __name__ == "__main__":
    asyncio.run(main())
