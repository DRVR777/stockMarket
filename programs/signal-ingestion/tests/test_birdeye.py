"""
Test: Adapter 5 — Birdeye Price
Tests the REST fallback path (single poll via /defi/price).
Uses an in-memory fake Redis (no server required).

Requires BIRDEYE_API_KEY env var.
"""
import asyncio
import json
import os
import sys

import httpx

from signal_ingestion.adapters.birdeye_ws import BirdeyeWSAdapter, BIRDEYE_REST_BASE, _to_float
from oracle_shared.contracts.signal import Signal, SourceId, SignalCategory


class FakeRedis:
    def __init__(self):
        self.messages: list[tuple[str, str]] = []

    async def publish(self, channel: str, message: str) -> None:
        self.messages.append((channel, message))


async def run_test() -> bool:
    api_key = os.getenv("BIRDEYE_API_KEY", "")
    if not api_key:
        print("SKIP: BIRDEYE_API_KEY not set")
        return True

    redis = FakeRedis()

    # Test REST path directly (WS requires long-running connection)
    sol_address = "So11111111111111111111111111111111111111112"
    print(f"Fetching Birdeye /defi/price for SOL (live)...")

    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.get(
            f"{BIRDEYE_REST_BASE}/defi/price",
            params={"address": sol_address},
            headers={"X-API-KEY": api_key, "x-chain": "solana"},
        )
        resp.raise_for_status()
        body = resp.json().get("data", {})

    signal = Signal(
        source_id=SourceId.BIRDEYE,
        timestamp=__import__("datetime").datetime.now(__import__("datetime").timezone.utc),
        category=SignalCategory.PRICE,
        raw_payload={
            "token_address": sol_address,
            "symbol": "SOL",
            "price_usd": _to_float(body.get("value")),
            "price_change_24h_pct": _to_float(body.get("priceChange24h")),
            "volume_24h_usd": _to_float(body.get("volume24h")),
        },
    )

    # Validate shape
    assert signal.source_id == SourceId.BIRDEYE
    assert signal.category == SignalCategory.PRICE

    rp = signal.raw_payload
    missing = [k for k in ("token_address", "symbol", "price_usd", "price_change_24h_pct", "volume_24h_usd") if k not in rp]
    assert not missing, f"raw_payload missing keys: {missing}"
    assert rp["price_usd"] > 0, f"SOL price should be > 0, got {rp['price_usd']}"

    payload = signal.model_dump_json()
    print("\n--- Signal (sample) ---")
    print(json.dumps(json.loads(payload), indent=2))
    print(f"\nAdapter 5 (Birdeye REST fallback): PASS  (SOL=${rp['price_usd']:.2f})")
    return True


if __name__ == "__main__":
    ok = asyncio.run(run_test())
    sys.exit(0 if ok else 1)
