"""
Test: Adapter 1 — Polymarket REST
Runs a single live poll against the Polymarket CLOB API.
Uses an in-memory fake Redis (no server required).
"""
import asyncio
import json
import sys

from signal_ingestion.adapters.polymarket_rest import PolymarketRESTAdapter
from oracle_shared.contracts.signal import Signal, SourceId, SignalCategory


class FakeRedis:
    def __init__(self):
        self.messages: list[tuple[str, str]] = []

    async def publish(self, channel: str, message: str) -> None:
        self.messages.append((channel, message))


async def run_test() -> bool:
    redis = FakeRedis()
    adapter = PolymarketRESTAdapter(redis, poll_interval=9999)

    print("Polling Polymarket CLOB /markets (live)...")
    await adapter._poll()

    count = len(redis.messages)
    print(f"Signals published: {count}")

    if count == 0:
        print("FAIL: no signals published")
        return False

    # --- validate first signal ---
    channel, payload = redis.messages[0]
    assert channel == Signal.CHANNEL, f"wrong channel: {channel!r}"

    signal = Signal.model_validate_json(payload)

    assert signal.source_id == SourceId.POLYMARKET_REST, f"bad source_id: {signal.source_id}"
    assert signal.category == SignalCategory.PRICE, f"bad category: {signal.category}"
    assert isinstance(signal.signal_id, str) and signal.signal_id, "missing signal_id"
    assert signal.timestamp is not None, "missing timestamp"

    rp = signal.raw_payload
    missing = [k for k in ("market_id", "question", "outcome_prices", "volume_usd", "liquidity_usd", "end_date") if k not in rp]
    assert not missing, f"raw_payload missing keys: {missing}"

    assert isinstance(rp["outcome_prices"], dict), "outcome_prices must be a dict"
    assert signal.market_ids and signal.market_ids[0] == rp["market_id"], "market_ids mismatch"

    print("\n--- First Signal (sample) ---")
    print(json.dumps(json.loads(payload), indent=2))

    print(f"\n--- Stats ---")
    print(f"  Total signals : {count}")
    print(f"  Channel       : {channel}")
    print(f"  source_id     : {signal.source_id.value}")
    print(f"  category      : {signal.category.value}")
    print(f"  market_ids    : {signal.market_ids}")

    # spot-check a few more signals for shape consistency
    bad = []
    for i, (ch, pl) in enumerate(redis.messages[:50]):
        try:
            s = Signal.model_validate_json(pl)
            assert s.source_id == SourceId.POLYMARKET_REST
            assert s.category == SignalCategory.PRICE
            assert "market_id" in s.raw_payload
        except Exception as e:
            bad.append((i, str(e)))

    if bad:
        print(f"FAIL: shape errors in {len(bad)} signals: {bad[:3]}")
        return False

    print(f"\nShape check on first 50 signals: PASS")
    print("\nAdapter 1 (Polymarket REST): PASS")
    return True


if __name__ == "__main__":
    ok = asyncio.run(run_test())
    sys.exit(0 if ok else 1)
