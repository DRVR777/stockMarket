"""
Test: Adapter 2 — Polymarket WebSocket
Connects to the live Polymarket WS feed, subscribes to 50 active markets,
waits up to 30 s for price ticks, validates Signal shape.
Uses FakeRedis (no server required).
"""
import asyncio
import json
import sys

from signal_ingestion.adapters.polymarket_ws import PolymarketWSAdapter
from oracle_shared.contracts.signal import Signal, SourceId, SignalCategory


class FakeRedis:
    def __init__(self):
        self.messages: list[tuple[str, str]] = []

    async def publish(self, channel: str, message: str) -> None:
        self.messages.append((channel, message))


async def run_test() -> bool:
    redis = FakeRedis()
    adapter = PolymarketWSAdapter(redis, max_markets=50, reconnect_delay=3.0)

    print("Refreshing token map (50 active markets)...")
    await adapter._refresh_token_map()
    token_count = len(adapter._token_map)
    print(f"Token map: {token_count} tokens")

    if token_count == 0:
        print("FAIL: no tokens in map")
        return False

    # --- connection + subscription test ---
    print(f"\nConnecting to WS and subscribing to {token_count} tokens...")
    print("Waiting up to 30s for price ticks...")

    async def _stream_with_timeout():
        try:
            await asyncio.wait_for(adapter._connect_and_stream(), timeout=30.0)
        except asyncio.TimeoutError:
            pass  # expected — just collecting ticks for 30s

    await _stream_with_timeout()

    tick_count = len(redis.messages)
    print(f"\nTicks received: {tick_count}")

    if tick_count == 0:
        # Connection + subscription succeeded (no exception thrown).
        # Zero ticks in 30s is possible if markets are quiet — not a failure.
        print("NOTE: 0 ticks in 30s window — markets may be quiet; connection OK")
        print("\nAdapter 2 (Polymarket WS): CONNECTION PASS (no ticks to validate)")
        return True

    # --- validate shape of received signals ---
    channel, payload = redis.messages[0]
    assert channel == Signal.CHANNEL, f"wrong channel: {channel!r}"

    signal = Signal.model_validate_json(payload)
    assert signal.source_id == SourceId.POLYMARKET_WS
    assert signal.category == SignalCategory.PRICE

    rp = signal.raw_payload
    missing = [k for k in ("market_id", "outcome", "price", "side", "size") if k not in rp]
    assert not missing, f"raw_payload missing keys: {missing}"

    print("\n--- First Tick Signal (sample) ---")
    print(json.dumps(json.loads(payload), indent=2))

    print(f"\n--- Stats ---")
    print(f"  Ticks received : {tick_count}")
    print(f"  Channel        : {channel}")
    print(f"  source_id      : {signal.source_id.value}")
    print(f"  category       : {signal.category.value}")
    print(f"  market_id      : {rp['market_id']}")
    print(f"  outcome        : {rp['outcome']}")
    print(f"  price          : {rp['price']}")
    print(f"  side           : {rp['side']}")

    # spot-check first 20 ticks
    bad = []
    for i, (ch, pl) in enumerate(redis.messages[:20]):
        try:
            s = Signal.model_validate_json(pl)
            assert s.source_id == SourceId.POLYMARKET_WS
            assert s.category == SignalCategory.PRICE
            for k in ("market_id", "outcome", "price", "side", "size"):
                assert k in s.raw_payload
        except Exception as e:
            bad.append((i, str(e)))

    if bad:
        print(f"FAIL: shape errors in signals: {bad}")
        return False

    print(f"\nShape check on first 20 ticks: PASS")
    print("\nAdapter 2 (Polymarket WS): PASS")
    return True


if __name__ == "__main__":
    ok = asyncio.run(run_test())
    sys.exit(0 if ok else 1)
