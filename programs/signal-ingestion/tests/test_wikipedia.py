"""
Test: Adapter 4b — Wikipedia Recent Changes
Runs a single live poll against the MediaWiki API.
Uses an in-memory fake Redis (no server required).
No API key required.
"""
import asyncio
import json
import sys

from signal_ingestion.adapters.wikipedia_adapter import WikipediaAdapter
from oracle_shared.contracts.signal import Signal, SourceId, SignalCategory


class FakeRedis:
    def __init__(self):
        self.messages: list[tuple[str, str]] = []

    async def publish(self, channel: str, message: str) -> None:
        self.messages.append((channel, message))


async def run_test() -> bool:
    redis = FakeRedis()
    # Use a short watch list for testing
    adapter = WikipediaAdapter(
        redis,
        poll_interval=9999,
        watch_pages=["Prediction market", "Polymarket"],
    )

    print("Polling Wikipedia Recent Changes API (live)...")
    await adapter._poll()

    count = len(redis.messages)
    print(f"Signals published: {count}")

    if count == 0:
        print("WARN: no recent edits found (pages may not have been edited recently)")
        return True  # not a failure

    # --- validate first signal ---
    channel, payload = redis.messages[0]
    assert channel == Signal.CHANNEL, f"wrong channel: {channel!r}"

    signal = Signal.model_validate_json(payload)
    assert signal.source_id == SourceId.WIKIPEDIA
    assert signal.category == SignalCategory.NEWS

    rp = signal.raw_payload
    missing = [k for k in ("page_title", "summary", "edit_timestamp", "edit_comment", "diff_url") if k not in rp]
    assert not missing, f"raw_payload missing keys: {missing}"

    print("\n--- First Signal (sample) ---")
    print(json.dumps(json.loads(payload), indent=2))
    print(f"\nAdapter 4b (Wikipedia): PASS  ({count} signals)")
    return True


if __name__ == "__main__":
    ok = asyncio.run(run_test())
    sys.exit(0 if ok else 1)
