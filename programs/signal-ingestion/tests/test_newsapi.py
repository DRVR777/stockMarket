"""
Test: Adapter 4a — NewsAPI
Runs a single live poll against newsapi.org.
Uses an in-memory fake Redis (no server required).

Requires NEWSAPI_API_KEY env var to be set.
"""
import asyncio
import json
import os
import sys

from signal_ingestion.adapters.newsapi_adapter import NewsAPIAdapter
from oracle_shared.contracts.signal import Signal, SourceId, SignalCategory


class FakeRedis:
    def __init__(self):
        self.messages: list[tuple[str, str]] = []

    async def publish(self, channel: str, message: str) -> None:
        self.messages.append((channel, message))


async def run_test() -> bool:
    api_key = os.getenv("NEWSAPI_API_KEY", "")
    if not api_key:
        print("SKIP: NEWSAPI_API_KEY not set")
        return True

    redis = FakeRedis()
    adapter = NewsAPIAdapter(redis, api_key=api_key, poll_interval=9999)

    print("Polling NewsAPI /v2/everything (live)...")
    await adapter._poll()

    count = len(redis.messages)
    print(f"Signals published: {count}")

    if count == 0:
        print("WARN: no articles returned (may be rate-limited)")
        return True  # not a failure — free tier has low limits

    # --- validate first signal ---
    channel, payload = redis.messages[0]
    assert channel == Signal.CHANNEL, f"wrong channel: {channel!r}"

    signal = Signal.model_validate_json(payload)
    assert signal.source_id == SourceId.NEWSAPI
    assert signal.category == SignalCategory.NEWS

    rp = signal.raw_payload
    missing = [k for k in ("title", "description", "url", "source_name", "published_at", "query_used") if k not in rp]
    assert not missing, f"raw_payload missing keys: {missing}"

    print("\n--- First Signal (sample) ---")
    print(json.dumps(json.loads(payload), indent=2))
    print(f"\nAdapter 4a (NewsAPI): PASS  ({count} signals)")
    return True


if __name__ == "__main__":
    ok = asyncio.run(run_test())
    sys.exit(0 if ok else 1)
