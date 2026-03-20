"""
Test: Adapter 4c — Reddit
Runs a single live poll against the Reddit OAuth API.
Uses an in-memory fake Redis (no server required).

Requires REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET env vars.
"""
import asyncio
import json
import os
import sys

from signal_ingestion.adapters.reddit_adapter import RedditAdapter
from oracle_shared.contracts.signal import Signal, SourceId, SignalCategory


class FakeRedis:
    def __init__(self):
        self.messages: list[tuple[str, str]] = []

    async def publish(self, channel: str, message: str) -> None:
        self.messages.append((channel, message))


async def run_test() -> bool:
    client_id = os.getenv("REDDIT_CLIENT_ID", "")
    client_secret = os.getenv("REDDIT_CLIENT_SECRET", "")
    if not client_id or not client_secret:
        print("SKIP: REDDIT_CLIENT_ID / REDDIT_CLIENT_SECRET not set")
        return True

    redis = FakeRedis()
    adapter = RedditAdapter(
        redis,
        poll_interval=9999,
        subreddits=["Polymarket"],
        client_id=client_id,
        client_secret=client_secret,
    )

    print("Polling Reddit r/Polymarket (live)...")
    await adapter._poll()

    count = len(redis.messages)
    print(f"Signals published: {count}")

    if count == 0:
        print("WARN: no posts returned")
        return True

    # --- validate first signal ---
    channel, payload = redis.messages[0]
    assert channel == Signal.CHANNEL, f"wrong channel: {channel!r}"

    signal = Signal.model_validate_json(payload)
    assert signal.source_id == SourceId.REDDIT
    assert signal.category == SignalCategory.SOCIAL

    rp = signal.raw_payload
    missing = [k for k in ("subreddit", "post_id", "title", "score", "num_comments", "created_utc", "url") if k not in rp]
    assert not missing, f"raw_payload missing keys: {missing}"

    print("\n--- First Signal (sample) ---")
    print(json.dumps(json.loads(payload), indent=2))
    print(f"\nAdapter 4c (Reddit): PASS  ({count} signals)")
    return True


if __name__ == "__main__":
    ok = asyncio.run(run_test())
    sys.exit(0 if ok else 1)
