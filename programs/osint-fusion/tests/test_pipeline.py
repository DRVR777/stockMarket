"""Tests for osint-fusion (OSFE) pipeline components.

Uses FakeRedis and mock embeddings — no external APIs or servers required.
"""
import asyncio
import json
import sys
from datetime import datetime, timezone, timedelta
from typing import Any, Optional

from oracle_shared.contracts.signal import Signal, SignalCategory, SourceId
from oracle_shared.contracts.insight import Insight, DEFAULT_CREDIBILITY_WEIGHTS
from oracle_shared.contracts.market_state import MarketState

from osint_fusion.signal_subscriber import _TEXT_EXTRACTORS, _SKIP_SOURCES
from osint_fusion.credibility import CredibilityWeighter
from osint_fusion.insight_emitter import InsightEmitter


# ── FakeRedis ─────────────────────────────────────────────────────────────────

class FakeRedis:
    def __init__(self) -> None:
        self.published: list[tuple[str, str]] = []
        self._hashes: dict[str, dict[str, str]] = {}
        self._strings: dict[str, str] = {}

    async def publish(self, channel: str, message: str) -> None:
        self.published.append((channel, message))

    async def hget(self, key: str, field: str) -> Optional[str]:
        return self._hashes.get(key, {}).get(field)

    async def hset(self, key: str, field: str, value: str) -> None:
        self._hashes.setdefault(key, {})[field] = value

    async def hgetall(self, key: str) -> dict[str, str]:
        return self._hashes.get(key, {})

    async def get(self, key: str) -> Optional[str]:
        return self._strings.get(key)

    async def set(self, key: str, value: str) -> None:
        self._strings[key] = value


# ── Tests ─────────────────────────────────────────────────────────────────────

async def test_text_extraction() -> bool:
    """Verify text extraction rules for each source_id."""
    cases = [
        (SourceId.NEWSAPI.value, {"title": "Breaking News", "description": "Details here"}, "Breaking News. Details here"),
        (SourceId.WIKIPEDIA.value, {"page_title": "Election", "summary": "Recent changes"}, "Election: Recent changes"),
        (SourceId.REDDIT.value, {"title": "Hot post about markets"}, "Hot post about markets"),
        (SourceId.POLYMARKET_REST.value, {"question": "Will X happen?"}, "Will X happen?"),
        (SourceId.AI_OPINION.value, {"response_text": "Analysis suggests..."}, "Analysis suggests..."),
    ]
    for source_id, payload, expected in cases:
        extractor = _TEXT_EXTRACTORS.get(source_id)
        assert extractor is not None, f"Missing extractor for {source_id}"
        result = extractor(payload).strip()
        assert result == expected, f"{source_id}: got {result!r}, expected {expected!r}"

    # Verify skip sources
    for source_id in [SourceId.POLYGON_CLOB.value, SourceId.POLYMARKET_WS.value, SourceId.BIRDEYE.value]:
        assert source_id in _SKIP_SOURCES, f"{source_id} should be skipped"

    print("  text_extraction: PASS")
    return True


async def test_credibility_defaults() -> bool:
    """Default weights match the contract spec."""
    redis = FakeRedis()
    cw = CredibilityWeighter(redis)

    for category, expected in DEFAULT_CREDIBILITY_WEIGHTS.items():
        weight = await cw.get_weight(category)
        assert weight == expected, f"{category}: got {weight}, expected {expected}"

    print("  credibility_defaults: PASS")
    return True


async def test_credibility_redis_override() -> bool:
    """Redis overrides take precedence over defaults."""
    redis = FakeRedis()
    await redis.set("oracle:state:params:credibility_weights:news", "1.5")
    cw = CredibilityWeighter(redis)
    weight = await cw.get_weight("news")
    assert weight == 1.5, f"Expected 1.5, got {weight}"
    print("  credibility_override: PASS")
    return True


async def test_credibility_apply_delta() -> bool:
    """Delta application updates weight and clamps."""
    redis = FakeRedis()
    cw = CredibilityWeighter(redis)
    # news default = 1.0, delta +0.3 → 1.3
    new_weight = await cw.apply_delta("news", 0.3)
    assert new_weight == 1.3, f"Expected 1.3, got {new_weight}"
    # Apply large negative → should clamp to 0.1
    new_weight = await cw.apply_delta("news", -5.0)
    assert new_weight == 0.1, f"Expected 0.1, got {new_weight}"
    print("  credibility_delta: PASS")
    return True


async def test_insight_emitter() -> bool:
    """InsightEmitter publishes a valid Insight."""
    redis = FakeRedis()
    emitter = InsightEmitter(redis)

    insight = await emitter.emit(
        source_signal_id="sig-001",
        source_category="news",
        raw_text="Breaking: Major policy announcement expected",
        similarity_scores={"market_A": 0.82, "market_B": 0.71},
        credibility_weight=1.0,
    )

    assert len(redis.published) == 1
    channel, payload = redis.published[0]
    assert channel == Insight.CHANNEL

    parsed = Insight.model_validate_json(payload)
    assert parsed.source_signal_id == "sig-001"
    assert parsed.source_category == "news"
    assert set(parsed.associated_market_ids) == {"market_A", "market_B"}
    assert parsed.similarity_scores["market_A"] == 0.82
    assert parsed.source_credibility_weight == 1.0
    assert len(parsed.raw_text) > 0

    print(f"  insight_emitter: PASS  insight_id={parsed.insight_id}")
    return True


async def test_chroma_store_query() -> bool:
    """ChromaStore.query returns matches above threshold."""
    # This test requires chromadb installed
    try:
        from osint_fusion.chroma_store import ChromaStore
        import tempfile
        import os

        with tempfile.TemporaryDirectory() as tmpdir:
            # Override persist dir for test
            import osint_fusion.config as cfg
            old_dir = cfg.CHROMA_PERSIST_DIR
            cfg.CHROMA_PERSIST_DIR = tmpdir

            store = ChromaStore()

            # Add test markets with known embeddings
            # Use simple vectors to control similarity
            dim = 512
            base_vec = [1.0] + [0.0] * (dim - 1)
            similar_vec = [0.95] + [0.05] + [0.0] * (dim - 2)
            different_vec = [0.0] * (dim - 1) + [1.0]

            store.upsert_market("m1", "Will X happen?", base_vec)
            store.upsert_market("m2", "Similar question", similar_vec)
            store.upsert_market("m3", "Totally different", different_vec)

            matches = store.query(base_vec)
            # m1 should match perfectly (similarity=1.0), m2 should be high,
            # m3 should be low/absent
            assert "m1" in matches, f"Expected m1 in matches, got {matches}"
            assert matches["m1"] >= 0.99, f"m1 similarity should be ~1.0"

            cfg.CHROMA_PERSIST_DIR = old_dir

        print(f"  chroma_query: PASS  matches={matches}")
    except ImportError:
        print("  chroma_query: SKIP (chromadb not installed)")
    return True


async def run_all() -> bool:
    tests = [
        test_text_extraction,
        test_credibility_defaults,
        test_credibility_redis_override,
        test_credibility_apply_delta,
        test_insight_emitter,
        test_chroma_store_query,
    ]

    print("osint-fusion (OSFE) unit tests\n")
    passed = 0
    failed = 0
    for test in tests:
        try:
            ok = await test()
            if ok:
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"  {test.__name__}: FAIL — {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print(f"\n{'='*40}")
    print(f"Results: {passed} passed, {failed} failed")
    return failed == 0


if __name__ == "__main__":
    ok = asyncio.run(run_all())
    sys.exit(0 if ok else 1)
