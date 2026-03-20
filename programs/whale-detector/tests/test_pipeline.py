"""Integration tests for the whale-detector (WADE) pipeline.

Uses fakeredis for Redis operations — no running Redis server required.
Tests validate the complete pipeline from threshold flagging through
AnomalyEvent emission.
"""
from __future__ import annotations

import asyncio
import json
import statistics
from datetime import datetime, timezone

import pytest
import fakeredis.aioredis

from oracle_shared.contracts.signal import Signal, SignalCategory, SourceId
from oracle_shared.contracts.anomaly_event import AnomalyEvent
from oracle_shared.contracts.wallet_profile import WalletProfile, ReputationTier

from whale_detector.config import (
    LARGE_ORDER_THRESHOLD_USD_DEFAULT,
    PARAMS_STATE_KEY,
    MARKET_STATE_KEY,
    WALLET_FILLS_KEY_PREFIX,
)
from whale_detector.threshold_flagger import ThresholdFlagger
from whale_detector.wallet_registry import WalletRegistry
from whale_detector.anomaly_scorer import AnomalyScorer
from whale_detector.cascade_detector import CascadeDetector
from whale_detector.event_emitter import EventEmitter


@pytest.fixture
def redis_client():
    """Create a fresh fakeredis async client."""
    return fakeredis.aioredis.FakeRedis(decode_responses=True)


def _make_signal(size_usd: float = 10_000.0, wallet: str = "0xabc") -> Signal:
    """Helper to build a test Signal matching polygon_clob on-chain shape."""
    return Signal(
        source_id=SourceId.POLYGON_CLOB,
        timestamp=datetime.now(timezone.utc),
        category=SignalCategory.ON_CHAIN,
        raw_payload={
            "tx_hash": "0xdeadbeef",
            "wallet": wallet,
            "market_id": "12345",
            "outcome": "67890",
            "side": "buy",
            "price": 0.55,
            "size_usd": size_usd,
            "block_number": 100,
            "block_timestamp": datetime.now(timezone.utc).isoformat(),
        },
        market_ids=["12345"],
    )


# -- Step 2: ThresholdFlagger ------------------------------------------------


@pytest.mark.asyncio
async def test_threshold_flagger_flags_large_order(redis_client):
    flagger = ThresholdFlagger(redis_client)
    sig = _make_signal(size_usd=10_000.0)
    assert await flagger.evaluate(sig) is True


@pytest.mark.asyncio
async def test_threshold_flagger_passes_small_order(redis_client):
    flagger = ThresholdFlagger(redis_client)
    sig = _make_signal(size_usd=100.0)
    assert await flagger.evaluate(sig) is False


@pytest.mark.asyncio
async def test_threshold_flagger_reads_redis_override(redis_client):
    await redis_client.hset(PARAMS_STATE_KEY, "large_order_threshold_usd", "20000")
    flagger = ThresholdFlagger(redis_client)
    sig = _make_signal(size_usd=15_000.0)
    # 15k < 20k override => not flagged
    assert await flagger.evaluate(sig) is False


# -- Step 3: WalletRegistry --------------------------------------------------


@pytest.mark.asyncio
async def test_wallet_registry_creates_stub(redis_client):
    registry = WalletRegistry(redis_client)
    profile = await registry.get_or_create("0xnewwallet")
    assert profile.wallet_address == "0xnewwallet"
    assert profile.reputation_tier == ReputationTier.UNKNOWN


@pytest.mark.asyncio
async def test_wallet_registry_returns_existing(redis_client):
    registry = WalletRegistry(redis_client)
    profile1 = await registry.get_or_create("0xexisting")
    profile2 = await registry.get_or_create("0xexisting")
    assert profile1.wallet_address == profile2.wallet_address


# -- Step 4: AnomalyScorer ---------------------------------------------------


@pytest.mark.asyncio
async def test_anomaly_scorer_no_liquidity(redis_client):
    scorer = AnomalyScorer(redis_client)
    profile = WalletProfile(
        wallet_address="0xtest",
        first_seen_at=datetime.now(timezone.utc),
        last_active_at=datetime.now(timezone.utc),
    )
    score, reasons = await scorer.score(10_000.0, "mkt1", profile)
    # No liquidity data => size_factor = 0.5, no history => wallet_factor = 1.0,
    # no time data => time_factor = 0.5
    # (0.5 + 1.0 + 0.5) / 3 = 0.6667
    assert 0.6 <= score <= 0.7
    assert "large_order" in reasons


@pytest.mark.asyncio
async def test_anomaly_scorer_with_liquidity(redis_client):
    await redis_client.hset(
        MARKET_STATE_KEY, "mkt1",
        json.dumps({"liquidity_usd": 100_000.0}),
    )
    scorer = AnomalyScorer(redis_client)
    profile = WalletProfile(
        wallet_address="0xtest",
        typical_position_size_usd=5_000.0,
        first_seen_at=datetime.now(timezone.utc),
        last_active_at=datetime.now(timezone.utc),
    )
    # size_usd=10000, liquidity=100000 => size_factor = 0.1
    # 10000 / 5000 / 2.0 = 1.0 => wallet_factor = 1.0
    # no time => 0.5
    # (0.1 + 1.0 + 0.5) / 3 = 0.5333
    score, reasons = await scorer.score(10_000.0, "mkt1", profile)
    assert 0.5 <= score <= 0.55
    assert "size_deviation" in reasons  # 10k/5k >= 2x


# -- Step 5: CascadeDetector -------------------------------------------------


@pytest.mark.asyncio
async def test_cascade_not_triggered_single_wallet(redis_client):
    detector = CascadeDetector(redis_client)
    is_cascade, wallets = await detector.check("mkt1", "YES", "0xwallet1")
    assert is_cascade is False


@pytest.mark.asyncio
async def test_cascade_triggered_three_wallets(redis_client):
    import time
    detector = CascadeDetector(redis_client)
    now = time.time()
    await detector.check("mkt1", "YES", "0xw1", now)
    await detector.check("mkt1", "YES", "0xw2", now + 1)
    is_cascade, wallets = await detector.check("mkt1", "YES", "0xw3", now + 2)
    assert is_cascade is True
    assert len(wallets) >= 3


# -- Step 8: Wallet profile update -------------------------------------------


@pytest.mark.asyncio
async def test_wallet_profile_update_rolling_median(redis_client):
    registry = WalletRegistry(redis_client)
    profile = await registry.get_or_create("0xmedian")
    await registry.update_after_event(profile, 1000.0)
    await registry.update_after_event(profile, 2000.0)
    updated = await registry.update_after_event(profile, 3000.0)
    assert updated.typical_position_size_usd == statistics.median([1000.0, 2000.0, 3000.0])
    assert updated.total_trades_tracked == 3
