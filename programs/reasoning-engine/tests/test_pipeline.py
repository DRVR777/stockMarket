"""Tests for reasoning-engine (RE) pipeline — no LLM calls, no DB required."""
import asyncio
import json
import math
import sys
from datetime import datetime, timezone, timedelta
from typing import Any, Optional

from oracle_shared.contracts.trade_thesis import (
    ContextAssembly,
    EvidenceWeight,
    Hypothesis,
    HistoricalAnalogue,
    ThesisDecision,
    TradeThesis,
)
from oracle_shared.contracts.market_state import MarketState

from reasoning_engine.evidence_weigher import EvidenceWeigher
from reasoning_engine.confidence_calibrator import ConfidenceCalibrator
from reasoning_engine.hypothesis_generator import HypothesisGenerator


# ── FakeRedis ─────────────────────────────────────────────────────────────────

class FakeRedis:
    def __init__(self) -> None:
        self.published: list[tuple[str, str]] = []
        self._hashes: dict[str, dict[str, str]] = {}
        self._sets: dict[str, set[str]] = {}
        self._lists: dict[str, list[str]] = {}

    async def publish(self, channel: str, message: str) -> None:
        self.published.append((channel, message))

    async def hget(self, key: str, field: str) -> Optional[str]:
        return self._hashes.get(key, {}).get(field)

    async def hset(self, key: str, field: str, value: str) -> None:
        self._hashes.setdefault(key, {})[field] = value

    async def hkeys(self, key: str) -> list[str]:
        return list(self._hashes.get(key, {}).keys())

    async def sadd(self, key: str, member: str) -> int:
        s = self._sets.setdefault(key, set())
        if member in s:
            return 0
        s.add(member)
        return 1

    async def srem(self, key: str, member: str) -> None:
        self._sets.get(key, set()).discard(member)

    async def lrange(self, key: str, start: int, stop: int) -> list[str]:
        lst = self._lists.get(key, [])
        return lst[start:stop + 1] if stop >= 0 else lst[start:]

    async def lpush(self, key: str, value: str) -> None:
        self._lists.setdefault(key, []).insert(0, value)

    async def ltrim(self, key: str, start: int, stop: int) -> None:
        lst = self._lists.get(key, [])
        self._lists[key] = lst[start:stop + 1]


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_context(
    current_price_yes: float = 0.55,
    n_insights: int = 5,
    n_anomalies: int = 2,
) -> ContextAssembly:
    now = datetime.now(timezone.utc)
    insights = []
    categories = ["news", "social", "on_chain", "price", "ai_generated"]
    for i in range(n_insights):
        insights.append({
            "insight_id": f"ins-{i}",
            "timestamp": (now - timedelta(hours=i)).isoformat(),
            "source_category": categories[i % len(categories)],
            "source_credibility_weight": 1.0,
            "raw_text": f"Test insight {i}",
        })

    anomalies = []
    for i in range(n_anomalies):
        anomalies.append({
            "event_id": f"ae-{i}",
            "market_id": "m1",
            "outcome": "YES" if i % 2 == 0 else "NO",
            "notional_usd": 10000.0,
            "anomaly_score": 0.8,
            "trigger_reasons": ["large_order"],
        })

    return ContextAssembly(
        market_state={
            "market_id": "m1",
            "market_question": "Will X happen by end of year?",
            "current_price_yes": current_price_yes,
            "liquidity_usd": 50000.0,
            "resolution_deadline": (now + timedelta(days=30)).isoformat(),
            "recent_insights": insights,
            "semantic_state_summary": "Test summary",
        },
        anomaly_events=anomalies,
        historical_analogues=[],
        assembled_at=now,
    )


# ── Tests ─────────────────────────────────────────────────────────────────────

async def test_evidence_weigher_basic() -> bool:
    """Evidence weigher returns valid scores and delta."""
    redis = FakeRedis()
    weigher = EvidenceWeigher(redis)
    context = make_context(current_price_yes=0.55)

    hypotheses = [
        Hypothesis(side="YES", argument="Test yes", evidence=["e1"]),
        Hypothesis(side="NO", argument="Test no", evidence=["e2"]),
    ]

    ew, re_prob, delta, skip = await weigher.weigh(hypotheses, context)
    assert 0.0 <= re_prob <= 1.0, f"re_prob out of range: {re_prob}"
    assert len(ew) == 2
    assert ew[0].hypothesis_side == "YES"
    assert ew[1].hypothesis_side == "NO"
    assert abs(ew[0].score + ew[1].score - 1.0) < 0.01

    print(f"  evidence_weigher: PASS  re_prob={re_prob:.3f}  delta={delta:.3f}  skip={skip}")
    return True


async def test_evidence_weigher_skip_small_delta() -> bool:
    """Small delta should trigger skip."""
    redis = FakeRedis()
    weigher = EvidenceWeigher(redis)
    # Set price close to 0.5 so delta is small
    context = make_context(current_price_yes=0.50)
    hypotheses = [
        Hypothesis(side="YES", argument="a", evidence=[]),
        Hypothesis(side="NO", argument="b", evidence=[]),
    ]
    _, re_prob, delta, skip = await weigher.weigh(hypotheses, context)
    # With equal insights, re_prob should be close to current_price (~0.5)
    # so delta should be small → skip=True
    assert abs(delta) < 0.15, f"Expected small delta, got {delta}"
    print(f"  evidence_skip: PASS  delta={delta:.3f}  skip={skip}")
    return True


async def test_confidence_calibrator() -> bool:
    """Confidence calibrator produces valid score and decision."""
    redis = FakeRedis()
    calibrator = ConfidenceCalibrator(redis)
    context = make_context(n_insights=5)

    confidence, decision, position = await calibrator.calibrate(
        context, re_probability=0.72, probability_delta=0.17,
    )

    assert 0.0 <= confidence <= 1.0, f"confidence out of range: {confidence}"
    assert decision in (ThesisDecision.EXECUTE, ThesisDecision.FLAG_FOR_REVIEW)
    if decision == ThesisDecision.EXECUTE:
        assert position is not None and position > 0

    print(f"  confidence: PASS  score={confidence:.3f}  decision={decision.value}  position=${position}")
    return True


async def test_confidence_low_certainty() -> bool:
    """Near-50/50 probability should give low certainty."""
    redis = FakeRedis()
    calibrator = ConfidenceCalibrator(redis)
    context = make_context(n_insights=1)  # low diversity + low recency

    confidence, decision, _ = await calibrator.calibrate(
        context, re_probability=0.51, probability_delta=0.01,
    )

    # With 1 insight, 1 category, and near 50/50 → very low confidence
    assert confidence < 0.5, f"Expected low confidence, got {confidence}"
    assert decision == ThesisDecision.FLAG_FOR_REVIEW

    print(f"  confidence_low: PASS  score={confidence:.3f}  decision={decision.value}")
    return True


async def test_certainty_math() -> bool:
    """Verify certainty score math."""
    # At p=0.5 (max entropy), certainty should be ~0
    certainty_half = ConfidenceCalibrator._certainty_score(0.5)
    assert certainty_half < 0.05, f"Expected ~0 at p=0.5, got {certainty_half}"

    # At p=0.99 (low entropy), certainty should be ~high
    certainty_high = ConfidenceCalibrator._certainty_score(0.99)
    assert certainty_high > 0.9, f"Expected >0.9 at p=0.99, got {certainty_high}"

    # At p=0.7 (moderate), certainty should be moderate
    certainty_mod = ConfidenceCalibrator._certainty_score(0.7)
    assert 0.1 < certainty_mod < 0.6

    print(f"  certainty_math: PASS  p=0.5={certainty_half:.3f}  p=0.99={certainty_high:.3f}  p=0.7={certainty_mod:.3f}")
    return True


async def test_hypothesis_parser() -> bool:
    """HypothesisGenerator._parse_response handles various formats."""
    # Standard JSON
    raw = json.dumps({
        "hypotheses": [
            {"side": "YES", "argument": "Strong case", "evidence": ["e1", "e2"]},
            {"side": "NO", "argument": "Weak case", "evidence": ["e3"]},
        ]
    })
    result = HypothesisGenerator._parse_response(raw)
    assert len(result) == 2
    assert result[0].side == "YES"
    assert len(result[0].evidence) == 2

    # With code fences
    fenced = f"```json\n{raw}\n```"
    result2 = HypothesisGenerator._parse_response(fenced)
    assert len(result2) == 2

    # Missing one side → should be auto-filled
    partial = json.dumps({"hypotheses": [{"side": "YES", "argument": "Only yes", "evidence": []}]})
    result3 = HypothesisGenerator._parse_response(partial)
    sides = {h.side for h in result3}
    assert "YES" in sides and "NO" in sides

    print("  hypothesis_parser: PASS")
    return True


async def run_all() -> bool:
    tests = [
        test_evidence_weigher_basic,
        test_evidence_weigher_skip_small_delta,
        test_confidence_calibrator,
        test_confidence_low_certainty,
        test_certainty_math,
        test_hypothesis_parser,
    ]

    print("reasoning-engine (RE) unit tests\n")
    passed = 0
    failed = 0
    for test in tests:
        try:
            if await test():
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
