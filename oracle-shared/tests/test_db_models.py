"""Test: oracle_shared.db — model conversion (Pydantic ↔ SQLAlchemy).

No database required. Validates that from_contract() produces correct rows.
"""
import asyncio
import sys
from datetime import datetime, timezone

from oracle_shared.contracts.signal import Signal, SignalCategory, SourceId
from oracle_shared.contracts.anomaly_event import AnomalyEvent
from oracle_shared.contracts.insight import Insight
from oracle_shared.contracts.trade_thesis import (
    TradeThesis, ThesisDecision, Hypothesis, EvidenceWeight,
    ContextAssembly, HistoricalAnalogue,
)
from oracle_shared.contracts.wallet_profile import WalletProfile, ReputationTier
from oracle_shared.contracts.trade_execution import (
    TradeExecution, MarketType, ExecutionSource, ExecutionStatus,
)
from oracle_shared.contracts.post_mortem import PostMortem, SignalSummary
from oracle_shared.contracts.operator_alert import (
    OperatorAlert, AlertType, AlertSeverity,
)

from oracle_shared.db.models import (
    SignalRow, AnomalyEventRow, InsightRow, TradeThesisRow,
    WalletProfileRow, TradeExecutionRow, PostMortemRow, OperatorAlertRow,
)


now = datetime.now(timezone.utc)


def test_signal_row() -> bool:
    sig = Signal(
        source_id=SourceId.NEWSAPI,
        timestamp=now,
        category=SignalCategory.NEWS,
        raw_payload={"title": "Test", "description": "Desc"},
    )
    row = SignalRow.from_contract(sig)
    assert row.signal_id == sig.signal_id
    assert row.source_id == "newsapi"
    assert row.category == "news"
    assert row.raw_payload["title"] == "Test"
    print("  SignalRow: PASS")
    return True


def test_anomaly_event_row() -> bool:
    ae = AnomalyEvent(
        timestamp=now,
        wallet_address="0xabc",
        market_id="m1",
        outcome="YES",
        notional_usd=10000.0,
        anomaly_score=0.85,
        trigger_reasons=["large_order", "tier_1_wallet"],
        copy_trade_eligible=True,
        source_signal_id="sig-001",
    )
    row = AnomalyEventRow.from_contract(ae)
    assert row.event_id == ae.event_id
    assert row.anomaly_score == 0.85
    assert "large_order" in row.trigger_reasons
    print("  AnomalyEventRow: PASS")
    return True


def test_insight_row() -> bool:
    ins = Insight(
        timestamp=now,
        source_signal_id="sig-002",
        source_category="news",
        associated_market_ids=["m1", "m2"],
        similarity_scores={"m1": 0.82, "m2": 0.71},
        semantic_summary="Breaking news...",
        source_credibility_weight=1.0,
        raw_text="Full article text here",
    )
    row = InsightRow.from_contract(ins)
    assert row.insight_id == ins.insight_id
    assert row.associated_market_ids == ["m1", "m2"]
    assert row.similarity_scores["m1"] == 0.82
    print("  InsightRow: PASS")
    return True


def test_trade_thesis_row() -> bool:
    thesis = TradeThesis(
        created_at=now,
        market_id="m1",
        market_question="Will X happen?",
        direction="YES",
        re_probability_estimate=0.72,
        market_implied_probability=0.55,
        probability_delta=0.17,
        confidence_score=0.68,
        decision=ThesisDecision.EXECUTE,
        recommended_position_usd=500.0,
        hypotheses=[
            Hypothesis(side="YES", argument="Strong evidence for", evidence=["e1"]),
            Hypothesis(side="NO", argument="Counter argument", evidence=["e2"]),
        ],
        evidence_weights=[
            EvidenceWeight(hypothesis_side="YES", score=0.72, reasoning="..."),
            EvidenceWeight(hypothesis_side="NO", score=0.28, reasoning="..."),
        ],
        context_assembly=ContextAssembly(
            market_state={"market_id": "m1"},
            anomaly_events=[],
            historical_analogues=[],
            assembled_at=now,
        ),
    )
    row = TradeThesisRow.from_contract(thesis)
    assert row.thesis_id == thesis.thesis_id
    assert row.decision == "execute"
    assert row.probability_delta == 0.17
    assert len(row.hypotheses) == 2
    assert row.hypotheses[0]["side"] == "YES"
    print("  TradeThesisRow: PASS")
    return True


def test_wallet_profile_row() -> bool:
    wp = WalletProfile(
        wallet_address="0xshark",
        reputation_tier=ReputationTier.SHARK,
        win_rate=0.72,
        typical_position_size_usd=8000.0,
        first_seen_at=now,
        last_active_at=now,
        total_trades_tracked=25,
    )
    row = WalletProfileRow.from_contract(wp)
    assert row.wallet_address == "0xshark"
    assert row.reputation_tier == "Shark"
    assert row.win_rate == 0.72
    print("  WalletProfileRow: PASS")
    return True


def test_trade_execution_row() -> bool:
    te = TradeExecution(
        market_id="m1",
        market_type=MarketType.POLYMARKET,
        direction="buy",
        outcome="YES",
        entry_price=0.55,
        size_usd=1000.0,
        executed_at=now,
        execution_source=ExecutionSource.RE_THESIS,
        thesis_id="thesis-001",
    )
    row = TradeExecutionRow.from_contract(te)
    assert row.execution_id == te.execution_id
    assert row.market_type == "polymarket"
    assert row.execution_source == "re_thesis"
    assert row.status == "open"
    print("  TradeExecutionRow: PASS")
    return True


def test_post_mortem_row() -> bool:
    pm = PostMortem(
        generated_at=now,
        market_id="m1",
        market_question="Will X happen?",
        market_resolved_as="YES",
        thesis_was_correct=True,
        realized_pnl_usd=150.0,
        signals_present=[
            SignalSummary(signal_id="s1", category="news", summary="Article about X"),
        ],
        what_the_thesis_said="Predicted YES at 72%",
        what_happened="Market resolved YES",
        what_would_have_changed_outcome="Nothing — prediction was correct",
        source_weight_updates={"newsapi": 0.05, "reddit": -0.02},
        vault_path="/data/vault/markets/m1/pm-001.md",
    )
    row = PostMortemRow.from_contract(pm)
    assert row.postmortem_id == pm.postmortem_id
    assert row.thesis_was_correct is True
    assert row.source_weight_updates["newsapi"] == 0.05
    assert len(row.signals_present) == 1
    print("  PostMortemRow: PASS")
    return True


def test_operator_alert_row() -> bool:
    alert = OperatorAlert(
        created_at=now,
        alert_type=AlertType.ANOMALY,
        severity=AlertSeverity.ACTION_REQUIRED,
        title="Whale alert: $50,000 on market",
        body="Details...",
        action_required=True,
        action_options=["approve_copy_trade", "dismiss"],
        linked_event_id="evt-001",
    )
    row = OperatorAlertRow.from_contract(alert)
    assert row.alert_id == alert.alert_id
    assert row.alert_type == "anomaly"
    assert row.severity == "action_required"
    assert row.acknowledged is False
    print("  OperatorAlertRow: PASS")
    return True


def main() -> bool:
    tests = [
        test_signal_row,
        test_anomaly_event_row,
        test_insight_row,
        test_trade_thesis_row,
        test_wallet_profile_row,
        test_trade_execution_row,
        test_post_mortem_row,
        test_operator_alert_row,
    ]

    print("oracle_shared.db model conversion tests\n")
    passed = 0
    failed = 0
    for test in tests:
        try:
            if test():
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
    ok = main()
    sys.exit(0 if ok else 1)
