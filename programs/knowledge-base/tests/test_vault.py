"""Tests for KBPM — vault operations, writers, post-mortem pipeline.

No external APIs or servers required.
"""
import asyncio
import json
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from oracle_shared.contracts.trade_thesis import (
    ContextAssembly,
    EvidenceWeight,
    Hypothesis,
    ThesisDecision,
    TradeThesis,
)
from oracle_shared.contracts.anomaly_event import AnomalyEvent
from oracle_shared.contracts.trade_execution import (
    ExecutionSource,
    ExecutionStatus,
    ExitReason,
    MarketType,
    TradeExecution,
)

import knowledge_base.config as cfg
from knowledge_base.vault import init_vault, vault_path, write_md, read_md, append_section
from knowledge_base.market_writer import MarketWriter


# ── Helpers ───────────────────────────────────────────────────────────────────

now = datetime.now(timezone.utc)


def make_thesis() -> TradeThesis:
    return TradeThesis(
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
            Hypothesis(side="YES", argument="Strong case", evidence=["e1"]),
            Hypothesis(side="NO", argument="Counter", evidence=["e2"]),
        ],
        evidence_weights=[
            EvidenceWeight(hypothesis_side="YES", score=0.72, reasoning="..."),
            EvidenceWeight(hypothesis_side="NO", score=0.28, reasoning="..."),
        ],
        context_assembly=ContextAssembly(
            market_state={"market_id": "m1", "resolution_deadline": "2026-12-31"},
            anomaly_events=[],
            historical_analogues=[],
            assembled_at=now,
        ),
    )


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_vault_init() -> bool:
    """Vault initializer creates all subdirectories."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg.VAULT_DIR = tmpdir
        root = init_vault()
        for subdir in ["markets", "wallets", "signals", "theses", "osint"]:
            assert (root / subdir).is_dir(), f"Missing subdir: {subdir}"

    print("  vault_init: PASS")
    return True


def test_write_md() -> bool:
    """write_md creates a file with YAML front-matter."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg.VAULT_DIR = tmpdir
        init_vault()
        path = vault_path("theses", "test-thesis.md")
        result = write_md(
            path,
            {"thesis_id": "t1", "outcome": None},
            "# Test\nBody text",
        )
        assert result is True
        content = path.read_text()
        assert "thesis_id: t1" in content
        assert "outcome: null" in content
        assert "# Test" in content

    print("  write_md: PASS")
    return True


def test_write_md_no_overwrite() -> bool:
    """write_md with overwrite=False skips existing files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg.VAULT_DIR = tmpdir
        init_vault()
        path = vault_path("theses", "exists.md")
        write_md(path, {"id": "1"}, "first")
        result = write_md(path, {"id": "2"}, "second", overwrite=False)
        assert result is False
        assert "first" in path.read_text()

    print("  write_md_no_overwrite: PASS")
    return True


def test_append_section() -> bool:
    """append_section adds content under a section header."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg.VAULT_DIR = tmpdir
        init_vault()
        path = vault_path("markets", "test.md")
        write_md(path, {"id": "m1"}, "# Test\n\n## Theses\n\n## Resolution\n")
        append_section(path, "Theses", "- thesis_001\n")
        content = path.read_text()
        assert "- thesis_001" in content
        # Should be before ## Resolution
        theses_idx = content.index("## Theses")
        item_idx = content.index("- thesis_001")
        res_idx = content.index("## Resolution")
        assert theses_idx < item_idx < res_idx

    print("  append_section: PASS")
    return True


def test_market_writer() -> bool:
    """MarketWriter creates and appends to market files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg.VAULT_DIR = tmpdir
        init_vault()
        writer = MarketWriter()
        thesis = make_thesis()

        # First write creates the file
        writer.write_from_thesis(thesis)
        path = vault_path("markets", "m1.md")
        assert path.exists()
        content = path.read_text()
        assert "Will X happen?" in content
        assert thesis.thesis_id[:8] in content

        # Second write appends
        thesis2 = make_thesis()
        writer.write_from_thesis(thesis2)
        content2 = path.read_text()
        assert content2.count("[YES]") == 2

    print("  market_writer: PASS")
    return True


def test_market_writer_skip() -> bool:
    """MarketWriter ignores skip decisions."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg.VAULT_DIR = tmpdir
        init_vault()
        writer = MarketWriter()
        thesis = make_thesis()
        thesis.decision = ThesisDecision.SKIP
        writer.write_from_thesis(thesis)
        path = vault_path("markets", "m1.md")
        assert not path.exists()

    print("  market_writer_skip: PASS")
    return True


def run_all() -> bool:
    tests = [
        test_vault_init,
        test_write_md,
        test_write_md_no_overwrite,
        test_append_section,
        test_market_writer,
        test_market_writer_skip,
    ]

    print("knowledge-base (KBPM) unit tests\n")
    passed = 0
    failed = 0
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"  {test.__name__}: FAIL -- {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print(f"\n{'='*40}")
    print(f"Results: {passed} passed, {failed} failed")
    return failed == 0


if __name__ == "__main__":
    ok = run_all()
    sys.exit(0 if ok else 1)
