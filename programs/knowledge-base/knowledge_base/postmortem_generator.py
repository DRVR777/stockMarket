"""Task 7 — Post-mortem pipeline.

Triggered by TradeExecution close. Steps:
  (a) Read vault/markets/{market_id}.md
  (b) Read linked vault/theses/{thesis_id}.md
  (c) Call Claude Sonnet for post-mortem analysis
  (d) Parse into PostMortem object
  (e) Append to market doc
  (f) Update thesis outcome in vault + ChromaDB
  (g) Publish PostMortem to oracle:post_mortem
  (h) Persist to Postgres
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Optional

import chromadb

from oracle_shared.contracts.post_mortem import PostMortem, SignalSummary
from oracle_shared.contracts.trade_execution import TradeExecution
from oracle_shared.contracts.trade_thesis import ThesisOutcome
from oracle_shared.db import get_session
from oracle_shared.db.repository import PostMortemRepo, ThesisRepo
from oracle_shared.providers import LLMProvider, get_llm

from knowledge_base.config import (
    CHROMA_PERSIST_DIR,
    CHROMA_THESES_COLLECTION,
    VAULT_DIR,
)
from knowledge_base.market_writer import MarketWriter
from knowledge_base.vault import read_md, vault_path, write_md

logger = logging.getLogger(__name__)

PM_SYSTEM_PROMPT = """You are a trading post-mortem analyst. Given a market's full context \
(signals, thesis, execution result), produce a structured analysis.

Return ONLY valid JSON:
{
  "market_resolved_as": "<YES/NO/VOID or asset symbol>",
  "thesis_was_correct": <true/false/null>,
  "what_the_thesis_said": "<1-2 sentence summary>",
  "what_happened": "<1-2 sentence summary of actual resolution>",
  "what_would_have_changed_outcome": "<1-2 sentence counterfactual>",
  "source_weight_updates": {"<source_id>": <float delta>, ...},
  "signals_assessment": [{"signal_id": "...", "category": "...", "summary": "...", "was_useful": true/false}]
}"""


class PostMortemGenerator:
    """Generate post-mortems for closed trades."""

    def __init__(
        self,
        redis_client: Any,
        market_writer: MarketWriter,
        llm: LLMProvider | None = None,
    ) -> None:
        self._redis = redis_client
        self._market_writer = market_writer
        self._llm = llm or get_llm()
        self._chroma_client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
        self._collection = self._chroma_client.get_or_create_collection(
            name=CHROMA_THESES_COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )

    async def generate(
        self,
        market_id: str,
        thesis_id: Optional[str],
        execution: TradeExecution,
    ) -> Optional[PostMortem]:
        """Run the full post-mortem pipeline."""
        # (a) Read market doc
        market_doc = read_md(vault_path("markets", f"{market_id}.md"))

        # (b) Read thesis doc
        thesis_doc = ""
        if thesis_id:
            thesis_doc = read_md(vault_path("theses", f"{thesis_id}.md"))

        if not market_doc and not thesis_doc:
            logger.warning("PostMortemGenerator: no vault docs for market %s", market_id)
            return None

        # (c) Call LLM for analysis
        pm_data = await self._call_llm(market_id, market_doc, thesis_doc, execution)
        if pm_data is None:
            return None

        # (d) Build PostMortem object
        signals_assessment = pm_data.get("signals_assessment", [])
        signals_present = [
            SignalSummary(
                signal_id=s.get("signal_id", ""),
                category=s.get("category", ""),
                summary=s.get("summary", ""),
                was_useful=s.get("was_useful"),
            )
            for s in signals_assessment
        ]

        pm_path = str(vault_path("markets", f"{market_id}.md"))
        pm = PostMortem(
            generated_at=datetime.now(timezone.utc),
            market_id=market_id,
            market_question=market_id,  # ideally from market doc
            thesis_id=thesis_id,
            execution_id=execution.execution_id,
            market_resolved_as=pm_data.get("market_resolved_as", "unknown"),
            thesis_was_correct=pm_data.get("thesis_was_correct"),
            realized_pnl_usd=execution.realized_pnl_usd,
            signals_present=signals_present,
            what_the_thesis_said=pm_data.get("what_the_thesis_said", ""),
            what_happened=pm_data.get("what_happened", ""),
            what_would_have_changed_outcome=pm_data.get("what_would_have_changed_outcome", ""),
            source_weight_updates=pm_data.get("source_weight_updates", {}),
            vault_path=pm_path,
        )

        # (e) Append to market doc
        pm_text = (
            f"\n### Post-Mortem {pm.postmortem_id[:8]}\n"
            f"- Resolved as: {pm.market_resolved_as}\n"
            f"- Thesis correct: {pm.thesis_was_correct}\n"
            f"- PnL: ${pm.realized_pnl_usd or 0:,.2f}\n"
            f"- {pm.what_happened}\n"
        )
        self._market_writer.add_postmortem(market_id, pm_text)

        # (f) Update thesis outcome in vault + ChromaDB
        if thesis_id:
            outcome = self._determine_outcome(pm.thesis_was_correct)
            self._update_thesis_outcome(thesis_id, outcome)

        # (g) Publish PostMortem to Redis
        try:
            await self._redis.publish(PostMortem.CHANNEL, pm.model_dump_json())
        except (ConnectionError, OSError) as exc:
            logger.error("PostMortemGenerator: Redis publish failed: %s", exc)

        # (h) Persist to Postgres
        try:
            async with get_session() as session:
                await PostMortemRepo.save(session, pm)
                if thesis_id and pm.thesis_was_correct is not None:
                    outcome_str = self._determine_outcome(pm.thesis_was_correct).value
                    await ThesisRepo.set_outcome(session, thesis_id, outcome_str)
        except Exception:
            logger.warning("PostMortemGenerator: Postgres save failed", exc_info=True)

        logger.info(
            "PostMortemGenerator: generated PM %s for market %s  correct=%s  pnl=$%s",
            pm.postmortem_id[:8], market_id[:16],
            pm.thesis_was_correct, pm.realized_pnl_usd,
        )
        return pm

    async def _call_llm(
        self,
        market_id: str,
        market_doc: str,
        thesis_doc: str,
        execution: TradeExecution,
    ) -> Optional[dict]:
        """Call LLM provider for post-mortem analysis."""
        user_prompt = (
            f"Market ID: {market_id}\n\n"
            f"Market document:\n{market_doc[:3000]}\n\n"
            f"Thesis document:\n{thesis_doc[:2000]}\n\n"
            f"Execution result:\n"
            f"  Direction: {execution.direction}\n"
            f"  Entry: ${execution.entry_price:.4f}\n"
            f"  Exit: ${execution.exit_price or 0:.4f}\n"
            f"  PnL: ${execution.realized_pnl_usd or 0:.2f}\n"
            f"  Exit reason: {execution.exit_reason.value if execution.exit_reason else 'unknown'}\n"
        )

        try:
            return await self._llm.generate_json(user_prompt, system=PM_SYSTEM_PROMPT, max_tokens=1024)
        except Exception:
            logger.warning("PostMortemGenerator: LLM call failed", exc_info=True)
            return {
                "market_resolved_as": "unknown",
                "thesis_was_correct": None,
                "what_the_thesis_said": "Analysis unavailable",
                "what_happened": "Post-mortem generation failed",
                "what_would_have_changed_outcome": "N/A",
                "source_weight_updates": {},
                "signals_assessment": [],
            }

    def _update_thesis_outcome(self, thesis_id: str, outcome: ThesisOutcome) -> None:
        """Update thesis vault file and ChromaDB metadata."""
        path = vault_path("theses", f"{thesis_id}.md")
        if path.exists():
            text = path.read_text(encoding="utf-8")
            text = text.replace("outcome: null", f"outcome: {outcome.value}")
            path.write_text(text, encoding="utf-8")

        try:
            self._collection.update(
                ids=[thesis_id],
                metadatas=[{"outcome": outcome.value}],
            )
        except Exception:
            logger.warning("PostMortemGenerator: ChromaDB update failed for %s", thesis_id)

    @staticmethod
    def _determine_outcome(thesis_was_correct: Optional[bool]) -> ThesisOutcome:
        if thesis_was_correct is True:
            return ThesisOutcome.WIN
        elif thesis_was_correct is False:
            return ThesisOutcome.LOSS
        return ThesisOutcome.VOID
