"""Task 4 — Step 2: Hypothesis generation.

Calls Claude Sonnet with the ContextAssembly. Requests structured output:
exactly one YES hypothesis and one NO hypothesis with evidence.
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any

from oracle_shared.contracts.trade_thesis import ContextAssembly, Hypothesis
from oracle_shared.providers import LLMProvider, get_llm

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are an adversarial prediction market analyst. Your job is to argue \
both sides of a prediction market question with equal rigor.

Given the market context (current state, recent signals, whale activity, historical analogues), \
produce exactly TWO hypotheses:
1. A YES hypothesis — the strongest case that the market resolves YES
2. A NO hypothesis — the strongest case that the market resolves NO

Each hypothesis must include:
- "side": "YES" or "NO"
- "argument": A 2-4 sentence argument for this side
- "evidence": A list of 2-5 specific evidence items from the provided context

Respond ONLY with valid JSON in this exact format:
{"hypotheses": [{"side": "YES", "argument": "...", "evidence": ["...", "..."]}, {"side": "NO", "argument": "...", "evidence": ["...", "..."]}]}"""


class HypothesisGenerator:
    """Generate adversarial YES/NO hypotheses via Claude."""

    def __init__(self, llm: LLMProvider | None = None) -> None:
        self._llm = llm or get_llm()

    async def generate(
        self,
        market_question: str,
        context: ContextAssembly,
    ) -> list[Hypothesis]:
        """Generate adversarial YES/NO hypotheses via LLM."""
        user_prompt = self._build_prompt(market_question, context)

        raw_text = await self._llm.generate(user_prompt, system=SYSTEM_PROMPT, max_tokens=1024)
        hypotheses = self._parse_response(raw_text)

        logger.info(
            "HypothesisGenerator: generated %d hypotheses for '%s'",
            len(hypotheses),
            market_question[:60],
        )
        return hypotheses

    def _build_prompt(
        self,
        market_question: str,
        context: ContextAssembly,
    ) -> str:
        ms = context.market_state
        parts = [
            f"Market question: {market_question}",
            f"Current YES price: {ms.get('current_price_yes', '?')}",
            f"Liquidity: ${ms.get('liquidity_usd', 0):,.0f}",
            f"Resolution deadline: {ms.get('resolution_deadline', '?')}",
        ]

        # Recent insights
        insights = ms.get("recent_insights", [])[:5]
        if insights:
            parts.append("\nRecent signals:")
            for ins in insights:
                parts.append(
                    f"  - [{ins.get('source_category', '?')}] "
                    f"{ins.get('raw_text', '')[:150]}"
                )

        # Anomaly events
        if context.anomaly_events:
            parts.append(f"\nWhale activity ({len(context.anomaly_events)} events):")
            for ae in context.anomaly_events[:3]:
                parts.append(
                    f"  - ${ae.get('notional_usd', 0):,.0f} {ae.get('outcome', '?')} "
                    f"score={ae.get('anomaly_score', 0):.2f} "
                    f"triggers={ae.get('trigger_reasons', [])}"
                )

        # Historical analogues
        if context.historical_analogues:
            parts.append(f"\nHistorical analogues ({len(context.historical_analogues)}):")
            for ha in context.historical_analogues:
                parts.append(
                    f"  - thesis {ha.thesis_id[:8]}… "
                    f"similarity={ha.similarity:.2f} "
                    f"outcome={ha.outcome or 'unresolved'}"
                )

        # Semantic summary
        summary = ms.get("semantic_state_summary", "")
        if summary:
            parts.append(f"\nSemantic summary: {summary}")

        return "\n".join(parts)

    @staticmethod
    def _parse_response(raw: str) -> list[Hypothesis]:
        """Parse Claude's JSON response into Hypothesis objects."""
        # Strip markdown code fences if present
        text = raw.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(lines[1:-1] if lines[-1].startswith("```") else lines[1:])

        data = json.loads(text)
        hypotheses = []
        for h in data.get("hypotheses", []):
            hypotheses.append(Hypothesis(
                side=h["side"],
                argument=h["argument"],
                evidence=h.get("evidence", []),
            ))

        # Ensure we have at least YES and NO
        sides = {h.side for h in hypotheses}
        if "YES" not in sides:
            hypotheses.append(Hypothesis(side="YES", argument="Insufficient data for YES case", evidence=[]))
        if "NO" not in sides:
            hypotheses.append(Hypothesis(side="NO", argument="Insufficient data for NO case", evidence=[]))

        return hypotheses
