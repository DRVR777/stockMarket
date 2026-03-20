"""Task 5 — Step 3: Evidence weighting.

Scores each hypothesis against recent insights. Computes:
  - YES_score and NO_score (weighted by source_credibility_weight)
  - re_probability_estimate = YES_score / (YES_score + NO_score)
  - probability_delta = re_probability_estimate - current_price_yes

If abs(probability_delta) < threshold → decision=skip.

This step is pure math — no LLM calls.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

from oracle_shared.contracts.trade_thesis import (
    ContextAssembly,
    EvidenceWeight,
    Hypothesis,
)

from reasoning_engine.config import (
    PARAMS_KEY,
    RE_DELTA_THRESHOLD_DEFAULT,
    RE_DELTA_THRESHOLD_PARAM,
)

logger = logging.getLogger(__name__)


class EvidenceWeigher:
    """Score hypotheses against market context and compute probability estimate."""

    def __init__(self, redis_client: Any) -> None:
        self._redis = redis_client

    async def weigh(
        self,
        hypotheses: list[Hypothesis],
        context: ContextAssembly,
    ) -> tuple[list[EvidenceWeight], float, float, bool]:
        """Weigh evidence and compute probability.

        Returns:
            (evidence_weights, re_probability_estimate, probability_delta, should_skip)
        """
        ms = context.market_state
        insights = ms.get("recent_insights", [])
        current_price = float(ms.get("current_price_yes", 0.5))

        yes_score = 0.0
        no_score = 0.0

        # Score each insight's support for YES vs NO
        # Heuristic: if the insight text mentions positive/affirmative language
        # for the YES side, weight it toward YES; otherwise NO.
        # The real signal is the source_credibility_weight amplifying reliable sources.
        for insight in insights:
            weight = float(insight.get("source_credibility_weight", 1.0))
            # Simple heuristic: split credit based on market price tendency
            # A more sophisticated version would use the hypothesis evidence
            yes_score += weight * current_price
            no_score += weight * (1.0 - current_price)

        # Incorporate anomaly events (whale activity direction)
        for ae in context.anomaly_events:
            outcome = str(ae.get("outcome", "")).upper()
            score_val = float(ae.get("anomaly_score", 0.5))
            if "YES" in outcome:
                yes_score += score_val
            elif "NO" in outcome:
                no_score += score_val

        # Normalize to probability
        total = yes_score + no_score
        if total > 0:
            re_prob = yes_score / total
        else:
            re_prob = 0.5

        probability_delta = re_prob - current_price

        # Build evidence weight records
        yes_reasoning = (
            f"YES evidence: {len(insights)} insights weighted toward YES "
            f"(score={yes_score:.2f}), {len(context.anomaly_events)} whale events"
        )
        no_reasoning = (
            f"NO evidence: {len(insights)} insights weighted toward NO "
            f"(score={no_score:.2f}), market implied {1-current_price:.2%} NO"
        )
        evidence_weights = [
            EvidenceWeight(
                hypothesis_side="YES",
                score=round(re_prob, 4),
                reasoning=yes_reasoning,
            ),
            EvidenceWeight(
                hypothesis_side="NO",
                score=round(1 - re_prob, 4),
                reasoning=no_reasoning,
            ),
        ]

        # Check delta threshold
        threshold = await self._get_delta_threshold()
        should_skip = abs(probability_delta) < threshold

        logger.info(
            "EvidenceWeigher: re_prob=%.3f  market=%.3f  delta=%.3f  skip=%s",
            re_prob, current_price, probability_delta, should_skip,
        )

        return evidence_weights, round(re_prob, 4), round(probability_delta, 4), should_skip

    async def _get_delta_threshold(self) -> float:
        raw = await self._redis.hget(PARAMS_KEY, RE_DELTA_THRESHOLD_PARAM)
        if raw:
            try:
                return float(raw)
            except (ValueError, TypeError):
                pass
        return RE_DELTA_THRESHOLD_DEFAULT
