"""reasoning-engine entry point.

Usage:  uv run python -m reasoning_engine
        (or via Makefile: make reasoning-engine)

Runs:
  - Insight subscriber (signal-triggered analysis)
  - AnomalyEvent indexer (per-market Redis index + Postgres)
  - Scheduled full scan (APScheduler, default 30min)
  - SOE floor estimate handler (Redis request/reply)

The 4-step pipeline (context → hypotheses → evidence → confidence → emit)
runs for each triggered market.
"""
from __future__ import annotations

import asyncio
import logging
import os
import signal
import sys

from dotenv import load_dotenv

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", "..", "..", ".env"))

import redis.asyncio as aioredis

from oracle_shared.contracts.trade_thesis import ThesisDecision
from oracle_shared.db import init_db
from oracle_shared.providers import get_embedder, get_llm

from reasoning_engine.config import (
    LOG_FORMAT,
    LOG_LEVEL,
    REDIS_URL,
)
from reasoning_engine.chroma_store import ThesisChromaStore
from reasoning_engine.context_assembler import ContextAssembler
from reasoning_engine.hypothesis_generator import HypothesisGenerator
from reasoning_engine.evidence_weigher import EvidenceWeigher
from reasoning_engine.confidence_calibrator import ConfidenceCalibrator
from reasoning_engine.thesis_emitter import ThesisEmitter
from reasoning_engine.insight_subscriber import InsightSubscriber
from reasoning_engine.anomaly_indexer import AnomalyIndexer
from reasoning_engine.scheduler import ScheduledScanner
from reasoning_engine.floor_estimator import FloorEstimator

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format=LOG_FORMAT,
    stream=sys.stdout,
)
logger = logging.getLogger("reasoning_engine")


class REPipeline:
    """Orchestrates the full 4-step reasoning pipeline for a single market."""

    def __init__(
        self,
        redis_client: aioredis.Redis,
        assembler: ContextAssembler,
        generator: HypothesisGenerator,
        weigher: EvidenceWeigher,
        calibrator: ConfidenceCalibrator,
        emitter: ThesisEmitter,
    ) -> None:
        self.assembler = assembler
        self.generator = generator
        self.weigher = weigher
        self.calibrator = calibrator
        self.emitter = emitter

    async def analyze(self, market_id: str) -> None:
        """Run the full pipeline for a single market."""
        # Step 1: Context assembly
        context = await self.assembler.assemble(market_id)
        if context is None:
            return

        market_question = context.market_state.get("market_question", "")

        # Step 2: Hypothesis generation (Claude)
        hypotheses = await self.generator.generate(market_question, context)

        # Step 3: Evidence weighting
        evidence_weights, re_prob, delta, should_skip = await self.weigher.weigh(
            hypotheses, context,
        )

        if should_skip:
            # Emit skip thesis (for audit trail) and return
            await self.emitter.emit(
                market_id=market_id,
                market_question=market_question,
                hypotheses=hypotheses,
                evidence_weights=evidence_weights,
                context=context,
                re_probability=re_prob,
                probability_delta=delta,
                confidence=0.0,
                decision=ThesisDecision.SKIP,
                position_usd=None,
            )
            return

        # Step 4: Confidence calibration
        confidence, decision, position_usd = await self.calibrator.calibrate(
            context, re_prob, delta,
        )

        # Emit final thesis
        await self.emitter.emit(
            market_id=market_id,
            market_question=market_question,
            hypotheses=hypotheses,
            evidence_weights=evidence_weights,
            context=context,
            re_probability=re_prob,
            probability_delta=delta,
            confidence=confidence,
            decision=decision,
            position_usd=position_usd,
        )


async def main() -> None:
    """Start the reasoning engine."""
    logger.info("reasoning-engine (RE) starting  redis=%s", REDIS_URL)

    redis_client = aioredis.from_url(REDIS_URL, decode_responses=True)

    # Init Postgres tables
    try:
        await init_db()
        logger.info("RE: database tables initialized")
    except Exception:
        logger.warning("RE: database init failed — running without Postgres", exc_info=True)

    # Provider-based embedder + LLM (auto-detects Gemini/OpenAI/Anthropic)
    embedder = get_embedder()
    llm = get_llm()
    logger.info("RE: using LLM=%s  Embedder=%s", type(llm).__name__, type(embedder).__name__)

    # Initialize components
    chroma = ThesisChromaStore()
    assembler = ContextAssembler(redis_client, chroma, embedder.embed_single)
    generator = HypothesisGenerator(llm=llm)
    weigher = EvidenceWeigher(redis_client)
    calibrator = ConfidenceCalibrator(redis_client)
    emitter = ThesisEmitter(redis_client, chroma)

    pipeline = REPipeline(
        redis_client=redis_client,
        assembler=assembler,
        generator=generator,
        weigher=weigher,
        calibrator=calibrator,
        emitter=emitter,
    )

    # Subscribers and scheduler
    insight_sub = InsightSubscriber(redis_client, on_market=pipeline.analyze)
    anomaly_idx = AnomalyIndexer(redis_client)
    scanner = ScheduledScanner(redis_client, analyze_market=pipeline.analyze)
    floor_est = FloorEstimator(redis_client)

    # Graceful shutdown
    loop = asyncio.get_running_loop()
    components = [insight_sub, anomaly_idx, floor_est]

    def _shutdown(sig_name: str) -> None:
        logger.info("reasoning-engine received %s — shutting down", sig_name)
        for comp in components:
            asyncio.ensure_future(comp.stop())
        asyncio.ensure_future(scanner.stop())

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, lambda s=sig.name: _shutdown(s))
        except NotImplementedError:
            pass

    # Start scanner
    await scanner.start()

    # Run all subscribers concurrently
    try:
        await asyncio.gather(
            insight_sub.start(),
            anomaly_idx.start(),
            floor_est.start(),
        )
    finally:
        await scanner.stop()
        await redis_client.aclose()
        logger.info("reasoning-engine (RE) stopped")


if __name__ == "__main__":
    asyncio.run(main())
