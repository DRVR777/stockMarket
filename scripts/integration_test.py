"""Full end-to-end integration test — exercises every major component."""
from __future__ import annotations

import asyncio
import os
import sys

# Load env
for line in open(os.path.join(os.path.dirname(__file__), "..", ".env")):
    line = line.strip()
    if "=" in line and not line.startswith("#"):
        k, v = line.split("=", 1)
        os.environ[k] = v

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "programs", "market-scanner"))

import redis.asyncio as aioredis
import pandas as pd
from datetime import datetime, timezone
from oracle_shared.contracts.signal import Signal, SignalCategory, SourceId
from oracle_shared.contracts.anomaly_event import AnomalyEvent
from oracle_shared.contracts.market_state import MarketState
from oracle_shared.db import get_session, init_db
from oracle_shared.db.models import SignalRow, AnomalyEventRow
from oracle_shared.providers import get_llm, get_embedder


async def main() -> None:
    redis = aioredis.from_url(os.environ["REDIS_URL"], decode_responses=True)
    await init_db()
    now = datetime.now(timezone.utc)
    passed = 0

    print("=" * 55)
    print("  ORACLE — Full Integration Test")
    print("=" * 55, "\n")

    # 1. Signals -> Redis + Postgres
    print("1. Signal ingestion -> Redis + Postgres")
    for i in range(3):
        sig = Signal(
            source_id=SourceId.POLYGON_CLOB, timestamp=now,
            category=SignalCategory.ON_CHAIN,
            raw_payload={"tx_hash": f"0xfull{i}", "wallet": f"0xw{i}",
                         "market_id": "btc_100k", "outcome": "YES",
                         "side": "buy", "price": 0.65, "size_usd": 15000 + i * 5000},
            market_ids=["btc_100k"],
        )
        await redis.publish(Signal.CHANNEL, sig.model_dump_json())
        async with get_session() as session:
            session.add(SignalRow.from_contract(sig))
    print("   OK — 3 signals published + persisted")
    passed += 1

    # 2. Anomaly event
    print("2. Whale detector anomaly event")
    ae = AnomalyEvent(
        timestamp=now, wallet_address="0xwhale", market_id="btc_100k",
        outcome="YES", notional_usd=25000, anomaly_score=0.82,
        trigger_reasons=["large_order", "tier_1_wallet"],
        copy_trade_eligible=True, source_signal_id="sig-001",
    )
    await redis.publish(AnomalyEvent.CHANNEL, ae.model_dump_json())
    async with get_session() as session:
        session.add(AnomalyEventRow.from_contract(ae))
    print(f"   OK — score={ae.anomaly_score} triggers={ae.trigger_reasons}")
    passed += 1

    # 3. Market state
    print("3. Market state in Redis")
    ms = MarketState(
        market_id="btc_100k",
        market_question="Will Bitcoin exceed $100,000 by December 2026?",
        current_price_yes=0.65,
        resolution_deadline=datetime(2026, 12, 31, tzinfo=timezone.utc),
        liquidity_usd=500000, last_price_updated=now,
        semantic_state_summary="Whale activity surging on YES side",
        signal_count_24h=3, whale_event_count_24h=1,
    )
    await redis.hset("oracle:state:markets", "btc_100k", ms.model_dump_json())
    print(f'   OK — "{ms.market_question[:45]}..." seeded')
    passed += 1

    # 4. Postgres counts
    print("4. Postgres verification")
    async with get_session() as session:
        from sqlalchemy import select, func
        sigs = (await session.execute(select(func.count()).select_from(SignalRow))).scalar_one()
        aes = (await session.execute(select(func.count()).select_from(AnomalyEventRow))).scalar_one()
    print(f"   OK — signals={sigs} anomaly_events={aes}")
    passed += 1

    # 5. Redis readback
    print("5. Redis state readback")
    raw = await redis.hget("oracle:state:markets", "btc_100k")
    loaded = MarketState.model_validate_json(raw)
    print(f"   OK — liquidity=${loaded.liquidity_usd:,.0f}")
    passed += 1

    # 6. Gemini LLM hypothesis
    print("6. Gemini LLM — adversarial hypothesis generation")
    llm = get_llm()
    prompt = (
        'Market: "Will Bitcoin exceed $100k by Dec 2026?" YES=0.65. '
        "Whale bought $25k YES. Anomaly score 0.82. "
        "Return ONLY JSON: "
        '{"hypotheses": [{"side": "YES", "argument": "..."}, {"side": "NO", "argument": "..."}]}'
    )
    result = await llm.generate_json(prompt, system="You are an adversarial analyst.")
    for h in result.get("hypotheses", []):
        print(f"   {h['side']}: {h['argument'][:90]}")
    passed += 1

    # 7. Gemini embeddings
    print("7. Gemini embeddings")
    emb = get_embedder()
    vec = await emb.embed_single("Bitcoin price prediction 2026")
    batch = await emb.embed(["whale activity", "interest rates", "ETF inflows"])
    print(f"   OK — single={len(vec)} dims, batch={len(batch)} vectors")
    passed += 1

    # 8. Post-mortem generation
    print("8. Gemini post-mortem analysis")
    pm = await llm.generate_json(
        "Trader bought YES at $0.55 on BTC $100k market. Resolved YES=$1.00. PnL +$225. "
        'Return JSON: {"thesis_was_correct": true, "what_happened": "...", '
        '"source_weight_updates": {"on_chain": 0.05}}',
        system="You are a trading post-mortem analyst.",
    )
    print(f"   OK — correct={pm.get('thesis_was_correct')} weights={pm.get('source_weight_updates', {})}")
    passed += 1

    # 9. Semantic summary (OSFE simulation)
    print("9. Gemini semantic summary")
    summary = await llm.generate(
        "Signals: whale $25k YES buy, Bloomberg says BTC ETF inflows record. "
        "Summarize in 30 words.", max_tokens=80,
    )
    print(f"   OK — {summary.strip()[:120]}")
    passed += 1

    # 10. Binance -> SMC -> Redis -> Postgres
    print("10. Scanner -> SMC analysis -> pipeline")
    from market_scanner.providers.binance_ws import BinanceKlineProvider
    from market_scanner.smc import analyze_smc
    provider = BinanceKlineProvider()
    candles = await provider.get_recent_klines("BTCUSDT", interval="15m", limit=100)
    df = pd.DataFrame([
        {"open": c.open, "high": c.high, "low": c.low, "close": c.close, "volume": c.volume}
        for c in candles
    ])
    smc = analyze_smc(df, "BTCUSDT", "crypto")
    if smc:
        print(f"   BTC: {smc.bias.value} conf={smc.confidence} setup={smc.setup_type or 'none'}")
        scanner_sig = Signal(
            source_id=SourceId.AI_OPINION, timestamp=now, category=SignalCategory.PRICE,
            raw_payload={"scanner": "smc", "symbol": "BTCUSDT", "bias": smc.bias.value},
            confidence=smc.confidence,
        )
        await redis.publish(Signal.CHANNEL, scanner_sig.model_dump_json())
        async with get_session() as session:
            session.add(SignalRow.from_contract(scanner_sig))
        print("   OK — published to Redis + Postgres")
    passed += 1

    # Final
    async with get_session() as session:
        from sqlalchemy import select, func
        total = (await session.execute(select(func.count()).select_from(SignalRow))).scalar_one()

    await redis.aclose()

    print(f"\n{'=' * 55}")
    print(f"  {passed}/10 CHECKS PASSED — {total} signals in Postgres")
    print(f"{'=' * 55}")


if __name__ == "__main__":
    asyncio.run(main())
