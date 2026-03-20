"""Tests for SOE — statistical model, entry/exit logic, chain adapter interface.

No chain connections or API keys needed.
"""
import asyncio
import json
import sys
import statistics
from datetime import datetime, timezone
from typing import Any, AsyncIterator, Optional

from oracle_shared.contracts.trade_execution import (
    ExecutionSource, ExecutionStatus, MarketType, TradeExecution,
)

from solana_executor.chains.base import ChainAdapter, OHLCVBar, PriceTick, SwapResult
from solana_executor.statistical_model import AssetModel, ModelStore
from solana_executor.entry_exit import EntryExitEngine


# ── Mock chain adapter ────────────────────────────────────────────────────────

class MockChainAdapter(ChainAdapter):
    """Mock chain adapter for testing."""

    @property
    def chain_name(self) -> str:
        return "mock_chain"

    async def get_price(self, token_address: str) -> float:
        return 100.0

    async def get_ohlcv(self, token_address: str, days: int = 30) -> list[OHLCVBar]:
        return [OHLCVBar(timestamp=float(i), open=100+i, high=105+i, low=95+i, close=100+i, volume=1000) for i in range(days)]

    async def subscribe_prices(self, token_addresses: list[str]) -> AsyncIterator[PriceTick]:
        yield PriceTick(token_address="mock", symbol="MOCK", price_usd=100.0, timestamp=0)

    async def execute_swap(self, token_in: str, token_out: str, amount_usd: float, slippage_bps: int = 50) -> SwapResult:
        return SwapResult(tx_hash="0xmock", executed_price=100.0, amount_in=amount_usd, amount_out=amount_usd/100, chain="mock_chain")

    async def get_balance(self, token_address: str) -> float:
        return 10000.0


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

async def test_statistical_model() -> bool:
    """AssetModel computes MA20 and std_dev correctly."""
    model = AssetModel(token_address="t1", symbol="TEST", chain="mock")
    bars = [OHLCVBar(timestamp=float(i), open=0, high=0, low=0, close=100.0 + i, volume=0) for i in range(30)]
    model.update_from_ohlcv(bars)

    assert len(model.prices_30d) == 30
    expected_ma = statistics.mean([100.0 + i for i in range(10, 30)])
    assert abs(model.ma_20 - expected_ma) < 0.01
    assert model.std_dev > 0

    model.update_price(95.0)
    assert model.current_price == 95.0
    assert model.price_velocity != 0

    print(f"  statistical_model: PASS  ma20={model.ma_20:.2f}  std={model.std_dev:.2f}")
    return True


async def test_model_store() -> bool:
    """ModelStore persists and loads correctly."""
    redis = FakeRedis()
    store = ModelStore(redis)

    model = AssetModel(token_address="t1", symbol="TEST", chain="mock")
    model.prices_30d = [100 + i for i in range(30)]
    model.ma_20 = 119.5
    model.current_price = 115.0

    await store.save(model)
    loaded = await store.load("t1")

    assert loaded is not None
    assert loaded.symbol == "TEST"
    assert loaded.ma_20 == 119.5
    assert len(loaded.prices_30d) == 30

    print("  model_store: PASS")
    return True


async def test_chain_adapter_interface() -> bool:
    """Mock chain adapter satisfies the interface."""
    adapter = MockChainAdapter()
    assert adapter.chain_name == "mock_chain"

    price = await adapter.get_price("token")
    assert price == 100.0

    bars = await adapter.get_ohlcv("token", 10)
    assert len(bars) == 10
    assert all(isinstance(b, OHLCVBar) for b in bars)

    result = await adapter.execute_swap("in", "out", 1000.0)
    assert isinstance(result, SwapResult)
    assert result.chain == "mock_chain"

    print("  chain_adapter: PASS")
    return True


async def test_entry_below_ma() -> bool:
    """Entry triggers when price < MA20 and within floor zone."""
    redis = FakeRedis()
    chain = MockChainAdapter()
    engine = EntryExitEngine(redis, chain)

    model = AssetModel(token_address="t1", symbol="TEST", chain="mock")
    model.ma_20 = 120.0
    model.current_price = 110.0  # below MA
    model.ai_floor_estimate = 108.0  # price within 5% of floor (108 * 1.05 = 113.4)

    execution = await engine.check_entry(model)
    assert execution is not None
    assert execution.direction == "buy"
    assert execution.entry_price == 110.0
    assert execution.execution_source == ExecutionSource.SOE_MEAN_REVERSION

    print(f"  entry_below_ma: PASS  exec_id={execution.execution_id[:8]}")
    return True


async def test_entry_rejected_above_ma() -> bool:
    """No entry when price >= MA20."""
    redis = FakeRedis()
    chain = MockChainAdapter()
    engine = EntryExitEngine(redis, chain)

    model = AssetModel(token_address="t1", symbol="TEST", chain="mock")
    model.ma_20 = 100.0
    model.current_price = 105.0  # above MA

    execution = await engine.check_entry(model)
    assert execution is None

    print("  entry_rejected_above_ma: PASS")
    return True


async def test_exit_take_profit() -> bool:
    """Position closes on take-profit."""
    redis = FakeRedis()
    chain = MockChainAdapter()
    engine = EntryExitEngine(redis, chain)

    # Create a fake open position
    ex = TradeExecution(
        market_id="t1",
        market_type=MarketType.SOLANA,
        direction="buy",
        entry_price=100.0,
        size_usd=500.0,
        executed_at=datetime.now(timezone.utc),
        execution_source=ExecutionSource.SOE_MEAN_REVERSION,
    )
    await redis.hset("oracle:state:positions", ex.execution_id, ex.model_dump_json())

    # Price at +9% (above 8% take-profit)
    closed = await engine.check_exits({"t1": 109.0})
    assert len(closed) == 1
    assert closed[0].exit_reason.value == "take_profit"
    assert closed[0].realized_pnl_usd > 0

    print(f"  exit_take_profit: PASS  pnl=${closed[0].realized_pnl_usd:.2f}")
    return True


async def test_exit_stop_loss() -> bool:
    """Position closes on stop-loss."""
    redis = FakeRedis()
    chain = MockChainAdapter()
    engine = EntryExitEngine(redis, chain)

    ex = TradeExecution(
        market_id="t1",
        market_type=MarketType.SOLANA,
        direction="buy",
        entry_price=100.0,
        size_usd=500.0,
        executed_at=datetime.now(timezone.utc),
        execution_source=ExecutionSource.SOE_MEAN_REVERSION,
    )
    await redis.hset("oracle:state:positions", ex.execution_id, ex.model_dump_json())

    # Price at -5% (below 4% stop-loss)
    closed = await engine.check_exits({"t1": 95.0})
    assert len(closed) == 1
    assert closed[0].exit_reason.value == "stop_loss"
    assert closed[0].realized_pnl_usd < 0

    print(f"  exit_stop_loss: PASS  pnl=${closed[0].realized_pnl_usd:.2f}")
    return True


async def run_all() -> bool:
    tests = [
        test_statistical_model,
        test_model_store,
        test_chain_adapter_interface,
        test_entry_below_ma,
        test_entry_rejected_above_ma,
        test_exit_take_profit,
        test_exit_stop_loss,
    ]

    print("solana-executor (SOE) unit tests\n")
    passed = 0
    failed = 0
    for test in tests:
        try:
            if await test():
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
    ok = asyncio.run(run_all())
    sys.exit(0 if ok else 1)
