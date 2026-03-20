"""Tasks 5 & 7 — Entry and exit logic.

Chain-agnostic mean-reversion strategy:
  Entry: price < MA20 AND price <= floor * (1 + entry_pct)
  Exit:  take-profit or stop-loss

Works with any asset that has an AssetModel.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any, Optional

from oracle_shared.contracts.trade_execution import (
    ExitReason,
    ExecutionSource,
    ExecutionStatus,
    MarketType,
    TradeExecution,
)
from oracle_shared.db import get_session
from oracle_shared.db.repository import ExecutionRepo

from solana_executor.chains.base import ChainAdapter, SwapResult
from solana_executor.statistical_model import AssetModel
from solana_executor.config import (
    CIRCUIT_BREAKER_KEY,
    DAILY_LOSS_CEILING_USD,
    DAILY_PNL_KEY,
    ENTRY_FLOOR_PCT,
    MAX_CONCURRENT_POSITIONS,
    MAX_POSITION_USD,
    PAPER_TRADING,
    PARAMS_KEY,
    POSITIONS_KEY,
    STOP_LOSS_PCT,
    TAKE_PROFIT_PCT,
)

logger = logging.getLogger(__name__)


class EntryExitEngine:
    """Chain-agnostic entry/exit logic for mean-reversion trading."""

    def __init__(
        self,
        redis_client: Any,
        chain: ChainAdapter,
    ) -> None:
        self._redis = redis_client
        self._chain = chain

    # ── Entry check (Task 5) ──────────────────────────────────────────────────

    async def check_entry(self, model: AssetModel) -> Optional[TradeExecution]:
        """Check entry conditions and execute if met. Returns TradeExecution or None."""
        price = model.current_price
        if price <= 0 or model.ma_20 <= 0:
            return None

        # Condition (a): price below 20-day MA
        if price >= model.ma_20:
            return None

        # Condition (b): price within floor zone
        floor_pct = await self._get_param("soe_entry_floor_pct", ENTRY_FLOOR_PCT)
        floor_ceiling = model.ai_floor_estimate * (1 + floor_pct)
        if model.ai_floor_estimate > 0 and price > floor_ceiling:
            return None

        # Circuit breaker check
        if await self._is_circuit_breaker_active():
            logger.debug("EntryExitEngine: circuit breaker active — skipping entry")
            return None

        # Position limit check
        open_count = await self._count_open_positions()
        max_concurrent = int(await self._get_param(
            "soe_max_concurrent_positions", MAX_CONCURRENT_POSITIONS
        ))
        if open_count >= max_concurrent:
            return None

        # Execute
        max_pos = await self._get_param("soe_max_position_usd", MAX_POSITION_USD)
        return await self._execute_buy(model, min(max_pos, MAX_POSITION_USD))

    async def _execute_buy(
        self, model: AssetModel, size_usd: float,
    ) -> TradeExecution:
        """Execute a buy (paper or live)."""
        tx_hash = ""

        if not PAPER_TRADING:
            try:
                from solana_executor.chains.solana import USDC_MINT
                result = await self._chain.execute_swap(
                    token_in=USDC_MINT,
                    token_out=model.token_address,
                    amount_usd=size_usd,
                )
                tx_hash = result.tx_hash
            except Exception:
                logger.exception("EntryExitEngine: swap execution failed")
                return None

        # Determine market type from chain
        market_type_map = {"solana": MarketType.SOLANA, "polymarket": MarketType.POLYMARKET}
        market_type = market_type_map.get(self._chain.chain_name, MarketType.SOLANA)

        execution = TradeExecution(
            market_id=model.token_address,
            market_type=market_type,
            direction="buy",
            entry_price=model.current_price,
            size_usd=size_usd,
            executed_at=datetime.now(timezone.utc),
            execution_source=ExecutionSource.SOE_MEAN_REVERSION,
            tx_hash=tx_hash or None,
        )

        # Store in Redis
        await self._redis.hset(
            POSITIONS_KEY, execution.execution_id, execution.model_dump_json()
        )
        # Publish
        await self._redis.publish(
            TradeExecution.CHANNEL, execution.model_dump_json()
        )
        # Persist to Postgres
        try:
            async with get_session() as session:
                await ExecutionRepo.save(session, execution)
        except Exception:
            logger.warning("EntryExitEngine: Postgres save failed", exc_info=True)

        logger.info(
            "EntryExitEngine: ENTRY %s  %s @ $%.4f  size=$%.2f  paper=%s",
            model.symbol, model.token_address[:12], model.current_price,
            size_usd, PAPER_TRADING,
        )
        return execution

    # ── Exit check (Task 7) ───────────────────────────────────────────────────

    async def check_exits(self, current_prices: dict[str, float]) -> list[TradeExecution]:
        """Check all open positions for take-profit or stop-loss."""
        closed: list[TradeExecution] = []
        positions = await self._redis.hgetall(POSITIONS_KEY)

        tp_pct = await self._get_param("soe_take_profit_pct", TAKE_PROFIT_PCT)
        sl_pct = await self._get_param("soe_stop_loss_pct", STOP_LOSS_PCT)

        for exec_id, raw in positions.items():
            try:
                ex = TradeExecution.model_validate_json(raw)
            except Exception:
                continue

            if ex.status != ExecutionStatus.OPEN:
                continue
            if ex.execution_source != ExecutionSource.SOE_MEAN_REVERSION:
                continue

            price = current_prices.get(ex.market_id, 0.0)
            if price <= 0:
                continue

            exit_reason: Optional[ExitReason] = None

            # Take profit
            if price >= ex.entry_price * (1 + tp_pct):
                exit_reason = ExitReason.TAKE_PROFIT
            # Stop loss
            elif price <= ex.entry_price * (1 - sl_pct):
                exit_reason = ExitReason.STOP_LOSS

            if exit_reason:
                closed_ex = await self._close_position(ex, price, exit_reason)
                if closed_ex:
                    closed.append(closed_ex)

        return closed

    async def _close_position(
        self,
        ex: TradeExecution,
        exit_price: float,
        reason: ExitReason,
    ) -> Optional[TradeExecution]:
        """Close a position and update everywhere."""
        pnl = (exit_price - ex.entry_price) / ex.entry_price * ex.size_usd

        ex.status = ExecutionStatus.CLOSED
        ex.exit_price = exit_price
        ex.exit_at = datetime.now(timezone.utc)
        ex.exit_reason = reason
        ex.realized_pnl_usd = round(pnl, 2)

        # Update Redis
        await self._redis.hset(POSITIONS_KEY, ex.execution_id, ex.model_dump_json())
        # Publish
        await self._redis.publish(TradeExecution.CHANNEL, ex.model_dump_json())
        # Update daily PnL
        await self._update_daily_pnl(pnl)
        # Persist
        try:
            async with get_session() as session:
                await ExecutionRepo.close_position(
                    session, ex.execution_id, exit_price, reason.value, pnl,
                )
        except Exception:
            logger.warning("EntryExitEngine: Postgres close failed", exc_info=True)

        logger.info(
            "EntryExitEngine: EXIT %s  reason=%s  entry=$%.4f  exit=$%.4f  pnl=$%.2f",
            ex.market_id[:12], reason.value, ex.entry_price, exit_price, pnl,
        )
        return ex

    # ── Circuit breaker (Task 8) ──────────────────────────────────────────────

    async def _update_daily_pnl(self, pnl: float) -> None:
        """Update daily PnL and check circuit breaker."""
        current = float(await self._redis.get(DAILY_PNL_KEY) or "0")
        new_pnl = current + pnl
        await self._redis.set(DAILY_PNL_KEY, str(round(new_pnl, 2)))

        ceiling = await self._get_param("soe_daily_loss_ceiling_usd", DAILY_LOSS_CEILING_USD)
        if new_pnl <= -ceiling:
            await self._trigger_circuit_breaker(new_pnl)

    async def _trigger_circuit_breaker(self, daily_pnl: float) -> None:
        """Activate circuit breaker and emit OperatorAlert."""
        from oracle_shared.contracts.operator_alert import (
            AlertSeverity, AlertType, OperatorAlert,
        )

        await self._redis.set(
            CIRCUIT_BREAKER_KEY,
            json.dumps({"active": True, "triggered_at": datetime.now(timezone.utc).isoformat()}),
        )

        alert = OperatorAlert(
            created_at=datetime.now(timezone.utc),
            alert_type=AlertType.CIRCUIT_BREAKER,
            severity=AlertSeverity.WARNING,
            title=f"SOE circuit breaker: daily PnL ${daily_pnl:,.2f}",
            body=f"SOE daily loss ceiling breached. PnL: ${daily_pnl:,.2f}. All entries paused.",
            action_required=False,
            action_options=["acknowledge", "reset_circuit_breaker"],
        )
        await self._redis.publish(OperatorAlert.CHANNEL, alert.model_dump_json())
        logger.warning("EntryExitEngine: CIRCUIT BREAKER TRIGGERED  pnl=$%.2f", daily_pnl)

    async def _is_circuit_breaker_active(self) -> bool:
        raw = await self._redis.get(CIRCUIT_BREAKER_KEY)
        if raw is None:
            return False
        try:
            return json.loads(raw).get("active", False)
        except (json.JSONDecodeError, TypeError):
            return False

    async def _count_open_positions(self) -> int:
        positions = await self._redis.hgetall(POSITIONS_KEY)
        count = 0
        for raw in positions.values():
            try:
                ex = TradeExecution.model_validate_json(raw)
                if (ex.status == ExecutionStatus.OPEN
                        and ex.execution_source == ExecutionSource.SOE_MEAN_REVERSION):
                    count += 1
            except Exception:
                continue
        return count

    async def _get_param(self, name: str, default: float) -> float:
        raw = await self._redis.hget(PARAMS_KEY, name)
        if raw:
            try:
                return float(raw)
            except (ValueError, TypeError):
                pass
        return default
