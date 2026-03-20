"""Abstract chain adapter interface.

Any blockchain can be supported by implementing this interface.
The mean-reversion engine and entry/exit logic are chain-agnostic —
they only interact through this adapter.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import AsyncIterator, Optional


@dataclass
class PriceTick:
    """A single price update from any chain."""
    token_address: str
    symbol: str
    price_usd: float
    timestamp: float  # Unix timestamp


@dataclass
class OHLCVBar:
    """One OHLCV bar (daily candle)."""
    timestamp: float
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass
class SwapResult:
    """Result of a swap execution."""
    tx_hash: str
    executed_price: float
    amount_in: float
    amount_out: float
    chain: str


class ChainAdapter(ABC):
    """Abstract interface for blockchain price feeds and execution.

    Implement this to add support for any new chain/DEX.
    """

    @property
    @abstractmethod
    def chain_name(self) -> str:
        """Return the chain identifier (e.g. 'solana', 'ethereum', 'base')."""
        ...

    @abstractmethod
    async def get_price(self, token_address: str) -> float:
        """Get the current USD price for a token."""
        ...

    @abstractmethod
    async def get_ohlcv(
        self,
        token_address: str,
        days: int = 30,
    ) -> list[OHLCVBar]:
        """Get daily OHLCV bars for a token."""
        ...

    @abstractmethod
    async def subscribe_prices(
        self,
        token_addresses: list[str],
    ) -> AsyncIterator[PriceTick]:
        """Subscribe to real-time price ticks for multiple tokens.

        Yields PriceTick objects indefinitely until the connection is closed.
        """
        ...

    @abstractmethod
    async def execute_swap(
        self,
        token_in: str,
        token_out: str,
        amount_usd: float,
        slippage_bps: int = 50,
    ) -> SwapResult:
        """Execute a swap on-chain.

        ``amount_usd`` is the USD value to swap.
        Returns a SwapResult with the transaction hash and execution details.
        """
        ...

    @abstractmethod
    async def get_balance(self, token_address: str) -> float:
        """Get the wallet's balance for a token in USD."""
        ...

    async def close(self) -> None:
        """Clean up connections. Override if needed."""
        pass
