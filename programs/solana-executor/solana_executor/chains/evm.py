"""EVM chain adapter — works with Ethereum, Base, Arbitrum, Polygon, etc.

Uses free public APIs:
  - Price data: CoinGecko or DexScreener (free, no auth)
  - Execution: Uniswap/1inch aggregator APIs
  - Balance: Etherscan-compatible block explorers

To add a new EVM chain, just pass a different chain_id and rpc_url.
"""
from __future__ import annotations

import json
import logging
import os
from typing import AsyncIterator

import httpx

from solana_executor.chains.base import ChainAdapter, OHLCVBar, PriceTick, SwapResult

logger = logging.getLogger(__name__)

# DexScreener free API — works for any EVM chain
DEXSCREENER_BASE = "https://api.dexscreener.com/latest"

# 1inch aggregator (free, multi-chain)
ONEINCH_BASE = "https://api.1inch.dev/swap/v6.0"

# Common chain IDs
CHAINS = {
    "ethereum": {"id": 1, "name": "Ethereum", "explorer": "https://api.etherscan.io"},
    "base": {"id": 8453, "name": "Base", "explorer": "https://api.basescan.org"},
    "arbitrum": {"id": 42161, "name": "Arbitrum", "explorer": "https://api.arbiscan.io"},
    "polygon": {"id": 137, "name": "Polygon", "explorer": "https://api.polygonscan.com"},
    "optimism": {"id": 10, "name": "Optimism", "explorer": "https://api-optimistic.etherscan.io"},
    "avalanche": {"id": 43114, "name": "Avalanche", "explorer": "https://api.snowtrace.io"},
    "bsc": {"id": 56, "name": "BNB Chain", "explorer": "https://api.bscscan.com"},
}

# USDC addresses per chain
USDC_ADDRESSES = {
    "ethereum": "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48",
    "base": "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913",
    "arbitrum": "0xaf88d065e77c8cC2239327C5EDb3A432268e5831",
    "polygon": "0x3c499c542cEF5E3811e1192ce70d8cC03d5c3359",
    "bsc": "0x8AC76a51cc950d9822D68b83fE1Ad97B32Cd580d",
}


class EVMAdapter(ChainAdapter):
    """Multi-chain EVM adapter using free APIs.

    Supports Ethereum, Base, Arbitrum, Polygon, Optimism, Avalanche, BSC.
    """

    def __init__(
        self,
        chain: str = "ethereum",
        rpc_url: str | None = None,
    ) -> None:
        if chain not in CHAINS:
            raise ValueError(f"Unknown chain: {chain}. Supported: {list(CHAINS.keys())}")
        self._chain = chain
        self._chain_config = CHAINS[chain]
        self._chain_id = self._chain_config["id"]
        self._rpc_url = rpc_url or os.getenv(f"{chain.upper()}_RPC_URL", "")

    @property
    def chain_name(self) -> str:
        return self._chain

    async def get_price(self, token_address: str) -> float:
        """Get current USD price via DexScreener (free, no auth)."""
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(
                f"{DEXSCREENER_BASE}/dex/tokens/{token_address}",
            )
            resp.raise_for_status()
            data = resp.json()

        pairs = data.get("pairs", [])
        if not pairs:
            return 0.0

        # Find the highest-liquidity pair
        best = max(pairs, key=lambda p: float(p.get("liquidity", {}).get("usd", 0)))
        return float(best.get("priceUsd", 0))

    async def get_ohlcv(
        self,
        token_address: str,
        days: int = 30,
    ) -> list[OHLCVBar]:
        """Get daily OHLCV from CoinGecko by contract address.

        Falls back to DexScreener price history if CoinGecko fails.
        """
        try:
            return await self._ohlcv_coingecko(token_address, days)
        except Exception:
            logger.debug("EVMAdapter: CoinGecko OHLCV failed, using DexScreener")
            return await self._ohlcv_dexscreener(token_address)

    async def _ohlcv_coingecko(
        self, token_address: str, days: int,
    ) -> list[OHLCVBar]:
        """OHLCV from CoinGecko contract API."""
        platform = {
            "ethereum": "ethereum",
            "base": "base",
            "arbitrum": "arbitrum-one",
            "polygon": "polygon-pos",
            "bsc": "binance-smart-chain",
            "avalanche": "avalanche",
            "optimism": "optimistic-ethereum",
        }.get(self._chain, "ethereum")

        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.get(
                f"https://api.coingecko.com/api/v3/coins/{platform}/contract/{token_address}/market_chart",
                params={"vs_currency": "usd", "days": str(days)},
            )
            resp.raise_for_status()
            data = resp.json()

        prices = data.get("prices", [])
        if not prices:
            return []

        # CoinGecko returns [timestamp_ms, price] pairs — approximate OHLCV
        bars: list[OHLCVBar] = []
        for ts, price in prices:
            bars.append(OHLCVBar(
                timestamp=ts / 1000,
                open=price,
                high=price * 1.005,  # approximate
                low=price * 0.995,
                close=price,
                volume=0,
            ))
        return bars[-days:]

    async def _ohlcv_dexscreener(self, token_address: str) -> list[OHLCVBar]:
        """Basic price history from DexScreener."""
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(
                f"{DEXSCREENER_BASE}/dex/tokens/{token_address}",
            )
            resp.raise_for_status()
            data = resp.json()

        pairs = data.get("pairs", [])
        if not pairs:
            return []

        # DexScreener doesn't have full OHLCV — use current price as single bar
        best = max(pairs, key=lambda p: float(p.get("liquidity", {}).get("usd", 0)))
        price = float(best.get("priceUsd", 0))
        return [OHLCVBar(
            timestamp=0, open=price, high=price, low=price, close=price, volume=0,
        )]

    async def subscribe_prices(
        self,
        token_addresses: list[str],
    ) -> AsyncIterator[PriceTick]:
        """Poll DexScreener for price updates (no WebSocket available for free).

        Polls every 5 seconds — sufficient for 4h/daily timeframes.
        """
        import asyncio
        while True:
            for addr in token_addresses:
                try:
                    price = await self.get_price(addr)
                    if price > 0:
                        yield PriceTick(
                            token_address=addr,
                            symbol=addr[:8],
                            price_usd=price,
                            timestamp=0,
                        )
                except Exception:
                    pass
            await asyncio.sleep(5)

    async def execute_swap(
        self,
        token_in: str,
        token_out: str,
        amount_usd: float,
        slippage_bps: int = 50,
    ) -> SwapResult:
        """Execute swap via 1inch aggregator (quote only — live swap needs wallet).

        For live execution, would need:
        - Private key in env (EVM_PRIVATE_KEY)
        - web3.py for transaction signing
        - 1inch swap API for the transaction data
        """
        # Get quote from 1inch
        usdc = USDC_ADDRESSES.get(self._chain, token_in)
        amount_raw = int(amount_usd * 1e6)  # USDC has 6 decimals

        async with httpx.AsyncClient(timeout=15.0) as client:
            try:
                resp = await client.get(
                    f"{ONEINCH_BASE}/{self._chain_id}/quote",
                    params={
                        "src": token_in,
                        "dst": token_out,
                        "amount": str(amount_raw),
                    },
                    headers={"Accept": "application/json"},
                )
                if resp.status_code == 200:
                    data = resp.json()
                    out_amount = float(data.get("dstAmount", 0))
                    return SwapResult(
                        tx_hash=f"paper_{self._chain}_{token_out[:8]}",
                        executed_price=amount_raw / out_amount if out_amount > 0 else 0,
                        amount_in=amount_usd,
                        amount_out=out_amount,
                        chain=self._chain,
                    )
            except Exception:
                pass

        # Fallback: paper trade at current price
        price = await self.get_price(token_out)
        return SwapResult(
            tx_hash=f"paper_{self._chain}_{token_out[:8]}",
            executed_price=price,
            amount_in=amount_usd,
            amount_out=amount_usd / price if price > 0 else 0,
            chain=self._chain,
        )

    async def get_balance(self, token_address: str) -> float:
        """Get token balance — placeholder for paper trading."""
        return 0.0
