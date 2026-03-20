"""Solana chain adapter — Birdeye (prices) + Jupiter (execution).

Implements ChainAdapter for the Solana blockchain.
"""
from __future__ import annotations

import json
import logging
import os
from typing import AsyncIterator

import httpx
import websockets
import websockets.exceptions

from solana_executor.chains.base import ChainAdapter, OHLCVBar, PriceTick, SwapResult
from solana_executor.config import (
    BIRDEYE_REST_BASE,
    BIRDEYE_WS_URL,
    JUPITER_QUOTE_URL,
    JUPITER_SWAP_URL,
    SLIPPAGE_BPS,
)

logger = logging.getLogger(__name__)

# USDC on Solana (used as quote currency for swaps)
USDC_MINT = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"


class SolanaAdapter(ChainAdapter):
    """Birdeye for prices, Jupiter for execution on Solana."""

    def __init__(self, api_key: str | None = None) -> None:
        self._api_key = api_key or os.getenv("BIRDEYE_API_KEY", "")
        self._headers = {
            "X-API-KEY": self._api_key,
            "x-chain": "solana",
        }

    @property
    def chain_name(self) -> str:
        return "solana"

    async def get_price(self, token_address: str) -> float:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(
                f"{BIRDEYE_REST_BASE}/defi/price",
                params={"address": token_address},
                headers=self._headers,
            )
            resp.raise_for_status()
            return float(resp.json().get("data", {}).get("value", 0.0))

    async def get_ohlcv(
        self,
        token_address: str,
        days: int = 30,
    ) -> list[OHLCVBar]:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.get(
                f"{BIRDEYE_REST_BASE}/defi/ohlcv",
                params={
                    "address": token_address,
                    "type": "1D",
                    "time_from": 0,
                    "time_to": 9999999999,
                },
                headers=self._headers,
            )
            resp.raise_for_status()
            items = resp.json().get("data", {}).get("items", [])

        bars: list[OHLCVBar] = []
        for item in items[-days:]:
            bars.append(OHLCVBar(
                timestamp=float(item.get("unixTime", 0)),
                open=float(item.get("o", 0)),
                high=float(item.get("h", 0)),
                low=float(item.get("l", 0)),
                close=float(item.get("c", 0)),
                volume=float(item.get("v", 0)),
            ))
        return bars

    async def subscribe_prices(
        self,
        token_addresses: list[str],
    ) -> AsyncIterator[PriceTick]:
        headers = {"X-API-KEY": self._api_key}
        async with websockets.connect(
            BIRDEYE_WS_URL,
            additional_headers=headers,
            ping_interval=20,
            ping_timeout=30,
        ) as ws:
            for addr in token_addresses:
                await ws.send(json.dumps({
                    "type": "SUBSCRIBE_PRICE",
                    "data": {"address": addr, "chain": "solana"},
                }))

            async for raw in ws:
                try:
                    data = json.loads(raw)
                    event = data.get("data", data)
                    addr = event.get("address", "")
                    price = float(event.get("price", 0))
                    if addr and price > 0:
                        yield PriceTick(
                            token_address=addr,
                            symbol=event.get("symbol", addr[:8]),
                            price_usd=price,
                            timestamp=float(event.get("timestamp", 0)),
                        )
                except (json.JSONDecodeError, TypeError, ValueError):
                    continue

    async def execute_swap(
        self,
        token_in: str,
        token_out: str,
        amount_usd: float,
        slippage_bps: int = SLIPPAGE_BPS,
    ) -> SwapResult:
        """Execute a swap via Jupiter v6.

        For buys: token_in=USDC, token_out=target token
        For sells: token_in=target token, token_out=USDC
        """
        # Convert USD to lamport-scale amount (USDC has 6 decimals)
        amount_raw = int(amount_usd * 1_000_000)

        async with httpx.AsyncClient(timeout=30.0) as client:
            # Step 1: Get quote
            quote_resp = await client.get(
                JUPITER_QUOTE_URL,
                params={
                    "inputMint": token_in,
                    "outputMint": token_out,
                    "amount": str(amount_raw),
                    "slippageBps": str(slippage_bps),
                },
            )
            quote_resp.raise_for_status()
            quote = quote_resp.json()

            out_amount = float(quote.get("outAmount", 0))
            in_amount = float(quote.get("inAmount", amount_raw))

            # Step 2: In live mode, would sign and submit via solders
            # For now, return the quote result
            # (Live execution requires SOLANA_PRIVATE_KEY + solders integration)
            logger.info(
                "SolanaAdapter: Jupiter quote  %s -> %s  in=%s  out=%s",
                token_in[:8], token_out[:8], in_amount, out_amount,
            )

            return SwapResult(
                tx_hash="paper_" + token_out[:8],
                executed_price=in_amount / out_amount if out_amount > 0 else 0.0,
                amount_in=in_amount / 1_000_000,
                amount_out=out_amount / 1_000_000,
                chain="solana",
            )

    async def get_balance(self, token_address: str) -> float:
        """Get token balance via Birdeye wallet endpoint.

        In paper mode, returns a simulated balance.
        """
        # For live mode, would query Solana RPC for token account balance
        # Placeholder: return 0 (paper mode doesn't need real balances)
        return 0.0
