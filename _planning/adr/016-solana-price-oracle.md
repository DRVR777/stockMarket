# ADR-016: Solana Price Oracle and DEX

## Status
accepted

## Context
SOE needs real-time Solana asset prices for statistical model computation, and a DEX router to execute swaps when entry/exit conditions are met.

## Decision
**Price feeds: Birdeye API** (`api.birdeye.so`)
- Endpoint: `/defi/price` for spot price, `/defi/ohlcv` for historical OHLCV (N-day MA computation)
- WebSocket: `wss://public-api.birdeye.so/socket` for real-time price updates
- Free tier: sufficient for monitoring up to 20 configured assets
- API key stored as `BIRDEYE_API_KEY` environment variable

**Execution routing: Jupiter Aggregator v6 API** (`quote-api.jup.ag`)
- Best swap routes across Solana DEXes (Raydium, Orca, Meteora, etc.)
- No API key required for quote API
- Execution via `solders` + `solana-py` for transaction signing and submission
- Slippage tolerance: configurable, default 0.5%

## Consequences
- SOE has two external dependencies: Birdeye (price monitoring) and Jupiter (execution)
- If Birdeye WebSocket drops: fall back to polling `/defi/price` every 5 seconds
- If Jupiter quote API is unavailable: pause execution, emit circuit breaker alert — do not execute at unknown price
- Solana wallet keypair (ADR-021) used by `solders.Keypair` for transaction signing

## Alternatives Considered
- Pyth Network: on-chain oracle, excellent for DeFi but adds Solana RPC dependency and slightly more latency
- DexScreener: free, no auth, but less reliable and no official SLA
- Helius price API: good alternative if Birdeye free tier is limiting
