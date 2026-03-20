# ADR-017: Polymarket Wallet and Keypair Management

## Status
accepted

## Context
ORACLE must place and manage orders on Polymarket's CLOB. This requires a Polygon wallet private key for signing orders via py-clob-client.

## Decision
**Server-side keypair stored as environment variable.**
- Library: `py-clob-client` (Polymarket's official Python SDK)
- Private key: stored as `POLYMARKET_PRIVATE_KEY` env var (hex string, 0x-prefixed)
- API credentials: `POLYMARKET_API_KEY`, `POLYMARKET_API_SECRET`, `POLYMARKET_API_PASSPHRASE` (derived from wallet via py-clob-client key derivation)
- Chain: Polygon mainnet (`POLYGON_CHAIN_ID=137`)
- Funding: USDC on Polygon, deposited to the Polymarket CLOB contract

Wallet address is logged at startup for operator verification. Private key is never logged.

Capital controls: maximum USDC balance in the Polymarket account is the operator's responsibility. ORACLE's circuit breaker (ADR-006) enforces per-trade and daily loss limits but does not control wallet funding.

## Consequences
- SIL (execution path) and any copy-trade executor both use the same `py-clob-client` instance
- The client is initialized once at startup and shared via dependency injection — not re-initialized per trade
- If private key env var is missing at startup: program refuses to start, logs clear error

## Alternatives Considered
- HashiCorp Vault / AWS Secrets Manager: production-grade, unnecessary ops complexity for personal use
- Hardware wallet with signing proxy: most secure, too much friction for initial build
