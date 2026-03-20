# ADR-021: SOE Execution Wallet Separation

## Status
accepted

## Context
SOE trades Solana-native assets. ORACLE also trades on Polymarket via a Polygon wallet. Different chains, different keypairs. PRD does not specify whether these are kept separate.

## Decision
**Separate wallets — dedicated Solana keypair for SOE, dedicated Polygon keypair for Polymarket.**
- Solana keypair: stored as `SOLANA_PRIVATE_KEY` env var (base58-encoded, loaded via `solders.Keypair.from_base58_string()`)
- Polygon keypair: stored as `POLYMARKET_PRIVATE_KEY` env var (ADR-017)
- No cross-chain capital movement by ORACLE — operator manually allocates SOL/USDC to each wallet

Capital is fully siloed: Polymarket USDC balance and Solana wallet balance are independent. SOE circuit breaker (ADR-006) only tracks SOE daily PnL against the Solana wallet; Polymarket circuit breaker tracks Polymarket daily PnL independently.

## Consequences
- SOE uses `solders` + `solana-py` with `SOLANA_PRIVATE_KEY`; it never touches `POLYMARKET_PRIVATE_KEY`
- Operator must fund two separate wallets
- If either private key env var is missing at startup: the affected executor refuses to start; the other can still run
- Combined daily PnL (Polymarket + SOE) is surfaced on the dashboard but not used as a single circuit breaker

## Alternatives Considered
- Shared custody with chain abstraction: unnecessarily complex, higher risk of cross-chain errors
