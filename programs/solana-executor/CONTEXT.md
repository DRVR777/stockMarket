# programs/solana-executor — Task Router

## What This Program Is
SOE — autonomous mean-reversion trader. Monitors Solana asset prices via Birdeye, maintains a per-asset statistical model, queries RE for AI floor estimates, executes buys in the floor zone, exits at take-profit or stop-loss. Starts in paper trading mode.

---

## Before Writing Any Code

All ADRs accepted. All contracts defined. No stops remain — proceed after spec-review PASS.
Paper trading mode is mandatory until Step 8 checkpoint is approved.

---

## Statistical Model (per asset)

Maintained in Redis at `oracle:state:soe_model:{token_address}`:
- `prices_30d`: list of daily close prices (OHLCV from Birdeye, last 30 entries)
- `ma_20`: 20-day simple moving average (computed on update)
- `std_dev`: standard deviation of prices_30d
- `price_velocity`: `(price_now - price_24h_ago) / price_24h_ago`
- `ai_floor_estimate`: last value from RE, with `estimated_at` timestamp
- `current_price`: latest Birdeye WS tick

Configurable params (read from `oracle:state:params`):
- `soe_entry_floor_pct`: default 0.05 (price within 5% of AI floor)
- `soe_take_profit_pct`: default 0.08 (8% above entry)
- `soe_stop_loss_pct`: default 0.04 (4% below entry)
- `soe_max_position_usd`: default 500
- `soe_max_concurrent_positions`: default 3
- `soe_daily_loss_ceiling_usd`: default 200

---

## Task Routing

Build in this exact sequence. Each row is one discrete unit of work.

| Your Task | Load These | Specification |
|-----------|-----------|---------------|
| 1. Asset config loader | ADR-016, ADR-015 | Read configured assets from `oracle:state:params:soe_assets` (JSON list of `{token_address, symbol}`). Initialize Redis model key per asset if not present. |
| 2. Birdeye OHLCV backfill | ADR-016, ADR-015 | On startup: for each asset, call `GET /defi/ohlcv?address={token}&type=1D&limit=30` from Birdeye REST. Populate `prices_30d`, compute `ma_20` and `std_dev`. Store in Redis model. |
| 3. Birdeye WS price monitor | ADR-016, ADR-015 | Subscribe to Birdeye WS price feed for each configured asset. On each tick: update `oracle:state:soe_model:{token}:current_price`. Update `price_velocity` if 24h data is available. Trigger entry check (step 5). |
| 4. RE floor estimate requester | ADR-009, ADR-014 | Every 6 hours per asset (APScheduler): publish to `oracle:re_floor_request` with `{asset, price_history: prices_30d, request_id: uuid}`. Subscribe to `oracle:re_floor_response:{request_id}`. On response: update `oracle:state:soe_model:{token}:ai_floor_estimate`. |
| 5. Entry logic | trade-execution.md, ADR-006 | On each price tick: check entry conditions — (a) `current_price < ma_20` AND (b) `current_price <= ai_floor_estimate * (1 + soe_entry_floor_pct)`. If both true AND circuit breaker not active AND open positions < soe_max_concurrent_positions: trigger paper or live buy (see step 6). |
| 6. Buy executor — paper first | trade-execution.md, ADR-021 | Paper mode (default): create TradeExecution(status=open, execution_source=soe_mean_reversion) and write to Redis only — do NOT call Jupiter. Live mode (enabled after checkpoint): call Jupiter quote API for best swap route, sign and submit tx via `solders.Keypair`. Publish TradeExecution to `oracle:trade_execution`. |
| 7. Exit monitor | trade-execution.md, ADR-006 | For each open SOE position: on each price tick, check two conditions — (a) take-profit: `current_price >= entry_price * (1 + soe_take_profit_pct)` → close, reason=take_profit; (b) stop-loss: `current_price <= entry_price * (1 - soe_stop_loss_pct)` → close, reason=stop_loss. On close: update TradeExecution(status=closed, exit_price, exit_at, exit_reason, realized_pnl_usd). Publish updated TradeExecution. |
| 8. Circuit breaker | trade-execution.md, ADR-006, ADR-015 | On each position close: add `realized_pnl_usd` to `oracle:state:daily_pnl:soe` (Redis string, reset at midnight UTC). If `soe_daily_pnl <= -soe_daily_loss_ceiling_usd`: set `oracle:state:circuit_breaker:soe = {active: true, triggered_at: now}`. Publish OperatorAlert(alert_type=circuit_breaker, severity=warning). Entry logic (step 5) reads circuit breaker state before executing. |
| Fix a bug | The failing unit, relevant contracts | Reproduce → fix → verify TradeExecution output matches contract shape. |

---

## Checkpoints

| After step | Present | Human options |
|------------|---------|---------------|
| Step 5 complete (paper mode) | Entry signals logged for last 7 days of price data (backsim) | approve entry formula / adjust params |
| Step 7 complete (paper mode) | Paper P&L over 48h of live price monitoring | approve / adjust take-profit or stop-loss |
| Before enabling live mode | Review paper trading results, confirm wallet funded | approve live execution / continue paper trading |

---

## Inference Logging
All inferences → `../../_meta/gaps/pending.txt`
Format: `[timestamp] | solana-executor | [file] | inferred "[what]" — no file states this`
