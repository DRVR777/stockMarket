# programs/whale-detector — Task Router

## What This Program Is
WADE — surveillance layer. Subscribes to `oracle:signal` for on-chain events, scores anomalies by order size/wallet/timing, maintains wallet registry in Redis, surfaces copy-trade opportunities via OperatorAlert.

---

## Before Writing Any Code

All ADRs accepted. All contracts defined. No stops remain — proceed after spec-review PASS.

---

## Task Routing

Build in this exact sequence. Each row is one discrete unit of work.

| Your Task | Load These | Specification |
|-----------|-----------|---------------|
| 1. Signal subscriber | signal.md, ADR-014 | Subscribe to `oracle:signal`. Filter: `category == "on_chain"` AND `source_id == "polygon_clob"`. Pass each to the threshold check. Ignore all other categories. |
| 2. Threshold flagging | signal.md, anomaly-event.md, ADR-015 | Read `oracle:state:params:large_order_threshold_usd` (default: 5000). If `raw_payload.size_usd >= threshold` → flag as Large Order Event and push to anomaly queue. |
| 3. Wallet registry lookup | wallet-profile.md, ADR-015 | On each flagged event: `HGET oracle:state:wallets {wallet_address}`. If found → deserialize WalletProfile, attach to event. If not found → create stub WalletProfile(reputation_tier=Unknown), store it. |
| 4. Anomaly scoring | anomaly-event.md, wallet-profile.md, ADR-015 | Score = weighted sum of three factors (equal weight 1/3 each): (a) `size_usd / market_liquidity_usd` clamped 0–1, (b) `size_usd / wallet.typical_position_size_usd` normalized (>2x = 1.0), (c) `1 - (hours_to_resolution / 168)` clamped 0–1 (168h = 7 days). Read `market_liquidity_usd` from `oracle:state:markets:{market_id}`. |
| 5. Cascade detection | anomaly-event.md, ADR-015 | Maintain a Redis sorted set `oracle:state:cascade:{market_id}:{outcome}` scored by timestamp. On each flagged event: ZADD the wallet_address, ZRANGEBYSCORE last 300 seconds. If count >= 3 distinct wallets → add `"cascade_buy"` to trigger_reasons, populate cascade_wallets. Expire the set after 600s. |
| 6. AnomalyEvent emit | anomaly-event.md, ADR-014 | Assemble AnomalyEvent from all prior steps. Set `copy_trade_eligible = (anomaly_score >= oracle:state:params:copy_trade_threshold, default 0.7)`. Publish to `oracle:anomaly_event`. |
| 7. OperatorAlert emit | operator-alert.md, ADR-010, ADR-019 | If `copy_trade_eligible`: create OperatorAlert(alert_type=anomaly, severity=action_required, action_options=["approve_copy_trade","dismiss"]). Publish to `oracle:operator_alert`. |
| 8. WalletProfile update | wallet-profile.md, ADR-015 | After each event processed: recalculate `typical_position_size_usd` (rolling median of last 20 fills, stored as a Redis list `oracle:state:wallet_fills:{address}`). Recalculate reputation_tier using thresholds in wallet-profile.md. HSET updated profile back. |
| Fix a bug | The failing unit, the relevant contract | Reproduce → fix → verify output matches contract shape. |

---

## Checkpoints

| After step | Present | Human options |
|------------|---------|---------------|
| Step 4 complete | Anomaly scores from 10 real events vs expected | approve formula / adjust weights |
| Step 7 complete | Sample OperatorAlert JSON from live event | approve / revise |
| All steps complete | Event throughput, false-positive rate estimate | approve for Audit / revise |

---

## Inference Logging
All inferences → `../../_meta/gaps/pending.txt`
Format: `[timestamp] | whale-detector | [file] | inferred "[what]" — no file states this`
