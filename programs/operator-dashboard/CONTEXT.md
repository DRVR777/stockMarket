# programs/operator-dashboard — Task Router

## What This Program Is
Real-time operator interface. FastAPI backend + vanilla JS frontend at localhost:8080. Subscribes to all event bus channels, pushes live data to browser via WebSocket, handles operator actions (copy-trade approval, parameter changes). Also relays alerts to Telegram.

---

## Before Writing Any Code

All ADRs accepted. All contracts defined. No stops remain — proceed after spec-review PASS.

---

## Task Routing

Build in this exact sequence. Each row is one discrete unit of work.

| Your Task | Load These | Specification |
|-----------|-----------|---------------|
| 1. FastAPI app skeleton | ADR-018, ADR-023 | Create FastAPI app with uvicorn. Single `main.py`. Static file mount for `static/index.html`. WebSocket endpoint at `/ws`. HTTP endpoints: `POST /action`, `GET /params`, `POST /params`. Run on `0.0.0.0:8080`. |
| 2. Redis subscriber bridge | ADR-014, ADR-015, operator-alert.md | Background asyncio task: subscribe to all 8 `oracle:*` channels. On each message: parse into the relevant Pydantic contract object, serialize to JSON, broadcast to all connected WebSocket clients. Keep a `connected_clients: set[WebSocket]` — remove disconnected clients. |
| 3. Anomaly alert view | anomaly-event.md, operator-alert.md | Frontend JS: on WS message with `type=anomaly_event` or `type=operator_alert` (alert_type=anomaly): prepend to `#alerts-list` div, sorted by anomaly_score desc. Each row shows: wallet_address (truncated), market question (40 chars), notional_usd, anomaly_score, trigger_reasons, copy_trade_eligible badge. If `action_required=true`: show [Approve Copy Trade] button. |
| 4. Thesis display view | trade-thesis.md | Frontend JS: on WS message with `type=trade_thesis`: maintain a table of active theses (decision != skip). Columns: market (40 chars), direction, re_prob, mkt_prob, delta (colored red/green), confidence, decision badge. Auto-remove theses after 24h or when outcome is labeled. |
| 5. SOE position view | trade-execution.md, ADR-015 | Frontend JS: on WS message with `type=trade_execution`: maintain table of open SOE positions. Columns: symbol, entry_price, current_price (update on each Birdeye tick relayed via WS), unrealized_pnl_usd (computed client-side), exit_conditions. Color row red if stop_loss is within 1%. |
| 6. Post-mortem feed | post-mortem.md | Frontend JS: on WS message with `type=post_mortem`: prepend to `#postmortem-feed` div. Each entry: market question, resolved_as, thesis_was_correct badge (green/red/grey), realized_pnl_usd. Expandable to show full what_happened + what_would_have_changed_outcome. |
| 7. Parameter control panel | ADR-006, ADR-010, ADR-015 | Frontend JS: on page load, `GET /params` → render editable form. Fields: large_order_threshold_usd, copy_trade_threshold, re_delta_threshold, re_confidence_min, re_scan_interval_minutes, copy_trade_mode (dropdown: manual/semi/auto), soe_entry_floor_pct, soe_take_profit_pct, soe_stop_loss_pct, soe_max_position_usd, soe_daily_loss_ceiling_usd. On submit: `POST /params` → server writes to `oracle:state:params` Redis hash. |
| 8. Copy-trade action handler | operator-alert.md, ADR-010, ADR-014 | `POST /action` with `{action: "approve_copy_trade", alert_id, anomaly_event_id}`. Server: (a) mark alert acknowledged in Redis, (b) publish `oracle:copy_trade_approved:{anomaly_event_id}` — SIL execution path subscribes to this channel and executes the copy trade. Also handle `{action: "dismiss", alert_id}`: just acknowledge. |
| 9. Telegram relay | ADR-019, operator-alert.md | Background asyncio task: subscribe to `oracle:operator_alert`. For each alert with `severity=action_required` or `severity=warning`: send Telegram message via `python-telegram-bot`. Format: `title\n\nbody\n\nSeverity: {severity}`. Rate-limit: max 1 message per 3 seconds via asyncio.sleep queue. |
| Fix a bug | The failing unit, relevant contracts | Reproduce → fix → verify WS message structure matches contract shape. |

---

## Checkpoints

| After step | Present | Human options |
|------------|---------|---------------|
| Step 2 complete | WS message stream in browser console from live Redis events | approve format / revise |
| Step 7 complete | Full dashboard screenshot with all 5 views populated | approve UI / revise layout |
| Step 9 complete | Sample Telegram alert from live AnomalyEvent | approve / revise message format |

---

## Inference Logging
All inferences → `../../_meta/gaps/pending.txt`
Format: `[timestamp] | operator-dashboard | [file] | inferred "[what]" — no file states this`
