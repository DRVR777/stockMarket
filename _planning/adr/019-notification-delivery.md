# ADR-019: Alert Notification Delivery Mechanism

## Status
accepted

## Context
WADE and RE must push alerts to the operator when anomalies are detected or high-confidence theses are generated. The operator may not be watching the dashboard continuously.

## Decision
**Telegram bot** for out-of-band push notifications.
- Library: `python-telegram-bot` (async)
- Bot token: `TELEGRAM_BOT_TOKEN` env var
- Operator chat ID: `TELEGRAM_CHAT_ID` env var
- Trigger events: AnomalyEvent with `copy_trade_eligible=true`, TradeThesis with `decision=execute`, circuit breaker activations, system errors

Message format:
```
🚨 WHALE ALERT
Market: [market question]
Wallet: [address] (Tier: Shark)
Size: $12,400 | Score: 0.91
[Copy Trade] button → approve via dashboard
```

Telegram is the out-of-band channel. The dashboard shows the same alerts with full detail and action buttons.

## Consequences
- whale-detector and reasoning-engine both publish OperatorAlert objects to `oracle:operator_alert` channel
- operator-dashboard subscribes and relays to Telegram bot AND to browser WebSocket
- If Telegram is unreachable: log error, continue operation — do not block trade logic on notification delivery
- Rate limiting: max 1 message per 3 seconds to the same chat (Telegram limit) — queue and batch if needed

## Alternatives Considered
- Discord webhook: good for team setups, slightly more complex for interactive approval flow
- Email: too slow for trading alerts
- SMS via Twilio: paid, unnecessary when Telegram is free and faster
