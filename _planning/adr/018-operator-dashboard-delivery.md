# ADR-018: Operator Dashboard Delivery Mode

## Status
accepted

## Context
The operator dashboard must show live anomaly alerts, active theses, SOE positions, post-mortem feed, and a parameter control panel. It must support copy-trade approval actions.

## Decision
**Local web app: FastAPI backend + plain HTML/JS frontend (no framework).**
- Backend: FastAPI with WebSocket endpoints for live data push to browser
- Frontend: single `index.html` with vanilla JS — no React, no build step
- Served at `http://localhost:8080` — accessible from any browser on the local network
- Live data: WebSocket connection from browser to FastAPI, which relays Redis pub/sub events
- Actions (copy-trade approve, parameter change): HTTP POST to FastAPI endpoints

Why no framework: the dashboard is a read-heavy monitoring tool with a handful of action buttons. A build pipeline adds friction with no benefit here.

## Consequences
- operator-dashboard program runs a FastAPI server as its main process
- All operator actions go through FastAPI → Redis state update → event bus notification to executors
- Dashboard is not authenticated (local-only assumption) — add basic auth if exposing beyond localhost
- Mobile access: works in mobile browser on same local network

## Alternatives Considered
- CLI TUI (Textual): no browser needed, harder to implement copy-trade approval buttons cleanly
- Telegram bot: good for alerts (covered by ADR-019), poor for parameter control UI
- Grafana: excellent for metrics charts, cannot handle copy-trade approval actions
