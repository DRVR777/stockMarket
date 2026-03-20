# ADR-020: Reasoning Engine Scheduled Scan Trigger

## Status
accepted

## Context
RE must run a full market scan across all active Polymarket markets on a configurable schedule (PRD suggests every 30 minutes) in addition to signal-triggered analysis. The trigger mechanism must be internal to RE.

## Decision
**APScheduler** — `AsyncIOScheduler` with an interval job.
- Library: `apscheduler` (Python)
- Job: `full_market_scan()` runs every N minutes (default: 30, configurable via `oracle:state:params:re_scan_interval_minutes`)
- Scheduler starts when RE starts, runs in the same asyncio event loop as the Redis subscriber
- Scan reads all entries from `oracle:state:markets` Redis hash, runs Steps 1–4 for each

Two analysis paths in RE:
1. **Signal-triggered:** OSFE publishes Insight → RE immediately assembles context for associated markets and runs the pipeline
2. **Scheduled scan:** APScheduler fires → RE runs full pipeline across all active markets regardless of new signals

Both paths produce TradeThesis objects (or skip) via the same pipeline code.

## Consequences
- RE has one main asyncio event loop managing: Redis subscriber (signal-triggered), APScheduler (scheduled scan), and on-demand SOE floor estimate requests
- Scheduled scan may produce ThesisObjects at scale — each is logged to KBPM regardless (ADR-008)
- If RE is restarted, the next scheduled scan fires after N minutes — no backfill of missed scans

## Alternatives Considered
- OS cron: external dependency, harder to reconfigure at runtime
- Redis-based job queue (RQ/Celery): durable across restarts but adds ops overhead
- Separate scheduler process: unnecessary separation for a single interval job
