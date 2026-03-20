# CLAUDE.md — whale-detector

## What Is In This Directory

| Name | Purpose |
|------|---------|
| `whale_detector/` | Python package: entry point, config, and 6 pipeline modules |
| `tests/` | Unit and integration tests (FakeRedis, no server needed) |
| `pyproject.toml` | Package metadata and dependencies |
| `CONTEXT.md` | Task router with build sequence |
| `MANIFEST.md` | Structural envelope and dependency declarations |

## Quick Rules For This Directory

- All pipeline modules follow the same pattern: class with async methods
- Subscribe to `Signal.CHANNEL`, publish to `AnomalyEvent.CHANNEL` and `OperatorAlert.CHANNEL`
- All config values come from `whale_detector.config` — no magic numbers in pipeline code
- Use `uv` for package management
- Log every inference to `../../_meta/gaps/pending.txt`

## Cross-References

- Signal contract: `../../oracle-shared/oracle_shared/contracts/signal.py`
- AnomalyEvent contract: `../../oracle-shared/oracle_shared/contracts/anomaly_event.py`
- WalletProfile contract: `../../oracle-shared/oracle_shared/contracts/wallet_profile.py`
- OperatorAlert contract: `../../oracle-shared/oracle_shared/contracts/operator_alert.py`
- ADRs: `../../_planning/adr/` (014, 015, 010, 019)
