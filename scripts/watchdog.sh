#!/bin/bash
# ORACLE Watchdog — checks all services are alive and active.
# Run via systemd timer (every 2 minutes).
# Restarts any service that has stopped or is in a failed state.

SERVICES=(
    oracle-market-scanner
    oracle-signal-ingestion
    oracle-whale-detector
    oracle-osint-fusion
    oracle-reasoning-engine
    oracle-knowledge-base
    oracle-operator-dashboard
)

LOG=/root/stockMarket/logs/watchdog.log
exec >> "$LOG" 2>&1

for svc in "${SERVICES[@]}"; do
    state=$(systemctl is-active "$svc" 2>/dev/null)
    if [[ "$state" != "active" ]]; then
        echo "$(date -Is) WATCHDOG: $svc is '$state' — restarting"
        systemctl restart "$svc"
    fi
done

# Check market-scanner heartbeat: daemon writes oracle:daemon_status to Redis every 5 min.
# If the key is missing or older than 10 minutes, force-restart.
LAST_BEAT=$(redis-cli get oracle:daemon_status 2>/dev/null | python3 -c "
import sys, json, datetime
try:
    d = json.loads(sys.stdin.read())
    ts = d.get('timestamp', '')
    if ts:
        t = datetime.datetime.fromisoformat(ts)
        age = (datetime.datetime.now() - t).total_seconds()
        print(int(age))
    else:
        print(9999)
except Exception:
    print(9999)
" 2>/dev/null)

if [[ -n "$LAST_BEAT" && "$LAST_BEAT" -gt 600 ]]; then
    echo "$(date -Is) WATCHDOG: market-scanner heartbeat stale (${LAST_BEAT}s) — restarting"
    systemctl restart oracle-market-scanner
fi
