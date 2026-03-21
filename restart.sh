#!/bin/bash
# Restart all ORACLE services via systemd.
# Usage: ./restart.sh [service-name]
#   ./restart.sh              — restarts all oracle services
#   ./restart.sh market-scanner — restarts only oracle-market-scanner

set -e

SERVICES=(
    oracle-market-scanner
    oracle-signal-ingestion
    oracle-whale-detector
    oracle-osint-fusion
    oracle-reasoning-engine
    oracle-knowledge-base
    oracle-operator-dashboard
)

if [[ -n "$1" ]]; then
    TARGET="oracle-$1"
    echo "Restarting $TARGET..."
    systemctl restart "$TARGET"
    systemctl status "$TARGET" --no-pager -l
else
    echo "Restarting all ORACLE services..."
    for svc in "${SERVICES[@]}"; do
        echo -n "  $svc ... "
        systemctl restart "$svc" && echo "OK" || echo "FAILED"
    done
    echo ""
    systemctl status "${SERVICES[@]}" --no-pager | grep -E "(● oracle|Active:)"
fi
