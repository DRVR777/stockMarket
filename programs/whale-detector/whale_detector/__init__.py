"""whale-detector (WADE) — Whale & Anomaly Detection Engine.

Subscribes to on-chain signals from signal-ingestion, scores anomalies by
order size / wallet history / timing, maintains the wallet registry in Redis,
and surfaces copy-trade opportunities as AnomalyEvents and OperatorAlerts.
"""
