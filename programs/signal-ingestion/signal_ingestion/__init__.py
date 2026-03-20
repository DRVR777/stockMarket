"""signal-ingestion (SIL) — polls and streams all data sources, normalizes to canonical Signal objects.

Adapters run concurrently via asyncio. Each adapter publishes Signal objects to the
``oracle:signal`` Redis channel using the schema defined in oracle_shared.contracts.signal.
"""
