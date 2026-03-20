# MANIFEST — adapters

## Envelope

| Field       | Value                    |
|-------------|--------------------------|
| name        | adapters                 |
| type        | adapter-collection       |
| depth       | 6                        |
| status      | active                   |
| path        | programs/oracle/programs/signal-ingestion/signal_ingestion/adapters |

## Purpose

One Python module per external data source. Each adapter follows the shared interface
(`start`, `stop`, `_normalize`, `_publish`) and emits canonical Signal objects to
the `oracle:signal` Redis channel.

## Contents

| Module | Adapter | Source | Category |
|--------|---------|--------|----------|
| `polymarket_rest.py` | PolymarketRESTAdapter | Polymarket CLOB REST API | price |
| `polymarket_ws.py` | PolymarketWSAdapter | Polymarket CLOB WebSocket | price |
| `polygon_onchain.py` | PolygonOnchainAdapter | Alchemy Polygon WS (OrderFilled) | on_chain |
| `newsapi_adapter.py` | NewsAPIAdapter | NewsAPI /v2/everything | news |
| `wikipedia_adapter.py` | WikipediaAdapter | MediaWiki Recent Changes API | news |
| `reddit_adapter.py` | RedditAdapter | Reddit OAuth2 /hot endpoints | social |
| `birdeye_ws.py` | BirdeyeWSAdapter | Birdeye WS + REST fallback | price |
| `ai_opinion.py` | AIOpinionAdapter | RE request/reply via Redis | ai_generated |

## Needs

- `oracle_shared.contracts.signal` — Signal, SignalCategory, SourceId
- Redis async client (shared connection pool)
- External API credentials per adapter (see `.env.example`)

## Returns

- Signal objects published to `Signal.CHANNEL` (`oracle:signal`)
