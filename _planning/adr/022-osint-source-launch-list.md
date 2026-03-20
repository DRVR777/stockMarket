# ADR-022: OSINT Source Registry at Launch

## Status
accepted

## Context
SIL must poll a configurable set of OSINT sources. PRD lists many possible sources but does not specify which are included at launch. Scope must be bounded to ship a working v1.

## Decision
**Four sources at launch:**

1. **NewsAPI** (`newsapi.org/v2/everything`)
   - Queries: derived from active Polymarket market questions (keywords extracted by OSFE)
   - Poll interval: every 5 minutes per market cluster
   - API key: `NEWSAPI_API_KEY` env var
   - Free tier: 100 requests/day — upgrade to Developer ($449/mo) for production volume

2. **Wikipedia Recent Changes** (`api.wikimedia.org/feed/v1/wikipedia/en/featured/...`)
   - Monitors recent edits to pages matching active market topics
   - Poll interval: every 15 minutes
   - No API key required

3. **Reddit API** (`oauth.reddit.com`) — subreddits: r/Polymarket, r/PredictIt, r/politics, r/sports
   - Hot posts + new posts sorted by recency
   - Poll interval: every 10 minutes
   - Credentials: `REDDIT_CLIENT_ID`, `REDDIT_CLIENT_SECRET` env vars (script-type app)

4. **Polymarket market descriptions** (from SIL's own Polymarket REST polling)
   - Market question text is used as the seed embedding for each market in ChromaDB
   - No additional API call needed — SIL already polls the market list

**Post-launch additions (Phase 2+):** FEC filings API, FDA drug approval calendar, sports result APIs (The Odds API), broader social velocity tracking.

The source registry is a configurable list — adding a new source requires adding one SIL adapter class and one entry in `oracle:state:params:osint_sources`. No other program changes.

## Consequences
- SIL implements four adapter classes at launch: `NewsAPIAdapter`, `WikipediaAdapter`, `RedditAdapter`, `PolymarketDescriptionAdapter`
- NewsAPI free tier may be a bottleneck — monitor request counts and consider upgrading early
- Reddit rate limits: 60 requests/minute for OAuth app — well within bounds for 4 subreddits at 10-min intervals

## Alternatives Considered
- Starting with fewer sources (just NewsAPI): faster to build, weaker signal coverage
- Including government APIs at launch: adds significant complexity per agency (FEC, FDA each have different schemas)
