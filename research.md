# Data Sources Research Report

## Overview

The AIA-Forecaster-CC system uses **7 distinct data source categories** to produce FX probability forecasts. Every single one can be disabled via CLI flags, and most can be replaced or augmented through the `company/` folder override mechanism. The architecture follows a decorator-based registry pattern where data sources self-register on import, and a `company/` Python package (if present) is auto-discovered at startup to inject proprietary replacements.

---

## 1. Complete Data Source Inventory

### 1.1 Web Search (Agentic)

| Property | Value |
|----------|-------|
| **Source file** | `src/aia_forecaster/search/web.py` |
| **Registry type** | Web search provider (`@web_search_provider("duckduckgo")`) |
| **Default provider** | DuckDuckGo via `duckduckgo-search` library (v7.0+) |
| **API key required** | No |
| **Cost** | Free |
| **CLI control** | `--sources web` enables it; omitting `web` disables it |
| **Provider swap** | `--web-provider brave` or `--web-provider duckduckgo,brave` |
| **Can be disabled** | Yes — omit `web` from `--sources` |
| **Can be replaced** | Yes — register a `@web_search_provider("custom")` in `company/search/` |

**How it works:**
- Agents generate search queries via LLM in an iterative loop (up to `MAX_SEARCH_ITERATIONS`, default 5).
- Each query is sanitized: `site:`, `before:`/`after:` operators, boolean `AND`/`OR`/`NOT`, parentheses, and quotes are stripped. Max 300 chars.
- DuckDuckGo's `timelimit` parameter filters by recency: `d` (day), `w` (week), `m` (month), `y` (year) — derived from `cutoff_date`.
- Results are merged across all active providers, deduplicated by URL, and filtered through a shared domain blacklist.
- The blacklist includes 11 irrelevant utility domains (calculator.net, timeanddate.com, etc.). Prediction markets (Polymarket, Metaculus, Kalshi, Manifold) are explicitly **not** blacklisted — they're valuable probability signals.

**Company override examples:**
- `company.example/search/brave.py` — fully functional Brave Search provider using `https://api.search.brave.com/res/v1/web/search`. Requires `BRAVE_SEARCH_API_KEY` env var. Supports freshness filtering (`pd`/`pw`/`pm`/`py`).
- Additional blacklisted domains can be added via `add_blacklisted_domains()` in `company/config.py`.

---

### 1.2 RSS Feed Aggregation

| Property | Value |
|----------|-------|
| **Source file** | `src/aia_forecaster/search/rss.py` |
| **Registry type** | Data source (`@data_source("rss")`) |
| **Feed count** | 24 built-in feeds across 7 categories |
| **API key required** | No |
| **Cost** | Free |
| **CLI control** | `--sources rss` enables it; not included by default |
| **Can be disabled** | Yes — omit `rss` from `--sources` (it is off by default) |
| **Can be replaced** | Yes — `register_feed()` and `register_feeds()` add custom feeds |

**Default is OFF.** The default `--sources` value is `bis,web` (BIS speeches + web search). RSS must be explicitly opted into via `--sources rss,bis,web`.

**Built-in feed catalog (24 feeds):**

**Central Banks (8 feeds):**
| Feed | URL | Currencies |
|------|-----|-----------|
| Federal Reserve | `federalreserve.gov/feeds/press_all.xml` | USD |
| ECB | `ecb.europa.eu/rss/press.html` | EUR |
| Bank of Japan | `boj.or.jp/en/rss/whatsnew.xml` | JPY |
| Bank of England | `bankofengland.co.uk/rss/news` | GBP |
| RBA | `rba.gov.au/rss/rss-cb-media-releases.xml` | AUD |
| RBNZ | `rbnz.govt.nz/rss/news.xml` | NZD |
| SNB | `snb.ch/en/mmr/reference/rss_en/source/rss_en.en.xml` | CHF |
| Bank of Canada | `bankofcanada.ca/content_type/press-releases/feed/` | CAD |

**FX-Specific News (4 feeds):**
| Feed | URL | Currencies |
|------|-----|-----------|
| FXStreet | `fxstreet.com/rss` | All |
| Forexlive | `forexlive.com/feed/` | All |
| DailyFX | `dailyfx.com/feeds/all` | All |
| Investing.com | `investing.com/rss/news_14.rss` | All |

**Macro Data (3 feeds):**
| Feed | URL | Currencies |
|------|-----|-----------|
| BLS | `bls.gov/feed/bls_latest.rss` | USD |
| BEA | `bea.gov/news/feed` | USD |
| Eurostat | `ec.europa.eu/eurostat/web/main/news/euro-indicators/feed` | EUR |

**Commodity (1 feed):**
| Feed | URL | Currencies |
|------|-----|-----------|
| OilPrice | `oilprice.com/rss/main` | CAD, NOK, AUD |

**Geopolitical (1 feed):**
| Feed | URL | Currencies |
|------|-----|-----------|
| WTO | `wto.org/english/news_e/news_e.rss` | All |

**General Financial (3 feeds):**
| Feed | URL | Currencies |
|------|-----|-----------|
| Reuters | `feeds.reuters.com/reuters/businessNews` | All |
| BBC Business | `feeds.bbci.co.uk/news/business/rss.xml` | All |
| CNBC | `cnbc.com/id/100727362/device/rss/rss.html` | All |

**Regional (4 feeds):**
| Feed | URL | Currencies |
|------|-----|-----------|
| Japan Times | `japantimes.co.jp/feed/` | JPY |
| Kyodo News | `english.kyodonews.net/rss/all.xml` | JPY |
| Guardian | `theguardian.com/business/rss` | GBP |
| SMH | `smh.com.au/rss/business.xml` | AUD |

**Feed filtering logic:**
- Only feeds relevant to the pair's currencies are fetched (pair-specific filtering via `_feeds_for_pair()`).
- Headlines are keyword-matched against ~135 FX keywords + currency-specific keywords (e.g., "yen", "boj", "ueda" for JPY).
- Entries are deduplicated by SHA-256 hash of `title:link`.
- Temporal filter: entries older than `max_age_hours` (default 48h) are dropped.
- Feed health tracking: in-memory `_feed_health` dict records successes/failures per URL (diagnostic only, doesn't prevent retries).
- Timeout: 10 seconds per feed.

**Extension points:**
- `register_feed(FeedConfig(...))` — add one custom feed.
- `register_feeds([...])` — add multiple feeds at once.
- `register_currency_keywords("CNH", ["renminbi", "pboc", ...])` — extend keyword matching for new currencies.

---

### 1.3 BIS Central Bank Speeches

| Property | Value |
|----------|-------|
| **Source file** | `src/aia_forecaster/search/bis.py` |
| **Registry type** | Data source (`@data_source("bis_speeches")`) |
| **Feed URL** | `https://www.bis.org/doclist/cbspeeches.rss` (RSS 1.0/RDF format) |
| **API key required** | No |
| **Cost** | Free |
| **CLI control** | `--sources bis` enables it (included by default) |
| **Can be disabled** | Yes — omit `bis` from `--sources` |
| **Can be replaced** | Yes — `unregister("bis_speeches")` then register a new source |

**How it works:**
- Fetches the BIS speeches RSS feed (RDF/RSS 1.0 with `cb:` namespace).
- Parses structured XML with namespaces for `rdf:`, `rss:`, `dc:`, and `cb:` (CentralBankWiki spec).
- Extracts the actual central bank from `<description>` text, not `cb:institutionAbbrev` (which is always "BIS").
- Maps institutions to currencies via `INSTITUTION_CURRENCY_MAP` (22 entries covering G10 central banks + Eurozone national central banks).
- Fallback: speaker surname matching via `KNOWN_SPEAKERS` dict (20 entries: Powell→USD, Lagarde→EUR, Ueda→JPY, Bailey→GBP, etc.).
- Two-level pair relevance: Level 1 is direct currency match from institution extraction; Level 2 is keyword matching on description text.
- Default lookback: 336 hours (7 days), max 15 results.
- Timeout: 15 seconds.
- Extracts PDF links from `cb:resourceLink` for direct speech document access.

---

### 1.4 FX Spot Rates

| Property | Value |
|----------|-------|
| **Source file** | `src/aia_forecaster/fx/rates.py` |
| **Type** | Price data (not a registry source) |
| **Primary API** | `https://api.exchangerate.host/latest?base={BASE}&symbols={QUOTE}` |
| **Fallback API** | `https://open.er-api.com/v6/latest/{BASE}` |
| **API key required** | No (both free) |
| **Cost** | Free |
| **Can be disabled** | No — spot rate is always required |
| **Can be replaced** | Not directly pluggable; would require modifying `rates.py` |

**How it works:**
- Attempts exchangerate.host first, falls back to open.er-api.com on failure.
- Both return JSON with a `rates` object keyed by currency code.
- Timeout: 10 seconds each.
- Raises `RuntimeError` if both APIs fail.

**Limitation:** No company override hook exists for spot rates. To use Bloomberg or internal spot rates, you'd need to modify `rates.py` or wrap `get_spot_rate()`.

---

### 1.5 Volatility Data — REMOVED

Volatility data and Yahoo Finance dependency have been removed. The base rate system no longer computes probabilities — it provides plain market context (spot, strike distances, consensus) to agents, who estimate probabilities themselves from evidence.

---

### 1.6 Consensus Provider (Directional View)

| Property | Value |
|----------|-------|
| **Source file** | `src/aia_forecaster/fx/base_rates.py` |
| **Type** | Pluggable pricing input |
| **Registration** | `set_consensus_provider(fn)` |
| **Default** | None (falls back to spot = zero drift) |
| **Can be disabled** | Yes — simply don't register a provider (default behavior) |
| **Can be replaced** | Yes — this is the primary company override point |

**How it works:**
- A registered callable receives `(pair, spot, tenor)` and returns `(consensus_rate, source_label)` or `None`.
- When set, the consensus rate replaces spot as the center of the probability distribution.
- When absent or when the provider returns None/raises, spot is used with zero drift.
- The source label (e.g., "bloomberg_survey", "internal_model") appears in the agent's base rate context block.
- Error handling: any exception from the provider is caught and logged; system falls back to spot.

**Company override:**
- `company.example/consensus.py` — stub returning `None` (template for implementation).
- `company.example/consensus_sample.py` — working example with hardcoded forecasts for USDJPY, EURUSD, GBPUSD across 5 tenors. Source label: `"sample_hardcoded"`.
- To switch: change the import in `company/__init__.py` from `company.consensus` to `company.consensus_sample`.

---

### 1.7 LLM Provider

| Property | Value |
|----------|-------|
| **Source file** | `src/aia_forecaster/llm/client.py` |
| **Type** | Inference backend |
| **Default** | `ChatOpenAI` (langchain-openai) with model from `LLM_MODEL` env var |
| **Fallback** | Cerebras (`https://api.cerebras.ai/v1`) if `CEREBRAS_API_KEY` is set |
| **Registration** | `set_llm_provider(factory)` |
| **Can be disabled** | No — LLM is required |
| **Can be replaced** | Yes — via `set_llm_provider()` in `company/llm.py` |

**How it works:**
- Strips provider prefix from model name (e.g., `"openai/gpt-5-mini"` → `"gpt-5-mini"`).
- Reasoning models (`o1`, `o3`, `gpt-5` prefixes) get a minimum of 2048 `max_tokens` to accommodate hidden reasoning tokens.
- On primary failure, automatically tries Cerebras if API key is configured.
- All LLM calls go through `LLMClient.complete()` (text) or `LLMClient.complete_json()` (parsed JSON).

**Company override examples:**
- Azure OpenAI: `AzureChatOpenAI(azure_deployment=..., api_version=...)`
- Anthropic: `ChatAnthropic(model_name=..., temperature=..., max_tokens=...)`
- Ollama (local): `ChatOllama(model=..., temperature=..., num_predict=...)`

---

## 2. Relevance Filtering (Post-Processing Layer)

### 2.1 Heuristic Filter

| Property | Value |
|----------|-------|
| **Source file** | `src/aia_forecaster/search/relevance.py` |
| **Config key** | `relevance_filtering_enabled` (default: `True`) |
| **Threshold** | `relevance_threshold` (default: `0.10`) |

**Scoring rubric (0.0–1.0):**
- +0.40: Direct pair mention in title (e.g., "USD/JPY" or "USDJPY")
- +0.25: Direct pair mention in snippet only
- +0.25: Both base AND quote currency keywords present
- +0.15: Only one currency keyword present
- +0.02: Per general FX keyword hit (max +0.15 total, ~35 keywords)
- +0.10: Source is a pair-relevant central bank URL
- -0.20: Different pair prominently mentioned in title
- -0.15: Unrelated asset class in title (gold, crypto, stocks — unless commodity-currency exemption applies)

**Commodity-currency exemptions:** AUD (gold, iron ore, coal, copper, mining), CAD (oil, crude, WTI, brent, energy, natural gas), NOK (oil, crude, brent, energy, natural gas), NZD (dairy, milk, agriculture), ZAR (gold, platinum, mining), CLP (copper, mining).

### 2.2 LLM Relevance Filter (Two-Tier)

| Property | Value |
|----------|-------|
| **Source file** | `src/aia_forecaster/search/llm_relevance.py` |
| **Config key** | `llm_relevance_enabled` (default: `True`) |
| **Batch size** | 10 results per LLM call |

**How it works:**
1. Heuristic pre-filter runs first (threshold 0.10 — very permissive).
2. Survivors are batched (10 per call) and sent to the LLM as an FX research analyst prompt.
3. LLM decides "keep" or "drop" for each result with a reason.
4. Tenor-aware: includes guidance on what's actionable for short-term (1D-2W), medium-term (1M-3M), and long-term (6M+) horizons.
5. **Fails open:** if the LLM call fails, all results are kept.
6. Temperature: 0.0 for consistency.

---

## 3. Configuration & Control Surface

### 3.1 Environment Variables (`.env`)

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_MODEL` | `gpt-4o` | LLM model string (accepts `provider/model-name`) |
| `OPENAI_API_KEY` | (empty) | OpenAI API key |
| `CEREBRAS_API_KEY` | (empty) | Cerebras fallback API key |
| `CEREBRAS_MODEL` | `llama-4-scout-17b-16e-instruct` | Cerebras model name |
| `NUM_AGENTS` | `10` | Number of parallel forecasting agents |
| `MAX_SEARCH_ITERATIONS` | `5` | Max agentic search iterations per agent |
| `PLATT_ALPHA` | `1.732...` (sqrt(3)) | Platt scaling coefficient |
| `DEFAULT_PAIR` | `USDJPY` | Default currency pair |
| `DB_PATH` | `data/forecasts/forecasts.db` | SQLite database path |
| `FORECAST_MODE` | `hitting` | Default forecast mode |
| `RELEVANCE_THRESHOLD` | `0.10` | Heuristic relevance filter threshold |
| `RELEVANCE_FILTERING_ENABLED` | `True` | Enable/disable heuristic filter |
| `LLM_RELEVANCE_ENABLED` | `True` | Enable/disable LLM two-tier filter |
| `WEB_SEARCH_PROVIDER` | `duckduckgo` | Default web search provider |
| `BLOOMBERG_API_KEY` | (empty) | For company Bloomberg extension |
| `BRAVE_SEARCH_API_KEY` | (empty) | For company Brave Search extension |

### 3.2 CLI Flags

| Flag | Effect |
|------|--------|
| `--sources rss,bis,web` | Enable specific data sources. Default: `bis,web`. Options: `rss`, `bis`, `web`. |
| `--sources rss` | RSS only (no web search, no BIS). Sets `SearchMode.RSS_ONLY`. |
| `--sources web` | Web search only (no RSS, no BIS). Sets `SearchMode.WEB_ONLY`. |
| `--sources bis,web` | Default behavior: BIS speeches + agentic web search. |
| `--sources rss,bis,web` | All sources active. Sets `SearchMode.HYBRID`. |
| `--web-provider brave` | Swap DuckDuckGo for Brave Search. |
| `--web-provider duckduckgo,brave` | Query both providers in parallel, merge and deduplicate results. |

### 3.3 SourceConfig Model

Defined in `src/aia_forecaster/models.py`:

```python
class SourceConfig(BaseModel):
    registry_sources: list[str] = ["bis_speeches"]   # passive sources
    web_search_enabled: bool = True                   # agentic search toggle
    web_provider: str = "duckduckgo"                  # web search backend(s)
```

Derives `SearchMode` automatically:
- Both registry sources AND web → `HYBRID`
- Registry sources only → `RSS_ONLY`
- Web only → `WEB_ONLY`
- Neither → falls back to `HYBRID` (agents get no data)

---

## 4. Company Override Architecture

### 4.1 Auto-Discovery Mechanism

At startup, `src/aia_forecaster/__init__.py` calls `_load_extensions()`:
```python
def _load_extensions():
    try:
        import company  # triggers company/__init__.py
    except ImportError:
        pass  # no company package — running upstream
```

Similarly, `registry.py._load_builtins()` and `web_providers.py._load_providers()` both attempt `import company.search` after loading built-in sources.

### 4.2 Expected `company/` Directory Structure

```
company/
├── __init__.py              # Main entry: registers pairs, consensus, imports search
├── config.py                # API keys, blacklist extensions
├── pairs.py                 # Custom currency pair registration
├── consensus.py             # Consensus provider (stub or implementation)
├── consensus_sample.py      # Working example with hardcoded data
├── llm.py                   # Custom LLM backend (Azure, Anthropic, Ollama)
└── search/
    ├── __init__.py           # Import data source modules to trigger decorators
    ├── bloomberg.py          # @data_source("bloomberg") — proprietary news
    └── brave.py              # @web_search_provider("brave") — alternative search
```

### 4.3 What Each Override Does

| Override | File | Registration API | What it replaces |
|----------|------|-----------------|------------------|
| Custom pairs | `company/pairs.py` | `register_pair(PairConfig(...))` | Adds new currency pairs (exotics, NDFs) |
| Currency keywords | `company/pairs.py` | `register_currency_keywords("CNH", [...])` | Extends RSS/BIS keyword matching for new currencies |
| Custom RSS feeds | `company/search/*.py` | `register_feed(FeedConfig(...))` | Adds proprietary news feeds to RSS source |
| Custom data source | `company/search/*.py` | `@data_source("name")` | Adds entirely new data sources (Bloomberg, Refinitiv, etc.) |
| Custom web provider | `company/search/*.py` | `@web_search_provider("name")` | Adds/replaces web search backends (Brave, Bing, etc.) |
| Consensus forecasts | `company/consensus.py` | `set_consensus_provider(fn)` | Provides directional FX forecasts as distribution center |
| LLM backend | `company/llm.py` | `set_llm_provider(factory)` | Replaces ChatOpenAI with Azure, Anthropic, Ollama, etc. |
| Domain blacklist | `company/config.py` | `add_blacklisted_domains([...])` | Blocks additional domains from search results |

### 4.4 Activation Order

1. Poetry runs `forecast` → `aia_forecaster.main:main()`
2. `aia_forecaster/__init__.py._load_extensions()` → `import company`
3. `company/__init__.py` executes:
   - `register_custom_pairs()` (custom pairs + keywords)
   - `import company.search` (triggers `@data_source` and `@web_search_provider` decorators)
   - `set_consensus_provider(get_consensus)` (from `company/consensus.py`)
   - Optionally: `register_llm_connector()` (from `company/llm.py`)
4. `registry._load_builtins()` imports `rss`, `bis`, then `company.search` (idempotent)
5. `web_providers._load_providers()` imports `web` (DuckDuckGo), then `company.search` (idempotent)
6. CLI flags override any remaining config (model, agents, pair, web-provider, sources)

---

## 5. Disableability Summary

| Source | Default State | Disable Method | Impact of Disabling |
|--------|:---:|---|---|
| **DuckDuckGo web search** | ON | `--sources bis` (omit `web`) | No agentic search — agents only see passive data |
| **RSS feeds** | OFF | Already off; enable with `--sources rss,...` | N/A |
| **BIS speeches** | ON | `--sources web` (omit `bis`) | No central bank speech data |
| **Spot rates** | ON | Cannot disable | System cannot function without spot |
| ~~Yahoo Finance vol~~ | REMOVED | N/A — volatility computation deleted | N/A |
| **Consensus provider** | OFF | Already off; enable via `company/` | N/A |
| **LLM** | ON | Cannot disable | System cannot function without LLM |
| **Heuristic relevance filter** | ON | `RELEVANCE_FILTERING_ENABLED=false` | More noise in agent evidence |
| **LLM relevance filter** | ON | `LLM_RELEVANCE_ENABLED=false` | Saves LLM calls, more noise |
| **Cerebras fallback** | ON (if key set) | Remove `CEREBRAS_API_KEY` from `.env` | No fallback on primary LLM failure |

**Fully disableable (5):** Web search, RSS feeds, BIS speeches, heuristic filter, LLM relevance filter.
**Replaceable via company/ (6):** Web search provider, data sources, consensus, LLM, RSS feeds, currency pairs.
**Not pluggable (1):** Spot rates — no `set_*_provider()` hook.

---

## 6. Fallback Chains

### LLM
1. Primary: `LLM_MODEL` via `ChatOpenAI` (or custom `set_llm_provider()` factory)
2. Cerebras: `https://api.cerebras.ai/v1` with `CEREBRAS_MODEL` (auto-triggered on primary failure)
3. Hard fail if both unavailable

### Spot Rate
1. `api.exchangerate.host/latest`
2. `open.er-api.com/v6/latest/{BASE}`
3. `RuntimeError` if both fail

### Consensus
1. Registered `_consensus_provider` callable
2. Spot rate (zero drift) — always available

### Web Search
1. Active providers queried in parallel (default: DuckDuckGo only)
2. Each provider fails independently (returns empty list on error)
3. Results merged and deduplicated

### RSS/BIS
1. Each feed/source fetched independently
2. Individual feed failures logged and skipped (fail-open)
3. Empty list returned if all feeds fail

---

## 7. Agent Search Execution Flow

Each agent goes through two phases depending on its `SearchMode`:

### Phase 1: Passive Evidence (RSS_ONLY or HYBRID)
- `fetch_all_sources(pair, cutoff_date, source_names, max_age_hours, max_results)` called
- Fans out to all active registry sources (e.g., `rss`, `bis_speeches`) in parallel via `asyncio.gather()`
- Returns `dict[source_name → list[SearchResult]]`
- Relevance filtering applied to results

### Phase 2: Agentic Search Loop (WEB_ONLY or HYBRID)
For each iteration (up to `MAX_SEARCH_ITERATIONS`):
1. **Query generation:** LLM receives the question, base rate context, and evidence gathered so far. Generates ONE short search query (<150 chars, plain keywords).
2. **Query execution:** `search_web(query, max_results=5, cutoff_date=...)` fans out to all active web providers in parallel.
3. **Relevance filter:** Heuristic first-pass → optional LLM second-pass.
4. **Sufficiency check:** LLM decides "SEARCH" (continue) or "FORECAST" (ready to estimate).

### Agent Diversity (in Ensemble)
The ensemble engine creates 10 agents with deliberate diversity:
- Search modes cycle: `WEB_ONLY` → `HYBRID`
- Temperatures spread: 0.4–1.0
- Max iterations vary: 3–7
- All agents run in parallel via `asyncio.gather()`

---

## 8. What's NOT Pluggable (Gaps)

| Component | Current State | What would be needed |
|-----------|---------------|---------------------|
| **Spot rates** | Hardcoded API chain in `rates.py` | Add `set_spot_provider(fn)` hook |
| ~~Volatility~~ | REMOVED — no longer needed | N/A |
| **Relevance keywords** | Hardcoded in `rss.py` + `relevance.py` | Already partially pluggable via `register_currency_keywords()` |
| **Agent prompts** | Hardcoded in `forecaster.py` | No override mechanism |
| **Calibration alpha** | Env var only (`PLATT_ALPHA`) | Already configurable |

---

## 9. Data Flow Diagram

```
CLI (--sources, --web-provider)
  │
  ├── SourceConfig
  │     ├── registry_sources: ["bis_speeches"]     ← passive data sources
  │     ├── web_search_enabled: true                ← agentic search toggle
  │     └── web_provider: "duckduckgo"              ← which search backend
  │
  ├── Registry (registry.py)
  │     ├── "rss"          → fetch_fx_news()        ← 24 RSS feeds, pair-filtered
  │     ├── "bis_speeches" → fetch_bis_speeches()   ← BIS central bank feed
  │     └── "bloomberg"    → fetch_bloomberg()       ← company extension (example)
  │
  ├── Web Providers (web_providers.py)
  │     ├── "duckduckgo"   → search_duckduckgo()    ← built-in, free
  │     └── "brave"        → search_brave()          ← company extension (example)
  │
  ├── FX Data (not in registry)
  │     ├── Spot rates     → get_spot_rate()         ← exchangerate.host + fallback
  │     └── Consensus      → get_consensus()         ← pluggable via company/
  │
  └── Filtering (applied to all search results)
        ├── Heuristic      → score_relevance()       ← keyword-based, 0.0–1.0
        └── LLM            → filter_relevant_llm()   ← two-tier, batched, fail-open
```

---

## 10. Practical Usage Scenarios

### Run with all default sources (BIS + web search)
```bash
forecast USDJPY 2026-03-15
```

### Run with all sources including RSS
```bash
forecast USDJPY 2026-03-15 --sources rss,bis,web
```

### Run RSS-only (no web search)
```bash
forecast USDJPY 2026-03-15 --sources rss,bis
```

### Run web-only (no passive sources)
```bash
forecast USDJPY 2026-03-15 --sources web
```

### Use Brave Search instead of DuckDuckGo
```bash
forecast USDJPY 2026-03-15 --web-provider brave
```

### Query both DuckDuckGo and Brave in parallel
```bash
forecast USDJPY 2026-03-15 --web-provider duckduckgo,brave
```

### A/B compare source configurations
```bash
forecast USDJPY 2026-03-15 --sources bis,web
forecast USDJPY 2026-03-15 --sources rss,bis,web
forecast compare data/forecasts/USDJPY_2026-03-15_above_bis_speeches+web.json \
                 data/forecasts/USDJPY_2026-03-15_above_bis_speeches+rss+web.json
```
