# Data Sources in AIA-Forecaster-CC: Complete Reference

## Overview

The system uses a **dual-registry architecture** for data ingestion:

1. **Data Source Registry** (`registry.py`) — "passive" sources that fetch structured data for a currency pair (RSS feeds, BIS speeches, custom connectors). These run once at the start of each agent's research phase.
2. **Web Search Provider Registry** (`web_providers.py`) — "active" search backends that the LLM queries iteratively during its agentic search loop (DuckDuckGo, Brave, custom APIs).

Both registries share the same extension mechanism: a `company/` folder that is auto-imported at startup.

| Layer | Registry Module | Decorator | Default Sources | Purpose |
|-------|----------------|-----------|-----------------|---------|
| Passive data sources | `search/registry.py` | `@data_source("name")` | `rss`, `bis_speeches` | Background evidence gathering |
| Web search providers | `search/web_providers.py` | `@web_search_provider("name")` | `duckduckgo` | Iterative LLM-directed search |

---

## Part 1: The Data Source Registry (Passive Sources)

### File: `src/aia_forecaster/search/registry.py`

### How Registration Works

A data source is any async function with this signature:

```python
async def my_source(pair: str, cutoff_date: date, **kwargs) -> list[SearchResult]:
```

Registration happens two ways:

**1. Decorator (preferred):**
```python
from aia_forecaster.search.registry import data_source

@data_source("my_feed")
async def fetch_my_feed(pair: str, cutoff_date, **kwargs) -> list[SearchResult]:
    ...
    return [SearchResult(...)]
```

**2. Imperative:**
```python
from aia_forecaster.search.registry import register

register("my_feed", fetch_my_feed)
```

Both store the function in a singleton `_Registry` instance at module level. The decorator also stamps `fn._data_source_name = name` on the function for introspection.

### Internal Registry Class (`_Registry`)

| Method | Behavior |
|--------|----------|
| `register(name, fn)` | Adds to `_sources` dict. Warns if name already exists (overwrites). |
| `unregister(name)` | Removes from `_sources` (silent no-op if missing). |
| `names` | Property: returns `list[str]` of all registered source names. |
| `get(name)` | Returns the function or `None`. |
| `all()` | Returns a copy of the full `dict[str, DataSourceFn]`. |

### Built-in Sources Loaded at Startup

The `_load_builtins()` function runs lazily on first call to `list_sources()` or `fetch_all()`. It imports:

```python
import aia_forecaster.search.rss       # triggers @data_source("rss")
import aia_forecaster.search.bis       # triggers @data_source("bis_speeches")
import company.search                  # triggers any company @data_source() decorators
```

The `company.search` import is wrapped in a bare `ImportError` catch — if no `company/` package exists, it silently passes. Other exceptions log a warning.

### How Agents Consume Data Sources

In `agents/forecaster.py`, the `ForecastingAgent.forecast()` and `ForecastingAgent.research()` methods call:

```python
source_results = await fetch_all_sources(
    pair=question.pair,
    cutoff_date=question.cutoff_date,
    source_names=self.source_names,  # None = all sources
    max_age_hours=48,
    max_results=10,
)
```

`fetch_all()` in `registry.py`:
1. Calls `_load_builtins()` to ensure all sources are registered
2. Filters to `source_names` if provided, otherwise uses all
3. Runs all source functions **in parallel** via `asyncio.gather`
4. Wraps each call in `_safe_fetch()` which inspects the function's signature to only pass kwargs it accepts
5. Returns `dict[str, list[SearchResult]]` mapping source name → results
6. Individual source failures are caught and logged — they return empty lists, never crash the pipeline

### `source_names` Filtering

The `ForecastingAgent` constructor accepts `source_names: list[str] | None`. When set, only those named sources are fetched:

```python
ForecastingAgent(agent_id=0, source_names=["rss"])  # Only RSS, no BIS
```

This is threaded from `SourceConfig.registry_sources` via `EnsembleEngine._create_agents()`.

---

## Part 2: Built-in Data Source — RSS (`rss`)

### File: `src/aia_forecaster/search/rss.py`

### Feed Configuration

Each feed is a `FeedConfig(url, category, currencies)` dataclass:

```python
FeedConfig("https://www.federalreserve.gov/feeds/press_all.xml", "central_bank", ["USD"])
```

The `currencies` field scopes which pairs trigger this feed. Empty `currencies` means "all pairs."

### Default Feeds (25 total)

| Category | Count | Example Feeds |
|----------|-------|---------------|
| `central_bank` | 8 | Fed, ECB, BOJ, BOE, RBA, RBNZ, SNB, BOC |
| `fx_specific` | 4 | FXStreet, ForexLive, DailyFX, Investing.com |
| `macro_data` | 3 | BLS (USD), BEA (USD), Eurostat (EUR) |
| `commodity` | 1 | OilPrice.com (CAD, NOK, AUD) |
| `geopolitical` | 1 | WTO |
| `general` | 8 | Reuters, BBC Business, CNBC, Japan Times (JPY), Kyodo News (JPY), The Guardian (GBP), SMH (AUD) |

### Feed Selection Per Pair

`_feeds_for_pair(pair)` filters feeds:
- Feeds with empty `currencies` list → always included
- Feeds with `currencies` containing either the base or quote currency → included

Example: For `USDJPY`, feeds tagged `["USD"]`, `["JPY"]`, or `[]` (general) are fetched. Feeds tagged `["GBP"]` only are excluded.

### Keyword Filtering

After fetching, each RSS entry is filtered against `CURRENCY_KEYWORDS` for the pair's currencies plus `GENERAL_FX_KEYWORDS`. Only entries whose title+summary contain at least one matching keyword are kept.

**Currency keywords** (10 currencies mapped):
- `JPY`: jpy, yen, japan, boj, bank of japan, ueda, japanese, nikkei, tokyo
- `USD`: usd, dollar, fed, federal reserve, fomc, powell, treasury, nonfarm, payroll, cpi, pce
- `EUR`: eur, euro, ecb, european central bank, lagarde, eurozone, eurostat, german
- (+ GBP, CHF, AUD, CAD, NZD, NOK, SEK)

**General FX keywords** (30): forex, fx, exchange rate, currency, carry trade, risk-on, risk-off, safe haven, interest rate, inflation, gdp, employment, trade balance, central bank, monetary policy, rate hike, rate cut, hawkish, dovish, quantitative, yield, rate decision, forward guidance, tariff, pmi, bond yield, vix, retail sales, current account

### Temporal Filtering

Entries older than `max_age_hours` (default: 48) are dropped based on `published_parsed`.

### Deduplication

SHA-256 hash of `"{title}:{link}"`, truncated to 16 hex chars. Duplicates across feeds are dropped.

### Feed Health Tracking

`_record_feed_result(url, success)` tracks per-feed health in `_feed_health` dict:
- On success: records timestamp, resets failure count
- On failure: increments failure count
- Accessible via `get_feed_health()` for diagnostics

### Extension Points for RSS

```python
from aia_forecaster.search.rss import register_feed, register_feeds, register_currency_keywords, FeedConfig

# Add a single custom feed
register_feed(FeedConfig("https://internal.example.com/fx-rss", "fx_specific", ["USD", "EUR"]))

# Add multiple feeds
register_feeds([FeedConfig(...), FeedConfig(...)])

# Add keywords for a new or existing currency
register_currency_keywords("CNH", ["renminbi", "pboc", "yuan", "china"])
```

These functions append to the module-level `FX_FEEDS` and `CURRENCY_KEYWORDS` lists/dicts — no restart needed, but they must run before the RSS source is fetched.

---

## Part 3: Built-in Data Source — BIS Speeches (`bis_speeches`)

### File: `src/aia_forecaster/search/bis.py`

### What It Does

Fetches and parses the BIS central bank speeches RSS feed (`https://www.bis.org/doclist/cbspeeches.rss`), which uses RSS 1.0 (RDF) format with a custom `cb:` namespace.

### Currency Extraction

Two-level approach to determine which currency a speech relates to:

**Level 1 — Institution mapping** (22 institutions mapped):
- G10 central banks: Fed → USD, BOE → GBP, BOJ → JPY, ECB → EUR, RBA → AUD, RBNZ → NZD, BOC → CAD, SNB → CHF, Riksbank → SEK, Norges Bank → NOK
- Eurozone national central banks: Bundesbank, Banque de France, Banca d'Italia, Banco de España, etc. → all map to EUR

**Level 2 — Speaker surname fallback** (20 speakers mapped):
- Powell, Jefferson, Waller → USD
- Bailey, Broadbent → GBP
- Ueda → JPY
- Lagarde, de Guindos → EUR
- etc.

### Pair Matching

A speech matches a pair if:
1. Its extracted currency is one of the pair's two currencies, OR
2. The title+description contain currency-specific keywords (NOT general FX keywords — this is intentionally more restrictive than RSS filtering to avoid false positives from generic central banking discussions)

### Defaults

- `max_age_hours`: 336 (14 days) — speeches are higher-signal, longer-lived than news
- `max_results`: 15
- `_FETCH_TIMEOUT`: 15 seconds

### XML Parsing Details

The `cb:speech` element contains:
- `cb:simpleTitle` — clean title
- `cb:occurrenceDate` — when the speech was given (preferred over `dc:date`)
- `cb:person/cb:surname` — speaker surname
- `cb:resource/cb:resourceLink` — PDF URL (used as the result URL when available)

---

## Part 4: Web Search Provider Registry (Active Sources)

### File: `src/aia_forecaster/search/web_providers.py`

### How It Differs from Data Sources

| Aspect | Data Source Registry | Web Search Provider Registry |
|--------|---------------------|------------------------------|
| When called | Once per research phase (background) | Per-iteration in agentic search loop |
| Who decides queries | Hardcoded (pair + cutoff) | LLM generates queries dynamically |
| Parallelism | All sources fetched in parallel | All **active** providers queried per search |
| Filtering | Source handles its own relevance | Shared blacklist + relevance filter post-hoc |
| Default | `bis_speeches` (RSS opt-in via `--sources rss`) | `duckduckgo` |

### Registration

```python
from aia_forecaster.search.web_providers import web_search_provider

@web_search_provider("brave")
async def search_brave(query: str, max_results: int = 10, cutoff_date: date | None = None) -> list[SearchResult]:
    ...
```

### Active vs. Registered Providers

- **Registered**: All providers whose `@web_search_provider()` decorator has run
- **Active**: The subset currently in use — controlled by `_active` list (default: `["duckduckgo"]`)

Key functions:
```python
set_web_provider("brave")                    # Single provider
set_web_providers(["duckduckgo", "brave"])    # Multiple — queried in parallel
get_web_providers()                           # Returns active list
list_web_providers()                          # Returns all registered (available)
```

`set_web_providers()` validates that all names are registered — raises `ValueError` if not.

### Dispatch (`search_web()`)

The main entry point called by agents:

1. Calls `_load_providers()` (imports `aia_forecaster.search.web` for DuckDuckGo, then `company.search` for company providers)
2. Resolves active provider functions
3. Fans out to **all active providers in parallel** via `asyncio.gather`
4. Merges results, **deduplicates by URL** (first occurrence wins, case-insensitive, trailing-slash-stripped)
5. Applies **shared blacklist filtering** (BLACKLISTED_DOMAINS + IRRELEVANT_DOMAINS)

### Blacklist System

**IRRELEVANT_DOMAINS** (hardcoded, 11 entries): Calculator, converter, and date utility sites that are never relevant to FX analysis.

**BLACKLISTED_DOMAINS** (user-extensible, starts empty):
```python
from aia_forecaster.search.web_providers import add_blacklisted_domains
add_blacklisted_domains(["internal-wiki.example.com"])
```

**Prediction markets are intentionally NOT blacklisted** — they provide valuable probability signals.

---

## Part 5: Built-in Web Provider — DuckDuckGo

### File: `src/aia_forecaster/search/web.py`

### Query Sanitization

Before sending to DDG, queries are cleaned:
- Strip `site:` operators
- Strip `before:/after:` date operators
- Strip boolean operators (AND/OR/NOT → spaces)
- Strip parentheses and quotation marks
- Collapse whitespace
- Truncate to 300 chars at word boundary

### Time Limit Mapping

Cutoff date is mapped to DDG's `timelimit` parameter:
- ≤1 day → `"d"` (past day)
- ≤7 days → `"w"` (past week)
- ≤31 days → `"m"` (past month)
- >31 days → `"y"` (past year)
- Future cutoff → `None` (no restriction)

### Implementation Notes

- Uses `duckduckgo_search.DDGS` synchronous client (called from async context)
- Requests `max_results + 5` from DDG, truncates to `max_results` (over-fetches to handle filtering)
- Blacklist filtering is NOT applied here — it happens in the `search_web()` dispatch layer

---

## Part 6: Relevance Filtering

### File: `src/aia_forecaster/search/relevance.py`

Applied post-hoc to results from both passive sources and web search. Controlled by:
```python
# config.py
relevance_filtering_enabled: bool = True
relevance_threshold: float = 0.20
```

### Scoring Rubric (0.0–1.0)

**Positive signals:**
| Signal | Score |
|--------|-------|
| Direct pair mention in title (e.g. "USD/JPY") | +0.40 |
| Direct pair mention in snippet only | +0.25 |
| Both base AND quote currency keywords present | +0.25 |
| Only one currency keyword present | +0.15 |
| Per general FX keyword hit (max +0.15) | +0.02 each |
| Source is pair-specific central bank URL | +0.10 |

**Negative signals:**
| Signal | Score |
|--------|-------|
| Different pair prominently in title | -0.20 |
| Unrelated asset class in title (gold, crypto, etc.) | -0.15 |

**Commodity exemptions**: For pairs involving commodity currencies (AUD, CAD, NOK, NZD, ZAR, CLP), mentions of their respective commodities (gold, oil, dairy, copper, etc.) are NOT penalized.

---

## Part 7: The `company/` Folder System

### Purpose

The `company/` folder is the **extension mechanism** for adding proprietary data sources, custom LLM backends, consensus providers, and pair configurations without modifying upstream code.

### Activation: How `company/` Gets Loaded

There is **no explicit activation step**. The system auto-discovers it:

1. **`registry.py` `_load_builtins()`** runs `import company.search` (triggered by any call to `list_sources()` or `fetch_all()`)
2. **`web_providers.py` `_load_providers()`** also runs `import company.search` (triggered by `search_web()` or `list_web_providers()`)
3. **`company/__init__.py`** runs on first import, executing all registration calls

Both registries import `company.search` independently, but Python's import system ensures `company/__init__.py` and `company/search/__init__.py` each only execute once.

### Setting Up a `company/` Folder

```bash
# Copy the template
cp -r company.example company

# The .gitignore ignores `internal/` but NOT `company/`
# Add `company/` to .gitignore if desired, or commit to your fork
```

### `company/__init__.py` — The Entry Point

This file runs on `import company` and orchestrates all registrations:

```python
# 1. Register custom currency pairs
from company.pairs import register_custom_pairs
register_custom_pairs()

# 2. Import search subpackage (triggers @data_source and @web_search_provider decorators)
import company.search  # noqa: F401

# 3. Register consensus provider
from company.consensus import get_consensus
from aia_forecaster.fx import set_consensus_provider
set_consensus_provider(get_consensus)

# 4. (Optional) Register custom LLM backend
# from company.llm import register_llm_connector
# register_llm_connector()
```

### `company/search/__init__.py` — Data Source Activation

This is where you **enable or disable** data sources and web search providers. By default in the template, everything is commented out:

```python
# Uncomment as you add data source modules:
# import company.search.bloomberg  # noqa: F401

# Uncomment to register additional web search providers:
# import company.search.brave  # noqa: F401
```

**To enable a data source:** Uncomment the import line. The `@data_source()` or `@web_search_provider()` decorator runs on import, registering it in the global registry.

**To disable a data source:** Comment out or remove the import line.

**To disable a built-in source:** Call `unregister()` from `company/__init__.py`:
```python
from aia_forecaster.search.registry import unregister
unregister("rss")           # Disable RSS feeds
unregister("bis_speeches")  # Disable BIS speeches
```

### `company/search/bloomberg.py` — Example Data Source

Template for a proprietary data source:

```python
@data_source("bloomberg")
async def fetch_bloomberg(pair: str, cutoff_date: date | None = None, **kwargs) -> list[SearchResult]:
    api_key = os.environ.get("BLOOMBERG_API_KEY")
    if not api_key:
        return []
    # TODO: Replace with actual Bloomberg API calls
    return []
```

Key pattern: returns empty list when API key is missing — graceful degradation.

### `company/search/brave.py` — Example Web Search Provider

Full working implementation of a Brave Search provider:

```python
@web_search_provider("brave")
async def search_brave(query: str, max_results: int = 10, cutoff_date: date | None = None) -> list[SearchResult]:
    api_key = os.environ.get("BRAVE_SEARCH_API_KEY", "")
    if not api_key:
        return []
    # Full implementation with freshness filter, httpx client, etc.
```

**To activate Brave alongside DuckDuckGo** (both queried in parallel):
1. Uncomment `import company.search.brave` in `company/search/__init__.py`
2. Set `BRAVE_SEARCH_API_KEY` env var
3. Call `set_web_providers(["duckduckgo", "brave"])` or use CLI: `--web-provider duckduckgo,brave`

### `company/config.py` — Configuration Overrides

Handles company-specific env vars and blacklist additions:

```python
BLOOMBERG_API_KEY = os.environ.get("BLOOMBERG_API_KEY", "")

EXTRA_BLACKLISTED_DOMAINS = [
    # "internal-wiki.example.com",
]

if EXTRA_BLACKLISTED_DOMAINS:
    from aia_forecaster.search.web import add_blacklisted_domains
    add_blacklisted_domains(EXTRA_BLACKLISTED_DOMAINS)
```

### `company/pairs.py` — Custom Currency Pairs

```python
from aia_forecaster.fx.pairs import PairConfig, register_pair
from aia_forecaster.search.rss import register_currency_keywords

def register_custom_pairs():
    register_pair(PairConfig(pair="USDCNH", base="USD", quote="CNH", ...))
    register_currency_keywords("CNH", ["renminbi", "yuan", "pboc", "china"])
```

### `company/consensus.py` — Consensus Provider

Plugs in a directional forecast to replace the forward rate as the distribution center:

```python
def get_consensus(pair: str, spot: float, tenor: str) -> tuple[float, str] | None:
    # Return (rate, source_label) or None to fall back to forward
    return None
```

A working demo with hardcoded forecasts is in `company/consensus_sample.py`. To use it, change the import in `company/__init__.py`:
```python
from company.consensus_sample import get_consensus  # instead of company.consensus
```

### `company/llm.py` — Custom LLM Backend

Examples for Azure OpenAI, Anthropic Claude, and Ollama (local). Uncomment one and call from `company/__init__.py`.

---

## Part 8: SourceConfig and the `--sources` CLI Flag

### Model: `models.py` — `SourceConfig`

```python
class SourceConfig(BaseModel):
    registry_sources: list[str] = ["rss", "bis_speeches"]   # Passive sources
    web_search_enabled: bool = True                          # Active web search
    web_provider: str = "duckduckgo"                         # Which web backend
```

`SourceConfig` derives a `SearchMode` via `get_search_mode()`:
- Both registry sources AND web search → `HYBRID`
- Only registry sources → `RSS_ONLY`
- Only web search → `WEB_ONLY`
- Neither → `HYBRID` (fallback, agents get no data)

### CLI: `--sources` Flag

```bash
forecast USDJPY --sources rss           # RSS only, no web, no BIS
forecast USDJPY --sources rss,bis       # RSS + BIS, no web search
forecast USDJPY --sources rss,web       # RSS + web search, no BIS
forecast USDJPY --sources rss,bis,web   # All sources (RSS is opt-in)
forecast USDJPY --sources web           # Web search only, no passive sources
```

The parser (`_parse_source_config()` in `main.py`) maps CLI tokens to registry names:
- `rss` → `"rss"`
- `bis` → `"bis_speeches"`
- `web` → enables `web_search_enabled` (sentinel, not a registry source)

### CLI: `--web-provider` Flag

```bash
forecast USDJPY --web-provider brave             # Brave instead of DuckDuckGo
forecast USDJPY --web-provider duckduckgo,brave   # Both in parallel
```

Can also be set via env: `WEB_SEARCH_PROVIDER=brave` (loaded from `.env` by `Settings`).

### How SourceConfig Flows Through the Pipeline

1. CLI parses `--sources` → `SourceConfig`
2. `ProbabilitySurfaceGenerator(source_config=...)` passes it to `EnsembleEngine`
3. `EnsembleEngine._create_agents()` derives:
   - `forced_mode = source_config.get_search_mode()` — all agents use this mode
   - `source_names = source_config.registry_sources or None` — only these passive sources
4. Each `ForecastingAgent` receives `source_names` → passes to `fetch_all_sources(source_names=...)`
5. When `source_config is None` (default), agents cycle through `WEB_ONLY → HYBRID` for diversity (RSS is opt-in)

---

## Part 9: Agent Search Mode Diversity

### Default Behavior (no `--sources` flag)

Agents are assigned diverse search modes via round-robin:

| Agent 0 | Agent 1 | Agent 2 | Agent 3 | Agent 4 | Agent 5 | Agent 6 | Agent 7 | Agent 8 | Agent 9 |
|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|
| RSS_ONLY | WEB_ONLY | HYBRID | RSS_ONLY | WEB_ONLY | HYBRID | RSS_ONLY | WEB_ONLY | HYBRID | RSS_ONLY |

Temperature spreads linearly from 0.4 (agent 0) to 1.0 (agent 9).
Search iterations cycle: 3, 4, 5, 6, 7, 3, 4, 5, 6, 7.

### With `--sources` (SourceConfig set)

All agents use the **same** forced search mode and source names. No mode cycling — the run isolates specific data sources for A/B comparison.

### SearchMode Behavior in Agents

| Mode | Passive Sources | Web Search Loop | Tenor-Specific Web Search |
|------|----------------|-----------------|---------------------------|
| `RSS_ONLY` | Yes (72h window, 20 results) | No | No (returns empty brief) |
| `WEB_ONLY` | No | Yes (up to N iterations) | Yes (1-2 iterations) |
| `HYBRID` | Yes (48h window, 10 results) | Yes (up to N iterations) | Yes (1-2 iterations) |

---

## Part 10: Step-by-Step — Enabling a New Data Source

### Scenario: Adding a Reuters News API data source

**1. Create the source file:**

```python
# company/search/reuters.py
from aia_forecaster.search.registry import data_source
from aia_forecaster.models import SearchResult

@data_source("reuters")
async def fetch_reuters(pair: str, cutoff_date, **kwargs) -> list[SearchResult]:
    max_results = kwargs.get("max_results", 15)
    # ... fetch from Reuters API ...
    return [SearchResult(query=f"reuters:{pair}", title=..., snippet=..., url=..., source="reuters")]
```

**2. Enable it in `company/search/__init__.py`:**

```python
import company.search.reuters  # noqa: F401 — triggers @data_source("reuters")
```

**3. That's it.** On next run, `_load_builtins()` → `import company.search` → `import company.search.reuters` → `@data_source("reuters")` registers it → `fetch_all()` includes it alongside RSS and BIS.

### Scenario: Disabling the built-in BIS source

Add to `company/__init__.py` (after `import company.search`):

```python
from aia_forecaster.search.registry import unregister
unregister("bis_speeches")
```

Or use the CLI: `--sources rss,web` (omit `bis`).

### Scenario: Replacing DuckDuckGo with a paid search API

**1. Create provider:**

```python
# company/search/serper.py
from aia_forecaster.search.web_providers import web_search_provider

@web_search_provider("serper")
async def search_serper(query, max_results=10, cutoff_date=None):
    # ... call Serper API ...
    return [SearchResult(...)]
```

**2. Enable in `company/search/__init__.py`:**

```python
import company.search.serper  # noqa: F401
```

**3. Activate via CLI:**

```bash
forecast USDJPY --web-provider serper
# or both:
forecast USDJPY --web-provider duckduckgo,serper
```

**Or set permanently via `.env`:**

```
WEB_SEARCH_PROVIDER=serper
```

---

## Part 11: Upstream Merge Safety

The `company/` folder design ensures zero merge conflicts when pulling upstream changes:

1. Upstream `.gitignore` ignores `internal/` (not `company/`)
2. All company code lives in `company/` — no upstream files are modified
3. The auto-import mechanism (`import company.search`) is in upstream code and stable
4. Registration APIs (`register_pair`, `register_feed`, `data_source`, `set_consensus_provider`, etc.) are the stable contract between upstream and company code

```bash
git fetch upstream
git merge upstream/main   # Always clean — no overlapping files
```

---

## Part 12: Summary — What's Active by Default

### Without a `company/` folder:

| Source | Type | Active | Registration |
|--------|------|--------|-------------|
| RSS (25 feeds) | Passive | Yes | Built-in, `@data_source("rss")` |
| BIS Speeches | Passive | Yes | Built-in, `@data_source("bis_speeches")` |
| DuckDuckGo | Web search | Yes | Built-in, `@web_search_provider("duckduckgo")` |

### With `company.example` copied to `company/` (as-is):

| Source | Type | Active | Registration |
|--------|------|--------|-------------|
| RSS (25 feeds) | Passive | Yes | Built-in |
| BIS Speeches | Passive | Yes | Built-in |
| DuckDuckGo | Web search | Yes | Built-in |
| Bloomberg | Passive | **No** — import commented out in `company/search/__init__.py` | `@data_source("bloomberg")` |
| Brave | Web search | **No** — import commented out in `company/search/__init__.py` | `@web_search_provider("brave")` |
| USDCNH pair | Pair config | **Yes** — registered in `register_custom_pairs()` | `register_pair(...)` |
| Consensus | Provider | **Yes** — but returns `None` (blank stub), so system falls back to forward rate | `set_consensus_provider(...)` |
| Custom LLM | Provider | **No** — commented out in `company/__init__.py` | `set_llm_provider(...)` |

### Key Runtime Behavior

- **No `company/`**: 2 sources active by default (BIS + DuckDuckGo). RSS opt-in via `--sources rss,bis,web`. Agents cycle WEB_ONLY → HYBRID.
- **With `company/` (template)**: Same 2 sources + consensus stub (no-op). Same agent behavior.
- **With `--sources rss`**: Only RSS. All agents forced to `RSS_ONLY` mode.
- **With `--sources web`**: Only DuckDuckGo. All agents forced to `WEB_ONLY` mode.
- **With `--web-provider brave`**: Brave replaces DuckDuckGo (requires `BRAVE_SEARCH_API_KEY`).

---
---

# Consensus Provider: Deep Dive Research Report

## 1. What Is the Consensus Provider?

The consensus provider is a **pluggable hook** that injects an external directional forecast (e.g., analyst surveys, Bloomberg FXFC, internal models) into the probability computation pipeline. When registered, it **replaces the forward rate** as the center of the probability distribution that anchors all forecasts. When absent, the system falls back to using the interest-rate-parity forward as the distribution center.

The key design philosophy: **forward ≠ consensus**. The forward rate is mechanical carry math (`F = S × exp((r_quote − r_base) × T)`), not a directional view. A consensus rate represents what the market or analysts actually *expect* the rate to be.

---

## 2. Registration & Lifecycle

### 2.1 The Provider Interface

Defined in `src/aia_forecaster/fx/base_rates.py`:

```python
ConsensusProvider = Callable[[str, float, "Tenor"], tuple[float, str] | None]
```

The provider receives:
- `pair` (str): e.g. `"USDJPY"`
- `spot` (float): current spot rate
- `tenor` (Tenor): forecast horizon enum

And returns either:
- `(consensus_rate, source_label)` — a point estimate and a human-readable label (e.g. `"bloomberg_survey"`, `"internal_model"`, `"sample_hardcoded"`)
- `None` — signals "no consensus available for this pair/tenor"

### 2.2 Registration Flow

The provider is registered at application startup via the `company/` extension mechanism:

1. `src/aia_forecaster/__init__.py` calls `_load_extensions()` on import
2. This tries to `import company`, which runs `company/__init__.py`
3. `company/__init__.py` imports `get_consensus` from `company/consensus.py` and calls:
   ```python
   from aia_forecaster.fx import set_consensus_provider
   set_consensus_provider(get_consensus)
   ```
4. This stores the callable in the module-level `_consensus_provider` variable

The `set_consensus_provider()` function:
- Accepts a callable or `None` (to clear)
- Stores it in a module-global `_consensus_provider`
- Logs registration/clearing

### 2.3 The company/ Extension System

The `company/` directory is designed as a **source-agnostic** plugin layer:

- **`company/consensus.py`** — The stub implementation. Returns `None` for all pairs (i.e., forward-only mode). Has a `# TODO` to replace with Bloomberg or internal API calls.
- **`company/consensus_sample.py`** — A working demo with hardcoded forecasts for USDJPY, EURUSD, GBPUSD across 1D–6M tenors. Source label is `"sample_hardcoded"`.
- **`company/__init__.py`** — By default imports from `consensus.py` (stub). User can swap to `consensus_sample` to test.

If the `company/` directory doesn't exist (upstream/open-source usage), `_load_extensions()` catches the `ImportError` and silently proceeds — the system runs entirely on forward rates.

---

## 3. How Consensus Is Queried

The `get_consensus()` function in `base_rates.py` (line 72) wraps the provider call with error handling:

```python
def get_consensus(pair, spot, tenor):
    if _consensus_provider is None:
        return None           # No provider registered
    try:
        return _consensus_provider(pair.upper(), spot, tenor)
    except Exception:
        logger.warning(...)   # Log and ignore — fall back to forward
        return None
```

Key behaviors:
- **No provider registered** → returns `None` immediately
- **Provider raises an exception** → logs warning with traceback, returns `None`
- **Provider returns `None`** → passed through as-is

This means the system is **maximally fault-tolerant**: any failure in the consensus path silently degrades to forward-rate mode.

---

## 4. Where Consensus Enters the Math

### 4.1 ABOVE Mode (`compute_base_rates()`, line 406)

The probability math is:
```
d2 = (ln(C/K) − ½σ_t²) / σ_t
P(S_T > K) = Φ(d2)
```
where `C` = center = consensus rate (if available) or forward rate.

Decision logic (line 442):
```python
consensus_result = get_consensus(pair, spot, tenor)
if consensus_result is not None:
    center = consensus_rate          # Use consensus
    forward = None                   # Skip forward computation entirely
else:
    forward = compute_forward_rate(pair, spot, tenor)
    center = forward                 # Fall back to forward
```

**Important optimization**: When consensus is available, the forward rate is **not computed at all**. This saves a Yahoo Finance API call for the USD rate (via `^IRX`). The forward is only computed when no consensus is available.

### 4.2 HITTING Mode (`compute_hitting_base_rate()`, line 545)

Uses the same consensus-or-forward decision. The drift term uses the center:

```python
h = log(barrier / spot)              # Log-distance to barrier
nu_T = log(center / spot) − ½σ_t²   # Drift toward center
```

Then feeds into the first-passage probability formula (`_first_passage_probability()`, line 495). The drift direction matters:
- If center is on the same side of spot as the barrier → drift is **toward** the barrier → higher touch probability
- If center is on the opposite side → drift is **away** → lower touch probability

### 4.3 Return Values

Both functions return a dict with these consensus-related fields:

| Field | With consensus | Without consensus |
|-------|---------------|-------------------|
| `center` | consensus_rate | forward_rate |
| `center_source` | source_label (e.g. `"bloomberg_survey"`) | `"forward"` |
| `consensus_rate` | the rate value | `None` |
| `consensus_source` | the label | `None` |
| `forward_rate` | `None` | computed forward |
| `forward_source` | `None` | e.g. `"USD:dynamic\|JPY:fallback"` |
| `r_base` | `None` | interest rate |
| `r_quote` | `None` | interest rate |

When consensus is used, all forward/rate fields are `None` because the forward was never computed.

---

## 5. How Consensus Reaches the LLM Prompts

### 5.1 The Context Formatting Layer

`format_base_rate_context()` (line 684) in `base_rates.py` is the bridge between math and prompts. It produces a multi-line text block that gets injected into agent prompts. The consensus vs. forward distinction controls what agents see.

#### 5.1.1 ABOVE Mode Output

**With consensus:**
```
BASE RATE CONTEXT (statistical anchor):
Current spot: USD/JPY = 150.00
1M consensus: USD/JPY = 148.00 (src: bloomberg_survey)
Annualized vol: 10.0% (dynamic)
Target: below 149.00 in 1M
  From bloomberg_survey: -1.00 (-0.67%)
  From spot:    -1.00 (-0.67%)
Historical 1M range (1-sigma): +/-2.74 JPY
Required move from bloomberg_survey: 0.36 standard deviations
Statistical base rate: P(below 149.00) = 0.640 (64.0%)
Note: Base rate is anchored to bloomberg_survey (consensus view).
This is a log-normal baseline. Adjust based on current evidence.
```

**Without consensus (forward only):**
```
BASE RATE CONTEXT (statistical anchor):
Current spot: USD/JPY = 150.00
1M forward: USD/JPY = 149.51 (carry: USD 4.50% vs JPY 0.50%, net -4.00%, src: dynamic/fallback)
Annualized vol: 10.0% (dynamic)
Target: below 149.00 in 1M
  From forward: -0.51 (-0.34%)
  From spot:    -1.00 (-0.67%)
...
Note: Base rate is anchored to forward (carry-adjusted).
```

Key differences in what the agent sees:
1. **Line 3**: "consensus" line vs "forward" line — different labels, different rate values
2. **Move calculations**: "From bloomberg_survey" vs "From forward"
3. **Note**: "(consensus view)" vs "(carry-adjusted)"
4. **Forward line shows carry details**: interest rate breakdown, net differential, source provenance. Consensus line shows only the rate and source label — it's opaque.

#### 5.1.2 HITTING Mode Output

Similar structure but with barrier-specific language:

**With consensus:**
```
BASE RATE CONTEXT (statistical anchor — HITTING/BARRIER mode):
...
Expected drift (bloomberg_survey) is toward this barrier.
...
This is a drift-adjusted first-passage baseline anchored to the bloomberg_survey.
Adjust based on current evidence.
```

**Without consensus:**
```
...
Expected drift (forward) is toward this barrier.
...
This is a drift-adjusted first-passage baseline anchored to the forward.
Adjust based on current evidence.
```

The `_format_market_context()` helper function (line 643) handles the conditional rendering:
- If `consensus_rate` and `consensus_source` are both present → render consensus line
- Else if `forward_rate` is present and source isn't `"no_rates"` → render forward line with carry breakdown
- If neither → empty string (no market context line at all)

### 5.2 Prompt Injection Points

The base rate context block appears in **three** agent prompts:

1. **Query Generation Prompt** (`QUERY_GENERATION_PROMPT`, line 42): Shown as `{base_rate_section}` — gives agents context while deciding what to search for. This means consensus influences the agent's *search strategy*, not just its probability output.

2. **Forecast Prompt** (`FORECAST_PROMPT`, line 90): Shown as `{base_rate_section}` after the evidence — the primary anchoring point where agents use base rates to produce their probability estimate.

3. **Batch Pricing Prompt** (`BATCH_PRICING_PROMPT` / `BATCH_PRICING_PROMPT_HITTING`, lines 218/252): Shown as `{base_rates_block}` — used during Phase 2 to price all strikes at a given tenor. Here, `format_base_rate_context()` is called **per strike**, but only the "Statistical base rate" line is extracted into the pricing block (not the full context).

### 5.3 The Batch Pricing Extraction

In `price_tenor()` (line 894, Phase 2), the full context is generated per strike but only the statistical base rate line is passed to the LLM:

```python
for strike in strikes:
    ctx = format_base_rate_context(pair, spot, strike, tenor, forecast_mode)
    for line in ctx.split("\n"):
        if "Statistical base rate" in line:
            base_rates_lines.append(f"  Strike {strike}: {line.strip()}")
            break
```

This means in batch pricing, agents see something like:
```
BASE RATES (statistical anchors):
  Strike 148.00: Statistical base rate: P(above 148.00) = 0.640 (64.0%)
  Strike 150.00: Statistical base rate: P(above 150.00) = 0.480 (48.0%)
  Strike 152.00: Statistical base rate: P(above 152.00) = 0.320 (32.0%)
```

The consensus vs forward distinction is **hidden** at the batch pricing level — agents only see the resulting probability number, not whether it came from a consensus-centered or forward-centered distribution. This is by design: the statistical anchor should influence the agent's calibration without biasing it toward a specific methodology.

---

## 6. Fallback Cascade

The system has a multi-level fallback chain:

```
Consensus Provider
    ├─ Provider registered and returns (rate, label) → USE CONSENSUS as center
    ├─ Provider registered but returns None → FALL THROUGH
    ├─ Provider registered but raises exception → LOG WARNING, FALL THROUGH
    └─ No provider registered → FALL THROUGH
         │
         ▼
Forward Rate (interest-rate parity)
    ├─ Both rates available → F = S × exp((r_q − r_b) × T)
    │   ├─ Dynamic rate (Yahoo Finance ^IRX for USD) → preferred
    │   └─ Static fallback (FALLBACK_POLICY_RATES) → used for non-USD
    ├─ One rate available → use 0.0 for the missing one
    └─ Neither rate available → forward = spot (i.e., no drift)
```

When forward also degrades to spot (no rates), the distribution is centered at the current spot, which means:
- P(above spot) ≈ 0.5 (slightly below due to the −½σ² drift correction)
- The base rate becomes purely volatility-driven

---

## 7. The Supervisor's "Consensus" (Unrelated — Naming Overlap)

There's a naming overlap worth noting. The supervisor agent in `supervisor.py` uses `_build_consensus_causal_summary()` and `_build_consensus_factors()` — these aggregate **causal factors** across multiple forecasting agents (e.g., "7/10 agents identified BOJ policy as bearish"). This is a *consensus among agents*, not the external consensus provider. The two are completely unrelated systems sharing the word "consensus".

Similarly, `CellExplanation.consensus_summary` (in `models.py`, line 377) in the explanation module is about *agent agreement on direction*, not the consensus provider.

---

## 8. Per-Cell vs Per-Pair Granularity

The consensus provider operates at the **(pair, tenor)** level — it returns one rate per pair-tenor combination. However:

- Base rates are computed at the **(pair, spot, strike, tenor)** level
- The consensus rate is the **same** for all strikes at a given tenor
- Only the strike position relative to the consensus changes the probability

This means for a given tenor, all strikes share the same distribution center (consensus rate), but each strike gets a different probability because it's at a different position on the distribution.

For different tenors, the consensus provider is called separately, so it can return different rates (e.g., USDJPY 148.00 for 1M vs 145.00 for 3M), creating a **term structure of consensus expectations**.

---

## 9. What Agents Actually See: Example Walkthrough

### Scenario: USDJPY, spot=150.00, strike=148.00, tenor=1M

**With consensus (148.00, "analyst_survey"):**

The agent prompt includes:
```
BASE RATE CONTEXT (statistical anchor):
Current spot: USD/JPY = 150.00
1M consensus: USD/JPY = 148.00 (src: analyst_survey)
Annualized vol: 10.0% (dynamic)
Target: above 148.00 in 1M
  From analyst_survey: +0.00 (+0.00%)
  From spot:    -2.00 (-1.33%)
Historical 1M range (1-sigma): +/-2.74 JPY
Required move from analyst_survey: 0.00 standard deviations
Statistical base rate: P(above 148.00) = 0.479 (47.9%)
Note: Base rate is anchored to analyst_survey (consensus view).
This is a log-normal baseline. Adjust based on current evidence.
```

Here, the strike equals the consensus, so the move from consensus is 0 — roughly a coin flip adjusted for log-normal drift. The agent sees that the market expects exactly this level.

**Without consensus (forward=149.51):**

```
BASE RATE CONTEXT (statistical anchor):
Current spot: USD/JPY = 150.00
1M forward: USD/JPY = 149.51 (carry: USD 4.50% vs JPY 0.50%, net -4.00%,
  src: dynamic/fallback)
Annualized vol: 10.0% (dynamic)
Target: above 148.00 in 1M
  From forward: +1.51 (+1.01%)
  From spot:    -2.00 (-1.33%)
Historical 1M range (1-sigma): +/-2.74 JPY
Required move from forward: 0.55 standard deviations
Statistical base rate: P(above 148.00) = 0.709 (70.9%)
Note: Base rate is anchored to forward (carry-adjusted).
This is a log-normal baseline. Adjust based on current evidence.
```

The forward is higher than the strike (149.51 vs 148.00), so being above 148.00 is quite likely from the forward's perspective. The agent sees carry-specific detail (rate differential breakdown).

**Key difference**: The consensus-centered distribution gives ~48% probability of being above 148.00, while the forward-centered one gives ~71%. This is a **massive** difference in the statistical anchor the agent receives, which will significantly influence its final probability output.

---

## 10. Platt Scaling Interaction

After agents produce their probabilities (influenced by the consensus-or-forward base rates), the probabilities go through Platt scaling:

```
p_calibrated = p^α / (p^α + (1−p)^α)    where α = √3 ≈ 1.73
```

This pushes probabilities away from 0.5 toward 0 or 1, correcting LLM hedging bias. The consensus provider affects the **input** to this calibration — if consensus shifts the base rate anchor, it shifts the agent's raw probability, which then gets calibrated.

This means consensus has an **amplified** effect: a modest shift in the center (e.g., consensus 148 vs forward 149.5) changes the base rate (48% vs 71%), which the agent uses as an anchor, and then Platt scaling pushes it further from 0.5.

---

## 11. Edge Cases & Gotchas

1. **Partial coverage**: The provider can return consensus for USDJPY 1M but `None` for USDJPY 1D. In this case, 1M cells use consensus as center and 1D cells use forward. Different cells on the same surface can have different center sources.

2. **Forward not computed when consensus exists**: When consensus is available, `r_base`, `r_quote`, and `forward_rate` are all `None` in the return dict. Any downstream code that assumes these exist will break. The current codebase handles this correctly (the market context formatter checks for `None`).

3. **Consensus == spot**: If the consensus rate equals the current spot, the system effectively says "no expected move" — probabilities are symmetrical around spot (modulo the −½σ² drift adjustment).

4. **Stale consensus**: There's no built-in staleness check. If the provider returns a rate that was surveyed weeks ago while the market has moved significantly, the base rate will anchor to an outdated level. The provider is responsible for freshness.

5. **Source label propagation**: The `source_label` string from the provider flows all the way through to the prompt text (`"src: analyst_survey"`) and the `center_source` field in the return dict. It appears in the final formatted output that agents read.

6. **No forward shown alongside consensus**: When consensus is active, the forward is NOT computed and NOT shown to agents. They only see the consensus line. This is different from the original design intent described in the docstring ("The carry-adjusted forward is still computed internally and shown to agents for context") — the current implementation skips the forward entirely for efficiency.

---

## 12. Summary: Data Flow Diagram

```
company/__init__.py
  │
  │  set_consensus_provider(get_consensus)
  ▼
base_rates._consensus_provider   (module-level callable)
  │
  │  get_consensus(pair, spot, tenor)
  ▼
compute_base_rates()  /  compute_hitting_base_rate()
  │
  │  Returns dict with center, center_source, consensus_rate, ...
  ▼
format_base_rate_context()
  │
  │  Produces human-readable text block
  ▼
ForecastingAgent._build_base_rate_section()
  │
  │  {base_rate_section} in prompts
  ├──► QUERY_GENERATION_PROMPT   (influences search strategy)
  ├──► FORECAST_PROMPT           (anchors probability estimate)
  └──► BATCH_PRICING_PROMPT      (statistical base rate per strike)
         │
         │  Agent raw probabilities
         ▼
    Platt scaling → calibrated probabilities
```

---

## 13. Key Files Reference

| File | Role in consensus flow |
|------|----------------------|
| `src/aia_forecaster/fx/base_rates.py` | Core: provider storage, `get_consensus()`, math, formatting |
| `src/aia_forecaster/fx/__init__.py` | Exports `set_consensus_provider`, `get_consensus` |
| `src/aia_forecaster/__init__.py` | Triggers `company/` import at startup |
| `company.example/__init__.py` | Calls `set_consensus_provider(get_consensus)` |
| `company.example/consensus.py` | Stub provider (returns `None`) |
| `company.example/consensus_sample.py` | Demo provider with hardcoded rates |
| `src/aia_forecaster/agents/forecaster.py` | Consumes `format_base_rate_context()` in prompts |
| `src/aia_forecaster/agents/supervisor.py` | Does NOT use consensus provider (uses agent consensus — different concept) |
