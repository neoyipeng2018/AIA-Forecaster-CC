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
