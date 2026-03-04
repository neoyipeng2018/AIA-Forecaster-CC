# AIA-Forecaster-CC: Deep Technical Research Report

## Executive Summary

AIA-Forecaster-CC is an FX probability surface forecaster that uses LLM ensembles to answer questions like *"What is the probability that USD/JPY will be above 155.0 in 1 month?"*. The output is a grid of calibrated probabilities across multiple strikes (price levels) and tenors (time horizons) -- comparable to what you'd derive from an options market, but driven by news and macro analysis rather than market pricing.

The system adapts four techniques from the AIA Forecaster paper (Alur, Stadie et al., Bridgewater AIA Labs, arXiv:2511.07678, Nov 2025):

1. **Agentic, adaptive search** -- LLM agents control their own query strategy
2. **Multi-agent ensembling** -- 10 independent agents with built-in diversity
3. **Statistical calibration** -- Platt scaling corrects LLM hedging bias
4. **Foreknowledge bias mitigation** -- temporal cutoffs and prediction market blacklists

---

## 1. System Architecture

### 1.1 Three-Phase Pipeline

The system processes a currency pair through three distinct phases:

```
Phase 1: RESEARCH (M agents in parallel)
  Each agent independently searches news via an agentic loop:
  generate query -> search -> evaluate -> decide "search more" or "ready" -> repeat
  Output: M ResearchBriefs with themes, causal factors, and evidence

Phase 1.5: TENOR-SPECIFIC RESEARCH (optional, M agents x T tenors)
  Lightweight follow-up: each agent researches tenor-specific catalysts
  Output: per-(agent, tenor) TenorResearchBrief

Phase 2: PRICING (M agents x T tenors)
  Each agent prices ALL strikes for each tenor in a single LLM call,
  using its own research + statistical base rates as anchors
  Output: Raw probability grid (M x S x T)

Phase 3: AGGREGATION & CALIBRATION
  Per-cell mean of M agent estimates
  -> Supervisor reviews surface for anomalies
  -> PAVA monotonicity enforcement
  -> Platt scaling calibration (alpha = sqrt(3))
  -> Final monotonicity pass
  Output: heatmap, 3D surface, JSON, PDF
```

### 1.2 Why Two Phases?

The shared-research approach avoids redundant search across cells. For a 5x5 grid:

| Approach | Formula | LLM Calls |
|----------|---------|-----------|
| Naive per-cell | 10 agents x 5 strikes x 5 tenors x ~12 calls/cell | ~3,000 |
| Shared research | 10 agents x ~7 research + 10 x 5 pricing + 1 supervisor | ~120 |

This is a **~94% reduction** in LLM calls.

### 1.3 File Structure

```
src/aia_forecaster/
  __init__.py           -- Package init, loads company extensions
  config.py             -- Settings from .env (pydantic-settings)
  models.py             -- All data models (Tenor, ForecastQuestion, CausalFactor, etc.)
  main.py               -- CLI entry point ("forecast" command)

  agents/
    forecaster.py       -- Individual forecasting agent (research + pricing)
    supervisor.py       -- Disagreement reconciliation + surface review

  ensemble/
    engine.py           -- Orchestrates M parallel agents

  calibration/
    platt.py            -- Platt scaling (fixes LLM hedging bias)
    monotonicity.py     -- PAVA algorithm (enforces logical constraints)

  fx/
    base_rates.py       -- Forward rates, consensus, vol, statistical anchors
    rates.py            -- Spot rate fetching (exchangerate.host + fallback)
    pairs.py            -- Currency pair configs + strike generation
    surface.py          -- ProbabilitySurfaceGenerator (main orchestrator)
    explanation.py      -- Evidence extraction from agent outputs (no LLM)
    compare.py          -- Compare 2+ surfaces visually
    pdf_report.py       -- PDF report generation

  llm/
    client.py           -- LLMClient (langchain-openai, pluggable provider)

  search/
    registry.py         -- @data_source decorator, pluggable source framework
    rss.py              -- 22+ curated RSS feeds (central banks, FX, macro)
    bis.py              -- BIS speeches (central bank communications)
    web.py              -- DuckDuckGo search implementation
    web_providers.py    -- Pluggable web search dispatch + blacklist
    relevance.py        -- Heuristic relevance scoring (no LLM cost)
    foreknowledge.py    -- Foreknowledge bias detection (LLM-as-judge)

  evaluation/
    metrics.py          -- Brier score + decomposition

  storage/
    cache.py            -- File-based search cache (SHA256 keys, TTL)
    database.py         -- SQLite for persisting forecast runs

company.example/        -- Template for company-specific extensions
tests/                  -- 10 test files, ~150+ test cases
data/
  cache/                -- ~190 cached search result JSON files
  forecasts/            -- Output: PNG, HTML, JSON, PDF per run
```

---

## 2. Data Models (`models.py`)

The entire pipeline is typed with Pydantic. Key models:

### 2.1 Tenor

Custom `str` subclass supporting arbitrary time horizons: `1D`, `3D`, `5D`, `2W`, `1M`, `3M`, `6M`, `1Y`, etc. Pattern: `<integer><unit>` where unit is D/W/M/Y.

Properties:
- `days` -- calendar days (D=1, W=7, M=30, Y=365)
- `trading_days` -- approx trading days (252/365 ratio)
- `label` -- human-readable ("1 day", "3 months")

Predefined constants: `Tenor.D1`, `Tenor.W1`, `Tenor.M1`, `Tenor.M3`, `Tenor.M6`, `Tenor.Y1`.

### 2.2 CausalFactor

Structured causal reasoning linking events to FX impacts:

```python
CausalFactor:
  event: str       # "BOJ signals rate hike in March"
  channel: str     # "rate differential narrowing"
  direction: str   # "bearish"
  magnitude: str   # "strong" / "moderate" / "weak"
  confidence: str  # "high" / "medium" / "low"
```

This is the backbone of the system's reasoning transparency -- agents don't just output probabilities; they explain *why* through structured causal chains.

### 2.3 ForecastMode

Two distinct probability semantics:
- **ABOVE**: P(price > strike at expiry) -- terminal distribution
- **HITTING**: P(price touches strike within tenor) -- first-passage / barrier touch

### 2.4 SourceConfig

Controls which data sources are active. Has `registry_sources` (e.g., ["rss", "bis_speeches"]), `web_search_enabled`, and `web_provider`. Generates filename suffixes like "rss+web" or "rss+bis+brave" for output provenance.

### 2.5 Key Pipeline Models

| Model | Purpose |
|-------|---------|
| `ForecastQuestion` | Binary question (pair, spot, strike, tenor, cutoff_date) |
| `ResearchBrief` | Phase 1 output: themes, causal factors, evidence, macro summary |
| `TenorResearchBrief` | Phase 1.5 output: tenor-specific catalysts and evidence |
| `BatchPricingResult` | One agent's prices for all strikes in a single tenor |
| `SurfaceCell` | One cell in the probability grid (strike x tenor -> probability) |
| `ProbabilitySurface` | The full output grid with metadata, causal factors, regime |
| `EnsembleResult` | Aggregated agent forecasts + supervisor reconciliation |
| `CalibratedForecast` | Post-Platt-scaling result (raw + calibrated + alpha) |
| `ForecastRun` | Complete pipeline run record for persistence |

---

## 3. Search Layer (`search/`)

### 3.1 Data Source Registry (`registry.py`)

A decorator-based pluggable system. Any async function returning `list[SearchResult]` can be registered:

```python
@data_source("my_source")
async def fetch_data(pair: str, cutoff_date, **kwargs) -> list[SearchResult]:
    ...
```

All registered sources run in parallel via `asyncio.gather()` with error isolation -- one failing source won't break others. Smart parameter inspection: only passes kwargs the function's signature accepts.

Built-in sources auto-loaded: `rss`, `bis_speeches`. Company extensions auto-discovered from `company.search`.

### 3.2 RSS Feeds (`rss.py`)

22+ curated feeds organized by category:
- **Central banks**: Fed, ECB, BOJ, BOE, RBA, RBNZ, SNB, BoC
- **FX-specific**: FXStreet, ForexLive, DailyFX, Investing.com
- **Macro data**: BLS, BEA, Eurostat
- **Commodity/energy**: OilPrice (for AUD/CAD/NOK)
- **Geopolitical**: WTO
- **General**: Reuters, BBC, CNBC, Japan Times, Kyodo, Guardian

Feed selection is currency-pair-aware: only feeds relevant to the pair's base/quote currencies are fetched. Headline matching uses keyword sets per currency (e.g., JPY: "jpy", "yen", "japan", "boj", "ueda", "nikkei"; USD: "usd", "dollar", "fed", "fomc", "powell", "nonfarm", "cpi").

Feed health tracking records success/failure per URL for diagnostics.

### 3.3 BIS Speeches (`bis.py`)

Fetches Bank for International Settlements central bank speeches. High-signal, low-frequency source.

Currency extraction uses two-tier resolution:
1. **Institution name** matching against 35 known central banks (high precision)
2. **Speaker surname** fallback against 20 known governors (lower precision)

Pair matching: direct institutional currency match first, then keyword-based.

Default max age is 336 hours (14 days) vs RSS's 48 hours -- speeches are less frequent but more impactful.

### 3.4 Web Search (`web.py` + `web_providers.py`)

DuckDuckGo implementation with safety guardrails:

**Blacklisted domains** (prevent foreknowledge leakage):
- Prediction markets: polymarket.com, metaculus.com, manifold.markets, kalshi.com, predictit.org, smarkets.com

**Filtered domains** (noise reduction):
- Utility sites: calculator.net, timeanddate.com, convertunits.com, epochconverter.com

**Query sanitization**: Strips advanced operators (site:, AND/OR, parentheses, quotes) that DDG doesn't support well. Truncates to 300 chars at word boundary.

**Temporal filtering**: Cutoff date mapped to DDG timelimit: d (day), w (week), m (month), y (year).

**Multi-provider architecture**: `web_providers.py` supports multiple web search backends running in parallel. Results are deduplicated by URL and filtered through the shared blacklist. Company extensions can register Brave, Google, etc.

### 3.5 Relevance Scoring (`relevance.py`)

Fast heuristic scoring (0.0-1.0, no LLM cost):

| Signal | Score |
|--------|-------|
| Pair in title (e.g., "EUR/USD") | +0.40 |
| Pair in snippet only | +0.25 |
| Both currencies mentioned | +0.25 |
| One currency keyword | +0.15 |
| Per general FX keyword (capped) | +0.02 each, max +0.15 |
| Central bank source for the pair | +0.10 |
| Different pair in title | -0.20 |
| Unrelated asset class | -0.15 |

Special commodity currency exemptions: AUD/gold/iron/copper, CAD/oil/crude, NOK/oil, NZD/dairy, ZAR/gold/platinum.

Default threshold: 0.20 (permissive -- catches most relevant articles while filtering blatant off-topic).

### 3.6 Foreknowledge Detection (`foreknowledge.py`)

LLM-as-a-judge pipeline that checks whether search results contain post-cutoff information. Detects:
1. Events/data from after the cutoff date
2. Past-tense language about future events
3. Specific data points only knowable post-cutoff

Uses low temperature (0.1) for consistency. Returns `FlaggedResult` with `has_foreknowledge`, `confidence`, and `evidence`.

The paper found ~1.65% of search results contain foreknowledge bias.

---

## 4. Forecasting Agent (`agents/forecaster.py`)

### 4.1 Architecture

Each `ForecastingAgent` supports three workflows:
1. **Binary forecasting** (`forecast()`) -- single question -> probability
2. **Pair-level research** (`research()`) -- Phase 1 shared research
3. **Tenor-specific research** (`research_tenor()`) -- Phase 1.5 lightweight
4. **Batch pricing** (`price_tenor()`) -- Phase 2 strike-grid pricing

### 4.2 Prompt Templates (10 total)

The agent uses carefully engineered prompts at each stage:

**Research phase:**
- `RESEARCH_QUERY_PROMPT` -- generate search queries for pair-level outlook
- `RESEARCH_ASSESS_PROMPT` -- decide SEARCH or DONE
- `RESEARCH_SUMMARY_PROMPT` -- synthesize into macro brief with structured causal factors

**Tenor research phase:**
- `TENOR_RESEARCH_QUERY_PROMPT` -- tenor-specific search (horizon-aware: SHORT/MEDIUM/LONG focus areas)
- `TENOR_RESEARCH_SUMMARY_PROMPT` -- extract tenor-specific causal factors (same event can be bullish short-term but bearish long-term)

**Pricing phase:**
- `BATCH_PRICING_PROMPT` -- price all strikes at a tenor (ABOVE mode)
- `BATCH_PRICING_PROMPT_HITTING` -- price barrier/touch probabilities (HITTING mode)

**Single question (non-surface):**
- `QUERY_GENERATION_PROMPT` -- generate search query
- `ASSESS_PROMPT` -- decide SEARCH or FORECAST
- `FORECAST_PROMPT` -- produce final probability

### 4.3 Agentic Search Loop

The research workflow for each agent:

1. **RSS/passive phase**: Fetch RSS feeds (if HYBRID or RSS_ONLY), apply relevance filter, limit to 10-20 results
2. **Web search loop** (if HYBRID or WEB_ONLY): For 3-7 iterations:
   - LLM generates a search query (conditioned on evidence so far)
   - Execute search via DuckDuckGo
   - Filter by relevance
   - LLM assesses sufficiency: SEARCH (more needed) or DONE (ready)
3. **Summarize**: LLM produces ResearchBrief with key_themes, causal_factors, macro_summary

### 4.4 Batch Pricing

Each agent prices all strikes for one tenor in a single LLM call:

1. Build base rates block (statistical anchor per strike from `base_rates.py`)
2. Merge pair-level + tenor-specific evidence (deduplicated by URL)
3. Build causal factors block (pair-level + tenor-specific)
4. LLM produces probabilities for ALL strikes at once
5. Prompt enforces monotonicity constraints and instructs "Do NOT hedge toward 0.5"

### 4.5 Built-in Diversity

Agents are differentiated along three dimensions:

| Dimension | Range | Purpose |
|-----------|-------|---------|
| Temperature | 0.4 -> 1.0 | Exploration vs precision |
| Search depth | 3 -> 7 iterations | Thoroughness variation |
| Search mode | RSS_ONLY -> WEB_ONLY -> HYBRID (cycling) | Source diversity |

This prevents groupthink and reduces correlation among agent estimates.

---

## 5. Supervisor Agent (`agents/supervisor.py`)

### 5.1 Key Insight

The paper found that **naive LLM aggregation** (asking an LLM to evaluate all forecasts and pick the best) performs **worse** than simple averaging. The supervisor's value comes from resolving *specific disagreements*, not holistic re-evaluation.

### 5.2 Regime Detection

Classifies the current macro environment into one of:
- **risk_on**: Carry trades active, EM/high-yield strengthening, VIX low
- **risk_off**: Safe-haven flows into USD/JPY/CHF, VIX elevated
- **policy_divergence**: Central bank rate differentials, forward guidance gaps
- **carry_unwind**: Sharp reversal in funding currencies, position liquidation
- **mixed**: No single regime dominates

Each regime maps to channel weights for reweighting causal factors:
```
risk_on:           carry trade (1.5x), risk appetite (1.5x), safe-haven (0.5x)
risk_off:          safe-haven demand (1.5x), deleveraging (1.5x), carry trade (0.5x)
policy_divergence: rate differential (1.5x), yield spread (1.5x), safe-haven (0.7x)
carry_unwind:      positioning (1.5x), forced liquidation (1.5x), rate diff (0.5x)
mixed:             no reweighting
```

### 5.3 Disagreement Analysis

When agent spread > 0.10, the supervisor:

1. **Identifies disagreement type**:
   - **Factual**: Agents disagree on what happened (e.g., BOJ hike vs hold)
   - **Channel**: Same event, different transmission (e.g., oil spike -> risk-off OR inflation)
   - **Magnitude**: Same event/channel, different size estimate
   - **Missing factor**: Some agents found a factor others missed

2. **Targeted search**: Runs specific queries to resolve each disagreement point

3. **Reconciliation**: Produces reconciled forecast with HIGH/MEDIUM/LOW confidence
   - HIGH: overrides the simple mean
   - MEDIUM/LOW: falls back to mean (conservative approach)

### 5.4 Surface Review

The supervisor also reviews the full probability grid for anomalies:
- Non-monotonicity violations
- Tenor inconsistency (longer tenors should show more regression to 0.5)
- Implausible values inconsistent with evidence
- Causal factor mismatches
- Temporal mismatches (fast factors driving long tenors)

Only HIGH confidence adjustments are applied.

---

## 6. Base Rate System (`fx/base_rates.py`)

### 6.1 Purpose

Statistical anchors so LLM agents don't start from scratch. Each forecasting agent receives a "BASE RATE CONTEXT" block showing:
- Current spot rate
- Forward/consensus rate with interest-rate breakdown
- Annualized volatility
- Target strike with move required
- 1-sigma expected range
- Statistical base rate with z-score

### 6.2 Forward Rate (Interest Rate Parity)

```
Forward = Spot x exp((r_quote - r_base) x T_years)
```

**Critical design decision**: Forward is pure carry math (interest-rate differential), NOT a directional view. It reflects funding costs, not where the market thinks the price will go.

Interest rate resolution:
1. Dynamic: Yahoo Finance `^IRX` (13-week T-bill, USD only)
2. Static fallback: `FALLBACK_POLICY_RATES` (USD 4.50%, JPY 0.50%, EUR 2.75%, GBP 4.25%, etc.)
3. Cache: 4-hour TTL for rates

### 6.3 Consensus Provider (Pluggable)

```python
set_consensus_provider(fn)  # (pair, spot, tenor) -> (rate, source_label) | None
```

When registered, consensus replaces forward as the distribution center. Forward is still computed and shown to agents for carry context. Falls back to forward when provider returns None.

Source label (e.g., "analyst_survey", "bloomberg") appears in agent context.

### 6.4 Volatility

- Dynamic: 60-day realized vol from Yahoo Finance (annualized via sqrt(252))
- Fallback: Static per pair (USDJPY 10%, EURUSD 8%, etc.)
- Cache: 1-hour TTL
- Sanity bounds: rejects < 1% or > 50%

### 6.5 ABOVE Mode Probability

```
sigma_t = annual_vol x sqrt(trading_days / 252)
d2 = (ln(Center / K) - 0.5 x sigma_t^2) / sigma_t
P(S_T > K) = Phi(d2)    [Phi = standard normal CDF]
```

### 6.6 HITTING Mode Probability (First-Passage)

For barrier above spot (h > 0):
```
d1 = (nu_T - h) / sigma_t
d2 = (-nu_T - h) / sigma_t
P(touch) = Phi(d1) + exp(2 * nu_T * h / sigma_t^2) * Phi(d2)
```

Where:
- `h = ln(barrier / spot)` (log-distance)
- `nu_T = ln(forward / spot) - 0.5 * sigma_t^2` (total log-space drift)

Key properties: P(touch) >= P(above) always; P(touch) at spot = 1.0; decreases with distance from spot in both directions.

Overflow protection: clamps exponent to [-500, 500] to avoid numerical issues.

---

## 7. Calibration (`calibration/`)

### 7.1 Platt Scaling (`platt.py`)

LLMs systematically hedge toward 0.5 due to RLHF training. Mathematical correction:

```
p_calibrated = p^alpha / (p^alpha + (1-p)^alpha)    where alpha = sqrt(3) ~ 1.73
```

Equivalent to `sigmoid(alpha * logit(p))`.

Effect: A raw 0.60 becomes ~0.65; a raw 0.30 becomes ~0.23. The paper found that prompting changes are largely ineffective at fixing hedging bias -- mathematical corrections are more robust.

The fixed alpha comes from Neyman & Roughgarden (2022), shown to be robust across benchmarks.

### 7.2 Monotonicity Enforcement (`monotonicity.py`)

Uses the **Pool Adjacent Violators Algorithm (PAVA)**, an isotonic regression technique.

**ABOVE mode constraints** (per tenor):
- P(above K) must be non-increasing as K increases

**HITTING mode constraints**:
- **Strike axis (per tenor)**: P(touch) decreases as barrier moves away from spot (both directions)
- **Tenor axis (per strike)**: P(touch) must be non-decreasing as tenor increases (more time = more chances)
- **At spot**: must be highest probability

PAVA properties: optimal in L2 norm (closest monotonic sequence), preserves mean, O(n) time.

Applied **twice**: once on raw probabilities (pre-Platt), once on calibrated probabilities (post-Platt).

---

## 8. Ensemble Engine (`ensemble/engine.py`)

Orchestrates M parallel agents through the pipeline:

### 8.1 Agent Diversity Creation

```python
def _create_agents():
    # Search modes: cycle through RSS_ONLY, WEB_ONLY, HYBRID
    # Temperatures: spread across [0.4, 1.0]
    # Search iterations: vary 3-7
```

When `source_config` is set, all agents use the forced mode (for A/B testing).

### 8.2 Phase 1: Shared Research

```python
shared_research = await engine.research(pair, cutoff_date)
# M agents run in parallel -> M ResearchBriefs
```

Each agent independently researches the pair. Exceptions are filtered (continues if some agents fail).

### 8.3 Phase 1.5: Tenor-Specific Research

```python
tenor_briefs = await engine.research_tenors(shared_research, tenors)
# M agents x T tenors -> per-(agent, tenor) TenorResearchBrief
```

Each agent gets 1-2 focused searches per tenor. Configurable via `settings.tenor_research_max_iterations`.

### 8.4 Phase 2: Pricing

```python
cell_data = await engine.price_surface(shared_research, strikes, tenors, spot, mode)
# M agents x T tenors LLM calls -> raw probability grid
```

Batched by tenor (one LLM call per agent per tenor). Results aggregated into per-cell mean probabilities.

### 8.5 Supervisor Logic

For single-question mode: if supervisor has HIGH confidence, its reconciled probability overrides the mean. Otherwise the simple mean stands.

---

## 9. Surface Generation (`fx/surface.py`)

The `ProbabilitySurfaceGenerator` is the main orchestrator that wires everything together.

### 9.1 Full Pipeline

1. Fetch spot rate (async, from exchangerate.host with fallback)
2. Generate strikes (around spot, configurable step/count/explicit list)
3. Phase 1: M agents research in parallel
4. Phase 1.5: Tenor-specific research (if enabled)
5. Phase 2: Each agent prices all strikes per tenor
6. Build SurfaceCell objects with synthetic EnsembleResult per cell
7. Supervisor surface review (anomaly detection + targeted search)
8. Raw monotonicity enforcement (PAVA)
9. Platt scaling calibration
10. Post-calibration monotonicity enforcement
11. Causal factor aggregation (consensus factors from briefs)
12. Regime detection (risk_on/risk_off/policy_divergence/carry_unwind/mixed)

### 9.2 Factor Aggregation

Causal factors from all agents are grouped by channel (case-insensitive). Majority-vote determines direction (bullish/bearish/contested). Only factors cited by 2+ agents are kept (or 1+ if < 3 agents total). Sorted by citation count.

### 9.3 Output Formats

Each surface run produces 6-7 outputs:

| Format | Description |
|--------|-------------|
| Console table | Color-coded probability grid (Rich library) |
| Heatmap PNG | 2D probability heatmap (strikes x tenors) |
| Scatter PNG | Prob vs strike, prob vs tenor, raw vs calibrated |
| CDF PNG | P(spot < K) curve (ABOVE mode only) |
| 3D Surface HTML | Interactive Plotly surface (rotatable, zoomable, hover tooltips) |
| JSON | Full data: probabilities, evidence, reasoning, causal factors |
| PDF Report | Multi-page: title, causal factors, probability table, charts, per-tenor narrative |

### 9.4 Interactive 3D Surface (Plotly)

Rich hover tooltips per cell showing:
- Strike, tenor, calibrated & raw probability
- Tenor-specific causal factors (event, channel, direction, magnitude, confidence)
- Tenor relevance summary
- Top 3 evidence sources with citation counts
- Disagreements
- Supervisor reasoning

---

## 10. Explanation System (`fx/explanation.py`)

Heuristic-based extraction of consensus, disagreements, and evidence from ensemble data -- **no LLM calls** (fast, deterministic, reproducible).

- **Evidence deduplication**: By URL across agents, with citation counts
- **Consensus extraction**: Identifies majority direction (above/below 0.5), extracts first sentences from aligned agents
- **Disagreement detection**: Measures spread and stdev, identifies outlier agents (> 1 stdev from mean)
- **Irrelevant evidence filtering**: Blacklists calculator/date-converter sites

---

## 11. Comparison System (`fx/compare.py`)

Side-by-side analysis of probability surfaces generated with different configurations:

- **Diff heatmaps**: Individual surface heatmaps + pairwise difference heatmaps (RdBu diverging colorscale)
- **Overlay scatter**: Probability-vs-strike and probability-vs-tenor curves overlaid per surface
- **Interactive Plotly**: Dropdown to toggle between surfaces and diff views

Enables A/B testing of data sources, LLM models, agent counts, etc.

---

## 12. PDF Report (`fx/pdf_report.py`)

Multi-page PDF using fpdf2:

1. **Title page**: Pair, spot, mode, regime, data sources, grid dimensions
2. **Causal factors narrative**: Per-factor rendering with directional icons (green +/red -/orange ~)
3. **Probability grid table**: Color-coded (green >= 0.6, red <= 0.4), spot row highlighted
4. **Charts**: Embedded heatmap, scatter, CDF images
5. **Per-tenor narrative**: Strike probabilities, causal factors, top evidence, disagreements, supervisor notes

Unicode sanitization handles smart quotes, en-dashes, ellipsis for PDF compatibility.

---

## 13. LLM Client (`llm/client.py`)

Thin wrapper around langchain-openai with:

- **Pluggable provider**: `set_llm_provider(factory)` where factory is `(model_name, temperature, max_tokens) -> BaseChatModel`
- **Reasoning model support**: o1/o3/gpt-5 get minimum 2048-token budget for chain-of-thought
- **Provider prefix stripping**: "openai/gpt-4o" -> "gpt-4o"
- **Markdown JSON extraction**: Strips ```json blocks from LLM responses (common LLM behavior)
- **Message format conversion**: Dict-based messages to LangChain message types

---

## 14. Storage Layer

### 14.1 Search Cache (`storage/cache.py`)

File-based cache (no Redis dependency):
- Keys: SHA256 hash of query string -> first 16 hex chars -> `.json` file
- Value: `{"ts": timestamp, "results": [...]}`
- TTL: 6 hours (configurable)
- Lazy deletion: expired entries removed on access
- Used for deduplicating search API calls across agents and runs

### 14.2 Forecast Database (`storage/database.py`)

SQLite persistence for forecast runs:
- Schema: `forecast_runs` table with summary columns + full JSON blob
- Auto-generated 12-char hex IDs
- Dual storage: summary columns for efficient queries, `data_json` for full replay
- Powers `forecast list` and `forecast evaluate` commands

---

## 15. Evaluation (`evaluation/metrics.py`)

### 15.1 Brier Score

```
BS = (1/n) * sum((p_i - o_i)^2)
```

- 0 = perfect, 0.25 = always guessing 0.5, 1 = worst
- Strictly proper scoring rule (incentivizes truthful forecasting)

### 15.2 Brier Decomposition

```
BS = Reliability - Resolution + Uncertainty
```

- **Reliability** (calibration error): lower is better
- **Resolution** (discrimination ability): higher is better
- **Uncertainty** (base rate variance): constant for given dataset

### 15.3 Target Benchmarks (from paper)

- ForecastBench: Brier 0.1076 (vs superforecasters at 0.1110)
- MarketLiquid: Brier 0.1258 (vs market consensus at 0.1106)

---

## 16. Company Extension System

The architecture supports proprietary extensions without forking. Copy `company.example/` to `company/` (gitignored) and customize:

### Available Extension Points

| Extension | Registration API | Purpose |
|-----------|-----------------|---------|
| Custom pairs | `register_pair(PairConfig)` | Add exotic/NDF pairs (USDCNH, USDSGD, etc.) |
| RSS feeds | `register_feed(FeedConfig)` | Add custom news sources |
| Currency keywords | `register_currency_keywords(ccy, keywords)` | Extend keyword matching |
| Data sources | `@data_source("name")` | Bloomberg, internal DB, CSV, etc. |
| Web search | `@web_search_provider("name")` | Brave, Google, etc. |
| LLM provider | `set_llm_provider(factory)` | Azure, Anthropic, Ollama, etc. |
| Consensus | `set_consensus_provider(fn)` | Analyst forecasts, internal models |
| Domain blacklist | `add_blacklisted_domains(domains)` | Additional blocked sites |

Extensions are auto-discovered: `aia_forecaster/__init__.py` imports `company/__init__.py` at startup if it exists. Silent failure if missing (upstream behavior).

### Example Extensions in `company.example/`

- **`pairs.py`**: USDCNH (offshore yuan) with renminbi/pboc keywords
- **`llm.py`**: Factory patterns for Azure OpenAI, Anthropic, Ollama
- **`search/bloomberg.py`**: Bloomberg data source placeholder
- **`search/brave.py`**: Full Brave Search implementation with freshness filters

---

## 17. Configuration (`config.py`)

Pydantic `BaseSettings` with environment variable integration:

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_MODEL` | `gpt-4o` | LLM model |
| `NUM_AGENTS` | `10` | Forecasting agents |
| `MAX_SEARCH_ITERATIONS` | `5` | Max search iterations per agent |
| `PLATT_ALPHA` | `sqrt(3)` | Calibration coefficient |
| `FORECAST_MODE` | `hitting` | Default mode |
| `REGIME_WEIGHTING_ENABLED` | `true` | Regime-aware supervisor |
| `TENOR_RESEARCH_ENABLED` | `true` | Phase 1.5 |
| `TENOR_RESEARCH_MAX_ITERATIONS` | `2` | Max iterations for Phase 1.5 |
| `RELEVANCE_THRESHOLD` | `0.20` | Min relevance score |
| `RELEVANCE_FILTERING_ENABLED` | `true` | Enable filtering |
| `WEB_SEARCH_PROVIDER` | `duckduckgo` | Default web search |
| `CACHE_TTL_HOURS` | `6` | Search cache TTL |
| `DEFAULT_PAIR` | `USDJPY` | Default currency pair |

---

## 18. CLI (`main.py`)

Poetry-installed `forecast` command with shorthand syntax support:

```bash
# Shorthand (auto-rewrites to full form)
forecast USDJPY 2026-02-15

# Explicit subcommands
forecast surface --pair USDJPY --cutoff 2026-02-15 --strikes 11 --tenors 1D,1W,1M,3M
forecast question "Will USD/JPY be above 155 in 1 week?"
forecast evaluate <RUN_ID> 1
forecast list --limit 20
forecast compare file1.json file2.json
```

The shorthand detection (`_is_pair_shorthand`) checks if the first arg is a 6-char uppercase string matching a registered pair.

---

## 19. Test Suite

10 test files with ~150+ test cases covering:

| Test File | Focus | Key Tests |
|-----------|-------|-----------|
| `test_base_rates.py` | Statistical foundation | ATM=0.5, OTM~0, ITM~1, tenor monotonicity, symmetry |
| `test_calibration.py` | Platt scaling | Identity at 0.5, push from 0.5, symmetry, alpha=sqrt(3) |
| `test_ensemble.py` | Ensemble logic | Brier score, decomposition, supervisor override logic |
| `test_explanation.py` | Evidence extraction | First sentence, dedup, consensus/disagreement |
| `test_hitting_mode.py` | Barrier probability | P(hit at barrier)=1, P(hit)>=P(above), distance monotonicity |
| `test_models.py` | Pydantic validation | Defaults, probability bounds, enum values |
| `test_monotonicity.py` | PAVA algorithm | Properties, violations, tenor ordering, mean preservation |
| `test_relevance.py` | Search relevance | Pair presence, currency keywords, commodity exemptions |
| `test_rss_feeds.py` | RSS configuration | Feed validity, pair-specific selection, keywords |
| `test_search.py` | Web search + BIS | Blacklist, institution mapping, cache TTL, XML parsing |

---

## 20. Key Design Decisions and Trade-offs

### 20.1 Forward != Consensus

The forward rate is pure carry math (interest-rate parity), not a directional view. This was a deliberate distinction informed by domain expertise -- the forward reflects funding costs, not where the market thinks the price is going. Consensus (analyst surveys, internal models) is a separate, pluggable concept.

### 20.2 Statistical Anchoring

Agents receive base rate context to ground their estimates in market math. Without this, LLMs would start from pure priors and tend to cluster around 0.5. The base rate gives them a quantitative starting point to adjust from.

### 20.3 Two-Phase Design

Separating research from pricing reduces LLM calls by ~94% while maintaining quality. Each agent develops a comprehensive macro view once, then applies it efficiently across the entire strike-tenor grid.

### 20.4 Causal Chain Reasoning

Instead of black-box probability estimates, agents produce structured causal factors (event -> channel -> direction -> magnitude). This enables:
- Transparent reasoning
- Supervisor disagreement analysis at the factor level
- Regime-aware channel weighting
- Tenor-specific factor assessment (same event can have opposite effects at different horizons)

### 20.5 Conservative Supervisor

The supervisor only overrides the simple mean when it has HIGH confidence. MEDIUM/LOW confidence falls back to averaging. This reflects the paper's finding that simple averaging is a strong baseline.

### 20.6 Mathematical Over Prompting

Platt scaling is applied as a mathematical post-hoc correction rather than trying to fix hedging bias through prompt engineering. The paper found prompting changes are "largely ineffective" -- math is more robust.

### 20.7 Extensibility Without Forking

The company extension system (auto-discovered `company/` directory) allows proprietary customization without code duplication. All extension points use decorator/registration patterns, keeping the core codebase clean.

---

## 21. Data Flow Example

A concrete example of running `forecast USDJPY`:

1. CLI parses shorthand -> `forecast --pair USDJPY --cutoff 2026-02-25 surface`
2. Fetch spot rate: 155.74 from exchangerate.host
3. Generate 5 strikes: [153.74, 154.74, 155.74, 156.74, 157.74]
4. Default tenors: [1D, 1W, 1M, 3M, 6M]
5. Create 10 agents with diversified configs

**Phase 1 (parallel):**
- Agent 0 (RSS_ONLY, T=0.4): fetches 22 RSS feeds for USDJPY, keyword-filters, produces ResearchBrief
- Agent 1 (WEB_ONLY, T=0.52): runs 4 DuckDuckGo searches, produces ResearchBrief
- Agent 2 (HYBRID, T=0.64): both RSS + 5 web searches, produces ResearchBrief
- ... (agents 3-9 similarly)

**Phase 1.5 (parallel, 10 agents x 5 tenors = 50 tasks):**
- Each agent does 1-2 targeted tenor searches per tenor
- Short tenors focus on positioning/technicals/event risk
- Long tenors focus on policy trajectories/structural flows

**Phase 2 (parallel, 10 agents x 5 tenors = 50 LLM calls):**
- Each call: agent receives research + base rates -> prices all 5 strikes at once
- Returns: `{"153.74": 0.85, "154.74": 0.72, "155.74": 0.55, "156.74": 0.38, "157.74": 0.22}`

**Phase 3:**
- Per-cell mean across 10 agents
- Supervisor detects regime (e.g., "policy_divergence")
- Supervisor reviews surface, applies HIGH-confidence adjustments
- PAVA ensures monotonicity
- Platt scaling pushes away from 0.5
- Final PAVA pass

**Output:**
- Console: colored probability table
- `data/forecasts/USDJPY_2026-02-25_above.png` (heatmap)
- `data/forecasts/USDJPY_2026-02-25_above_scatter.png`
- `data/forecasts/USDJPY_2026-02-25_above_cdf.png`
- `data/forecasts/USDJPY_2026-02-25_above.html` (3D interactive)
- `data/forecasts/USDJPY_2026-02-25_above.json` (full data)
- `data/forecasts/USDJPY_2026-02-25_above.pdf` (report)

---

## 22. Dependencies

| Package | Role |
|---------|------|
| `langchain-openai` | LLM calls (ChatOpenAI wrapper) |
| `pydantic` / `pydantic-settings` | Data models + config from .env |
| `duckduckgo-search` | Web search API (free, no key required) |
| `feedparser` | RSS feed parsing |
| `httpx` | Async HTTP (spot rates, BIS speeches) |
| `yfinance` | Interest rates + volatility from Yahoo Finance |
| `matplotlib` | Heatmaps, scatter plots, CDF charts |
| `plotly` | Interactive 3D surface (HTML) |
| `fpdf2` | PDF report generation |
| `rich` | Console output formatting |

---

## 23. Existing Forecast Data

The repository contains forecast outputs from multiple runs spanning 2026-02-13 to 2026-02-25, all for USDJPY. The `data/cache/` directory has ~190 cached search result files. The `data/forecasts/forecasts.db` SQLite database stores historical runs accessible via `forecast list`.

Sample output filenames show the evolution of the system:
- Early: `USDJPY_2026-02-13.json` (basic)
- Debug: `USDJPY_debug.json`, `USDJPY_debug2.json`
- Causal versions: `USDJPY_causal_test.json` through `USDJPY_causal_v4.json`
- Source-labeled: `USDJPY_2026-02-20_above_rss+web.json`
- Latest: `USDJPY_2026-02-25_above.json` (current production format)

---

## 24. Summary of Strengths

1. **Principled architecture**: Directly implements peer-reviewed research with clear theoretical grounding
2. **Ensemble diversity**: 10 agents with varied temperatures, search depths, and sources reduce correlation
3. **Statistical rigor**: Forward rates, realized volatility, Platt scaling, and PAVA all grounded in financial math
4. **Transparency**: Structured causal factors and comprehensive explanations at every level
5. **Extensibility**: Pluggable providers for LLM, search, consensus, pairs -- all without forking
6. **Comprehensive output**: 7 output formats from console to interactive 3D to PDF reports
7. **Test coverage**: 150+ tests covering mathematical properties, search logic, and model validation
8. **Efficiency**: Two-phase design reduces LLM calls by ~94% vs naive per-cell approach
