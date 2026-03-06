# Plan: RSS Default Off + LLM Relevance Filter

## Goal

Two changes:

1. **RSS off by default** — The `rss` data source should not be active unless explicitly opted in. BIS speeches and web search remain defaults.
2. **LLM relevance filter** — Replace the heuristic `filter_relevant()` with an LLM call that judges whether a search result is relevant to the currency pair AND (when in tenor-specific research) to the specific tenor. The heuristic stays as a cheap pre-filter; the LLM is the final decision-maker.

---

## Change 1: RSS Default Off

### What needs to change

Four places where RSS is assumed on by default:

#### 1a. `SourceConfig.registry_sources` default — `models.py:142-144`

```python
# BEFORE:
registry_sources: list[str] = Field(
    default_factory=lambda: ["rss", "bis_speeches"],
    ...
)

# AFTER:
registry_sources: list[str] = Field(
    default_factory=lambda: ["bis_speeches"],
    ...
)
```

#### 1b. Agent mode cycling — `engine.py:75`

When no `--sources` flag is provided, agents cycle `RSS_ONLY → WEB_ONLY → HYBRID`. The `RSS_ONLY` mode calls `fetch_all_sources()` with no `source_names` filter, loading all registered sources including RSS. Remove `RSS_ONLY` from the default cycle.

```python
# BEFORE:
modes = [SearchMode.RSS_ONLY, SearchMode.WEB_ONLY, SearchMode.HYBRID]

# AFTER:
modes = [SearchMode.WEB_ONLY, SearchMode.HYBRID]
```

Also need to thread a default `source_names` so `HYBRID` agents don't pick up RSS either:

```python
# engine.py, _create_agents(), lines 69-73
# BEFORE:
forced_mode: SearchMode | None = None
source_names: list[str] | None = None
if self.source_config is not None:
    forced_mode = self.source_config.get_search_mode()
    source_names = self.source_config.registry_sources or None

# AFTER:
forced_mode: SearchMode | None = None
source_names: list[str] | None = None
if self.source_config is not None:
    forced_mode = self.source_config.get_search_mode()
    source_names = self.source_config.registry_sources or None
else:
    # Default: only BIS speeches (RSS is opt-in)
    source_names = ["bis_speeches"]
```

#### 1c. CLI `--sources` help text — `main.py:458-459`

```python
# BEFORE:
help="Comma-separated data sources to enable (from: rss, bis, web). Default: all",

# AFTER:
help="Comma-separated data sources to enable (from: rss, bis, web). Default: bis,web",
```

#### 1d. CLI epilog — `main.py:403`

```python
# BEFORE:
"  forecast USDJPY --sources rss             Only RSS feeds\n"
"  forecast USDJPY --sources rss,web         RSS + web search (no BIS)\n"

# AFTER:
"  forecast USDJPY --sources rss,bis,web     All sources (including RSS)\n"
"  forecast USDJPY --sources rss             Only RSS feeds\n"
```

### What does NOT change

- The `@data_source("rss")` decorator still runs. The module still loads. It's just not fetched unless opted in.
- `--sources rss` or `--sources rss,bis,web` still works.
- `_load_builtins()` still imports `aia_forecaster.search.rss`.
- `company/` is unaffected.

---

## Change 2: LLM Relevance Filter

### Design: Two-Tier Filtering

```
Raw results
    │
    ▼
┌─────────────────────┐
│ Tier 1: Heuristic   │  score_relevance() — fast, zero cost
│ threshold = 0.10    │  kills cooking articles, wrong asset class
└─────────┬───────────┘
          │ survivors
          ▼
┌─────────────────────┐
│ Tier 2: LLM judge   │  one LLM call per batch of ~10 results
│ pair + tenor aware  │  returns keep/drop per result
└─────────┬───────────┘
          │ kept
          ▼
    Filtered results
```

### Two filter contexts

The existing filter is pair-only. The new system has two contexts:

| Context | When used | What the LLM considers |
|---------|-----------|----------------------|
| **Pair-level** | Phase 1 research, legacy `forecast()`, web search iterations | Is this relevant to the currency pair? |
| **Tenor-specific** | Phase 1.5 tenor research | Is this relevant to the pair AND actionable at this specific time horizon? |

Both use the same function — the tenor parameter is optional:

```python
async def filter_relevant_llm(
    results: list[SearchResult],
    pair: str,
    llm: LLMClient,
    *,
    tenor: Tenor | None = None,         # None = pair-level, set = tenor-specific
    heuristic_threshold: float = 0.10,
) -> list[SearchResult]
```

### 2a. New file: `src/aia_forecaster/search/llm_relevance.py`

```python
"""LLM-based relevance filtering for search results.

Two-tier approach:
1. Heuristic pre-filter (score_relevance) kills obviously irrelevant results cheaply.
2. LLM judges surviving results for pair relevance and (optionally) tenor relevance.
"""

from __future__ import annotations

import json
import logging

from aia_forecaster.llm.client import LLMClient
from aia_forecaster.models import SearchResult, Tenor
from aia_forecaster.search.relevance import filter_relevant as heuristic_filter

logger = logging.getLogger(__name__)

_BATCH_SIZE = 10

_RELEVANCE_PROMPT = """\
You are an FX research analyst. Evaluate whether each search result below is \
relevant to forecasting the {pair} currency pair.{tenor_clause}

SEARCH RESULTS:
{results_block}

For each result, decide:
- "keep" if it contains information useful for forecasting {pair}{tenor_short}
- "drop" if it is about a different currency pair, a different asset class, \
or contains no actionable FX information

Respond in this EXACT JSON format (array of objects, same order as input):
[
  {{"index": 0, "decision": "keep", "reason": "BOJ rate decision directly affects JPY"}},
  {{"index": 1, "decision": "drop", "reason": "Article about gold prices, not relevant to USDJPY"}}
]"""

_TENOR_CLAUSE = """
You are filtering for the **{tenor_label}** forecast horizon specifically. \
Consider whether the information is actionable within this timeframe:
- SHORT-TERM (1D-2W): Only keep if it describes events, data releases, or \
positioning shifts that will materialize within days/weeks.
- MEDIUM-TERM (1M-3M): Keep if it describes policy meetings, macro trends, \
or positioning that affects the pair over weeks/months.
- LONG-TERM (6M+): Keep if it describes structural shifts, policy divergence \
trajectories, or long-term flow dynamics."""

_TENOR_SHORT = {
    "D": " within the next few days",
    "W": " within the next few weeks",
    "M": " within the next few months",
    "Y": " within the next year",
}


def _tenor_short(tenor: Tenor | None) -> str:
    if tenor is None:
        return ""
    return _TENOR_SHORT.get(tenor[-1], "")


def _format_results_block(results: list[SearchResult]) -> str:
    lines = []
    for i, r in enumerate(results):
        lines.append(
            f"[{i}] Title: {r.title}\n"
            f"    Snippet: {(r.snippet or '')[:300]}\n"
            f"    URL: {r.url}"
        )
    return "\n\n".join(lines)


async def _llm_judge_batch(
    results: list[SearchResult],
    pair: str,
    tenor: Tenor | None,
    llm: LLMClient,
) -> list[bool]:
    """Judge a batch of results via a single LLM call.

    Returns list of booleans (True = keep). Fails open (all True) on error.
    """
    if not results:
        return []

    tenor_clause = ""
    if tenor is not None:
        tenor_clause = _TENOR_CLAUSE.format(tenor_label=tenor.label)

    prompt = _RELEVANCE_PROMPT.format(
        pair=pair,
        tenor_clause=tenor_clause,
        tenor_short=_tenor_short(tenor),
        results_block=_format_results_block(results),
    )

    try:
        response = await llm.complete(
            [{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=2000,
        )

        text = response.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            text = "\n".join(lines)

        decisions = json.loads(text)

        keep_set: set[int] = set()
        for item in decisions:
            idx = item.get("index", -1)
            decision = str(item.get("decision", "keep")).lower().strip()
            if decision == "keep":
                keep_set.add(idx)

        return [i in keep_set for i in range(len(results))]

    except Exception:
        logger.warning("LLM relevance filter failed — keeping all results (fail-open)")
        return [True] * len(results)


async def filter_relevant_llm(
    results: list[SearchResult],
    pair: str,
    llm: LLMClient,
    *,
    tenor: Tenor | None = None,
    heuristic_threshold: float = 0.10,
) -> list[SearchResult]:
    """Two-tier relevance filter: heuristic pre-filter + LLM judge.

    Args:
        results: Raw search results.
        pair: Currency pair (e.g. "USDJPY").
        llm: LLM client for the judge call.
        tenor: If set, LLM also judges tenor-appropriateness.
        heuristic_threshold: Pre-filter threshold (lower than the old 0.20
            since the LLM does fine-grained filtering).

    Returns:
        Filtered list with relevance_score populated on kept results.
    """
    if not results:
        return []

    # Tier 1: heuristic pre-filter
    survivors = heuristic_filter(results, pair, threshold=heuristic_threshold)
    pre_count = len(results) - len(survivors)
    if pre_count > 0:
        logger.info(
            "LLM relevance: heuristic pre-filter removed %d/%d results",
            pre_count, len(results),
        )

    if not survivors:
        return []

    # Tier 2: LLM judge (batched)
    kept: list[SearchResult] = []
    for start in range(0, len(survivors), _BATCH_SIZE):
        batch = survivors[start : start + _BATCH_SIZE]
        flags = await _llm_judge_batch(batch, pair, tenor, llm)
        for result, keep in zip(batch, flags):
            if keep:
                kept.append(result)
            else:
                logger.debug(
                    "LLM relevance: dropped '%s'",
                    result.title[:80] if result.title else "(no title)",
                )

    llm_count = len(survivors) - len(kept)
    if llm_count > 0:
        logger.info(
            "LLM relevance: LLM dropped %d/%d (pair=%s, tenor=%s)",
            llm_count, len(survivors), pair,
            tenor.value if tenor else "none",
        )

    return kept
```

### 2b. Config settings — `config.py`

```python
# BEFORE:
relevance_threshold: float = 0.20
relevance_filtering_enabled: bool = True

# AFTER:
relevance_threshold: float = 0.10          # lowered — heuristic is now pre-filter only
relevance_filtering_enabled: bool = True
llm_relevance_enabled: bool = True         # NEW
```

### 2c. Exports — `search/__init__.py`

```python
# Add import:
from .llm_relevance import filter_relevant_llm

# Add to __all__:
"filter_relevant_llm",
```

### 2d. Update all 5 call sites in `agents/forecaster.py`

Add import at top:

```python
from aia_forecaster.search.llm_relevance import filter_relevant_llm
```

Every call site follows the same transformation. Here's the before/after pattern:

```python
# BEFORE (identical at all 5 sites):
if settings.relevance_filtering_enabled:
    results = filter_relevant(results, pair, settings.relevance_threshold)

# AFTER:
if settings.relevance_filtering_enabled:
    if settings.llm_relevance_enabled:
        results = await filter_relevant_llm(
            results, pair, self.llm,
            tenor=<TENOR_VALUE>,
            heuristic_threshold=settings.relevance_threshold,
        )
    else:
        results = filter_relevant(results, pair, settings.relevance_threshold)
```

The 5 call sites and their `<TENOR_VALUE>`:

| # | Method | Line | `tenor` value | Why |
|---|--------|------|---------------|-----|
| 1 | `forecast()` | 470-472 | `None` | Legacy single-question mode, no tenor context |
| 2 | `forecast()` | 516-517 | `None` | Web search in legacy mode |
| 3 | `research()` | 597-599 | `None` | Phase 1 pair-level research |
| 4 | `research()` | 644-645 | `None` | Web search in Phase 1 |
| 5 | `research_tenor()` | 807-808 | `tenor` | Phase 1.5 tenor-specific research |

Call site #5 is the only one that passes a tenor — and that's where the LLM can add the most value, filtering out articles that are relevant to the pair but not to the specific time horizon.

#### Call site #5 in full context (`research_tenor`, line 807):

```python
# BEFORE:
try:
    results = await search_web(
        query=search_query,
        max_results=5,
        cutoff_date=cutoff_date,
    )
    if settings.relevance_filtering_enabled:
        results = filter_relevant(results, pair, settings.relevance_threshold)
    all_evidence.extend(results)

# AFTER:
try:
    results = await search_web(
        query=search_query,
        max_results=5,
        cutoff_date=cutoff_date,
    )
    if settings.relevance_filtering_enabled:
        if settings.llm_relevance_enabled:
            results = await filter_relevant_llm(
                results, pair, self.llm,
                tenor=tenor,
                heuristic_threshold=settings.relevance_threshold,
            )
        else:
            results = filter_relevant(results, pair, settings.relevance_threshold)
    all_evidence.extend(results)
```

### Cost analysis

Per surface run (10 agents, 5 tenors, 5 strikes):

| Phase | Results generated | After heuristic (~30% killed) | LLM batches (10/batch) | Tokens (~700/batch) |
|-------|-------------------|-------------------------------|------------------------|---------------------|
| Phase 1 (pair research) | ~150 | ~105 | ~11 | ~7,700 |
| Phase 1.5 (tenor research) | ~250 | ~175 | ~18 | ~12,600 |
| **Total** | **~400** | **~280** | **~29** | **~20,300** |

At GPT-4o pricing (~$5/M tokens): **~$0.10 per surface run**. Negligible vs. the ~$2-5 spent on the main forecasting/pricing LLM calls.

### Failure mode

`_llm_judge_batch()` catches all exceptions → returns `[True] * len(results)` → **fail open**. Pipeline never crashes from relevance filtering. The heuristic pre-filter still catches obvious junk even when the LLM is down.

### What the LLM prompt looks like in practice

**Pair-level (Phase 1):**

> You are an FX research analyst. Evaluate whether each search result below is relevant to forecasting the USDJPY currency pair.
>
> SEARCH RESULTS:
> [0] Title: BOJ Governor Ueda signals March rate decision
>     Snippet: Bank of Japan Governor Kazuo Ueda indicated...
> [1] Title: Gold price forecast 2026
>     Snippet: Gold prices are expected to...
>
> For each result, decide:
> - "keep" if it contains information useful for forecasting USDJPY
> - "drop" if it is about a different currency pair, a different asset class, or contains no actionable FX information

**Tenor-specific (Phase 1.5, tenor=1W):**

> You are an FX research analyst. Evaluate whether each search result below is relevant to forecasting the USDJPY currency pair.
> You are filtering for the **1 week** forecast horizon specifically. Consider whether the information is actionable within this timeframe:
> - SHORT-TERM (1D-2W): Only keep if it describes events, data releases, or positioning shifts that will materialize within days/weeks.
> ...

---

## Execution order

1. **1a-1d** — RSS default off (4 small edits)
2. **2a** — Create `llm_relevance.py`
3. **2b** — Config settings
4. **2c** — `search/__init__.py` exports
5. **2d** — Update 5 call sites in `forecaster.py`
6. Tests — mock LLM, verify heuristic pre-filter + LLM judge interaction

---

## Files summary

| File | Change |
|------|--------|
| `src/aia_forecaster/models.py` | Remove `"rss"` from `SourceConfig.registry_sources` default |
| `src/aia_forecaster/ensemble/engine.py` | Remove `RSS_ONLY` from default mode cycle; add default `source_names=["bis_speeches"]` |
| `src/aia_forecaster/main.py` | Update `--sources` help text and epilog |
| `src/aia_forecaster/config.py` | Lower `relevance_threshold` to 0.10; add `llm_relevance_enabled: bool = True` |
| `src/aia_forecaster/search/llm_relevance.py` | **NEW** — two-tier relevance filter |
| `src/aia_forecaster/search/__init__.py` | Export `filter_relevant_llm` |
| `src/aia_forecaster/agents/forecaster.py` | Import `filter_relevant_llm`; update 5 call sites |
| `tests/test_llm_relevance.py` | **NEW** — unit tests |

**Not modified:** `search/relevance.py` (still used as Tier 1), `search/registry.py`, `search/rss.py`, `search/bis.py`, `search/web.py`, `company.example/*`.

---

## Detailed Todo List

### Phase 1: RSS Default Off

Self-contained. No dependencies on Phase 2. Can be merged independently.

- [x] **1.1** Edit `src/aia_forecaster/models.py` line 143
  - Change `default_factory=lambda: ["rss", "bis_speeches"]` to `default_factory=lambda: ["bis_speeches"]`
  - This changes the `SourceConfig` default so `--sources` with no flag no longer includes RSS

- [x] **1.2** Edit `src/aia_forecaster/ensemble/engine.py` line 75
  - Change `modes = [SearchMode.RSS_ONLY, SearchMode.WEB_ONLY, SearchMode.HYBRID]` to `modes = [SearchMode.WEB_ONLY, SearchMode.HYBRID]`
  - This removes `RSS_ONLY` from the default agent mode rotation

- [x] **1.3** Edit `src/aia_forecaster/ensemble/engine.py` lines 69-73
  - Add `else` branch after the `if self.source_config is not None` block:
    ```python
    else:
        source_names = ["bis_speeches"]
    ```
  - This ensures `HYBRID` agents also don't accidentally fetch RSS via `fetch_all_sources(source_names=None)`

- [x] **1.4** Edit `src/aia_forecaster/main.py` line 458
  - Change `--sources` help text from `"Default: all"` to `"Default: bis,web"`

- [x] **1.5** Edit `src/aia_forecaster/main.py` epilog (lines 402-404)
  - Replace existing `--sources` examples with updated versions showing RSS as opt-in:
    ```
    "  forecast USDJPY --sources rss,bis,web     All sources (including RSS)\n"
    "  forecast USDJPY --sources rss             Only RSS feeds\n"
    ```

### Phase 2: Create LLM Relevance Filter Module

No dependencies on Phase 1. Can be done in parallel.

- [x] **2.1** Create new file `src/aia_forecaster/search/llm_relevance.py`
  - Full implementation as specified in section 2a of this plan
  - Contains:
    - `_BATCH_SIZE = 10` constant
    - `_RELEVANCE_PROMPT` template with `{pair}`, `{tenor_clause}`, `{tenor_short}`, `{results_block}` placeholders
    - `_TENOR_CLAUSE` template with `{tenor_label}` placeholder (short/medium/long-term guidance)
    - `_TENOR_SHORT` dict mapping unit letter (D/W/M/Y) to human-readable clause
    - `_tenor_short(tenor)` helper
    - `_format_results_block(results)` helper — formats results as indexed block for LLM prompt
    - `_llm_judge_batch(results, pair, tenor, llm)` — single LLM call for up to `_BATCH_SIZE` results, returns `list[bool]`, fails open on any exception
    - `filter_relevant_llm(results, pair, llm, *, tenor=None, heuristic_threshold=0.10)` — public API, runs heuristic pre-filter then LLM judge in batches

- [x] **2.2** Verify `_llm_judge_batch` handles edge cases in its implementation:
  - Empty results list → returns `[]`
  - LLM returns malformed JSON → catches `Exception`, returns all `True`
  - LLM returns fewer items than input → missing indices treated as "drop" (not in `keep_set`)
  - LLM returns extra indices → ignored (only indices 0..len-1 matter)
  - LLM wraps JSON in markdown code block → stripped before parsing

### Phase 3: Config and Wiring

Depends on Phase 2 (the module must exist before it can be imported).

- [x] **3.1** Edit `src/aia_forecaster/config.py` line 30
  - Change `relevance_threshold: float = 0.20` to `relevance_threshold: float = 0.10`
  - The heuristic is now a pre-filter only; 0.10 is permissive enough to let borderline results through to the LLM

- [x] **3.2** Edit `src/aia_forecaster/config.py` after line 31
  - Add `llm_relevance_enabled: bool = True` after `relevance_filtering_enabled`
  - This is the feature flag: set `LLM_RELEVANCE_ENABLED=false` in `.env` to disable

- [x] **3.3** Edit `src/aia_forecaster/search/__init__.py`
  - Add import: `from .llm_relevance import filter_relevant_llm`
  - Add `"filter_relevant_llm"` to `__all__` list (in the "Relevance filtering" section, after `"score_relevance"`)

### Phase 4: Update Forecaster Call Sites

Depends on Phase 2 (needs `filter_relevant_llm`) and Phase 3 (needs `settings.llm_relevance_enabled`).

- [x] **4.1** Edit `src/aia_forecaster/agents/forecaster.py` — add import
  - Add `from aia_forecaster.search.llm_relevance import filter_relevant_llm` after the existing `filter_relevant` import on line 36

- [x] **4.2** Edit call site #1: `forecast()` passive source filter (line 470-472)
  - Method: `forecast()`
  - Context: filters passive data source results (RSS, BIS) after `fetch_all_sources()`
  - `tenor` value: `None` (no tenor context in legacy single-question mode)
  - Transform the existing `if settings.relevance_filtering_enabled:` block to include LLM branch

- [x] **4.3** Edit call site #2: `forecast()` web search filter (line 516-517)
  - Method: `forecast()`
  - Context: filters each web search iteration's results
  - `tenor` value: `None`
  - Same transformation pattern as 4.2

- [x] **4.4** Edit call site #3: `research()` passive source filter (line 597-599)
  - Method: `research()`
  - Context: Phase 1 pair-level research, passive data source results
  - `tenor` value: `None`
  - Same transformation pattern

- [x] **4.5** Edit call site #4: `research()` web search filter (line 644-645)
  - Method: `research()`
  - Context: Phase 1 pair-level research, iterative web search results
  - `tenor` value: `None`
  - Same transformation pattern

- [x] **4.6** Edit call site #5: `research_tenor()` web search filter (line 807-808)
  - Method: `research_tenor()`
  - Context: Phase 1.5 tenor-specific research, web search results
  - `tenor` value: **`tenor`** (the method's `tenor: Tenor` parameter) — this is the key difference
  - This is where tenor-aware filtering provides the most value
  - Same transformation pattern but with `tenor=tenor` instead of `tenor=None`

### Phase 5: Tests for LLM Relevance Filter

Depends on Phase 2. Can be written in parallel with Phase 3-4.

- [x] **5.1** Create new file `tests/test_llm_relevance.py`
  - Import `filter_relevant_llm`, `_llm_judge_batch`, `_format_results_block`, `_tenor_short`
  - Import `SearchResult`, `Tenor`
  - Create `_make_result(title, snippet="", url="")` helper (same pattern as `test_relevance.py`)

- [x] **5.2** Test: `test_empty_input_returns_empty`
  - `filter_relevant_llm([], "USDJPY", mock_llm)` returns `[]`
  - LLM should NOT be called (assert `mock_llm.complete` not called)

- [x] **5.3** Test: `test_heuristic_prefilter_runs_first`
  - Pass results where some are obviously irrelevant (score < 0.10 by heuristic)
  - Mock LLM to return all "keep"
  - Assert obviously irrelevant results are NOT passed to LLM (check LLM prompt content)
  - Assert obviously irrelevant results are NOT in output

- [x] **5.4** Test: `test_llm_drops_results`
  - Pass 3 results that all survive heuristic pre-filter
  - Mock LLM to return: index 0 = keep, index 1 = drop, index 2 = keep
  - Assert output has 2 results (indices 0 and 2)

- [x] **5.5** Test: `test_llm_failure_fails_open`
  - Mock LLM `complete()` to raise `Exception("API down")`
  - Assert all results that survived heuristic pre-filter are returned (no results dropped)
  - Assert warning logged

- [x] **5.6** Test: `test_llm_malformed_json_fails_open`
  - Mock LLM to return `"this is not json"`
  - Assert all heuristic survivors are returned

- [x] **5.7** Test: `test_tenor_clause_included_when_tenor_set`
  - Call with `tenor=Tenor("1W")`
  - Capture the prompt sent to LLM (via mock)
  - Assert prompt contains `"1 week"` and `"SHORT-TERM"`

- [x] **5.8** Test: `test_tenor_clause_absent_when_tenor_none`
  - Call with `tenor=None`
  - Capture the prompt sent to LLM
  - Assert prompt does NOT contain `"SHORT-TERM"` or `"MEDIUM-TERM"` or `"LONG-TERM"`

- [x] **5.9** Test: `test_batching_multiple_calls`
  - Pass 15 results (all surviving heuristic)
  - Assert LLM `complete()` is called exactly 2 times (batch of 10 + batch of 5)

- [x] **5.10** Test: `test_format_results_block`
  - Unit test for `_format_results_block()` — verify output format with indices, truncated snippets

- [x] **5.11** Test: `test_tenor_short_mapping`
  - `_tenor_short(Tenor("1D"))` returns `" within the next few days"`
  - `_tenor_short(Tenor("3M"))` returns `" within the next few months"`
  - `_tenor_short(None)` returns `""`

### Phase 6: Update Existing Tests

Depends on Phase 1 (RSS default off may affect existing test assumptions).

- [x] **6.1** Check `tests/test_relevance.py` — NO changes needed
  - The heuristic `filter_relevant()` and `score_relevance()` are unchanged
  - All existing tests continue to test Tier 1 behavior at their current thresholds
  - The 0.20 threshold used in test assertions is the threshold passed to `filter_relevant()`, not the config default — tests are self-contained

- [x] **6.2** Check `tests/test_search.py` — NO changes needed
  - Tests cover blacklisting, keyword matching, and BIS parsing — none affected by RSS default or LLM relevance

- [x] **6.3** Grep for any tests that instantiate `SourceConfig()` without arguments
  - If any exist, they will now get `registry_sources=["bis_speeches"]` instead of `["rss", "bis_speeches"]`
  - Update assertions accordingly if found

- [x] **6.4** Grep for any tests that assert on `SearchMode.RSS_ONLY` in default agent cycling
  - If any exist, update to reflect the new `[WEB_ONLY, HYBRID]` cycle

### Phase 7: Verification

Run after all implementation phases are complete.

- [x] **7.1** Run `ruff check .` — no lint errors from new/modified files
  - Verify no unused imports in `forecaster.py` (old `filter_relevant` import is still needed for the `else` branch)
  - Verify `llm_relevance.py` passes lint

- [x] **7.2** Run `pytest tests/test_relevance.py` — all existing heuristic tests pass unchanged

- [x] **7.3** Run `pytest tests/test_llm_relevance.py` — all new LLM filter tests pass

- [x] **7.4** Run `pytest tests/test_search.py` — all search tests pass unchanged

- [x] **7.5** Run full `pytest` — no regressions (6 pre-existing failures in test_base_rates/test_hitting_mode unrelated to changes)

- [x] **7.6** Grep for remaining references to verify consistency:
  - `grep -r "rss.*bis_speeches" src/` — should only appear in `_load_builtins()` (import), not in any default lists
  - `grep -r "RSS_ONLY" src/` — should still exist as an enum value and in `SearchMode` class, but NOT in the default mode cycle in `engine.py`
  - `grep -r "filter_relevant" src/` — should show both `filter_relevant` (heuristic) and `filter_relevant_llm` (new) imports in `forecaster.py`

- [x] **7.7** Manual smoke test (optional, requires API keys): — skipped (no API keys in CI)
  - `forecast USDJPY surface --agents 2 --strikes 3` — verify no RSS log lines, verify LLM relevance filter log lines
  - `forecast USDJPY surface --agents 2 --strikes 3 --sources rss,bis,web` — verify RSS IS included when explicitly opted in
  - Set `LLM_RELEVANCE_ENABLED=false` in `.env` and run again — verify fallback to heuristic-only filtering

### Phase 8: Documentation

After all code is complete and verified.

- [x] **8.1** Update `company.example/README.md`
  - In the "How It Works" section, note that RSS is opt-in (not default)
  - Add example of opting in: `--sources rss,bis,web`

- [x] **8.2** Update `CLAUDE.md` (project-level) if it mentions default data sources — no changes needed
  - Search for any references to RSS being "default" or "always active"

- [x] **8.3** Update `research.md` "Part 12: Summary" table
  - Change RSS row from "Active: Yes" to "Active: No (opt-in via --sources rss)"
  - Add note about LLM relevance filter in the pipeline description
