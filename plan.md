# Plan: Remove All Caches

## Inventory

Four caches to remove:

| # | Cache | File | Type |
|---|-------|------|------|
| 1 | SearchCache | `storage/cache.py` | File-based JSON (web, RSS, BIS) |
| 2 | Vol cache | `fx/base_rates.py` `_vol_cache` | In-memory dict |
| 3 | Rate cache | `fx/base_rates.py` `_rate_cache` | In-memory dict |
| 4 | Spot cache | `fx/rates.py` `_rate_cache` | In-memory dict |

## Step 1: Delete `storage/cache.py` and remove exports

**Delete:** `src/aia_forecaster/storage/cache.py`

**Edit:** `src/aia_forecaster/storage/__init__.py`

```python
# Before
from .cache import SearchCache
from .database import ForecastDatabase

__all__ = ["SearchCache", "ForecastDatabase"]

# After
from .database import ForecastDatabase

__all__ = ["ForecastDatabase"]
```

## Step 2: Remove SearchCache from web search (`search/web.py`)

Remove the import, module-level `_cache` instance, and the cache check/set calls.

```python
# REMOVE these lines:
from aia_forecaster.storage.cache import SearchCache   # line 16
_cache = SearchCache()                                  # line 23

# In search_duckduckgo(), REMOVE the cache check block (lines 92-96):
    cache_key = f"web:{query}:{cutoff_date}"
    cached = _cache.get(cache_key)
    if cached is not None:
        logger.debug("Cache hit for query: %s", query)
        return [SearchResult(**r) for r in cached]

# And REMOVE the cache write block (lines 136-137):
    if results:
        _cache.set(cache_key, [r.model_dump(mode="json") for r in results])
```

The function becomes a straight-through fetch. Final shape:

```python
@web_search_provider("duckduckgo")
async def search_duckduckgo(
    query: str,
    max_results: int = 10,
    cutoff_date: date | None = None,
) -> list[SearchResult]:
    if not query or not query.strip():
        logger.warning("Empty search query -- skipping web search")
        return []

    search_query = _sanitize_query(query)
    if not search_query:
        logger.warning("Query empty after sanitization -- skipping web search")
        return []

    timelimit = _compute_timelimit(cutoff_date) if cutoff_date else None

    results: list[SearchResult] = []
    try:
        with DDGS() as ddgs:
            raw_results = list(ddgs.text(
                search_query,
                max_results=max_results + 5,
                timelimit=timelimit,
            ))

        for r in raw_results:
            url = r.get("href", r.get("link", ""))
            results.append(
                SearchResult(
                    query=query,
                    title=r.get("title", ""),
                    snippet=r.get("body", r.get("snippet", "")),
                    url=url,
                    source="duckduckgo",
                    timestamp=datetime.now(),
                )
            )
            if len(results) >= max_results:
                break

    except Exception:
        logger.exception("DuckDuckGo search failed for query: %s", query)

    return results
```

## Step 3: Remove SearchCache from RSS (`search/rss.py`)

```python
# REMOVE:
from aia_forecaster.storage.cache import SearchCache   # line 15
_cache = SearchCache()                                  # line 191

# In fetch_fx_news(), REMOVE cache check (lines 263-266):
    cache_key = f"rss:{currency_pair}:{max_age_hours}"
    cached = _cache.get(cache_key)
    if cached is not None:
        return [SearchResult(**r) for r in cached]

# And REMOVE cache write (line 325):
    _cache.set(cache_key, [r.model_dump(mode="json") for r in results])
```

## Step 4: Remove SearchCache from BIS (`search/bis.py`)

```python
# REMOVE:
from aia_forecaster.storage.cache import SearchCache   # line 25
_cache = SearchCache()                                  # line 220

# In fetch_bis_speeches(), REMOVE cache check (lines 238-241):
    cache_key = f"bis_speeches:{pair}:{max_age_hours}"
    cached = _cache.get(cache_key)
    if cached is not None:
        return [SearchResult(**r) for r in cached]

# And REMOVE cache write (line 291):
    _cache.set(cache_key, [r.model_dump(mode="json") for r in results])
```

## Step 5: Remove vol cache from `fx/base_rates.py`

Remove the module-level cache dict and TTL constant. Change `get_annualized_vol` to always compute fresh (or fall back).

```python
# REMOVE these lines:
_vol_cache: dict[str, tuple[float, float]] = {}           # line 111
_CACHE_TTL = 3600  # 1 hour                               # line 112
```

Rewrite `get_annualized_vol`:

```python
def get_annualized_vol(pair: str) -> float:
    """Return the best available annualized volatility for *pair*.

    Priority:
      1. Freshly computed realized vol from Yahoo Finance
      2. Static fallback from FALLBACK_VOL

    Raises ValueError if the pair has no fallback and dynamic fetch fails.
    """
    pair = pair.upper()

    realized = _compute_realized_vol(pair)
    if realized is not None:
        fallback = FALLBACK_VOL.get(pair)
        if fallback is not None:
            delta = abs(realized - fallback) / fallback
            if delta > 0.25:
                logger.info(
                    "%s realized vol %.1f%% differs from fallback %.1f%% by %.0f%%"
                    " -- using realized",
                    pair, realized * 100, fallback * 100, delta * 100,
                )
        else:
            logger.info(
                "%s dynamic vol computed: %.1f%% (no static fallback existed)",
                pair, realized * 100,
            )
        return realized

    if pair in FALLBACK_VOL:
        logger.debug("Using static fallback vol for %s: %.1f%%", pair, FALLBACK_VOL[pair] * 100)
        return FALLBACK_VOL[pair]

    raise ValueError(
        f"No volatility data for {pair}. "
        f"Dynamic fetch failed and no static fallback available."
    )
```

Fix `vol_source` derivation in `compute_base_rates` and `compute_hitting_base_rate`. Currently it checks `pair in _vol_cache` to decide "dynamic" vs "fallback". After removal, track the source explicitly by having `get_annualized_vol` return a tuple:

```python
def get_annualized_vol(pair: str) -> tuple[float, str]:
    """Return (annualized_vol, source) where source is 'dynamic' or 'fallback'."""
    pair = pair.upper()

    realized = _compute_realized_vol(pair)
    if realized is not None:
        # ... logging as above ...
        return realized, "dynamic"

    if pair in FALLBACK_VOL:
        logger.debug("Using static fallback vol for %s: %.1f%%", pair, FALLBACK_VOL[pair] * 100)
        return FALLBACK_VOL[pair], "fallback"

    raise ValueError(...)
```

Then update callers in `compute_base_rates` and `compute_hitting_base_rate`:

```python
# Before (both functions):
    annual_vol = get_annualized_vol(pair)
    vol_source = "dynamic" if pair in _vol_cache else "fallback"

# After:
    annual_vol, vol_source = get_annualized_vol(pair)
```

## Step 6: Remove rate cache from `fx/base_rates.py`

```python
# REMOVE:
_rate_cache: dict[str, tuple[float, float]] = {}    # line 165
_RATE_CACHE_TTL = 14400                              # line 166
```

Rewrite `get_short_rate`:

```python
def get_short_rate(currency: str) -> tuple[float, str]:
    """Return the best available short-term rate for *currency*.

    Priority:
      1. Freshly fetched dynamic rate from Yahoo Finance
      2. Static fallback from FALLBACK_POLICY_RATES

    Returns:
        Tuple of (rate_as_decimal, source_label).
    """
    currency = currency.upper()

    dynamic = _fetch_dynamic_rate(currency)
    if dynamic is not None:
        fallback = FALLBACK_POLICY_RATES.get(currency)
        if fallback is not None:
            delta_bp = abs(dynamic - fallback) * 10_000
            if delta_bp > 25:
                logger.info(
                    "%s dynamic rate %.2f%% differs from fallback %.2f%% by %.0fbp",
                    currency, dynamic * 100, fallback * 100, delta_bp,
                )
        return dynamic, "dynamic"

    fallback = FALLBACK_POLICY_RATES.get(currency)
    if fallback is not None:
        return fallback, "fallback"

    return 0.0, "none"
```

## Step 7: Remove spot rate cache from `fx/rates.py`

```python
# REMOVE:
_rate_cache: dict[str, tuple[float, float]] = {}   # line 13
_CACHE_TTL = 300                                     # line 14

# REMOVE import:
import time                                          # line 6
```

Rewrite `get_spot_rate`:

```python
async def get_spot_rate(pair: str = "USDJPY") -> float:
    """Fetch the current spot rate for a currency pair.

    Uses exchangerate.host with open.er-api.com as fallback.
    """
    base, quote = _pair_to_api_format(pair)

    url = f"https://api.exchangerate.host/latest?base={base}&symbols={quote}"
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            data = resp.json()
            return float(data["rates"][quote])
    except Exception:
        logger.warning("exchangerate.host failed, trying fallback")

    url = f"https://open.er-api.com/v6/latest/{base}"
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            data = resp.json()
            return float(data["rates"][quote])
    except Exception:
        logger.exception("All FX rate APIs failed for %s", pair)
        raise RuntimeError(f"Could not fetch spot rate for {pair}")
```

## Step 8: Remove cache settings from config

**Edit:** `src/aia_forecaster/config.py`

```python
# REMOVE these lines:
    cache_dir: Path = Path("data/cache")       # line 40
    cache_ttl_hours: int = 6                    # line 42
```

**Edit:** `.env.example` -- remove:

```
CACHE_DIR=data/cache
CACHE_TTL_HOURS=6
```

## Step 9: Update tests

### `tests/test_search.py`

Remove `TestSearchCache` class entirely (lines 63-99) and the import:

```python
# REMOVE:
from aia_forecaster.storage.cache import SearchCache   # line 20
```

Remove the `import json`, `import tempfile`, `import time`, `from pathlib import Path` imports if they are no longer used by remaining tests.

### `tests/test_base_rates.py`

Remove `_vol_cache` import and all `_vol_cache.pop(...)` cleanup lines:

```python
# REMOVE from import (line 8):
    _vol_cache,

# REMOVE each occurrence:
    _vol_cache.pop("XYZABC", None)   # lines 87, 155
    _vol_cache.pop("USDJPY", None)   # lines 117, 124, 131, 139
```

Update `test_caches_dynamic_vol` -- this test verifies caching behavior and should be removed entirely:

```python
# REMOVE (lines 128-134):
    def test_caches_dynamic_vol(self):
        """Dynamic vol should be cached after first fetch."""
        with patch("aia_forecaster.fx.base_rates._compute_realized_vol", return_value=0.12) as mock:
            _vol_cache.pop("USDJPY", None)
            get_annualized_vol("USDJPY")
            get_annualized_vol("USDJPY")  # second call should use cache
            assert mock.call_count == 1
```

Update `test_result_includes_vol_metadata` -- `vol_source` is still returned, no change needed.

If `get_annualized_vol` return type changes to `tuple[float, str]`, update any test that calls it directly:

```python
# Before:
    vol = get_annualized_vol("USDJPY")
    assert vol == FALLBACK_VOL["USDJPY"]

# After:
    vol, source = get_annualized_vol("USDJPY")
    assert vol == FALLBACK_VOL["USDJPY"]
    assert source == "fallback"
```

## Step 10: Cleanup

1. **Delete directory:** `data/cache/` (and any leftover `.json` files in it).
2. **Update `.gitignore`:** Remove the `data/cache/` line.

## Order of Operations

The steps above are ordered to minimize broken intermediate states. Execute them in this sequence:

1. Steps 5-6 first (base_rates.py) -- self-contained, return type change propagates to callers
2. Step 7 (rates.py) -- fully independent
3. Steps 2-4 (web, rss, bis) -- all depend on SearchCache but are independent of each other
4. Step 1 (delete cache.py, update __init__) -- only after all consumers removed
5. Step 8 (config cleanup) -- only after cache.py deleted
6. Step 9 (tests) -- after all production code changes
7. Step 10 (filesystem cleanup)

## Impact Assessment

### Performance

- **Vol + rate fetches:** Each `compute_base_rates()` call will hit Yahoo Finance instead of serving from memory. During a single pipeline run, `get_annualized_vol("USDJPY")` may be called dozens of times (once per cell in the probability surface). This will be noticeably slower (~1-2s per yfinance download vs instant cache hit).
- **Spot rate:** Called once per pipeline run, so 5-min cache removal has negligible impact.
- **Search results:** Each agent already generates unique queries, so the web search cache rarely helps. RSS/BIS are called once per agent, so removing the cache means re-fetching feeds per agent instead of once per 6 hours -- adds ~10-20s of network overhead per pipeline run.

### Mitigation (optional, if slowdown is unacceptable)

If the repeated Yahoo Finance calls in a single run become a bottleneck, the simplest fix is to compute vol/rates once at pipeline entry and pass them through as arguments rather than re-fetching. This is simpler than a cache -- it's just normal variable passing:

```python
# In the pipeline orchestrator (e.g., main.py or surface.py):
vol = get_annualized_vol(pair)      # one fetch
rate = get_short_rate(currency)     # one fetch
# Pass vol, rate to all downstream functions
```

This is a design refactor, not a cache, and may be done as a follow-up if needed.

## Files Modified Summary

| File | Action |
|------|--------|
| `src/aia_forecaster/storage/cache.py` | DELETE |
| `src/aia_forecaster/storage/__init__.py` | Remove SearchCache export |
| `src/aia_forecaster/search/web.py` | Remove cache import, instance, get/set calls |
| `src/aia_forecaster/search/rss.py` | Remove cache import, instance, get/set calls |
| `src/aia_forecaster/search/bis.py` | Remove cache import, instance, get/set calls |
| `src/aia_forecaster/fx/base_rates.py` | Remove `_vol_cache`, `_rate_cache`, TTL constants; change `get_annualized_vol` return type; simplify `get_short_rate` |
| `src/aia_forecaster/fx/rates.py` | Remove `_rate_cache`, `_CACHE_TTL`, `import time` |
| `src/aia_forecaster/config.py` | Remove `cache_dir`, `cache_ttl_hours` |
| `.env.example` | Remove `CACHE_DIR`, `CACHE_TTL_HOURS` |
| `tests/test_search.py` | Delete `TestSearchCache` class, remove unused imports |
| `tests/test_base_rates.py` | Remove `_vol_cache` import/cleanup, delete cache behavior test, update `get_annualized_vol` call sites for new return type |
| `data/cache/` | DELETE directory |
| `.gitignore` | Remove `data/cache/` line |

---

## Detailed Todo List

### Phase 1: In-memory caches in `fx/base_rates.py` (vol + rate)

These are the most structurally complex changes because `get_annualized_vol` changes its return type from `float` to `tuple[float, str]`, which ripples to callers.

- [x] **1.1** Remove `_vol_cache` dict and `_CACHE_TTL` constant (lines 111-112)
- [x] **1.2** Remove `import time` (line 13) -- all `time.time()` usages in this file are cache-related
- [x] **1.3** Remove `_rate_cache` dict and `_RATE_CACHE_TTL` constant (lines 165-166)
- [x] **1.4** Rewrite `get_annualized_vol` (line 384):
  - Remove cache check block (lines 397-400)
  - Remove cache write (line 405)
  - Change return type from `float` to `tuple[float, str]`
  - Return `(realized, "dynamic")` on dynamic success
  - Return `(FALLBACK_VOL[pair], "fallback")` on fallback
  - Keep the `ValueError` raise for unsupported pairs
  - Keep the logging for large fallback-vs-dynamic divergence
- [x] **1.5** Rewrite `get_short_rate` (line 233):
  - Remove cache check block (lines 250-253)
  - Remove cache write (line 258)
  - Straight-through: try dynamic, then fallback, then `(0.0, "none")`
  - Keep the logging for >25bp divergence
- [x] **1.6** Update `compute_base_rates` (line 437):
  - Change `annual_vol = get_annualized_vol(pair)` to `annual_vol, vol_source = get_annualized_vol(pair)`
  - Remove `vol_source = "dynamic" if pair in _vol_cache else "fallback"` (line 471)
- [x] **1.7** Update `compute_hitting_base_rate` (line 577):
  - Change `annual_vol = get_annualized_vol(pair)` to `annual_vol, vol_source = get_annualized_vol(pair)`
  - Remove `vol_source = "dynamic" if pair in _vol_cache else "fallback"` (line 607)

### Phase 2: Spot rate cache in `fx/rates.py`

Fully independent of Phase 1 -- can be done in parallel.

- [x] **2.1** Remove `import time` (line 6)
- [x] **2.2** Remove `_rate_cache` dict (line 13)
- [x] **2.3** Remove `_CACHE_TTL` constant (line 14)
- [x] **2.4** Rewrite `get_spot_rate`:
  - Remove cache check block (lines 34-37)
  - Remove both `_rate_cache[pair] = (rate, time.time())` writes (lines 49, 62)
  - Keep the dual-API structure and error handling as-is

### Phase 3: SearchCache removal from consumers (web, RSS, BIS)

All three consumers are independent of each other -- can be done in parallel. Must be completed before Phase 4.

- [x] **3.1** Edit `search/web.py`:
  - Remove `from aia_forecaster.storage.cache import SearchCache` (line 16)
  - Remove `_cache = SearchCache()` (line 23)
  - Remove cache check block in `search_duckduckgo` (lines 92-96): the `cache_key`, `_cache.get()`, and early return
  - Remove cache write block (lines 136-137): `if results: _cache.set(...)`
- [x] **3.2** Edit `search/rss.py`:
  - Remove `from aia_forecaster.storage.cache import SearchCache` (line 15)
  - Remove `_cache = SearchCache()` (line 191)
  - Remove cache check block in `fetch_fx_news` (lines 263-266): `cache_key`, `_cache.get()`, and early return
  - Remove cache write (line 325): `_cache.set(...)`
- [x] **3.3** Edit `search/bis.py`:
  - Remove `from aia_forecaster.storage.cache import SearchCache` (line 25)
  - Remove `_cache = SearchCache()` (line 220)
  - Remove cache check block in `fetch_bis_speeches` (lines 238-241): `cache_key`, `_cache.get()`, and early return
  - Remove cache write (line 291): `_cache.set(...)`

### Phase 4: Delete `SearchCache` class and update exports

Only safe after all consumers (Phase 3) are done.

- [x] **4.1** Delete file `src/aia_forecaster/storage/cache.py`
- [x] **4.2** Edit `src/aia_forecaster/storage/__init__.py`:
  - Remove `from .cache import SearchCache`
  - Remove `"SearchCache"` from `__all__`

### Phase 5: Config cleanup

Only safe after `cache.py` is deleted (Phase 4) -- `SearchCache.__init__` reads `settings.cache_dir` and `settings.cache_ttl_hours` at import time.

- [x] **5.1** Edit `src/aia_forecaster/config.py`:
  - Remove `cache_dir: Path = Path("data/cache")` (line 40)
  - Remove `cache_ttl_hours: int = 6` (line 42)
  - Remove `from pathlib import Path` if no longer used (check: `db_path` still uses `Path`, so keep it)
- [x] **5.2** Edit `.env.example`:
  - Remove `CACHE_DIR=data/cache` line
  - Remove `CACHE_TTL_HOURS=6` line

### Phase 6: Test updates

After all production code is changed.

- [x] **6.1** Edit `tests/test_search.py`:
  - Remove `from aia_forecaster.storage.cache import SearchCache` import (line 20)
  - Delete entire `TestSearchCache` class (lines 63-99)
  - Remove now-unused imports: `import json` (line 3), `import tempfile` (line 4), `import time` (line 5), `from pathlib import Path` (line 6) -- verify each is unused by remaining tests first
- [x] **6.2** Edit `tests/test_base_rates.py`:
  - Remove `_vol_cache` from the import block (line 8)
  - Remove all 6 `_vol_cache.pop(...)` calls (lines 87, 117, 124, 131, 139, 155)
  - Delete `test_caches_dynamic_vol` test method entirely (lines 128-134)
  - Update `test_returns_fallback_when_dynamic_fails` (line 114): change `vol = get_annualized_vol(...)` to `vol, source = get_annualized_vol(...)`, assert `source == "fallback"`
  - Update `test_returns_dynamic_when_available` (line 121): change `vol = get_annualized_vol(...)` to `vol, source = get_annualized_vol(...)`, assert `source == "dynamic"`
  - Update `test_case_insensitive` (line 136): change `vol = get_annualized_vol(...)` to `vol, source = get_annualized_vol(...)`
  - Update `test_z_score_known_value` (line 94): the `patch("...get_annualized_vol", return_value=0.10)` must change to `return_value=(0.10, "fallback")` since the function now returns a tuple
- [x] **6.3** Verify `tests/test_hitting_mode.py` needs no changes:
  - It calls `compute_base_rates` and `compute_hitting_base_rate` (not `get_annualized_vol` directly)
  - These functions unpack the tuple internally, so test code is unaffected

### Phase 7: Filesystem and git cleanup

- [x] **7.1** Delete `data/cache/` directory and all its contents
- [x] **7.2** Edit `.gitignore`: remove the `data/cache/` line

### Phase 8: Verification

- [x] **8.1** Run `pytest` -- all tests should pass
- [x] **8.2** Run `ruff check .` -- no lint errors from removed imports or unused variables
- [x] **8.3** Grep for any remaining references to `SearchCache`, `_vol_cache`, `_rate_cache`, `_CACHE_TTL`, `_RATE_CACHE_TTL`, `cache_dir`, `cache_ttl_hours` -- should find zero hits in `src/` and `tests/`
- [x] **8.4** Verify no `import time` remains in `fx/rates.py` or `fx/base_rates.py` (unless a future change reintroduces it)
- [x] **8.5** Verify `data/cache/` directory does not exist
