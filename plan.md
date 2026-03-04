# Plan: Remove Foreknowledge Detection & Prediction Market Blacklist

## Motivation

1. **Foreknowledge detection** is unnecessary because the system is used for live forward-looking forecasts, not backtesting. The `check_foreknowledge()` and `filter_foreknowledge()` functions are already defined but **never called** anywhere in the codebase. This is pure dead code cleanup.

2. **Prediction market blacklist** should be removed. Prediction markets (Polymarket, Metaculus, Kalshi, etc.) are a rich source of probability information for forward-looking forecasts. The blacklist was designed to prevent data leakage during backtesting -- irrelevant for live use.

## What stays

- **Phase 1** (pair-level research) -- unchanged
- **Phase 1.5** (tenor-specific research) -- unchanged
- **Phase 2** (batch pricing) -- unchanged
- **`cutoff_date`** parameter stays everywhere. It still serves real purposes:
  - DuckDuckGo `timelimit` filtering (restricts search results by recency)
  - RSS feed `max_age_hours` filtering
  - BIS speech temporal filtering
  - Prompt context (agents see the date to ground their reasoning)
- **`IRRELEVANT_DOMAINS`** list stays (calculator.net, timeanddate.com, etc. -- these are genuinely noise)
- **`add_blacklisted_domains()`** function stays -- companies can still block their own internal domains via the extension system
- **`_is_blacklisted()`** function stays -- it still checks `IRRELEVANT_DOMAINS`

---

## Step 1: Delete `foreknowledge.py`

Delete the entire file. It's 119 lines of unused code.

**File:** `src/aia_forecaster/search/foreknowledge.py` -- DELETE

---

## Step 2: Remove `FlaggedResult` model

**File:** `src/aia_forecaster/models.py`

Delete:
```python
class FlaggedResult(BaseModel):
    """Search result flagged for potential foreknowledge bias."""

    result: SearchResult
    has_foreknowledge: bool
    confidence: Confidence
    evidence: str = ""
```

Note: `Confidence` enum stays -- it's used by `SupervisorResult`.

---

## Step 3: Remove foreknowledge exports from `search/__init__.py`

**File:** `src/aia_forecaster/search/__init__.py`

Remove these imports and any `__all__` entries:
```python
from .foreknowledge import check_foreknowledge, filter_foreknowledge
```

---

## Step 4: Empty the prediction market blacklist

**File:** `src/aia_forecaster/search/web_providers.py`

Change the `BLACKLISTED_DOMAINS` list from:
```python
# Prediction market domains to blacklist (can leak foreknowledge)
BLACKLISTED_DOMAINS = [
    "polymarket.com",
    "metaculus.com",
    "manifold.markets",
    "kalshi.com",
    "predictit.org",
    "smarkets.com",
]
```

To:
```python
# Custom blacklisted domains (add via add_blacklisted_domains() or company extensions).
# Prediction markets are intentionally NOT blacklisted -- they provide valuable
# probability signals for forward-looking forecasts.
BLACKLISTED_DOMAINS: list[str] = []
```

The `add_blacklisted_domains()` function, `_is_blacklisted()` helper, and the filtering logic in `search_web()` all stay intact. They still serve two purposes:
1. Filtering `IRRELEVANT_DOMAINS` (calculators, date tools, etc.)
2. Allowing company extensions to block their own domains via `add_blacklisted_domains()`

---

## Step 5: Update tests

**File:** `tests/test_search.py`

The `TestBlacklist` class currently tests:
```python
class TestBlacklist:
    def test_blacklisted_domains(self):
        assert _is_blacklisted("https://polymarket.com/market/123")
        assert _is_blacklisted("https://www.metaculus.com/questions/1234")
        assert _is_blacklisted("https://manifold.markets/foo")
        assert _is_blacklisted("https://kalshi.com/event/test")

    def test_non_blacklisted(self):
        assert not _is_blacklisted("https://reuters.com/article/test")
        assert not _is_blacklisted("https://bbc.com/news/test")
        assert not _is_blacklisted("https://fxstreet.com/news/test")
```

Replace `test_blacklisted_domains` -- prediction markets are no longer blocked:
```python
class TestBlacklist:
    def test_prediction_markets_not_blocked(self):
        """Prediction markets are valuable signal for forward-looking forecasts."""
        assert not _is_blacklisted("https://polymarket.com/market/123")
        assert not _is_blacklisted("https://www.metaculus.com/questions/1234")
        assert not _is_blacklisted("https://manifold.markets/foo")
        assert not _is_blacklisted("https://kalshi.com/event/test")

    def test_irrelevant_domains_still_blocked(self):
        """Calculator and utility sites are still filtered."""
        assert _is_blacklisted("https://calculator.net/some-tool")
        assert _is_blacklisted("https://timeanddate.com/countdown")

    def test_news_sites_not_blocked(self):
        assert not _is_blacklisted("https://reuters.com/article/test")
        assert not _is_blacklisted("https://bbc.com/news/test")
        assert not _is_blacklisted("https://fxstreet.com/news/test")

    def test_add_blacklisted_domains(self):
        """Company extensions can still add custom blacklisted domains."""
        add_blacklisted_domains(["internal-wiki.example.com"])
        assert _is_blacklisted("https://internal-wiki.example.com/page")
        # Clean up
        BLACKLISTED_DOMAINS.remove("internal-wiki.example.com")
```

Also need to update the import line in `tests/test_search.py`:
```python
# Before:
from aia_forecaster.search.web import BLACKLISTED_DOMAINS, _is_blacklisted

# After (import from web_providers where they're defined):
from aia_forecaster.search.web_providers import (
    BLACKLISTED_DOMAINS,
    _is_blacklisted,
    add_blacklisted_domains,
)
```

---

## Step 6: Run tests and verify

```bash
poetry run pytest -x -v
```

Ensure all tests pass after the changes.

---

## TODO List

### Phase A: Remove Foreknowledge Detection (dead code cleanup)

- [x] **A1.** Delete file `src/aia_forecaster/search/foreknowledge.py`
- [x] **A2.** Remove `FlaggedResult` class from `src/aia_forecaster/models.py`
  - Delete the class definition (5 lines)
  - Verify `Confidence` enum has no other dead references after this removal
- [x] **A3.** Remove foreknowledge imports from `src/aia_forecaster/search/__init__.py`
  - No-op: confirmed file had no foreknowledge imports
- [x] **A4.** Verify no other file imports from `foreknowledge.py`
  - Grep for `foreknowledge` across all `.py` files
  - Only comment references remained; updated `web_providers.py` and `registry.py` comments

### Phase B: Remove Prediction Market Blacklist

- [x] **B1.** Empty `BLACKLISTED_DOMAINS` in `src/aia_forecaster/search/web_providers.py`
  - Replace the 6-domain list with an empty `list[str]`
  - Update the comment to explain prediction markets are intentionally allowed
- [x] **B2.** Update `tests/test_search.py` imports
  - Change `from aia_forecaster.search.web import BLACKLISTED_DOMAINS, _is_blacklisted` to import from `web_providers` instead
  - Add `add_blacklisted_domains` to the import
- [x] **B3.** Rewrite `TestBlacklist.test_blacklisted_domains` in `tests/test_search.py`
  - Replace with `test_prediction_markets_not_blocked` (assert polymarket, metaculus, manifold, kalshi are NOT blocked)
- [x] **B4.** Add `test_irrelevant_domains_still_blocked` test in `tests/test_search.py`
  - Assert calculator.net, timeanddate.com are still blocked
- [x] **B5.** Keep existing `test_non_blacklisted` test (reuters, bbc, fxstreet -- renamed to `test_news_sites_not_blocked`)
- [x] **B6.** Add `test_add_blacklisted_domains` test in `tests/test_search.py`
  - Verify company extensions can still add custom domains at runtime
  - Clean up after test (remove the added domain)

### Phase C: Verify & Clean Up

- [x] **C1.** Run full test suite: `poetry run pytest -x -v`
  - 140 tests pass; 2 pre-existing failures (test_base_rates, test_hitting_mode) unrelated to changes
- [x] **C2.** Grep for any remaining references to deleted symbols
  - `foreknowledge`, `FlaggedResult`, `check_foreknowledge`, `filter_foreknowledge`
  - Only plan.md and research.md references remain (expected)
- [x] **C3.** Verify `CLAUDE.md` and `README.md` references
  - Updated CLAUDE.md: removed foreknowledge section, updated search notes, removed blacklist mention
  - Updated README.md: removed foreknowledge.py from tree, updated blacklist to note prediction markets allowed
- [ ] **C4.** Run a quick smoke test: `forecast USDJPY --strikes 3 --agents 3` to confirm the pipeline still works end-to-end

---

## Summary of changes

| Action | File | What |
|--------|------|------|
| DELETE | `search/foreknowledge.py` | Entire file (119 lines) |
| EDIT | `search/__init__.py` | Remove foreknowledge imports |
| EDIT | `models.py` | Remove `FlaggedResult` class |
| EDIT | `search/web_providers.py` | Empty `BLACKLISTED_DOMAINS` list, update comment |
| EDIT | `tests/test_search.py` | Update blacklist tests, fix imports |

**Estimated lines removed:** ~130
**Estimated lines modified:** ~20
