# Plan: Remove Volatility, Base Rate Probability Math & Yahoo Finance

## Goal

Delete all probability computation from the base rate system. The base rate probabilities are only ever used as text hints inside LLM agent prompts — they never enter any downstream math. The agents produce their own probability estimates from evidence, which flow into the ensemble mean and then Platt calibration. The base rate is advisory only.

Replace the current vol-dependent probability engine with a simple **context formatter** that tells agents: here's spot, here's the strike, here's the distance, here's consensus if available. Let the LLM figure out the probability.

## What Gets Deleted

| Item | File | Lines |
|------|------|-------|
| `FALLBACK_VOL` dict | `base_rates.py` | 90-98 |
| `ANNUALIZED_VOL` alias | `base_rates.py` | 102 |
| `_YF_SUFFIX` constant | `base_rates.py` | 104 |
| `_norm = NormalDist(0, 1)` | `base_rates.py` | 106 |
| `_compute_realized_vol()` | `base_rates.py` | 109-160 |
| `get_annualized_vol()` | `base_rates.py` | 163-206 |
| `compute_base_rates()` | `base_rates.py` | 209-279 |
| `_first_passage_probability()` | `base_rates.py` | 282-329 |
| `compute_hitting_base_rate()` | `base_rates.py` | 332-416 |
| `yfinance` dependency | `pyproject.toml` | 22 |
| `NormalDist` import | `base_rates.py` | 14 |
| `math` import (most usage) | `base_rates.py` | 12 |

## What Stays

| Item | Why |
|------|-----|
| `set_consensus_provider()` / `get_consensus()` | Still needed — consensus shifts the center for agent context |
| `format_base_rate_context()` | Rewritten — now formats spot/strike/distance/consensus as plain text, no probability number |
| `_format_market_context()` | Still needed — formats consensus line |
| `ConsensusProvider` type alias | Used by consensus hook |

## Blast Radius

### Files to Modify

| File | Severity | Change |
|------|----------|--------|
| `src/aia_forecaster/fx/base_rates.py` | **Heavy** | Delete ~300 lines of probability math. Rewrite `format_base_rate_context()` to plain text. |
| `src/aia_forecaster/agents/forecaster.py` | **Light** | Lines 921-924 grep for `"Statistical base rate"` in the context output. Change to grep for a new marker or just pass the full context per strike. |
| `src/aia_forecaster/fx/__init__.py` | **Light** | Remove `compute_base_rates` from exports. |
| `pyproject.toml` | **Trivial** | Remove `yfinance` line. |
| `tests/test_base_rates.py` | **Heavy** | Delete and rewrite — all current tests exercise the probability math. |
| `tests/test_hitting_mode.py` | **Moderate** | Delete `TestComputeHittingBaseRate` and `TestFormatBaseRateContextHitting` classes. Keep `TestEnforceHittingMonotonicity`, `TestGenerateStrikesHittingMode`, `TestForecastModeEnum`. |
| `README.md` | **Light** | Remove vol/yfinance references. |

### Files That DON'T Change

| File | Why |
|------|-----|
| `src/aia_forecaster/agents/supervisor.py` | Receives pre-built context strings |
| `src/aia_forecaster/fx/rates.py` | Spot rate fetching — unrelated |
| `src/aia_forecaster/fx/pairs.py` | Strike generation — no vol dependency |
| `src/aia_forecaster/search/*` | All search modules |
| `src/aia_forecaster/ensemble/*` | Ensemble orchestration |
| `src/aia_forecaster/calibration/*` | Platt scaling operates on agent output, not base rates |
| `src/aia_forecaster/models.py` | Data models — no vol fields |
| `src/aia_forecaster/config.py` | No vol settings |
| All `company.example/` files | No vol references |

---

## Step-by-Step Implementation

### Step 1: Rewrite `base_rates.py`

**File:** `src/aia_forecaster/fx/base_rates.py`

Delete everything except the consensus provider hook and `format_base_rate_context`. The entire file shrinks from ~560 lines to ~150.

```python
"""Base rate context for FX probability anchoring.

Provides agents with spot, strike, distance, and consensus context.
Agents use this as orientation when producing their own probability
estimates — no probability is pre-computed here.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

from aia_forecaster.models import ForecastMode, Tenor

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Consensus provider hook
# ---------------------------------------------------------------------------

ConsensusProvider = Callable[[str, float, "Tenor"], tuple[float, str] | None]

_consensus_provider: ConsensusProvider | None = None


def set_consensus_provider(provider: ConsensusProvider | None) -> None:
    """Register (or clear) a consensus-rate provider.

    The provider is called with ``(pair, spot, tenor)`` and should return
    ``(consensus_rate, source_label)`` or ``None``.

    Pass ``None`` to remove a previously registered provider.
    """
    global _consensus_provider
    _consensus_provider = provider
    if provider is not None:
        logger.info("Consensus provider registered: %s", provider)
    else:
        logger.info("Consensus provider cleared")


def get_consensus(
    pair: str, spot: float, tenor: "Tenor",
) -> tuple[float, str] | None:
    """Query the consensus provider for a point estimate.

    Returns (consensus_rate, source_label) or None.
    """
    if _consensus_provider is None:
        return None
    try:
        return _consensus_provider(pair.upper(), spot, tenor)
    except Exception:
        logger.warning(
            "Consensus provider raised for %s %s; ignoring",
            pair, tenor, exc_info=True,
        )
        return None


# ---------------------------------------------------------------------------
# Context formatting
# ---------------------------------------------------------------------------


def _format_market_context(
    consensus_rate: float | None,
    consensus_source: str | None,
    base: str,
    quote: str,
    spot: float,
    price_fmt: str,
    horizon: str,
) -> str:
    """Build the consensus line for the context block."""
    if consensus_rate is not None and consensus_source is not None:
        return (
            f"{horizon} consensus: {base}/{quote} = {consensus_rate:{price_fmt}} "
            f"(src: {consensus_source})\n"
        )
    return ""


def format_base_rate_context(
    pair: str,
    spot: float,
    strike: float,
    tenor: Tenor,
    forecast_mode: ForecastMode = ForecastMode.ABOVE,
) -> str:
    """Produce a context block for LLM agent prompts.

    Gives the agent spot, strike, distance, tenor, and consensus
    information. No probability is pre-computed — agents estimate
    probabilities themselves from evidence.

    Returns a non-empty string for any valid pair.
    """
    pair = pair.upper()
    horizon = tenor.label
    base, quote = pair[:3], pair[3:]
    price_fmt = ".2f" if "JPY" in pair else ".4f"

    # Resolve consensus
    consensus_result = get_consensus(pair, spot, tenor)
    if consensus_result is not None:
        consensus_rate, consensus_source = consensus_result
        center = consensus_rate
        center_source = consensus_source
    else:
        consensus_rate = None
        consensus_source = None
        center = spot
        center_source = "spot"

    market_note = _format_market_context(
        consensus_rate, consensus_source, base, quote, spot, price_fmt, horizon,
    )

    move_from_spot = strike - spot
    spot_sign = "+" if move_from_spot >= 0 else ""
    move_pct = move_from_spot / spot if spot != 0 else 0.0
    pct_sign = "+" if move_pct >= 0 else ""

    if forecast_mode == ForecastMode.HITTING:
        direction = "above" if strike >= spot else "below"

        drift_note = ""
        if center_source != "spot" and center != spot:
            drift_dir = "toward" if (
                (strike > spot and center > spot) or (strike < spot and center < spot)
            ) else "away from"
            drift_note = (
                f"Expected drift ({center_source}) is {drift_dir} this barrier.\n"
            )

        anchor_note = (
            f"anchored to {center_source}"
            if center_source != "spot"
            else "no consensus view available — anchored to spot"
        )

        return (
            f"MARKET CONTEXT (HITTING/BARRIER mode):\n"
            f"Current spot: {base}/{quote} = {spot:{price_fmt}}\n"
            f"{market_note}"
            f"Barrier: {strike:{price_fmt}} ({direction} spot, "
            f"distance: {spot_sign}{move_from_spot:{price_fmt}}, {pct_sign}{move_pct:.2%})\n"
            f"{drift_note}"
            f"Tenor: {horizon}\n"
            f"Note: {anchor_note}. "
            f"P(touch) ~ 1.0 near spot, decreasing with distance. "
            f"Longer tenors increase P(touch). P(touch) >= P(above) always.\n"
            f"Estimate probabilities based on evidence and this context."
        )

    # Default: ABOVE mode
    direction = "above" if strike >= center else "below"

    if center_source != "spot":
        move_from_center = strike - center
        center_sign = "+" if move_from_center >= 0 else ""
        move_lines = (
            f"  From {center_source}: {center_sign}{move_from_center:{price_fmt}} "
            f"({center_sign}{move_from_center / spot:.2%})\n"
            f"  From spot: {spot_sign}{move_from_spot:{price_fmt}} "
            f"({pct_sign}{move_pct:.2%})\n"
        )
        anchor_note = f"anchored to {center_source} (consensus view)"
    else:
        move_lines = (
            f"  From spot: {spot_sign}{move_from_spot:{price_fmt}} "
            f"({pct_sign}{move_pct:.2%})\n"
        )
        anchor_note = "no consensus view available — anchored to spot"

    return (
        f"MARKET CONTEXT:\n"
        f"Current spot: {base}/{quote} = {spot:{price_fmt}}\n"
        f"{market_note}"
        f"Target: {direction} {strike:{price_fmt}} in {horizon}\n"
        f"{move_lines}"
        f"Note: {anchor_note}.\n"
        f"Estimate probabilities based on evidence and this context."
    )
```

---

### Step 2: Update `forecaster.py` — Fix Base Rate Line Extraction

**File:** `src/aia_forecaster/agents/forecaster.py`

Lines 913-930 currently grep for `"Statistical base rate"` inside the context output to build a per-strike summary for batch pricing. Since `format_base_rate_context()` no longer emits that line, simplify to just show the distance:

Replace lines 913-930:
```python
        # Build base rates block
        base_rates_lines = []
        for strike in strikes:
            try:
                ctx = format_base_rate_context(
                    pair=pair, spot=spot, strike=strike, tenor=tenor,
                    forecast_mode=forecast_mode,
                )
                # Extract just the base rate line
                for line in ctx.split("\n"):
                    if "Statistical base rate" in line:
                        base_rates_lines.append(f"  Strike {strike}: {line.strip()}")
                        break
                else:
                    base_rates_lines.append(f"  Strike {strike}: (base rate context available)")
            except ValueError:
                base_rates_lines.append(f"  Strike {strike}: (no base rate data)")
        base_rates_block = "\n".join(base_rates_lines)
```

With:
```python
        # Build market context block (per-strike distance info)
        base_rates_lines = []
        for strike in strikes:
            dist = strike - spot
            dist_sign = "+" if dist >= 0 else ""
            pct = dist / spot if spot != 0 else 0.0
            pct_sign = "+" if pct >= 0 else ""
            base_rates_lines.append(
                f"  Strike {strike}: {dist_sign}{dist:{'.2f' if 'JPY' in pair else '.4f'}} "
                f"({pct_sign}{pct:.2%} from spot)"
            )
        base_rates_block = "\n".join(base_rates_lines)
```

Also update the prompt label on line 234 and 278. The templates already say `BASE RATES (statistical anchors):` — change to `STRIKE DISTANCES:` in both `BATCH_PRICING_PROMPT` and `BATCH_PRICING_PROMPT_HITTING`:

In `BATCH_PRICING_PROMPT` (line 234):
```diff
-BASE RATES (statistical anchors):
+STRIKE DISTANCES (from current spot):
```

In `BATCH_PRICING_PROMPT_HITTING` (line 278):
```diff
-BASE RATES (statistical anchors):
+STRIKE DISTANCES (from current spot):
```

And in both prompts, update instruction #1:
```diff
-1. Start from the base rates as statistical anchors.
+1. Consider the strike distances and market context.
```

---

### Step 3: Update `fx/__init__.py`

**File:** `src/aia_forecaster/fx/__init__.py`

Remove `compute_base_rates` from exports (it no longer exists):

```python
from .base_rates import (
    format_base_rate_context,
    get_consensus,
    set_consensus_provider,
)
from .rates import get_spot_rate
from .pairs import get_pair_config, generate_strikes

__all__ = [
    "format_base_rate_context",
    "get_consensus",
    "get_spot_rate",
    "get_pair_config",
    "generate_strikes",
    "set_consensus_provider",
]
```

---

### Step 4: Remove `yfinance` from `pyproject.toml`

```bash
poetry remove yfinance
```

This also drops transitive dependencies (`numpy`, `pandas`, etc. if nothing else needs them).

---

### Step 5: Rewrite Tests

**File:** `tests/test_base_rates.py` — delete and replace:

```python
"""Tests for base rate context formatting."""

from unittest.mock import patch

from aia_forecaster.fx.base_rates import (
    format_base_rate_context,
    get_consensus,
    set_consensus_provider,
)
from aia_forecaster.models import ForecastMode, Tenor


class TestFormatBaseRateContext:
    def test_produces_nonempty(self):
        result = format_base_rate_context("USDJPY", spot=153.0, strike=155.0, tenor=Tenor.W1)
        assert "MARKET CONTEXT" in result
        assert "153.00" in result
        assert "155.00" in result

    def test_works_for_any_pair(self):
        """No FALLBACK_VOL needed — any pair works."""
        result = format_base_rate_context("XYZABC", spot=0.90, strike=0.91, tenor=Tenor.W1)
        assert "MARKET CONTEXT" in result
        assert "0.90" in result

    def test_shows_distance(self):
        result = format_base_rate_context("USDJPY", spot=153.0, strike=155.0, tenor=Tenor.W1)
        assert "+2.00" in result
        assert "1.31%" in result or "1.30%" in result  # rounding

    def test_shows_tenor(self):
        result = format_base_rate_context("USDJPY", spot=153.0, strike=155.0, tenor=Tenor.M3)
        assert "3 months" in result

    def test_no_probability_in_output(self):
        """Context should NOT contain pre-computed probabilities."""
        result = format_base_rate_context("USDJPY", spot=153.0, strike=155.0, tenor=Tenor.W1)
        assert "Statistical base rate" not in result
        assert "P(above" not in result
        assert "Annualized vol" not in result
        assert "sigma" not in result.lower()

    def test_eurusd_4_decimal_formatting(self):
        result = format_base_rate_context("EURUSD", spot=1.0800, strike=1.0900, tenor=Tenor.M1)
        assert "1.0800" in result
        assert "1.0900" in result

    def test_hitting_mode(self):
        result = format_base_rate_context(
            "USDJPY", spot=153.0, strike=155.0, tenor=Tenor.W1,
            forecast_mode=ForecastMode.HITTING,
        )
        assert "HITTING" in result or "BARRIER" in result
        assert "Barrier" in result
        assert "P(touch)" in result

    def test_above_mode(self):
        result = format_base_rate_context(
            "USDJPY", spot=153.0, strike=155.0, tenor=Tenor.W1,
            forecast_mode=ForecastMode.ABOVE,
        )
        assert "Target" in result
        assert "above" in result

    def test_case_insensitive_pair(self):
        result = format_base_rate_context("usdjpy", spot=153.0, strike=155.0, tenor=Tenor.W1)
        assert "MARKET CONTEXT" in result


class TestConsensusProvider:
    def test_no_provider_returns_none(self):
        set_consensus_provider(None)
        assert get_consensus("USDJPY", 153.0, Tenor.W1) is None

    def test_provider_called(self):
        def mock_provider(pair, spot, tenor):
            return (150.0, "test_model")

        set_consensus_provider(mock_provider)
        result = get_consensus("USDJPY", 153.0, Tenor.W1)
        assert result == (150.0, "test_model")
        set_consensus_provider(None)  # cleanup

    def test_provider_exception_returns_none(self):
        def bad_provider(pair, spot, tenor):
            raise RuntimeError("boom")

        set_consensus_provider(bad_provider)
        result = get_consensus("USDJPY", 153.0, Tenor.W1)
        assert result is None
        set_consensus_provider(None)  # cleanup

    def test_consensus_appears_in_context(self):
        def mock_provider(pair, spot, tenor):
            return (150.0, "analyst_survey")

        set_consensus_provider(mock_provider)
        result = format_base_rate_context("USDJPY", spot=153.0, strike=155.0, tenor=Tenor.W1)
        assert "150.00" in result
        assert "analyst_survey" in result
        set_consensus_provider(None)  # cleanup
```

**File:** `tests/test_hitting_mode.py` — delete `TestComputeHittingBaseRate` and `TestFormatBaseRateContextHitting` classes. Keep the rest unchanged:

```python
"""Tests for hitting (barrier/touch) probability mode."""

from aia_forecaster.calibration.monotonicity import (
    enforce_hitting_monotonicity,
)
from aia_forecaster.fx.pairs import generate_strikes
from aia_forecaster.models import ForecastMode, Tenor


class TestEnforceHittingMonotonicity:
    # ... all existing tests unchanged ...


class TestGenerateStrikesHittingMode:
    # ... all existing tests unchanged ...


class TestForecastModeEnum:
    # ... all existing tests unchanged ...
```

---

### Step 6: Update README.md

| Line(s) | Current | New |
|---------|---------|-----|
| 100 | "forward rates, volatility" | "consensus, market context" |
| 180 | "Forward rates, consensus, vol, statistical anchors" | "Consensus, market context" |
| 231 | Yahoo Finance in diagram | Remove |
| 334 | "Forward rates, consensus, vol, statistical anchoring" | "Consensus, market context" |
| 471-475 | Volatility subsection | Delete |
| 976, 986 | "Annualized vol: 9.0% (dynamic)" | Remove these lines |
| 1023 | yfinance row in deps table | Delete row |
| 1035 | "Base rates from forward rates and volatility" | "Market context from spot and consensus" |

---

### Step 7: Verify

```bash
# Remove yfinance
poetry remove yfinance
poetry install

# Verify no stale imports
grep -r "yfinance\|import yf\|FALLBACK_VOL\|ANNUALIZED_VOL\|get_annualized_vol\|compute_base_rates\|compute_hitting_base_rate\|_first_passage\|_compute_realized_vol\|NormalDist\|sigma_t\|tenor_range_1sigma" src/ tests/

# Run tests
pytest tests/test_base_rates.py tests/test_hitting_mode.py -v

# Full suite
pytest
```

---

## Summary of Deletions

| What | Lines Deleted | Replacement |
|------|:---:|---|
| `_compute_realized_vol()` | ~50 | Nothing |
| `get_annualized_vol()` | ~45 | Nothing |
| `compute_base_rates()` | ~70 | Nothing |
| `compute_hitting_base_rate()` | ~85 | Nothing |
| `_first_passage_probability()` | ~50 | Nothing |
| `FALLBACK_VOL` / `ANNUALIZED_VOL` | ~15 | Nothing |
| `format_base_rate_context()` | ~130 | ~80 lines (plain text, no math) |
| `yfinance` dependency | 1 | Nothing |
| **Total** | **~445 lines deleted** | **~80 lines of simple formatting** |

## What Agents See Before vs After

### Before (current)
```
BASE RATE CONTEXT (statistical anchor):
Current spot: USD/JPY = 153.00
Annualized vol: 10.0% (fallback)
Target: above 155.00 in 1 week
  From spot: +2.00 (+1.31%)
Historical 1 week range (1-sigma): +/-0.96 JPY
Required move: 2.08 standard deviations
Statistical base rate: P(above 155.00) = 0.019 (1.9%)
Note: No consensus view available. Base rate is anchored to spot.
This is a log-normal baseline. Adjust based on current evidence.
```

### After (proposed)
```
MARKET CONTEXT:
Current spot: USD/JPY = 153.00
Target: above 155.00 in 1 week
  From spot: +2.00 (+1.31%)
Note: no consensus view available — anchored to spot.
Estimate probabilities based on evidence and this context.
```

The agent still knows spot, strike, distance, direction, and tenor. It just doesn't get a pre-computed probability number or vol stats. The LLM is fully capable of reasoning "a +1.3% move in 1 week is moderately unlikely" from the distance alone.

## Risk Assessment

**Very low risk.** The base rate probability was purely advisory text — it never entered any formula. The actual forecast pipeline is: agent research → agent probability → ensemble mean → Platt calibration → output. None of those steps change. The AIA Forecaster paper emphasizes that **search quality is the single most important factor**, not the statistical prior.

**Side benefit:** The system now works for ANY currency pair without needing volatility data. No more `ValueError` for unsupported pairs. No more Yahoo Finance rate-limits or data gaps. No more `numpy`/`pandas` transitive dependencies (if nothing else pulls them in).

---

## TODO List

### Phase 1: Core Deletion — `base_rates.py`

This is the critical path. Everything else depends on this file being rewritten first.

- [x] **1.1** Read the current `src/aia_forecaster/fx/base_rates.py` (560 lines)
- [x] **1.2** Delete `FALLBACK_VOL` dict (lines 90-98)
- [x] **1.3** Delete `ANNUALIZED_VOL = FALLBACK_VOL` alias (line 102)
- [x] **1.4** Delete `_YF_SUFFIX = "=X"` constant (line 104)
- [x] **1.5** Delete `_norm = NormalDist(0, 1)` (line 106)
- [x] **1.6** Delete `_compute_realized_vol()` function (lines 109-160)
- [x] **1.7** Delete `get_annualized_vol()` function (lines 163-206)
- [x] **1.8** Delete `compute_base_rates()` function (lines 209-279)
- [x] **1.9** Delete `_first_passage_probability()` function (lines 282-329)
- [x] **1.10** Delete `compute_hitting_base_rate()` function (lines 332-416)
- [x] **1.11** Remove unused imports: `math`, `NormalDist` from `statistics`
- [x] **1.12** Change `_format_market_context()` signature — currently takes a `stats` dict, change to take `consensus_rate` and `consensus_source` directly (since there's no stats dict anymore)
- [x] **1.13** Rewrite `format_base_rate_context()` ABOVE mode — replace probability math with plain text: spot, strike, distance (absolute + percentage), consensus if available, tenor, anchor note
- [x] **1.14** Rewrite `format_base_rate_context()` HITTING mode — same plain text approach: spot, barrier, distance, direction, drift note if consensus, tenor, P(touch) behavioral hints (near spot ~1.0, decreases with distance, etc.)
- [x] **1.15** Verify the rewritten file keeps: `set_consensus_provider()`, `get_consensus()`, `ConsensusProvider` type alias, `format_base_rate_context()`, `_format_market_context()` — and nothing else

### Phase 2: Update Consumers — `forecaster.py`

- [x] **2.1** Read `src/aia_forecaster/agents/forecaster.py` lines 913-930 (the `_price_single_tenor` method's base rate line extraction)
- [x] **2.2** Replace the `format_base_rate_context()` + grep-for-`"Statistical base rate"` loop with inline distance formatting: `Strike {strike}: +/-{dist} ({pct}% from spot)` per strike
- [x] **2.3** In `BATCH_PRICING_PROMPT` (line 234): change `BASE RATES (statistical anchors):` → `STRIKE DISTANCES (from current spot):`
- [x] **2.4** In `BATCH_PRICING_PROMPT_HITTING` (line 278): change `BASE RATES (statistical anchors):` → `STRIKE DISTANCES (from current spot):`
- [x] **2.5** In `BATCH_PRICING_PROMPT` instruction #1 (line 238): change `Start from the base rates as statistical anchors.` → `Consider the strike distances and market context.`
- [x] **2.6** In `BATCH_PRICING_PROMPT_HITTING` instruction #1 (line 282): same change as 2.5
- [x] **2.7** Verify `_build_base_rate_section()` (line 425) still works — it calls `format_base_rate_context()` which still exists with the same signature, so it should need no changes
- [x] **2.8** Verify `QUERY_GENERATION_PROMPT` (line 49, `{base_rate_section}`) and `FORECAST_PROMPT` (line 102, `{base_rate_section}`) still work — they consume the string output of `format_base_rate_context()` directly, so they need no changes

### Phase 3: Update Exports — `fx/__init__.py`

- [x] **3.1** Remove `compute_base_rates` from the import list in `src/aia_forecaster/fx/__init__.py` (line 2)
- [x] **3.2** Remove `"compute_base_rates"` from the `__all__` list (line 11)
- [x] **3.3** Verify remaining exports are correct: `format_base_rate_context`, `get_consensus`, `set_consensus_provider`, `get_spot_rate`, `get_pair_config`, `generate_strikes`

### Phase 4: Remove `yfinance` Dependency

- [x] **4.1** Run `poetry remove yfinance`
- [x] **4.2** Verify `pyproject.toml` no longer contains `yfinance`
- [x] **4.3** Run `poetry install` to sync the environment
- [x] **4.4** Check if `numpy` and `pandas` are still needed by other deps — if not, they'll be auto-removed by poetry

### Phase 5: Rewrite Tests — `test_base_rates.py`

- [x] **5.1** Delete the entire current `tests/test_base_rates.py` (179 lines — all tests exercise deleted probability math)
- [x] **5.2** Write new `TestFormatBaseRateContext` class:
  - [x] **5.2a** `test_produces_nonempty` — USDJPY context contains "MARKET CONTEXT", spot, strike
  - [x] **5.2b** `test_works_for_any_pair` — exotic pair "XYZABC" works (no FALLBACK_VOL needed)
  - [x] **5.2c** `test_shows_distance` — output contains "+2.00" and "1.31%" for 153→155
  - [x] **5.2d** `test_shows_tenor` — output contains "3 months" for Tenor.M3
  - [x] **5.2e** `test_no_probability_in_output` — no "Statistical base rate", "P(above", "Annualized vol", "sigma"
  - [x] **5.2f** `test_eurusd_4_decimal_formatting` — EURUSD uses ".4f" format
  - [x] **5.2g** `test_hitting_mode` — HITTING mode output contains "HITTING" or "BARRIER", "Barrier", "P(touch)"
  - [x] **5.2h** `test_above_mode` — ABOVE mode output contains "Target", "above"
  - [x] **5.2i** `test_case_insensitive_pair` — lowercase "usdjpy" works
- [x] **5.3** Write new `TestConsensusProvider` class:
  - [x] **5.3a** `test_no_provider_returns_none` — None provider returns None
  - [x] **5.3b** `test_provider_called` — registered provider returns expected value
  - [x] **5.3c** `test_provider_exception_returns_none` — broken provider returns None (fail-safe)
  - [x] **5.3d** `test_consensus_appears_in_context` — consensus rate and source label appear in formatted output

### Phase 6: Update Tests — `test_hitting_mode.py`

- [x] **6.1** Delete `TestComputeHittingBaseRate` class (lines 18-96) — all 8 test methods exercise deleted probability functions
- [x] **6.2** Delete `TestFormatBaseRateContextHitting` class (lines 99-125) — 3 test methods check for "log-normal", "P(touch", "first-passage" which are gone
- [x] **6.3** Remove now-unused imports: `compute_base_rates`, `compute_hitting_base_rate`, `format_base_rate_context` from `aia_forecaster.fx.base_rates`
- [x] **6.4** Remove `from unittest.mock import patch` if no longer used
- [x] **6.5** Verify remaining test classes are untouched and still pass:
  - `TestEnforceHittingMonotonicity` (5 tests) — tests calibration monotonicity enforcement, no base rate dependency
  - `TestGenerateStrikesHittingMode` (4 tests) — tests strike generation, no base rate dependency
  - `TestForecastModeEnum` (2 tests) — tests enum values, no base rate dependency

### Phase 7: Update Documentation

- [x] **7.1** Update `README.md`:
  - [x] **7.1a** Line 100: "forward rates, volatility" → "consensus, market context"
  - [x] **7.1b** Line 180: "Forward rates, consensus, vol, statistical anchors" → "Consensus, market context"
  - [x] **7.1c** Line 231: Remove Yahoo Finance from architecture diagram
  - [x] **7.1d** Line 334: "Forward rates, consensus, vol, statistical anchoring" → "Consensus, market context"
  - [x] **7.1e** Lines 471-475: Delete the "Volatility" subsection about Yahoo Finance
  - [x] **7.1f** Lines 976, 986: Remove "Annualized vol: 9.0% (dynamic)" from example outputs
  - [x] **7.1g** Line 1023: Delete yfinance row from dependencies table
  - [x] **7.1h** Line 1035: "Base rates from forward rates and volatility" → "Market context from spot and consensus"
- [x] **7.2** Update `research.md`:
  - [x] **7.2a** Section 1.5 (Volatility Data): Replace with note that volatility was removed; base rates now use plain market context
  - [x] **7.2b** Section 5 (Disableability Summary): Remove Yahoo Finance vol row
  - [x] **7.2c** Section 6 (Fallback Chains): Remove "Volatility" fallback chain
  - [x] **7.2d** Section 8 (What's NOT Pluggable): Remove volatility gap (no longer relevant)

### Phase 8: Update Memory Files

- [x] **8.1** Update `memory/MEMORY.md`:
  - [x] **8.1a** Remove "Probability Math" section (ABOVE mode formula, HITTING mode formula, first-passage reference)
  - [x] **8.1b** Update "Key Design Decisions" — remove log-normal convexity note, test tolerance note
  - [x] **8.1c** Update "File Structure" — `base_rates.py` description from "consensus, vol, base rate computation, LLM context formatting" to "consensus provider, market context formatting"
  - [x] **8.1d** Update `__init__.py` exports line — remove `compute_base_rates`
- [x] **8.2** Update or delete `memory/base_rates_details.md` — likely contains extended notes about the probability math being removed

### Phase 9: Verification

- [x] **9.1** Run stale import check:
  ```
  grep -r "yfinance\|import yf\|FALLBACK_VOL\|ANNUALIZED_VOL\|get_annualized_vol\|compute_base_rates\|compute_hitting_base_rate\|_first_passage\|_compute_realized_vol\|NormalDist\|sigma_t\|tenor_range_1sigma" src/ tests/
  ```
- [x] **9.2** Run `pytest tests/test_base_rates.py -v` — all new tests pass
- [x] **9.3** Run `pytest tests/test_hitting_mode.py -v` — remaining tests pass
- [x] **9.4** Run `pytest` — full test suite green
- [x] **9.5** Run `poetry install` — no yfinance in environment
- [x] **9.6** Spot-check: `python -c "from aia_forecaster.fx import format_base_rate_context"` — imports cleanly
- [x] **9.7** Spot-check: `python -c "import yfinance"` — should fail (not installed)
- [x] **9.8** Spot-check: Run a quick forecast to verify agent prompts look correct:
  ```
  forecast USDJPY 2026-03-15 --agents 1 --sources web -v
  ```
  Verify the verbose output shows "MARKET CONTEXT" (not "BASE RATE CONTEXT") and no vol/sigma references in agent prompts
