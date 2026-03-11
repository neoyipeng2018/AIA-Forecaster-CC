# Plan: Remove Forward Rate Fallback When No Consensus

## Goal

When no consensus provider is registered (or it returns `None`), the system currently falls back to the **forward rate** (carry-adjusted via interest-rate parity) as the distribution center. This change removes that fallback entirely:

- **No consensus → center = spot** (zero directional assumption)
- **No forward/carry line in prompts** — agents see only volatility-based statistical anchors
- **Forward computation becomes dead code** — removed

## Why

The forward rate is mechanical carry math, not a directional view. Presenting it to agents as a "center" misleads them into thinking there's a market expectation when there isn't one. Using spot as center (zero drift) is the honest neutral position — agents should rely on their research evidence, not carry math, for direction.

---

## Affected Files

| File | Changes |
|------|---------|
| `src/aia_forecaster/fx/base_rates.py` | Core logic: center=spot when no consensus, simplify prompt formatting, remove dead forward code |
| `src/aia_forecaster/fx/__init__.py` | Remove `compute_forward_rate`, `get_short_rate` exports |
| `tests/test_base_rates.py` | Fix 3 failing tests (ATM, symmetry, z_score) + update assertions |
| `tests/test_hitting_mode.py` | Fix 2 failing tests (format context string assertions) |

Files that need **no changes**: `agents/forecaster.py`, `agents/supervisor.py`, `fx/surface.py`, `fx/explanation.py`, `fx/pdf_report.py` — they consume `format_base_rate_context()` which will continue to return a string, just with different content.

---

## Step-by-Step Implementation

### Step 1: Modify `compute_base_rates()` (line 406)

Replace the forward fallback with spot-as-center.

**Current code (lines 441–456):**
```python
# Check for consensus first — if available, skip forward computation
consensus_result = get_consensus(pair, spot, tenor)
if consensus_result is not None:
    consensus_rate, consensus_source = consensus_result
    center = consensus_rate
    center_source = consensus_source
    forward = None
    fwd_source = None
    r_base = None
    r_quote = None
else:
    consensus_rate = None
    consensus_source = None
    forward, r_base, r_quote, fwd_source = compute_forward_rate(pair, spot, tenor)
    center = forward
    center_source = "forward"
```

**New code:**
```python
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
```

**Update the return dict (lines 475–492)** — remove forward/rate fields:

**Current:**
```python
return {
    "sigma_t": sigma_t,
    "move_required": move_required,
    "move_pct": move_pct,
    "z_score": z_score,
    "base_rate_above": base_rate_above,
    "tenor_range_1sigma": tenor_range_1sigma,
    "annualized_vol": annual_vol,
    "vol_source": vol_source,
    "forward_rate": forward,
    "forward_source": fwd_source,
    "r_base": r_base,
    "r_quote": r_quote,
    "center": center,
    "center_source": center_source,
    "consensus_rate": consensus_rate,
    "consensus_source": consensus_source,
}
```

**New:**
```python
return {
    "sigma_t": sigma_t,
    "move_required": move_required,
    "move_pct": move_pct,
    "z_score": z_score,
    "base_rate_above": base_rate_above,
    "tenor_range_1sigma": tenor_range_1sigma,
    "annualized_vol": annual_vol,
    "vol_source": vol_source,
    "center": center,
    "center_source": center_source,
    "consensus_rate": consensus_rate,
    "consensus_source": consensus_source,
}
```

### Step 2: Modify `compute_hitting_base_rate()` (line 545)

Same pattern — replace forward fallback with spot.

**Current code (lines 576–591):**
```python
consensus_result = get_consensus(pair, spot, tenor)
if consensus_result is not None:
    consensus_rate, consensus_source = consensus_result
    center = consensus_rate
    center_source = consensus_source
    forward = None
    fwd_source = None
    r_base = None
    r_quote = None
else:
    consensus_rate = None
    consensus_source = None
    forward, r_base, r_quote, fwd_source = compute_forward_rate(pair, spot, tenor)
    center = forward
    center_source = "forward"
```

**New code:**
```python
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
```

**Update the `base_dict` (lines 597–610)** — remove forward/rate fields:

**Current:**
```python
base_dict = {
    "sigma_t": sigma_t,
    "tenor_range_1sigma": tenor_range_1sigma,
    "annualized_vol": annual_vol,
    "vol_source": vol_source,
    "forward_rate": forward,
    "forward_source": fwd_source,
    "r_base": r_base,
    "r_quote": r_quote,
    "center": center,
    "center_source": center_source,
    "consensus_rate": consensus_rate,
    "consensus_source": consensus_source,
}
```

**New:**
```python
base_dict = {
    "sigma_t": sigma_t,
    "tenor_range_1sigma": tenor_range_1sigma,
    "annualized_vol": annual_vol,
    "vol_source": vol_source,
    "center": center,
    "center_source": center_source,
    "consensus_rate": consensus_rate,
    "consensus_source": consensus_source,
}
```

### Step 3: Simplify `_format_market_context()` (line 643)

Remove the entire forward formatting branch. When no consensus, return empty string — no market context line.

**Current code (lines 643–681):**
```python
def _format_market_context(
    stats: dict, base: str, quote: str, spot: float, price_fmt: str, horizon: str,
) -> str:
    """Build the forward + consensus lines for the context block."""
    lines: list[str] = []

    # Consensus line (directional view — shown when provider is set)
    cons = stats.get("consensus_rate")
    cons_src = stats.get("consensus_source")
    if cons is not None and cons_src is not None:
        lines.append(
            f"{horizon} consensus: {base}/{quote} = {cons:{price_fmt}} "
            f"(src: {cons_src})"
        )
    else:
        # Forward line (carry math — only shown when no consensus is available)
        fwd = stats.get("forward_rate")
        fwd_src = stats.get("forward_source", "")
        if fwd is not None and fwd_src != "no_rates":
            r_b = stats.get("r_base", 0)
            r_q = stats.get("r_quote", 0)
            diff = r_q - r_b
            diff_sign = "+" if diff >= 0 else ""

            src_parts = fwd_src.split("|") if "|" in fwd_src else []
            if src_parts:
                src_note = "/".join(p.split(":")[-1] for p in src_parts)
            else:
                src_note = fwd_src

            lines.append(
                f"{horizon} forward: {base}/{quote} = {fwd:{price_fmt}} "
                f"(carry: {base} {r_b:.2%} vs {quote} {r_q:.2%}, "
                f"net {diff_sign}{diff:.2%}, src: {src_note})"
            )

    if not lines:
        return ""
    return "\n".join(lines) + "\n"
```

**New code:**
```python
def _format_market_context(
    stats: dict, base: str, quote: str, spot: float, price_fmt: str, horizon: str,
) -> str:
    """Build the consensus line for the context block (empty when no consensus)."""
    cons = stats.get("consensus_rate")
    cons_src = stats.get("consensus_source")
    if cons is not None and cons_src is not None:
        return (
            f"{horizon} consensus: {base}/{quote} = {cons:{price_fmt}} "
            f"(src: {cons_src})\n"
        )
    return ""
```

### Step 4: Update `format_base_rate_context()` — ABOVE mode (line 749)

When `center_source == "spot"`, the "From center" line and note text need to reflect that there's no directional view. The move-from-spot and move-from-center become identical, so only show one.

**Current code (lines 755–789):**
```python
center = stats["center"]
center_src = stats["center_source"]
direction = "above" if strike >= center else "below"

# Show move from center (primary) and from spot (for context)
move_from_center = strike - center
move_from_spot = strike - spot
center_sign = "+" if move_from_center >= 0 else ""
spot_sign = "+" if move_from_spot >= 0 else ""

center_label = center_src if center_src != "forward" else "forward"

vol_note = (
    f"Annualized vol: {stats['annualized_vol']:.1%} ({stats['vol_source']})\n"
)
market_note = _format_market_context(stats, base, quote, spot, price_fmt, horizon)

return (
    f"BASE RATE CONTEXT (statistical anchor):\n"
    f"Current spot: {base}/{quote} = {spot:{price_fmt}}\n"
    f"{market_note}"
    f"{vol_note}"
    f"Target: {direction} {strike:{price_fmt}} in {horizon}\n"
    f"  From {center_label}: {center_sign}{move_from_center:{price_fmt}} "
    f"({center_sign}{move_from_center / spot:.2%})\n"
    f"  From spot:    {spot_sign}{move_from_spot:{price_fmt}} "
    f"({spot_sign}{move_from_spot / spot:.2%})\n"
    f"Historical {horizon} range (1-sigma): +/-{stats['tenor_range_1sigma']:{price_fmt}} {quote}\n"
    f"Required move from {center_label}: {abs(stats['z_score']):.2f} standard deviations\n"
    f"Statistical base rate: P({direction} {strike:{price_fmt}}) "
    f"= {stats['base_rate_above']:.3f} ({stats['base_rate_above']:.1%})\n"
    f"Note: Base rate is anchored to {center_label}"
    f"{' (consensus view)' if center_src != 'forward' else ' (carry-adjusted)'}.\n"
    f"This is a log-normal baseline. Adjust based on current evidence."
)
```

**New code:**
```python
center = stats["center"]
center_src = stats["center_source"]
direction = "above" if strike >= center else "below"

move_from_spot = strike - spot
spot_sign = "+" if move_from_spot >= 0 else ""

vol_note = (
    f"Annualized vol: {stats['annualized_vol']:.1%} ({stats['vol_source']})\n"
)
market_note = _format_market_context(stats, base, quote, spot, price_fmt, horizon)

# When consensus is available, also show move from consensus
if center_src != "spot":
    move_from_center = strike - center
    center_sign = "+" if move_from_center >= 0 else ""
    center_label = center_src
    move_lines = (
        f"  From {center_label}: {center_sign}{move_from_center:{price_fmt}} "
        f"({center_sign}{move_from_center / spot:.2%})\n"
        f"  From spot:    {spot_sign}{move_from_spot:{price_fmt}} "
        f"({spot_sign}{move_from_spot / spot:.2%})\n"
    )
    anchor_note = (
        f"Required move from {center_label}: {abs(stats['z_score']):.2f} standard deviations\n"
    )
    tail_note = (
        f"Note: Base rate is anchored to {center_label} (consensus view).\n"
    )
else:
    move_lines = (
        f"  From spot: {spot_sign}{move_from_spot:{price_fmt}} "
        f"({spot_sign}{move_from_spot / spot:.2%})\n"
    )
    anchor_note = (
        f"Required move: {abs(stats['z_score']):.2f} standard deviations\n"
    )
    tail_note = (
        f"Note: No consensus view available. Base rate is anchored to spot.\n"
    )

return (
    f"BASE RATE CONTEXT (statistical anchor):\n"
    f"Current spot: {base}/{quote} = {spot:{price_fmt}}\n"
    f"{market_note}"
    f"{vol_note}"
    f"Target: {direction} {strike:{price_fmt}} in {horizon}\n"
    f"{move_lines}"
    f"Historical {horizon} range (1-sigma): +/-{stats['tenor_range_1sigma']:{price_fmt}} {quote}\n"
    f"{anchor_note}"
    f"Statistical base rate: P({direction} {strike:{price_fmt}}) "
    f"= {stats['base_rate_above']:.3f} ({stats['base_rate_above']:.1%})\n"
    f"{tail_note}"
    f"This is a log-normal baseline. Adjust based on current evidence."
)
```

### Step 5: Update `format_base_rate_context()` — HITTING mode (line 702)

Same pattern as ABOVE — adjust the drift note and anchor label for `center_source == "spot"`.

**Current code (lines 719–747):**
```python
drift_note = ""
if center is not None and center != spot:
    drift_dir = "toward" if (
        (strike > spot and center > spot) or (strike < spot and center < spot)
    ) else "away from"
    drift_note = (
        f"Expected drift ({center_src}) is {drift_dir} this barrier.\n"
    )

anchor_label = f"the {center_src}" if center_src != "forward" else "the forward"

return (
    f"BASE RATE CONTEXT (statistical anchor — HITTING/BARRIER mode):\n"
    f"Current spot: {base}/{quote} = {spot:{price_fmt}}\n"
    f"{market_note}"
    f"{vol_note}"
    f"Barrier: {strike:{price_fmt}} ({direction} spot, "
    f"distance: {dist_sign}{distance_pct:.2%})\n"
    f"{drift_note}"
    f"Historical {horizon} range (1-sigma): +/-{stats['tenor_range_1sigma']:{price_fmt}} {quote}\n"
    f"Statistical base rate: P(touch {strike:{price_fmt}} within {horizon}) "
    f"= {stats['base_rate_hitting']:.3f} ({stats['base_rate_hitting']:.1%})\n"
    f"For reference, P(above {strike:{price_fmt}} at expiry) "
    f"= {stats['base_rate_above']:.3f}\n"
    f"Note: P(touch) >= P(above) always. P(touch) ~ 1.0 near spot, "
    f"decreasing with distance. Longer tenors increase P(touch).\n"
    f"This is a drift-adjusted first-passage baseline anchored to {anchor_label}. "
    f"Adjust based on current evidence."
)
```

**New code:**
```python
drift_note = ""
if center_src != "spot" and center != spot:
    drift_dir = "toward" if (
        (strike > spot and center > spot) or (strike < spot and center < spot)
    ) else "away from"
    drift_note = (
        f"Expected drift ({center_src}) is {drift_dir} this barrier.\n"
    )

if center_src != "spot":
    anchor_tail = f"anchored to the {center_src}"
else:
    anchor_tail = "anchored to spot (no consensus view available)"

return (
    f"BASE RATE CONTEXT (statistical anchor — HITTING/BARRIER mode):\n"
    f"Current spot: {base}/{quote} = {spot:{price_fmt}}\n"
    f"{market_note}"
    f"{vol_note}"
    f"Barrier: {strike:{price_fmt}} ({direction} spot, "
    f"distance: {dist_sign}{distance_pct:.2%})\n"
    f"{drift_note}"
    f"Historical {horizon} range (1-sigma): +/-{stats['tenor_range_1sigma']:{price_fmt}} {quote}\n"
    f"Statistical base rate: P(touch {strike:{price_fmt}} within {horizon}) "
    f"= {stats['base_rate_hitting']:.3f} ({stats['base_rate_hitting']:.1%})\n"
    f"For reference, P(above {strike:{price_fmt}} at expiry) "
    f"= {stats['base_rate_above']:.3f}\n"
    f"Note: P(touch) >= P(above) always. P(touch) ~ 1.0 near spot, "
    f"decreasing with distance. Longer tenors increase P(touch).\n"
    f"This is a first-passage baseline {anchor_tail}. "
    f"Adjust based on current evidence."
)
```

### Step 6: Remove dead code from `base_rates.py`

The following become unused after Steps 1–5 and should be removed:

- `compute_forward_rate()` function (lines 255–303)
- `get_short_rate()` function (lines 223–252)
- `_fetch_dynamic_rate()` function (lines 159–220)
- `get_policy_rate()` function (lines 134–136)
- `_YIELD_TICKERS` dict (lines 151–157)
- `_POLICY_RATES_UPDATED` constant (line 120)
- `FALLBACK_POLICY_RATES` dict (lines 122–131)
- The comment block about policy rates (lines 112–118)
- The comment block about dynamic interest-rate fetching (lines 139–150)
- The `_YF_SUFFIX` constant (line 107)

**Keep**: `FALLBACK_VOL`, `get_annualized_vol`, `_compute_realized_vol` — still needed for the volatility computation. Also keep the consensus provider hook, `_norm`, and all consensus-related code.

### Step 7: Update `fx/__init__.py`

Remove the forward/rate exports that are now dead code.

**Current (lines 1–22):**
```python
from .base_rates import (
    compute_base_rates,
    compute_forward_rate,
    format_base_rate_context,
    get_consensus,
    get_short_rate,
    set_consensus_provider,
)
from .rates import get_spot_rate
from .pairs import get_pair_config, generate_strikes

__all__ = [
    "compute_base_rates",
    "compute_forward_rate",
    "format_base_rate_context",
    "get_consensus",
    "get_short_rate",
    "get_spot_rate",
    "get_pair_config",
    "generate_strikes",
    "set_consensus_provider",
]
```

**New:**
```python
from .base_rates import (
    compute_base_rates,
    format_base_rate_context,
    get_consensus,
    set_consensus_provider,
)
from .rates import get_spot_rate
from .pairs import get_pair_config, generate_strikes

__all__ = [
    "compute_base_rates",
    "format_base_rate_context",
    "get_consensus",
    "get_spot_rate",
    "get_pair_config",
    "generate_strikes",
    "set_consensus_provider",
]
```

### Step 8: Update docstrings in `base_rates.py`

**Module docstring (line 1):**
```python
"""Historical base rate computation for FX probability anchoring.

Computes statistical base rates for currency pair moves using
log-normal assumptions and dynamically fetched realized volatilities.
Uses spot as the distribution center when no consensus provider is
registered. Gives forecasting agents a quantitative anchor to adjust from.
"""
```

**`compute_base_rates()` docstring — update priority list:**
```python
"""Compute statistical base rate for P(price > strike at expiry).

The distribution center is chosen by priority:
  1. Consensus forecast (if a provider is registered and returns a value)
  2. Spot rate (zero directional assumption)
...
"""
```

**`compute_hitting_base_rate()` docstring — same update:**
```python
"""Compute the base rate for a barrier/touch probability.

Uses the first-passage formula for geometric Brownian motion.
The drift target is chosen by priority:
  1. Consensus forecast (if available)
  2. Spot rate (zero drift)

When no consensus is available the drift is zero, which is
equivalent to the reflection-principle formula.
...
"""
```

**`set_consensus_provider()` docstring (line 52) — remove forward mention:**
```python
def set_consensus_provider(provider: ConsensusProvider | None) -> None:
    """Register (or clear) a consensus-rate provider.

    The provider is called with ``(pair, spot, tenor)`` and should return
    ``(consensus_rate, source_label)`` or ``None``.

    When a consensus rate is available it becomes the center of the
    probability distribution.  When absent, spot is used (zero drift).

    Pass ``None`` to remove a previously registered provider.
    """
```

**Update the module-level consensus comment block (lines 20–45)** — remove references to the forward being shown to agents:
```python
# Register a callable to supply consensus forecasts from an external source
# (analyst surveys, internal models, market-implied estimates, etc.).
#
# When set, the consensus rate becomes the center of the probability
# distribution.  When no consensus is available, spot is used (zero drift).
#
# The provider receives (pair, spot, tenor) and should return either:
#   - (consensus_rate, source_label) on success
#   - None to signal "no consensus available for this pair/tenor"
```

### Step 9: Fix tests in `test_base_rates.py`

**`test_atm_strike_gives_half` (line 16)** — Currently fails because forward != spot. With center=spot, ATM strike will give exactly 0.5. **No code change needed** — the assertion `abs(stats["base_rate_above"] - 0.5) < 1e-10` will now pass.

**`test_symmetry` (line 54)** — Currently fails because forward breaks symmetry. With center=spot, symmetry is exact. **No code change needed** — will now pass.

**`test_z_score_known_value` (line 92)** — The expected z_score formula uses a linear approximation `(2.0 / 153.0) / sigma_t`, but the actual code uses `ln(center/strike)`. With center=spot, the log formula is exact. Fix the test to use the exact formula:

```python
def test_z_score_known_value(self):
    """Verify z-score calculation with a known case."""
    with patch("aia_forecaster.fx.base_rates.get_annualized_vol", return_value=(0.10, "fallback")):
        stats = compute_base_rates("USDJPY", spot=153.0, strike=155.0, tenor=Tenor.W1)
        expected_sigma_t = 0.10 * math.sqrt(5 / 252)
        assert abs(stats["sigma_t"] - expected_sigma_t) < 1e-10
        # z_score = -d2 where d2 = (ln(center/strike) - 0.5*sigma_t^2) / sigma_t
        # With center=spot=153.0, strike=155.0:
        d2 = (math.log(153.0 / 155.0) - 0.5 * expected_sigma_t**2) / expected_sigma_t
        expected_z = -d2
        assert abs(stats["z_score"] - expected_z) < 1e-10
```

### Step 10: Fix tests in `test_hitting_mode.py`

**`test_hitting_mode_output` (line 100):** Currently asserts `"reflection-principle"` which doesn't appear in the output. Change to check for `"first-passage"`:

```python
def test_hitting_mode_output(self):
    """Hitting mode should produce barrier-specific context."""
    result = format_base_rate_context(
        "USDJPY", spot=153.0, strike=155.0, tenor=Tenor.W1,
        forecast_mode=ForecastMode.HITTING,
    )
    assert "HITTING" in result or "BARRIER" in result
    assert "P(touch" in result
    assert "first-passage" in result.lower()
```

**`test_above_mode_output` (line 110):** Currently asserts `"normal-distribution"` which doesn't appear. Change to check for `"log-normal"`:

```python
def test_above_mode_output(self):
    """Above mode should produce the original context format."""
    result = format_base_rate_context(
        "USDJPY", spot=153.0, strike=155.0, tenor=Tenor.W1,
        forecast_mode=ForecastMode.ABOVE,
    )
    assert "Statistical base rate" in result
    assert "P(above" in result
    assert "log-normal" in result.lower()
```

---

## What Changes for Agents (Prompt Diffs)

### ABOVE mode — no consensus

**Before:**
```
BASE RATE CONTEXT (statistical anchor):
Current spot: USD/JPY = 150.00
1 week forward: USD/JPY = 149.89 (carry: USD 4.50% vs JPY 0.50%, net -4.00%, src: dynamic/fallback)
Annualized vol: 10.0% (dynamic)
Target: above 152.00 in 1 week
  From forward: +2.11 (+1.41%)
  From spot:    +2.00 (+1.33%)
Historical 1 week range (1-sigma): +/-2.11 JPY
Required move from forward: 1.00 standard deviations
Statistical base rate: P(above 152.00) = 0.159 (15.9%)
Note: Base rate is anchored to forward (carry-adjusted).
This is a log-normal baseline. Adjust based on current evidence.
```

**After:**
```
BASE RATE CONTEXT (statistical anchor):
Current spot: USD/JPY = 150.00
Annualized vol: 10.0% (dynamic)
Target: above 152.00 in 1 week
  From spot: +2.00 (+1.33%)
Historical 1 week range (1-sigma): +/-2.11 JPY
Required move: 0.95 standard deviations
Statistical base rate: P(above 152.00) = 0.172 (17.2%)
Note: No consensus view available. Base rate is anchored to spot.
This is a log-normal baseline. Adjust based on current evidence.
```

Key differences:
- No forward line (no carry details, no interest rate breakdown)
- Single "From spot" line (no redundant "From forward" + "From spot")
- Base rate probability slightly different (centered at spot vs forward)
- Explicit note that no consensus is available

### HITTING mode — no consensus

**Before:**
```
...
Expected drift (forward) is toward this barrier.
...
This is a drift-adjusted first-passage baseline anchored to the forward.
Adjust based on current evidence.
```

**After:**
```
...
(no drift note — drift is zero)
...
This is a first-passage baseline anchored to spot (no consensus view available).
Adjust based on current evidence.
```

### With consensus — both modes

**No change.** Consensus prompts are identical before and after.

---

## Math Impact

### ABOVE mode — no consensus

| Property | Before (forward center) | After (spot center) |
|----------|------------------------|---------------------|
| Center | `S × exp((r_q − r_b) × T)` | `S` |
| Drift | Carry-induced | Zero |
| ATM base rate | ≠ 0.5 (shifted by carry) | = 0.5 (exact, minus −½σ² correction) |
| Symmetry | Broken by carry | Exact |

For USDJPY (large rate differential: USD 4.5% vs JPY 0.5%), the forward is below spot for positive T. This means the forward-centered distribution was already biased bearish on USDJPY. Removing this bias is correct if we don't want carry math acting as a directional view.

### HITTING mode — no consensus

With center=spot, the drift `nu_T = ln(spot/spot) − ½σ² = −½σ²` is purely the log-normal drift correction (negative, tiny). The first-passage formula effectively reduces to the **reflection principle** — symmetric probabilities for barriers equidistant above and below spot. This is the cleanest neutral baseline.

---

## Detailed Todo List

### Phase 1: Core Math — Replace Forward with Spot

Self-contained. All changes in `src/aia_forecaster/fx/base_rates.py`. No external dependencies.

- [x] **1.1** Edit `compute_base_rates()` (line 441–456): Replace the `else` branch
  - Remove: `forward, r_base, r_quote, fwd_source = compute_forward_rate(...)` and `center = forward` / `center_source = "forward"`
  - Add: `center = spot` / `center_source = "spot"`
  - Remove: `forward = None`, `fwd_source = None`, `r_base = None`, `r_quote = None` from the consensus-present branch (no longer needed — those variables aren't returned)
  - Net: the if/else simplifies from 12 lines to 7

- [x] **1.2** Edit `compute_base_rates()` return dict (lines 475–492): Remove 4 keys
  - Remove: `"forward_rate"`, `"forward_source"`, `"r_base"`, `"r_quote"`
  - Keep all other keys unchanged

- [x] **1.3** Edit `compute_hitting_base_rate()` (lines 576–591): Same transformation as 1.1
  - Remove: `compute_forward_rate()` call in the `else` branch
  - Add: `center = spot` / `center_source = "spot"`

- [x] **1.4** Edit `compute_hitting_base_rate()` `base_dict` (lines 597–610): Remove 4 keys
  - Remove: `"forward_rate"`, `"forward_source"`, `"r_base"`, `"r_quote"`

- [x] **1.5** Verify: after 1.1–1.4, `compute_forward_rate` is no longer called anywhere in the two functions. Confirm with a grep.

### Phase 2: Prompt Formatting — Remove Forward Line

Depends on Phase 1 (the dict no longer has forward fields). All changes in `src/aia_forecaster/fx/base_rates.py`.

- [x] **2.1** Rewrite `_format_market_context()` (lines 643–681)
  - Remove the entire `else` branch that renders the forward line (carry details, rate breakdown, src_note parsing)
  - Keep only the consensus branch: if `consensus_rate` and `consensus_source` are present, render the consensus line; otherwise return `""`
  - Function shrinks from ~40 lines to ~10

- [x] **2.2** Update `format_base_rate_context()` — ABOVE mode (lines 749–789)
  - Replace: the single-path rendering that always shows "From {center_label}" + "From spot" with a conditional:
    - If `center_src != "spot"` (consensus): show both "From {consensus_label}" and "From spot" lines, plus "Required move from {label}" and "(consensus view)" note
    - If `center_src == "spot"` (no consensus): show only "From spot" line, "Required move:" without a label, and "No consensus view available" note
  - Remove: the `center_label` variable that mapped `"forward"` to `"forward"` (no longer needed)

- [x] **2.3** Update `format_base_rate_context()` — HITTING mode (lines 702–747)
  - Change the drift note guard from `center is not None and center != spot` to `center_src != "spot" and center != spot` — semantically identical after Phase 1, but clearer
  - Replace `anchor_label` logic:
    - Was: `f"the {center_src}" if center_src != "forward" else "the forward"`
    - Now: if `center_src != "spot"` → `f"anchored to the {center_src}"`, else → `"anchored to spot (no consensus view available)"`
  - Update the final sentence from `"drift-adjusted first-passage baseline"` to `"first-passage baseline"` (drift is only mentioned when consensus provides it)

- [x] **2.4** Verify: run `python -c "from aia_forecaster.fx.base_rates import format_base_rate_context; print(format_base_rate_context('USDJPY', 150.0, 152.0, __import__('aia_forecaster.models', fromlist=['Tenor']).Tenor.W1))"` and confirm output has no forward line, shows "From spot" only, and says "No consensus view available".

### Phase 3: Dead Code Removal

Depends on Phase 1 (forward functions are no longer called). All changes in `src/aia_forecaster/fx/base_rates.py`.

- [x] **3.1** Delete `compute_forward_rate()` (lines 255–303) — 49 lines
  - This was the only consumer of `get_short_rate()` and the interest rate data

- [x] **3.2** Delete `get_short_rate()` (lines 223–252) — 30 lines
  - This was the only consumer of `_fetch_dynamic_rate()` and `FALLBACK_POLICY_RATES`

- [x] **3.3** Delete `_fetch_dynamic_rate()` (lines 159–220) — 62 lines
  - This contained the Yahoo Finance `^IRX` fetching logic

- [x] **3.4** Delete `get_policy_rate()` (lines 134–136) — 3 lines
  - Simple dict lookup, no other callers

- [x] **3.5** Delete data constants and comments:
  - `_YIELD_TICKERS` dict (lines 151–157)
  - `_POLICY_RATES_UPDATED` constant (line 120)
  - `FALLBACK_POLICY_RATES` dict (lines 122–131)
  - Comment block about policy rates (lines 112–118)
  - Comment block about dynamic interest-rate fetching (lines 139–150)
  - `_YF_SUFFIX` constant (line 107)

- [x] **3.6** Verify: grep for `compute_forward_rate`, `get_short_rate`, `_fetch_dynamic_rate`, `get_policy_rate`, `FALLBACK_POLICY_RATES`, `_YIELD_TICKERS`, `_YF_SUFFIX` across the entire `src/` tree — all should return zero matches.

### Phase 4: Export Cleanup

Depends on Phase 3 (deleted functions can't be exported). Changes in `src/aia_forecaster/fx/__init__.py`.

- [x] **4.1** Edit `fx/__init__.py` imports (line 1–8)
  - Remove `compute_forward_rate` and `get_short_rate` from the `from .base_rates import (...)` block

- [x] **4.2** Edit `fx/__init__.py` `__all__` list (lines 12–22)
  - Remove `"compute_forward_rate"` and `"get_short_rate"` from the list

- [x] **4.3** Verify: `python -c "from aia_forecaster.fx import __all__; print(__all__)"` — should not contain `compute_forward_rate` or `get_short_rate`.

### Phase 5: Docstring Updates

No dependencies on other phases (can run in parallel with Phase 3–4). All changes in `src/aia_forecaster/fx/base_rates.py`.

- [x] **5.1** Update module docstring (line 1–6)
  - Change `"Falls back to static estimates when market data is unavailable."` to `"Uses spot as the distribution center when no consensus provider is registered."`

- [x] **5.2** Update the module-level consensus comment block (lines 20–45)
  - Remove: `"The carry-adjusted forward is still computed internally and shown to agents for context."`
  - Add: `"When no consensus is available, spot is used (zero drift)."`
  - Remove the paragraph about the forward being kept for context

- [x] **5.3** Update `set_consensus_provider()` docstring (lines 53–63)
  - Remove: `"The forward is still computed and shown to agents for carry context."`
  - Change: `"When a consensus rate is available it replaces the forward..."` to `"When a consensus rate is available it becomes the center of the probability distribution. When absent, spot is used (zero drift)."`

- [x] **5.4** Update `compute_base_rates()` docstring (lines 412–436)
  - Change priority list from `"1. Consensus forecast ... 2. FX forward rate (carry-adjusted)"` to `"1. Consensus forecast ... 2. Spot rate (zero directional assumption)"`
  - Remove: `"The forward is always computed for carry context..."`
  - Remove: `forward_rate, forward_source, r_base, r_quote` from the Returns docstring

- [x] **5.5** Update `compute_hitting_base_rate()` docstring (lines 551–571)
  - Change priority list from `"1. Consensus forecast ... 2. FX forward rate (carry-adjusted)"` to `"1. Consensus forecast ... 2. Spot rate (zero drift)"`
  - Update: `"When no rate data or consensus is available the drift is zero"` → `"When no consensus is available the drift is zero"`
  - Remove: `forward_rate, forward_source, r_base, r_quote` from the Returns docstring

### Phase 6: Test Fixes

Depends on Phases 1–2 (the math and formatting changes). Changes in `tests/test_base_rates.py` and `tests/test_hitting_mode.py`.

- [x] **6.1** `test_base_rates.py` — `test_atm_strike_gives_half` (line 16)
  - **No code change needed.** Currently fails because forward shifts center away from spot. With center=spot, the assertion `abs(stats["base_rate_above"] - 0.5) < 1e-10` will pass. Confirm by running the test.

- [x] **6.2** `test_base_rates.py` — `test_symmetry` (line 54)
  - **No code change needed.** Currently fails because forward breaks symmetry. With center=spot, `P(above spot+x) + P(above spot-x) ≈ 1.0` holds exactly. Confirm by running.

- [x] **6.3** `test_base_rates.py` — `test_z_score_known_value` (line 92)
  - **Code change needed.** The test's expected z_score uses a linear approximation `(2.0 / 153.0) / sigma_t` that doesn't match the log-normal formula.
  - Replace the expected z_score calculation with the exact log-normal formula:
    ```python
    d2 = (math.log(153.0 / 155.0) - 0.5 * expected_sigma_t**2) / expected_sigma_t
    expected_z = -d2
    ```
  - Also needs to mock `get_consensus` to return `None` (or ensure no consensus provider is registered) so we get deterministic spot-centered results. Currently uses `patch("...get_annualized_vol")` but not consensus — verify this is sufficient since the default `_consensus_provider` is `None`.

- [x] **6.4** `test_hitting_mode.py` — `test_hitting_mode_output` (line 100)
  - Change assertion from `assert "reflection-principle" in result.lower()` to `assert "first-passage" in result.lower()`
  - The string `"reflection-principle"` never appeared in the formatted output (it was in a docstring), so this test was already broken. The new output contains `"first-passage baseline"`.

- [x] **6.5** `test_hitting_mode.py` — `test_above_mode_output` (line 110)
  - Change assertion from `assert "normal-distribution" in result.lower()` to `assert "log-normal" in result.lower()`
  - Same issue: `"normal-distribution"` never appeared in output. The actual output says `"log-normal baseline"`.

- [x] **6.6** Run full test suite and confirm results:
  - `python -m pytest tests/test_base_rates.py -v` — all 19 tests should pass (3 previously-failing now pass)
  - `python -m pytest tests/test_hitting_mode.py -v` — all tests should pass (2 previously-failing now pass)

### Phase 7: Cross-Codebase Verification

Depends on all previous phases. No code changes — just checks.

- [x] **7.1** Grep for any remaining references to removed symbols across the entire repo:
  - `grep -r "compute_forward_rate" . --include="*.py"` — should return 0 matches
  - `grep -r "get_short_rate" . --include="*.py"` — 0 matches
  - `grep -r "FALLBACK_POLICY_RATES" . --include="*.py"` — 0 matches
  - `grep -r "_fetch_dynamic_rate" . --include="*.py"` — 0 matches
  - `grep -r "get_policy_rate" . --include="*.py"` — 0 matches
  - `grep -r "_YIELD_TICKERS" . --include="*.py"` — 0 matches

- [x] **7.2** Grep for `"forward"` in `base_rates.py` — should only appear in:
  - The word "forward" in comments/docstrings that remain relevant (e.g., "forward guidance" in prompt templates elsewhere — but those are in `forecaster.py`, not `base_rates.py`)
  - Confirm: zero occurrences of `forward_rate`, `forward_source`, `fwd_source`, `fwd`, or `compute_forward` in `base_rates.py`

- [x] **7.3** Verify `agents/forecaster.py` is unaffected:
  - It calls `format_base_rate_context()` which still exists and returns a string
  - It does NOT import `compute_forward_rate`, `get_short_rate`, or access dict fields `forward_rate` / `r_base` / `r_quote`
  - No changes needed — just confirm via grep

- [x] **7.4** Verify `agents/supervisor.py` is unaffected:
  - It does not import from `fx.base_rates` at all
  - Its `_build_consensus_causal_summary()` is about agent consensus (different concept), not the consensus provider
  - No changes needed — confirm

- [x] **7.5** Verify `fx/surface.py`, `fx/explanation.py`, `fx/pdf_report.py` are unaffected:
  - None of them import or reference `compute_forward_rate`, `forward_rate`, `r_base`, `r_quote`
  - Confirm via grep

- [x] **7.6** Verify the `company.example/` consensus provider still works:
  - `company.example/consensus.py` returns `None` → system uses spot (correct)
  - `company.example/consensus_sample.py` returns `(rate, label)` → system uses consensus (unchanged behavior)
  - `company.example/__init__.py` calls `set_consensus_provider(get_consensus)` → still valid

- [x] **7.7** Run `ruff check src/aia_forecaster/fx/base_rates.py` — no lint errors (no unused imports, no undefined names)

- [x] **7.8** Run `ruff check src/aia_forecaster/fx/__init__.py` — no lint errors

- [x] **7.9** Run the full test suite: `python -m pytest tests/ -v`
  - All tests in `test_base_rates.py` pass (19/19)
  - All tests in `test_hitting_mode.py` pass
  - No regressions in other test files

### Phase 8: Memory & Documentation Updates

Depends on all code being finalized. Non-code changes.

- [x] **8.1** Update auto-memory file `MEMORY.md`
  - Remove the "Forward Rate" subsection under "Key Architecture: Base Rate System"
  - Remove references to `compute_forward_rate()`, `get_short_rate()`, `FALLBACK_POLICY_RATES`, `^IRX`, rate caching
  - Update "Key Design Decisions" to reflect: forward removed, center=spot when no consensus
  - Update "File Structure" entry for `base_rates.py` — remove "forward rates" from its description

- [x] **8.2** Update `research.md` — Consensus Provider report (Section 4, 6, 11)
  - Section 4: Update "Where Consensus Enters the Math" to reflect center=spot (not forward) as fallback
  - Section 6: Simplify the "Fallback Cascade" diagram — remove the forward tier entirely
  - Section 11: Remove edge case #2 about `forward_rate` being `None` (field no longer exists)
  - Update the data flow diagram in Section 12

- [x] **8.3** Verify `CLAUDE.md` (project-level) — no references to forward rate that need updating
  - The CLAUDE.md discusses the consensus provider system but shouldn't reference forward internals

---

## Execution Order

Phases can be partially parallelized:

```
Phase 1 (core math)
  │
  ├──► Phase 2 (prompt formatting) ──► Phase 3 (dead code) ──► Phase 4 (exports)
  │                                                                │
  │                                                                ▼
  ├──► Phase 5 (docstrings)                              Phase 7 (verification)
  │                                                                │
  └──► Phase 6 (test fixes) ────────────────────────────────────────┘
                                                                   │
                                                                   ▼
                                                          Phase 8 (docs/memory)
```

Total: ~200 lines removed, ~50 lines added. Net reduction.
