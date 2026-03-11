"""Historical base rate computation for FX probability anchoring.

Computes statistical base rates for currency pair moves using
log-normal assumptions and dynamically fetched realized volatilities.
Uses spot as the distribution center when no consensus provider is
registered. Gives forecasting agents a quantitative anchor to adjust from.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable
from statistics import NormalDist

from aia_forecaster.models import ForecastMode, Tenor

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Consensus provider hook
# ---------------------------------------------------------------------------
# Register a callable to supply consensus forecasts from an external source
# (analyst surveys, internal models, market-implied estimates, etc.).
#
# When set, the consensus rate becomes the center of the probability
# distribution.  When no consensus is available, spot is used (zero drift).
#
# The provider receives (pair, spot, tenor) and should return either:
#   - (consensus_rate, source_label) on success
#   - None to signal "no consensus available for this pair/tenor"
#
# Example:
#
#     from aia_forecaster.fx.base_rates import set_consensus_provider
#
#     def my_consensus(pair: str, spot: float, tenor: Tenor) -> tuple[float, str] | None:
#         rate = internal_model.get_consensus(pair, str(tenor))
#         if rate is None:
#             return None
#         return rate, "analyst_consensus"
#
#     set_consensus_provider(my_consensus)

ConsensusProvider = Callable[[str, float, "Tenor"], tuple[float, str] | None]

_consensus_provider: ConsensusProvider | None = None


def set_consensus_provider(provider: ConsensusProvider | None) -> None:
    """Register (or clear) a consensus-rate provider.

    The provider is called with ``(pair, spot, tenor)`` and should return
    ``(consensus_rate, source_label)`` or ``None``.

    When a consensus rate is available it becomes the center of the
    probability distribution.  When absent, spot is used (zero drift).

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

    Returns (consensus_rate, source_label) or None if no provider is
    registered or the provider has no data for this pair/tenor.
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

# Static fallback volatilities — used when live data is unavailable.
FALLBACK_VOL: dict[str, float] = {
    "USDJPY": 0.10,
    "EURUSD": 0.08,
    "GBPUSD": 0.09,
    "AUDUSD": 0.10,
    "NZDUSD": 0.11,
    "USDCAD": 0.07,
    "USDCHF": 0.08,
}

# Backward-compatible alias: code that imports ANNUALIZED_VOL will still work,
# but the values may be overridden at runtime by dynamic lookups.
ANNUALIZED_VOL = FALLBACK_VOL

_YF_SUFFIX = "=X"

_norm = NormalDist(0, 1)


def _compute_realized_vol(pair: str, lookback_days: int = 60) -> float | None:
    """Fetch recent daily closes from Yahoo Finance and compute annualized vol.

    Uses log-returns standard deviation scaled to annualized.  The lookback
    window defaults to 60 trading days (~3 months) which balances recency
    against noise.

    Returns None on any failure (network, missing data, etc.).
    """
    try:
        import yfinance as yf  # noqa: F811 — lazy import to keep startup fast
    except ImportError:
        logger.debug("yfinance not installed; skipping dynamic vol for %s", pair)
        return None

    ticker = f"{pair.upper()}{_YF_SUFFIX}"
    try:
        data = yf.download(
            ticker,
            period=f"{lookback_days + 10}d",  # small buffer for weekends/holidays
            interval="1d",
            progress=False,
            auto_adjust=True,
        )
        if data is None or len(data) < 10:
            logger.warning("Insufficient price data for %s (%d rows)", pair, len(data) if data is not None else 0)
            return None

        closes = data["Close"].dropna()
        if hasattr(closes, "squeeze"):
            closes = closes.squeeze()
        if len(closes) < 10:
            return None

        # Log returns
        import numpy as np

        log_returns = np.diff(np.log(closes.values.astype(float)))
        daily_vol = float(np.std(log_returns, ddof=1))
        annualized = daily_vol * math.sqrt(252)

        # Sanity check: reject implausible values
        if annualized < 0.01 or annualized > 0.50:
            logger.warning(
                "Computed vol %.4f for %s looks implausible; discarding", annualized, pair
            )
            return None

        return annualized
    except Exception:
        logger.warning("Failed to fetch dynamic vol for %s", pair, exc_info=True)
        return None


def get_annualized_vol(pair: str) -> tuple[float, str]:
    """Return the best available annualized volatility for *pair*.

    Priority:
      1. Freshly computed realized vol from Yahoo Finance
      2. Static fallback from FALLBACK_VOL

    Returns:
        Tuple of (annualized_vol, source) where source is "dynamic" or "fallback".

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
                    "%s realized vol %.1f%% differs from fallback %.1f%% by %.0f%% "
                    "— using realized",
                    pair,
                    realized * 100,
                    fallback * 100,
                    delta * 100,
                )
        else:
            logger.info(
                "%s dynamic vol computed: %.1f%% (no static fallback existed)",
                pair,
                realized * 100,
            )
        return realized, "dynamic"

    if pair in FALLBACK_VOL:
        logger.debug("Using static fallback vol for %s: %.1f%%", pair, FALLBACK_VOL[pair] * 100)
        return FALLBACK_VOL[pair], "fallback"

    raise ValueError(
        f"No volatility data for {pair}. "
        f"Dynamic fetch failed and no static fallback available."
    )


def compute_base_rates(
    pair: str,
    spot: float,
    strike: float,
    tenor: Tenor,
) -> dict:
    """Compute statistical base rate for P(price > strike at expiry).

    The distribution center is chosen by priority:
      1. Consensus forecast (if a provider is registered and returns a value)
      2. Spot rate (zero directional assumption)

    The probability is computed under a log-normal model:
        d2 = (ln(C/K) − ½σ_t²) / σ_t
        P(S_T > K) = Φ(d2)
    where C is the distribution center (consensus or spot).

    Args:
        pair: Currency pair (e.g. "USDJPY").
        spot: Current spot rate.
        strike: Target price level.
        tenor: Forecast horizon.

    Returns:
        Dict with keys: sigma_t, move_required, move_pct, z_score,
        base_rate_above, tenor_range_1sigma, annualized_vol, vol_source,
        center, center_source, consensus_rate, consensus_source.
    """
    pair = pair.upper()
    annual_vol, vol_source = get_annualized_vol(pair)

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

    days = tenor.trading_days
    sigma_t = annual_vol * math.sqrt(days / 252)
    tenor_range_1sigma = spot * sigma_t

    move_required = strike - center
    move_pct = move_required / spot

    if sigma_t > 0 and strike > 0 and center > 0:
        d2 = (math.log(center / strike) - 0.5 * sigma_t**2) / sigma_t
        base_rate_above = _norm.cdf(d2)
        z_score = -d2
    else:
        z_score = 0.0
        base_rate_above = 0.5

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


def _first_passage_probability(
    h: float,
    nu_T: float,
    sigma_t: float,
) -> float:
    """First-passage probability for Brownian motion with drift.

    Computes P(process touches level h before time T) where the log-price
    process has total drift nu_T and tenor-scaled volatility sigma_t.

    For barrier above spot (h > 0):
        P = Φ((nu_T − h)/σ_t) + exp(2·nu_T·h/σ_t²) · Φ((−nu_T − h)/σ_t)

    For barrier below spot (h < 0), uses the minimum of the process:
        P = Φ((−nu_T + h)/σ_t) + exp(2·nu_T·h/σ_t²) · Φ((nu_T + h)/σ_t)

    Reduces to 2·(1 − Φ(|h|/σ_t)) when nu_T = 0 (zero drift / reflection
    principle), which matches the previous implementation.

    Args:
        h: Log-distance to barrier, ln(barrier/spot). Positive = above spot.
        nu_T: Total log-space drift = ln(F/S) − ½·σ_t².
        sigma_t: Tenor-scaled volatility (σ_annual · √(T_trading)).
    """
    if abs(h) < 1e-12:
        return 1.0  # barrier at spot — always touched

    if sigma_t <= 1e-12:
        # No volatility: only hits if drift carries us there
        return 1.0 if (h > 0 and nu_T >= h) or (h < 0 and nu_T <= h) else 0.0

    # Clamp the exponent to avoid overflow (exp(700) ~ 1e304)
    exponent = 2.0 * nu_T * h / (sigma_t**2)
    exponent = max(-500.0, min(500.0, exponent))

    if h > 0:
        # Barrier above spot
        d1 = (nu_T - h) / sigma_t
        d2 = (-nu_T - h) / sigma_t
        p = _norm.cdf(d1) + math.exp(exponent) * _norm.cdf(d2)
    else:
        # Barrier below spot (h < 0)
        abs_h = -h
        d1 = (-nu_T - abs_h) / sigma_t
        d2 = (nu_T - abs_h) / sigma_t
        p = _norm.cdf(d1) + math.exp(exponent) * _norm.cdf(d2)

    return max(0.0, min(1.0, p))


def compute_hitting_base_rate(
    pair: str,
    spot: float,
    barrier: float,
    tenor: Tenor,
) -> dict:
    """Compute the base rate for a barrier/touch probability.

    Uses the first-passage formula for geometric Brownian motion.
    The drift target is chosen by priority:
      1. Consensus forecast (if available)
      2. Spot rate (zero drift)

    When no consensus is available the drift is zero, which is
    equivalent to the reflection-principle formula.

    Args:
        pair: Currency pair.
        spot: Current spot rate.
        barrier: Barrier/touch level.
        tenor: Forecast horizon.

    Returns:
        Dict with keys: sigma_t, distance_pct, base_rate_hitting, base_rate_above,
        tenor_range_1sigma, annualized_vol, vol_source,
        center, center_source, consensus_rate, consensus_source.
    """
    pair = pair.upper()
    annual_vol, vol_source = get_annualized_vol(pair)

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

    days = tenor.trading_days
    sigma_t = annual_vol * math.sqrt(days / 252)
    tenor_range_1sigma = spot * sigma_t

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

    if barrier <= 0 or spot <= 0:
        return {
            **base_dict,
            "distance_pct": 0.0,
            "base_rate_hitting": 1.0,
            "base_rate_above": 0.5,
        }

    # Log-distance to barrier and drift toward center
    h = math.log(barrier / spot)
    nu_T = math.log(center / spot) - 0.5 * sigma_t**2

    base_rate_hitting = _first_passage_probability(h, nu_T, sigma_t)

    # P(above at expiry) using center
    if sigma_t > 0 and center > 0:
        d2 = (math.log(center / barrier) - 0.5 * sigma_t**2) / sigma_t
        base_rate_above = _norm.cdf(d2)
    else:
        base_rate_above = 0.5

    distance_pct = (barrier - spot) / spot

    return {
        **base_dict,
        "distance_pct": distance_pct,
        "base_rate_hitting": base_rate_hitting,
        "base_rate_above": base_rate_above,
    }


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


def format_base_rate_context(
    pair: str,
    spot: float,
    strike: float,
    tenor: Tenor,
    forecast_mode: ForecastMode = ForecastMode.ABOVE,
) -> str:
    """Produce a human-readable base rate context block for LLM prompts.

    Returns an empty string if the pair is entirely unsupported
    (no dynamic vol and no fallback).
    """
    pair = pair.upper()

    horizon = tenor.label
    base, quote = pair[:3], pair[3:]
    price_fmt = ".2f" if "JPY" in pair else ".4f"

    if forecast_mode == ForecastMode.HITTING:
        try:
            stats = compute_hitting_base_rate(pair, spot, strike, tenor)
        except ValueError:
            return ""

        center = stats["center"]
        center_src = stats["center_source"]
        direction = "above" if strike >= spot else "below"
        distance_pct = stats["distance_pct"]
        dist_sign = "+" if distance_pct >= 0 else ""

        vol_note = (
            f"Annualized vol: {stats['annualized_vol']:.1%} ({stats['vol_source']})\n"
        )
        market_note = _format_market_context(stats, base, quote, spot, price_fmt, horizon)

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

    # Default: ABOVE mode
    try:
        stats = compute_base_rates(pair, spot, strike, tenor)
    except ValueError:
        return ""

    center = stats["center"]
    center_src = stats["center_source"]
    direction = "above" if strike >= center else "below"

    move_from_spot = strike - spot
    spot_sign = "+" if move_from_spot >= 0 else ""

    vol_note = (
        f"Annualized vol: {stats['annualized_vol']:.1%} ({stats['vol_source']})\n"
    )
    market_note = _format_market_context(stats, base, quote, spot, price_fmt, horizon)

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
