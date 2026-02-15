"""Historical base rate computation for FX probability anchoring.

Computes statistical base rates for currency pair moves using
log-normal assumptions and dynamically fetched realized volatilities.
Falls back to static estimates when market data is unavailable.
Gives forecasting agents a quantitative anchor to adjust from.
"""

from __future__ import annotations

import logging
import math
import time
from statistics import NormalDist

from aia_forecaster.models import Tenor

logger = logging.getLogger(__name__)

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

# Yahoo Finance ticker format for FX pairs (e.g., "USDJPY=X")
_YF_SUFFIX = "=X"

# Cache: pair -> (annualized_vol, timestamp)
_vol_cache: dict[str, tuple[float, float]] = {}
_CACHE_TTL = 3600  # 1 hour — vol doesn't change dramatically intra-session

# Trading days per tenor
TENOR_DAYS: dict[Tenor, int] = {
    Tenor.D1: 1,
    Tenor.W1: 5,
    Tenor.M1: 21,
    Tenor.M3: 63,
    Tenor.M6: 126,
}

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


def get_annualized_vol(pair: str) -> float:
    """Return the best available annualized volatility for *pair*.

    Priority:
      1. Cached dynamic vol (if fresh)
      2. Freshly computed realized vol from Yahoo Finance
      3. Static fallback from FALLBACK_VOL

    Raises ValueError if the pair has no fallback and dynamic fetch fails.
    """
    pair = pair.upper()

    # 1. Check cache
    if pair in _vol_cache:
        vol, ts = _vol_cache[pair]
        if time.time() - ts < _CACHE_TTL:
            return vol

    # 2. Try dynamic computation
    realized = _compute_realized_vol(pair)
    if realized is not None:
        _vol_cache[pair] = (realized, time.time())
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
        return realized

    # 3. Static fallback
    if pair in FALLBACK_VOL:
        logger.debug("Using static fallback vol for %s: %.1f%%", pair, FALLBACK_VOL[pair] * 100)
        return FALLBACK_VOL[pair]

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
    """Compute historical base rate statistics for a given cell.

    Args:
        pair: Currency pair (e.g. "USDJPY").
        spot: Current spot rate.
        strike: Target price level.
        tenor: Forecast horizon.

    Returns:
        Dict with keys: sigma_t, move_required, move_pct, z_score,
        base_rate_above, tenor_range_1sigma, annualized_vol, vol_source.
    """
    pair = pair.upper()
    annual_vol = get_annualized_vol(pair)

    # Track whether we used dynamic or fallback
    vol_source = "dynamic" if pair in _vol_cache else "fallback"

    days = TENOR_DAYS[tenor]

    # Tenor-scaled volatility: sigma_T = sigma_annual * sqrt(days / 252)
    sigma_t = annual_vol * math.sqrt(days / 252)

    # 1-sigma range in price units
    tenor_range_1sigma = spot * sigma_t

    # Required move
    move_required = strike - spot
    move_pct = move_required / spot

    # Z-score: how many standard deviations is this move?
    z_score = move_pct / sigma_t if sigma_t > 0 else 0.0

    # Base rate: P(price > strike) = 1 - phi(z)
    base_rate_above = 1 - _norm.cdf(z_score)

    return {
        "sigma_t": sigma_t,
        "move_required": move_required,
        "move_pct": move_pct,
        "z_score": z_score,
        "base_rate_above": base_rate_above,
        "tenor_range_1sigma": tenor_range_1sigma,
        "annualized_vol": annual_vol,
        "vol_source": vol_source,
    }


def format_base_rate_context(
    pair: str,
    spot: float,
    strike: float,
    tenor: Tenor,
) -> str:
    """Produce a human-readable base rate context block for LLM prompts.

    Returns an empty string if the pair is entirely unsupported
    (no dynamic vol and no fallback).
    """
    pair = pair.upper()

    try:
        stats = compute_base_rates(pair, spot, strike, tenor)
    except ValueError:
        return ""

    base, quote = pair[:3], pair[3:]
    direction = "above" if strike >= spot else "below"
    move_sign = "+" if stats["move_required"] >= 0 else ""

    tenor_map = {
        Tenor.D1: "1 day",
        Tenor.W1: "1 week",
        Tenor.M1: "1 month",
        Tenor.M3: "3 months",
        Tenor.M6: "6 months",
    }
    horizon = tenor_map.get(tenor, tenor.value)

    # Format price precision based on pair
    price_fmt = ".2f" if "JPY" in pair else ".4f"

    vol_note = (
        f"Annualized vol: {stats['annualized_vol']:.1%} ({stats['vol_source']})\n"
    )

    return (
        f"BASE RATE CONTEXT (statistical anchor):\n"
        f"Current spot: {base}/{quote} = {spot:{price_fmt}}\n"
        f"{vol_note}"
        f"Target: {direction} {strike:{price_fmt}} in {horizon} "
        f"-> requires {move_sign}{stats['move_required']:{price_fmt}} move "
        f"({move_sign}{stats['move_pct']:.2%})\n"
        f"Historical {horizon} range (1-sigma): +/-{stats['tenor_range_1sigma']:{price_fmt}} {quote}\n"
        f"Required move: {abs(stats['z_score']):.2f} standard deviations\n"
        f"Statistical base rate: P({direction} {strike:{price_fmt}}) "
        f"= {stats['base_rate_above']:.3f} ({stats['base_rate_above']:.1%})\n"
        f"This is a normal-distribution baseline. Adjust based on current evidence."
    )
