"""Historical base rate computation for FX probability anchoring.

Computes statistical base rates for currency pair moves using
log-normal assumptions and hardcoded annualized volatilities.
Gives forecasting agents a quantitative anchor to adjust from.
"""

from __future__ import annotations

import math
from statistics import NormalDist

from aia_forecaster.models import Tenor

# Hardcoded annualized volatilities (stable enough as anchors)
ANNUALIZED_VOL: dict[str, float] = {
    "USDJPY": 0.10,  # ~10%
    "EURUSD": 0.08,  # ~8%
    "GBPUSD": 0.09,  # ~9%
}

# Trading days per tenor
TENOR_DAYS: dict[Tenor, int] = {
    Tenor.D1: 1,
    Tenor.W1: 5,
    Tenor.M1: 21,
    Tenor.M3: 63,
    Tenor.M6: 126,
}

_norm = NormalDist(0, 1)


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
        base_rate_above, tenor_range_1sigma.
    """
    pair = pair.upper()
    annual_vol = ANNUALIZED_VOL.get(pair)
    if annual_vol is None:
        raise ValueError(
            f"No volatility data for {pair}. "
            f"Supported: {list(ANNUALIZED_VOL.keys())}"
        )

    days = TENOR_DAYS[tenor]

    # Tenor-scaled volatility: σ_T = σ_annual * sqrt(days / 252)
    sigma_t = annual_vol * math.sqrt(days / 252)

    # 1-sigma range in price units
    tenor_range_1sigma = spot * sigma_t

    # Required move
    move_required = strike - spot
    move_pct = move_required / spot

    # Z-score: how many standard deviations is this move?
    z_score = move_pct / sigma_t if sigma_t > 0 else 0.0

    # Base rate: P(price > strike) = 1 - Φ(z)
    base_rate_above = 1 - _norm.cdf(z_score)

    return {
        "sigma_t": sigma_t,
        "move_required": move_required,
        "move_pct": move_pct,
        "z_score": z_score,
        "base_rate_above": base_rate_above,
        "tenor_range_1sigma": tenor_range_1sigma,
    }


def format_base_rate_context(
    pair: str,
    spot: float,
    strike: float,
    tenor: Tenor,
) -> str:
    """Produce a human-readable base rate context block for LLM prompts.

    Returns an empty string if the pair is unsupported.
    """
    pair = pair.upper()
    if pair not in ANNUALIZED_VOL:
        return ""

    stats = compute_base_rates(pair, spot, strike, tenor)

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

    return (
        f"BASE RATE CONTEXT (statistical anchor):\n"
        f"Current spot: {base}/{quote} = {spot:{price_fmt}}\n"
        f"Target: {direction} {strike:{price_fmt}} in {horizon} "
        f"→ requires {move_sign}{stats['move_required']:{price_fmt}} move "
        f"({move_sign}{stats['move_pct']:.2%})\n"
        f"Historical {horizon} range (1σ): ±{stats['tenor_range_1sigma']:{price_fmt}} {quote}\n"
        f"Required move: {abs(stats['z_score']):.2f} standard deviations\n"
        f"Statistical base rate: P({direction} {strike:{price_fmt}}) "
        f"≈ {stats['base_rate_above']:.3f} ({stats['base_rate_above']:.1%})\n"
        f"This is a normal-distribution baseline. Adjust based on current evidence."
    )
