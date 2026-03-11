"""Market context for FX agent prompts.

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
    pair: str, spot: float, tenor: Tenor,
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
    """Produce a market context block for LLM agent prompts.

    Gives the agent spot, strike, distance, tenor, and consensus
    information. Agents estimate probabilities themselves from evidence.

    Returns a non-empty string for any valid pair.
    """
    pair = pair.upper()
    horizon = tenor.label
    base, quote = pair[:3], pair[3:]
    price_fmt = ".2f" if "JPY" in pair else ".4f"

    consensus_result = get_consensus(pair, spot, tenor)
    if consensus_result is not None:
        consensus_rate: float | None = consensus_result[0]
        consensus_source: str | None = consensus_result[1]
        center = consensus_rate
        center_source = consensus_source
    else:
        consensus_rate = None
        consensus_source = None
        center = spot
        center_source = "spot"

    market_note = _format_market_context(
        consensus_rate, consensus_source, base, quote, price_fmt, horizon,
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
