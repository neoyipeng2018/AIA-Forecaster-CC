"""Consensus provider — plug in your own FX consensus forecasts.

Implement ``get_consensus()`` to return a directional consensus rate
for a given pair and tenor.  The system will use it as the distribution
center instead of the forward rate (carry math).

See ``consensus_sample.py`` for a working example with hardcoded data.
"""

from __future__ import annotations


def get_consensus(
    pair: str, spot: float, tenor: str
) -> tuple[float, str] | None:
    """Return a directional consensus rate for *pair* at the given *tenor*.

    Parameters
    ----------
    pair : str
        Currency pair, e.g. ``"USDJPY"``.
    spot : float
        Current spot rate.
    tenor : str
        Forecast horizon, e.g. ``"1W"``, ``"1M"``, ``"3M"``.

    Returns
    -------
    tuple[float, str] | None
        ``(consensus_rate, source_label)`` — the rate the market/analysts
        expect and a label describing the source (e.g. ``"bloomberg_survey"``,
        ``"internal_model"``).  Return ``None`` if no consensus is available
        for this pair/tenor, and the system will fall back to the forward rate.
    """
    # TODO: Replace with your implementation.
    # Example: query Bloomberg FXFC or an internal API.
    return None
