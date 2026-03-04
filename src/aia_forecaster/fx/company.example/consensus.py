"""Example consensus provider.

Copy this folder to ``company/`` and implement ``get_consensus`` with your
internal data source (e.g. Bloomberg, Reuters, internal model).

Wire it up at pipeline startup::

    from aia_forecaster.fx import set_consensus_provider
    from aia_forecaster.fx.company.consensus import get_consensus

    set_consensus_provider(get_consensus)
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
    #
    # if pair == "USDJPY" and tenor == "3M":
    #     return (148.50, "bloomberg_survey")
    #
    return None
