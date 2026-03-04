"""Sample consensus provider with hardcoded forecasts for demonstration.

This shows a working example you can run immediately::

    # In company/__init__.py, switch the import:
    from .consensus_sample import get_consensus

Then run::

    poetry run forecast USDJPY -v

You should see "sample_hardcoded" as the source label in the agent context,
and the distribution will center on the consensus rate instead of the forward.
"""

from __future__ import annotations

# Hardcoded analyst consensus forecasts by (pair, tenor).
# Replace these with your own data source.
_SAMPLE_FORECASTS: dict[str, dict[str, float]] = {
    "USDJPY": {
        "1D": 149.80,
        "1W": 149.50,
        "1M": 148.00,
        "3M": 145.00,
        "6M": 142.00,
    },
    "EURUSD": {
        "1D": 1.0450,
        "1W": 1.0480,
        "1M": 1.0550,
        "3M": 1.0700,
        "6M": 1.0900,
    },
    "GBPUSD": {
        "1D": 1.2650,
        "1W": 1.2680,
        "1M": 1.2750,
        "3M": 1.2900,
        "6M": 1.3050,
    },
}

_SOURCE_LABEL = "sample_hardcoded"


def get_consensus(
    pair: str, spot: float, tenor: str
) -> tuple[float, str] | None:
    """Return a hardcoded consensus rate for demonstration purposes.

    Returns None for unsupported pair/tenor combos, which makes the
    system fall back to the forward rate for those cells.
    """
    pair_forecasts = _SAMPLE_FORECASTS.get(pair)
    if pair_forecasts is None:
        return None

    rate = pair_forecasts.get(tenor)
    if rate is None:
        return None

    return (rate, _SOURCE_LABEL)
