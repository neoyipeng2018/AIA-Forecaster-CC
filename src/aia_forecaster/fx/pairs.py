"""Currency pair definitions and strike generation."""

from __future__ import annotations

from dataclasses import dataclass

from aia_forecaster.models import ForecastMode, Tenor


@dataclass
class PairConfig:
    pair: str
    base: str
    quote: str
    pip_size: float  # Smallest meaningful price move
    typical_daily_range: float  # Typical daily range in price units


# Supported pairs with their characteristics
PAIR_CONFIGS: dict[str, PairConfig] = {
    "USDJPY": PairConfig(
        pair="USDJPY", base="USD", quote="JPY",
        pip_size=0.01, typical_daily_range=1.0,
    ),
    "EURUSD": PairConfig(
        pair="EURUSD", base="EUR", quote="USD",
        pip_size=0.0001, typical_daily_range=0.008,
    ),
    "GBPUSD": PairConfig(
        pair="GBPUSD", base="GBP", quote="USD",
        pip_size=0.0001, typical_daily_range=0.010,
    ),
}

# Default tenors
DEFAULT_TENORS = [Tenor.D1, Tenor.W1, Tenor.M1, Tenor.M3, Tenor.M6]


def get_pair_config(pair: str) -> PairConfig:
    """Get configuration for a currency pair."""
    pair = pair.upper()
    if pair not in PAIR_CONFIGS:
        raise ValueError(f"Unsupported pair: {pair}. Supported: {list(PAIR_CONFIGS.keys())}")
    return PAIR_CONFIGS[pair]


def generate_strikes(
    spot: float,
    pair: str = "USDJPY",
    num_strikes: int = 11,
    forecast_mode: ForecastMode = ForecastMode.ABOVE,
) -> list[float]:
    """Generate strike prices around the current spot rate.

    For USDJPY at spot=155.00 with num_strikes=11:
    → [150.0, 151.0, 152.0, 153.0, 154.0, 155.0, 156.0, 157.0, 158.0, 159.0, 160.0]

    In hitting mode, forces num_strikes to be odd for symmetry around spot.

    Args:
        spot: Current spot rate.
        pair: Currency pair string.
        num_strikes: Number of strikes to generate (odd number centers on spot).
        forecast_mode: Forecast mode — hitting forces odd num_strikes for symmetry.

    Returns:
        Sorted list of strike prices.
    """
    config = get_pair_config(pair)

    # In hitting mode, force odd num_strikes so spot is centered
    if forecast_mode == ForecastMode.HITTING and num_strikes % 2 == 0:
        num_strikes += 1

    # Use percentage-based offsets scaled to the pair
    # For USDJPY (~155): ±1 yen per strike → roughly ±0.65%
    # For EURUSD (~1.08): ±0.01 per strike → roughly ±0.9%
    if pair == "USDJPY":
        step = 1.0  # 1 yen steps
    else:
        step = config.typical_daily_range  # 1 daily range per step

    half = num_strikes // 2
    center = round(spot, 2) if pair == "USDJPY" else round(spot, 4)
    strikes = [round(center + (i - half) * step, 4) for i in range(num_strikes)]

    return strikes
