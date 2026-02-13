"""Platt scaling calibration to correct LLM hedging bias.

LLMs systematically hedge toward 0.5 due to RLHF training. Platt scaling
pushes probabilities away from 0.5 toward the extremes.

The fixed coefficient α = √3 ≈ 1.73 comes from Neyman & Roughgarden (2022)
and is shown to be robust across benchmarks in the AIA Forecaster paper.

Mathematically equivalent forms:
  p_cal = sigmoid(α * logit(p))
  p_cal = p^α / (p^α + (1-p)^α)
"""

from __future__ import annotations

import math

from aia_forecaster.config import settings
from aia_forecaster.models import CalibratedForecast

# Default: √3 ≈ 1.7320508075688772
DEFAULT_ALPHA = math.sqrt(3)


def platt_scale(p: float, alpha: float | None = None) -> float:
    """Apply Platt scaling to a probability.

    Args:
        p: Raw probability in [0, 1].
        alpha: Extremization coefficient. Default √3.

    Returns:
        Calibrated probability pushed away from 0.5.
    """
    if alpha is None:
        alpha = settings.platt_alpha

    # Edge cases
    if p <= 0.0:
        return 0.0
    if p >= 1.0:
        return 1.0

    p_a = p**alpha
    q_a = (1 - p) ** alpha
    return p_a / (p_a + q_a)


def calibrate(p: float, alpha: float | None = None) -> CalibratedForecast:
    """Calibrate a probability and return a full CalibratedForecast."""
    if alpha is None:
        alpha = settings.platt_alpha
    return CalibratedForecast(
        raw_probability=p,
        calibrated_probability=platt_scale(p, alpha),
        alpha=alpha,
    )
