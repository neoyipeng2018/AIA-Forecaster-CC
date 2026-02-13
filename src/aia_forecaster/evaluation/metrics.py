"""Evaluation metrics for forecast quality.

Primary metric: Brier score
  BS = (1/n) * Σ(p_i - o_i)²

Range: 0 (perfect) to 1 (worst)
Baseline: 0.25 (always predicting 0.5)
Strictly proper scoring rule (incentivizes truthful forecasting)
"""

from __future__ import annotations


def brier_score(forecasts: list[float], outcomes: list[int]) -> float:
    """Compute the Brier score for a set of forecasts.

    Args:
        forecasts: List of predicted probabilities in [0, 1].
        outcomes: List of binary outcomes (0 or 1).

    Returns:
        Brier score (lower is better).
    """
    if len(forecasts) != len(outcomes):
        raise ValueError("forecasts and outcomes must have the same length")
    if not forecasts:
        raise ValueError("Cannot compute Brier score on empty lists")

    return sum((p - o) ** 2 for p, o in zip(forecasts, outcomes)) / len(forecasts)


def brier_score_decomposition(
    forecasts: list[float], outcomes: list[int], n_bins: int = 10
) -> dict[str, float]:
    """Decompose Brier score into reliability, resolution, and uncertainty.

    - Reliability: measures calibration (lower is better)
    - Resolution: measures discrimination ability (higher is better)
    - Uncertainty: base rate variance (constant for a given dataset)

    BS = Reliability - Resolution + Uncertainty
    """
    n = len(forecasts)
    if n == 0:
        raise ValueError("Cannot compute decomposition on empty lists")

    base_rate = sum(outcomes) / n
    uncertainty = base_rate * (1 - base_rate)

    # Bin forecasts
    bins: dict[int, list[tuple[float, int]]] = {i: [] for i in range(n_bins)}
    for p, o in zip(forecasts, outcomes):
        bin_idx = min(int(p * n_bins), n_bins - 1)
        bins[bin_idx].append((p, o))

    reliability = 0.0
    resolution = 0.0

    for bin_items in bins.values():
        if not bin_items:
            continue
        n_k = len(bin_items)
        mean_p = sum(p for p, _ in bin_items) / n_k
        mean_o = sum(o for _, o in bin_items) / n_k

        reliability += n_k * (mean_p - mean_o) ** 2
        resolution += n_k * (mean_o - base_rate) ** 2

    reliability /= n
    resolution /= n

    return {
        "brier_score": brier_score(forecasts, outcomes),
        "reliability": reliability,
        "resolution": resolution,
        "uncertainty": uncertainty,
    }
