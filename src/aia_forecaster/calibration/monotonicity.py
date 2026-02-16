"""Monotonicity enforcement for cumulative tail probabilities.

For P(price > strike), probabilities must be non-increasing as strike
increases: a higher threshold should have equal or lower probability of
being exceeded.

Uses the Pool Adjacent Violators Algorithm (PAVA) to find the closest
non-increasing sequence in L2 norm — the standard isotonic regression
approach.
"""

from __future__ import annotations

import logging

from aia_forecaster.models import CalibratedForecast, ProbabilitySurface, SurfaceCell, Tenor

logger = logging.getLogger(__name__)


def enforce_decreasing(probabilities: list[float]) -> list[float]:
    """Enforce non-increasing monotonicity on a sequence of probabilities.

    Given probabilities ordered by ascending strike, returns the closest
    non-increasing sequence (in L2 norm) using PAVA.

    Args:
        probabilities: Probabilities sorted by ascending strike.

    Returns:
        Adjusted probabilities satisfying p[0] >= p[1] >= ... >= p[n-1],
        clamped to [0, 1].
    """
    n = len(probabilities)
    if n <= 1:
        return list(probabilities)

    # PAVA for non-increasing: merge adjacent blocks that violate the constraint.
    # Each block tracks (sum_of_values, count).
    blocks: list[tuple[float, int]] = [(p, 1) for p in probabilities]

    i = 0
    while i < len(blocks) - 1:
        mean_curr = blocks[i][0] / blocks[i][1]
        mean_next = blocks[i + 1][0] / blocks[i + 1][1]
        if mean_curr < mean_next:
            # Violation: current mean < next mean, but we need non-increasing
            merged_sum = blocks[i][0] + blocks[i + 1][0]
            merged_count = blocks[i][1] + blocks[i + 1][1]
            blocks[i] = (merged_sum, merged_count)
            blocks.pop(i + 1)
            # Step back to recheck the previous boundary
            if i > 0:
                i -= 1
        else:
            i += 1

    # Expand blocks to individual values, clamped to [0, 1]
    result: list[float] = []
    for total, count in blocks:
        val = max(0.0, min(1.0, total / count))
        result.extend([val] * count)

    return result


def enforce_raw_surface_monotonicity(
    cell_probabilities: dict[tuple[float, "Tenor"], float],
    strikes: list[float],
    tenors: list["Tenor"],
) -> int:
    """Apply PAVA to raw (pre-calibration) probabilities per tenor.

    Modifies *cell_probabilities* in-place so that for each tenor the
    probabilities are non-increasing as strike increases.

    Returns:
        Number of cells whose probabilities were adjusted.
    """
    sorted_strikes = sorted(strikes)
    total_adjusted = 0

    for tenor in tenors:
        probs = [cell_probabilities.get((s, tenor), 0.5) for s in sorted_strikes]
        adjusted = enforce_decreasing(probs)

        for strike, orig, adj in zip(sorted_strikes, probs, adjusted):
            if abs(orig - adj) > 1e-10:
                total_adjusted += 1
                logger.info(
                    "Raw monotonicity fix [%s, strike=%.2f]: %.4f -> %.4f",
                    tenor.value, strike, orig, adj,
                )
                cell_probabilities[(strike, tenor)] = adj

    if total_adjusted:
        logger.info("Raw monotonicity: adjusted %d cells", total_adjusted)
    else:
        logger.info("Raw monotonicity: no violations detected")

    return total_adjusted


def enforce_surface_monotonicity(surface: ProbabilitySurface) -> int:
    """Enforce monotonicity across strikes for each tenor in a probability surface.

    Modifies cells in-place: for each tenor, sorts cells by ascending strike
    and applies PAVA so that calibrated probabilities are non-increasing.

    Args:
        surface: The probability surface to fix.

    Returns:
        Number of cells whose probabilities were adjusted.
    """
    # Group cells by tenor
    by_tenor: dict[Tenor, list[SurfaceCell]] = {}
    for cell in surface.cells:
        by_tenor.setdefault(cell.tenor, []).append(cell)

    total_adjusted = 0

    for tenor, cells in by_tenor.items():
        # Only process cells that have calibrated forecasts
        valid = [c for c in cells if c.calibrated is not None]
        if len(valid) <= 1:
            continue

        # Sort by ascending strike
        valid.sort(key=lambda c: c.strike)

        original = [c.calibrated.calibrated_probability for c in valid]
        adjusted = enforce_decreasing(original)

        for cell, orig, adj in zip(valid, original, adjusted):
            if abs(orig - adj) > 1e-10:
                total_adjusted += 1
                logger.info(
                    "Monotonicity fix [%s, strike=%.2f]: %.4f -> %.4f",
                    tenor.value,
                    cell.strike,
                    orig,
                    adj,
                )
                cell.calibrated = CalibratedForecast(
                    raw_probability=cell.calibrated.raw_probability,
                    calibrated_probability=adj,
                    alpha=cell.calibrated.alpha,
                )

    if total_adjusted:
        logger.info("Monotonicity: adjusted %d cells across %d tenors", total_adjusted, len(by_tenor))
    else:
        logger.info("Monotonicity: no violations detected")

    return total_adjusted
