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


def enforce_increasing(probabilities: list[float]) -> list[float]:
    """Enforce non-decreasing monotonicity on a sequence of probabilities.

    Returns the closest non-decreasing sequence (in L2 norm) using PAVA.

    Args:
        probabilities: Probabilities in order.

    Returns:
        Adjusted probabilities satisfying p[0] <= p[1] <= ... <= p[n-1],
        clamped to [0, 1].
    """
    # Reverse, apply non-increasing PAVA, reverse back
    reversed_result = enforce_decreasing(list(reversed(probabilities)))
    return list(reversed(reversed_result))


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


def enforce_hitting_monotonicity(
    cell_probabilities: dict[tuple[float, "Tenor"], float],
    strikes: list[float],
    tenors: list["Tenor"],
    spot: float,
) -> int:
    """Enforce hitting-mode monotonicity on both strike and tenor axes.

    Strike axis (per tenor):
    - Above spot: non-increasing as strike increases (moves away from spot)
    - Below spot: non-increasing as strike decreases (moves away from spot)

    Tenor axis (per strike):
    - Non-decreasing as tenor increases: more time = higher P(touch)

    Modifies *cell_probabilities* in-place.

    Returns:
        Number of cells whose probabilities were adjusted.
    """
    sorted_strikes = sorted(strikes)
    sorted_tenors = sorted(tenors, key=lambda t: t.days)
    total_adjusted = 0

    # --- Strike-axis monotonicity (per tenor) ---
    for tenor in sorted_tenors:
        # Split strikes into below-spot, at-spot, and above-spot
        below = [s for s in sorted_strikes if s < spot]
        above = [s for s in sorted_strikes if s > spot]
        at_spot = [s for s in sorted_strikes if s == spot]

        # Above spot: enforce non-increasing as strike increases (away from spot)
        if above:
            above_strikes = above  # already ascending = away from spot
            probs = [cell_probabilities.get((s, tenor), 0.5) for s in above_strikes]
            adjusted = enforce_decreasing(probs)
            for strike, orig, adj in zip(above_strikes, probs, adjusted):
                if abs(orig - adj) > 1e-10:
                    total_adjusted += 1
                    logger.info(
                        "Hitting strike fix [%s, strike=%.2f above spot]: %.4f -> %.4f",
                        tenor.value, strike, orig, adj,
                    )
                    cell_probabilities[(strike, tenor)] = adj

        # Below spot: enforce non-increasing as strike decreases (away from spot)
        # Reverse so we go from closest-to-spot to farthest-from-spot
        if below:
            below_away = list(reversed(below))  # closest to spot first
            probs = [cell_probabilities.get((s, tenor), 0.5) for s in below_away]
            adjusted = enforce_decreasing(probs)
            for strike, orig, adj in zip(below_away, probs, adjusted):
                if abs(orig - adj) > 1e-10:
                    total_adjusted += 1
                    logger.info(
                        "Hitting strike fix [%s, strike=%.2f below spot]: %.4f -> %.4f",
                        tenor.value, strike, orig, adj,
                    )
                    cell_probabilities[(strike, tenor)] = adj

        # At-spot strikes should be highest; enforce they are >= neighbors
        if at_spot:
            spot_strike = at_spot[0]
            spot_p = cell_probabilities.get((spot_strike, tenor), 0.5)
            for s in sorted_strikes:
                if s == spot_strike:
                    continue
                neighbor_p = cell_probabilities.get((s, tenor), 0.5)
                if neighbor_p > spot_p:
                    total_adjusted += 1
                    logger.info(
                        "Hitting spot fix [%s, strike=%.2f > spot %.2f]: %.4f -> %.4f",
                        tenor.value, s, spot_strike, neighbor_p, spot_p,
                    )
                    cell_probabilities[(s, tenor)] = spot_p

    # --- Tenor-axis monotonicity (per strike) ---
    # P(touch) must be non-decreasing as tenor increases
    for strike in sorted_strikes:
        probs = [cell_probabilities.get((strike, t), 0.5) for t in sorted_tenors]
        adjusted = enforce_increasing(probs)
        for tenor, orig, adj in zip(sorted_tenors, probs, adjusted):
            if abs(orig - adj) > 1e-10:
                total_adjusted += 1
                logger.info(
                    "Hitting tenor fix [%s, strike=%.2f]: %.4f -> %.4f",
                    tenor.value, strike, orig, adj,
                )
                cell_probabilities[(strike, tenor)] = adj

    if total_adjusted:
        logger.info("Hitting monotonicity: adjusted %d cells", total_adjusted)
    else:
        logger.info("Hitting monotonicity: no violations detected")

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
