"""Tests for monotonicity enforcement (PAVA)."""

from aia_forecaster.calibration.monotonicity import (
    enforce_decreasing,
    enforce_surface_monotonicity,
)
from aia_forecaster.models import (
    CalibratedForecast,
    ProbabilitySurface,
    SurfaceCell,
    Tenor,
)


class TestEnforceDecreasing:
    def test_already_monotone(self):
        """A non-increasing sequence should be unchanged."""
        probs = [0.9, 0.7, 0.5, 0.3, 0.1]
        result = enforce_decreasing(probs)
        assert result == probs

    def test_single_element(self):
        assert enforce_decreasing([0.6]) == [0.6]

    def test_empty(self):
        assert enforce_decreasing([]) == []

    def test_simple_violation(self):
        """Two adjacent elements out of order should be averaged."""
        # 0.3 < 0.7 is a violation → averaged to 0.5
        result = enforce_decreasing([0.3, 0.7])
        assert len(result) == 2
        assert abs(result[0] - 0.5) < 1e-10
        assert abs(result[1] - 0.5) < 1e-10

    def test_single_violation_in_middle(self):
        """One violation in an otherwise monotone sequence."""
        # [0.9, 0.4, 0.6, 0.2] → 0.4 < 0.6 violates
        # PAVA merges 0.4 and 0.6 → both become 0.5
        result = enforce_decreasing([0.9, 0.4, 0.6, 0.2])
        assert len(result) == 4
        assert result[0] == 0.9
        assert abs(result[1] - 0.5) < 1e-10
        assert abs(result[2] - 0.5) < 1e-10
        assert result[3] == 0.2

    def test_cascading_violation(self):
        """Violation that cascades backward after merging."""
        # [0.5, 0.3, 0.8] → merge 0.3,0.8 → 0.55 > 0.5 → merge all three
        result = enforce_decreasing([0.5, 0.3, 0.8])
        mean = (0.5 + 0.3 + 0.8) / 3
        for r in result:
            assert abs(r - mean) < 1e-10

    def test_fully_reversed(self):
        """Fully increasing sequence should become constant (mean)."""
        probs = [0.1, 0.3, 0.5, 0.7, 0.9]
        result = enforce_decreasing(probs)
        mean = sum(probs) / len(probs)
        for r in result:
            assert abs(r - mean) < 1e-10

    def test_constant_sequence(self):
        """A constant sequence is trivially non-increasing."""
        probs = [0.5, 0.5, 0.5]
        result = enforce_decreasing(probs)
        assert result == probs

    def test_result_is_non_increasing(self):
        """Property test: output is always non-increasing."""
        probs = [0.45, 0.80, 0.30, 0.65, 0.20]
        result = enforce_decreasing(probs)
        for i in range(len(result) - 1):
            assert result[i] >= result[i + 1] - 1e-10

    def test_preserves_mean(self):
        """PAVA preserves the overall mean of the sequence."""
        probs = [0.45, 0.80, 0.30, 0.65, 0.20]
        result = enforce_decreasing(probs)
        assert abs(sum(result) / len(result) - sum(probs) / len(probs)) < 1e-10

    def test_clamped_to_unit_interval(self):
        """Result should be clamped to [0, 1]."""
        probs = [0.9, 0.8, 0.7]
        result = enforce_decreasing(probs)
        for r in result:
            assert 0.0 <= r <= 1.0

    def test_minimizes_l2_distance(self):
        """PAVA result should be closer to input than any other non-increasing sequence."""
        import random

        random.seed(42)
        probs = [random.random() for _ in range(6)]
        result = enforce_decreasing(probs)

        l2_pava = sum((r - p) ** 2 for r, p in zip(result, probs))

        # Compare against a naive non-increasing sequence (sorted descending)
        naive = sorted(probs, reverse=True)
        l2_naive = sum((n - p) ** 2 for n, p in zip(naive, probs))

        assert l2_pava <= l2_naive + 1e-10


def _make_cell(strike: float, tenor: Tenor, prob: float) -> SurfaceCell:
    return SurfaceCell(
        strike=strike,
        tenor=tenor,
        question=f"Will X be above {strike} in {tenor.value}?",
        calibrated=CalibratedForecast(
            raw_probability=prob,
            calibrated_probability=prob,
            alpha=1.73,
        ),
    )


class TestEnforceSurfaceMonotonicity:
    def test_fixes_single_tenor(self):
        """Violations within a tenor are corrected."""
        surface = ProbabilitySurface(pair="USDJPY", spot_rate=155.0, cells=[
            _make_cell(153.0, Tenor.W1, 0.80),
            _make_cell(154.0, Tenor.W1, 0.90),  # violation: 0.90 > 0.80
            _make_cell(155.0, Tenor.W1, 0.50),
            _make_cell(156.0, Tenor.W1, 0.30),
            _make_cell(157.0, Tenor.W1, 0.10),
        ])
        n = enforce_surface_monotonicity(surface)
        assert n > 0

        # Verify monotonicity
        w1_cells = sorted(
            [c for c in surface.cells if c.tenor == Tenor.W1],
            key=lambda c: c.strike,
        )
        probs = [c.calibrated.calibrated_probability for c in w1_cells]
        for i in range(len(probs) - 1):
            assert probs[i] >= probs[i + 1] - 1e-10

    def test_independent_across_tenors(self):
        """Each tenor is fixed independently."""
        surface = ProbabilitySurface(pair="USDJPY", spot_rate=155.0, cells=[
            # W1: has violation
            _make_cell(153.0, Tenor.W1, 0.40),
            _make_cell(154.0, Tenor.W1, 0.70),  # violation
            # M1: already monotone
            _make_cell(153.0, Tenor.M1, 0.80),
            _make_cell(154.0, Tenor.M1, 0.60),
        ])
        enforce_surface_monotonicity(surface)

        # M1 should be unchanged
        m1_cells = [c for c in surface.cells if c.tenor == Tenor.M1]
        m1_cells.sort(key=lambda c: c.strike)
        assert m1_cells[0].calibrated.calibrated_probability == 0.80
        assert m1_cells[1].calibrated.calibrated_probability == 0.60

    def test_no_violations_returns_zero(self):
        """When there are no violations, nothing is adjusted."""
        surface = ProbabilitySurface(pair="USDJPY", spot_rate=155.0, cells=[
            _make_cell(153.0, Tenor.W1, 0.80),
            _make_cell(154.0, Tenor.W1, 0.60),
            _make_cell(155.0, Tenor.W1, 0.40),
        ])
        n = enforce_surface_monotonicity(surface)
        assert n == 0

    def test_skips_cells_without_calibration(self):
        """Cells with calibrated=None are skipped."""
        surface = ProbabilitySurface(pair="USDJPY", spot_rate=155.0, cells=[
            _make_cell(153.0, Tenor.W1, 0.80),
            SurfaceCell(strike=154.0, tenor=Tenor.W1, question="test"),  # no calibration
            _make_cell(155.0, Tenor.W1, 0.40),
        ])
        n = enforce_surface_monotonicity(surface)
        assert n == 0
        # The None cell should still be None
        none_cell = [c for c in surface.cells if c.calibrated is None]
        assert len(none_cell) == 1

    def test_preserves_raw_probability(self):
        """raw_probability should not change when calibrated_probability is adjusted."""
        surface = ProbabilitySurface(pair="USDJPY", spot_rate=155.0, cells=[
            _make_cell(153.0, Tenor.W1, 0.40),
            _make_cell(154.0, Tenor.W1, 0.70),  # violation
        ])
        enforce_surface_monotonicity(surface)

        for cell in surface.cells:
            if cell.strike == 153.0:
                assert cell.calibrated.raw_probability == 0.40
            elif cell.strike == 154.0:
                assert cell.calibrated.raw_probability == 0.70
