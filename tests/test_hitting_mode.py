"""Tests for hitting (barrier/touch) probability mode."""

import math
from unittest.mock import patch

from aia_forecaster.calibration.monotonicity import (
    enforce_hitting_monotonicity,
)
from aia_forecaster.fx.base_rates import (
    compute_base_rates,
    compute_hitting_base_rate,
    format_base_rate_context,
)
from aia_forecaster.fx.pairs import generate_strikes
from aia_forecaster.models import ForecastMode, Tenor


class TestComputeHittingBaseRate:
    def test_atm_barrier_gives_one(self):
        """Barrier at spot gives P(hit) = 1.0."""
        stats = compute_hitting_base_rate("USDJPY", spot=153.0, barrier=153.0, tenor=Tenor.W1)
        assert abs(stats["base_rate_hitting"] - 1.0) < 1e-10

    def test_far_barrier_gives_near_zero(self):
        """A barrier very far from spot should have P(hit) near 0."""
        stats = compute_hitting_base_rate("USDJPY", spot=153.0, barrier=180.0, tenor=Tenor.W1)
        assert stats["base_rate_hitting"] < 0.01

    def test_far_below_barrier_gives_near_zero(self):
        """A barrier far below spot should also have P(hit) near 0."""
        stats = compute_hitting_base_rate("USDJPY", spot=153.0, barrier=126.0, tenor=Tenor.W1)
        assert stats["base_rate_hitting"] < 0.01

    def test_hitting_gte_above_for_above_spot(self):
        """P(hit barrier) >= P(above strike) for barriers ABOVE spot.

        This property holds because touching a barrier above spot is easier than
        finishing above it. For barriers below spot, P(above) is ~1 (trivially)
        so the comparison isn't meaningful.
        """
        spot = 153.0
        for strike in [153.0, 154.0, 155.0, 156.0, 157.0]:
            for tenor in [Tenor.D1, Tenor.W1, Tenor.M1, Tenor.M3]:
                hitting = compute_hitting_base_rate("USDJPY", spot, strike, tenor)
                above = compute_base_rates("USDJPY", spot, strike, tenor)
                assert hitting["base_rate_hitting"] >= above["base_rate_above"] - 1e-10, (
                    f"P(hit)={hitting['base_rate_hitting']:.6f} < "
                    f"P(above)={above['base_rate_above']:.6f} at strike={strike}, tenor={tenor.value}"
                )

    def test_longer_tenor_higher_hitting(self):
        """Longer tenors should give higher P(hit) for out-of-spot barriers."""
        rates = []
        for tenor in [Tenor.D1, Tenor.W1, Tenor.M1, Tenor.M3, Tenor.M6]:
            stats = compute_hitting_base_rate("USDJPY", spot=153.0, barrier=156.0, tenor=tenor)
            rates.append(stats["base_rate_hitting"])

        # Each rate should be >= the previous one
        for i in range(1, len(rates)):
            assert rates[i] >= rates[i - 1] - 1e-10, (
                f"Tenor ordering violated: {rates[i-1]:.6f} > {rates[i]:.6f}"
            )

    def test_approximately_symmetric_distance(self):
        """Barriers equidistant above and below spot should have approximately same P(hit).

        Not perfectly symmetric because the formula uses |ln(B/S)| which differs
        slightly for +offset vs -offset in absolute price terms.
        Tolerance is proportional to offset/spot.
        """
        spot = 153.0
        for offset in [1.0, 2.0, 5.0]:
            above = compute_hitting_base_rate("USDJPY", spot, spot + offset, Tenor.M1)
            below = compute_hitting_base_rate("USDJPY", spot, spot - offset, Tenor.M1)
            tolerance = 0.01 * (offset / spot) * 100  # proportional to relative offset
            assert abs(above["base_rate_hitting"] - below["base_rate_hitting"]) < tolerance, (
                f"Asymmetry at offset={offset}: "
                f"above={above['base_rate_hitting']:.6f}, below={below['base_rate_hitting']:.6f}"
            )

    def test_decreases_with_distance(self):
        """P(hit) should decrease as barrier moves away from spot in either direction."""
        spot = 153.0
        # Above spot
        prev_p = 1.0
        for strike in [154.0, 155.0, 156.0, 157.0, 158.0]:
            stats = compute_hitting_base_rate("USDJPY", spot, strike, Tenor.M1)
            assert stats["base_rate_hitting"] <= prev_p + 1e-10
            prev_p = stats["base_rate_hitting"]

        # Below spot
        prev_p = 1.0
        for strike in [152.0, 151.0, 150.0, 149.0, 148.0]:
            stats = compute_hitting_base_rate("USDJPY", spot, strike, Tenor.M1)
            assert stats["base_rate_hitting"] <= prev_p + 1e-10
            prev_p = stats["base_rate_hitting"]


class TestFormatBaseRateContextHitting:
    def test_hitting_mode_output(self):
        """Hitting mode should produce barrier-specific context."""
        result = format_base_rate_context(
            "USDJPY", spot=153.0, strike=155.0, tenor=Tenor.W1,
            forecast_mode=ForecastMode.HITTING,
        )
        assert "HITTING" in result or "BARRIER" in result
        assert "P(touch" in result
        assert "reflection-principle" in result.lower()

    def test_above_mode_output(self):
        """Above mode should produce the original context format."""
        result = format_base_rate_context(
            "USDJPY", spot=153.0, strike=155.0, tenor=Tenor.W1,
            forecast_mode=ForecastMode.ABOVE,
        )
        assert "Statistical base rate" in result
        assert "P(above" in result
        assert "normal-distribution" in result.lower()

    def test_default_is_above(self):
        """Default mode should be ABOVE for backward compatibility."""
        result = format_base_rate_context(
            "USDJPY", spot=153.0, strike=155.0, tenor=Tenor.W1,
        )
        assert "P(above" in result


class TestEnforceHittingMonotonicity:
    def test_no_violations(self):
        """Already correct surface should have zero adjustments."""
        strikes = [151.0, 152.0, 153.0, 154.0, 155.0]
        tenors = [Tenor.W1]
        spot = 153.0
        cell_probs = {
            (151.0, Tenor.W1): 0.70,
            (152.0, Tenor.W1): 0.85,
            (153.0, Tenor.W1): 0.99,  # highest at spot
            (154.0, Tenor.W1): 0.85,
            (155.0, Tenor.W1): 0.70,
        }
        n = enforce_hitting_monotonicity(cell_probs, strikes, tenors, spot)
        assert n == 0

    def test_fixes_above_spot_violation(self):
        """If a farther strike has higher P than a closer one (above spot), fix it."""
        strikes = [153.0, 154.0, 155.0, 156.0]
        tenors = [Tenor.W1]
        spot = 153.0
        cell_probs = {
            (153.0, Tenor.W1): 0.99,
            (154.0, Tenor.W1): 0.70,
            (155.0, Tenor.W1): 0.80,  # violation: 0.80 > 0.70
            (156.0, Tenor.W1): 0.50,
        }
        n = enforce_hitting_monotonicity(cell_probs, strikes, tenors, spot)
        assert n > 0
        # After fix, above-spot should be non-increasing with distance
        assert cell_probs[(154.0, Tenor.W1)] >= cell_probs[(155.0, Tenor.W1)] - 1e-10
        assert cell_probs[(155.0, Tenor.W1)] >= cell_probs[(156.0, Tenor.W1)] - 1e-10

    def test_fixes_below_spot_violation(self):
        """If a farther strike below spot has higher P than a closer one, fix it."""
        strikes = [150.0, 151.0, 152.0, 153.0]
        tenors = [Tenor.W1]
        spot = 153.0
        cell_probs = {
            (150.0, Tenor.W1): 0.60,  # violation: further from spot but higher
            (151.0, Tenor.W1): 0.50,
            (152.0, Tenor.W1): 0.85,
            (153.0, Tenor.W1): 0.99,
        }
        n = enforce_hitting_monotonicity(cell_probs, strikes, tenors, spot)
        assert n > 0
        # After fix: 152 >= 151 >= 150 (closest to farthest from spot)
        assert cell_probs[(152.0, Tenor.W1)] >= cell_probs[(151.0, Tenor.W1)] - 1e-10
        assert cell_probs[(151.0, Tenor.W1)] >= cell_probs[(150.0, Tenor.W1)] - 1e-10

    def test_spot_is_highest(self):
        """Spot strike probability should be >= all others after enforcement."""
        strikes = [151.0, 152.0, 153.0, 154.0, 155.0]
        tenors = [Tenor.W1]
        spot = 153.0
        cell_probs = {
            (151.0, Tenor.W1): 0.70,
            (152.0, Tenor.W1): 0.85,
            (153.0, Tenor.W1): 0.80,  # spot is NOT highest — will be kept, but neighbors clamped
            (154.0, Tenor.W1): 0.90,  # violation: higher than spot
            (155.0, Tenor.W1): 0.60,
        }
        enforce_hitting_monotonicity(cell_probs, strikes, tenors, spot)
        spot_p = cell_probs[(153.0, Tenor.W1)]
        for s in strikes:
            assert cell_probs[(s, Tenor.W1)] <= spot_p + 1e-10


class TestGenerateStrikesHittingMode:
    def test_odd_num_strikes_in_hitting(self):
        """Hitting mode should force odd num_strikes."""
        strikes = generate_strikes(153.0, "USDJPY", 4, ForecastMode.HITTING)
        assert len(strikes) % 2 == 1  # forced odd

    def test_even_stays_even_in_above(self):
        """Above mode should not change even num_strikes."""
        strikes = generate_strikes(153.0, "USDJPY", 4, ForecastMode.ABOVE)
        assert len(strikes) == 4

    def test_odd_stays_odd_in_hitting(self):
        """Already odd num_strikes should stay odd in hitting."""
        strikes = generate_strikes(153.0, "USDJPY", 5, ForecastMode.HITTING)
        assert len(strikes) == 5

    def test_spot_is_center_in_hitting(self):
        """Spot should be approximately centered in hitting mode strikes."""
        spot = 153.0
        strikes = generate_strikes(spot, "USDJPY", 5, ForecastMode.HITTING)
        mid_idx = len(strikes) // 2
        assert abs(strikes[mid_idx] - spot) < 1.0


class TestForecastModeEnum:
    def test_values(self):
        assert ForecastMode.ABOVE.value == "above"
        assert ForecastMode.HITTING.value == "hitting"

    def test_construction_from_string(self):
        assert ForecastMode("above") == ForecastMode.ABOVE
        assert ForecastMode("hitting") == ForecastMode.HITTING
