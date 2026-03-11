"""Tests for hitting (barrier/touch) probability mode."""

from aia_forecaster.calibration.monotonicity import (
    enforce_hitting_monotonicity,
)
from aia_forecaster.fx.pairs import generate_strikes
from aia_forecaster.models import ForecastMode, Tenor


class TestEnforceHittingMonotonicity:
    def test_no_violations(self):
        """Already correct surface should have zero adjustments."""
        strikes = [151.0, 152.0, 153.0, 154.0, 155.0]
        tenors = [Tenor.W1]
        spot = 153.0
        cell_probs = {
            (151.0, Tenor.W1): 0.70,
            (152.0, Tenor.W1): 0.85,
            (153.0, Tenor.W1): 0.99,
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
            (155.0, Tenor.W1): 0.80,
            (156.0, Tenor.W1): 0.50,
        }
        n = enforce_hitting_monotonicity(cell_probs, strikes, tenors, spot)
        assert n > 0
        assert cell_probs[(154.0, Tenor.W1)] >= cell_probs[(155.0, Tenor.W1)] - 1e-10
        assert cell_probs[(155.0, Tenor.W1)] >= cell_probs[(156.0, Tenor.W1)] - 1e-10

    def test_fixes_below_spot_violation(self):
        """If a farther strike below spot has higher P than a closer one, fix it."""
        strikes = [150.0, 151.0, 152.0, 153.0]
        tenors = [Tenor.W1]
        spot = 153.0
        cell_probs = {
            (150.0, Tenor.W1): 0.60,
            (151.0, Tenor.W1): 0.50,
            (152.0, Tenor.W1): 0.85,
            (153.0, Tenor.W1): 0.99,
        }
        n = enforce_hitting_monotonicity(cell_probs, strikes, tenors, spot)
        assert n > 0
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
            (153.0, Tenor.W1): 0.80,
            (154.0, Tenor.W1): 0.90,
            (155.0, Tenor.W1): 0.60,
        }
        enforce_hitting_monotonicity(cell_probs, strikes, tenors, spot)
        spot_p = cell_probs[(153.0, Tenor.W1)]
        for s in strikes:
            assert cell_probs[(s, Tenor.W1)] <= spot_p + 1e-10

    def test_fixes_tenor_violation(self):
        """P(touch) must be non-decreasing along tenor axis."""
        strikes = [155.0]
        tenors = [Tenor.D1, Tenor.W1, Tenor.M1, Tenor.M3, Tenor.M6]
        spot = 153.0
        cell_probs = {
            (155.0, Tenor.D1): 0.10,
            (155.0, Tenor.W1): 0.30,
            (155.0, Tenor.M1): 0.60,
            (155.0, Tenor.M3): 0.80,
            (155.0, Tenor.M6): 0.70,
        }
        n = enforce_hitting_monotonicity(cell_probs, strikes, tenors, spot)
        assert n > 0
        tenor_probs = [cell_probs[(155.0, t)] for t in tenors]
        for i in range(1, len(tenor_probs)):
            assert tenor_probs[i] >= tenor_probs[i - 1] - 1e-10

    def test_no_tenor_violation_when_increasing(self):
        """Already non-decreasing tenor sequence should have zero tenor adjustments."""
        strikes = [155.0]
        tenors = [Tenor.D1, Tenor.W1, Tenor.M1]
        spot = 153.0
        cell_probs = {
            (155.0, Tenor.D1): 0.10,
            (155.0, Tenor.W1): 0.30,
            (155.0, Tenor.M1): 0.60,
        }
        n = enforce_hitting_monotonicity(cell_probs, strikes, tenors, spot)
        assert n == 0


class TestGenerateStrikesHittingMode:
    def test_odd_num_strikes_in_hitting(self):
        """Hitting mode should force odd num_strikes."""
        strikes = generate_strikes(153.0, "USDJPY", 4, ForecastMode.HITTING)
        assert len(strikes) % 2 == 1

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
