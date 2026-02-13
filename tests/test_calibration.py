"""Tests for Platt scaling calibration."""

import math

from aia_forecaster.calibration.platt import calibrate, platt_scale


class TestPlattScale:
    def test_identity_at_half(self):
        """Platt scaling should leave 0.5 unchanged."""
        assert abs(platt_scale(0.5) - 0.5) < 1e-10

    def test_pushes_above_half_higher(self):
        """Probabilities above 0.5 should increase."""
        for p in [0.55, 0.6, 0.7, 0.8, 0.9]:
            assert platt_scale(p) > p

    def test_pushes_below_half_lower(self):
        """Probabilities below 0.5 should decrease."""
        for p in [0.45, 0.4, 0.3, 0.2, 0.1]:
            assert platt_scale(p) < p

    def test_edge_cases(self):
        """Edge cases: 0 and 1 should be preserved."""
        assert platt_scale(0.0) == 0.0
        assert platt_scale(1.0) == 1.0

    def test_symmetry(self):
        """platt_scale(p) + platt_scale(1-p) should equal 1."""
        for p in [0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9]:
            total = platt_scale(p) + platt_scale(1 - p)
            assert abs(total - 1.0) < 1e-10

    def test_default_alpha(self):
        """Default alpha should be sqrt(3)."""
        alpha = math.sqrt(3)
        p = 0.7
        expected = p**alpha / (p**alpha + (1 - p) ** alpha)
        assert abs(platt_scale(p) - expected) < 1e-10

    def test_custom_alpha(self):
        """Custom alpha should work correctly."""
        p = 0.6
        alpha = 2.0
        result = platt_scale(p, alpha=alpha)
        expected = p**alpha / (p**alpha + (1 - p) ** alpha)
        assert abs(result - expected) < 1e-10

    def test_alpha_one_is_identity(self):
        """With alpha=1, calibration should be identity."""
        for p in [0.1, 0.3, 0.5, 0.7, 0.9]:
            assert abs(platt_scale(p, alpha=1.0) - p) < 1e-10

    def test_known_values(self):
        """Test against hand-computed values."""
        # p=0.6, alpha=sqrt(3): 0.6^1.732 / (0.6^1.732 + 0.4^1.732)
        p = 0.6
        alpha = math.sqrt(3)
        p_a = p**alpha  # ≈ 0.4813
        q_a = (1 - p) ** alpha  # ≈ 0.2771
        expected = p_a / (p_a + q_a)  # ≈ 0.6347
        assert abs(platt_scale(p) - expected) < 1e-4

    def test_calibrate_returns_model(self):
        """calibrate() should return a CalibratedForecast."""
        result = calibrate(0.7)
        assert result.raw_probability == 0.7
        assert result.calibrated_probability > 0.7
        assert abs(result.alpha - math.sqrt(3)) < 1e-10
