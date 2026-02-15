"""Tests for historical base rate computation."""

import math

from aia_forecaster.fx.base_rates import (
    ANNUALIZED_VOL,
    compute_base_rates,
    format_base_rate_context,
)
from aia_forecaster.models import Tenor


class TestComputeBaseRates:
    def test_atm_strike_gives_half(self):
        """ATM strike (strike == spot) should give base rate ≈ 0.50."""
        for pair in ANNUALIZED_VOL:
            spot = 153.0 if "JPY" in pair else 1.08
            stats = compute_base_rates(pair, spot=spot, strike=spot, tenor=Tenor.W1)
            assert abs(stats["base_rate_above"] - 0.5) < 1e-10
            assert abs(stats["z_score"]) < 1e-10

    def test_far_otm_strike_near_zero(self):
        """A strike very far above spot should have base rate near 0."""
        stats = compute_base_rates("USDJPY", spot=153.0, strike=180.0, tenor=Tenor.W1)
        assert stats["base_rate_above"] < 0.001

    def test_deep_itm_strike_near_one(self):
        """A strike well below spot should have base rate near 1."""
        stats = compute_base_rates("USDJPY", spot=153.0, strike=130.0, tenor=Tenor.W1)
        assert stats["base_rate_above"] > 0.999

    def test_longer_tenor_wider_range(self):
        """Longer tenors should produce wider 1σ ranges."""
        stats_1d = compute_base_rates("USDJPY", spot=153.0, strike=155.0, tenor=Tenor.D1)
        stats_1w = compute_base_rates("USDJPY", spot=153.0, strike=155.0, tenor=Tenor.W1)
        stats_1m = compute_base_rates("USDJPY", spot=153.0, strike=155.0, tenor=Tenor.M1)

        assert stats_1d["tenor_range_1sigma"] < stats_1w["tenor_range_1sigma"]
        assert stats_1w["tenor_range_1sigma"] < stats_1m["tenor_range_1sigma"]

    def test_longer_tenor_higher_base_rate_for_otm(self):
        """For an OTM strike, longer tenors should give higher base rates."""
        rates = []
        for tenor in [Tenor.D1, Tenor.W1, Tenor.M1, Tenor.M3, Tenor.M6]:
            stats = compute_base_rates("USDJPY", spot=153.0, strike=156.0, tenor=tenor)
            rates.append(stats["base_rate_above"])

        # Each rate should be >= the previous one
        for i in range(1, len(rates)):
            assert rates[i] >= rates[i - 1]

    def test_symmetry(self):
        """P(above spot+x) + P(above spot-x) ≈ 1."""
        spot = 153.0
        offsets = [1.0, 2.0, 5.0]
        for x in offsets:
            above = compute_base_rates("USDJPY", spot=spot, strike=spot + x, tenor=Tenor.W1)
            below = compute_base_rates("USDJPY", spot=spot, strike=spot - x, tenor=Tenor.W1)
            total = above["base_rate_above"] + below["base_rate_above"]
            assert abs(total - 1.0) < 1e-10

    def test_all_supported_pairs(self):
        """All supported pairs should compute without error."""
        pair_spots = {"USDJPY": 153.0, "EURUSD": 1.0800, "GBPUSD": 1.2700}
        for pair, spot in pair_spots.items():
            stats = compute_base_rates(pair, spot=spot, strike=spot * 1.01, tenor=Tenor.M1)
            assert 0.0 <= stats["base_rate_above"] <= 1.0
            assert stats["sigma_t"] > 0

    def test_unsupported_pair_raises(self):
        """Unsupported pair should raise ValueError."""
        try:
            compute_base_rates("USDCHF", spot=0.90, strike=0.91, tenor=Tenor.W1)
            assert False, "Should have raised ValueError"
        except ValueError:
            pass

    def test_z_score_known_value(self):
        """Verify z-score calculation with a known case."""
        # USDJPY: spot=153, strike=155, 1W
        # σ_T = 0.10 * sqrt(5/252) ≈ 0.01409
        # move_pct = 2/153 ≈ 0.01307
        # z = 0.01307 / 0.01409 ≈ 0.928
        stats = compute_base_rates("USDJPY", spot=153.0, strike=155.0, tenor=Tenor.W1)
        expected_sigma_t = 0.10 * math.sqrt(5 / 252)
        assert abs(stats["sigma_t"] - expected_sigma_t) < 1e-10
        expected_z = (2.0 / 153.0) / expected_sigma_t
        assert abs(stats["z_score"] - expected_z) < 1e-10


class TestFormatBaseRateContext:
    def test_produces_nonempty_for_supported_pair(self):
        """Should produce a non-empty string for supported pairs."""
        result = format_base_rate_context("USDJPY", spot=153.0, strike=155.0, tenor=Tenor.W1)
        assert "BASE RATE CONTEXT" in result
        assert "153.00" in result
        assert "155.00" in result

    def test_returns_empty_for_unsupported_pair(self):
        """Should return empty string for unsupported pairs."""
        result = format_base_rate_context("USDCHF", spot=0.90, strike=0.91, tenor=Tenor.W1)
        assert result == ""

    def test_contains_key_information(self):
        """Output should contain all key statistics."""
        result = format_base_rate_context("USDJPY", spot=153.0, strike=155.0, tenor=Tenor.W1)
        assert "spot" in result.lower()
        assert "standard deviation" in result.lower()
        assert "base rate" in result.lower()
        assert "1 week" in result

    def test_eurusd_formatting(self):
        """EURUSD should use 4-decimal formatting."""
        result = format_base_rate_context("EURUSD", spot=1.0800, strike=1.0900, tenor=Tenor.M1)
        assert "1.0800" in result
        assert "1.0900" in result

    def test_case_insensitive_pair(self):
        """Should accept lowercase pair names."""
        result = format_base_rate_context("usdjpy", spot=153.0, strike=155.0, tenor=Tenor.W1)
        assert "BASE RATE CONTEXT" in result
