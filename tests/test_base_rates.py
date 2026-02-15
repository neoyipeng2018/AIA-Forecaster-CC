"""Tests for historical base rate computation."""

import math
from unittest.mock import patch

from aia_forecaster.fx.base_rates import (
    FALLBACK_VOL,
    _vol_cache,
    compute_base_rates,
    format_base_rate_context,
    get_annualized_vol,
)
from aia_forecaster.models import Tenor


class TestComputeBaseRates:
    def test_atm_strike_gives_half(self):
        """ATM strike (strike == spot) should give base rate ~ 0.50."""
        for pair in FALLBACK_VOL:
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
        """Longer tenors should produce wider 1-sigma ranges."""
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
        """P(above spot+x) + P(above spot-x) ~ 1."""
        spot = 153.0
        offsets = [1.0, 2.0, 5.0]
        for x in offsets:
            above = compute_base_rates("USDJPY", spot=spot, strike=spot + x, tenor=Tenor.W1)
            below = compute_base_rates("USDJPY", spot=spot, strike=spot - x, tenor=Tenor.W1)
            total = above["base_rate_above"] + below["base_rate_above"]
            assert abs(total - 1.0) < 1e-10

    def test_all_fallback_pairs(self):
        """All pairs with fallback data should compute without error."""
        pair_spots = {
            "USDJPY": 153.0,
            "EURUSD": 1.0800,
            "GBPUSD": 1.2700,
            "AUDUSD": 0.6500,
            "NZDUSD": 0.5900,
            "USDCAD": 1.3500,
            "USDCHF": 0.8900,
        }
        for pair, spot in pair_spots.items():
            if pair not in FALLBACK_VOL:
                continue
            stats = compute_base_rates(pair, spot=spot, strike=spot * 1.01, tenor=Tenor.M1)
            assert 0.0 <= stats["base_rate_above"] <= 1.0
            assert stats["sigma_t"] > 0

    def test_unsupported_pair_no_fallback_raises(self):
        """Pair with no fallback and no dynamic data should raise ValueError."""
        # Patch _compute_realized_vol to return None
        with patch("aia_forecaster.fx.base_rates._compute_realized_vol", return_value=None):
            _vol_cache.pop("XYZABC", None)
            try:
                compute_base_rates("XYZABC", spot=1.0, strike=1.01, tenor=Tenor.W1)
                assert False, "Should have raised ValueError"
            except ValueError:
                pass

    def test_z_score_known_value(self):
        """Verify z-score calculation with a known case."""
        # Force fallback vol so we get deterministic results
        with patch("aia_forecaster.fx.base_rates.get_annualized_vol", return_value=0.10):
            stats = compute_base_rates("USDJPY", spot=153.0, strike=155.0, tenor=Tenor.W1)
            expected_sigma_t = 0.10 * math.sqrt(5 / 252)
            assert abs(stats["sigma_t"] - expected_sigma_t) < 1e-10
            expected_z = (2.0 / 153.0) / expected_sigma_t
            assert abs(stats["z_score"] - expected_z) < 1e-10

    def test_result_includes_vol_metadata(self):
        """Result should include annualized_vol and vol_source."""
        stats = compute_base_rates("USDJPY", spot=153.0, strike=155.0, tenor=Tenor.W1)
        assert "annualized_vol" in stats
        assert "vol_source" in stats
        assert stats["annualized_vol"] > 0
        assert stats["vol_source"] in ("dynamic", "fallback")


class TestGetAnnualizedVol:
    def test_returns_fallback_when_dynamic_fails(self):
        """Should return fallback vol when dynamic computation fails."""
        with patch("aia_forecaster.fx.base_rates._compute_realized_vol", return_value=None):
            _vol_cache.pop("USDJPY", None)
            vol = get_annualized_vol("USDJPY")
            assert vol == FALLBACK_VOL["USDJPY"]

    def test_returns_dynamic_when_available(self):
        """Should prefer dynamic vol over fallback."""
        with patch("aia_forecaster.fx.base_rates._compute_realized_vol", return_value=0.12):
            _vol_cache.pop("USDJPY", None)
            vol = get_annualized_vol("USDJPY")
            assert vol == 0.12

    def test_caches_dynamic_vol(self):
        """Dynamic vol should be cached after first fetch."""
        with patch("aia_forecaster.fx.base_rates._compute_realized_vol", return_value=0.12) as mock:
            _vol_cache.pop("USDJPY", None)
            get_annualized_vol("USDJPY")
            get_annualized_vol("USDJPY")  # second call should use cache
            assert mock.call_count == 1

    def test_case_insensitive(self):
        """Should handle lowercase pair names."""
        with patch("aia_forecaster.fx.base_rates._compute_realized_vol", return_value=None):
            _vol_cache.pop("USDJPY", None)
            vol = get_annualized_vol("usdjpy")
            assert vol == FALLBACK_VOL["USDJPY"]


class TestFormatBaseRateContext:
    def test_produces_nonempty_for_supported_pair(self):
        """Should produce a non-empty string for supported pairs."""
        result = format_base_rate_context("USDJPY", spot=153.0, strike=155.0, tenor=Tenor.W1)
        assert "BASE RATE CONTEXT" in result
        assert "153.00" in result
        assert "155.00" in result

    def test_returns_empty_for_unsupported_pair(self):
        """Should return empty string for fully unsupported pairs."""
        with patch("aia_forecaster.fx.base_rates._compute_realized_vol", return_value=None):
            _vol_cache.pop("XYZABC", None)
            result = format_base_rate_context("XYZABC", spot=0.90, strike=0.91, tenor=Tenor.W1)
            assert result == ""

    def test_contains_key_information(self):
        """Output should contain all key statistics."""
        result = format_base_rate_context("USDJPY", spot=153.0, strike=155.0, tenor=Tenor.W1)
        assert "spot" in result.lower()
        assert "standard deviation" in result.lower()
        assert "base rate" in result.lower()
        assert "1 week" in result

    def test_contains_vol_source(self):
        """Output should show the volatility source (dynamic vs fallback)."""
        result = format_base_rate_context("USDJPY", spot=153.0, strike=155.0, tenor=Tenor.W1)
        assert "Annualized vol:" in result
        assert ("dynamic" in result or "fallback" in result)

    def test_eurusd_formatting(self):
        """EURUSD should use 4-decimal formatting."""
        result = format_base_rate_context("EURUSD", spot=1.0800, strike=1.0900, tenor=Tenor.M1)
        assert "1.0800" in result
        assert "1.0900" in result

    def test_case_insensitive_pair(self):
        """Should accept lowercase pair names."""
        result = format_base_rate_context("usdjpy", spot=153.0, strike=155.0, tenor=Tenor.W1)
        assert "BASE RATE CONTEXT" in result
