"""Tests for base rate context formatting."""

from aia_forecaster.fx.base_rates import (
    format_base_rate_context,
    get_consensus,
    set_consensus_provider,
)
from aia_forecaster.models import ForecastMode, Tenor


class TestFormatBaseRateContext:
    def test_produces_nonempty(self):
        result = format_base_rate_context("USDJPY", spot=153.0, strike=155.0, tenor=Tenor.W1)
        assert "MARKET CONTEXT" in result
        assert "153.00" in result
        assert "155.00" in result

    def test_works_for_any_pair(self):
        result = format_base_rate_context("XYZABC", spot=0.90, strike=0.91, tenor=Tenor.W1)
        assert "MARKET CONTEXT" in result
        assert "0.9000" in result

    def test_shows_distance(self):
        result = format_base_rate_context("USDJPY", spot=153.0, strike=155.0, tenor=Tenor.W1)
        assert "+2.00" in result
        assert "+1.31%" in result

    def test_shows_tenor(self):
        result = format_base_rate_context("USDJPY", spot=153.0, strike=155.0, tenor=Tenor.M3)
        assert "3 months" in result

    def test_no_probability_in_output(self):
        result = format_base_rate_context("USDJPY", spot=153.0, strike=155.0, tenor=Tenor.W1)
        assert "Statistical base rate" not in result
        assert "P(above" not in result
        assert "Annualized vol" not in result
        assert "sigma" not in result.lower()

    def test_eurusd_4_decimal_formatting(self):
        result = format_base_rate_context("EURUSD", spot=1.0800, strike=1.0900, tenor=Tenor.M1)
        assert "1.0800" in result
        assert "1.0900" in result

    def test_hitting_mode(self):
        result = format_base_rate_context(
            "USDJPY", spot=153.0, strike=155.0, tenor=Tenor.W1,
            forecast_mode=ForecastMode.HITTING,
        )
        assert "HITTING" in result or "BARRIER" in result
        assert "Barrier" in result
        assert "P(touch)" in result

    def test_above_mode(self):
        result = format_base_rate_context(
            "USDJPY", spot=153.0, strike=155.0, tenor=Tenor.W1,
            forecast_mode=ForecastMode.ABOVE,
        )
        assert "Target" in result
        assert "above" in result

    def test_case_insensitive_pair(self):
        result = format_base_rate_context("usdjpy", spot=153.0, strike=155.0, tenor=Tenor.W1)
        assert "MARKET CONTEXT" in result

    def test_negative_distance(self):
        result = format_base_rate_context("USDJPY", spot=155.0, strike=153.0, tenor=Tenor.W1)
        assert "-2.00" in result

    def test_hitting_below_spot(self):
        result = format_base_rate_context(
            "USDJPY", spot=155.0, strike=153.0, tenor=Tenor.W1,
            forecast_mode=ForecastMode.HITTING,
        )
        assert "below" in result


class TestConsensusProvider:
    def test_no_provider_returns_none(self):
        set_consensus_provider(None)
        assert get_consensus("USDJPY", 153.0, Tenor.W1) is None

    def test_provider_called(self):
        def mock_provider(pair: str, spot: float, tenor: Tenor) -> tuple[float, str]:
            return (150.0, "test_model")

        set_consensus_provider(mock_provider)
        result = get_consensus("USDJPY", 153.0, Tenor.W1)
        assert result == (150.0, "test_model")
        set_consensus_provider(None)

    def test_provider_exception_returns_none(self):
        def bad_provider(pair: str, spot: float, tenor: Tenor) -> tuple[float, str]:
            raise RuntimeError("boom")

        set_consensus_provider(bad_provider)
        result = get_consensus("USDJPY", 153.0, Tenor.W1)
        assert result is None
        set_consensus_provider(None)

    def test_consensus_appears_in_context(self):
        def mock_provider(pair: str, spot: float, tenor: Tenor) -> tuple[float, str]:
            return (150.0, "analyst_survey")

        set_consensus_provider(mock_provider)
        result = format_base_rate_context("USDJPY", spot=153.0, strike=155.0, tenor=Tenor.W1)
        assert "150.00" in result
        assert "analyst_survey" in result
        set_consensus_provider(None)

    def test_consensus_with_hitting_mode(self):
        def mock_provider(pair: str, spot: float, tenor: Tenor) -> tuple[float, str]:
            return (155.0, "internal_model")

        set_consensus_provider(mock_provider)
        result = format_base_rate_context(
            "USDJPY", spot=153.0, strike=156.0, tenor=Tenor.W1,
            forecast_mode=ForecastMode.HITTING,
        )
        assert "internal_model" in result
        assert "toward" in result
        set_consensus_provider(None)
