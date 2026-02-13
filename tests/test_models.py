"""Tests for Pydantic data models."""

from datetime import date

import pytest
from pydantic import ValidationError

from aia_forecaster.models import (
    AgentForecast,
    CalibratedForecast,
    Confidence,
    EnsembleResult,
    ForecastQuestion,
    SearchResult,
    SupervisorResult,
    Tenor,
)


class TestForecastQuestion:
    def test_defaults(self):
        q = ForecastQuestion(text="Will USD/JPY be above 155?")
        assert q.pair == "USDJPY"
        assert q.cutoff_date == date.today()
        assert q.strike is None
        assert q.tenor is None

    def test_full(self):
        q = ForecastQuestion(
            text="test",
            pair="EURUSD",
            strike=1.10,
            tenor=Tenor.M1,
            cutoff_date=date(2025, 1, 1),
        )
        assert q.tenor == Tenor.M1
        assert q.strike == 1.10


class TestAgentForecast:
    def test_probability_bounds(self):
        f = AgentForecast(agent_id=0, probability=0.75, reasoning="test")
        assert f.probability == 0.75

    def test_probability_out_of_bounds(self):
        with pytest.raises(ValidationError):
            AgentForecast(agent_id=0, probability=1.5, reasoning="test")

        with pytest.raises(ValidationError):
            AgentForecast(agent_id=0, probability=-0.1, reasoning="test")


class TestSearchResult:
    def test_creation(self):
        r = SearchResult(
            query="test query",
            title="Test Title",
            snippet="Some snippet",
            url="https://example.com",
        )
        assert r.source == ""
        assert r.timestamp is None


class TestEnsembleResult:
    def test_creation(self):
        forecasts = [
            AgentForecast(agent_id=i, probability=0.5 + i * 0.05, reasoning=f"agent {i}")
            for i in range(3)
        ]
        result = EnsembleResult(
            agent_forecasts=forecasts,
            mean_probability=0.55,
            final_probability=0.55,
        )
        assert len(result.agent_forecasts) == 3
        assert result.supervisor is None


class TestTenor:
    def test_values(self):
        assert Tenor.D1.value == "1D"
        assert Tenor.W1.value == "1W"
        assert Tenor.M1.value == "1M"
        assert Tenor.M3.value == "3M"
        assert Tenor.M6.value == "6M"


class TestConfidence:
    def test_values(self):
        assert Confidence.HIGH.value == "high"
        assert Confidence.MEDIUM.value == "medium"
        assert Confidence.LOW.value == "low"
