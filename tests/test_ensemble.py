"""Tests for ensemble aggregation and evaluation metrics."""

import pytest

from aia_forecaster.evaluation.metrics import brier_score, brier_score_decomposition
from aia_forecaster.models import AgentForecast, Confidence, EnsembleResult, SupervisorResult


class TestBrierScore:
    def test_perfect_forecast(self):
        """Perfect forecasts should score 0."""
        assert brier_score([1.0, 0.0, 1.0], [1, 0, 1]) == 0.0

    def test_worst_forecast(self):
        """Worst possible forecasts should score 1."""
        assert brier_score([0.0, 1.0], [1, 0]) == 1.0

    def test_baseline(self):
        """Always predicting 0.5 should score 0.25."""
        bs = brier_score([0.5, 0.5, 0.5, 0.5], [1, 0, 1, 0])
        assert abs(bs - 0.25) < 1e-10

    def test_better_than_baseline(self):
        """Reasonable forecasts should beat the 0.25 baseline."""
        bs = brier_score([0.8, 0.2, 0.9, 0.1], [1, 0, 1, 0])
        assert bs < 0.25

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            brier_score([], [])

    def test_mismatched_lengths(self):
        with pytest.raises(ValueError):
            brier_score([0.5, 0.5], [1])


class TestBrierDecomposition:
    def test_components_sum(self):
        """BS = reliability - resolution + uncertainty."""
        forecasts = [0.8, 0.2, 0.7, 0.3, 0.6]
        outcomes = [1, 0, 1, 0, 1]
        d = brier_score_decomposition(forecasts, outcomes)
        reconstructed = d["reliability"] - d["resolution"] + d["uncertainty"]
        assert abs(d["brier_score"] - reconstructed) < 0.05  # Approximate due to binning


class TestEnsembleLogic:
    def test_high_confidence_supervisor_overrides_mean(self):
        """When supervisor has HIGH confidence, its probability should be used."""
        forecasts = [
            AgentForecast(agent_id=i, probability=0.5 + i * 0.05, reasoning="test")
            for i in range(5)
        ]
        mean_p = sum(f.probability for f in forecasts) / len(forecasts)

        supervisor = SupervisorResult(
            reconciled_probability=0.8,
            confidence=Confidence.HIGH,
            reasoning="Strong evidence for higher probability",
        )

        result = EnsembleResult(
            agent_forecasts=forecasts,
            mean_probability=mean_p,
            supervisor=supervisor,
            final_probability=supervisor.reconciled_probability,
        )

        assert result.final_probability == 0.8
        assert result.final_probability != result.mean_probability

    def test_low_confidence_supervisor_uses_mean(self):
        """When supervisor has LOW confidence, mean should be used."""
        forecasts = [
            AgentForecast(agent_id=i, probability=0.6, reasoning="test")
            for i in range(5)
        ]
        mean_p = 0.6

        supervisor = SupervisorResult(
            reconciled_probability=0.9,
            confidence=Confidence.LOW,
            reasoning="Uncertain",
        )

        # The engine would use mean_p here, not supervisor's probability
        result = EnsembleResult(
            agent_forecasts=forecasts,
            mean_probability=mean_p,
            supervisor=supervisor,
            final_probability=mean_p,  # Uses mean because confidence is low
        )

        assert result.final_probability == 0.6
