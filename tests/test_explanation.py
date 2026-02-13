"""Tests for the explanation extraction module."""

from datetime import datetime

from aia_forecaster.fx.explanation import (
    _deduplicate_evidence,
    _first_sentence,
    _summarize_consensus,
    _summarize_disagreements,
    explain_cell,
    explain_surface,
)
from aia_forecaster.models import (
    AgentForecast,
    CalibratedForecast,
    Confidence,
    EnsembleResult,
    ProbabilitySurface,
    SearchResult,
    SupervisorResult,
    SurfaceCell,
    Tenor,
)


def _make_search_result(url: str = "https://example.com/1", title: str = "Test") -> SearchResult:
    return SearchResult(query="test", title=title, snippet="snippet", url=url)


def _make_agent(
    agent_id: int = 0,
    probability: float = 0.6,
    reasoning: str = "The yen is weakening due to BOJ policy. This suggests upside.",
    evidence: list[SearchResult] | None = None,
) -> AgentForecast:
    return AgentForecast(
        agent_id=agent_id,
        probability=probability,
        reasoning=reasoning,
        evidence=evidence or [],
        iterations=3,
    )


def _make_cell(
    agents: list[AgentForecast] | None = None,
    supervisor: SupervisorResult | None = None,
) -> SurfaceCell:
    if agents is None:
        agents = [_make_agent(i, 0.6 + i * 0.02) for i in range(3)]
    ensemble = EnsembleResult(
        agent_forecasts=agents,
        mean_probability=sum(a.probability for a in agents) / len(agents),
        supervisor=supervisor,
        final_probability=sum(a.probability for a in agents) / len(agents),
    )
    cal = CalibratedForecast(
        raw_probability=ensemble.final_probability,
        calibrated_probability=ensemble.final_probability + 0.05,
        alpha=1.732,
    )
    return SurfaceCell(
        strike=155.0,
        tenor=Tenor.W1,
        question="Will USD/JPY be above 155.00 in 1 week?",
        calibrated=cal,
        ensemble=ensemble,
    )


class TestFirstSentence:
    def test_period(self):
        assert _first_sentence("Hello world. More text.") == "Hello world."

    def test_question_mark(self):
        assert _first_sentence("Is it rising? Yes.") == "Is it rising?"

    def test_no_terminator(self):
        text = "a" * 300
        assert len(_first_sentence(text)) == 200


class TestDeduplicateEvidence:
    def test_counts_citations(self):
        e1 = _make_search_result("https://example.com/1", "Article A")
        e2 = _make_search_result("https://example.com/1", "Article A")  # same URL
        e3 = _make_search_result("https://example.com/2", "Article B")

        agents = [
            _make_agent(0, 0.6, evidence=[e1, e3]),
            _make_agent(1, 0.7, evidence=[e2]),
        ]
        items = _deduplicate_evidence(agents)

        # example.com/1 cited by 2 agents, example.com/2 by 1
        assert len(items) == 2
        assert items[0].cited_by_agents == 2
        assert items[1].cited_by_agents == 1

    def test_empty_evidence(self):
        agents = [_make_agent(0, 0.5)]
        assert _deduplicate_evidence(agents) == []

    def test_top_n_limit(self):
        evidence = [_make_search_result(f"https://example.com/{i}") for i in range(10)]
        agents = [_make_agent(0, 0.5, evidence=evidence)]
        items = _deduplicate_evidence(agents, top_n=3)
        assert len(items) == 3

    def test_trailing_slash_normalization(self):
        e1 = _make_search_result("https://example.com/page/")
        e2 = _make_search_result("https://example.com/page")
        agents = [
            _make_agent(0, 0.6, evidence=[e1]),
            _make_agent(1, 0.7, evidence=[e2]),
        ]
        items = _deduplicate_evidence(agents)
        assert len(items) == 1
        assert items[0].cited_by_agents == 2


class TestSummarizeConsensus:
    def test_above_half(self):
        agents = [_make_agent(i, 0.7) for i in range(3)]
        summary = _summarize_consensus(agents)
        assert "above" in summary
        assert "0.700" in summary

    def test_below_half(self):
        agents = [_make_agent(i, 0.3) for i in range(3)]
        summary = _summarize_consensus(agents)
        assert "below" in summary

    def test_empty(self):
        assert _summarize_consensus([]) == ""


class TestSummarizeDisagreements:
    def test_low_spread_no_output(self):
        agents = [_make_agent(i, 0.60 + i * 0.01) for i in range(3)]
        assert _summarize_disagreements(agents, None) == ""

    def test_high_spread(self):
        agents = [
            _make_agent(0, 0.2),
            _make_agent(1, 0.8),
            _make_agent(2, 0.5),
        ]
        result = _summarize_disagreements(agents, None)
        assert "Spread" in result

    def test_supervisor_included_if_high_confidence(self):
        agents = [_make_agent(0, 0.2), _make_agent(1, 0.8)]
        supervisor = SupervisorResult(
            reconciled_probability=0.6,
            confidence=Confidence.HIGH,
            reasoning="Resolved by checking BOJ minutes.",
        )
        result = _summarize_disagreements(agents, supervisor)
        assert "Supervisor" in result
        assert "HIGH" in result


class TestExplainCell:
    def test_with_ensemble(self):
        cell = _make_cell()
        explanation = explain_cell(cell)
        assert explanation.num_agents == 3
        assert len(explanation.agent_probabilities) == 3
        assert explanation.calibrated_probability is not None

    def test_without_ensemble(self):
        cell = SurfaceCell(
            strike=155.0,
            tenor=Tenor.W1,
            question="test",
            calibrated=CalibratedForecast(raw_probability=0.5, calibrated_probability=0.55, alpha=1.732),
        )
        explanation = explain_cell(cell)
        assert explanation.num_agents == 0
        assert explanation.top_evidence == []


class TestExplainSurface:
    def test_generates_explanations_for_all_cells(self):
        cells = [
            _make_cell([_make_agent(i, 0.5 + i * 0.1) for i in range(3)]),
            _make_cell([_make_agent(i, 0.4 + i * 0.05) for i in range(3)]),
        ]
        surface = ProbabilitySurface(
            pair="USDJPY",
            spot_rate=154.0,
            cells=cells,
        )
        explanation = explain_surface(surface)
        assert explanation.pair == "USDJPY"
        assert len(explanation.cells) == 2
