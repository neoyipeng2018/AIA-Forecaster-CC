import json
from unittest.mock import AsyncMock

import pytest

from aia_forecaster.models import SearchResult, Tenor
from aia_forecaster.search.llm_relevance import (
    _format_results_block,
    _llm_judge_batch,
    _tenor_short,
    filter_relevant_llm,
)


def _make_result(title: str, snippet: str = "", url: str = "https://example.com") -> SearchResult:
    return SearchResult(
        query="test",
        title=title,
        snippet=snippet,
        url=url,
        source="test",
    )


def _mock_llm(response: str) -> AsyncMock:
    llm = AsyncMock()
    llm.complete = AsyncMock(return_value=response)
    return llm


# ---------------------------------------------------------------------------
# filter_relevant_llm
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_empty_input_returns_empty():
    llm = _mock_llm("")
    result = await filter_relevant_llm([], "USDJPY", llm)
    assert result == []
    llm.complete.assert_not_called()


@pytest.mark.asyncio
async def test_heuristic_prefilter_runs_first():
    results = [
        _make_result("USD/JPY bounces back above 152"),
        _make_result("Completely irrelevant cooking recipe"),
        _make_result("BOJ rate decision and Fed policy impact on yen dollar"),
    ]
    llm_response = json.dumps([
        {"index": 0, "decision": "keep", "reason": "relevant"},
        {"index": 1, "decision": "keep", "reason": "relevant"},
    ])
    llm = _mock_llm(llm_response)
    kept = await filter_relevant_llm(results, "USDJPY", llm, heuristic_threshold=0.10)

    titles = [r.title for r in kept]
    assert "Completely irrelevant cooking recipe" not in titles

    prompt_sent = llm.complete.call_args[0][0][0]["content"]
    assert "cooking" not in prompt_sent


@pytest.mark.asyncio
async def test_llm_drops_results():
    results = [
        _make_result("USD/JPY bounces back above 152"),
        _make_result("BOJ rate decision affects yen"),
        _make_result("Fed holds rates, dollar strengthens against yen"),
    ]
    llm_response = json.dumps([
        {"index": 0, "decision": "keep", "reason": "relevant"},
        {"index": 1, "decision": "drop", "reason": "not relevant enough"},
        {"index": 2, "decision": "keep", "reason": "relevant"},
    ])
    llm = _mock_llm(llm_response)
    kept = await filter_relevant_llm(results, "USDJPY", llm, heuristic_threshold=0.0)
    assert len(kept) == 2
    assert kept[0].title == "USD/JPY bounces back above 152"
    assert kept[1].title == "Fed holds rates, dollar strengthens against yen"


@pytest.mark.asyncio
async def test_llm_failure_fails_open():
    results = [
        _make_result("USD/JPY bounces back above 152"),
        _make_result("BOJ rate decision"),
    ]
    llm = AsyncMock()
    llm.complete = AsyncMock(side_effect=Exception("API down"))
    kept = await filter_relevant_llm(results, "USDJPY", llm, heuristic_threshold=0.0)
    assert len(kept) == 2


@pytest.mark.asyncio
async def test_llm_malformed_json_fails_open():
    results = [
        _make_result("USD/JPY bounces back above 152"),
        _make_result("BOJ rate decision"),
    ]
    llm = _mock_llm("this is not json")
    kept = await filter_relevant_llm(results, "USDJPY", llm, heuristic_threshold=0.0)
    assert len(kept) == 2


@pytest.mark.asyncio
async def test_tenor_clause_included_when_tenor_set():
    results = [_make_result("USD/JPY bounces back above 152")]
    llm_response = json.dumps([{"index": 0, "decision": "keep", "reason": "ok"}])
    llm = _mock_llm(llm_response)
    await filter_relevant_llm(results, "USDJPY", llm, tenor=Tenor("1W"), heuristic_threshold=0.0)

    prompt_sent = llm.complete.call_args[0][0][0]["content"]
    assert "1 week" in prompt_sent
    assert "SHORT-TERM" in prompt_sent


@pytest.mark.asyncio
async def test_tenor_clause_absent_when_tenor_none():
    results = [_make_result("USD/JPY bounces back above 152")]
    llm_response = json.dumps([{"index": 0, "decision": "keep", "reason": "ok"}])
    llm = _mock_llm(llm_response)
    await filter_relevant_llm(results, "USDJPY", llm, tenor=None, heuristic_threshold=0.0)

    prompt_sent = llm.complete.call_args[0][0][0]["content"]
    assert "SHORT-TERM" not in prompt_sent
    assert "MEDIUM-TERM" not in prompt_sent
    assert "LONG-TERM" not in prompt_sent


@pytest.mark.asyncio
async def test_batching_multiple_calls():
    results = [_make_result(f"USD/JPY article {i}") for i in range(15)]
    llm_response = json.dumps([{"index": i, "decision": "keep", "reason": "ok"} for i in range(10)])
    llm_response_2 = json.dumps([{"index": i, "decision": "keep", "reason": "ok"} for i in range(5)])
    llm = AsyncMock()
    llm.complete = AsyncMock(side_effect=[llm_response, llm_response_2])
    kept = await filter_relevant_llm(results, "USDJPY", llm, heuristic_threshold=0.0)
    assert llm.complete.call_count == 2
    assert len(kept) == 15


# ---------------------------------------------------------------------------
# _format_results_block
# ---------------------------------------------------------------------------


def test_format_results_block():
    results = [
        _make_result("Title A", snippet="Snippet A", url="https://a.com"),
        _make_result("Title B", snippet="Snippet B", url="https://b.com"),
    ]
    block = _format_results_block(results)
    assert "[0] Title: Title A" in block
    assert "[1] Title: Title B" in block
    assert "Snippet A" in block
    assert "https://b.com" in block


def test_format_results_block_truncates_snippet():
    long_snippet = "x" * 500
    results = [_make_result("Title", snippet=long_snippet)]
    block = _format_results_block(results)
    assert len(block.split("Snippet: ")[1].split("\n")[0]) <= 300


# ---------------------------------------------------------------------------
# _tenor_short
# ---------------------------------------------------------------------------


def test_tenor_short_day():
    assert _tenor_short(Tenor("1D")) == " within the next few days"


def test_tenor_short_month():
    assert _tenor_short(Tenor("3M")) == " within the next few months"


def test_tenor_short_none():
    assert _tenor_short(None) == ""


def test_tenor_short_week():
    assert _tenor_short(Tenor("2W")) == " within the next few weeks"


def test_tenor_short_year():
    assert _tenor_short(Tenor("1Y")) == " within the next year"
