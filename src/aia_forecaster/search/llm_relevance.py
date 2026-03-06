from __future__ import annotations

import json
import logging

from aia_forecaster.llm.client import LLMClient
from aia_forecaster.models import SearchResult, Tenor
from aia_forecaster.search.relevance import filter_relevant as heuristic_filter

logger = logging.getLogger(__name__)

_BATCH_SIZE = 10

_RELEVANCE_PROMPT = """\
You are an FX research analyst. Evaluate whether each search result below is \
relevant to forecasting the {pair} currency pair.{tenor_clause}

SEARCH RESULTS:
{results_block}

For each result, decide:
- "keep" if it contains information useful for forecasting {pair}{tenor_short}
- "drop" if it is about a different currency pair, a different asset class, \
or contains no actionable FX information

Respond in this EXACT JSON format (array of objects, same order as input):
[
  {{"index": 0, "decision": "keep", "reason": "BOJ rate decision directly affects JPY"}},
  {{"index": 1, "decision": "drop", "reason": "Article about gold prices, not relevant to USDJPY"}}
]"""

_TENOR_CLAUSE = """
You are filtering for the **{tenor_label}** forecast horizon specifically. \
Consider whether the information is actionable within this timeframe:
- SHORT-TERM (1D-2W): Only keep if it describes events, data releases, or \
positioning shifts that will materialize within days/weeks.
- MEDIUM-TERM (1M-3M): Keep if it describes policy meetings, macro trends, \
or positioning that affects the pair over weeks/months.
- LONG-TERM (6M+): Keep if it describes structural shifts, policy divergence \
trajectories, or long-term flow dynamics."""

_TENOR_SHORT: dict[str, str] = {
    "D": " within the next few days",
    "W": " within the next few weeks",
    "M": " within the next few months",
    "Y": " within the next year",
}


def _tenor_short(tenor: Tenor | None) -> str:
    if tenor is None:
        return ""
    return _TENOR_SHORT.get(tenor[-1], "")


def _format_results_block(results: list[SearchResult]) -> str:
    lines: list[str] = []
    for i, r in enumerate(results):
        lines.append(
            f"[{i}] Title: {r.title}\n"
            f"    Snippet: {(r.snippet or '')[:300]}\n"
            f"    URL: {r.url}"
        )
    return "\n\n".join(lines)


async def _llm_judge_batch(
    results: list[SearchResult],
    pair: str,
    tenor: Tenor | None,
    llm: LLMClient,
) -> list[bool]:
    """Judge a batch of results via a single LLM call.

    Returns list of booleans (True = keep). Fails open (all True) on error.
    """
    if not results:
        return []

    tenor_clause = ""
    if tenor is not None:
        tenor_clause = _TENOR_CLAUSE.format(tenor_label=tenor.label)

    prompt = _RELEVANCE_PROMPT.format(
        pair=pair,
        tenor_clause=tenor_clause,
        tenor_short=_tenor_short(tenor),
        results_block=_format_results_block(results),
    )

    try:
        response = await llm.complete(
            [{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=2000,
        )

        text = response.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            lines = [ln for ln in lines if not ln.strip().startswith("```")]
            text = "\n".join(lines)

        decisions = json.loads(text)

        keep_set: set[int] = set()
        for item in decisions:
            idx = item.get("index", -1)
            decision = str(item.get("decision", "keep")).lower().strip()
            if decision == "keep":
                keep_set.add(idx)

        return [i in keep_set for i in range(len(results))]

    except Exception:
        logger.warning("LLM relevance filter failed — keeping all results (fail-open)")
        return [True] * len(results)


async def filter_relevant_llm(
    results: list[SearchResult],
    pair: str,
    llm: LLMClient,
    *,
    tenor: Tenor | None = None,
    heuristic_threshold: float = 0.10,
) -> list[SearchResult]:
    """Two-tier relevance filter: heuristic pre-filter + LLM judge."""
    if not results:
        return []

    survivors = heuristic_filter(results, pair, threshold=heuristic_threshold)
    pre_count = len(results) - len(survivors)
    if pre_count > 0:
        logger.info(
            "LLM relevance: heuristic pre-filter removed %d/%d results",
            pre_count, len(results),
        )

    if not survivors:
        return []

    kept: list[SearchResult] = []
    for start in range(0, len(survivors), _BATCH_SIZE):
        batch = survivors[start : start + _BATCH_SIZE]
        flags = await _llm_judge_batch(batch, pair, tenor, llm)
        for result, keep in zip(batch, flags):
            if keep:
                kept.append(result)
            else:
                logger.debug(
                    "LLM relevance: dropped '%s'",
                    result.title[:80] if result.title else "(no title)",
                )

    llm_count = len(survivors) - len(kept)
    if llm_count > 0:
        logger.info(
            "LLM relevance: LLM dropped %d/%d (pair=%s, tenor=%s)",
            llm_count, len(survivors), pair,
            tenor.value if tenor else "none",
        )

    return kept
