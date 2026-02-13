"""Foreknowledge bias detection using LLM-as-a-judge.

Search APIs can leak future information via:
- Live data widgets (e.g., current stock prices on pages about past events)
- Updated Wikipedia pages
- Republished articles with post-cutoff edits

This module prompts an LLM to assess whether search results contain
information from after the specified cutoff date.
"""

from __future__ import annotations

import json
import logging
from datetime import date

from aia_forecaster.llm.client import LLMClient
from aia_forecaster.models import Confidence, FlaggedResult, SearchResult

logger = logging.getLogger(__name__)

FOREKNOWLEDGE_PROMPT = """\
You are a temporal information auditor. Your task is to determine whether \
a search result contains information from AFTER a specified cutoff date.

CUTOFF DATE: {cutoff_date}

SEARCH RESULT:
Title: {title}
Snippet: {snippet}
URL: {url}

Analyze this search result for signs of POST-CUTOFF information:
1. Does it mention events, data, or facts from after {cutoff_date}?
2. Does it use past-tense language about events that should be in the future?
3. Does it contain specific data points (prices, statistics) that could only be known after the cutoff?

Respond in this EXACT JSON format:
{{
  "has_foreknowledge": true/false,
  "confidence": "high" | "medium" | "low",
  "evidence": "Brief explanation of what post-cutoff information was found, or 'None detected'"
}}"""


async def check_foreknowledge(
    results: list[SearchResult],
    cutoff_date: date,
    llm: LLMClient | None = None,
) -> list[FlaggedResult]:
    """Check search results for foreknowledge bias.

    Args:
        results: Search results to check.
        cutoff_date: The temporal cutoff — any info after this date is foreknowledge.
        llm: LLM client for the judge.

    Returns:
        List of FlaggedResult for each input result.
    """
    if llm is None:
        llm = LLMClient()

    flagged: list[FlaggedResult] = []

    for result in results:
        prompt = FOREKNOWLEDGE_PROMPT.format(
            cutoff_date=cutoff_date.isoformat(),
            title=result.title,
            snippet=result.snippet[:1000],
            url=result.url,
        )

        try:
            response = await llm.complete(
                [{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=300,
            )

            # Parse response
            text = response.strip()
            if text.startswith("```"):
                lines = text.split("\n")
                lines = [l for l in lines if not l.strip().startswith("```")]
                text = "\n".join(lines)

            data = json.loads(text)
            flagged.append(
                FlaggedResult(
                    result=result,
                    has_foreknowledge=data.get("has_foreknowledge", False),
                    confidence=Confidence(data.get("confidence", "low")),
                    evidence=data.get("evidence", ""),
                )
            )
        except Exception:
            logger.warning("Foreknowledge check failed for: %s", result.title)
            flagged.append(
                FlaggedResult(
                    result=result,
                    has_foreknowledge=False,
                    confidence=Confidence.LOW,
                    evidence="Check failed",
                )
            )

    num_flagged = sum(1 for f in flagged if f.has_foreknowledge)
    if num_flagged:
        logger.warning("Foreknowledge detected in %d/%d results", num_flagged, len(results))

    return flagged


def filter_foreknowledge(flagged: list[FlaggedResult]) -> list[SearchResult]:
    """Return only the results that are NOT flagged for foreknowledge."""
    return [f.result for f in flagged if not f.has_foreknowledge]
