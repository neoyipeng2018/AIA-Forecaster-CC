"""Single forecasting agent with agentic, adaptive search loop.

Each agent independently:
1. Receives a binary forecasting question
2. Iteratively searches (web + RSS), conditioning each query on prior results
3. Decides when it has enough evidence to produce a forecast
4. Outputs a probability estimate and reasoning trace

The LLM has full discretion over what to search — this is the "agentic" part
that the AIA Forecaster paper shows dramatically outperforms fixed-query approaches.
"""

from __future__ import annotations

import json
import logging
import re

from aia_forecaster.config import settings
from aia_forecaster.fx.base_rates import format_base_rate_context
from aia_forecaster.llm.client import LLMClient
from aia_forecaster.models import AgentForecast, ForecastQuestion, SearchResult
from aia_forecaster.search.rss import fetch_fx_news
from aia_forecaster.search.web import search_web

logger = logging.getLogger(__name__)

QUERY_GENERATION_PROMPT = """\
You are an FX research analyst. You are investigating the following binary forecasting question:

QUESTION: {question}

Your information cutoff date is {cutoff_date}. Do NOT use any information after this date.

{base_rate_section}
{evidence_section}

Based on what you know and any evidence gathered so far, generate the SINGLE BEST search \
query to find information that would help you estimate the probability of this event. \
Focus on finding factual, quantitative data — central bank policy signals, economic \
indicators, market positioning, or geopolitical developments.

Respond with ONLY the search query string, nothing else."""

ASSESS_PROMPT = """\
You are an FX research analyst investigating this binary forecasting question:

QUESTION: {question}

You have performed {iteration} search iteration(s) so far and gathered the following evidence:

{evidence_summary}

Do you have SUFFICIENT evidence to make a well-informed probability estimate? \
Consider whether you have:
- Relevant macroeconomic data and central bank policy signals
- Current market conditions and recent price action context
- Geopolitical factors that could affect the currency pair
- Enough diverse sources to form a balanced view

Respond with EXACTLY one word: SEARCH (if you need more evidence) or FORECAST (if ready)."""

FORECAST_PROMPT = """\
You are an expert FX forecasting analyst. You must estimate the probability for this \
binary question:

QUESTION: {question}

Your information cutoff date is {cutoff_date}. Base your analysis ONLY on information \
available before this date.

EVIDENCE GATHERED:
{evidence_summary}

{base_rate_section}
Analyze the evidence carefully. Consider:
1. Base rates and historical precedents for similar FX moves
2. Current monetary policy stance and expected changes
3. Macro fundamentals (growth, inflation, trade balance, employment)
4. Market positioning and sentiment
5. Geopolitical risks and safe-haven flows
6. Technical levels and recent price action

Provide your analysis and then give a precise probability estimate.

IMPORTANT: Do NOT hedge toward 0.5 out of uncertainty. If evidence points in one direction, \
commit to a probability that reflects that evidence. Be specific (e.g., 0.35, 0.72), \
not vague (e.g., 0.50, 0.55).

Respond in this EXACT JSON format:
{{
  "reasoning": "Your detailed reasoning trace (2-4 paragraphs)",
  "probability": 0.XX
}}"""


def _format_evidence(evidence: list[SearchResult], max_chars: int = 6000) -> str:
    """Format evidence list into a readable summary for the LLM."""
    if not evidence:
        return "No evidence gathered yet."

    parts = []
    total_chars = 0
    for i, e in enumerate(evidence, 1):
        entry = f"[{i}] {e.title}\n    {e.snippet}\n    Source: {e.url}"
        if total_chars + len(entry) > max_chars:
            parts.append(f"... ({len(evidence) - i + 1} more results truncated)")
            break
        parts.append(entry)
        total_chars += len(entry)
    return "\n\n".join(parts)


class ForecastingAgent:
    """A single forecasting agent that performs agentic search and produces a probability."""

    def __init__(self, agent_id: int, llm: LLMClient | None = None):
        self.agent_id = agent_id
        self.llm = llm or LLMClient()

    def _build_base_rate_section(self, question: ForecastQuestion) -> str:
        """Build the base rate context block if spot/strike/tenor are available."""
        if question.spot is not None and question.strike is not None and question.tenor is not None:
            try:
                return format_base_rate_context(
                    pair=question.pair,
                    spot=question.spot,
                    strike=question.strike,
                    tenor=question.tenor,
                )
            except ValueError:
                return ""
        return ""

    async def forecast(self, question: ForecastQuestion) -> AgentForecast:
        """Run the full agentic search loop and return a forecast.

        Steps:
        1. Fetch RSS news as background context
        2. Loop up to max_search_iterations:
           a. Generate adaptive search query
           b. Execute web search
           c. Assess if enough evidence to forecast
        3. Generate probability estimate with reasoning
        """
        all_evidence: list[SearchResult] = []
        all_queries: list[str] = []
        base_rate_section = self._build_base_rate_section(question)

        # Start with RSS news as background context
        try:
            rss_news = await fetch_fx_news(
                currency_pair=question.pair,
                max_age_hours=48,
                max_results=10,
            )
            all_evidence.extend(rss_news)
            logger.info("Agent %d: Got %d RSS news items", self.agent_id, len(rss_news))
        except Exception:
            logger.warning("Agent %d: RSS fetch failed, continuing with web search", self.agent_id)

        # Agentic search loop
        for iteration in range(1, settings.max_search_iterations + 1):
            # Step 1: Generate search query
            evidence_section = ""
            if all_evidence:
                evidence_section = (
                    f"Evidence gathered so far:\n{_format_evidence(all_evidence)}"
                )

            query_prompt = QUERY_GENERATION_PROMPT.format(
                question=question.text,
                cutoff_date=question.cutoff_date.isoformat(),
                base_rate_section=base_rate_section,
                evidence_section=evidence_section,
            )
            search_query = await self.llm.complete(
                [{"role": "user", "content": query_prompt}],
                temperature=0.7 + (self.agent_id % 5) * 0.05,  # Slight diversity
                max_tokens=200,
            )
            search_query = search_query.strip().strip('"')
            all_queries.append(search_query)
            logger.info("Agent %d, iter %d: Searching '%s'", self.agent_id, iteration, search_query)

            # Step 2: Execute web search
            try:
                results = await search_web(
                    query=search_query,
                    max_results=5,
                    cutoff_date=question.cutoff_date,
                )
                all_evidence.extend(results)
            except Exception:
                logger.warning("Agent %d: Search failed for query '%s'", self.agent_id, search_query)

            # Step 3: Assess if we have enough evidence (skip on last iteration)
            if iteration < settings.max_search_iterations:
                assess_prompt = ASSESS_PROMPT.format(
                    question=question.text,
                    iteration=iteration,
                    evidence_summary=_format_evidence(all_evidence),
                )
                decision = await self.llm.complete(
                    [{"role": "user", "content": assess_prompt}],
                    temperature=0.3,
                    max_tokens=10,
                )
                decision = decision.strip().upper()
                if "FORECAST" in decision:
                    logger.info("Agent %d: Ready to forecast after %d iterations", self.agent_id, iteration)
                    break

        # Generate forecast
        forecast_prompt = FORECAST_PROMPT.format(
            question=question.text,
            cutoff_date=question.cutoff_date.isoformat(),
            evidence_summary=_format_evidence(all_evidence, max_chars=8000),
            base_rate_section=base_rate_section,
        )
        response = await self.llm.complete(
            [{"role": "user", "content": forecast_prompt}],
            temperature=0.5,
            max_tokens=2000,
        )

        # Parse response
        probability, reasoning = self._parse_forecast_response(response)

        return AgentForecast(
            agent_id=self.agent_id,
            probability=probability,
            reasoning=reasoning,
            search_queries=all_queries,
            evidence=all_evidence,
            iterations=len(all_queries),
        )

    def _parse_forecast_response(self, response: str) -> tuple[float, str]:
        """Extract probability and reasoning from the LLM response."""
        # Try JSON parse first
        try:
            # Find JSON in the response
            text = response.strip()
            if text.startswith("```"):
                lines = text.split("\n")
                lines = [l for l in lines if not l.strip().startswith("```")]
                text = "\n".join(lines)

            data = json.loads(text)
            prob = float(data["probability"])
            reasoning = data.get("reasoning", "")
            return max(0.0, min(1.0, prob)), reasoning
        except (json.JSONDecodeError, KeyError, ValueError):
            pass

        # Fallback: regex extraction
        prob_match = re.search(r'"probability"\s*:\s*(0?\.\d+|1\.0|0|1)', response)
        if prob_match:
            prob = float(prob_match.group(1))
        else:
            # Last resort: find any decimal between 0 and 1
            decimals = re.findall(r'\b(0\.\d+)\b', response)
            prob = float(decimals[-1]) if decimals else 0.5

        return max(0.0, min(1.0, prob)), response
