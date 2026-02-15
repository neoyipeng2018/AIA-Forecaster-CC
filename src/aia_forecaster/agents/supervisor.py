"""Supervisor agent for reconciling disagreements among forecasting agents.

The supervisor's value comes from resolving SPECIFIC disagreements via targeted
search — NOT from holistic re-evaluation. The paper shows that naive LLM
aggregation (asking an LLM to pick the best forecast) performs WORSE than
simple averaging because LLMs overemphasize outlier opinions.

Algorithm:
1. Read all M reasoning traces
2. Identify sources of disagreement (factual vs. interpretive)
3. Generate targeted search queries to resolve disagreements
4. Execute searches and gather additional evidence
5. Produce reconciled forecast with confidence level (high/medium/low)
6. High confidence → replace mean; Medium/Low → fall back to mean
"""

from __future__ import annotations

import json
import logging
import re

from aia_forecaster.llm.client import LLMClient
from aia_forecaster.models import (
    AgentForecast,
    Confidence,
    ForecastQuestion,
    ResearchBrief,
    SearchResult,
    SupervisorResult,
    Tenor,
)
from aia_forecaster.search.web import search_web

logger = logging.getLogger(__name__)

DISAGREEMENT_PROMPT = """\
You are a senior FX supervisor reviewing forecasts from {num_agents} independent analysts.

QUESTION: {question}

INDIVIDUAL FORECASTS:
{forecasts_summary}

STATISTICS:
- Mean probability: {mean_prob:.4f}
- Min: {min_prob:.4f}, Max: {max_prob:.4f}, Spread: {spread:.4f}

Identify the KEY DISAGREEMENTS among these analysts. Focus on:
1. Factual disagreements (different assumptions about data or events)
2. Interpretive disagreements (same facts, different conclusions)
3. Missing information that some agents found and others didn't

For each disagreement, suggest a targeted search query that could help resolve it.

Respond in this EXACT JSON format:
{{
  "disagreements": [
    {{
      "description": "Brief description of the disagreement",
      "search_query": "Targeted search query to resolve it"
    }}
  ]
}}"""

RECONCILIATION_PROMPT = """\
You are a senior FX supervisor producing a RECONCILED forecast.

QUESTION: {question}

INDIVIDUAL FORECASTS:
{forecasts_summary}

Mean probability: {mean_prob:.4f}

ADDITIONAL EVIDENCE GATHERED TO RESOLVE DISAGREEMENTS:
{additional_evidence}

Based on the individual forecasts AND the additional evidence:
1. Assess whether the disagreements have been resolved
2. If you have HIGH confidence in a specific probability (the evidence clearly \
favors one interpretation), provide it
3. If evidence is ambiguous or doesn't resolve the disagreement, set confidence to "low"

IMPORTANT: Do NOT simply re-average or pick an outlier. Only override the mean if you \
have strong, specific evidence that resolves a factual disagreement.

Respond in this EXACT JSON format:
{{
  "confidence": "high" | "medium" | "low",
  "reconciled_probability": 0.XX,
  "reasoning": "Explanation of your reconciliation"
}}"""


SURFACE_REVIEW_PROMPT = """\
You are a senior FX supervisor reviewing a FULL probability surface for {pair}.

Spot rate: {spot}
Information cutoff: {cutoff_date}

PROBABILITY GRID (P(above strike) for each strike × tenor):
{grid_text}

SHARED RESEARCH THEMES (from {num_agents} agents):
{research_themes}

Review this probability surface for anomalies:
1. Non-monotonicity: P(above strike) should decrease as strike increases (for each tenor)
2. Tenor inconsistency: Longer tenors should generally show more regression toward 0.5 \
(probabilities for extreme strikes should move toward 0.5 at longer horizons)
3. Implausible values: Any cell that seems inconsistent with the evidence
4. Missing risk factors: Evidence themes that should shift specific cells but don't seem to

For each anomalous cell, decide if you have HIGH confidence in a correction.
Only override cells where specific evidence clearly demands it.

Respond in this EXACT JSON format:
{{
  "anomalies_found": true/false,
  "search_queries": ["targeted query 1", "targeted query 2"],
  "adjustments": [
    {{
      "strike": 155.00,
      "tenor": "1W",
      "current_probability": 0.XX,
      "adjusted_probability": 0.XX,
      "confidence": "high" | "medium" | "low",
      "reasoning": "Why this cell needs adjustment"
    }}
  ],
  "surface_reasoning": "Overall assessment of the surface quality"
}}"""


def _format_forecasts(forecasts: list[AgentForecast]) -> str:
    parts = []
    for f in forecasts:
        # Truncate reasoning to keep prompt manageable
        reasoning_short = f.reasoning[:500] + "..." if len(f.reasoning) > 500 else f.reasoning
        parts.append(
            f"Agent {f.agent_id}: p={f.probability:.4f}\n"
            f"  Reasoning: {reasoning_short}"
        )
    return "\n\n".join(parts)


class SupervisorAgent:
    """Reconciles disagreements among forecasting agents via targeted search."""

    def __init__(self, llm: LLMClient | None = None):
        self.llm = llm or LLMClient()

    async def reconcile(
        self,
        forecasts: list[AgentForecast],
        question: ForecastQuestion,
    ) -> SupervisorResult:
        """Analyze disagreements, search for resolution, and reconcile."""
        probs = [f.probability for f in forecasts]
        mean_prob = sum(probs) / len(probs)
        spread = max(probs) - min(probs)

        # If forecasts are already in tight agreement, skip reconciliation
        if spread < 0.10:
            logger.info("Supervisor: Spread %.3f < 0.10, skipping reconciliation", spread)
            return SupervisorResult(
                reconciled_probability=None,
                confidence=Confidence.LOW,
                reasoning=f"Agents in tight agreement (spread={spread:.3f}). Using mean.",
            )

        # Step 1: Identify disagreements
        disagreement_prompt = DISAGREEMENT_PROMPT.format(
            num_agents=len(forecasts),
            question=question.text,
            forecasts_summary=_format_forecasts(forecasts),
            mean_prob=mean_prob,
            min_prob=min(probs),
            max_prob=max(probs),
            spread=spread,
        )
        disagreement_response = await self.llm.complete(
            [{"role": "user", "content": disagreement_prompt}],
            temperature=0.3,
            max_tokens=1500,
        )

        # Parse disagreements and search queries
        search_queries = self._parse_disagreements(disagreement_response)
        logger.info("Supervisor: Found %d disagreements to resolve", len(search_queries))

        # Step 2: Execute targeted searches
        additional_evidence: list[SearchResult] = []
        for query in search_queries[:3]:  # Limit to 3 searches
            try:
                results = await search_web(
                    query=query,
                    max_results=5,
                    cutoff_date=question.cutoff_date,
                )
                additional_evidence.extend(results)
            except Exception:
                logger.warning("Supervisor: Search failed for '%s'", query)

        # Step 3: Reconcile
        evidence_text = "No additional evidence found."
        if additional_evidence:
            parts = []
            for i, e in enumerate(additional_evidence, 1):
                parts.append(f"[{i}] {e.title}\n    {e.snippet}")
            evidence_text = "\n\n".join(parts[:15])

        reconciliation_prompt = RECONCILIATION_PROMPT.format(
            question=question.text,
            forecasts_summary=_format_forecasts(forecasts),
            mean_prob=mean_prob,
            additional_evidence=evidence_text,
        )
        reconciliation_response = await self.llm.complete(
            [{"role": "user", "content": reconciliation_prompt}],
            temperature=0.3,
            max_tokens=1500,
        )

        return self._parse_reconciliation(reconciliation_response, additional_evidence)

    async def review_surface(
        self,
        pair: str,
        spot: float,
        cutoff_date,
        strikes: list[float],
        tenors: list[Tenor],
        cell_probabilities: dict[tuple[float, Tenor], float],
        briefs: list[ResearchBrief],
    ) -> dict[tuple[float, Tenor], float]:
        """Review the full probability surface and return adjustments.

        Unlike per-cell reconciliation, this reviews the entire grid for
        cross-cell consistency and does targeted searches only for
        genuine anomalies.

        Returns:
            Dict mapping (strike, tenor) → adjusted probability.
            Only contains cells that were adjusted with HIGH confidence.
        """
        # Build grid text
        grid_lines = ["Strike    " + "  ".join(f"{t.value:>6}" for t in tenors)]
        for strike in strikes:
            row = f"{strike:<10.2f}"
            for tenor in tenors:
                p = cell_probabilities.get((strike, tenor), float("nan"))
                row += f"  {p:>6.3f}"
            grid_lines.append(row)
        grid_text = "\n".join(grid_lines)

        # Collect research themes across all agents
        all_themes: list[str] = []
        for brief in briefs:
            all_themes.extend(brief.key_themes)
        # Deduplicate
        seen: set[str] = set()
        unique_themes: list[str] = []
        for theme in all_themes:
            normalized = theme.lower().strip()
            if normalized not in seen:
                seen.add(normalized)
                unique_themes.append(theme)
        research_themes = "\n".join(f"- {t}" for t in unique_themes[:15])

        prompt = SURFACE_REVIEW_PROMPT.format(
            pair=pair,
            spot=spot,
            cutoff_date=cutoff_date,
            grid_text=grid_text,
            num_agents=len(briefs),
            research_themes=research_themes or "No themes available.",
        )

        response = await self.llm.complete(
            [{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=2000,
        )

        # Parse response
        adjustments: dict[tuple[float, Tenor], float] = {}
        try:
            text = response.strip()
            if text.startswith("```"):
                lines = text.split("\n")
                lines = [l for l in lines if not l.strip().startswith("```")]
                text = "\n".join(lines)
            data = json.loads(text)

            surface_reasoning = data.get("surface_reasoning", "")
            logger.info("Supervisor surface review: %s", surface_reasoning[:200])

            # Execute targeted searches if any
            search_queries = data.get("search_queries", [])
            additional_evidence: list[SearchResult] = []
            for query in search_queries[:3]:
                try:
                    results = await search_web(
                        query=query,
                        max_results=5,
                        cutoff_date=cutoff_date,
                    )
                    additional_evidence.extend(results)
                except Exception:
                    logger.warning("Supervisor: Surface search failed for '%s'", query)

            # Apply HIGH confidence adjustments only
            for adj in data.get("adjustments", []):
                conf = adj.get("confidence", "low").lower()
                if conf != "high":
                    continue
                try:
                    strike = float(adj["strike"])
                    tenor = Tenor(adj["tenor"])
                    adjusted_p = max(0.0, min(1.0, float(adj["adjusted_probability"])))
                    adjustments[(strike, tenor)] = adjusted_p
                    logger.info(
                        "Supervisor adjusted [%s, %.2f]: %.4f → %.4f (%s)",
                        tenor.value, strike,
                        adj.get("current_probability", 0),
                        adjusted_p,
                        adj.get("reasoning", "")[:100],
                    )
                except (KeyError, ValueError) as e:
                    logger.warning("Supervisor: Invalid adjustment entry: %s", e)

        except (json.JSONDecodeError, KeyError) as e:
            logger.warning("Supervisor: Failed to parse surface review: %s", e)

        return adjustments

    def _parse_disagreements(self, response: str) -> list[str]:
        """Extract search queries from the disagreement analysis."""
        try:
            text = response.strip()
            if text.startswith("```"):
                lines = text.split("\n")
                lines = [l for l in lines if not l.strip().startswith("```")]
                text = "\n".join(lines)
            data = json.loads(text)
            return [d["search_query"] for d in data.get("disagreements", [])]
        except (json.JSONDecodeError, KeyError):
            logger.warning("Supervisor: Failed to parse disagreements, extracting queries by regex")
            return re.findall(r'"search_query"\s*:\s*"([^"]+)"', response)

    def _parse_reconciliation(
        self, response: str, evidence: list[SearchResult]
    ) -> SupervisorResult:
        """Parse the reconciliation response into a SupervisorResult."""
        try:
            text = response.strip()
            if text.startswith("```"):
                lines = text.split("\n")
                lines = [l for l in lines if not l.strip().startswith("```")]
                text = "\n".join(lines)
            data = json.loads(text)

            confidence_str = data.get("confidence", "low").lower()
            confidence = Confidence(confidence_str)
            prob = float(data.get("reconciled_probability", 0.5))
            prob = max(0.0, min(1.0, prob))
            reasoning = data.get("reasoning", "")

            return SupervisorResult(
                reconciled_probability=prob,
                confidence=confidence,
                reasoning=reasoning,
                additional_evidence=evidence,
            )
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.warning("Supervisor: Failed to parse reconciliation: %s", e)
            return SupervisorResult(
                reconciled_probability=None,
                confidence=Confidence.LOW,
                reasoning=f"Parse error: {e}. Falling back to mean.",
                additional_evidence=evidence,
            )
