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

from aia_forecaster.config import settings
from aia_forecaster.llm.client import LLMClient
from aia_forecaster.models import (
    AgentForecast,
    CausalFactor,
    Confidence,
    ForecastQuestion,
    ResearchBrief,
    SearchResult,
    SupervisorResult,
    Tenor,
)
from aia_forecaster.search.web import search_web

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Regime detection
# ---------------------------------------------------------------------------

REGIME_DETECTION_PROMPT = """\
You are a senior FX macro strategist. Based on the research evidence below, identify the \
current dominant market REGIME for {pair}.

RESEARCH THEMES AND CAUSAL FACTORS:
{causal_factors_summary}

AGENT MACRO SUMMARIES (first 3):
{macro_excerpts}

Classify the current regime as ONE of:
- **risk_on**: Risk appetite is dominant — carry trades active, EM/high-yield currencies \
strengthening, VIX low, equities rallying. Channels: carry trade, portfolio flows, \
risk sentiment dominate.
- **risk_off**: Risk aversion is dominant — safe-haven flows into USD/JPY/CHF, \
VIX elevated, flight to quality. Channels: safe-haven demand, deleveraging dominate.
- **policy_divergence**: Central bank policy differences are the primary driver — rate \
differentials, forward guidance divergence, QE/QT differences. Channels: rate differential, \
yield spread, forward guidance dominate.
- **carry_unwind**: Active unwinding of carry positions — sharp reversal in funding \
currencies (JPY, CHF), position liquidation, margin calls. Channels: positioning, \
forced liquidation, margin dynamics dominate.
- **mixed**: No single regime dominates — multiple competing forces, regime transition, \
or conflicting signals.

Also state which causal CHANNELS are most important in the current regime.

Respond in this EXACT JSON format:
{{
  "regime": "risk_on | risk_off | policy_divergence | carry_unwind | mixed",
  "confidence": "high | medium | low",
  "dominant_channels": ["channel1", "channel2"],
  "reasoning": "1-2 sentence explanation"
}}"""

# Mapping from regime → which causal channels carry the most weight
REGIME_CHANNEL_WEIGHTS: dict[str, dict[str, float]] = {
    "risk_on": {
        "carry trade": 1.5, "portfolio flows": 1.3, "risk appetite": 1.5,
        "risk sentiment": 1.5, "equity correlation": 1.2,
        "rate differential": 0.8, "safe-haven demand": 0.5,
    },
    "risk_off": {
        "safe-haven demand": 1.5, "deleveraging": 1.5, "risk appetite": 1.3,
        "risk sentiment": 1.3, "flight to quality": 1.5,
        "carry trade": 0.5, "rate differential": 0.7,
    },
    "policy_divergence": {
        "rate differential": 1.5, "yield spread": 1.5, "forward guidance": 1.5,
        "monetary policy": 1.5, "central bank": 1.3,
        "carry trade": 1.0, "safe-haven demand": 0.7,
    },
    "carry_unwind": {
        "positioning": 1.5, "forced liquidation": 1.5, "margin dynamics": 1.5,
        "carry trade": 1.5, "deleveraging": 1.3,
        "rate differential": 0.5, "safe-haven demand": 1.0,
    },
    "mixed": {},  # No reweighting in mixed regime
}

DISAGREEMENT_PROMPT = """\
You are a senior FX supervisor reviewing forecasts from {num_agents} independent analysts.

QUESTION: {question}

INDIVIDUAL FORECASTS AND CAUSAL CHAINS:
{forecasts_summary}

STATISTICS:
- Mean probability: {mean_prob:.4f}
- Min: {min_prob:.4f}, Max: {max_prob:.4f}, Spread: {spread:.4f}

Your job is to find exactly WHERE the analysts diverge by comparing their causal chains.

Classify each disagreement into one of these types:
1. **Factual**: Agents disagree about what happened or will happen \
(e.g., one assumes BOJ will hike, another assumes hold). \
→ Search for the factual answer.
2. **Channel**: Agents agree on the event but disagree on HOW it transmits to the pair \
(e.g., both see oil price spike, but one routes through "risk-off → USD safe haven" while \
the other routes through "inflation expectations → Fed hawkishness"). \
→ Search for evidence on which transmission channel is dominant right now.
3. **Magnitude**: Agents agree on event and channel but disagree on how much it matters. \
→ Search for quantitative evidence (positioning data, vol surface, historical analogues).
4. **Missing factor**: Some agents identified a causal factor that others missed entirely. \
→ Search to verify whether the missing factor is real and material.

For each disagreement, suggest a TARGETED search query that resolves the specific divergence point.

Respond in this EXACT JSON format:
{{
  "disagreements": [
    {{
      "type": "factual | channel | magnitude | missing_factor",
      "description": "What specifically the agents disagree about",
      "agents_bullish": [0, 2],
      "agents_bearish": [1, 3],
      "search_query": "Targeted search query to resolve this specific divergence"
    }}
  ]
}}"""

RECONCILIATION_PROMPT = """\
You are a senior FX supervisor producing a RECONCILED forecast.

QUESTION: {question}

INDIVIDUAL FORECASTS AND CAUSAL CHAINS:
{forecasts_summary}

Mean probability: {mean_prob:.4f}

IDENTIFIED DISAGREEMENTS:
{disagreements_summary}

ADDITIONAL EVIDENCE GATHERED TO RESOLVE DISAGREEMENTS:
{additional_evidence}

For each disagreement identified above, assess whether the new evidence resolves it:
- **Factual**: Does the evidence confirm one side's facts? If yes, weight those agents higher.
- **Channel**: Does the evidence reveal which transmission channel is currently dominant? \
Consider: what is the market's current regime (risk-on/off, carry-dominated, policy-driven)?
- **Magnitude**: Does the evidence provide quantitative support for the impact size?
- **Missing factor**: Is the newly identified factor confirmed as real and material?

IMPORTANT:
- Do NOT simply re-average or pick an outlier
- Only override the mean if the evidence clearly resolves a specific divergence point
- If competing causal channels are BOTH active, the correct answer is often between the \
extremes — but not necessarily at the mean. Weight by channel dominance.

Respond in this EXACT JSON format:
{{
  "confidence": "high" | "medium" | "low",
  "reconciled_probability": 0.XX,
  "resolved_disagreements": ["brief description of what was resolved"],
  "unresolved_disagreements": ["brief description of what remains ambiguous"],
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

CONSENSUS CAUSAL FACTORS (factors identified by multiple agents):
{causal_factors_summary}

Review this probability surface for anomalies:
1. Non-monotonicity: P(above strike) should decrease as strike increases (for each tenor)
2. Tenor inconsistency: Longer tenors should generally show more regression toward 0.5 \
(probabilities for extreme strikes should move toward 0.5 at longer horizons)
3. Implausible values: Any cell that seems inconsistent with the evidence
4. Causal factor mismatch: Check that strong, high-confidence causal factors are actually \
reflected in the surface. A "strong bullish" factor at short tenors should visibly shift \
near-spot probabilities upward. If a consensus factor isn't reflected, that cell needs review.
5. Temporal mismatch: Fast-acting factors (positioning, sentiment) should mainly affect \
short tenors (1D, 1W). Slow-acting factors (policy divergence, trade flows) should mainly \
affect long tenors (3M, 6M). Flag cells where the wrong factor type dominates.

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


def _format_causal_factors_for_supervisor(factors: list[CausalFactor]) -> str:
    """Format causal factors for supervisor prompts."""
    if not factors:
        return "  (no causal factors reported)"
    lines = []
    for f in factors:
        lines.append(
            f"  - {f.event} → [{f.channel}] → {f.direction} "
            f"(magnitude: {f.magnitude}, confidence: {f.confidence})"
        )
    return "\n".join(lines)


def _format_forecasts(forecasts: list[AgentForecast], briefs: list[ResearchBrief] | None = None) -> str:
    """Format forecasts with causal chains for the supervisor.

    If briefs are provided, causal factors from the matching brief are included.
    """
    brief_map: dict[int, ResearchBrief] = {}
    if briefs:
        for b in briefs:
            brief_map[b.agent_id] = b

    parts = []
    for f in forecasts:
        reasoning_short = f.reasoning[:500] + "..." if len(f.reasoning) > 500 else f.reasoning
        brief = brief_map.get(f.agent_id)
        causal_section = ""
        if brief and brief.causal_factors:
            causal_section = f"\n  Causal factors:\n{_format_causal_factors_for_supervisor(brief.causal_factors)}"
        parts.append(
            f"Agent {f.agent_id}: p={f.probability:.4f}\n"
            f"  Reasoning: {reasoning_short}{causal_section}"
        )
    return "\n\n".join(parts)


def _build_consensus_causal_summary(briefs: list[ResearchBrief]) -> str:
    """Identify causal factors cited by multiple agents and format for the supervisor.

    Groups factors by (event_keyword, channel) similarity and reports frequency.
    """
    if not briefs:
        return "No causal factors available."

    # Collect all factors with agent attribution
    all_factors: list[tuple[int, CausalFactor]] = []
    for brief in briefs:
        for f in brief.causal_factors:
            all_factors.append((brief.agent_id, f))

    if not all_factors:
        return "No causal factors reported by any agent."

    # Simple dedup: group by (channel, direction) — exact event wording varies
    groups: dict[tuple[str, str], list[tuple[int, CausalFactor]]] = {}
    for agent_id, f in all_factors:
        key = (f.channel.lower().strip(), f.direction.lower().strip())
        groups.setdefault(key, []).append((agent_id, f))

    lines = []
    for (channel, direction), entries in sorted(groups.items(), key=lambda x: -len(x[1])):
        count = len(entries)
        agent_ids = sorted({aid for aid, _ in entries})
        representative = entries[0][1]
        magnitudes = [e.magnitude for _, e in entries]
        most_common_mag = max(set(magnitudes), key=magnitudes.count)
        lines.append(
            f"- [{count}/{len(briefs)} agents] {representative.event}\n"
            f"  Channel: {channel} | Direction: {direction} | "
            f"Typical magnitude: {most_common_mag} | Agents: {agent_ids}"
        )

    return "\n".join(lines)


class SupervisorAgent:
    """Reconciles disagreements among forecasting agents via targeted search."""

    def __init__(self, llm: LLMClient | None = None):
        self.llm = llm or LLMClient()

    async def detect_regime(
        self,
        pair: str,
        briefs: list[ResearchBrief],
    ) -> tuple[str, list[str], str]:
        """Detect the current macro regime from agent research.

        Returns:
            (regime, dominant_channels, reasoning) tuple.
            regime is one of: risk_on, risk_off, policy_divergence, carry_unwind, mixed.
        """
        causal_summary = _build_consensus_causal_summary(briefs)
        macro_excerpts = "\n\n".join(
            f"Agent {b.agent_id}: {b.macro_summary[:400]}..."
            if len(b.macro_summary) > 400 else f"Agent {b.agent_id}: {b.macro_summary}"
            for b in briefs[:3]
        )

        prompt = REGIME_DETECTION_PROMPT.format(
            pair=pair,
            causal_factors_summary=causal_summary,
            macro_excerpts=macro_excerpts or "No macro summaries available.",
        )
        response = await self.llm.complete(
            [{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=500,
        )

        # Parse
        try:
            text = response.strip()
            if text.startswith("```"):
                lines = text.split("\n")
                lines = [l for l in lines if not l.strip().startswith("```")]
                text = "\n".join(lines)
            match = re.search(r'\{[\s\S]*\}', text)
            if match:
                text = match.group()
            data = json.loads(text)
            regime = data.get("regime", "mixed").lower().strip()
            if regime not in REGIME_CHANNEL_WEIGHTS:
                regime = "mixed"
            dominant_channels = data.get("dominant_channels", [])
            reasoning = data.get("reasoning", "")
            logger.info("Regime detected: %s (channels: %s)", regime, dominant_channels)
            return regime, dominant_channels, reasoning
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.warning("Regime detection failed: %s — defaulting to mixed", e)
            return "mixed", [], f"Parse error: {e}"

    async def reconcile(
        self,
        forecasts: list[AgentForecast],
        question: ForecastQuestion,
        briefs: list[ResearchBrief] | None = None,
    ) -> SupervisorResult:
        """Analyze disagreements via causal chain comparison, search for resolution, and reconcile."""
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

        # Step 1: Identify disagreements (now channel-aware)
        disagreement_prompt = DISAGREEMENT_PROMPT.format(
            num_agents=len(forecasts),
            question=question.text,
            forecasts_summary=_format_forecasts(forecasts, briefs),
            mean_prob=mean_prob,
            min_prob=min(probs),
            max_prob=max(probs),
            spread=spread,
        )
        disagreement_response = await self.llm.complete(
            [{"role": "user", "content": disagreement_prompt}],
            temperature=0.3,
            max_tokens=2000,
        )

        # Parse disagreements and search queries
        search_queries = self._parse_disagreements(disagreement_response)
        disagreements_summary = self._format_disagreements_summary(disagreement_response)
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

        # Step 2b: Regime detection (if enabled and briefs available)
        regime_context = ""
        if settings.regime_weighting_enabled and briefs:
            try:
                regime, dominant_channels, regime_reasoning = await self.detect_regime(
                    question.pair, briefs
                )
                regime_context = (
                    f"\nCURRENT REGIME: {regime}\n"
                    f"Dominant channels: {', '.join(dominant_channels) if dominant_channels else 'N/A'}\n"
                    f"Implication: Weight agents whose causal channels align with the "
                    f"dominant regime channels more heavily.\n"
                )
            except Exception as e:
                logger.warning("Regime detection failed: %s", e)

        # Step 3: Reconcile (now with disagreement context + regime)
        evidence_text = "No additional evidence found."
        if additional_evidence:
            parts = []
            for i, e in enumerate(additional_evidence, 1):
                parts.append(f"[{i}] {e.title}\n    {e.snippet}")
            evidence_text = "\n\n".join(parts[:15])

        reconciliation_prompt = RECONCILIATION_PROMPT.format(
            question=question.text,
            forecasts_summary=_format_forecasts(forecasts, briefs),
            mean_prob=mean_prob,
            disagreements_summary=disagreements_summary + regime_context,
            additional_evidence=evidence_text,
        )
        reconciliation_response = await self.llm.complete(
            [{"role": "user", "content": reconciliation_prompt}],
            temperature=0.3,
            max_tokens=2000,
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

        causal_factors_summary = _build_consensus_causal_summary(briefs)

        # Regime detection for surface review
        regime_section = ""
        if settings.regime_weighting_enabled and briefs:
            try:
                regime, dominant_channels, regime_reasoning = await self.detect_regime(
                    pair, briefs
                )
                regime_section = (
                    f"\nCURRENT REGIME: {regime}\n"
                    f"Dominant channels: {', '.join(dominant_channels) if dominant_channels else 'N/A'}\n"
                    f"Regime reasoning: {regime_reasoning}\n"
                    f"Use this to assess whether the surface properly reflects "
                    f"the dominant regime's causal channels.\n"
                )
            except Exception as e:
                logger.warning("Regime detection for surface review failed: %s", e)

        prompt = SURFACE_REVIEW_PROMPT.format(
            pair=pair,
            spot=spot,
            cutoff_date=cutoff_date,
            grid_text=grid_text,
            num_agents=len(briefs),
            research_themes=research_themes or "No themes available.",
            causal_factors_summary=causal_factors_summary + regime_section,
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

    def _format_disagreements_summary(self, response: str) -> str:
        """Format the raw disagreement analysis into a readable summary for the reconciliation prompt."""
        try:
            text = response.strip()
            if text.startswith("```"):
                lines = text.split("\n")
                lines = [l for l in lines if not l.strip().startswith("```")]
                text = "\n".join(lines)
            data = json.loads(text)
            parts = []
            for i, d in enumerate(data.get("disagreements", []), 1):
                dtype = d.get("type", "unknown")
                desc = d.get("description", "")
                parts.append(f"[{i}] ({dtype}) {desc}")
            return "\n".join(parts) if parts else "No structured disagreements identified."
        except (json.JSONDecodeError, KeyError):
            return "Could not parse disagreements (raw analysis was used for search queries)."

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
