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
from datetime import date

from aia_forecaster.config import settings
from aia_forecaster.fx.base_rates import format_base_rate_context
from aia_forecaster.llm.client import LLMClient
from aia_forecaster.models import (
    AgentForecast,
    BatchPricingResult,
    CausalFactor,
    ForecastMode,
    ForecastQuestion,
    ResearchBrief,
    SearchMode,
    SearchResult,
    Tenor,
)
from aia_forecaster.search.registry import fetch_all as fetch_all_sources
from aia_forecaster.search.web import search_web

logger = logging.getLogger(__name__)

QUERY_GENERATION_PROMPT = """\
You are an FX research analyst. You are investigating the following binary forecasting question:

QUESTION: {question}

Your information cutoff date is {cutoff_date}. Do NOT use any information after this date.

{base_rate_section}
{evidence_section}

Based on what you know and any evidence gathered so far, generate the SINGLE BEST search \
query to find information that would help you estimate the probability of this event.

Prioritize FX-specific data sources:
- Central bank forward guidance, rate decisions, and dot plots (Fed, ECB, BOJ, BOE, etc.)
- Economic calendar events: NFP, CPI, GDP, PMI, employment, retail sales
- CFTC Commitments of Traders (COT) positioning data and risk reversals
- Key technical levels: support/resistance, round-number barriers, 200-day MA
- Trade balance and capital flow data (current account, portfolio flows)
- Cross-pair correlations and risk sentiment indicators (VIX, DXY, yield spreads)
- Geopolitical risk events affecting safe-haven flows (JPY, CHF, USD)

IMPORTANT query formatting rules:
- Keep the query SHORT (under 150 characters, ideally 5-10 words)
- Use plain keywords only — NO boolean operators (AND/OR), NO parentheses, NO quotation marks
- Do NOT use site: filters or before:/after: date filters
- Focus on ONE specific topic per query

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

# --- Phase 1: Pair-level research prompts ---

RESEARCH_QUERY_PROMPT = """\
You are an FX research analyst investigating the outlook for {pair}.

Your information cutoff date is {cutoff_date}. Do NOT use any information after this date.

{evidence_section}

Generate the SINGLE BEST search query to find information about the current and near-term \
outlook for {base}/{quote} over the next 1 day to 6 months.

Prioritize FX-specific data sources:
- Central bank forward guidance, rate decisions, and dot plots for both currencies
- Economic calendar events: NFP, CPI, GDP, PMI, employment, retail sales
- CFTC Commitments of Traders (COT) positioning data and risk reversals
- Key technical levels: support/resistance, round-number barriers, 200-day MA
- Trade balance and capital flow data (current account, portfolio flows)
- Cross-pair correlations and risk sentiment indicators (VIX, DXY, yield spreads)
- Geopolitical risk events affecting safe-haven flows (JPY, CHF, USD)

IMPORTANT query formatting rules:
- Keep the query SHORT (under 150 characters, ideally 5-10 words)
- Use plain keywords only — NO boolean operators (AND/OR), NO parentheses, NO quotation marks
- Do NOT use site: filters or before:/after: date filters
- Focus on ONE specific topic per query

Respond with ONLY the search query string, nothing else."""

RESEARCH_ASSESS_PROMPT = """\
You are an FX research analyst investigating the outlook for {pair}.

You have performed {iteration} search iteration(s) so far and gathered the following evidence:

{evidence_summary}

Do you have SUFFICIENT evidence to form a well-informed view on {pair}'s likely direction? \
Consider whether you have:
- Relevant macroeconomic data and central bank policy signals
- Current market conditions and recent price action context
- Geopolitical factors that could affect this currency pair
- Enough diverse sources to form a balanced view

Respond with EXACTLY one word: SEARCH (if you need more evidence) or DONE (if ready)."""

RESEARCH_SUMMARY_PROMPT = """\
You are an FX research analyst. Summarize your research on {pair} into a concise macro brief.

EVIDENCE GATHERED:
{evidence_summary}

Produce a summary covering:
1. Key macro themes affecting {pair}
2. Central bank policy direction for both currencies
3. Near-term risks and catalysts
4. Overall directional bias (bullish/bearish/neutral on {base} vs {quote})

CAUSAL ANALYSIS (REQUIRED):
For each material factor you identified, state the causal chain explicitly:
- Event/Condition: What is happening or expected to happen
- Channel: How it transmits to {pair} (e.g., rate differential, risk appetite, trade flows, safe-haven demand, carry trade, portfolio rebalancing)
- Direction: Does this make {base} stronger (bullish on pair) or weaker (bearish on pair)?
- Magnitude: strong / moderate / weak
- Confidence: high / medium / low (how sure are you this factor is active?)

Be specific about the transmission channel — two analysts can agree on the event but disagree \
on which channel dominates, producing opposite forecasts. Making the channel explicit is the \
whole point.

You MUST include ALL three fields in your JSON response. Do NOT omit causal_factors.

Respond in this EXACT JSON format:
{{
  "causal_factors": [
    {{
      "event": "BOJ signals rate hike in March",
      "channel": "rate differential narrowing",
      "direction": "bearish",
      "magnitude": "strong",
      "confidence": "high"
    }},
    {{
      "event": "Fed holds rates, pushes back on cuts",
      "channel": "rate differential widening",
      "direction": "bullish",
      "magnitude": "moderate",
      "confidence": "high"
    }}
  ],
  "key_themes": ["theme1", "theme2", "theme3"],
  "macro_summary": "2-3 paragraph summary of your analysis"
}}"""

# --- Phase 2: Batch pricing prompt ---

BATCH_PRICING_PROMPT = """\
You are an expert FX forecasting analyst. Given your research on {pair}, estimate the \
probability that {base}/{quote} will be ABOVE each strike price at the {tenor_label} horizon.

Current spot: {spot}
Tenor: {tenor_label}

YOUR EVIDENCE:
{evidence_summary}

YOUR MACRO ANALYSIS:
{macro_summary}

YOUR CAUSAL FACTORS:
{causal_factors_block}

BASE RATES (statistical anchors):
{base_rates_block}

INSTRUCTIONS:
1. Start from the base rates as statistical anchors.
2. For EACH causal factor above, assess whether it is relevant at THIS tenor ({tenor_label}). \
Some factors act fast (positioning, sentiment → days/weeks) while others act slowly \
(trade flows, policy divergence → months). Weight accordingly.
3. State how each active factor shifts your probability distribution (up/down, by roughly how much).
4. Probabilities MUST be non-increasing as strike increases (higher price = less likely to be above).
5. Do NOT hedge toward 0.5 — if evidence points in one direction, commit to it.

Respond in this EXACT JSON format (no extra fields):
{{
  "reasoning": "Brief explanation of your probability distribution",
  "probabilities": {{{strike_keys}}}
}}"""

BATCH_PRICING_PROMPT_HITTING = """\
You are an expert FX forecasting analyst. Given your research on {pair}, estimate the \
probability that {base}/{quote} will TOUCH or REACH each barrier price at any point within \
the {tenor_label} horizon.

This is a BARRIER/HITTING probability — "Will the price touch this level at any point \
during the period?", NOT "Will it be above this level at the end?"

Key properties of barrier probabilities:
- P(touch) is highest (~1.0) for barriers near the current spot price
- P(touch) DECREASES with distance from spot, in BOTH directions (above and below)
- P(touch) >= P(above) always — touching a level is easier than finishing above it
- Longer tenors always give higher P(touch) — more time means more chance to reach a level

Current spot: {spot}
Tenor: {tenor_label}

YOUR EVIDENCE:
{evidence_summary}

YOUR MACRO ANALYSIS:
{macro_summary}

YOUR CAUSAL FACTORS:
{causal_factors_block}

BASE RATES (statistical anchors):
{base_rates_block}

INSTRUCTIONS:
1. Start from the base rates as statistical anchors.
2. For EACH causal factor above, assess whether it is relevant at THIS tenor ({tenor_label}). \
Some factors act fast (positioning, sentiment → days/weeks) while others act slowly \
(trade flows, policy divergence → months). Weight accordingly.
3. State how each active factor shifts your probability distribution (up/down, by roughly how much).
4. Probabilities MUST decrease with distance from current spot (both above and below).
5. Do NOT hedge toward 0.5 — if evidence points in one direction, commit to it. \
Barriers near spot should have probabilities near 1.0.

Respond in this EXACT JSON format (no extra fields):
{{
  "reasoning": "Brief explanation of your probability distribution",
  "probabilities": {{{strike_keys}}}
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


def _format_causal_factors(factors: list[CausalFactor]) -> str:
    """Format causal factors into a readable block for inclusion in prompts."""
    if not factors:
        return "No causal factors identified yet."
    lines = []
    for i, f in enumerate(factors, 1):
        lines.append(
            f"[{i}] {f.event}\n"
            f"    Channel: {f.channel}\n"
            f"    Direction: {f.direction} | Magnitude: {f.magnitude} | Confidence: {f.confidence}"
        )
    return "\n".join(lines)


def _parse_causal_factors(raw: list[dict]) -> list[CausalFactor]:
    """Parse raw dicts from LLM JSON into validated CausalFactor objects."""
    factors: list[CausalFactor] = []
    for item in raw:
        try:
            factors.append(CausalFactor(
                event=str(item.get("event", "")),
                channel=str(item.get("channel", "")),
                direction=str(item.get("direction", "")).lower(),
                magnitude=str(item.get("magnitude", "moderate")).lower(),
                confidence=str(item.get("confidence", "medium")).lower(),
            ))
        except Exception:
            logger.warning("Skipping unparseable causal factor: %s", item)
    return factors


class ForecastingAgent:
    """A single forecasting agent that performs agentic search and produces a probability."""

    def __init__(
        self,
        agent_id: int,
        llm: LLMClient | None = None,
        search_mode: SearchMode = SearchMode.HYBRID,
        temperature: float | None = None,
        max_search_iterations: int | None = None,
    ):
        self.agent_id = agent_id
        self.llm = llm or LLMClient()
        self.search_mode = search_mode
        self.temperature = temperature if temperature is not None else 0.7
        self.max_search_iterations = (
            max_search_iterations
            if max_search_iterations is not None
            else settings.max_search_iterations
        )

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

        Search mode controls which information sources the agent uses:
        - HYBRID (default): RSS background + iterative web search
        - RSS_ONLY: Only RSS feeds, no web search — faster, curated sources
        - WEB_ONLY: Only web search, no RSS — broader, unfiltered evidence
        """
        all_evidence: list[SearchResult] = []
        all_queries: list[str] = []
        base_rate_section = self._build_base_rate_section(question)

        logger.info("Agent %d: search_mode=%s", self.agent_id, self.search_mode.value)

        # Passive data source gathering (RSS_ONLY and HYBRID modes)
        if self.search_mode in (SearchMode.RSS_ONLY, SearchMode.HYBRID):
            max_results = 20 if self.search_mode == SearchMode.RSS_ONLY else 10
            try:
                source_results = await fetch_all_sources(
                    pair=question.pair,
                    cutoff_date=question.cutoff_date,
                    max_age_hours=72 if self.search_mode == SearchMode.RSS_ONLY else 48,
                    max_results=max_results,
                )
                for source_name, results in source_results.items():
                    all_evidence.extend(results)
                    logger.info("Agent %d: Got %d items from '%s'", self.agent_id, len(results), source_name)
            except Exception:
                logger.warning("Agent %d: Data source fetch failed", self.agent_id)

        # Web search loop (WEB_ONLY and HYBRID modes)
        if self.search_mode in (SearchMode.WEB_ONLY, SearchMode.HYBRID):
            for iteration in range(1, self.max_search_iterations + 1):
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
                    temperature=self.temperature,
                    max_tokens=200,
                )
                search_query = search_query.strip().strip('"')

                # Skip empty or trivially short queries
                if not search_query or len(search_query) < 5:
                    logger.warning("Agent %d, iter %d: Empty/short query — skipping", self.agent_id, iteration)
                    all_queries.append(search_query)
                    continue

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
                if iteration < self.max_search_iterations:
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
            search_mode=self.search_mode,
        )

    async def research(self, pair: str, cutoff_date: date) -> ResearchBrief:
        """Phase 1: Perform pair-level research (no cell-specific question).

        Reuses the same agentic search loop but with a broad pair-level question
        instead of a cell-specific one. Returns a ResearchBrief with evidence
        and macro summary.
        """
        all_evidence: list[SearchResult] = []
        all_queries: list[str] = []
        base, quote = pair[:3], pair[3:]

        logger.info("Agent %d: research mode, pair=%s, search_mode=%s",
                     self.agent_id, pair, self.search_mode.value)

        # Passive data source gathering (RSS_ONLY and HYBRID modes)
        if self.search_mode in (SearchMode.RSS_ONLY, SearchMode.HYBRID):
            max_results = 20 if self.search_mode == SearchMode.RSS_ONLY else 10
            try:
                source_results = await fetch_all_sources(
                    pair=pair,
                    cutoff_date=cutoff_date,
                    max_age_hours=72 if self.search_mode == SearchMode.RSS_ONLY else 48,
                    max_results=max_results,
                )
                for source_name, results in source_results.items():
                    all_evidence.extend(results)
                    logger.info("Agent %d: Got %d items from '%s'", self.agent_id, len(results), source_name)
            except Exception:
                logger.warning("Agent %d: Data source fetch failed", self.agent_id)

        # Web search loop (WEB_ONLY and HYBRID modes)
        if self.search_mode in (SearchMode.WEB_ONLY, SearchMode.HYBRID):
            for iteration in range(1, self.max_search_iterations + 1):
                evidence_section = ""
                if all_evidence:
                    evidence_section = (
                        f"Evidence gathered so far:\n{_format_evidence(all_evidence)}"
                    )

                query_prompt = RESEARCH_QUERY_PROMPT.format(
                    pair=pair,
                    base=base,
                    quote=quote,
                    cutoff_date=cutoff_date.isoformat(),
                    evidence_section=evidence_section,
                )
                search_query = await self.llm.complete(
                    [{"role": "user", "content": query_prompt}],
                    temperature=self.temperature,
                    max_tokens=200,
                )
                search_query = search_query.strip().strip('"')

                # Skip empty or trivially short queries
                if not search_query or len(search_query) < 5:
                    logger.warning("Agent %d, iter %d: Empty/short query — skipping",
                                   self.agent_id, iteration)
                    all_queries.append(search_query)
                    continue

                all_queries.append(search_query)
                logger.info("Agent %d, iter %d: Searching '%s'",
                            self.agent_id, iteration, search_query)

                try:
                    results = await search_web(
                        query=search_query,
                        max_results=5,
                        cutoff_date=cutoff_date,
                    )
                    all_evidence.extend(results)
                except Exception:
                    logger.warning("Agent %d: Search failed for query '%s'",
                                   self.agent_id, search_query)

                if iteration < self.max_search_iterations:
                    assess_prompt = RESEARCH_ASSESS_PROMPT.format(
                        pair=pair,
                        iteration=iteration,
                        evidence_summary=_format_evidence(all_evidence),
                    )
                    decision = await self.llm.complete(
                        [{"role": "user", "content": assess_prompt}],
                        temperature=0.3,
                        max_tokens=10,
                    )
                    decision = decision.strip().upper()
                    if "DONE" in decision:
                        logger.info("Agent %d: Research complete after %d iterations",
                                    self.agent_id, iteration)
                        break

        # Generate macro summary
        key_themes: list[str] = []
        causal_factors: list[CausalFactor] = []
        macro_summary = ""
        if all_evidence:
            try:
                summary_prompt = RESEARCH_SUMMARY_PROMPT.format(
                    pair=pair,
                    base=base,
                    quote=quote,
                    evidence_summary=_format_evidence(all_evidence, max_chars=8000),
                )
                summary_response = await self.llm.complete(
                    [{"role": "user", "content": summary_prompt}],
                    temperature=0.3,
                    max_tokens=2500,
                )
                summary_data = self._parse_json_response(summary_response)
                if not summary_data:
                    logger.warning(
                        "Agent %d: summary JSON parse returned empty — "
                        "response length=%d",
                        self.agent_id, len(summary_response),
                    )
                key_themes = summary_data.get("key_themes", [])
                macro_summary = summary_data.get("macro_summary", "")
                raw_causal = summary_data.get("causal_factors", [])
                causal_factors = _parse_causal_factors(raw_causal)
                if causal_factors:
                    logger.info(
                        "Agent %d: extracted %d causal factors",
                        self.agent_id, len(causal_factors),
                    )
            except Exception:
                logger.warning("Agent %d: Failed to generate macro summary", self.agent_id)
                macro_summary = _format_evidence(all_evidence, max_chars=3000)

        return ResearchBrief(
            agent_id=self.agent_id,
            key_themes=key_themes,
            causal_factors=causal_factors,
            evidence=all_evidence,
            search_queries=all_queries,
            search_mode=self.search_mode,
            macro_summary=macro_summary,
            iterations=len(all_queries),
        )

    async def price_tenor(
        self,
        pair: str,
        tenor: Tenor,
        strikes: list[float],
        spot: float,
        brief: ResearchBrief,
        cutoff_date: date,
        forecast_mode: ForecastMode = ForecastMode.ABOVE,
    ) -> BatchPricingResult:
        """Phase 2: Price all strikes for a single tenor using pre-gathered evidence.

        One LLM call produces probabilities for all strikes at the given tenor,
        ensuring cross-strike coherence.
        """
        base, quote = pair[:3], pair[3:]
        tenor_map = {
            Tenor.D1: "1 day", Tenor.W1: "1 week", Tenor.M1: "1 month",
            Tenor.M3: "3 months", Tenor.M6: "6 months",
        }
        tenor_label = tenor_map.get(tenor, tenor.value)

        # Build base rates block
        base_rates_lines = []
        for strike in strikes:
            try:
                ctx = format_base_rate_context(
                    pair=pair, spot=spot, strike=strike, tenor=tenor,
                    forecast_mode=forecast_mode,
                )
                # Extract just the base rate line
                for line in ctx.split("\n"):
                    if "Statistical base rate" in line:
                        base_rates_lines.append(f"  Strike {strike}: {line.strip()}")
                        break
                else:
                    base_rates_lines.append(f"  Strike {strike}: (base rate context available)")
            except ValueError:
                base_rates_lines.append(f"  Strike {strike}: (no base rate data)")
        base_rates_block = "\n".join(base_rates_lines)

        # Build strike keys for JSON template
        strike_keys = ", ".join(f'"{s:.2f}": 0.XX' for s in strikes)

        # Format causal factors from research phase
        causal_factors_block = _format_causal_factors(brief.causal_factors)

        # Select prompt based on forecast mode
        prompt_template = (
            BATCH_PRICING_PROMPT_HITTING
            if forecast_mode == ForecastMode.HITTING
            else BATCH_PRICING_PROMPT
        )

        prompt = prompt_template.format(
            pair=pair,
            base=base,
            quote=quote,
            spot=spot,
            tenor_label=tenor_label,
            evidence_summary=_format_evidence(brief.evidence, max_chars=6000),
            macro_summary=brief.macro_summary or "No macro summary available.",
            causal_factors_block=causal_factors_block,
            base_rates_block=base_rates_block,
            strike_keys=strike_keys,
        )

        response = await self.llm.complete(
            [{"role": "user", "content": prompt}],
            temperature=0.5,
            max_tokens=4096,
        )

        # Parse response
        data = self._parse_json_response(response)
        raw_probs = data.get("probabilities", {})
        reasoning = data.get("reasoning", "")
        # Causal factors come from the research phase (brief), not the pricing response
        causal_factors = brief.causal_factors

        # Normalize keys and clamp values
        probabilities: dict[str, float] = {}
        for strike in strikes:
            key = f"{strike:.2f}"
            p = raw_probs.get(key, raw_probs.get(str(strike), 0.5))
            try:
                probabilities[key] = max(0.0, min(1.0, float(p)))
            except (ValueError, TypeError):
                probabilities[key] = 0.5

        return BatchPricingResult(
            agent_id=self.agent_id,
            tenor=tenor,
            probabilities=probabilities,
            reasoning=reasoning,
            causal_factors=causal_factors,
        )

    def _parse_json_response(self, response: str) -> dict:
        """Parse a JSON response, handling markdown code blocks."""
        text = response.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            text = "\n".join(lines)
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # Try to find JSON object in the response
            match = re.search(r'\{[\s\S]*\}', text)
            if match:
                try:
                    return json.loads(match.group())
                except json.JSONDecodeError:
                    pass
            logger.warning(
                "JSON parse failed — raw response (first 500 chars): %s",
                response[:500],
            )
            return {}

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
