"""Ensemble engine: runs M agents in parallel and aggregates forecasts.

Key insights from the paper:
- 10 independent forecasts per question is standard (diminishing returns beyond ~15)
- Simple mean is a strong baseline; median and trimmed mean perform comparably
- The supervisor agent's value comes from resolving specific disagreements,
  not from holistic re-evaluation
- If supervisor has HIGH confidence → use supervisor's probability
  Otherwise → use simple mean
"""

from __future__ import annotations

import asyncio
import logging
from datetime import date

from rich.progress import Progress, SpinnerColumn, TextColumn

from aia_forecaster.agents.forecaster import ForecastingAgent
from aia_forecaster.agents.supervisor import SupervisorAgent
from aia_forecaster.config import settings
from aia_forecaster.llm.client import LLMClient
from aia_forecaster.models import (
    AgentForecast,
    BatchPricingResult,
    Confidence,
    EnsembleResult,
    ForecastQuestion,
    ResearchBrief,
    SearchMode,
    SharedResearch,
    Tenor,
)

logger = logging.getLogger(__name__)


class EnsembleEngine:
    """Orchestrates parallel forecasting agents and supervisor reconciliation."""

    def __init__(self, llm: LLMClient | None = None, num_agents: int | None = None):
        self.llm = llm or LLMClient()
        self.num_agents = num_agents or settings.num_agents

    def _create_agents(self) -> list[ForecastingAgent]:
        """Create agents with diverse search modes."""
        agents = []
        for i in range(self.num_agents):
            mode_index = i % 3
            if mode_index == 0:
                mode = SearchMode.RSS_ONLY
            elif mode_index == 1:
                mode = SearchMode.WEB_ONLY
            else:
                mode = SearchMode.HYBRID
            agents.append(ForecastingAgent(agent_id=i, llm=self.llm, search_mode=mode))
        return agents

    async def research(self, pair: str, cutoff_date: date) -> SharedResearch:
        """Phase 1: Run M agents in parallel to gather shared research.

        Each agent independently researches the broad pair outlook using the
        same agentic search loop, but without a cell-specific question.

        Returns:
            SharedResearch with all agent briefs.
        """
        agents = self._create_agents()

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            transient=True,
        ) as progress:
            task = progress.add_task(
                f"Phase 1: {self.num_agents} agents researching {pair}...", total=None
            )

            results = await asyncio.gather(
                *[agent.research(pair, cutoff_date) for agent in agents],
                return_exceptions=True,
            )

            progress.update(task, completed=True)

        briefs: list[ResearchBrief] = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error("Agent %d research failed: %s", i, result)
            else:
                briefs.append(result)

        if not briefs:
            raise RuntimeError("All research agents failed")

        logger.info(
            "Phase 1 complete: %d/%d agents produced research briefs",
            len(briefs), self.num_agents,
        )

        return SharedResearch(pair=pair, cutoff_date=cutoff_date, briefs=briefs)

    async def price_surface(
        self,
        shared_research: SharedResearch,
        strikes: list[float],
        tenors: list[Tenor],
        spot: float,
    ) -> dict[tuple[float, Tenor], dict]:
        """Phase 2: Each agent prices all cells using its own evidence.

        Batched by tenor: one LLM call per agent per tenor prices all strikes.
        → num_agents × num_tenors LLM calls total.

        Returns:
            Dict mapping (strike, tenor) → {
                "agent_probabilities": list[float],
                "mean_probability": float,
                "agent_briefs": list[ResearchBrief],
            }
        """
        briefs = shared_research.briefs
        pair = shared_research.pair
        cutoff_date = shared_research.cutoff_date

        # Create agents matching the briefs (same agent_id, search_mode)
        agents = []
        for brief in briefs:
            agents.append(
                ForecastingAgent(
                    agent_id=brief.agent_id,
                    llm=self.llm,
                    search_mode=brief.search_mode,
                )
            )

        # Run all (agent × tenor) pricing calls in parallel
        pricing_tasks = []
        task_keys = []  # Track (brief_index, tenor) for each task
        for idx, (agent, brief) in enumerate(zip(agents, briefs)):
            for tenor in tenors:
                pricing_tasks.append(
                    agent.price_tenor(pair, tenor, strikes, spot, brief, cutoff_date)
                )
                task_keys.append((idx, tenor))

        total_calls = len(pricing_tasks)
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            transient=True,
        ) as progress:
            task = progress.add_task(
                f"Phase 2: Pricing {total_calls} (agent×tenor) batches...",
                total=None,
            )

            results = await asyncio.gather(*pricing_tasks, return_exceptions=True)
            progress.update(task, completed=True)

        # Aggregate: for each (strike, tenor) collect probabilities from all agents
        cell_data: dict[tuple[float, Tenor], dict] = {}
        for strike in strikes:
            for tenor in tenors:
                cell_data[(strike, tenor)] = {
                    "agent_probabilities": [],
                    "mean_probability": 0.5,
                    "agent_briefs": briefs,
                }

        succeeded = 0
        for (idx, tenor), result in zip(task_keys, results):
            if isinstance(result, Exception):
                logger.error(
                    "Agent %d pricing failed for %s: %s",
                    briefs[idx].agent_id, tenor.value, result,
                )
                continue
            succeeded += 1
            pricing: BatchPricingResult = result
            for strike in strikes:
                key = f"{strike:.2f}"
                p = pricing.probabilities.get(key, 0.5)
                cell_data[(strike, tenor)]["agent_probabilities"].append(p)

        logger.info(
            "Phase 2 complete: %d/%d pricing calls succeeded", succeeded, total_calls
        )

        # Compute means
        for key, data in cell_data.items():
            probs = data["agent_probabilities"]
            if probs:
                data["mean_probability"] = sum(probs) / len(probs)
            else:
                data["mean_probability"] = 0.5

        return cell_data

    async def run(self, question: ForecastQuestion) -> EnsembleResult:
        """Run the full ensemble: parallel agents → aggregation → supervisor.

        Returns:
            EnsembleResult with individual forecasts, mean, supervisor result,
            and final probability.
        """
        # Stage 1: Run M agents in parallel with diverse search modes
        agents = self._create_agents()

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            transient=True,
        ) as progress:
            task = progress.add_task(
                f"Running {self.num_agents} forecasting agents...", total=None
            )

            results = await asyncio.gather(
                *[agent.forecast(question) for agent in agents],
                return_exceptions=True,
            )

            progress.update(task, completed=True)

        # Filter out failures
        forecasts: list[AgentForecast] = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error("Agent %d failed: %s", i, result)
            else:
                forecasts.append(result)

        if not forecasts:
            raise RuntimeError("All forecasting agents failed")

        # Compute simple mean
        probs = [f.probability for f in forecasts]
        mean_prob = sum(probs) / len(probs)

        # Log per-mode breakdown
        by_mode: dict[str, list[float]] = {}
        for f in forecasts:
            by_mode.setdefault(f.search_mode.value, []).append(f.probability)
        mode_summary = ", ".join(
            f"{m}={sum(ps)/len(ps):.4f}(n={len(ps)})" for m, ps in sorted(by_mode.items())
        )
        logger.info(
            "Ensemble: %d/%d agents succeeded. Mean=%.4f, Min=%.4f, Max=%.4f | %s",
            len(forecasts), self.num_agents, mean_prob, min(probs), max(probs), mode_summary,
        )

        # Stage 2: Supervisor reconciliation
        supervisor = SupervisorAgent(llm=self.llm)
        try:
            supervisor_result = await supervisor.reconcile(forecasts, question)
            logger.info(
                "Supervisor: confidence=%s, reconciled_p=%s",
                supervisor_result.confidence,
                supervisor_result.reconciled_probability,
            )
        except Exception as e:
            logger.error("Supervisor failed: %s", e)
            supervisor_result = None

        # Determine final probability
        final_prob = mean_prob
        if (
            supervisor_result
            and supervisor_result.confidence == Confidence.HIGH
            and supervisor_result.reconciled_probability is not None
        ):
            final_prob = supervisor_result.reconciled_probability
            logger.info("Using supervisor's high-confidence estimate: %.4f", final_prob)
        else:
            logger.info("Using ensemble mean: %.4f", final_prob)

        return EnsembleResult(
            agent_forecasts=forecasts,
            mean_probability=mean_prob,
            supervisor=supervisor_result,
            final_probability=final_prob,
        )
