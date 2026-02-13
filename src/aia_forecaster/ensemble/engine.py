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

from rich.progress import Progress, SpinnerColumn, TextColumn

from aia_forecaster.agents.forecaster import ForecastingAgent
from aia_forecaster.agents.supervisor import SupervisorAgent
from aia_forecaster.config import settings
from aia_forecaster.llm.client import LLMClient
from aia_forecaster.models import (
    AgentForecast,
    Confidence,
    EnsembleResult,
    ForecastQuestion,
)

logger = logging.getLogger(__name__)


class EnsembleEngine:
    """Orchestrates parallel forecasting agents and supervisor reconciliation."""

    def __init__(self, llm: LLMClient | None = None, num_agents: int | None = None):
        self.llm = llm or LLMClient()
        self.num_agents = num_agents or settings.num_agents

    async def run(self, question: ForecastQuestion) -> EnsembleResult:
        """Run the full ensemble: parallel agents → aggregation → supervisor.

        Returns:
            EnsembleResult with individual forecasts, mean, supervisor result,
            and final probability.
        """
        # Stage 1: Run M agents in parallel
        agents = [ForecastingAgent(agent_id=i, llm=self.llm) for i in range(self.num_agents)]

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
        logger.info(
            "Ensemble: %d/%d agents succeeded. Mean=%.4f, Min=%.4f, Max=%.4f",
            len(forecasts), self.num_agents, mean_prob, min(probs), max(probs),
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
