"""Probability surface generation over (strike, tenor) grid."""

from __future__ import annotations

import html as html_mod
import logging
import textwrap
from datetime import date
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from rich.console import Console
from rich.table import Table

from aia_forecaster.calibration.monotonicity import (
    enforce_hitting_monotonicity,
    enforce_raw_surface_monotonicity,
    enforce_surface_monotonicity,
)
from aia_forecaster.calibration.platt import calibrate
from aia_forecaster.ensemble.engine import EnsembleEngine
from aia_forecaster.fx.pairs import DEFAULT_TENORS, generate_strikes
from aia_forecaster.fx.rates import get_spot_rate
from aia_forecaster.llm.client import LLMClient
from aia_forecaster.models import (
    AgentForecast,
    CausalFactor,
    EvidenceItem,
    EnsembleResult,
    ForecastMode,
    ProbabilitySurface,
    ResearchBrief,
    SearchResult,
    SourceConfig,
    SurfaceCell,
    Tenor,
    TenorResearchBrief,
)

logger = logging.getLogger(__name__)
console = Console()


def _aggregate_factors(
    factors_with_agents: list[tuple[int, CausalFactor]],
    total_agents: int,
) -> list[CausalFactor]:
    """Shared aggregation: group by channel, majority-vote direction, return consensus.

    Used by both pair-level and tenor-level aggregation.

    Args:
        factors_with_agents: List of (agent_id, CausalFactor) tuples.
        total_agents: Total number of agents contributing (for annotation).

    Returns:
        Consensus factors sorted by citation count descending.
    """
    if not factors_with_agents:
        return []

    # Group by channel only — same channel with opposite directions gets merged
    groups: dict[str, list[tuple[int, CausalFactor]]] = {}
    for agent_id, f in factors_with_agents:
        key = f.channel.lower().strip()
        groups.setdefault(key, []).append((agent_id, f))

    # Build consensus factors, sorted by total citation count.
    agents_with_factors = len({aid for aid, _ in factors_with_agents})
    min_count = 2 if agents_with_factors >= 3 else 1
    consensus: list[CausalFactor] = []
    for _channel, entries in sorted(groups.items(), key=lambda x: -len(x[1])):
        if len(entries) < min_count:
            continue

        agent_count = len({aid for aid, _ in entries})

        # Split by direction and reconcile
        bullish = [(aid, f) for aid, f in entries if f.direction.lower().strip() == "bullish"]
        bearish = [(aid, f) for aid, f in entries if f.direction.lower().strip() == "bearish"]
        n_bull = len({aid for aid, _ in bullish})
        n_bear = len({aid for aid, _ in bearish})

        if n_bull > n_bear:
            net_direction = "bullish"
            representative = bullish[0][1]
        elif n_bear > n_bull:
            net_direction = "bearish"
            representative = bearish[0][1]
        else:
            # Tie — pick whichever has higher confidence, default bearish
            net_direction = "contested"
            representative = entries[0][1]

        # Annotation for the event text
        if n_bull > 0 and n_bear > 0:
            event_prefix = f"[{agent_count}/{total_agents} agents, {n_bull} bullish vs {n_bear} bearish]"
        else:
            event_prefix = f"[{agent_count}/{total_agents} agents]"

        magnitudes = [e.magnitude for _, e in entries]
        confidences = [e.confidence for _, e in entries]
        consensus.append(CausalFactor(
            event=f"{event_prefix} {representative.event}",
            channel=representative.channel,
            direction=net_direction,
            magnitude=max(set(magnitudes), key=magnitudes.count),
            confidence=max(set(confidences), key=confidences.count),
        ))

    return consensus


def _build_consensus_factors(briefs: list[ResearchBrief]) -> list[CausalFactor]:
    """Aggregate causal factors across all agent briefs into consensus factors.

    Groups by channel (regardless of direction), reconciles conflicting
    bullish/bearish assessments via majority vote, and only returns factors
    cited by 2+ agents (or all if fewer than 2 agents).
    """
    if not briefs:
        return []

    all_factors: list[tuple[int, CausalFactor]] = []
    for brief in briefs:
        for f in brief.causal_factors:
            all_factors.append((brief.agent_id, f))

    return _aggregate_factors(all_factors, total_agents=len(briefs))


def _build_tenor_consensus_factors(
    tenor_briefs: list[TenorResearchBrief],
) -> list[CausalFactor]:
    """Aggregate tenor-specific causal factors across agents for a single tenor.

    Same logic as pair-level aggregation but operating on TenorResearchBrief
    causal_factors instead of ResearchBrief causal_factors.
    """
    if not tenor_briefs:
        return []

    all_factors: list[tuple[int, CausalFactor]] = []
    for tb in tenor_briefs:
        for f in tb.causal_factors:
            all_factors.append((tb.agent_id, f))

    return _aggregate_factors(all_factors, total_agents=len(tenor_briefs))


def _question_text(
    pair: str,
    strike: float,
    tenor: Tenor,
    forecast_mode: ForecastMode = ForecastMode.ABOVE,
) -> str:
    """Formulate the binary question for a (strike, tenor) cell."""
    base, quote = pair[:3], pair[3:]
    horizon = tenor.label
    if forecast_mode == ForecastMode.HITTING:
        return f"Will {base}/{quote} touch/reach {strike} within {horizon}?"
    return f"Will {base}/{quote} be above {strike} in {horizon}?"


class ProbabilitySurfaceGenerator:
    """Generates the full probability surface for a currency pair.

    Uses a two-phase approach to avoid redundant search:
      Phase 1: Shared research — agents gather evidence for the pair (not per-cell)
      Phase 2: Batch pricing — each agent prices all cells using its own evidence
    This reduces LLM calls from ~3,000 to ~170 (a ~94% reduction).
    """

    def __init__(
        self,
        llm: LLMClient | None = None,
        num_agents: int | None = None,
        source_config: SourceConfig | None = None,
    ):
        self.llm = llm or LLMClient()
        self.source_config = source_config
        self.engine = EnsembleEngine(
            llm=self.llm, num_agents=num_agents, source_config=source_config,
        )

    async def generate(
        self,
        pair: str = "USDJPY",
        num_strikes: int = 5,
        tenors: list[Tenor] | None = None,
        cutoff_date: date | None = None,
        forecast_mode: ForecastMode = ForecastMode.HITTING,
        strike_step: float | None = None,
        custom_strikes: list[float] | None = None,
    ) -> ProbabilitySurface:
        """Generate the probability surface using two-phase shared research.

        Phase 1: Shared research — M agents independently research the pair outlook
        Phase 2: Batch pricing — each agent prices all (strike, tenor) cells
        Phase 3: Surface-level supervisor review + PAVA monotonicity + Platt calibration

        Args:
            pair: Currency pair.
            num_strikes: Number of strikes around spot.
            tenors: List of tenors to evaluate. Default: all.
            cutoff_date: Information cutoff date.
            forecast_mode: 'hitting' (barrier touch) or 'above' (terminal distribution).
            strike_step: Override the default interval between auto-generated strikes.
            custom_strikes: Explicit list of strike prices (bypasses auto-generation).

        Returns:
            ProbabilitySurface with calibrated probabilities for each cell.
        """
        if tenors is None:
            tenors = DEFAULT_TENORS
        if cutoff_date is None:
            cutoff_date = date.today()

        # Get spot rate and generate strikes
        spot = await get_spot_rate(pair)
        if custom_strikes is not None:
            strikes = sorted(custom_strikes)
        else:
            strikes = generate_strikes(
                spot, pair, num_strikes,
                forecast_mode=forecast_mode, step=strike_step,
            )
        console.print(f"[bold]Spot rate:[/bold] {pair} = {spot}")
        console.print(f"[bold]Strikes:[/bold] {strikes}")
        console.print(f"[bold]Tenors:[/bold] {[t.value for t in tenors]}")

        total_cells = len(strikes) * len(tenors)
        num_agents = self.engine.num_agents
        from aia_forecaster.config import settings as app_settings
        tenor_research_calls = (
            num_agents * len(tenors) * 3  # ~3 calls per (agent, tenor): query + search + summary
            if app_settings.tenor_research_enabled
            else 0
        )
        estimated_calls = (
            num_agents * 7  # ~7 LLM calls per agent in research (queries + assess + summary)
            + tenor_research_calls  # Phase 1.5 tenor research
            + num_agents * len(tenors)  # pricing calls
            + 1  # supervisor surface review
        )
        console.print(
            f"\n[bold]Generating {total_cells} cell surface "
            f"(~{estimated_calls} LLM calls via shared research)...[/bold]\n"
        )

        # --- Phase 1: Shared Research ---
        console.print("[bold]Phase 1: Shared Research[/bold]")
        shared_research = await self.engine.research(pair, cutoff_date)
        console.print(
            f"  [green]{len(shared_research.briefs)} agents produced research briefs[/green]"
        )
        for brief in shared_research.briefs:
            themes = ", ".join(brief.key_themes[:3]) if brief.key_themes else "—"
            console.print(
                f"  [dim]Agent {brief.agent_id} ({brief.search_mode.value}): "
                f"{brief.iterations} searches, {len(brief.evidence)} evidence items, "
                f"themes: {themes}[/dim]"
            )

        # Aggregate consensus causal factors from all briefs
        consensus_factors = _build_consensus_factors(shared_research.briefs)
        if consensus_factors:
            console.print(f"\n  [bold]Consensus causal factors ({len(consensus_factors)}):[/bold]")
            for cf in consensus_factors:
                icon = "[green]+[/green]" if cf.direction == "bullish" else "[red]-[/red]"
                console.print(
                    f"    {icon} {cf.event}\n"
                    f"      [dim]{cf.channel} | {cf.direction} | "
                    f"magnitude: {cf.magnitude} | confidence: {cf.confidence}[/dim]"
                )

        # --- Phase 1.5: Tenor-Specific Research ---
        if app_settings.tenor_research_enabled:
            console.print("\n[bold]Phase 1.5: Tenor-Specific Research[/bold]")
            tenor_briefs = await self.engine.research_tenors(shared_research, tenors)
            shared_research.tenor_briefs = tenor_briefs
            total_tenor_evidence = sum(
                sum(len(tb.evidence) for tb in tbs)
                for tbs in tenor_briefs.values()
            )
            console.print(
                f"  [green]{sum(len(tbs) for tbs in tenor_briefs.values())} "
                f"tenor briefs produced, {total_tenor_evidence} new evidence items[/green]"
            )
            for tenor_str, tbs in sorted(tenor_briefs.items()):
                if tbs:
                    ev_count = sum(len(tb.evidence) for tb in tbs)
                    cf_count = sum(len(tb.causal_factors) for tb in tbs)
                    console.print(
                        f"  [dim]{tenor_str}: {len(tbs)} agents, "
                        f"{ev_count} evidence items, {cf_count} causal factors[/dim]"
                    )

        # --- Phase 2: Batch Pricing ---
        console.print("\n[bold]Phase 2: Batch Pricing[/bold]")
        cell_data = await self.engine.price_surface(
            shared_research, strikes, tenors, spot,
            forecast_mode=forecast_mode,
        )

        # Build surface cells with synthetic EnsembleResult per cell
        surface = ProbabilitySurface(
            pair=pair,
            spot_rate=spot,
            forecast_mode=forecast_mode,
            causal_factors=consensus_factors,
            source_config=self.source_config,
        )
        cell_probabilities: dict[tuple[float, Tenor], float] = {}

        for tenor in tenors:
            for strike in strikes:
                q_text = _question_text(pair, strike, tenor, forecast_mode)
                data = cell_data.get((strike, tenor), {})
                agent_probs = data.get("agent_probabilities", [])
                mean_prob = data.get("mean_probability", 0.5)

                # Build synthetic AgentForecast objects for explanation compatibility
                agent_forecasts = []
                briefs = data.get("agent_briefs", shared_research.briefs)
                cell_tenor_briefs: list[TenorResearchBrief] = data.get("tenor_briefs", [])
                # Build agent_id → TenorResearchBrief lookup for this tenor
                tb_by_agent: dict[int, TenorResearchBrief] = {
                    tb.agent_id: tb for tb in cell_tenor_briefs
                }
                for idx, p in enumerate(agent_probs):
                    brief = briefs[idx] if idx < len(briefs) else None
                    agent_id = brief.agent_id if brief else idx
                    tb = tb_by_agent.get(agent_id)

                    # Merge evidence: tenor-specific FIRST (so it ranks higher
                    # in dedup), then pair-level (dedup by URL)
                    merged_evidence: list[SearchResult] = []
                    seen_urls: set[str] = set()
                    if tb and tb.evidence:
                        for e in tb.evidence:
                            key = e.url.rstrip("/").lower()
                            if key not in seen_urls:
                                merged_evidence.append(e)
                                seen_urls.add(key)
                    if brief:
                        for e in brief.evidence:
                            key = e.url.rstrip("/").lower()
                            if key not in seen_urls:
                                merged_evidence.append(e)
                                seen_urls.add(key)

                    # Build tenor-enriched reasoning so consensus differs per tenor
                    reasoning = brief.macro_summary if brief else ""
                    if tb and tb.relevance_summary:
                        reasoning = f"{reasoning}\n\n[{tenor.label}] {tb.relevance_summary}".strip()

                    # Merge search queries (tenor-specific first)
                    search_queries = list(tb.search_queries) if tb else []
                    if brief:
                        search_queries.extend(brief.search_queries)

                    agent_forecasts.append(AgentForecast(
                        agent_id=agent_id,
                        probability=p,
                        reasoning=reasoning,
                        search_queries=search_queries,
                        evidence=merged_evidence,
                        iterations=brief.iterations if brief else 0,
                        search_mode=brief.search_mode if brief else "hybrid",
                    ))

                ensemble_result = EnsembleResult(
                    agent_forecasts=agent_forecasts,
                    mean_probability=mean_prob,
                    supervisor=None,
                    final_probability=mean_prob,
                )

                # Aggregate tenor-specific causal factors across agents
                tenor_consensus_factors = _build_tenor_consensus_factors(cell_tenor_briefs)
                relevance_parts: list[str] = []
                for tb in cell_tenor_briefs:
                    if tb.relevance_summary:
                        relevance_parts.append(tb.relevance_summary)

                cell_probabilities[(strike, tenor)] = mean_prob

                console.print(
                    f"  [dim]{q_text}[/dim] → mean={mean_prob:.4f} "
                    f"(n={len(agent_probs)})"
                )

                surface.cells.append(SurfaceCell(
                    strike=strike,
                    tenor=tenor,
                    question=q_text,
                    ensemble=ensemble_result,
                    causal_factors=tenor_consensus_factors,
                    tenor_relevance=relevance_parts[0] if relevance_parts else "",
                ))

        # --- Phase 3: Surface-level Supervisor ---
        console.print("\n[bold]Phase 3: Surface Supervisor Review[/bold]")
        from aia_forecaster.agents.supervisor import SupervisorAgent

        supervisor = SupervisorAgent(llm=self.llm)

        # Capture regime for surface metadata
        if app_settings.regime_weighting_enabled and shared_research.briefs:
            try:
                regime, dominant_channels, regime_reasoning = await supervisor.detect_regime(
                    pair, shared_research.briefs
                )
                surface.regime = regime
                surface.regime_dominant_channels = dominant_channels
                console.print(
                    f"  [bold]Regime:[/bold] {regime} "
                    f"[dim](channels: {', '.join(dominant_channels)})[/dim]"
                )
            except Exception as e:
                logger.warning("Regime detection failed: %s", e)

        try:
            adjustments = await supervisor.review_surface(
                pair=pair,
                spot=spot,
                cutoff_date=cutoff_date,
                strikes=strikes,
                tenors=tenors,
                cell_probabilities=cell_probabilities,
                briefs=shared_research.briefs,
                forecast_mode=forecast_mode,
            )
            if adjustments:
                console.print(
                    f"  [yellow]Supervisor adjusted {len(adjustments)} cell(s)[/yellow]"
                )
                # Apply high-confidence adjustments
                for cell in surface.cells:
                    key = (cell.strike, cell.tenor)
                    if key in adjustments:
                        adjusted_p = adjustments[key]
                        cell_probabilities[key] = adjusted_p
                        if cell.ensemble:
                            cell.ensemble.final_probability = adjusted_p
                        console.print(
                            f"    [{cell.tenor.value}, {cell.strike:.2f}]: "
                            f"adjusted to {adjusted_p:.4f}"
                        )
            else:
                console.print("  [green]No adjustments needed[/green]")
        except Exception as e:
            logger.error("Supervisor surface review failed: %s", e)
            console.print(f"  [red]Supervisor failed: {e}[/red]")

        # --- Monotonicity (PAVA on raw means) ---
        if forecast_mode == ForecastMode.HITTING:
            n_adjusted = enforce_hitting_monotonicity(
                cell_probabilities, strikes, tenors, spot,
            )
        else:
            n_adjusted = enforce_raw_surface_monotonicity(
                cell_probabilities, strikes, tenors,
            )
        if n_adjusted:
            console.print(
                f"[yellow]Monotonicity (raw): adjusted {n_adjusted} cell(s)[/yellow]"
            )
        else:
            console.print("[green]Monotonicity: no violations[/green]")

        # --- Calibration (Platt scaling on monotone sequence) ---
        console.print("[bold]Calibrating (Platt scaling)...[/bold]")
        for cell in surface.cells:
            final_p = cell_probabilities.get(
                (cell.strike, cell.tenor),
                cell.ensemble.final_probability if cell.ensemble else 0.5,
            )
            cell.calibrated = calibrate(final_p)

        return surface


def print_surface(surface: ProbabilitySurface) -> None:
    """Print the probability surface as a rich table."""
    # Collect unique strikes and tenors
    strikes = sorted(set(c.strike for c in surface.cells))
    tenors = sorted(set(c.tenor for c in surface.cells), key=lambda t: t.days)

    # Build lookup
    lookup: dict[tuple[float, Tenor], float | None] = {}
    for c in surface.cells:
        p = c.calibrated.calibrated_probability if c.calibrated else None
        lookup[(c.strike, c.tenor)] = p

    mode_label = (
        "Barrier/Touch" if surface.forecast_mode == ForecastMode.HITTING else "Above Strike"
    )
    table = Table(title=f"{surface.pair} {mode_label} Probability Surface (spot={surface.spot_rate})")
    table.add_column("Strike", style="bold")
    for tenor in tenors:
        table.add_column(tenor.value, justify="center")

    for strike in strikes:
        row = [f"{strike:.2f}"]
        for tenor in tenors:
            p = lookup.get((strike, tenor))
            if p is not None:
                # Color code: green for high, red for low
                if p >= 0.6:
                    row.append(f"[green]{p:.3f}[/green]")
                elif p <= 0.4:
                    row.append(f"[red]{p:.3f}[/red]")
                else:
                    row.append(f"{p:.3f}")
            else:
                row.append("[dim]—[/dim]")
        table.add_row(*row)

    console.print(table)


def plot_surface(surface: ProbabilitySurface, output_path: str | Path) -> Path:
    """Render the probability surface as a heatmap and save to PNG.

    Args:
        surface: The probability surface data.
        output_path: File path for the saved image.

    Returns:
        The resolved output path.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    strikes = sorted(set(c.strike for c in surface.cells))
    tenors = sorted(
        set(c.tenor for c in surface.cells), key=lambda t: t.days
    )

    # Build lookup
    lookup: dict[tuple[float, Tenor], float | None] = {}
    for c in surface.cells:
        p = c.calibrated.calibrated_probability if c.calibrated else None
        lookup[(c.strike, c.tenor)] = p

    # Build 2D array (rows=strikes descending so higher strikes at top, cols=tenors)
    strikes_desc = list(reversed(strikes))
    data = np.full((len(strikes_desc), len(tenors)), np.nan)
    for i, strike in enumerate(strikes_desc):
        for j, tenor in enumerate(tenors):
            p = lookup.get((strike, tenor))
            if p is not None:
                data[i, j] = p

    fig, ax = plt.subplots(figsize=(max(8, len(tenors) * 1.8), max(5, len(strikes) * 0.7)))

    im = ax.imshow(data, cmap="RdYlGn", vmin=0.0, vmax=1.0, aspect="auto")

    # Annotate cells
    for i in range(len(strikes_desc)):
        for j in range(len(tenors)):
            val = data[i, j]
            if not np.isnan(val):
                color = "white" if val < 0.25 or val > 0.75 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=color, fontsize=11, fontweight="bold")

    # Axis labels
    ax.set_xticks(range(len(tenors)))
    ax.set_xticklabels([t.value for t in tenors], fontsize=11)
    ax.set_yticks(range(len(strikes_desc)))
    ax.set_yticklabels([f"{s:.2f}" for s in strikes_desc], fontsize=10)
    ax.set_xlabel("Tenor", fontsize=12)
    ax.set_ylabel("Strike", fontsize=12)

    # Spot rate marker
    if surface.spot_rate is not None:
        for i, s in enumerate(strikes_desc):
            if s <= surface.spot_rate:
                spot_y = max(0, i - 0.5) if i == 0 or strikes_desc[i - 1] > surface.spot_rate else i - (surface.spot_rate - s) / (strikes_desc[i - 1] - s) if i > 0 and strikes_desc[i - 1] != s else i
                ax.axhline(y=spot_y, color="blue", linewidth=2, linestyle="--", alpha=0.7)
                ax.text(
                    len(tenors) - 0.5, spot_y, f"  spot={surface.spot_rate:.2f}",
                    va="center", ha="left", color="blue", fontsize=9, fontweight="bold",
                    clip_on=False,
                )
                break

    base, quote = surface.pair[:3], surface.pair[3:]
    date_str = surface.generated_at.strftime("%Y-%m-%d")
    is_hitting = surface.forecast_mode == ForecastMode.HITTING
    mode_subtitle = "Barrier/Touch" if is_hitting else "Above Strike"
    ax.set_title(
        f"{base}/{quote} {mode_subtitle} Probability Surface\nspot={surface.spot_rate}  |  as-of {date_str}",
        fontsize=14, fontweight="bold", pad=12,
    )

    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.12)
    cbar.set_label("P(touch barrier)" if is_hitting else "P(above strike)", fontsize=11)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return output_path


def plot_surface_scatter(surface: ProbabilitySurface, output_path: str | Path) -> Path:
    """Render three 2D scatter plots of the probability surface and save to PNG.

    The three plots are:
      1. Probability vs Strike (one series per tenor)
      2. Probability vs Tenor (one series per strike)
      3. Raw vs Calibrated probability (Platt scaling effect)

    Args:
        surface: The probability surface data.
        output_path: File path for the saved image.

    Returns:
        The resolved output path.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    strikes = sorted(set(c.strike for c in surface.cells))
    tenors = sorted(
        set(c.tenor for c in surface.cells), key=lambda t: t.days
    )
    tenor_labels = [t.value for t in tenors]
    tenor_days = [t.days for t in tenors]

    # Build lookup
    lookup: dict[tuple[float, Tenor], float | None] = {}
    for c in surface.cells:
        p = c.calibrated.calibrated_probability if c.calibrated else None
        lookup[(c.strike, c.tenor)] = p

    is_hitting = surface.forecast_mode == ForecastMode.HITTING
    p_label = "P(touch barrier)" if is_hitting else "P(above strike)"
    base, quote = surface.pair[:3], surface.pair[3:]
    date_str = surface.generated_at.strftime("%Y-%m-%d")
    mode_subtitle = "Barrier/Touch" if is_hitting else "Above Strike"

    # Color palette for series
    cmap_series = plt.cm.get_cmap("tab10")

    fig, axes = plt.subplots(1, 3, figsize=(22, 6))

    # --- Plot 1: Probability vs Strike (one series per tenor) ---
    ax1 = axes[0]
    for j, (tenor, t_label) in enumerate(zip(tenors, tenor_labels)):
        xs = []
        ys = []
        for strike in strikes:
            p = lookup.get((strike, tenor))
            if p is not None:
                xs.append(strike)
                ys.append(p)
        ax1.scatter(xs, ys, color=cmap_series(j), label=t_label, s=60, zorder=3)
        ax1.plot(xs, ys, color=cmap_series(j), alpha=0.4, linewidth=1.2)

    if surface.spot_rate is not None:
        ax1.axvline(x=surface.spot_rate, color="blue", linestyle="--", alpha=0.6, label=f"spot={surface.spot_rate:.2f}")
    ax1.set_xlabel("Strike", fontsize=11)
    ax1.set_ylabel(p_label, fontsize=11)
    ax1.set_ylim(-0.05, 1.05)
    ax1.set_title("Prob vs Strike", fontsize=12, fontweight="bold")
    ax1.legend(fontsize=8, title="Tenor")
    ax1.grid(True, alpha=0.3)

    # --- Plot 2: Probability vs Tenor (one series per strike) ---
    ax2 = axes[1]
    for i, strike in enumerate(strikes):
        xs = []
        ys = []
        for j, tenor in enumerate(tenors):
            p = lookup.get((strike, tenor))
            if p is not None:
                xs.append(tenor_days[j])
                ys.append(p)
        ax2.scatter(xs, ys, color=cmap_series(i), label=f"{strike:.2f}", s=60, zorder=3)
        ax2.plot(xs, ys, color=cmap_series(i), alpha=0.4, linewidth=1.2)

    ax2.set_xscale("log")
    ax2.set_xticks(tenor_days)
    ax2.set_xticklabels(tenor_labels)
    ax2.set_xlabel("Tenor", fontsize=11)
    ax2.set_ylabel(p_label, fontsize=11)
    ax2.set_ylim(-0.05, 1.05)
    ax2.set_title("Prob vs Tenor", fontsize=12, fontweight="bold")
    ax2.legend(fontsize=8, title="Strike")
    ax2.grid(True, alpha=0.3)

    # --- Plot 3: Raw vs Calibrated (Platt scaling effect) ---
    ax3 = axes[2]
    raw_ps = []
    cal_ps = []
    labels = []
    for c in surface.cells:
        if c.calibrated is not None:
            raw_ps.append(c.calibrated.raw_probability)
            cal_ps.append(c.calibrated.calibrated_probability)
            labels.append(f"{c.strike:.0f}/{c.tenor.value}")

    ax3.scatter(raw_ps, cal_ps, s=70, color="teal", edgecolors="black", linewidths=0.5, zorder=3)
    # 45° reference line
    ax3.plot([0, 1], [0, 1], color="grey", linestyle="--", alpha=0.5, label="no change")
    # Annotate points
    for x, y, lbl in zip(raw_ps, cal_ps, labels):
        ax3.annotate(lbl, (x, y), textcoords="offset points", xytext=(5, 5), fontsize=7, alpha=0.7)
    ax3.set_xlabel("Raw (pre-calibration)", fontsize=11)
    ax3.set_ylabel("Calibrated (post-Platt)", fontsize=11)
    ax3.set_xlim(-0.05, 1.05)
    ax3.set_ylim(-0.05, 1.05)
    ax3.set_aspect("equal")
    ax3.set_title("Platt Scaling Effect", fontsize=12, fontweight="bold")
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)

    fig.suptitle(
        f"{base}/{quote} {mode_subtitle} — Scatter Views  (spot={surface.spot_rate}, {date_str})",
        fontsize=14, fontweight="bold", y=1.02,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return output_path


# ---------------------------------------------------------------------------
# Tenor numeric mapping for 3D surface
# ---------------------------------------------------------------------------



_HOVER_LINE_WIDTH = 60  # max chars per line in hover tooltips


def _wrap_html(text: str, width: int = _HOVER_LINE_WIDTH) -> str:
    """Wrap *text* to *width* using ``<br>`` tags instead of newlines."""
    return "<br>".join(textwrap.wrap(text, width=width))


def _build_hover_text(
    surface: ProbabilitySurface,
    strikes: list[float],
    tenors: list[Tenor],
) -> list[list[str]]:
    """Build per-cell hover HTML from explanation data.

    Returns a 2D list [strike_idx][tenor_idx] of hover text strings.
    """
    from aia_forecaster.fx.explanation import explain_cell

    # Build cell lookup
    cell_lookup: dict[tuple[float, Tenor], SurfaceCell] = {}
    for c in surface.cells:
        cell_lookup[(c.strike, c.tenor)] = c

    hover: list[list[str]] = []
    for strike in strikes:
        row: list[str] = []
        for tenor in tenors:
            cell = cell_lookup.get((strike, tenor))
            if cell is None:
                row.append("")
                continue

            expl = explain_cell(cell)
            cal_p = expl.calibrated_probability
            raw_p = expl.raw_probability

            is_hitting = surface.forecast_mode == ForecastMode.HITTING
            p_label = "P(touch)" if is_hitting else "P(above)"
            strike_label = "Barrier" if is_hitting else "Strike"

            lines = [
                f"<b>{strike_label}:</b> {strike:.2f}  |  <b>Tenor:</b> {tenor.value}",
                f"<b>{p_label}:</b> {cal_p:.3f}" + (f"  (raw: {raw_p:.3f})" if raw_p else ""),
                f"<b>Agents:</b> {expl.num_agents}",
            ]

            # Tenor-specific causal factors (cell-level from Phase 1.5)
            if expl.causal_factors:
                lines.append(f"<br><b>Tenor Causal Factors ({tenor.value}):</b>")
                for cf in expl.causal_factors[:5]:
                    icon = "+" if cf.direction == "bullish" else ("~" if cf.direction == "contested" else "-")
                    event_text = _wrap_html(
                        html_mod.escape(cf.event), width=_HOVER_LINE_WIDTH - 4
                    )
                    detail = (
                        f"{html_mod.escape(cf.channel)} → {cf.direction} "
                        f"| {cf.magnitude} | {cf.confidence}"
                    )
                    detail_wrapped = _wrap_html(detail, width=_HOVER_LINE_WIDTH - 6)
                    lines.append(
                        f"  {icon} {event_text}"
                        f"<br>    <i>{detail_wrapped}</i>"
                    )
                if expl.tenor_relevance:
                    rel_text = _wrap_html(
                        html_mod.escape(expl.tenor_relevance),
                        width=_HOVER_LINE_WIDTH - 4,
                    )
                    lines.append(f"  <i>{rel_text}</i>")
            elif surface.causal_factors:
                # Fallback to surface-level causal factors if no tenor-specific ones
                regime_tag = ""
                if surface.regime:
                    regime_tag = f"  [regime: {html_mod.escape(surface.regime)}]"
                lines.append(f"<br><b>Causal Factors{regime_tag}:</b>")
                for cf in surface.causal_factors[:5]:
                    icon = "+" if cf.direction == "bullish" else ("~" if cf.direction == "contested" else "-")
                    event_text = _wrap_html(
                        html_mod.escape(cf.event), width=_HOVER_LINE_WIDTH - 4
                    )
                    detail = (
                        f"{html_mod.escape(cf.channel)} → {cf.direction} "
                        f"| {cf.magnitude} | {cf.confidence}"
                    )
                    detail_wrapped = _wrap_html(detail, width=_HOVER_LINE_WIDTH - 6)
                    lines.append(
                        f"  {icon} {event_text}"
                        f"<br>    <i>{detail_wrapped}</i>"
                    )

            # Disagreements
            if expl.disagreement_notes:
                body = _wrap_html(html_mod.escape(expl.disagreement_notes))
                lines.append(f"<b>Disagreements:</b><br>{body}")

            # Top evidence (up to 3)
            if expl.top_evidence:
                lines.append(f"<br><b>Sources ({len(expl.top_evidence)}):</b>")
                for ev in expl.top_evidence[:3]:
                    title = _wrap_html(
                        html_mod.escape(ev.title), width=_HOVER_LINE_WIDTH - 4
                    )
                    snippet = _wrap_html(
                        html_mod.escape(ev.snippet), width=_HOVER_LINE_WIDTH - 6
                    )
                    cited = f" [{ev.cited_by_agents} agents]" if ev.cited_by_agents > 1 else ""
                    lines.append(f"  • {title}{cited}")
                    lines.append(f"    <i>{snippet}</i>")
                    url_wrapped = _wrap_html(
                        html_mod.escape(ev.url), width=_HOVER_LINE_WIDTH - 4
                    )
                    lines.append(f"    {url_wrapped}")

            # Supervisor
            if expl.supervisor_reasoning:
                sup_text = _wrap_html(html_mod.escape(expl.supervisor_reasoning))
                lines.append(
                    f"<br><b>Supervisor ({expl.supervisor_confidence}):</b>"
                    f"<br>{sup_text}"
                )

            row.append("<br>".join(lines))
        hover.append(row)

    return hover


def _collect_all_evidence(surface: ProbabilitySurface) -> list[EvidenceItem]:
    """Collect and deduplicate evidence across all surface cells.

    Returns evidence sorted by citation count (descending).
    """
    from aia_forecaster.fx.explanation import explain_cell

    url_map: dict[str, EvidenceItem] = {}
    for cell in surface.cells:
        expl = explain_cell(cell)
        for ev in expl.top_evidence:
            key = ev.url.rstrip("/").lower()
            if key in url_map:
                # Take the higher citation count
                url_map[key].cited_by_agents = max(
                    url_map[key].cited_by_agents, ev.cited_by_agents
                )
            else:
                url_map[key] = ev.model_copy()
    return sorted(url_map.values(), key=lambda e: -e.cited_by_agents)


def _build_citations_html(
    evidence: list[EvidenceItem],
    pair: str,
    date_str: str,
) -> str:
    """Build an HTML section with clickable citation links."""
    if not evidence:
        return ""

    rows: list[str] = []
    for i, ev in enumerate(evidence, 1):
        title_esc = html_mod.escape(ev.title)
        snippet_esc = html_mod.escape(ev.snippet[:300])
        url_esc = html_mod.escape(ev.url)
        source_esc = html_mod.escape(ev.source) if ev.source else ""
        agents_label = (
            f"{ev.cited_by_agents} agents"
            if ev.cited_by_agents > 1
            else "1 agent"
        )
        rows.append(
            f'<tr id="ref-{i}">'
            f'<td style="vertical-align:top;padding:8px;color:#666;font-weight:bold;">[{i}]</td>'
            f'<td style="padding:8px;">'
            f'<a href="{url_esc}" target="_blank" rel="noopener noreferrer" '
            f'style="color:#1a73e8;text-decoration:none;font-weight:600;">{title_esc}</a>'
            f'<span style="color:#888;font-size:0.85em;margin-left:8px;">({agents_label})</span>'
            f'<br><span style="color:#555;font-size:0.9em;">{snippet_esc}</span>'
            f'<br><span style="color:#999;font-size:0.8em;">{url_esc}'
            + (f" &middot; {source_esc}" if source_esc else "")
            + "</span></td></tr>"
        )

    return (
        '<div style="max-width:1100px;margin:30px auto;font-family:-apple-system,BlinkMacSystemFont,'
        "'Segoe UI',Roboto,sans-serif;\">"
        f'<h2 style="border-bottom:2px solid #1a73e8;padding-bottom:8px;color:#202124;">'
        f"Sources &amp; Citations &mdash; {html_mod.escape(pair)} ({html_mod.escape(date_str)})</h2>"
        f'<p style="color:#666;font-size:0.9em;margin-bottom:16px;">'
        f"{len(evidence)} unique sources cited across forecast agents. "
        f"Click any title to open the original article.</p>"
        f'<table style="width:100%;border-collapse:collapse;">'
        + "".join(rows)
        + "</table></div>"
    )


def plot_surface_3d(surface: ProbabilitySurface, output_path: str | Path) -> Path:
    """Render an interactive 3D probability surface and save as HTML.

    The surface can be rotated, zoomed, and hovered to inspect per-cell
    explanations including consensus reasoning, evidence sources, and
    agent disagreements.

    Args:
        surface: The probability surface data.
        output_path: File path for the saved HTML file.

    Returns:
        The resolved output path.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    strikes = sorted(set(c.strike for c in surface.cells))
    tenors = sorted(
        set(c.tenor for c in surface.cells), key=lambda t: t.days
    )
    tenor_labels = [t.value for t in tenors]
    tenor_days = [t.days for t in tenors]

    # Build probability grid [strike_idx][tenor_idx]
    lookup: dict[tuple[float, Tenor], float | None] = {}
    for c in surface.cells:
        p = c.calibrated.calibrated_probability if c.calibrated else None
        lookup[(c.strike, c.tenor)] = p

    z_data = []
    for strike in strikes:
        row = []
        for tenor in tenors:
            p = lookup.get((strike, tenor))
            row.append(p if p is not None else float("nan"))
        z_data.append(row)

    z_arr = np.array(z_data)

    # Build hover text
    hover_text = _build_hover_text(surface, strikes, tenors)

    base, quote = surface.pair[:3], surface.pair[3:]
    date_str = surface.generated_at.strftime("%Y-%m-%d")
    is_hitting = surface.forecast_mode == ForecastMode.HITTING
    cbar_title = "P(touch barrier)" if is_hitting else "P(above strike)"

    fig = go.Figure()

    # Main probability surface
    fig.add_trace(go.Surface(
        x=tenor_days,
        y=strikes,
        z=z_arr,
        customdata=np.array(hover_text, dtype=object),
        hovertemplate="%{customdata}<extra></extra>",
        colorscale="RdYlGn",
        cmin=0.0,
        cmax=1.0,
        colorbar=dict(
            title=dict(text=cbar_title, side="right"),
            thickness=20,
            len=0.75,
        ),
        opacity=0.92,
    ))

    # Spot rate plane (translucent horizontal plane at spot rate)
    if surface.spot_rate is not None:
        spot_z = np.full((2, len(tenor_days)), 0.5)  # arbitrary z for reference plane
        fig.add_trace(go.Surface(
            x=[tenor_days[0], tenor_days[-1]],
            y=[surface.spot_rate, surface.spot_rate],
            z=spot_z,
            showscale=False,
            opacity=0.3,
            colorscale=[[0, "royalblue"], [1, "royalblue"]],
            hovertemplate=(
                f"<b>Spot rate:</b> {surface.spot_rate:.4f}<extra></extra>"
            ),
        ))

    mode_subtitle = "Barrier/Touch" if is_hitting else "Above Strike"
    z_title = "P(touch barrier)" if is_hitting else "P(above strike)"

    fig.update_layout(
        title=dict(
            text=f"{base}/{quote} {mode_subtitle} Probability Surface — {date_str}",
            font=dict(size=18),
        ),
        scene=dict(
            xaxis=dict(
                title="Tenor",
                tickvals=tenor_days,
                ticktext=tenor_labels,
                type="log",
            ),
            yaxis=dict(
                title="Barrier" if is_hitting else "Strike",
                tickformat=".2f",
            ),
            zaxis=dict(
                title=z_title,
                range=[0, 1],
            ),
            camera=dict(
                eye=dict(x=1.8, y=-1.4, z=0.8),
            ),
        ),
        hoverlabel=dict(
            bgcolor="rgba(255,255,255,0.95)",
            font_size=12,
            font_family="monospace",
            align="left",
        ),
        margin=dict(l=0, r=0, t=60, b=0),
        template="plotly_white",
        autosize=True,
    )

    # Build chart div (not full HTML — we assemble the page ourselves)
    chart_html = fig.to_html(
        include_plotlyjs=True,
        full_html=False,
        default_width="100%",
        default_height="100%",
        config={
            "displayModeBar": True,
            "scrollZoom": True,
            "modeBarButtonsToAdd": ["hoverclosest", "hovercompare"],
        },
    )

    # Assemble full HTML page (citations moved to PDF report for cleaner chart rendering)
    full_page = (
        "<!DOCTYPE html>\n<html>\n<head>\n"
        f"<title>{base}/{quote} Probability Surface — {date_str}</title>\n"
        '<meta charset="utf-8">\n'
        "<style>html,body{margin:0;padding:0;width:100%;height:100%;overflow:hidden;background:#fafafa;}"
        ".js-plotly-plot,.plot-container,.plotly{width:100%!important;height:100%!important;}</style>\n"
        "</head>\n<body>\n"
        f"{chart_html}\n"
        "</body>\n</html>"
    )

    output_path.write_text(full_page, encoding="utf-8")

    return output_path


def plot_cdf(surface: ProbabilitySurface, output_path: str | Path) -> Path | None:
    """Render CDF curves — P(spot < K) at each strike — and save to PNG.

    Each tenor gets its own curve. The result is directly comparable to
    digital-put prices in the options market: a digital put at strike K
    costs exactly CDF(K) in risk-neutral terms.

    Only applicable to ABOVE mode (terminal distribution).  Returns None
    for HITTING mode since barrier-touch probabilities are not a CDF.

    Args:
        surface: The probability surface data.
        output_path: File path for the saved image.

    Returns:
        The resolved output path, or None if not applicable.
    """
    if surface.forecast_mode != ForecastMode.ABOVE:
        logger.info("CDF chart skipped — only applicable to ABOVE mode")
        return None

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    strikes = sorted(set(c.strike for c in surface.cells))
    tenors = sorted(
        set(c.tenor for c in surface.cells), key=lambda t: t.days
    )

    # Build lookup: P(above K) per (strike, tenor)
    lookup: dict[tuple[float, Tenor], float | None] = {}
    for c in surface.cells:
        p = c.calibrated.calibrated_probability if c.calibrated else None
        lookup[(c.strike, c.tenor)] = p

    base, quote = surface.pair[:3], surface.pair[3:]
    date_str = surface.generated_at.strftime("%Y-%m-%d")

    # Dark theme matching the reference image
    with plt.style.context("dark_background"):
        fig, ax = plt.subplots(figsize=(max(10, len(strikes) * 1.0), 6))
        fig.patch.set_facecolor("#1e2330")
        ax.set_facecolor("#2a2f3e")

        cmap_series = plt.cm.get_cmap("tab10")

        for j, tenor in enumerate(tenors):
            xs: list[float] = []
            ys: list[float] = []
            for strike in strikes:
                p_above = lookup.get((strike, tenor))
                if p_above is not None:
                    xs.append(strike)
                    ys.append((1.0 - p_above) * 100.0)  # CDF = 1 - P(above)

            color = cmap_series(j)
            ax.plot(xs, ys, color=color, linewidth=2, alpha=0.9, zorder=3)
            ax.scatter(xs, ys, color=color, s=50, zorder=4,
                       label=tenor.value, edgecolors="white", linewidths=0.5)

        # Spot rate reference line (yellow dashed, matching reference image)
        if surface.spot_rate is not None:
            ax.axvline(
                x=surface.spot_rate, color="#FFD700", linestyle="--",
                linewidth=2, alpha=0.8, label=f"spot = {surface.spot_rate:.4f}",
            )

        ax.set_xlabel(f"{base}/{quote}", fontsize=13, fontweight="bold")
        ax.set_ylabel("P(spot < K)  %", fontsize=13, fontweight="bold")
        ax.set_ylim(-2, 102)
        ax.set_title(
            f"P(spot < K) at each strike — {base}/{quote}\n"
            f"spot = {surface.spot_rate}  |  {date_str}",
            fontsize=14, fontweight="bold", pad=12,
        )
        ax.legend(fontsize=9, title="Tenor", loc="upper left")
        ax.grid(True, alpha=0.2, linestyle="--")

        fig.tight_layout()
        fig.savefig(output_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close(fig)

    return output_path
