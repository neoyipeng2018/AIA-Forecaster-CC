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
    EnsembleResult,
    ForecastMode,
    ProbabilitySurface,
    ResearchBrief,
    SourceConfig,
    SurfaceCell,
    Tenor,
)

logger = logging.getLogger(__name__)
console = Console()


def _build_consensus_factors(briefs: list[ResearchBrief]) -> list[CausalFactor]:
    """Aggregate causal factors across all agent briefs into consensus factors.

    Groups by (channel, direction), keeps the most common magnitude,
    and only returns factors cited by 2+ agents (or all if fewer than 2 agents).
    """
    if not briefs:
        return []

    # Collect all factors with agent ids
    all_factors: list[tuple[int, CausalFactor]] = []
    for brief in briefs:
        for f in brief.causal_factors:
            all_factors.append((brief.agent_id, f))

    if not all_factors:
        return []

    # Group by (channel_normalized, direction_normalized)
    groups: dict[tuple[str, str], list[tuple[int, CausalFactor]]] = {}
    for agent_id, f in all_factors:
        key = (f.channel.lower().strip(), f.direction.lower().strip())
        groups.setdefault(key, []).append((agent_id, f))

    # Build consensus factors, sorted by citation count.
    # Require 2+ agents when enough agents produced factors; otherwise show all.
    agents_with_factors = len({aid for aid, _ in all_factors})
    min_count = 2 if agents_with_factors >= 3 else 1
    consensus: list[CausalFactor] = []
    for (_channel, _direction), entries in sorted(groups.items(), key=lambda x: -len(x[1])):
        if len(entries) < min_count:
            continue
        magnitudes = [e.magnitude for _, e in entries]
        confidences = [e.confidence for _, e in entries]
        representative = entries[0][1]
        agent_count = len({aid for aid, _ in entries})
        consensus.append(CausalFactor(
            event=f"[{agent_count}/{len(briefs)} agents] {representative.event}",
            channel=representative.channel,
            direction=representative.direction,
            magnitude=max(set(magnitudes), key=magnitudes.count),
            confidence=max(set(confidences), key=confidences.count),
        ))

    return consensus


def _question_text(
    pair: str,
    strike: float,
    tenor: Tenor,
    forecast_mode: ForecastMode = ForecastMode.ABOVE,
) -> str:
    """Formulate the binary question for a (strike, tenor) cell."""
    base, quote = pair[:3], pair[3:]
    tenor_map = {
        Tenor.D1: "1 day",
        Tenor.W1: "1 week",
        Tenor.M1: "1 month",
        Tenor.M3: "3 months",
        Tenor.M6: "6 months",
    }
    horizon = tenor_map.get(tenor, tenor.value)
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

        Returns:
            ProbabilitySurface with calibrated probabilities for each cell.
        """
        if tenors is None:
            tenors = DEFAULT_TENORS
        if cutoff_date is None:
            cutoff_date = date.today()

        # Get spot rate and generate strikes
        spot = await get_spot_rate(pair)
        strikes = generate_strikes(spot, pair, num_strikes, forecast_mode=forecast_mode)
        console.print(f"[bold]Spot rate:[/bold] {pair} = {spot}")
        console.print(f"[bold]Strikes:[/bold] {strikes}")
        console.print(f"[bold]Tenors:[/bold] {[t.value for t in tenors]}")

        total_cells = len(strikes) * len(tenors)
        num_agents = self.engine.num_agents
        estimated_calls = (
            num_agents * 7  # ~7 LLM calls per agent in research (queries + assess + summary)
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
                for idx, p in enumerate(agent_probs):
                    brief = briefs[idx] if idx < len(briefs) else None
                    agent_forecasts.append(AgentForecast(
                        agent_id=brief.agent_id if brief else idx,
                        probability=p,
                        reasoning=brief.macro_summary if brief else "",
                        search_queries=brief.search_queries if brief else [],
                        evidence=brief.evidence if brief else [],
                        iterations=brief.iterations if brief else 0,
                        search_mode=brief.search_mode if brief else "hybrid",
                    ))

                ensemble_result = EnsembleResult(
                    agent_forecasts=agent_forecasts,
                    mean_probability=mean_prob,
                    supervisor=None,
                    final_probability=mean_prob,
                )

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
                ))

        # --- Phase 3: Surface-level Supervisor ---
        console.print("\n[bold]Phase 3: Surface Supervisor Review[/bold]")
        from aia_forecaster.agents.supervisor import SupervisorAgent

        supervisor = SupervisorAgent(llm=self.llm)

        # Capture regime for surface metadata
        from aia_forecaster.config import settings as app_settings

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
    tenors = sorted(set(c.tenor for c in surface.cells), key=lambda t: DEFAULT_TENORS.index(t))

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
        set(c.tenor for c in surface.cells), key=lambda t: DEFAULT_TENORS.index(t)
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
        set(c.tenor for c in surface.cells), key=lambda t: DEFAULT_TENORS.index(t)
    )
    tenor_labels = [t.value for t in tenors]
    tenor_days = [_TENOR_DAYS[t] for t in tenors]

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

_TENOR_DAYS: dict[Tenor, int] = {
    Tenor.D1: 1,
    Tenor.W1: 7,
    Tenor.M1: 30,
    Tenor.M3: 90,
    Tenor.M6: 180,
}


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

            # Causal factors (surface-level, shown per cell in hover)
            if surface.causal_factors:
                regime_tag = ""
                if surface.regime:
                    regime_tag = f"  [regime: {html_mod.escape(surface.regime)}]"
                lines.append(f"<br><b>Causal Factors{regime_tag}:</b>")
                for cf in surface.causal_factors[:5]:
                    icon = "+" if cf.direction == "bullish" else "-"
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

            # Consensus
            if expl.consensus_summary:
                body = _wrap_html(html_mod.escape(expl.consensus_summary))
                lines.append(f"<br><b>Consensus:</b><br>{body}")

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
        set(c.tenor for c in surface.cells), key=lambda t: DEFAULT_TENORS.index(t)
    )
    tenor_labels = [t.value for t in tenors]
    tenor_days = [_TENOR_DAYS[t] for t in tenors]

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
        margin=dict(l=20, r=20, t=60, b=20),
        template="plotly_white",
    )

    fig.write_html(
        str(output_path),
        include_plotlyjs=True,
        full_html=True,
        config={
            "displayModeBar": True,
            "scrollZoom": True,
            "modeBarButtonsToAdd": ["hoverclosest", "hovercompare"],
        },
    )

    return output_path
