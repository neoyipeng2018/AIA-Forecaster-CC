"""Explanation extraction from ensemble forecast data.

Provides per-cell and per-surface evidence summaries without LLM calls —
all heuristic-based extraction from agent reasoning traces and evidence lists.
"""

from __future__ import annotations

import statistics
from datetime import datetime

from rich.console import Console
from rich.panel import Panel

from aia_forecaster.models import (
    AgentForecast,
    CausalFactor,
    CellExplanation,
    Confidence,
    EvidenceItem,
    ForecastMode,
    ProbabilitySurface,
    SupervisorResult,
    SurfaceCell,
    SurfaceExplanation,
)

console = Console()


# ---------------------------------------------------------------------------
# Evidence aggregation
# ---------------------------------------------------------------------------


def _is_irrelevant_evidence(url: str, title: str, snippet: str) -> bool:
    """Return True if this evidence is clearly not FX-relevant."""
    combined = (url + " " + title + " " + snippet).lower()
    # Calculator / date-tool / converter sites
    irrelevant_domains = [
        "calculator.net", "calculateconvert.com", "gigacalculator.com",
        "calculatorsoup.com", "timeanddate.com", "convertunits.com",
        "rapidtables.com", "epochconverter.com", "daysuntil.net",
    ]
    url_lower = url.lower()
    if any(d in url_lower for d in irrelevant_domains):
        return True
    # Generic date-calculator content (title/snippet both about date math)
    date_calc_keywords = ["date calculator", "days between dates", "date difference calculator"]
    if any(kw in combined for kw in date_calc_keywords):
        return True
    return False


def _deduplicate_evidence(agents: list[AgentForecast], top_n: int = 5) -> list[EvidenceItem]:
    """Aggregate evidence across agents, counting citation frequency.

    Uses URL-based deduplication. If multiple agents cite the same URL,
    merge into one EvidenceItem with cited_by_agents = count.
    Sorted by citation frequency descending. Filters out irrelevant sources.
    """
    url_map: dict[str, EvidenceItem] = {}
    for agent in agents:
        for result in agent.evidence:
            if _is_irrelevant_evidence(result.url, result.title, result.snippet):
                continue
            normalized_url = result.url.rstrip("/").lower()
            if normalized_url in url_map:
                url_map[normalized_url].cited_by_agents += 1
            else:
                url_map[normalized_url] = EvidenceItem(
                    title=result.title,
                    snippet=result.snippet,
                    url=result.url,
                    source=result.source,
                    cited_by_agents=1,
                )
    items = sorted(url_map.values(), key=lambda e: -e.cited_by_agents)
    return items[:top_n]


# ---------------------------------------------------------------------------
# Consensus & disagreement extraction
# ---------------------------------------------------------------------------


def _first_sentence(text: str) -> str:
    """Extract the first sentence from text."""
    earliest = len(text)
    for end in (".", "!", "?"):
        idx = text.find(end)
        if idx != -1 and idx < earliest:
            earliest = idx
    if earliest < len(text):
        return text[: earliest + 1].strip()
    # No sentence terminator found — return first 200 chars
    return text[:200].strip()


def _extract_tenor_reasoning(reasoning: str) -> str:
    """Extract the tenor-specific section from enriched agent reasoning.

    Tenor reasoning is appended as ``[<tenor label>] <summary>`` after
    a blank line in the agent reasoning field.  Returns the tenor part
    only, or empty string if none found.
    """
    for line in reasoning.split("\n"):
        stripped = line.strip()
        if stripped.startswith("[") and "] " in stripped:
            return stripped
    return ""


def _summarize_consensus(agents: list[AgentForecast]) -> str:
    """Identify majority direction and extract key reasoning themes.

    Heuristic: find majority direction (above/below 0.5), then take the
    first sentence from each agent whose probability aligns with the majority.
    Combine into a concise summary.  Also extracts tenor-specific reasoning
    when agents have been enriched with ``[tenor] ...`` lines.
    """
    if not agents:
        return ""

    probs = [a.probability for a in agents]
    mean_p = statistics.mean(probs)
    direction = "above" if mean_p >= 0.5 else "below"

    # Agents aligned with majority direction
    aligned = [a for a in agents if (a.probability >= 0.5) == (mean_p >= 0.5)]
    if not aligned:
        aligned = agents

    # Extract first sentence from each aligned agent (deduplicate)
    seen: set[str] = set()
    sentences: list[str] = []
    for a in aligned:
        s = _first_sentence(a.reasoning)
        if s and s not in seen:
            seen.add(s)
            sentences.append(s)
        if len(sentences) >= 3:
            break

    # Extract tenor-specific reasoning (deduplicate across agents)
    tenor_seen: set[str] = set()
    tenor_sentences: list[str] = []
    for a in aligned:
        tenor_line = _extract_tenor_reasoning(a.reasoning)
        if tenor_line:
            ts = _first_sentence(tenor_line)
            if ts and ts not in tenor_seen:
                tenor_seen.add(ts)
                tenor_sentences.append(ts)
            if len(tenor_sentences) >= 2:
                break

    if not sentences and not tenor_sentences:
        return f"Agents lean {direction} 0.5 (mean={mean_p:.3f})."

    parts = [f"Agents lean {direction} 0.5 (mean={mean_p:.3f})."]
    if sentences:
        parts.append(" ".join(sentences))
    if tenor_sentences:
        parts.append("Tenor view: " + " ".join(tenor_sentences))
    return " ".join(parts)


def _summarize_disagreements(
    agents: list[AgentForecast],
    supervisor: SupervisorResult | None,
) -> str:
    """Describe where agents diverged and how it was resolved."""
    if not agents or len(agents) < 2:
        return ""

    probs = [a.probability for a in agents]
    spread = max(probs) - min(probs)
    mean_p = statistics.mean(probs)
    stdev = statistics.stdev(probs) if len(probs) > 1 else 0.0

    # Low disagreement
    if spread < 0.15:
        return ""

    # Identify outliers (>1 stddev from mean)
    outliers = [
        a for a in agents
        if stdev > 0 and abs(a.probability - mean_p) > stdev
    ]

    parts = [f"Spread={spread:.3f} (stdev={stdev:.3f})."]

    if outliers:
        outlier_strs = [
            f"Agent {a.agent_id}: {a.probability:.3f}" for a in outliers
        ]
        parts.append(f"Outliers: {', '.join(outlier_strs)}.")

    if supervisor and supervisor.confidence == Confidence.HIGH and supervisor.reasoning:
        parts.append(
            f"Supervisor (HIGH confidence) resolved: {_first_sentence(supervisor.reasoning)}"
        )

    return " ".join(parts)


# ---------------------------------------------------------------------------
# Cell & surface explanation
# ---------------------------------------------------------------------------


def explain_cell(cell: SurfaceCell) -> CellExplanation:
    """Generate a CellExplanation from a SurfaceCell with ensemble data."""
    if cell.ensemble is None:
        return CellExplanation(
            strike=cell.strike,
            tenor=cell.tenor,
            calibrated_probability=(
                cell.calibrated.calibrated_probability if cell.calibrated else None
            ),
            raw_probability=(
                cell.calibrated.raw_probability if cell.calibrated else None
            ),
            tenor_catalysts=cell.tenor_catalysts,
            tenor_relevance=cell.tenor_relevance,
        )

    agents = cell.ensemble.agent_forecasts
    supervisor = cell.ensemble.supervisor

    return CellExplanation(
        strike=cell.strike,
        tenor=cell.tenor,
        calibrated_probability=(
            cell.calibrated.calibrated_probability if cell.calibrated else None
        ),
        raw_probability=(
            cell.calibrated.raw_probability if cell.calibrated else None
        ),
        num_agents=len(agents),
        agent_probabilities=[a.probability for a in agents],
        top_evidence=_deduplicate_evidence(agents),
        consensus_summary=_summarize_consensus(agents),
        disagreement_notes=_summarize_disagreements(agents, supervisor),
        supervisor_confidence=(
            supervisor.confidence.value if supervisor else None
        ),
        supervisor_reasoning=(
            supervisor.reasoning if supervisor else ""
        ),
        tenor_catalysts=cell.tenor_catalysts,
        tenor_relevance=cell.tenor_relevance,
    )


def explain_surface(surface: ProbabilitySurface) -> SurfaceExplanation:
    """Generate a full SurfaceExplanation from a ProbabilitySurface."""
    return SurfaceExplanation(
        pair=surface.pair,
        spot_rate=surface.spot_rate,
        generated_at=surface.generated_at,
        cells=[explain_cell(cell) for cell in surface.cells],
        forecast_mode=surface.forecast_mode,
        causal_factors=surface.causal_factors,
        regime=surface.regime,
        regime_dominant_channels=surface.regime_dominant_channels,
    )


# ---------------------------------------------------------------------------
# Rich CLI output
# ---------------------------------------------------------------------------


def _format_causal_factors_rich(factors: list[CausalFactor]) -> str:
    """Format causal factors for Rich CLI display."""
    if not factors:
        return "[dim]No causal factors identified.[/dim]"
    lines = []
    for f in factors:
        icon = "[green]+[/green]" if f.direction == "bullish" else "[red]-[/red]"
        lines.append(
            f"  {icon} {f.event}\n"
            f"    [dim]{f.channel} → {f.direction} | "
            f"magnitude: {f.magnitude} | confidence: {f.confidence}[/dim]"
        )
    return "\n".join(lines)


def print_explanation(explanation: SurfaceExplanation) -> None:
    """Print per-tenor explanations using Rich formatting."""
    console.print(f"\n[bold]Evidence & Reasoning — {explanation.pair}[/bold]\n")

    # Causal factors and regime header
    if explanation.causal_factors:
        regime_label = ""
        if explanation.regime:
            channels = ", ".join(explanation.regime_dominant_channels) if explanation.regime_dominant_channels else "—"
            regime_label = f"\n[bold]Regime:[/bold] {explanation.regime} [dim](dominant: {channels})[/dim]"

        console.print(Panel(
            f"[bold]Causal Factors ({len(explanation.causal_factors)}):[/bold]\n"
            + _format_causal_factors_rich(explanation.causal_factors)
            + regime_label,
            title="Causal Map",
            border_style="cyan",
        ))

    is_hitting = explanation.forecast_mode == ForecastMode.HITTING
    p_verb = "touch" if is_hitting else "above"

    # Group cells by tenor
    from aia_forecaster.models import Tenor
    tenor_order: list[Tenor] = []
    tenor_cells: dict[str, list[CellExplanation]] = {}
    for cell in explanation.cells:
        if cell.calibrated_probability is None:
            continue
        key = cell.tenor.value
        if key not in tenor_cells:
            tenor_order.append(cell.tenor)
            tenor_cells[key] = []
        tenor_cells[key].append(cell)

    for tenor in sorted(tenor_order, key=lambda t: t.days):
        cells = tenor_cells[tenor.value]
        cells.sort(key=lambda c: c.strike)

        # Pick representative cell (richest consensus/evidence)
        rep = max(cells, key=lambda c: (
            len(c.consensus_summary),
            len(c.top_evidence),
            len(c.tenor_catalysts),
        ))

        lines: list[str] = []

        # Strike probabilities
        lines.append(f"[bold]Strike Probabilities:[/bold]")
        for c in cells:
            raw_str = f"  (raw={c.raw_probability:.3f})" if c.raw_probability is not None else ""
            lines.append(
                f"  P({p_verb} {c.strike:.2f}) = {c.calibrated_probability:.3f}{raw_str}"
            )

        # Tenor-specific catalysts
        if rep.tenor_catalysts:
            lines.append(f"\n[bold cyan]Tenor Catalysts ({tenor.value}):[/bold cyan]")
            for i, cat in enumerate(rep.tenor_catalysts[:5], 1):
                lines.append(f"  {i}. {cat}")
            if rep.tenor_relevance:
                lines.append(f"  [dim]{rep.tenor_relevance}[/dim]")

        # Consensus — separate tenor view for readability
        if rep.consensus_summary:
            _tenor_marker = "Tenor view: "
            if _tenor_marker in rep.consensus_summary:
                _general, _tenor_part = rep.consensus_summary.split(_tenor_marker, 1)
                lines.append(f"\n[bold]Consensus:[/bold] {_general.strip()}")
                lines.append(f"  [bold cyan]Tenor view ({tenor.value}):[/bold cyan] {_tenor_part.strip()}")
            else:
                lines.append(f"\n[bold]Consensus:[/bold] {rep.consensus_summary}")

        # Top evidence — merge across all cells at this tenor
        all_evidence: dict[str, EvidenceItem] = {}
        for c in cells:
            for ev in c.top_evidence:
                norm_url = ev.url.rstrip("/").lower()
                if norm_url not in all_evidence or ev.cited_by_agents > all_evidence[norm_url].cited_by_agents:
                    all_evidence[norm_url] = ev
        merged_evidence = sorted(all_evidence.values(), key=lambda e: -e.cited_by_agents)[:5]

        if merged_evidence:
            lines.append(f"\n[bold]Top Evidence ({len(merged_evidence)} items):[/bold]")
            for i, e in enumerate(merged_evidence, 1):
                lines.append(
                    f"  [{i}] {e.title} "
                    f"[dim](cited by {e.cited_by_agents} agent{'s' if e.cited_by_agents > 1 else ''})[/dim]"
                )
                lines.append(f"      {e.snippet[:200]}")
                lines.append(f"      [dim]{e.url}[/dim]")

        # Disagreements — show the most detailed note from any cell at this tenor
        disagreements = [c.disagreement_notes for c in cells if c.disagreement_notes]
        if disagreements:
            best = max(disagreements, key=len)
            lines.append(f"\n[bold]Disagreements:[/bold] {best}")

        # Supervisor — show if any cell has supervisor reasoning
        sup_cells = [c for c in cells if c.supervisor_reasoning]
        if sup_cells:
            s = sup_cells[0]
            lines.append(
                f"\n[dim]Supervisor ({s.supervisor_confidence}): "
                f"{s.supervisor_reasoning[:300]}[/dim]"
            )

        console.print(Panel(
            "\n".join(lines),
            title=f"{tenor.value}  [{rep.num_agents} agents]",
            border_style="dim",
        ))
