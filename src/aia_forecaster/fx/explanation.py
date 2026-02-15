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
    CellExplanation,
    Confidence,
    EvidenceItem,
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


def _summarize_consensus(agents: list[AgentForecast]) -> str:
    """Identify majority direction and extract key reasoning themes.

    Heuristic: find majority direction (above/below 0.5), then take the
    first sentence from each agent whose probability aligns with the majority.
    Combine into a concise summary.
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

    if not sentences:
        return f"Agents lean {direction} 0.5 (mean={mean_p:.3f})."

    return f"Agents lean {direction} 0.5 (mean={mean_p:.3f}). " + " ".join(sentences)


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
    )


def explain_surface(surface: ProbabilitySurface) -> SurfaceExplanation:
    """Generate a full SurfaceExplanation from a ProbabilitySurface."""
    return SurfaceExplanation(
        pair=surface.pair,
        spot_rate=surface.spot_rate,
        generated_at=surface.generated_at,
        cells=[explain_cell(cell) for cell in surface.cells],
    )


# ---------------------------------------------------------------------------
# Rich CLI output
# ---------------------------------------------------------------------------


def print_explanation(explanation: SurfaceExplanation) -> None:
    """Print per-cell explanations using Rich formatting."""
    console.print(f"\n[bold]Evidence & Reasoning — {explanation.pair}[/bold]\n")

    for cell in explanation.cells:
        if cell.calibrated_probability is None:
            continue

        # Build content
        lines: list[str] = []

        # Header
        lines.append(
            f"[bold]P(above {cell.strike:.2f}) @ {cell.tenor.value}[/bold]: "
            f"{cell.calibrated_probability:.3f}  "
            f"(raw={cell.raw_probability:.3f}, agents={cell.num_agents})"
        )

        # Consensus
        if cell.consensus_summary:
            lines.append(f"\n[bold]Consensus:[/bold] {cell.consensus_summary}")

        # Top evidence
        if cell.top_evidence:
            lines.append(f"\n[bold]Top Evidence ({len(cell.top_evidence)} items):[/bold]")
            for i, e in enumerate(cell.top_evidence[:3], 1):
                lines.append(
                    f"  [{i}] {e.title} "
                    f"[dim](cited by {e.cited_by_agents} agent{'s' if e.cited_by_agents > 1 else ''})[/dim]"
                )
                lines.append(f"      {e.snippet[:200]}")
                lines.append(f"      [dim]{e.url}[/dim]")

        # Disagreements
        if cell.disagreement_notes:
            lines.append(f"\n[bold]Disagreements:[/bold] {cell.disagreement_notes}")

        # Supervisor
        if cell.supervisor_reasoning:
            lines.append(
                f"\n[dim]Supervisor ({cell.supervisor_confidence}): "
                f"{cell.supervisor_reasoning[:300]}[/dim]"
            )

        console.print(Panel(
            "\n".join(lines),
            title=f"{cell.strike:.2f} / {cell.tenor.value}",
            border_style="dim",
        ))
