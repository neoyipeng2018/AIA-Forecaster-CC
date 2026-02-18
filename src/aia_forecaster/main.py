"""CLI entry point for the AIA Forecaster."""

from __future__ import annotations

import argparse
import asyncio
import logging
import re
import sys
from datetime import date
from pathlib import Path

from rich.console import Console
from rich.panel import Panel

from aia_forecaster.calibration.platt import calibrate
from aia_forecaster.config import settings
from aia_forecaster.ensemble.engine import EnsembleEngine
from aia_forecaster.evaluation.metrics import brier_score
from aia_forecaster.fx.pairs import PAIR_CONFIGS
from aia_forecaster.fx.surface import (
    ProbabilitySurfaceGenerator,
    plot_surface,
    plot_surface_3d,
    plot_surface_scatter,
    print_surface,
)
from aia_forecaster.llm.client import LLMClient
from aia_forecaster.models import ForecastMode, ForecastQuestion, ForecastRun, Tenor
from aia_forecaster.storage.database import ForecastDatabase

console = Console()

# Shorthand detection: "forecast USDJPY 2025-02-13" → surface command
_SUBCOMMANDS = {"question", "surface", "evaluate", "list"}
_PAIR_PATTERN = re.compile(r"^[A-Z]{6}$")


def _is_pair_shorthand(argv: list[str]) -> bool:
    """Check if argv uses the PAIR [DATE] shorthand form."""
    if len(argv) < 2:
        return False
    first = argv[1]
    if first in _SUBCOMMANDS or first.startswith("-"):
        return False
    return bool(_PAIR_PATTERN.match(first.upper()))


def _rewrite_shorthand(argv: list[str]) -> list[str]:
    """Rewrite 'forecast USDJPY 2025-02-13 [flags]' into canonical form.

    Result: 'forecast --pair USDJPY --cutoff 2025-02-13 surface [flags]'
    """
    new_argv = [argv[0]]
    pair = argv[1].upper()

    if pair not in PAIR_CONFIGS:
        supported = ", ".join(sorted(PAIR_CONFIGS.keys()))
        print(f"Error: Unknown currency pair '{pair}'. Supported: {supported}", file=sys.stderr)
        sys.exit(1)

    new_argv.extend(["--pair", pair])

    rest_start = 2
    if len(argv) > 2 and not argv[2].startswith("-"):
        cutoff = argv[2]
        try:
            date.fromisoformat(cutoff)
        except ValueError:
            print(f"Error: Invalid date '{cutoff}'. Expected YYYY-MM-DD.", file=sys.stderr)
            sys.exit(1)
        new_argv.extend(["--cutoff", cutoff])
        rest_start = 3

    new_argv.append("surface")
    new_argv.extend(argv[rest_start:])
    return new_argv


async def run_question(args: argparse.Namespace) -> None:
    """Run a single binary question through the full pipeline."""
    question = ForecastQuestion(
        text=args.question,
        pair=args.pair,
        cutoff_date=date.fromisoformat(args.cutoff) if args.cutoff else date.today(),
    )

    console.print(Panel(f"[bold]{question.text}[/bold]\nPair: {question.pair} | Cutoff: {question.cutoff_date}"))

    llm = LLMClient(model=args.model) if args.model else LLMClient()
    engine = EnsembleEngine(llm=llm, num_agents=args.agents)

    # Run ensemble
    console.print("\n[bold]Stage 1 & 2: Ensemble + Supervisor[/bold]")
    ensemble_result = await engine.run(question)

    # Calibrate
    console.print("\n[bold]Stage 3: Platt Scaling Calibration[/bold]")
    cal = calibrate(ensemble_result.final_probability)

    # Build run record
    from datetime import datetime

    run = ForecastRun(
        question=question,
        ensemble=ensemble_result,
        calibrated=cal,
        completed_at=datetime.utcnow(),
    )

    # Save to database
    db = ForecastDatabase()
    run_id = db.save_run(run)

    # Print results
    console.print("\n")
    console.print(Panel.fit(
        f"[bold green]Results[/bold green]\n\n"
        f"Agents: {len(ensemble_result.agent_forecasts)}\n"
        f"Individual forecasts: {[f'{f.probability:.3f}' for f in ensemble_result.agent_forecasts]}\n"
        f"Mean probability: {ensemble_result.mean_probability:.4f}\n"
        f"Supervisor confidence: {ensemble_result.supervisor.confidence.value if ensemble_result.supervisor else 'N/A'}\n"
        f"Final (pre-calibration): {ensemble_result.final_probability:.4f}\n"
        f"[bold]Calibrated probability: {cal.calibrated_probability:.4f}[/bold]\n"
        f"\nRun ID: {run_id}",
        title="Forecast Complete",
    ))

    # Print detailed reasoning traces
    if args.verbose:
        console.print("\n[bold]Agent Reasoning Traces:[/bold]")
        for f in ensemble_result.agent_forecasts:
            console.print(f"\n[dim]{'=' * 60}[/dim]")
            console.print(
                f"[bold]Agent {f.agent_id}[/bold] (p={f.probability:.4f}, "
                f"{f.iterations} search iterations)"
            )
            if f.search_queries:
                console.print("\n[bold]Search queries:[/bold]")
                for i, q in enumerate(f.search_queries, 1):
                    console.print(f"  {i}. {q}")
            if f.evidence:
                console.print(f"\n[bold]Evidence ({len(f.evidence)} items):[/bold]")
                for i, e in enumerate(f.evidence[:5], 1):
                    console.print(f"  [{i}] {e.title}")
                    console.print(f"      {e.snippet[:200]}")
                    console.print(f"      [dim]{e.url}[/dim]")
            console.print("\n[bold]Reasoning:[/bold]")
            console.print(f.reasoning)

        if ensemble_result.supervisor:
            s = ensemble_result.supervisor
            console.print(f"\n[bold]{'=' * 60}[/bold]")
            console.print(
                f"[bold]Supervisor Reconciliation[/bold] "
                f"(confidence={s.confidence.value})"
            )
            console.print(f"\n{s.reasoning}")
            if s.additional_evidence:
                console.print(
                    f"\n[bold]Additional evidence ({len(s.additional_evidence)} items):[/bold]"
                )
                for i, e in enumerate(s.additional_evidence[:5], 1):
                    console.print(f"  [{i}] {e.title}: {e.snippet[:200]}")


async def run_surface(args: argparse.Namespace) -> None:
    """Generate a probability surface for a currency pair."""
    tenors = None
    if args.tenors:
        tenors = [Tenor(t.strip()) for t in args.tenors.split(",")]

    llm = LLMClient(model=args.model) if args.model else LLMClient()
    generator = ProbabilitySurfaceGenerator(llm=llm, num_agents=args.agents)

    cutoff = date.fromisoformat(args.cutoff) if args.cutoff else None

    forecast_mode = ForecastMode(getattr(args, "mode", "hitting"))

    surface = await generator.generate(
        pair=args.pair,
        num_strikes=args.strikes,
        tenors=tenors,
        cutoff_date=cutoff,
        forecast_mode=forecast_mode,
    )

    console.print("\n")
    print_surface(surface)

    # Show per-cell evidence and reasoning
    if getattr(args, "explain", False) or args.verbose:
        from aia_forecaster.fx.explanation import explain_surface, print_explanation

        explanation = explain_surface(surface)
        print_explanation(explanation)

    # Save heatmap
    output = args.output
    if not output:
        cutoff_str = (cutoff or date.today()).isoformat()
        mode_suffix = f"_{forecast_mode.value}" if forecast_mode != ForecastMode.HITTING else ""
        output = f"data/forecasts/{args.pair}_{cutoff_str}{mode_suffix}.png"
    path = plot_surface(surface, output)
    console.print(f"\n[bold]Heatmap saved:[/bold] {path}")

    # Save interactive 3D surface
    html_path = Path(str(path).replace(".png", ".html"))
    plot_surface_3d(surface, html_path)
    console.print(f"[bold]Interactive 3D surface saved:[/bold] {html_path}")

    # Save 2D scatter plots
    scatter_path = Path(str(path).replace(".png", "_scatter.png"))
    plot_surface_scatter(surface, scatter_path)
    console.print(f"[bold]Scatter plots saved:[/bold] {scatter_path}")

    # Save full surface data (including reasoning/evidence) as companion JSON
    json_path = Path(str(path).replace(".png", ".json"))
    json_path.write_text(surface.model_dump_json(indent=2))
    console.print(f"[bold]Surface JSON saved:[/bold] {json_path}")


async def run_evaluate(args: argparse.Namespace) -> None:
    """Evaluate a past forecast against its outcome."""
    db = ForecastDatabase()
    run = db.get_run(args.run_id)
    if run is None:
        console.print(f"[red]Run {args.run_id} not found[/red]")
        return

    if run.calibrated is None:
        console.print("[red]Run has no calibrated forecast[/red]")
        return

    outcome = args.outcome
    p = run.calibrated.calibrated_probability
    bs = brier_score([p], [outcome])

    console.print(Panel.fit(
        f"Question: {run.question.text}\n"
        f"Forecast: {p:.4f}\n"
        f"Outcome: {outcome}\n"
        f"[bold]Brier Score: {bs:.4f}[/bold]\n"
        f"  (0=perfect, 0.25=baseline, 1=worst)",
        title="Evaluation",
    ))


async def run_list(args: argparse.Namespace) -> None:
    """List recent forecast runs."""
    db = ForecastDatabase()
    runs = db.list_runs(limit=args.limit)

    if not runs:
        console.print("[dim]No forecast runs found.[/dim]")
        return

    from rich.table import Table

    table = Table(title="Recent Forecast Runs")
    table.add_column("ID", style="bold")
    table.add_column("Question")
    table.add_column("Pair")
    table.add_column("Raw P")
    table.add_column("Cal P")
    table.add_column("Agents")
    table.add_column("Date")

    for r in runs:
        table.add_row(
            r["id"],
            (r["question_text"][:50] + "...") if len(r["question_text"]) > 50 else r["question_text"],
            r["pair"],
            f"{r['raw_probability']:.4f}" if r["raw_probability"] else "—",
            f"{r['calibrated_probability']:.4f}" if r["calibrated_probability"] else "—",
            str(r["num_agents"] or "—"),
            r["started_at"][:19] if r["started_at"] else "—",
        )

    console.print(table)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="forecast",
        description="AIA Forecaster — FX probability forecasting",
        epilog=(
            "Quick start:\n"
            "  forecast USDJPY 2025-02-13          Generate probability surface\n"
            "  forecast EURUSD 2025-02-13 --strikes 7\n"
            "  forecast USDJPY                      Surface with today as cutoff\n"
            "\n"
            "Subcommands:\n"
            "  forecast question \"Will USD/JPY be above 155 in 1 week?\"\n"
            "  forecast evaluate RUN_ID 1\n"
            "  forecast list\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model", help="LLM model string (e.g., gpt-4o, openai/o3)")
    parser.add_argument("--agents", type=int, default=settings.num_agents, help="Number of forecasting agents")
    parser.add_argument("--pair", default=settings.default_pair, help="Currency pair (e.g., USDJPY)")
    parser.add_argument("--cutoff", help="Information cutoff date (YYYY-MM-DD)")
    parser.add_argument("-v", "--verbose", action="store_true", help="Show detailed reasoning traces")

    subparsers = parser.add_subparsers(dest="command", required=True)

    # question command
    q_parser = subparsers.add_parser("question", help="Forecast a single binary question")
    q_parser.add_argument("question", help="Binary question to forecast")
    q_parser.add_argument("--model", default=argparse.SUPPRESS, help=argparse.SUPPRESS)
    q_parser.add_argument("--agents", type=int, default=argparse.SUPPRESS, help=argparse.SUPPRESS)

    # surface command
    s_parser = subparsers.add_parser("surface", help="Generate probability surface")
    s_parser.add_argument("--strikes", type=int, default=5, help="Number of strikes")
    s_parser.add_argument("--tenors", help="Comma-separated tenors (e.g., 1W,1M)")
    s_parser.add_argument(
        "--mode", choices=["above", "hitting"], default="hitting",
        help="Forecast mode: 'hitting' (barrier touch, default) or 'above' (terminal distribution)",
    )
    s_parser.add_argument("-o", "--output", help="Output path for heatmap PNG (default: data/forecasts/PAIR_DATE.png)")
    s_parser.add_argument("-e", "--explain", action="store_true", help="Show per-cell evidence and reasoning")
    s_parser.add_argument("--model", default=argparse.SUPPRESS, help=argparse.SUPPRESS)
    s_parser.add_argument("--agents", type=int, default=argparse.SUPPRESS, help=argparse.SUPPRESS)

    # evaluate command
    e_parser = subparsers.add_parser("evaluate", help="Evaluate a past forecast")
    e_parser.add_argument("run_id", help="Forecast run ID")
    e_parser.add_argument("outcome", type=int, choices=[0, 1], help="Actual outcome (0 or 1)")

    # list command
    l_parser = subparsers.add_parser("list", help="List recent forecast runs")
    l_parser.add_argument("--limit", type=int, default=20, help="Max runs to show")

    return parser


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # Shorthand: "forecast USDJPY 2025-02-13" → "forecast --pair USDJPY --cutoff 2025-02-13 surface"
    if _is_pair_shorthand(sys.argv):
        sys.argv = _rewrite_shorthand(sys.argv)

    parser = build_parser()
    args = parser.parse_args()

    command_map = {
        "question": run_question,
        "surface": run_surface,
        "evaluate": run_evaluate,
        "list": run_list,
    }

    handler = command_map.get(args.command)
    if handler is None:
        parser.print_help()
        sys.exit(1)

    try:
        asyncio.run(handler(args))
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted.[/yellow]")
        sys.exit(130)


if __name__ == "__main__":
    main()
