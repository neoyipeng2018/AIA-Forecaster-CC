"""CLI entry point for the AIA Forecaster."""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from datetime import date

from rich.console import Console
from rich.panel import Panel

from aia_forecaster.calibration.platt import calibrate
from aia_forecaster.config import settings
from aia_forecaster.ensemble.engine import EnsembleEngine
from aia_forecaster.evaluation.metrics import brier_score
from aia_forecaster.fx.surface import ProbabilitySurfaceGenerator, print_surface
from aia_forecaster.llm.client import LLMClient
from aia_forecaster.models import ForecastQuestion, ForecastRun, Tenor
from aia_forecaster.storage.database import ForecastDatabase

console = Console()


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

    # Print top reasoning traces
    if args.verbose:
        console.print("\n[bold]Agent Reasoning Traces:[/bold]")
        for f in ensemble_result.agent_forecasts:
            console.print(f"\n[dim]--- Agent {f.agent_id} (p={f.probability:.4f}) ---[/dim]")
            console.print(f.reasoning[:500])


async def run_surface(args: argparse.Namespace) -> None:
    """Generate a probability surface for a currency pair."""
    tenors = None
    if args.tenors:
        tenors = [Tenor(t.strip()) for t in args.tenors.split(",")]

    llm = LLMClient(model=args.model) if args.model else LLMClient()
    generator = ProbabilitySurfaceGenerator(llm=llm, num_agents=args.agents)

    surface = await generator.generate(
        pair=args.pair,
        num_strikes=args.strikes,
        tenors=tenors,
        cutoff_date=date.fromisoformat(args.cutoff) if args.cutoff else None,
    )

    console.print("\n")
    print_surface(surface)


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
    )
    parser.add_argument("--model", help="LLM model string (litellm format)")
    parser.add_argument("--agents", type=int, default=settings.num_agents, help="Number of forecasting agents")
    parser.add_argument("--pair", default=settings.default_pair, help="Currency pair (e.g., USDJPY)")
    parser.add_argument("--cutoff", help="Information cutoff date (YYYY-MM-DD)")
    parser.add_argument("-v", "--verbose", action="store_true", help="Show detailed reasoning traces")

    subparsers = parser.add_subparsers(dest="command", required=True)

    # question command
    q_parser = subparsers.add_parser("question", help="Forecast a single binary question")
    q_parser.add_argument("question", help="Binary question to forecast")

    # surface command
    s_parser = subparsers.add_parser("surface", help="Generate probability surface")
    s_parser.add_argument("--strikes", type=int, default=5, help="Number of strikes")
    s_parser.add_argument("--tenors", help="Comma-separated tenors (e.g., 1W,1M)")

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
