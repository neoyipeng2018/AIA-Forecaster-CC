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
from aia_forecaster.models import ForecastMode, ForecastQuestion, ForecastRun, SourceConfig, Tenor
from aia_forecaster.storage.database import ForecastDatabase

console = Console()

# Shorthand detection: "forecast USDJPY 2025-02-13" → surface command
_SUBCOMMANDS = {"question", "surface", "evaluate", "list", "compare"}
_PAIR_PATTERN = re.compile(r"^[A-Z]{6}$")

# Mapping from short CLI names to registry source names
_SOURCE_ALIASES: dict[str, str] = {
    "rss": "rss",
    "bis": "bis_speeches",
    "web": "web",  # sentinel — handled separately
}


def _parse_source_config(sources_str: str) -> SourceConfig:
    """Parse a comma-separated --sources flag into a SourceConfig.

    Accepted tokens: rss, bis, web (case-insensitive).
    """
    tokens = [t.strip().lower() for t in sources_str.split(",") if t.strip()]
    registry_sources: list[str] = []
    web_search_enabled = False
    for tok in tokens:
        if tok not in _SOURCE_ALIASES:
            supported = ", ".join(sorted(_SOURCE_ALIASES.keys()))
            print(f"Error: Unknown source '{tok}'. Supported: {supported}", file=sys.stderr)
            sys.exit(1)
        if tok == "web":
            web_search_enabled = True
        else:
            registry_sources.append(_SOURCE_ALIASES[tok])
    return SourceConfig(
        registry_sources=registry_sources,
        web_search_enabled=web_search_enabled,
    )


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

    # Parse --sources flag into SourceConfig (None = all sources / default)
    source_config: SourceConfig | None = None
    sources_arg = getattr(args, "sources", None)
    if sources_arg:
        source_config = _parse_source_config(sources_arg)
        console.print(
            f"[bold]Source config:[/bold] {source_config.label} "
            f"(mode={source_config.get_search_mode().value})"
        )

    llm = LLMClient(model=args.model) if args.model else LLMClient()
    generator = ProbabilitySurfaceGenerator(
        llm=llm, num_agents=args.agents, source_config=source_config,
    )

    cutoff = date.fromisoformat(args.cutoff) if args.cutoff else None

    forecast_mode = ForecastMode(getattr(args, "mode", "hitting"))

    # Parse custom strikes if provided
    custom_strikes: list[float] | None = None
    strike_list_arg = getattr(args, "strike_list", None)
    if strike_list_arg:
        custom_strikes = [float(s.strip()) for s in strike_list_arg.split(",")]

    strike_step: float | None = getattr(args, "strike_step", None)

    surface = await generator.generate(
        pair=args.pair,
        num_strikes=args.strikes,
        tenors=tenors,
        cutoff_date=cutoff,
        forecast_mode=forecast_mode,
        strike_step=strike_step,
        custom_strikes=custom_strikes,
    )

    console.print("\n")
    print_surface(surface)

    # Show per-cell evidence and reasoning
    if getattr(args, "explain", False) or args.verbose:
        from aia_forecaster.fx.explanation import explain_surface, print_explanation

        explanation = explain_surface(surface)
        print_explanation(explanation)

    # Build output filename with source label suffix
    output = args.output
    if not output:
        cutoff_str = (cutoff or date.today()).isoformat()
        mode_suffix = f"_{forecast_mode.value}" if forecast_mode != ForecastMode.HITTING else ""
        source_suffix = f"_{source_config.label}" if source_config else ""
        output = f"data/forecasts/{args.pair}_{cutoff_str}{mode_suffix}{source_suffix}.png"
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


async def run_compare(args: argparse.Namespace) -> None:
    """Compare 2+ saved probability surface JSON files."""
    from aia_forecaster.fx.compare import compare_surfaces

    paths = [Path(p) for p in args.files]
    for p in paths:
        if not p.exists():
            console.print(f"[red]File not found: {p}[/red]")
            return
        if p.suffix != ".json":
            console.print(f"[red]Expected .json file: {p}[/red]")
            return

    output_dir = Path(args.output_dir) if args.output_dir else paths[0].parent
    result = compare_surfaces(paths, output_dir)

    console.print(f"\n[bold green]Comparison complete![/bold green]")
    for name, path in result.items():
        console.print(f"  [bold]{name}:[/bold] {path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="forecast",
        description="AIA Forecaster — FX probability forecasting",
        epilog=(
            "Quick start:\n"
            "  forecast USDJPY 2025-02-13              Generate probability surface\n"
            "  forecast EURUSD 2025-02-13 --strikes 7   7 strikes, default step\n"
            "  forecast USDJPY --strike-step 0.5         Half-yen strike intervals\n"
            "  forecast USDJPY --strike-list 150,152.5,155,157.5,160\n"
            "  forecast USDJPY --tenors 1D,3D,5D,2W,1M   Flexible tenors (<N><D|W|M|Y>)\n"
            "  forecast USDJPY --sources rss             Only RSS feeds\n"
            "  forecast USDJPY --sources rss,web         RSS + web search (no BIS)\n"
            "\n"
            "Subcommands:\n"
            "  forecast question \"Will USD/JPY be above 155 in 1 week?\"\n"
            "  forecast compare file1.json file2.json\n"
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
    s_parser.add_argument("--strikes", type=int, default=5, help="Number of auto-generated strikes (default: 5)")
    s_parser.add_argument(
        "--strike-step", type=float, default=None,
        help="Interval between auto-generated strikes (e.g., 0.5 for USDJPY half-yen steps). "
             "Overrides pair-specific defaults.",
    )
    s_parser.add_argument(
        "--strike-list",
        help="Comma-separated explicit strike prices (e.g., 150.0,152.5,155.0,157.5,160.0). "
             "Overrides --strikes and --strike-step.",
    )
    s_parser.add_argument(
        "--tenors",
        help="Comma-separated tenors — any <number><unit> accepted "
             "(D=days, W=weeks, M=months, Y=years). "
             "Examples: 1D,3D,5D,2W,1M,3M,6M,1Y",
    )
    s_parser.add_argument(
        "--mode", choices=["above", "hitting"], default="above",
        help="Forecast mode: 'above' (terminal distribution, default) or 'hitting' (barrier touch)",
    )
    s_parser.add_argument(
        "--sources",
        help="Comma-separated data sources to enable (from: rss, bis, web). Default: all",
    )
    s_parser.add_argument("-o", "--output", help="Output path for heatmap PNG (default: data/forecasts/PAIR_DATE.png)")
    s_parser.add_argument("-e", "--explain", action="store_true", help="Show per-cell evidence and reasoning")
    s_parser.add_argument("--model", default=argparse.SUPPRESS, help=argparse.SUPPRESS)
    s_parser.add_argument("--agents", type=int, default=argparse.SUPPRESS, help=argparse.SUPPRESS)

    # compare command
    c_parser = subparsers.add_parser("compare", help="Compare 2+ saved probability surfaces")
    c_parser.add_argument("files", nargs="+", help="JSON surface files to compare")
    c_parser.add_argument(
        "--output-dir", help="Directory for comparison outputs (default: same as first file)",
    )

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
        "compare": run_compare,
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
