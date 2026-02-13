"""Probability surface generation over (strike, tenor) grid."""

from __future__ import annotations

import logging
from datetime import date

from rich.console import Console
from rich.table import Table

from aia_forecaster.calibration.platt import calibrate
from aia_forecaster.ensemble.engine import EnsembleEngine
from aia_forecaster.fx.pairs import DEFAULT_TENORS, generate_strikes
from aia_forecaster.fx.rates import get_spot_rate
from aia_forecaster.llm.client import LLMClient
from aia_forecaster.models import (
    ForecastQuestion,
    ProbabilitySurface,
    SurfaceCell,
    Tenor,
)

logger = logging.getLogger(__name__)
console = Console()


def _question_text(pair: str, strike: float, tenor: Tenor) -> str:
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
    return f"Will {base}/{quote} be above {strike} in {horizon}?"


class ProbabilitySurfaceGenerator:
    """Generates the full probability surface for a currency pair."""

    def __init__(self, llm: LLMClient | None = None, num_agents: int | None = None):
        self.engine = EnsembleEngine(llm=llm, num_agents=num_agents)

    async def generate(
        self,
        pair: str = "USDJPY",
        num_strikes: int = 5,
        tenors: list[Tenor] | None = None,
        cutoff_date: date | None = None,
    ) -> ProbabilitySurface:
        """Generate the probability surface.

        Args:
            pair: Currency pair.
            num_strikes: Number of strikes around spot.
            tenors: List of tenors to evaluate. Default: all.
            cutoff_date: Information cutoff date.

        Returns:
            ProbabilitySurface with calibrated probabilities for each cell.
        """
        if tenors is None:
            tenors = DEFAULT_TENORS
        if cutoff_date is None:
            cutoff_date = date.today()

        # Get spot rate and generate strikes
        spot = await get_spot_rate(pair)
        strikes = generate_strikes(spot, pair, num_strikes)
        console.print(f"[bold]Spot rate:[/bold] {pair} = {spot}")
        console.print(f"[bold]Strikes:[/bold] {strikes}")
        console.print(f"[bold]Tenors:[/bold] {[t.value for t in tenors]}")

        total_cells = len(strikes) * len(tenors)
        console.print(f"\n[bold]Generating {total_cells} forecasts...[/bold]\n")

        surface = ProbabilitySurface(pair=pair, spot_rate=spot)

        for tenor in tenors:
            for strike in strikes:
                q_text = _question_text(pair, strike, tenor)
                console.print(f"  [dim]{q_text}[/dim]")

                question = ForecastQuestion(
                    text=q_text,
                    pair=pair,
                    strike=strike,
                    tenor=tenor,
                    cutoff_date=cutoff_date,
                )

                try:
                    ensemble_result = await self.engine.run(question)
                    cal = calibrate(ensemble_result.final_probability)

                    cell = SurfaceCell(
                        strike=strike,
                        tenor=tenor,
                        question=q_text,
                        calibrated=cal,
                    )
                    surface.cells.append(cell)

                    console.print(
                        f"    → raw={cal.raw_probability:.4f}, "
                        f"calibrated={cal.calibrated_probability:.4f}"
                    )
                except Exception:
                    logger.exception("Failed to forecast: %s", q_text)
                    surface.cells.append(
                        SurfaceCell(strike=strike, tenor=tenor, question=q_text)
                    )

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

    table = Table(title=f"{surface.pair} Probability Surface (spot={surface.spot_rate})")
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
