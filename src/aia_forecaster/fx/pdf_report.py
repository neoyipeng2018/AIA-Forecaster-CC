"""PDF report generation for probability surface forecasts.

Produces a multi-page PDF combining:
  - Title page with key metadata (pair, spot, regime, date)
  - Causal factors narrative
  - Embedded chart images (heatmap, scatter, CDF)
  - Per-cell probability table with tenor catalysts and evidence
"""

from __future__ import annotations

import logging
from pathlib import Path

from fpdf import FPDF

from aia_forecaster.fx.explanation import explain_surface
from aia_forecaster.models import (
    CausalFactor,
    ForecastMode,
    ProbabilitySurface,
    Tenor,
)

logger = logging.getLogger(__name__)


def _sanitize(text: str) -> str:
    """Replace Unicode characters unsupported by built-in PDF fonts."""
    replacements = {
        "\u2013": "-",   # en-dash
        "\u2014": "--",  # em-dash
        "\u2018": "'",   # left single quote
        "\u2019": "'",   # right single quote
        "\u201c": '"',   # left double quote
        "\u201d": '"',   # right double quote
        "\u2026": "...", # ellipsis
        "\u2022": "-",   # bullet
        "\u00a0": " ",   # non-breaking space
        "\u200b": "",    # zero-width space
    }
    for char, replacement in replacements.items():
        text = text.replace(char, replacement)
    # Catch any remaining non-latin-1 characters
    return text.encode("latin-1", errors="replace").decode("latin-1")


class _ReportPDF(FPDF):
    """PDF subclass with branded header/footer."""

    def __init__(self, pair: str, date_str: str) -> None:
        super().__init__(orientation="P", unit="mm", format="A4")
        self._pair = pair
        self._date_str = date_str
        self.set_auto_page_break(auto=True, margin=20)

    def normalize_text(self, text: str) -> str:
        """Sanitize Unicode before passing to the PDF engine."""
        return super().normalize_text(_sanitize(text))

    def header(self) -> None:
        if self.page_no() == 1:
            return  # title page has its own header
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(120, 120, 120)
        base, quote = self._pair[:3], self._pair[3:]
        self.cell(
            0, 6,
            f"{base}/{quote} Probability Surface — {self._date_str}",
            new_x="LMARGIN", new_y="NEXT", align="L",
        )
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(4)

    def footer(self) -> None:
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(150, 150, 150)
        self.cell(0, 10, f"Page {self.page_no()}/{{nb}}", align="C")


def _add_title_page(
    pdf: _ReportPDF,
    surface: ProbabilitySurface,
) -> None:
    """Render the title page."""
    pdf.add_page()
    base, quote = surface.pair[:3], surface.pair[3:]
    date_str = surface.generated_at.strftime("%Y-%m-%d %H:%M UTC")
    is_hitting = surface.forecast_mode == ForecastMode.HITTING
    mode_label = "Barrier / Touch" if is_hitting else "Above Strike (Terminal)"

    # Title block
    pdf.ln(40)
    pdf.set_font("Helvetica", "B", 28)
    pdf.set_text_color(30, 30, 30)
    pdf.cell(0, 14, f"{base}/{quote}", new_x="LMARGIN", new_y="NEXT", align="C")

    pdf.set_font("Helvetica", "", 16)
    pdf.set_text_color(80, 80, 80)
    pdf.cell(0, 10, "Probability Surface Report", new_x="LMARGIN", new_y="NEXT", align="C")

    pdf.ln(12)
    pdf.set_draw_color(200, 200, 200)
    pdf.line(50, pdf.get_y(), 160, pdf.get_y())
    pdf.ln(12)

    # Metadata table
    pdf.set_font("Helvetica", "", 11)
    pdf.set_text_color(60, 60, 60)
    meta = [
        ("Spot Rate", f"{surface.spot_rate:.4f}"),
        ("Forecast Mode", mode_label),
        ("Generated", date_str),
    ]
    if surface.regime:
        channels = ", ".join(surface.regime_dominant_channels) if surface.regime_dominant_channels else "—"
        meta.append(("Regime", f"{surface.regime} (channels: {channels})"))
    if surface.source_config:
        meta.append(("Data Sources", surface.source_config.label))

    strikes = sorted(set(c.strike for c in surface.cells))
    tenors = sorted(set(c.tenor for c in surface.cells), key=lambda t: t.days)
    meta.append(("Strikes", f"{len(strikes)} ({strikes[0]:.2f} – {strikes[-1]:.2f})"))
    meta.append(("Tenors", ", ".join(t.value for t in tenors)))
    meta.append(("Grid Size", f"{len(strikes)} x {len(tenors)} = {len(surface.cells)} cells"))

    for label, value in meta:
        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(50, 8, f"{label}:", align="R")
        pdf.set_font("Helvetica", "", 11)
        pdf.cell(0, 8, f"  {value}", new_x="LMARGIN", new_y="NEXT")


def _add_causal_factors_page(
    pdf: _ReportPDF,
    surface: ProbabilitySurface,
) -> None:
    """Render the causal factors narrative page."""
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.set_text_color(30, 30, 30)
    pdf.cell(0, 10, "Causal Factors & Macro Regime", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)

    if surface.regime:
        pdf.set_font("Helvetica", "B", 11)
        pdf.set_text_color(50, 50, 50)
        pdf.cell(30, 7, "Regime:")
        pdf.set_font("Helvetica", "", 11)
        channels = ", ".join(surface.regime_dominant_channels) if surface.regime_dominant_channels else "—"
        pdf.cell(0, 7, f"{surface.regime}  (dominant channels: {channels})", new_x="LMARGIN", new_y="NEXT")
        pdf.ln(4)

    if not surface.causal_factors:
        pdf.set_font("Helvetica", "I", 10)
        pdf.set_text_color(120, 120, 120)
        pdf.cell(0, 8, "No causal factors identified.", new_x="LMARGIN", new_y="NEXT")
        return

    for i, cf in enumerate(surface.causal_factors):
        _render_causal_factor(pdf, cf, i + 1)


def _render_causal_factor(pdf: _ReportPDF, cf: CausalFactor, num: int) -> None:
    """Render a single causal factor block."""
    # Direction indicator
    direction = cf.direction.lower()
    if direction == "bullish":
        icon, r, g, b = "+", 34, 139, 34
    elif direction == "contested":
        icon, r, g, b = "~", 180, 130, 0
    else:
        icon, r, g, b = "-", 200, 40, 40

    pdf.set_font("Helvetica", "B", 10)
    pdf.set_text_color(r, g, b)
    pdf.cell(8, 7, icon)
    pdf.set_text_color(30, 30, 30)

    # Event text — wrap to fit page
    pdf.set_font("Helvetica", "B", 10)
    x_start = pdf.get_x()
    pdf.multi_cell(0, 6, cf.event, new_x="LMARGIN", new_y="NEXT")

    # Detail line
    pdf.set_x(18)
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(100, 100, 100)
    detail = f"{cf.channel}  |  {cf.direction}  |  magnitude: {cf.magnitude}  |  confidence: {cf.confidence}"
    pdf.multi_cell(0, 5, detail, new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)


def _add_chart_page(
    pdf: _ReportPDF,
    image_path: Path,
    title: str,
) -> None:
    """Embed a chart image as a full-width page."""
    if not image_path.exists():
        logger.warning("Chart not found, skipping: %s", image_path)
        return

    pdf.add_page()
    pdf.set_font("Helvetica", "B", 14)
    pdf.set_text_color(30, 30, 30)
    pdf.cell(0, 10, title, new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.ln(4)

    # Fit image to page width (190mm usable) while maintaining aspect ratio
    usable_w = 190
    max_h = 220  # leave room for title + footer
    try:
        pdf.image(str(image_path), x=10, w=usable_w)
    except Exception as e:
        logger.error("Failed to embed chart %s: %s", image_path, e)
        pdf.set_font("Helvetica", "I", 10)
        pdf.set_text_color(180, 40, 40)
        pdf.cell(0, 8, f"[Chart could not be embedded: {e}]", new_x="LMARGIN", new_y="NEXT")


def _add_probability_table(
    pdf: _ReportPDF,
    surface: ProbabilitySurface,
) -> None:
    """Render the probability grid as a table."""
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 14)
    pdf.set_text_color(30, 30, 30)

    is_hitting = surface.forecast_mode == ForecastMode.HITTING
    p_label = "P(touch)" if is_hitting else "P(above)"
    pdf.cell(0, 10, f"Probability Grid — {p_label}", new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.ln(4)

    strikes = sorted(set(c.strike for c in surface.cells))
    tenors = sorted(set(c.tenor for c in surface.cells), key=lambda t: t.days)

    # Build lookup
    lookup: dict[tuple[float, str], tuple[float | None, float | None]] = {}
    for c in surface.cells:
        cal = c.calibrated.calibrated_probability if c.calibrated else None
        raw = c.calibrated.raw_probability if c.calibrated else None
        lookup[(c.strike, c.tenor.value)] = (cal, raw)

    # Table dimensions
    n_cols = len(tenors) + 1  # strike col + tenor cols
    col_w_strike = 22
    col_w_tenor = min(28, (190 - col_w_strike) / len(tenors))

    # Header row
    pdf.set_font("Helvetica", "B", 9)
    pdf.set_fill_color(240, 240, 240)
    pdf.set_text_color(40, 40, 40)
    pdf.cell(col_w_strike, 8, "Strike", border=1, fill=True, align="C")
    for tenor in tenors:
        pdf.cell(col_w_tenor, 8, tenor.value, border=1, fill=True, align="C")
    pdf.ln()

    # Data rows
    pdf.set_font("Helvetica", "", 9)
    for strike in strikes:
        # Highlight row near spot
        is_near_spot = (
            surface.spot_rate is not None
            and abs(strike - surface.spot_rate) < (strikes[1] - strikes[0]) * 0.5
            if len(strikes) > 1 else False
        )
        if is_near_spot:
            pdf.set_fill_color(230, 240, 255)
        else:
            pdf.set_fill_color(255, 255, 255)

        pdf.set_font("Helvetica", "B", 9)
        pdf.set_text_color(40, 40, 40)
        pdf.cell(col_w_strike, 7, f"{strike:.2f}", border=1, fill=True, align="C")

        pdf.set_font("Helvetica", "", 9)
        for tenor in tenors:
            cal, raw = lookup.get((strike, tenor.value), (None, None))
            if cal is not None:
                # Color: green for high, red for low
                if cal >= 0.6:
                    pdf.set_text_color(34, 120, 34)
                elif cal <= 0.4:
                    pdf.set_text_color(180, 40, 40)
                else:
                    pdf.set_text_color(60, 60, 60)
                pdf.cell(col_w_tenor, 7, f"{cal:.3f}", border=1, fill=True, align="C")
            else:
                pdf.set_text_color(150, 150, 150)
                pdf.cell(col_w_tenor, 7, "—", border=1, fill=True, align="C")
        pdf.ln()

    pdf.set_text_color(40, 40, 40)

    # Legend
    pdf.ln(4)
    pdf.set_font("Helvetica", "I", 8)
    pdf.set_text_color(120, 120, 120)
    if surface.spot_rate is not None:
        pdf.cell(0, 5, f"Blue-highlighted row: nearest to spot ({surface.spot_rate:.4f})", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 5, "Green = high probability (>0.6), Red = low probability (<0.4)", new_x="LMARGIN", new_y="NEXT")
    alpha = surface.cells[0].calibrated.alpha if surface.cells and surface.cells[0].calibrated else 1.73
    pdf.cell(
        0, 5,
        f"Calibrated via Platt scaling (alpha = {alpha:.2f}). Values shown are post-calibration.",
        new_x="LMARGIN", new_y="NEXT",
    )


def _add_narrative_pages(
    pdf: _ReportPDF,
    surface: ProbabilitySurface,
) -> None:
    """Render per-tenor narrative: catalysts, evidence, and strike probabilities."""
    explanation = explain_surface(surface)

    pdf.add_page()
    pdf.set_font("Helvetica", "B", 14)
    pdf.set_text_color(30, 30, 30)
    pdf.cell(0, 10, "Tenor-by-Tenor Analysis", new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.ln(2)

    is_hitting = surface.forecast_mode == ForecastMode.HITTING
    p_verb = "touch" if is_hitting else "above"

    # Group cells by tenor (preserving tenor sort order)
    tenor_order: list[Tenor] = []
    tenor_cells: dict[str, list] = {}
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
        # Sort cells by strike
        cells.sort(key=lambda c: c.strike)

        # Pick representative cell for tenor-level data (richest consensus/evidence)
        rep = max(cells, key=lambda c: (
            len(c.consensus_summary),
            len(c.top_evidence),
            len(c.causal_factors),
        ))

        # Check if we need a new page
        if pdf.get_y() > 200:
            pdf.add_page()

        # Tenor header
        pdf.set_draw_color(60, 60, 60)
        pdf.line(10, pdf.get_y(), 200, pdf.get_y())
        pdf.ln(2)

        pdf.set_font("Helvetica", "B", 12)
        pdf.set_text_color(30, 30, 30)
        pdf.cell(0, 8, f"Tenor: {tenor.value}   [{rep.num_agents} agents]", new_x="LMARGIN", new_y="NEXT")
        pdf.ln(1)

        # Compact strike probability listing
        pdf.set_font("Helvetica", "B", 9)
        pdf.set_text_color(50, 50, 50)
        pdf.cell(0, 6, "Strike Probabilities:", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "", 9)
        for c in cells:
            raw_str = f"  (raw: {c.raw_probability:.3f})" if c.raw_probability is not None else ""
            # Color by probability
            if c.calibrated_probability >= 0.6:
                pdf.set_text_color(34, 120, 34)
            elif c.calibrated_probability <= 0.4:
                pdf.set_text_color(180, 40, 40)
            else:
                pdf.set_text_color(60, 60, 60)
            pdf.set_x(14)
            pdf.cell(
                0, 5.5,
                f"P({p_verb} {c.strike:.2f}) = {c.calibrated_probability:.3f}{raw_str}",
                new_x="LMARGIN", new_y="NEXT",
            )
        pdf.set_text_color(60, 60, 60)
        pdf.ln(1)

        # Tenor-specific causal factors
        if rep.causal_factors:
            pdf.set_font("Helvetica", "B", 9)
            pdf.set_text_color(26, 115, 232)
            pdf.cell(0, 6, f"Tenor Causal Factors ({tenor.value}):", new_x="LMARGIN", new_y="NEXT")
            for i, cf in enumerate(rep.causal_factors[:5], 1):
                _render_causal_factor(pdf, cf, i)
            if rep.tenor_relevance:
                pdf.set_x(14)
                pdf.set_font("Helvetica", "I", 7.5)
                pdf.set_text_color(100, 100, 120)
                pdf.multi_cell(0, 4, rep.tenor_relevance[:300], new_x="LMARGIN", new_y="NEXT")
            pdf.set_text_color(60, 60, 60)
        elif rep.tenor_catalysts:
            # Fallback for old data with plain-string catalysts
            pdf.set_font("Helvetica", "B", 9)
            pdf.set_text_color(26, 115, 232)
            pdf.cell(0, 6, f"Tenor Catalysts ({tenor.value}):", new_x="LMARGIN", new_y="NEXT")
            pdf.set_font("Helvetica", "", 8)
            pdf.set_text_color(60, 60, 80)
            for i, cat in enumerate(rep.tenor_catalysts[:5], 1):
                pdf.set_x(14)
                pdf.multi_cell(0, 4.5, f"{i}. {cat}", new_x="LMARGIN", new_y="NEXT")
            if rep.tenor_relevance:
                pdf.set_x(14)
                pdf.set_font("Helvetica", "I", 7.5)
                pdf.set_text_color(100, 100, 120)
                pdf.multi_cell(0, 4, rep.tenor_relevance[:300], new_x="LMARGIN", new_y="NEXT")
            pdf.set_text_color(60, 60, 60)

        # Top evidence — merge across all cells at this tenor for completeness
        all_evidence: dict[str, object] = {}
        for c in cells:
            for ev in c.top_evidence:
                norm_url = ev.url.rstrip("/").lower()
                if norm_url not in all_evidence or ev.cited_by_agents > all_evidence[norm_url].cited_by_agents:
                    all_evidence[norm_url] = ev
        merged_evidence = sorted(all_evidence.values(), key=lambda e: -e.cited_by_agents)[:5]

        if merged_evidence:
            pdf.set_font("Helvetica", "B", 9)
            pdf.set_text_color(50, 50, 50)
            pdf.cell(0, 6, f"Evidence ({len(merged_evidence)} sources):", new_x="LMARGIN", new_y="NEXT")
            pdf.set_font("Helvetica", "", 8)
            pdf.set_text_color(80, 80, 80)
            for ev in merged_evidence:
                cited = f" [{ev.cited_by_agents} agents]" if ev.cited_by_agents > 1 else ""
                pdf.set_x(14)
                title_line = f"- {ev.title}{cited}"
                pdf.multi_cell(0, 4.5, title_line, new_x="LMARGIN", new_y="NEXT")
                if ev.snippet:
                    pdf.set_x(18)
                    pdf.set_font("Helvetica", "I", 7.5)
                    pdf.set_text_color(110, 110, 110)
                    snippet = ev.snippet[:200] + ("..." if len(ev.snippet) > 200 else "")
                    pdf.multi_cell(0, 4, snippet, new_x="LMARGIN", new_y="NEXT")
                if ev.url:
                    pdf.set_x(18)
                    pdf.set_font("Helvetica", "U", 7)
                    pdf.set_text_color(26, 115, 232)
                    pdf.cell(0, 4, ev.url[:120], new_x="LMARGIN", new_y="NEXT", link=ev.url)
                pdf.set_font("Helvetica", "", 8)
                pdf.set_text_color(80, 80, 80)

        # Disagreements — show if any cell at this tenor had notable disagreement
        disagreements = [c.disagreement_notes for c in cells if c.disagreement_notes]
        if disagreements:
            # Pick the most detailed disagreement note
            best_disagreement = max(disagreements, key=len)
            pdf.set_font("Helvetica", "B", 9)
            pdf.set_text_color(180, 100, 30)
            pdf.cell(28, 6, "Disagreement:")
            pdf.set_font("Helvetica", "", 8)
            pdf.set_text_color(100, 80, 50)
            pdf.multi_cell(0, 4.5, best_disagreement, new_x="LMARGIN", new_y="NEXT")

        pdf.ln(4)


def generate_pdf_report(
    surface: ProbabilitySurface,
    output_path: str | Path,
    heatmap_path: Path | None = None,
    scatter_path: Path | None = None,
    cdf_path: Path | None = None,
) -> Path:
    """Generate a full PDF report for a probability surface forecast.

    Args:
        surface: The probability surface data.
        output_path: File path for the saved PDF.
        heatmap_path: Path to the heatmap PNG (optional).
        scatter_path: Path to the scatter plots PNG (optional).
        cdf_path: Path to the CDF chart PNG (optional).

    Returns:
        The resolved output path.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    date_str = surface.generated_at.strftime("%Y-%m-%d")
    pdf = _ReportPDF(surface.pair, date_str)
    pdf.alias_nb_pages()

    # Page 1: Title
    _add_title_page(pdf, surface)

    # Page 2: Causal factors narrative
    _add_causal_factors_page(pdf, surface)

    # Page 3: Probability grid table
    _add_probability_table(pdf, surface)

    # Pages 4+: Charts
    if heatmap_path and heatmap_path.exists():
        _add_chart_page(pdf, heatmap_path, "Probability Heatmap")
    if scatter_path and scatter_path.exists():
        _add_chart_page(pdf, scatter_path, "Scatter Analysis")
    if cdf_path and cdf_path.exists():
        _add_chart_page(pdf, cdf_path, "CDF — P(spot < K)")

    # Narrative pages: per-cell analysis
    _add_narrative_pages(pdf, surface)

    pdf.output(str(output_path))
    logger.info("PDF report saved: %s", output_path)

    return output_path
