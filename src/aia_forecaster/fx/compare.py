"""Comparison visualizations for probability surfaces generated with different source configs.

Provides side-by-side heatmaps, overlay scatter plots, and interactive Plotly HTML
to help users understand which data sources drive forecast differences.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from aia_forecaster.models import ProbabilitySurface, Tenor

logger = logging.getLogger(__name__)


@dataclass
class LabeledSurface:
    """A loaded probability surface with a display label."""

    label: str
    surface: ProbabilitySurface
    # Convenience lookups populated on load
    strikes: list[float] = field(default_factory=list)
    tenors: list[Tenor] = field(default_factory=list)
    grid: dict[tuple[float, Tenor], float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_surfaces(paths: list[Path]) -> list[LabeledSurface]:
    """Load JSON files into LabeledSurface objects.

    Label is derived from source_config.label if present, else the filename stem.
    """
    surfaces: list[LabeledSurface] = []
    for p in paths:
        data = json.loads(p.read_text())
        surface = ProbabilitySurface.model_validate(data)

        # Derive label
        if surface.source_config is not None:
            label = surface.source_config.label
        else:
            label = p.stem

        # Build lookup grid
        strikes = sorted({c.strike for c in surface.cells})
        tenors = sorted(
            {c.tenor for c in surface.cells},
            key=lambda t: t.days,
        )
        grid: dict[tuple[float, Tenor], float] = {}
        for c in surface.cells:
            p_val = c.calibrated.calibrated_probability if c.calibrated else None
            if p_val is not None:
                grid[(c.strike, c.tenor)] = p_val

        surfaces.append(LabeledSurface(
            label=label,
            surface=surface,
            strikes=strikes,
            tenors=tenors,
            grid=grid,
        ))

    return surfaces


# ---------------------------------------------------------------------------
# Grid alignment
# ---------------------------------------------------------------------------


def _align_grids(
    surfaces: list[LabeledSurface],
) -> tuple[list[float], list[Tenor]]:
    """Find the intersection of strikes and tenors across all surfaces."""
    strike_sets = [set(s.strikes) for s in surfaces]
    tenor_sets = [set(s.tenors) for s in surfaces]

    common_strikes = sorted(strike_sets[0].intersection(*strike_sets[1:]))
    common_tenors_set = tenor_sets[0].intersection(*tenor_sets[1:])
    common_tenors = sorted(common_tenors_set, key=lambda t: t.days)

    if not common_strikes or not common_tenors:
        logger.warning(
            "No overlapping strikes/tenors — falling back to union. "
            "Missing cells will show as NaN."
        )
        common_strikes = sorted(set().union(*strike_sets))
        common_tenors_set = set().union(*tenor_sets)
        common_tenors = sorted(common_tenors_set, key=lambda t: t.days)

    return common_strikes, common_tenors


def _surface_to_array(
    ls: LabeledSurface,
    strikes: list[float],
    tenors: list[Tenor],
) -> np.ndarray:
    """Convert a LabeledSurface's grid to a 2D numpy array [strike x tenor]."""
    arr = np.full((len(strikes), len(tenors)), np.nan)
    for i, strike in enumerate(strikes):
        for j, tenor in enumerate(tenors):
            val = ls.grid.get((strike, tenor))
            if val is not None:
                arr[i, j] = val
    return arr


# ---------------------------------------------------------------------------
# Static visualizations (matplotlib)
# ---------------------------------------------------------------------------


def plot_diff_heatmaps(
    surfaces: list[LabeledSurface],
    path: Path,
) -> Path:
    """Side-by-side heatmaps (RdYlGn) + difference heatmap (RdBu diverging).

    For N surfaces: N individual heatmaps + N*(N-1)/2 difference heatmaps.
    When N=2, this is 2 heatmaps + 1 diff = 3 panels.
    """
    strikes, tenors = _align_grids(surfaces)
    strikes_desc = list(reversed(strikes))  # Higher strikes at top
    tenor_labels = [t.value for t in tenors]

    arrays = []
    for ls in surfaces:
        arr = _surface_to_array(ls, strikes, tenors)
        arrays.append(arr[::-1])  # Reverse for display

    n = len(surfaces)
    n_diffs = n * (n - 1) // 2
    total_panels = n + n_diffs
    fig, axes = plt.subplots(
        1, total_panels,
        figsize=(6 * total_panels, max(5, len(strikes) * 0.7)),
        squeeze=False,
    )
    axes = axes[0]

    # Individual surface heatmaps
    for idx, (ls, arr) in enumerate(zip(surfaces, arrays)):
        ax = axes[idx]
        im = ax.imshow(arr, cmap="RdYlGn", vmin=0.0, vmax=1.0, aspect="auto")
        _annotate_heatmap(ax, arr, strikes_desc, tenor_labels)
        ax.set_title(ls.label, fontsize=12, fontweight="bold")
        fig.colorbar(im, ax=ax, shrink=0.8)

    # Difference heatmaps
    panel_idx = n
    for i in range(n):
        for j in range(i + 1, n):
            diff = arrays[i] - arrays[j]
            ax = axes[panel_idx]
            max_abs = max(np.nanmax(np.abs(diff)), 0.01)
            im = ax.imshow(
                diff, cmap="RdBu_r", vmin=-max_abs, vmax=max_abs, aspect="auto",
            )
            _annotate_heatmap(ax, diff, strikes_desc, tenor_labels, fmt="+.2f")
            ax.set_title(
                f"{surfaces[i].label} - {surfaces[j].label}",
                fontsize=11, fontweight="bold",
            )
            fig.colorbar(im, ax=ax, shrink=0.8)
            panel_idx += 1

    pair = surfaces[0].surface.pair
    fig.suptitle(
        f"{pair} Source Comparison — Probability Surfaces",
        fontsize=14, fontweight="bold", y=1.02,
    )
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def _annotate_heatmap(
    ax,
    data: np.ndarray,
    strike_labels: list[float],
    tenor_labels: list[str],
    fmt: str = ".2f",
) -> None:
    """Add text annotations and axis labels to a heatmap."""
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = data[i, j]
            if not np.isnan(val):
                color = "white" if abs(val) > 0.6 or val < -0.3 else "black"
                ax.text(
                    j, i, f"{val:{fmt}}", ha="center", va="center",
                    color=color, fontsize=9, fontweight="bold",
                )
    ax.set_xticks(range(len(tenor_labels)))
    ax.set_xticklabels(tenor_labels, fontsize=10)
    ax.set_yticks(range(len(strike_labels)))
    ax.set_yticklabels([f"{s:.2f}" for s in strike_labels], fontsize=9)
    ax.set_xlabel("Tenor", fontsize=11)
    ax.set_ylabel("Strike", fontsize=11)


def plot_overlay_scatter(
    surfaces: list[LabeledSurface],
    path: Path,
) -> Path:
    """Prob-vs-strike and prob-vs-tenor curves overlaid, different styles per source."""
    strikes, tenors = _align_grids(surfaces)
    tenor_labels = [t.value for t in tenors]
    tenor_days = [t.days for t in tenors]

    cmap = plt.cm.get_cmap("tab10")
    line_styles = ["-", "--", "-.", ":"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # --- Plot 1: Prob vs Strike (one series per surface, averaged across tenors) ---
    for idx, ls in enumerate(surfaces):
        color = cmap(idx)
        ls_style = line_styles[idx % len(line_styles)]
        for t_idx, tenor in enumerate(tenors):
            xs, ys = [], []
            for strike in strikes:
                val = ls.grid.get((strike, tenor))
                if val is not None:
                    xs.append(strike)
                    ys.append(val)
            label = f"{ls.label} ({tenor.value})" if t_idx == 0 else None
            # Use same color per surface, vary alpha by tenor
            alpha = 0.4 + 0.6 * (t_idx / max(len(tenors) - 1, 1))
            ax1.plot(
                xs, ys, color=color, linestyle=ls_style, alpha=alpha,
                linewidth=1.5, marker="o", markersize=4,
                label=f"{ls.label}" if t_idx == len(tenors) // 2 else None,
            )

    ax1.set_xlabel("Strike", fontsize=11)
    ax1.set_ylabel("Probability", fontsize=11)
    ax1.set_ylim(-0.05, 1.05)
    ax1.set_title("Prob vs Strike (all tenors)", fontsize=12, fontweight="bold")
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # --- Plot 2: Prob vs Tenor (one series per surface, averaged across strikes) ---
    for idx, ls in enumerate(surfaces):
        color = cmap(idx)
        ls_style = line_styles[idx % len(line_styles)]
        for s_idx, strike in enumerate(strikes):
            xs, ys = [], []
            for t_idx, tenor in enumerate(tenors):
                val = ls.grid.get((strike, tenor))
                if val is not None:
                    xs.append(tenor_days[t_idx])
                    ys.append(val)
            alpha = 0.4 + 0.6 * (s_idx / max(len(strikes) - 1, 1))
            ax2.plot(
                xs, ys, color=color, linestyle=ls_style, alpha=alpha,
                linewidth=1.5, marker="s", markersize=4,
                label=f"{ls.label}" if s_idx == len(strikes) // 2 else None,
            )

    ax2.set_xscale("log")
    ax2.set_xticks(tenor_days)
    ax2.set_xticklabels(tenor_labels)
    ax2.set_xlabel("Tenor", fontsize=11)
    ax2.set_ylabel("Probability", fontsize=11)
    ax2.set_ylim(-0.05, 1.05)
    ax2.set_title("Prob vs Tenor (all strikes)", fontsize=12, fontweight="bold")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    pair = surfaces[0].surface.pair
    labels = " vs ".join(ls.label for ls in surfaces)
    fig.suptitle(
        f"{pair} — {labels}",
        fontsize=14, fontweight="bold", y=1.02,
    )
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Interactive visualization (Plotly)
# ---------------------------------------------------------------------------


def plot_comparison_interactive(
    surfaces: list[LabeledSurface],
    path: Path,
) -> Path:
    """Plotly HTML with dropdown to toggle surfaces + diff views."""
    strikes, tenors = _align_grids(surfaces)
    strikes_desc = list(reversed(strikes))
    tenor_labels = [t.value for t in tenors]

    arrays = {}
    for ls in surfaces:
        arr = _surface_to_array(ls, strikes, tenors)
        arrays[ls.label] = arr[::-1]

    # Build all views: individual surfaces + pairwise diffs
    views: dict[str, np.ndarray] = {}
    for label, arr in arrays.items():
        views[label] = arr

    labels_list = list(arrays.keys())
    for i in range(len(labels_list)):
        for j in range(i + 1, len(labels_list)):
            diff_label = f"{labels_list[i]} - {labels_list[j]}"
            views[diff_label] = arrays[labels_list[i]] - arrays[labels_list[j]]

    fig = go.Figure()

    # Add a heatmap trace for each view (only first visible)
    view_names = list(views.keys())
    for idx, (name, arr) in enumerate(views.items()):
        is_diff = " - " in name
        colorscale = "RdBu_r" if is_diff else "RdYlGn"
        if is_diff:
            max_abs = max(float(np.nanmax(np.abs(arr))), 0.01)
            zmin, zmax = -max_abs, max_abs
        else:
            zmin, zmax = 0.0, 1.0

        # Build text annotations
        text = []
        for row in arr:
            text_row = []
            for val in row:
                if np.isnan(val):
                    text_row.append("")
                else:
                    text_row.append(f"{val:.3f}")
            text.append(text_row)

        fig.add_trace(go.Heatmap(
            z=arr.tolist(),
            x=tenor_labels,
            y=[f"{s:.2f}" for s in strikes_desc],
            text=text,
            texttemplate="%{text}",
            colorscale=colorscale,
            zmin=zmin,
            zmax=zmax,
            colorbar=dict(title="Diff" if is_diff else "Prob"),
            visible=(idx == 0),
            hovertemplate=(
                "Strike: %{y}<br>Tenor: %{x}<br>Value: %{z:.3f}<extra></extra>"
            ),
        ))

    # Dropdown menu
    buttons = []
    for idx, name in enumerate(view_names):
        visibility = [False] * len(view_names)
        visibility[idx] = True
        buttons.append(dict(
            label=name,
            method="update",
            args=[
                {"visible": visibility},
                {"title": f"{surfaces[0].surface.pair} — {name}"},
            ],
        ))

    pair = surfaces[0].surface.pair
    fig.update_layout(
        title=f"{pair} — {view_names[0]}",
        updatemenus=[dict(
            type="dropdown",
            direction="down",
            x=0.0,
            xanchor="left",
            y=1.15,
            yanchor="top",
            buttons=buttons,
        )],
        xaxis_title="Tenor",
        yaxis_title="Strike",
        template="plotly_white",
        height=max(500, len(strikes) * 50),
        width=max(700, len(tenors) * 120),
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(
        str(path),
        include_plotlyjs=True,
        full_html=True,
        config={"displayModeBar": True, "scrollZoom": True},
    )
    return path


# ---------------------------------------------------------------------------
# Top-level orchestrator
# ---------------------------------------------------------------------------


def compare_surfaces(
    paths: list[Path],
    output_dir: Path,
) -> dict[str, Path]:
    """Load surfaces and generate all comparison visualizations.

    Returns:
        Dict of output name → file path.
    """
    surfaces = load_surfaces(paths)
    if len(surfaces) < 2:
        raise ValueError("Need at least 2 surfaces to compare")

    labels_slug = "_vs_".join(ls.label for ls in surfaces)
    pair = surfaces[0].surface.pair
    base_name = f"compare_{pair}_{labels_slug}"

    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, Path] = {}

    heatmap_path = output_dir / f"{base_name}.png"
    plot_diff_heatmaps(surfaces, heatmap_path)
    outputs["heatmaps"] = heatmap_path

    scatter_path = output_dir / f"{base_name}_scatter.png"
    plot_overlay_scatter(surfaces, scatter_path)
    outputs["scatter"] = scatter_path

    html_path = output_dir / f"{base_name}.html"
    plot_comparison_interactive(surfaces, html_path)
    outputs["interactive"] = html_path

    return outputs
