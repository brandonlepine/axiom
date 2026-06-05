"""Publication-grade figures for activation-patching results (colorblind-safe).

Figures are generated from the tracked patching summaries and saved with stable stems as
both PNG and PDF. Signed effects use the colorblind-safe diverging map from
``axiom.figures.style`` (blue/orange, not blue/red) with a symmetric, outlier-robust
color range.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from axiom.alignment import PATCH_SPANS
from axiom.figures.style import apply_publication_style, diverging_cmap, save_figure


def _robust_symmetric_vlim(values: pd.Series) -> float:
    """Symmetric color limit at the 95th percentile of |value| (ignores NaN/outliers)."""
    finite = pd.to_numeric(values, errors="coerce").dropna().abs()
    return max(float(finite.quantile(0.95)), 1e-6) if not finite.empty else 1.0


def residual_span_heatmap(span_summary: pd.DataFrame, path: Path, title: str) -> list[Path]:
    """Layer x span heatmap of mean ``bias_effect`` (the localization result).

    Args:
        span_summary: columns ``layer``, ``span``, ``mean_bias_effect``.
        path: output stem (``.png`` / ``.pdf`` appended).
        title: figure title.

    Returns:
        Paths written (png, pdf). Empty input writes nothing.
    """
    if span_summary.empty:
        return []
    import matplotlib.pyplot as plt

    apply_publication_style()
    present = [s for s in PATCH_SPANS if s in set(span_summary["span"].astype(str))]
    pivot = (
        span_summary.assign(span=span_summary["span"].astype(str))
        .pivot(index="layer", columns="span", values="mean_bias_effect")
        .reindex(columns=present)
        .sort_index()
    )
    vlim = _robust_symmetric_vlim(pd.Series(pivot.values.ravel()))

    fig, ax = plt.subplots(figsize=(max(5, len(present) * 1.3), 8))
    im = ax.imshow(
        pivot.values, aspect="auto", origin="lower", cmap=diverging_cmap(),
        vmin=-vlim, vmax=vlim, interpolation="nearest",
    )
    ax.set_title(title)
    ax.set_xlabel("prompt span")
    ax.set_ylabel("layer")
    ax.set_xticks(range(len(present)))
    ax.set_xticklabels(present, rotation=45, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([str(int(y)) for y in pivot.index])
    ax.grid(False)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("mean bias_effect  (Δ logP continuation from injecting stereotype state)")
    fig.tight_layout()
    written = save_figure(fig, path)
    plt.close(fig)
    return written
