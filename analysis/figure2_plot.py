#!/usr/bin/env python3
"""
Figure 2: where in the network the allele signal lives, and whether it is really
the allele.

For every layer x readout cell we train a nonlinear probe on the allele
difference (diff_gbt) and the same probe on a control in which the variant is
replaced by a base that did NOT occur (rand_gbt). The control absorbs anything
the probe could learn from local sequence context alone, so

    gain = diff_gbt - rand_gbt

is the part attributable to the actual alternate allele. Cells near zero mean
the probe was reading context, not the variant.

The figure's point: gain is at noise level almost everywhere -- including every
HyenaDNA cell -- and the single cell that stands out (Caduceus, final layer,
+-1 kb) is still below the 0.599 hand-feature floor.

Inputs:  analysis/results/robustness/R6_variant_window.tsv
Outputs: paper/figures/figure2.pdf (+ _wide for the 1in-margin build)

Usage:  python -X utf8 analysis/figure2_plot.py --width 5.5 --name figure2
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402

SRC = "analysis/results/robustness/R6_variant_window.tsv"
OUT = "paper/figures"
TEXT_W = 5.5

INK, MUTED, GRID = "#0b0b0b", "#52514e", "#d8d7d2"
# diverging: one cool pole, one warm pole, neutral grey midpoint (never a hue at zero)
COOL, WARM, MID = "#2a78d6", "#eb6834", "#f2f1ee"
CMAP = LinearSegmentedColormap.from_list("gain", [COOL, MID, WARM])

POOLS = [("pos", "variant pos."), ("w64", "±64 bp"),
         ("w1k", "±1 kb"), ("mean", "whole seq.")]
MODELS = [("caduceus", "Caduceus-Ph"), ("hyena", "HyenaDNA")]

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.linewidth": 0.6,
    "figure.dpi": 200,
})


def layer_order(vals):
    """Numeric layers ascending, then the pooled 'final' readout."""
    nums = sorted(int(v) for v in vals if str(v) != "final")
    return [str(n) for n in nums] + (["final"] if "final" in set(map(str, vals)) else [])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--width", type=float, default=TEXT_W)
    ap.add_argument("--name", default="figure2")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    r = pd.read_csv(SRC, sep="\t")
    r["layer"] = r["layer"].astype(str)
    r["gain"] = r["diff_gbt"] - r["rand_gbt"]
    lim = float(np.abs(r["gain"]).max())
    norm = TwoSlopeNorm(vmin=-lim, vcenter=0.0, vmax=lim)

    h = 2.55 * args.width / TEXT_W
    widths = [len(layer_order(r[r.model == m].layer.unique())) for m, _ in MODELS]
    fig, axes = plt.subplots(1, 2, figsize=(args.width, h),
                             gridspec_kw={"width_ratios": widths, "wspace": 0.12})

    for ax, (key, label) in zip(axes, MODELS):
        g = r[r.model == key]
        layers = layer_order(g.layer.unique())
        M = np.full((len(POOLS), len(layers)), np.nan)
        for i, (pool, _) in enumerate(POOLS):
            for j, lay in enumerate(layers):
                cell = g[(g["pool"] == pool) & (g.layer == lay)]
                if len(cell):
                    M[i, j] = cell["gain"].iloc[0]

        ax.imshow(M, cmap=CMAP, norm=norm, aspect="auto")
        ax.set_xticks(range(len(layers)),
                      [("F" if l == "final" else l) for l in layers], fontsize=6.5)
        ax.set_xlabel("layer", fontsize=8)
        if ax is axes[0]:
            ax.set_yticks(range(len(POOLS)), [p[1] for p in POOLS], fontsize=7.5)
        else:
            ax.set_yticks(range(len(POOLS)), [])
        ax.set_title(label, fontsize=8.5, pad=4, color=INK)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.tick_params(length=0)

        # ring the one cell that clears its control by a visible margin
        best = np.unravel_index(np.nanargmax(M), M.shape)
        if M[best] > 0.03:
            ax.add_patch(plt.Rectangle((best[1] - .5, best[0] - .5), 1, 1,
                                       fill=False, ec=INK, lw=1.4, zorder=3))
            ax.annotate("survives\nthe control", xy=(best[1], best[0] - 0.5),
                        xytext=(best[1] - 3.4, best[0] - 1.45), fontsize=6.5,
                        ha="center", color=INK,
                        arrowprops=dict(arrowstyle="-", lw=0.7, color=INK))

    # explicit margins: bbox_inches is None so the axis labels must fit inside
    fig.subplots_adjust(left=0.135, right=0.855, top=0.90, bottom=0.20)
    cax = fig.add_axes([0.878, 0.20, 0.020, 0.70])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=CMAP), cax=cax,
                      ticks=[-0.02, 0, 0.02])
    cb.set_label("allele gain over control (AUROC)", fontsize=7)
    cb.ax.tick_params(labelsize=6.5, length=2)
    cb.outline.set_visible(False)

    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"{args.name}.{ext}"), bbox_inches=None)
    print(f"wrote {OUT}/{args.name}.pdf and .png at {args.width}in")


if __name__ == "__main__":
    main()
