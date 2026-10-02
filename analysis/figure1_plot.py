#!/usr/bin/env python3
"""
Figure 1 for paper/workshop_draft.md.

A  eQTLP-v2 per-fold test AUROC, `refalt` vs `ref_copy`, against the local-
   composition floor. Each fold's pair is joined by a line; every line is
   vertical, i.e. the swap moves no score, which is the result.
B  how far one substituted base moves the representation, as a fraction of the
   embedding norm, at four readouts from the variant position out to the whole
   sequence. Band = min-max over layers, marker = median.

Drawn at TEXT_W, the paper's text width, so the PDF is included at 1:1 and no
font is scaled down. If the venue's text width differs, change TEXT_W and rerun
rather than scaling the graphic in LaTeX.

Inputs:  analysis/results/robustness/F1_per_fold.tsv  (run figure1_data.py first)
         analysis/results/robustness/R6_variant_window.tsv
Outputs: paper/figures/figure1.pdf, .png

Usage:  python -X utf8 analysis/figure1_plot.py
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

OUT = "paper/figures"
TEXT_W = 5.5   # inches; the workshop build's text width. --width overrides.
# categorical slots 1-3 of the validated palette; the floor is a reference, so it is ink-grey
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#d8d7d2"

# sizes are true points at 1:1, so they read against the paper's 10pt body text
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "axes.linewidth": 0.6,
    "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "legend.frameon": False, "figure.dpi": 200,
})

POOLS = [("pos", "variant\npos."), ("w64", "\u00b164 bp"),
         ("w1k", "\u00b11 kb"), ("mean", "whole\nseq.")]


def panel_a(ax, df):
    rows = [("Floor (composition)", "Composition floor", MUTED, "D"),
            ("Caduceus", "Caduceus-Ph", AQUA, "o"),
            ("HyenaDNA", "HyenaDNA", ORANGE, "s"),
            ("CNN", "CNN (end to end)", BLUE, "^")]
    floor = df[df.model == "Floor (composition)"].refalt.mean()
    lo, hi = df.refalt.min() - 0.018, df.refalt.max() + 0.012
    top = len(rows) - 0.42

    ax.axvline(0.5, color=GRID, lw=0.8, ls=":", zorder=0)
    ax.axvline(floor, color=MUTED, lw=0.8, ls="--", zorder=0)
    ax.text(0.5, top, "chance", color=MUTED, fontsize=7, ha="center", va="bottom")
    ax.text(floor, top, "floor %.3f" % floor, color=MUTED, fontsize=7,
            ha="center", va="bottom")

    ticks, labels = [], []
    for i, (key, label, color, mk) in enumerate(rows):
        y = len(rows) - 1 - i
        ticks.append(y)
        labels.append(label)
        g = df[df.model == key]
        if key == "Floor (composition)":
            ax.scatter(g.refalt, np.full(len(g), y), s=17, marker=mk, color=color, zorder=2)
            ax.plot([g.refalt.mean()], [y], marker="|", ms=10, mew=1.5, color=color, zorder=3)
            continue
        # each fold (and seed) joins its refalt and ref_copy score
        for a, b in zip(g.refalt, g.ref_copy):
            ax.plot([a, b], [y + 0.15, y - 0.15], color=color, lw=0.6, alpha=0.55,
                    solid_capstyle="round", zorder=1)
        ax.scatter(g.ref_copy, np.full(len(g), y - 0.15), s=17, marker=mk,
                   facecolors="none", edgecolors=color, linewidths=0.9, zorder=2)
        ax.scatter(g.refalt, np.full(len(g), y + 0.15), s=17, marker=mk,
                   color=color, zorder=2)
        ax.plot([g.refalt.mean()], [y + 0.15], marker="|", ms=10, mew=1.5,
                color=color, zorder=3)
        ax.plot([g.ref_copy.mean()], [y - 0.15], marker="|", ms=10, mew=1.5,
                color=color, zorder=3)

    ax.set_yticks(ticks, labels, fontsize=8, color=INK)
    ax.tick_params(axis="y", length=0, pad=3)
    ax.set_ylim(-1.5, len(rows) - 0.2)   # room below the CNN row for the marker key
    ax.set_xlim(lo, hi)
    ax.set_xlabel("test AUROC (per fold)", labelpad=9)
    for s in ("left", "right", "top"):
        ax.spines[s].set_visible(False)

    # what the filled and open markers mean, stated once
    key = [Line2D([], [], marker="o", ls="none", ms=4.2, color=INK, label="ref / alt"),
           Line2D([], [], marker="o", ls="none", ms=4.2, mfc="none", mec=INK,
                  mew=0.9, label="ref copied over alt")]
    ax.legend(handles=key, loc="lower left", fontsize=7, handletextpad=0.4,
              borderpad=0.2, labelspacing=0.35)


def panel_b(ax, r):
    style = {"caduceus": ("Caduceus-Ph", AQUA, "o", "-"),
             "hyena": ("HyenaDNA", ORANGE, "s", "--")}
    x = np.arange(len(POOLS))
    for model, (label, color, mk, ls) in style.items():
        g = r[r.model == model]
        med = [g[g["pool"] == p]["shift"].median() for p, _ in POOLS]
        lo = [g[g["pool"] == p]["shift"].min() for p, _ in POOLS]
        hi = [g[g["pool"] == p]["shift"].max() for p, _ in POOLS]
        ax.fill_between(x, lo, hi, color=color, alpha=0.16, lw=0)
        ax.plot(x, med, ls=ls, lw=1.6, color=color, marker=mk, ms=5,
                mec="white", mew=0.7, label=label, zorder=3)

    ax.set_yscale("log")
    ax.set_ylim(3e-5, 3)
    ax.set_xlim(-0.45, len(POOLS) - 0.45)
    ax.set_xticks(x, [lab for _, lab in POOLS], fontsize=7)
    ax.set_ylabel("relative shift  ‖e_alt − e_ref‖ / ‖e_ref‖")
    ax.set_xlabel("readout window")
    ax.grid(axis="y", color=GRID, lw=0.5, zorder=0)
    ax.set_axisbelow(True)
    for s in ("right", "top"):
        ax.spines[s].set_visible(False)

    # the dilution, drawn rather than asserted
    ax.annotate("", xy=(2.93, 6.5e-4), xytext=(2.93, 0.42),
                arrowprops=dict(arrowstyle="-|>", lw=0.9, color=INK,
                                shrinkA=0, shrinkB=0))
    ax.text(2.84, 0.022, "≈1,000×\ndilution", fontsize=7,
            ha="right", va="center", color=INK)
    ax.legend(loc="lower left", fontsize=7.5, handlelength=2.2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--width", type=float, default=TEXT_W,
                    help="figure width in inches; must equal the build's text width")
    ap.add_argument("--name", default="figure1")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    df = pd.read_csv("analysis/results/robustness/F1_per_fold.tsv", sep="\t")
    r = pd.read_csv("analysis/results/robustness/R6_variant_window.tsv", sep="\t")

    fig, axes = plt.subplots(1, 2, figsize=(args.width, 2.45 * args.width / TEXT_W),
                             gridspec_kw={"width_ratios": [1.38, 1]})
    panel_a(axes[0], df)
    panel_b(axes[1], r)
    fig.tight_layout(w_pad=2.6, rect=[0, 0, 1, 0.94])
    fig.text(0.005, 0.995, "A   No model uses the allele",
             fontsize=9, fontweight="bold", va="top")
    fig.text(0.545, 0.995, "B   Pooling dilutes the substitution",
             fontsize=9, fontweight="bold", va="top")
    # no bbox_inches="tight": that crops to content and changes the width, so the
    # graphic would be rescaled in LaTeX and the fonts would no longer be true size
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"{args.name}.{ext}"))
    print(f"wrote {OUT}/{args.name}.pdf and .png at {args.width}in")


if __name__ == "__main__":
    main()
