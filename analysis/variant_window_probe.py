#!/usr/bin/env python3
"""
Robustness check R6 analysis: probes on variant_window_embed.py output
(eQTLP-v2 matched pairs, fixed chromosome folds; nothing here touches clinical data).

For every layer x pooling (mean / w1k / w64 / pos):
  shift          median ||alt - ref|| / ||ref|| (true alt, and the random-alt control)
  refalt, ref_copy, delta   linear probe on [ref, alt], scored with the true alt and alt := ref
  ref_only       linear probe on ref alone (context at that pooling)
  diff_lin       linear probe on alt - ref
  diff_gbt       gradient boosting on alt - ref
  rand_gbt       gradient boosting on alt_rand - ref (the control: a base change that
                 did NOT happen). diff_gbt > rand_gbt means the model encodes something
                 specific to the observed allele; diff_gbt ~ rand_gbt means the difference
                 vector only carries local context.

Usage:  python -X utf8 analysis/variant_window_probe.py [--model hyena caduceus]
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from hyena_probe import fit  # noqa: E402

N_FOLDS = 5
POOLS = ("mean", "w1k", "w64", "pos")


def fold_mean(y, s, fold):
    return float(np.mean([roc_auc_score(y[fold == k], s[fold == k]) for k in range(N_FOLDS)]))


def oof(X, y, fold, kind, X_test=None):
    X_test = X if X_test is None else X_test
    s = np.zeros(len(y))
    for k in range(N_FOLDS):
        te, tr = fold == k, fold != k
        if kind == "lin":
            m = fit(X[tr], y[tr], 0.1)
        else:
            m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=15,
                                               l2_regularization=1.0, random_state=0).fit(X[tr], y[tr])
        s[te] = m.predict_proba(X_test[te])[:, 1]
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", nargs="+", default=["hyena", "caduceus"])
    args = ap.parse_args()
    pairs = pd.read_csv("data/eQTL_v2/pairs.tsv", sep="\t")
    mp = pairs[pairs["matched"] == 1].reset_index(drop=True)
    rows = []
    for model in args.model:
        path = f"data/eQTL_v2/window_emb/{model}/window_emb.npz"
        if not os.path.exists(path):
            print(f"skip {model}: {path} not found")
            continue
        z = np.load(path, allow_pickle=True)
        assert (z["region_id"] == mp["region_id"].values).all()
        y, fold = z["y"], mp["fold_chrom"].values
        R, A, AR = (z[s].astype(np.float32) for s in ("ref", "alt", "alt_rand"))
        n_layers = R.shape[1]
        for li in range(n_layers):
            for pi, pool in enumerate(POOLS):
                r, a, ar = R[:, li, pi], A[:, li, pi], AR[:, li, pi]
                rel = lambda x: np.median(np.linalg.norm(x - r, axis=1) / np.linalg.norm(r, axis=1))
                s_ra = oof(np.hstack([r, a]), y, fold, "lin")
                s_copy = oof(np.hstack([r, a]), y, fold, "lin", X_test=np.hstack([r, r]))
                row = {"model": model, "layer": li if li < n_layers - 1 else "final", "pool": pool,
                       "shift": rel(a), "shift_rand": rel(ar),
                       "refalt": fold_mean(y, s_ra, fold), "ref_copy": fold_mean(y, s_copy, fold),
                       "ref_only": fold_mean(y, oof(r, y, fold, "lin"), fold),
                       "diff_lin": fold_mean(y, oof(a - r, y, fold, "lin"), fold),
                       "diff_gbt": fold_mean(y, oof(a - r, y, fold, "gbt"), fold),
                       "rand_gbt": fold_mean(y, oof(ar - r, y, fold, "gbt"), fold)}
                row["delta"] = row["refalt"] - row["ref_copy"]
                rows.append(row)
                print("  ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()),
                      flush=True)
    r = pd.DataFrame(rows)
    os.makedirs("analysis/results/robustness", exist_ok=True)
    r.to_csv("analysis/results/robustness/R6_variant_window.tsv", sep="\t", index=False)
    for model, g in r.groupby("model"):
        best = g.loc[g.diff_gbt.idxmax()]
        print(f"\n{model}: max |delta| {g.delta.abs().max():.4f}; best refalt {g.refalt.max():.3f}; "
              f"best diff_gbt {best.diff_gbt:.3f} (layer {best.layer}, {best['pool']}; rand control there "
              f"{best.rand_gbt:.3f}); mean diff_gbt - rand_gbt {(g.diff_gbt - g.rand_gbt).mean():+.4f}")


if __name__ == "__main__":
    main()
