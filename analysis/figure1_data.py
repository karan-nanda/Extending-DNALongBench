#!/usr/bin/env python3
"""
Per-fold numbers behind Figure 1, panel A (paper/workshop_draft.md 4).

For each model on eQTLP-v2 (matched pairs, fixed chromosome folds) this writes the
per-fold test AUROC of `refalt` and of `ref_copy`, next to the local-composition
floor computed on the same folds. The FM probes are re-fitted here (logistic,
C=0.1, the setting in Table 2) because R4_probes.tsv stores only fold means; the
CNN numbers are read from the runs in analysis/results/cnn_ablation.

Usage:  python -X utf8 analysis/figure1_data.py
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from eqtl_v2_baselines import oof_scores  # noqa: E402
from robust_eqtl_v2 import EMB, N_FOLDS, probe_scores  # noqa: E402

OUT = "analysis/results/robustness"
CNN_DIR = "analysis/results/cnn_ablation"
C_PROBE = 0.1


def per_fold_auc(y, s, fold):
    return [float(roc_auc_score(y[fold == k], s[fold == k])) for k in range(N_FOLDS)]


def main():
    pairs = pd.read_csv("data/eQTL_v2/pairs.tsv", sep="\t")
    X = pd.read_csv("data/eQTL_v2/features.tsv.gz", sep="\t", index_col=0)
    assert len(X) == len(pairs)

    keep = (pairs["matched"] == 1).values
    mp = pairs[keep].reset_index(drop=True)
    xm = X[keep].reset_index(drop=True)
    y = mp["y"].values
    fold = mp["fold_chrom"].values

    rows = []

    # local-composition floor, same pairs and folds as the probes
    s_floor = oof_scores(mp, xm, "fold_chrom", "composition", 0)
    for k, a in enumerate(per_fold_auc(y, s_floor, fold)):
        rows.append({"model": "Floor (composition)", "fold": k, "seed": 0,
                     "refalt": a, "ref_copy": np.nan})

    # frozen FM probes
    for model, path in EMB.items():
        z = np.load(path, allow_pickle=True)
        assert (z["region_id"] == mp["region_id"].values).all()
        er = np.hstack([z["ref_fwd"], z["ref_rev"]])
        ea = np.hstack([z["alt_fwd"], z["alt_rev"]])
        s_ra, s_copy, _ = probe_scores(er, ea, y, fold, C_PROBE)
        a_ra, a_copy = per_fold_auc(y, s_ra, fold), per_fold_auc(y, s_copy, fold)
        for k in range(N_FOLDS):
            rows.append({"model": model, "fold": k, "seed": 0,
                         "refalt": a_ra[k], "ref_copy": a_copy[k]})

    # end-to-end CNN, 5 folds x 3 seeds
    for k in range(N_FOLDS):
        for seed in range(3):
            f = os.path.join(CNN_DIR, f"matched_chrom_f{k}_refalt_s{seed}.json")
            ab = json.load(open(f))["ablation"]
            rows.append({"model": "CNN", "fold": k, "seed": seed,
                         "refalt": ab["alt"], "ref_copy": ab["ref_copy"]})

    r = pd.DataFrame(rows)
    r.to_csv(os.path.join(OUT, "F1_per_fold.tsv"), sep="\t", index=False)

    print("per-fold AUROC, eQTLP-v2 matched / chromosome folds")
    g = r.groupby("model").agg(refalt_mean=("refalt", "mean"), refalt_sd=("refalt", "std"),
                               ref_copy_mean=("ref_copy", "mean"), n=("refalt", "size"))
    g["delta"] = g["refalt_mean"] - g["ref_copy_mean"]
    print(g.round(4).to_string())
    print(f"\nwrote {OUT}/F1_per_fold.tsv")


if __name__ == "__main__":
    main()
