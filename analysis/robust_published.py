#!/usr/bin/env python3
"""
Robustness check R1/R2: an independent recount of the published eQTLP split claims
(matrix 9.1-9.3), written from scratch rather than reusing eqtl_leakage.py.

Per tissue, from data/eQTL/targets/*.data.tsv as released:
  - test positives / negatives, positive rate per split
  - AUROC of -distance on the test split, with distance taken two ways:
      (a) the released distance_to_tss column
      (b) recomputed from coordinates: |variant - TSS|, TSS = gene_start (+) or gene_end (-)
    and with and without parse_eQTL's 450 kb span cutoff
  - share of test rows whose gene / variant appears in train

Usage:  python -X utf8 analysis/robust_published.py
"""
import glob
import os

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

SEQ_CUTOFF = 450_000
TSS_FLANK, REGION_FLANK = 3000, 500


def main():
    rows = []
    for path in sorted(glob.glob("data/eQTL/targets/*.data.tsv")):
        tissue = os.path.basename(path).split(".")[0]
        d = pd.read_csv(path, sep="\t")
        d["y"] = (d["target"] == "positive").astype(int)
        tss = np.where(d["gene_strand"] == "+", d["gene_start"], d["gene_end"] - 1)
        d["dist_coord"] = np.abs(d["region_end"] - 1 - tss)
        span = (np.maximum(d["region_end"] + REGION_FLANK, tss + 1 + TSS_FLANK)
                - np.minimum(d["region_start"] - REGION_FLANK, tss - TSS_FLANK))
        d["in_cutoff"] = span <= SEQ_CUTOFF
        te, tr = d[d["subset"] == "test"], d[d["subset"] == "train"]
        tec = te[te["in_cutoff"]]
        rate = d.groupby("subset")["y"].mean()
        rows.append({
            "tissue": tissue,
            "test_pos": int(te.y.sum()), "test_neg": int((1 - te.y).sum()),
            "pos_rate_train": rate.get("train"), "pos_rate_valid": rate.get("valid"),
            "pos_rate_test": rate.get("test"),
            "split_share_train": (d.subset == "train").mean(), "split_share_test": (d.subset == "test").mean(),
            "auc_dist_column": roc_auc_score(te.y, -te["distance_to_tss"].abs()),
            "auc_dist_coords": roc_auc_score(te.y, -te["dist_coord"]),
            "auc_dist_coords_cutoff": roc_auc_score(tec.y, -tec["dist_coord"]),
            "test_rows_dropped_by_cutoff": int((~te["in_cutoff"]).sum()),
            "test_gene_in_train": te["gene_id"].isin(set(tr["gene_id"])).mean(),
            "test_variant_in_train": te["region_id"].isin(set(tr["region_id"])).mean(),
            "median_dist_pos": te.loc[te.y == 1, "dist_coord"].median(),
            "median_dist_neg": te.loc[te.y == 0, "dist_coord"].median(),
        })
    r = pd.DataFrame(rows)
    pd.set_option("display.width", 250, "display.max_columns", 30)
    print(r.round(3).to_string(index=False))
    print("\nmean AUROC  column {:.3f}  coords {:.3f}  coords+cutoff {:.3f}".format(
        r.auc_dist_column.mean(), r.auc_dist_coords.mean(), r.auc_dist_coords_cutoff.mean()))
    os.makedirs("analysis/results/robustness", exist_ok=True)
    r.to_csv("analysis/results/robustness/R1_published_split.tsv", sep="\t", index=False)


if __name__ == "__main__":
    main()
