#!/usr/bin/env python3
"""
No-GPU leakage audit of DNALongBench eQTLP (the published random 8:1:1 split).

For each tissue:
  - label balance per split
  - AUROC of variant-to-TSS distance alone on the test split
    (the loader pads variant..TSS with N to 450 kb, so padding length = distance)
  - test rows whose gene / variant / locus also appears in train

Usage:  python -X utf8 analysis/eqtl_leakage.py --root data/eQTL
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


# Published test AUROCs, Cheng et al. Nat Commun 2025, Table 7.
PUBLISHED = {  # tissue: (Expert, CNN, HyenaDNA, Caduceus-Ph, Caduceus-PS)
    "Cells_Cultured_fibroblasts":      (0.639, 0.547, 0.584, 0.597, 0.549),
    "Whole_Blood":                     (0.689, 0.577, 0.512, 0.594, 0.542),
    "Thyroid":                         (0.612, 0.487, 0.529, 0.527, 0.547),
    "Skin_Not_Sun_Exposed_Suprapubic": (0.710, 0.499, 0.471, 0.586, 0.529),
    "Skin_Sun_Exposed_Lower_leg":      (0.700, 0.499, 0.544, 0.574, 0.541),
    "Muscle_Skeletal":                 (0.621, 0.502, 0.487, 0.538, 0.523),
    "Nerve_Tibial":                    (0.683, 0.516, 0.511, 0.588, 0.552),
    "Artery_Tibial":                   (0.741, 0.576, 0.479, 0.547, 0.536),
    "Adipose_Subcutaneous":            (0.736, 0.551, 0.513, 0.541, 0.519),
}
MODELS = ("expert", "cnn", "hyena", "cad_ph", "cad_ps")


def hanley_mcneil_se(auc, n_pos, n_neg):
    """Standard error of an AUROC (Hanley & McNeil 1982)."""
    q1, q2 = auc / (2 - auc), 2 * auc ** 2 / (1 + auc)
    var = (auc * (1 - auc) + (n_pos - 1) * (q1 - auc ** 2)
           + (n_neg - 1) * (q2 - auc ** 2)) / (n_pos * n_neg)
    return np.sqrt(var)


def bootstrap_auc_ci(y, score, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    y, score = np.asarray(y), np.asarray(score)
    aucs = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() != y[i].max():
            aucs.append(roc_auc_score(y[i], score[i]))
    return np.percentile(aucs, [2.5, 97.5])


def parse_config(path):
    cfg = {}
    with open(path) as fh:
        for line in fh:
            row = line.split()
            if len(row) >= 2:
                cfg[row[0]] = row[1]
    return cfg


def tss(df):
    return np.where(df["gene_strand"] == "+", df["gene_start"], df["gene_end"] - 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/eQTL")
    ap.add_argument("--locus-kb", type=int, default=100,
                    help="variants within this many kb count as the same locus")
    args = ap.parse_args()

    rows = []
    for cfg_path in sorted(glob.glob(os.path.join(args.root, "config", "gtex_hg38.*.config"))):
        # gtex_hg38.<tissue>.config; the bare gtex_hg38.config is the combined set
        tissue = os.path.basename(cfg_path)[len("gtex_hg38."):-len(".config")] or "combined"
        cfg = parse_config(cfg_path)
        data_path = os.path.join(args.root, cfg["eQTL_file"])
        if not os.path.exists(data_path):
            print(f"skip {tissue}: {data_path} not found")
            continue
        df = pd.read_csv(data_path, sep="\t")
        df = df[df["gene_chrom"] == df["region_chrom"]].copy()
        df["y"] = (df["target"] == "positive").astype(int)
        df["dist"] = np.abs(df["region_start"] - tss(df))
        cutoff = int(cfg.get("seq_len_cutoff", 450000))
        df = df[df["dist"] <= cutoff]
        df["locus"] = df["region_chrom"] + ":" + (df["region_start"] // (args.locus_kb * 1000)).astype(str)

        tr, te = df[df["subset"] == "train"], df[df["subset"] == "test"]
        r = {"tissue": tissue, "n": len(df)}
        for s in ("train", "valid", "test"):
            d = df[df["subset"] == s]
            r[f"n_{s}"] = len(d)
            r[f"pos_{s}"] = d["y"].mean() if len(d) else np.nan
        n_pos, n_neg = int(te["y"].sum()), int((1 - te["y"]).sum())
        r["test_pos"], r["test_neg"] = n_pos, n_neg
        r["auroc_distance"] = roc_auc_score(te["y"], -te["dist"])
        r["dist_ci_lo"], r["dist_ci_hi"] = bootstrap_auc_ci(te["y"], -te["dist"])
        # half-width of a 95% CI around AUROC 0.55 at this test size
        r["ci95_halfwidth_at_0.55"] = 1.96 * hanley_mcneil_se(0.55, n_pos, n_neg)
        for m, v in zip(MODELS, PUBLISHED.get(tissue, (np.nan,) * 5)):
            r[f"pub_{m}"] = v
        r["med_dist_pos_kb"] = te.loc[te.y == 1, "dist"].median() / 1e3
        r["med_dist_neg_kb"] = te.loc[te.y == 0, "dist"].median() / 1e3
        r["test_gene_in_train"] = te["gene_id"].isin(set(tr["gene_id"])).mean()
        r["test_variant_in_train"] = te["region_id"].isin(set(tr["region_id"])).mean()
        r["test_locus_in_train"] = te["locus"].isin(set(tr["locus"])).mean()
        r["chroms_in_test"] = te["region_chrom"].nunique()
        rows.append(r)

    out = pd.DataFrame(rows)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)

    print("SPLIT COMPOSITION")
    print(out[["tissue", "n_train", "n_valid", "n_test", "pos_train", "pos_valid",
               "pos_test", "test_pos", "test_neg"]].round(3).to_string(index=False))
    print("\nDISTANCE-ONLY BASELINE vs PUBLISHED (test AUROC)")
    cols = ["tissue", "auroc_distance", "dist_ci_lo", "dist_ci_hi"] + [f"pub_{m}" for m in MODELS] \
        + ["ci95_halfwidth_at_0.55", "med_dist_pos_kb", "med_dist_neg_kb"]
    print(out[cols].round(3).to_string(index=False))
    mean = out[["auroc_distance"] + [f"pub_{m}" for m in MODELS]].mean()
    print("  mean: " + "  ".join(f"{k}={v:.3f}" for k, v in mean.items()))
    print("\nTRAIN/TEST OVERLAP (fraction of test rows)")
    print(out[["tissue", "test_gene_in_train", "test_variant_in_train",
               "test_locus_in_train", "chroms_in_test"]].round(3).to_string(index=False))
    os.makedirs("analysis/results", exist_ok=True)
    out.to_csv("analysis/results/eqtl_leakage.tsv", sep="\t", index=False)
    print("\nwritten: analysis/results/eqtl_leakage.tsv")


if __name__ == "__main__":
    main()
