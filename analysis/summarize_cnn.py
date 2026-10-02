#!/usr/bin/env python3
"""
Summarise analysis/eqtl_v2_cnn.py runs.

  - test AUROC per fold, by (version, scheme, mode, seed)
  - paired refalt - ref_only difference on identical test pairs, per fold,
    with a bootstrap CI over test pairs; > 0 means the model uses the allele
  - the CPU baselines for the same version/scheme, for reference

Usage:  python -X utf8 analysis/summarize_cnn.py [--dir analysis/results/cnn]
"""
import argparse
import glob
import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def paired_boot(y, a, b, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    d = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() != y[i].max():
            d.append(roc_auc_score(y[i], a[i]) - roc_auc_score(y[i], b[i]))
    return np.percentile(d, [2.5, 97.5])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="analysis/results/cnn")
    ap.add_argument("--baselines", default="analysis/results/eqtl_v2_baselines.tsv")
    args = ap.parse_args()

    rows = []
    for path in glob.glob(os.path.join(args.dir, "*.json")):
        with open(path) as fh:
            j = json.load(fh)
        a = j["args"]
        if a.get("limit"):
            continue
        rows.append({"version": a["version"], "scheme": a["scheme"], "fold": a["fold"],
                     "mode": a["mode"], "seed": a["seed"], "lr": a["lr"],
                     "test_auroc": j["test"]["auroc"], "best_valid": j["best_valid"],
                     "ablation": j.get("ablation"), "name": os.path.basename(path)[:-5]})
    if not rows:
        print("no finished runs yet")
        return
    r = pd.DataFrame(rows)

    print("TEST AUROC PER FOLD")
    t = r.pivot_table(index=["version", "scheme", "mode", "seed"], columns="fold",
                      values="test_auroc")
    t["mean"], t["sd"], t["n"] = t.mean(axis=1), t.std(axis=1), t.notna().sum(axis=1)
    print(t.round(3).to_string())

    print("\nPAIRED refalt - ref_only (identical test pairs; >0 = allele is used)")
    diffs = []
    for (v, s, f, sd), g in r.groupby(["version", "scheme", "fold", "seed"]):
        m = dict(zip(g["mode"], g["name"]))
        if not {"refalt", "ref_only"} <= m.keys():
            continue
        pa = pd.read_csv(os.path.join(args.dir, m["refalt"] + ".preds.tsv"), sep="\t")
        pb = pd.read_csv(os.path.join(args.dir, m["ref_only"] + ".preds.tsv"), sep="\t")
        x = pa.merge(pb, on=["region_id", "gene_id", "y"], suffixes=("_a", "_b"))
        y, sa, sb = x["y"].values, x["score_a"].values, x["score_b"].values
        d = roc_auc_score(y, sa) - roc_auc_score(y, sb)
        lo, hi = paired_boot(y, sa, sb)
        diffs.append(d)
        print(f"  {v}/{s} fold {f} seed {sd}: {d:+.3f}  [{lo:+.3f}, {hi:+.3f}]  (n={len(y)})")
    if diffs:
        print(f"  mean over folds: {np.mean(diffs):+.3f} ± {np.std(diffs):.3f} (n folds={len(diffs)})")

    ab = r[r["ablation"].notna()]
    if len(ab):
        print("\nTEST-TIME ABLATION (same trained refalt weights; alt - ref_copy > 0 = allele is used)")
        dd = []
        for _, row in ab.sort_values(["version", "scheme", "seed", "fold"]).iterrows():
            a = row["ablation"]
            p = pd.read_csv(os.path.join(args.dir, row["name"] + ".preds.tsv"), sep="\t")
            lo, hi = paired_boot(p["y"].values, p["score_alt"].values, p["score_ref_copy"].values)
            d = a["alt"] - a["ref_copy"]
            dd.append(d)
            print(f"  {row.version}/{row.scheme} fold {row.fold} seed {row.seed}: alt {a['alt']:.3f}  "
                  f"ref_copy {a['ref_copy']:.3f}  random_allele {a['random_allele']:.3f}  "
                  f"diff {d:+.4f} [{lo:+.4f}, {hi:+.4f}]  "
                  f"embedding changed in {a['frac_embedding_changed']:.1%} of pairs")
        print(f"  mean diff over folds: {np.mean(dd):+.4f} ± {np.std(dd):.4f}")

    if os.path.exists(args.baselines):
        b = pd.read_csv(args.baselines, sep="\t")
        print("\nCPU BASELINES, mean per-fold AUROC (same version/scheme)")
        for (v, s), _ in r.groupby(["version", "scheme"]):
            bb = b[(b.version == v) & (b.scheme == s)]
            print(f"  {v}/{s}: " + "  ".join(f"{m}={fm:.3f}" for m, fm in zip(bb.model, bb.fold_mean)))


if __name__ == "__main__":
    main()
