#!/usr/bin/env python3
"""
Linear probe on frozen HyenaDNA embeddings (analysis/hyena_embed.py), cross-validated
on the eQTLP-v2 folds.

For each reading order (fwd, rev, fwd+rev) and fold:
  refalt      logistic regression on [e_ref, e_alt] -- the Methods' classifier on
              frozen features
  ref_copy    the SAME fitted model with e_alt replaced by e_ref at test time
              (the plan's ablation); refalt - ref_copy > 0 means the allele is used
  ref_only    trained on e_ref alone
  diff_only   trained on e_alt - e_ref alone; ~0.5 means the pooled embedding carries
              no usable allele information
Also reports how far the SNP moves the pooled embedding, ||e_alt - e_ref|| / ||e_ref||.

Usage:  python -X utf8 analysis/hyena_probe.py --emb data/eQTL_v2/hyena_emb/matched_none/embeddings.npz
"""
import argparse
import os

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

N_FOLDS = 5


def fit(X, y, C):
    return make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=5000)).fit(X, y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb", required=True)
    ap.add_argument("--v2", default="data/eQTL_v2")
    ap.add_argument("--scheme", choices=("random", "gene", "chrom"), default="chrom")
    ap.add_argument("--C", type=float, default=0.1)
    args = ap.parse_args()

    z = np.load(args.emb, allow_pickle=True)
    pairs = pd.read_csv(os.path.join(args.v2, "pairs.tsv"), sep="\t")
    key = pd.DataFrame({"region_id": z["region_id"], "gene_id": z["gene_id"], "i": np.arange(len(z["y"]))})
    m = key.merge(pairs[["region_id", "gene_id", f"fold_{args.scheme}"]], on=["region_id", "gene_id"])
    assert len(m) == len(key), "embedding pairs missing from pairs.tsv"
    m = m.sort_values("i")
    fold, y = m[f"fold_{args.scheme}"].values, z["y"]

    orients = {"fwd": ("ref_fwd", "alt_fwd")}
    if "ref_rev" in z.files:
        orients["rev"] = ("ref_rev", "alt_rev")
    emb = {o: (z[r], z[a]) for o, (r, a) in orients.items()}
    if "rev" in emb:
        emb["fwd+rev"] = (np.hstack([emb["fwd"][0], emb["rev"][0]]),
                          np.hstack([emb["fwd"][1], emb["rev"][1]]))

    print(f"{len(y):,} pairs ({int(y.sum()):,} positive), scheme {args.scheme}, logistic C={args.C}\n")
    print("HOW FAR THE SNP MOVES THE POOLED EMBEDDING  ||e_alt - e_ref|| / ||e_ref||")
    for o, (er, ea) in emb.items():
        rel = np.linalg.norm(ea - er, axis=1) / np.linalg.norm(er, axis=1)
        print(f"  {o:<8} median {np.median(rel):.2e}   90th pct {np.quantile(rel, .9):.2e}   "
              f"exactly zero {np.mean(rel == 0):.1%}")

    rows = []
    for o, (er, ea) in emb.items():
        for k in range(N_FOLDS):
            te, tr = fold == k, fold != k
            X_ra = np.hstack([er, ea])
            model = fit(X_ra[tr], y[tr], args.C)
            s_ra = model.predict_proba(X_ra[te])[:, 1]
            s_copy = model.predict_proba(np.hstack([er, er])[te])[:, 1]
            s_ref = fit(er[tr], y[tr], args.C).predict_proba(er[te])[:, 1]
            d = ea - er
            s_diff = fit(d[tr], y[tr], args.C).predict_proba(d[te])[:, 1]
            rows.append({"orient": o, "fold": k,
                         "refalt": roc_auc_score(y[te], s_ra),
                         "ref_copy": roc_auc_score(y[te], s_copy),
                         "ref_only": roc_auc_score(y[te], s_ref),
                         "diff_only": roc_auc_score(y[te], s_diff)})
    r = pd.DataFrame(rows)
    r["refalt_minus_copy"] = r["refalt"] - r["ref_copy"]
    cols = ["refalt", "ref_copy", "refalt_minus_copy", "ref_only", "diff_only"]
    print("\nTEST AUROC, mean over folds ± SD")
    g = r.groupby("orient")[cols]
    summary = g.mean().round(4).astype(str) + " ± " + g.std().round(4).astype(str)
    print(summary.reindex([o for o in ("fwd", "rev", "fwd+rev") if o in emb]).to_string())
    out = os.path.join(os.path.dirname(args.emb), f"probe_{args.scheme}.tsv")
    r.to_csv(out, sep="\t", index=False)
    print(f"\nper-fold results: {out}")


if __name__ == "__main__":
    main()
