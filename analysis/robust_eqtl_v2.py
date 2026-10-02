#!/usr/bin/env python3
"""
Robustness checks R3 and R4 for eQTLP-v2 (matrix 10, 11.5, 12). CPU only.

R3  the local-composition floor (0.599) across
      - 7 matchings: matching seeds 0-4 at caliper 0.05, plus calipers 0.02 and 0.1
      - 10 random chromosome partitions into 5 folds (balanced on positives)
      - a different GBT seed per partition
R4  the "no allele use" result for frozen HyenaDNA and Caduceus embeddings, across
      - fold schemes: the fixed chrom folds, 10 random chrom partitions, gene, random
      - logistic C in {0.001, 0.01, 0.1, 1, 10} (fixed chrom folds)
      - nonlinear allele-only probes (gradient boosting, MLP) on e_alt - e_ref and on
        |e_alt - e_ref| (fixed chrom and gene folds)
      - a paired bootstrap CI for AUROC(refalt) - AUROC(ref_copy)
    and compares each FM with the floor on the same partitions (seed-0 matching, the
    pairs that were embedded).

Usage:  python -X utf8 analysis/robust_eqtl_v2.py [--part R3|R4]
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from build_eqtl_v2 import _balanced_assign, distance_match  # noqa: E402
from eqtl_v2_baselines import oof_scores  # noqa: E402
from hyena_probe import fit  # noqa: E402

N_FOLDS, N_PART = 5, 10
OUT = "analysis/results/robustness"
EMB = {"HyenaDNA": "data/eQTL_v2/hyena_emb/matched_none/embeddings.npz",
       "Caduceus": "data/eQTL_v2/caduceus_emb/matched_none/embeddings.npz"}


def random_chrom_folds(pairs, seed):
    """Chromosomes to 5 folds, balanced on positives, in a random tie-break order."""
    rng = np.random.default_rng(seed)
    chroms = pairs["region_chrom"].unique()
    jitter = dict(zip(chroms, rng.random(len(chroms))))
    sizes = pairs.groupby("region_chrom").size().to_dict()
    pos = pairs.groupby("region_chrom")["y"].sum().to_dict()
    # perturb the greedy order: random +-20% on the positive counts used for ordering
    pos_j = {c: pos[c] * (0.8 + 0.4 * jitter[c]) for c in chroms}
    assign = _balanced_assign(sizes, pos_j, N_FOLDS)
    return pairs["region_chrom"].map(assign).values


def fold_mean_auc(y, s, fold):
    return float(np.mean([roc_auc_score(y[fold == k], s[fold == k]) for k in range(N_FOLDS)]))


def r3(pairs, X):
    rows = []
    matchings = [(s, 0.05) for s in range(5)] + [(0, 0.02), (0, 0.1)]
    for mseed, cal in matchings:
        sel = distance_match(pairs, cal, mseed)
        p, x = pairs[sel].reset_index(drop=True), X[sel].reset_index(drop=True)
        for part in range(N_PART):
            p["fold_r"] = random_chrom_folds(p, 100 + part)
            y = p["y"].values
            for m in ("distance", "composition", "dist+comp+allele"):
                s = oof_scores(p, x, "fold_r", m, part)
                rows.append({"match_seed": mseed, "caliper": cal, "n_pairs": int(y.sum()),
                             "partition": part, "model": m,
                             "fold_mean_auroc": fold_mean_auc(y, s, p["fold_r"].values)})
        last = pd.DataFrame(rows)
        last = last[(last.match_seed == mseed) & (last.caliper == cal)]
        print(f"matching seed {mseed} caliper {cal}: " + "  ".join(
            f"{m} {g.fold_mean_auroc.mean():.3f} [{g.fold_mean_auroc.min():.3f}-{g.fold_mean_auroc.max():.3f}]"
            for m, g in last.groupby("model")), flush=True)
    r = pd.DataFrame(rows)
    r.to_csv(os.path.join(OUT, "R3_floor.tsv"), sep="\t", index=False)
    print("\nR3 summary: mean [min-max] of fold-mean AUROC over matchings x partitions")
    print(r.groupby("model").fold_mean_auroc.agg(["mean", "std", "min", "max"]).round(3).to_string())


def probe_scores(er, ea, y, fold, C):
    s_ra, s_copy, s_diff = np.zeros(len(y)), np.zeros(len(y)), np.zeros(len(y))
    X_ra, X_copy, d = np.hstack([er, ea]), np.hstack([er, er]), ea - er
    for k in range(N_FOLDS):
        te, tr = fold == k, fold != k
        m = fit(X_ra[tr], y[tr], C)
        s_ra[te] = m.predict_proba(X_ra[te])[:, 1]
        s_copy[te] = m.predict_proba(X_copy[te])[:, 1]
        s_diff[te] = fit(d[tr], y[tr], C).predict_proba(d[te])[:, 1]
    return s_ra, s_copy, s_diff


def nonlinear_diff(d, y, fold, kind, seed=0):
    s = np.zeros(len(y))
    for k in range(N_FOLDS):
        te, tr = fold == k, fold != k
        if kind == "gbt":
            m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=15,
                                               l2_regularization=1.0, random_state=seed)
        else:
            m = make_pipeline(StandardScaler(), MLPClassifier((256, 64), alpha=1e-3, max_iter=300,
                                                              early_stopping=True, random_state=seed))
        s[te] = m.fit(d[tr], y[tr]).predict_proba(d[te])[:, 1]
    return s


def paired_delta_ci(y, a, b, fold, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(n):
        idx = np.concatenate([rng.choice(np.flatnonzero(fold == k), (fold == k).sum()) for k in range(N_FOLDS)])
        deltas.append(fold_mean_auc(y[idx], a[idx], fold[idx]) - fold_mean_auc(y[idx], b[idx], fold[idx]))
    return np.quantile(deltas, [0.025, 0.975])


def r4(pairs, X):
    mp = pairs[pairs["matched"] == 1].reset_index(drop=True)
    xm = X[(pairs["matched"] == 1).values].reset_index(drop=True)
    y = mp["y"].values
    schemes = {"chrom_fixed": mp["fold_chrom"].values, "gene": mp["fold_gene"].values,
               "random": mp["fold_random"].values}
    for part in range(N_PART):
        schemes[f"chrom_rand{part}"] = random_chrom_folds(mp, 100 + part)
    floor = {}
    for name, fold in schemes.items():
        mp["fold_tmp"] = fold
        floor[name] = fold_mean_auc(y, oof_scores(mp, xm, "fold_tmp", "dist+comp+allele", 0), fold)

    rows = []
    for model, path in EMB.items():
        z = np.load(path, allow_pickle=True)
        assert (z["region_id"] == mp["region_id"].values).all()
        er = np.hstack([z["ref_fwd"], z["ref_rev"]])
        ea = np.hstack([z["alt_fwd"], z["alt_rev"]])
        for name, fold in schemes.items():
            Cs = (0.001, 0.01, 0.1, 1, 10) if name == "chrom_fixed" else (0.1,)
            for C in Cs:
                s_ra, s_copy, s_diff = probe_scores(er, ea, y, fold, C)
                row = {"model": model, "scheme": name, "C": C,
                       "refalt": fold_mean_auc(y, s_ra, fold), "ref_copy": fold_mean_auc(y, s_copy, fold),
                       "diff_only_logit": fold_mean_auc(y, s_diff, fold), "floor": floor[name]}
                row["delta"] = row["refalt"] - row["ref_copy"]
                if name == "chrom_fixed" and C == 0.1:
                    row["delta_ci_lo"], row["delta_ci_hi"] = paired_delta_ci(y, s_ra, s_copy, fold)
                if name in ("chrom_fixed", "gene") and C == 0.1:
                    d = ea - er
                    for kind in ("gbt", "mlp"):
                        row[f"diff_only_{kind}"] = fold_mean_auc(y, nonlinear_diff(d, y, fold, kind), fold)
                        row[f"absdiff_only_{kind}"] = fold_mean_auc(y, nonlinear_diff(np.abs(d), y, fold, kind), fold)
                rows.append(row)
                print("  ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()),
                      flush=True)
    r = pd.DataFrame(rows)
    r.to_csv(os.path.join(OUT, "R4_probes.tsv"), sep="\t", index=False)
    rp = r[r.scheme.str.startswith("chrom_rand")]
    print("\nR4 over 10 random chromosome partitions (C=0.1):")
    for model, g in rp.groupby("model"):
        print(f"  {model}: refalt {g.refalt.mean():.3f} [{g.refalt.min():.3f}-{g.refalt.max():.3f}]  "
              f"max |delta| {g.delta.abs().max():.4f}  diff_only {g.diff_only_logit.mean():.3f}  "
              f"below floor in {(g.refalt < g.floor).sum()}/{len(g)} partitions "
              f"(floor mean {g.floor.mean():.3f})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=("R3", "R4", "all"), default="all")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    pairs = pd.read_csv("data/eQTL_v2/pairs.tsv", sep="\t")
    X = pd.read_csv("data/eQTL_v2/features.tsv.gz", sep="\t", index_col=0)
    assert len(X) == len(pairs)
    if args.part in ("R3", "all"):
        r3(pairs, X)
    if args.part in ("R4", "all"):
        r4(pairs, X)


if __name__ == "__main__":
    main()
