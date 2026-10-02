#!/usr/bin/env python3
"""
CPU baselines for eQTLP-v2: every (version, scheme) with 5-fold CV, scored on
pooled out-of-fold predictions so every pair is tested exactly once.

Baselines
---------
  distance            score = -distance to TSS, no training
  gene_prior          positive rate of the pair's gene in the training folds
                      (pure memorisation; should collapse under gene/chrom holdout)
  composition         reference sequence around the variant only: GC, CpG o/e,
                      repeat (soft-mask) fraction, dinucleotides -- no distance
  dist+comp           distance features + composition
  dist+comp+allele    + ref/alt base, transition, CpG-disrupting
  dist+mask_count     distance + number of published-blacklist sites (= other
                      positive eQTLs) the loader would mask between variant and
                      TSS -- probes the label leak in the published masking

Usage:  python -X utf8 analysis/eqtl_v2_baselines.py --v2 data/eQTL_v2 --root data/eQTL
"""
import argparse
import bisect
import gzip
import mmap
import os
from collections import defaultdict
from itertools import product

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

N_FOLDS = 5
SCHEMES = ("random", "gene", "chrom")
WINDOWS = (50, 500, 5000)
DINUC = [a + b for a in "ACGT" for b in "ACGT"]
TSS_FLANK, REGION_FLANK = 3000, 500


class Fasta:
    """Minimal mmap reader over an uncompressed FASTA + .fai. Keeps case
    (hg38 is soft-masked: lowercase = repeat)."""

    def __init__(self, path):
        self.idx = {}
        with open(path + ".fai") as fh:
            for line in fh:
                name, length, offset, lb, lbytes = line.split("\t")[:5]
                self.idx[name] = (int(length), int(offset), int(lb), int(lbytes))
        self._fh = open(path, "rb")
        self.mm = mmap.mmap(self._fh.fileno(), 0, access=mmap.ACCESS_READ)

    def fetch(self, chrom, start, end):
        """0-based half-open; out-of-range positions come back as N."""
        length, off, lb, lbytes = self.idx[chrom]
        s, e = max(0, start), min(length, end)
        if s >= e:
            return "N" * (end - start)
        b0 = off + (s // lb) * lbytes + s % lb
        b1 = off + ((e - 1) // lb) * lbytes + (e - 1) % lb + 1
        seq = self.mm[b0:b1].replace(b"\n", b"").replace(b"\r", b"").decode()
        return "N" * (s - start) + seq + "N" * (end - e)


def composition(seq):
    up = seq.upper()
    n = len(up) - up.count("N")
    if n == 0:
        return 0.0, 0.0, 1.0
    gc = (up.count("G") + up.count("C")) / n
    c, g = up.count("C"), up.count("G")
    cpg_oe = up.count("CG") * n / (c * g) if c and g else 0.0
    rep = sum(ch.islower() for ch in seq) / len(seq)
    return gc, cpg_oe, rep


def load_blacklist(path):
    pos = defaultdict(list)
    with gzip.open(path, "rt") as fh:
        for line in fh:
            f = line.split("\t")
            pos[f[0]].append(int(f[1]))
    return {c: sorted(v) for c, v in pos.items()}


def build_features(pairs, fasta, blacklist):
    ref_ok = 0
    rows = []
    for r in pairs.itertuples(index=False):
        feat = {}
        v = r.region_start
        if fasta.fetch(r.region_chrom, v, r.region_end).upper() == r.allele1:
            ref_ok += 1
        for w in WINDOWS:
            gc, cpg, rep = composition(fasta.fetch(r.region_chrom, v - w, v + w + 1))
            feat[f"gc_{w}"], feat[f"cpg_oe_{w}"], feat[f"rep_{w}"] = gc, cpg, rep
        s500 = fasta.fetch(r.region_chrom, v - 500, v + 501).upper()
        denom = max(1, len(s500) - 1)
        for d in DINUC:
            feat[f"di_{d}"] = s500.count(d) / denom
        ctx = fasta.fetch(r.region_chrom, v - 1, v + 2).upper()
        feat["cpg_site"] = int("CG" in ctx)
        ref, alt = r.allele1.upper(), r.allele2.upper()
        for b in "ACGT":
            feat[f"ref_{b}"] = int(ref == b)
            feat[f"alt_{b}"] = int(alt == b)
        feat["transition"] = int({ref, alt} in ({"A", "G"}, {"C", "T"}))
        feat["log_dist"] = np.log10(r.dist + 1)
        feat["upstream"] = int(r.updown == "up")
        # sites the published loader would mask (same query interval as parse_eQTL)
        t0, t1 = r.tss - TSS_FLANK, r.tss + 1 + TSS_FLANK
        r0, r1 = r.region_start - REGION_FLANK, r.region_end + REGION_FLANK
        q0, q1 = (r1, t0) if t0 > r1 else (t1, r0)
        bl = blacklist.get(r.region_chrom, [])
        feat["mask_count"] = max(0, bisect.bisect_left(bl, q1) - bisect.bisect_left(bl, q0)) if q1 > q0 else 0
        rows.append(feat)
    print(f"reference allele matches FASTA: {ref_ok:,} / {len(pairs):,}")
    return pd.DataFrame(rows, index=pairs.index)


FEATURE_SETS = {
    "composition": lambda c: [x for x in c if x.startswith(("gc_", "cpg_oe_", "rep_", "di_"))],
    "dist+comp": lambda c: ["log_dist", "upstream"] + FEATURE_SETS["composition"](c),
    "dist+comp+allele": lambda c: FEATURE_SETS["dist+comp"](c)
    + [x for x in c if x.startswith(("ref_", "alt_"))] + ["transition", "cpg_site"],
    "dist+mask_count": lambda c: ["log_dist", "upstream", "mask_count"],
}


def oof_scores(pairs, X, fold_col, model_name, seed):
    score = np.zeros(len(pairs))
    y = pairs["y"].values
    for k in range(N_FOLDS):
        te = (pairs[fold_col] == k).values
        tr = ~te
        if model_name == "distance":
            score[te] = -pairs["dist"].values[te]
        elif model_name == "gene_prior":
            g = pairs.loc[tr].groupby("gene_id")["y"].agg(["sum", "count"])
            prior = y[tr].mean()
            rate = (g["sum"] + prior) / (g["count"] + 1)   # smoothed
            score[te] = pairs.loc[te, "gene_id"].map(rate).fillna(prior).values
        else:
            cols = FEATURE_SETS[model_name](X.columns)
            clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05,
                                                 max_leaf_nodes=15, l2_regularization=1.0,
                                                 random_state=seed)
            clf.fit(X.loc[tr, cols], y[tr])
            score[te] = clf.predict_proba(X.loc[te, cols])[:, 1]
    return score


def boot_ci(y, s, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() != y[i].max():
            out.append(roc_auc_score(y[i], s[i]))
    return np.percentile(out, [2.5, 97.5])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v2", default="data/eQTL_v2")
    ap.add_argument("--root", default="data/eQTL")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    pairs = pd.read_csv(os.path.join(args.v2, "pairs.tsv"), sep="\t")
    cache = os.path.join(args.v2, "features.tsv.gz")
    if os.path.exists(cache):
        X = pd.read_csv(cache, sep="\t", index_col=0)
    else:
        fasta = Fasta(os.path.join(args.root, "seqs", "hg38.fa"))
        bl = load_blacklist(os.path.join(args.root, "blacklist", "combined_blacklist.bed.gz"))
        X = build_features(pairs, fasta, bl)
        X.to_csv(cache, sep="\t")

    models = ["distance", "gene_prior", "composition", "dist+comp", "dist+comp+allele",
              "dist+mask_count"]
    rows = []
    for version, scheme in product(("all", "matched"), SCHEMES):
        sel = (pairs["matched"] == 1).values if version == "matched" else np.ones(len(pairs), bool)
        p, x = pairs[sel].reset_index(drop=True), X[sel].reset_index(drop=True)
        for m in models:
            s = oof_scores(p, x, f"fold_{scheme}", m, args.seed)
            y = p["y"].values
            lo, hi = boot_ci(y, s, seed=args.seed)
            per_fold = [roc_auc_score(y[p[f"fold_{scheme}"] == k], s[p[f"fold_{scheme}"] == k])
                        for k in range(N_FOLDS)]
            rows.append({"version": version, "scheme": scheme, "model": m,
                         "auroc": roc_auc_score(y, s), "ci_lo": lo, "ci_hi": hi,
                         "fold_mean": np.mean(per_fold), "fold_sd": np.std(per_fold),
                         "auprc": average_precision_score(y, s),
                         "prevalence": y.mean(), "n": len(y), "n_pos": int(y.sum())})
            print(f"{version:<8}{scheme:<8}{m:<18} fold-mean {np.mean(per_fold):.3f} "
                  f"± {np.std(per_fold):.3f}   pooled {rows[-1]['auroc']:.3f} [{lo:.3f}, {hi:.3f}]")

    res = pd.DataFrame(rows)
    os.makedirs("analysis/results", exist_ok=True)
    res.to_csv("analysis/results/eqtl_v2_baselines.tsv", sep="\t", index=False)
    # Primary metric is the mean of per-fold AUROCs. Pooled out-of-fold AUROC mixes
    # fold-level calibration: a score that is constant within a fold (gene_prior on
    # unseen genes) lands below 0.5 purely from fold-to-fold prevalence differences.
    piv = res.assign(cell=res.apply(lambda r: f"{r.fold_mean:.3f} ± {r.fold_sd:.3f}", axis=1))
    print("\nMEAN PER-FOLD AUROC ± SD (primary)")
    print(piv.pivot_table(index="model", columns=["version", "scheme"], values="cell",
                          aggfunc="first").reindex(models).to_string())
    piv = res.assign(cell=res.apply(lambda r: f"{r.auroc:.3f} ({r.ci_lo:.2f}-{r.ci_hi:.2f})", axis=1))
    print("\nPOOLED OUT-OF-FOLD AUROC (95% bootstrap CI)")
    print(piv.pivot_table(index="model", columns=["version", "scheme"], values="cell",
                          aggfunc="first").reindex(models).to_string())
    print("\nwritten: analysis/results/eqtl_v2_baselines.tsv")


if __name__ == "__main__":
    main()
