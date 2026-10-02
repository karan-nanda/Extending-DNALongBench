#!/usr/bin/env python3
"""
Build eQTLP-v2 from the published DNALongBench eQTL tables.

  1. Pool the 9 tissue tables into unique (variant, gene) pairs. A pair is
     positive if it is a SuSiE positive in any tissue; the ~111 pairs labelled
     positive in one tissue and negative in another are dropped.
  2. Distance-match: for each positive, take the unmatched negative with the
     nearest log10 distance to the TSS on the same side (upstream/downstream),
     within --caliper. Positives with no negative in range are dropped. This is
     the `matched` flag; `all` is every pair.
  3. Three 5-fold schemes, computed once on the full pool so both versions share
     folds:
        fold_random  stratified on label (what the published split amounts to)
        fold_gene    whole genes held out; genes that share a candidate variant
                     are kept together so no variant crosses folds
        fold_chrom   whole chromosomes held out, balanced on positives

The published loader masks positive eQTLs (the "blacklist" is exactly the
8,501 positives) between each candidate and its TSS -- label information in
the input. v2 does not use that blacklist; see `mask_count` in
eqtl_v2_baselines.py for how much it leaks.

Usage
-----
  python -X utf8 analysis/build_eqtl_v2.py --root data/eQTL --out data/eQTL_v2
  # loader-format files for one split (DNALongBench parse_eQTL columns + subset):
  python -X utf8 analysis/build_eqtl_v2.py --out data/eQTL_v2 --export matched chrom 0
"""
import argparse
import bisect
import glob
import os

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

TSS_FLANK = 3000        # parse_eQTL defaults from the published configs
REGION_FLANK = 500
SEQ_LEN_CUTOFF = 450_000
N_FOLDS = 5
SCHEMES = ("random", "gene", "chrom")
LOADER_COLS = ["gene_chrom", "gene_start", "gene_end", "region_chrom", "region_start",
               "region_end", "gene_id", "gene_strand", "region_id", "target", "subset",
               "allele1", "allele2", "distance_to_tss"]


def load_pairs(root):
    frames = []
    for path in sorted(glob.glob(os.path.join(root, "targets", "*.data.tsv"))):
        tissue = os.path.basename(path).split(".")[0]
        if tissue == "combined_data":
            continue
        d = pd.read_csv(path, sep="\t")
        d["tissue"] = tissue
        frames.append(d)
    a = pd.concat(frames, ignore_index=True)
    a = a[a["gene_chrom"] == a["region_chrom"]].reset_index(drop=True)

    # same window arithmetic as parse_eQTL in dnalongbench/utils.py
    tss = np.where(a["gene_strand"] == "+", a["gene_start"], a["gene_end"] - 1)
    t0, t1 = tss - TSS_FLANK, tss + 1 + TSS_FLANK
    r0, r1 = a["region_start"].values - REGION_FLANK, a["region_end"].values + REGION_FLANK
    gap = np.maximum(0, np.maximum(t0, r0) - np.minimum(t1, r1))
    a["tss"] = tss
    a = a[gap <= SEQ_LEN_CUTOFF].reset_index(drop=True)

    key = ["region_id", "gene_id"]
    agg = a.groupby(key).agg(
        n_pos=("target", lambda s: int((s == "positive").sum())),
        n_rows=("target", "size"),
        tissues=("tissue", lambda s: ";".join(sorted(set(s)))),
    )
    pairs = a.drop_duplicates(key).set_index(key).drop(columns=["tissue", "subset"]).join(agg)
    pairs = pairs.reset_index()
    conflict = (pairs["n_pos"] > 0) & (pairs["n_pos"] < pairs["n_rows"])
    n_conflict = int(conflict.sum())
    pairs = pairs[~conflict].reset_index(drop=True)
    pairs["y"] = (pairs["n_pos"] > 0).astype(int)
    pairs["target"] = np.where(pairs["y"] == 1, "positive", "negative")
    strand = np.where(pairs["gene_strand"] == "+", 1, -1)
    pairs["signed_dist"] = (pairs["region_start"] - pairs["tss"]) * strand   # <0 = upstream
    pairs["dist"] = pairs["signed_dist"].abs()
    pairs["updown"] = np.where(pairs["signed_dist"] < 0, "up", "down")
    return pairs, n_conflict


def distance_match(pairs, caliper, seed):
    """Greedy 1:1 nearest-neighbour match on log10 distance within up/down."""
    rng = np.random.default_rng(seed)
    logd = np.log10(pairs["dist"].values + 1)
    matched = np.zeros(len(pairs), bool)
    for side in ("up", "down"):
        side_mask = (pairs["updown"] == side).values
        neg = np.where(side_mask & (pairs["y"].values == 0))[0]
        neg = neg[np.argsort(logd[neg])]
        neg_d = list(logd[neg])
        neg_i = list(neg)
        pos = np.where(side_mask & (pairs["y"].values == 1))[0]
        for p in rng.permutation(pos):
            k = bisect.bisect_left(neg_d, logd[p])
            best = None
            for j in (k - 1, k):
                if 0 <= j < len(neg_d) and abs(neg_d[j] - logd[p]) <= caliper:
                    if best is None or abs(neg_d[j] - logd[p]) < abs(neg_d[best] - logd[p]):
                        best = j
            if best is None:
                continue
            matched[p] = matched[neg_i[best]] = True
            del neg_d[best], neg_i[best]
    return matched


def _balanced_assign(group_sizes, group_pos, n_folds):
    """Largest groups first, each to the fold with the fewest positives."""
    order = sorted(group_sizes, key=lambda g: (-group_pos[g], -group_sizes[g], str(g)))
    fold_pos, fold_n, assign = [0] * n_folds, [0] * n_folds, {}
    for g in order:
        f = min(range(n_folds), key=lambda i: (fold_pos[i], fold_n[i]))
        assign[g] = f
        fold_pos[f] += group_pos[g]
        fold_n[f] += group_sizes[g]
    return assign


def gene_components(pairs):
    """Union genes that share a candidate variant."""
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for _, grp in pairs.groupby("region_id")["gene_id"]:
        genes = list(grp)
        for g in genes[1:]:
            ra, rb = find(genes[0]), find(g)
            if ra != rb:
                parent[ra] = rb
    return pairs["gene_id"].map(find)


def assign_folds(pairs, seed):
    skf = StratifiedKFold(N_FOLDS, shuffle=True, random_state=seed)
    fold = np.zeros(len(pairs), int)
    for k, (_, te) in enumerate(skf.split(pairs, pairs["y"])):
        fold[te] = k
    pairs["fold_random"] = fold

    pairs["gene_group"] = gene_components(pairs)
    sizes = pairs.groupby("gene_group").size().to_dict()
    pos = pairs.groupby("gene_group")["y"].sum().to_dict()
    assign = _balanced_assign(sizes, pos, N_FOLDS)
    pairs["fold_gene"] = pairs["gene_group"].map(assign)

    sizes = pairs.groupby("region_chrom").size().to_dict()
    pos = pairs.groupby("region_chrom")["y"].sum().to_dict()
    assign = _balanced_assign(sizes, pos, N_FOLDS)
    pairs["fold_chrom"] = pairs["region_chrom"].map(assign)
    return pairs


def leakage(pairs, fold_col, locus_kb=100):
    """Mean over folds of the fraction of test pairs whose gene / variant /
    100-kb locus also appears in that fold's training pairs."""
    locus = pairs["region_chrom"] + ":" + (pairs["region_start"] // (locus_kb * 1000)).astype(str)
    out = {"gene": [], "variant": [], "locus": []}
    for k in range(N_FOLDS):
        te, tr = pairs[fold_col] == k, pairs[fold_col] != k
        out["gene"].append(pairs.loc[te, "gene_id"].isin(set(pairs.loc[tr, "gene_id"])).mean())
        out["variant"].append(pairs.loc[te, "region_id"].isin(set(pairs.loc[tr, "region_id"])).mean())
        out["locus"].append(locus[te].isin(set(locus[tr])).mean())
    return {k: float(np.mean(v)) for k, v in out.items()}


def export_split(out_dir, version, scheme, fold):
    """Write one split in DNALongBench parse_eQTL format: test = fold,
    valid = next fold, train = the rest."""
    pairs = pd.read_csv(os.path.join(out_dir, "pairs.tsv"), sep="\t")
    if version == "matched":
        pairs = pairs[pairs["matched"] == 1]
    f = pairs[f"fold_{scheme}"]
    pairs["subset"] = np.where(f == fold, "test",
                               np.where(f == (fold + 1) % N_FOLDS, "valid", "train"))
    pairs["distance_to_tss"] = pairs["dist"]
    name = f"{version}_{scheme}_f{fold}"
    os.makedirs(os.path.join(out_dir, "targets"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "config"), exist_ok=True)
    pairs[LOADER_COLS].to_csv(os.path.join(out_dir, "targets", f"{name}.data.tsv"),
                              sep="\t", index=False)
    with open(os.path.join(out_dir, "config", f"gtex_hg38.{name}.config"), "w") as fh:
        fh.write("organism\thuman\tstring\n"
                 "genome_fa\t../eQTL/seqs/hg38.fa\tstring\n"
                 f"seq_len_cutoff\t{SEQ_LEN_CUTOFF}\tint\n"
                 f"tss_flank_upstream\t{TSS_FLANK}\tint\n"
                 f"tss_flank_downstream\t{TSS_FLANK}\tint\n"
                 f"region_flank_upstream\t{REGION_FLANK}\tint\n"
                 f"region_flank_downstream\t{REGION_FLANK}\tint\n"
                 f"eQTL_file\ttargets/{name}.data.tsv\tstring\n")
    print(f"wrote {name}: " + ", ".join(f"{s}={int((pairs['subset'] == s).sum())}"
                                        for s in ("train", "valid", "test")))
    print("note: the published loader also requires eQTL_tabix_file; point it at an empty "
          "tabix-indexed BED to disable label-based masking.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/eQTL")
    ap.add_argument("--out", default="data/eQTL_v2")
    ap.add_argument("--caliper", type=float, default=0.05,
                    help="max |log10 distance| gap for a matched pair (0.05 = ~12%%)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--export", nargs=3, metavar=("VERSION", "SCHEME", "FOLD"))
    args = ap.parse_args()

    if args.export:
        v, s, k = args.export
        export_split(args.out, v, s, int(k))
        return

    pairs, n_conflict = load_pairs(args.root)
    pairs["matched"] = distance_match(pairs, args.caliper, args.seed).astype(int)
    pairs = assign_folds(pairs, args.seed)

    os.makedirs(args.out, exist_ok=True)
    pairs.to_csv(os.path.join(args.out, "pairs.tsv"), sep="\t", index=False)

    L = []
    A = L.append
    A(f"eQTLP-v2 build  |  source {args.root}  |  caliper {args.caliper}  |  seed {args.seed}")
    A(f"pooled pairs: {len(pairs):,} ({int(pairs.y.sum()):,} positive); "
      f"dropped {n_conflict} tissue-conflicting pairs")
    m = pairs[pairs.matched == 1]
    A(f"matched pairs: {len(m):,} ({int(m.y.sum()):,} positive, "
      f"{int(pairs.y.sum() - m.y.sum()):,} positives unmatched)")
    A(f"genes: {pairs.gene_id.nunique():,}  gene groups (shared variants merged): "
      f"{pairs.gene_group.nunique():,}  largest group: {pairs.gene_group.value_counts().iloc[0]} pairs")
    A("")
    A("distance-only AUROC (score = -distance; 0.5 = no distance signal)")
    for name, d in (("all", pairs), ("matched", m)):
        A(f"  {name:<8} {roc_auc_score(d.y, -d.dist):.3f}   median kb pos/neg: "
          f"{d[d.y == 1].dist.median() / 1e3:.1f} / {d[d.y == 0].dist.median() / 1e3:.1f}")
    A("")
    A("folds: positives per fold (all | matched)")
    for s in SCHEMES:
        col = f"fold_{s}"
        pa = pairs.groupby(col)["y"].sum().tolist()
        pm = m.groupby(col)["y"].sum().tolist()
        A(f"  {s:<7} {pa}  |  {pm}")
    A("")
    A("train->test leakage, mean over folds (fraction of test pairs; all version)")
    A(f"  {'scheme':<8}{'gene':>8}{'variant':>9}{'locus100kb':>12}")
    for s in SCHEMES:
        lk = leakage(pairs, f"fold_{s}")
        A(f"  {s:<8}{lk['gene']:>8.3f}{lk['variant']:>9.3f}{lk['locus']:>12.3f}")
    A("")
    A("chromosome folds: " + "; ".join(
        f"f{k}=" + ",".join(sorted(pairs.loc[pairs.fold_chrom == k, 'region_chrom'].unique(),
                                   key=lambda c: (len(c), c)))
        for k in range(N_FOLDS)))
    A(f"\nwritten: {os.path.join(args.out, 'pairs.tsv')}")
    report = "\n".join(L)
    with open(os.path.join(args.out, "build_report.txt"), "w", encoding="utf-8") as fh:
        fh.write(report + "\n")
    print(report)


if __name__ == "__main__":
    main()
