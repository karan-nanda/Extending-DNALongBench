#!/usr/bin/env python3
"""
Clinical evaluation set (eval-only, matrix 8.3): ClinVar regulatory-core pathogenic SNVs
vs matched benign SNVs, written as eQTL-style pairs so hyena_embed.py and
caduceus_embed.py can embed them unchanged (--v2 data/clinical).

Positives: the census regulatory core (census_1star/clinvar_filtered.tsv): location in
the census ELIGIBLE set, not overlapping CDS or a small structural RNA; SNVs only.
Negatives: benign / likely benign SNVs (>= --min-stars) passing the same filters,
re-derived here from the ClinVar VCF, variant_summary and GENCODE.

Each positive gets one benign match, chosen in this order:
  1. same gene and same location class, closest |log distance to TSS|
  2. same gene, any eligible location
  3. same location class, any gene, closest |log distance to TSS|
The level is recorded (match_level) so results can be stratified by it.

The "gene" of a variant is ClinVar's gene symbol, mapped to its GENCODE gene; the input
sequence spans the variant and that gene's TSS exactly as parse_eQTL does (build_strings).
Pairs whose span exceeds 450 kb are dropped (the variant would be cut off).

Usage:
  python -X utf8 analysis/build_clinical_eval.py
"""
import argparse
import gzip
import os
import re
import sys
from collections import defaultdict

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import build_clinvar_census as cc  # noqa: E402

SEQ_LEN = 450_000
TSS_FLANK, REGION_FLANK = 3000, 500     # as eqtl_v2_cnn.build_strings


def load_genes(gtf_path):
    """gene_name -> (chrom, start0, end, strand, gene_id); first (lowest-level) record wins."""
    pat = {k: re.compile(rf'{k} "([^"]+)"') for k in ("gene_id", "gene_name")}
    genes = {}
    with gzip.open(gtf_path, "rt", encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            f = line.split("\t", 8)
            if f[2] != "gene" or "_PAR_Y" in f[8]:
                continue
            name = pat["gene_name"].search(f[8]).group(1)
            if name not in genes:
                genes[name] = (f[0], int(f[3]) - 1, int(f[4]), f[6], pat["gene_id"].search(f[8]).group(1))
    return genes


def benign_pool(args, gtf):
    """Benign/likely-benign SNVs passing the census regulatory-core filters."""
    rows = {}
    with gzip.open(args.vcf, "rt", encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            f = line.rstrip("\n").split("\t")
            info = cc.parse_info(f[7])
            if not cc.is_benign(info.get("CLNSIG", "")):
                continue
            stars = cc.STARS.get(info.get("CLNREVSTAT", ""), 0)
            if stars < args.min_stars or info.get("CLNVC") != "single_nucleotide_variant":
                continue
            if cc.classify(cc.so_terms(info)) not in ("noncoding", "unannotated"):
                continue
            rows[f[2]] = {"chrom": f[0], "pos": int(f[1]), "id": f[2], "ref": f[3], "alt": f[4],
                          "clnsig": info.get("CLNSIG"), "stars": stars,
                          "gene": cc.gene_symbol(info), "location": "other"}
    print(f"benign SNVs, non-coding/unannotated, >= {args.min_stars} star: {len(rows):,}")
    with gzip.open(args.summary, "rt", encoding="utf-8") as fh:
        ix = {h: i for i, h in enumerate(fh.readline().lstrip("#").rstrip("\n").split("\t"))}
        for line in fh:
            f = line.rstrip("\n").split("\t")
            if f[ix["Assembly"]] == "GRCh38" and f[ix["VariationID"]] in rows:
                rows[f[ix["VariationID"]]]["location"] = cc.locate(f[ix["Name"]], cc.SPLICE_WINDOW)[0]
    keep = []
    for r in rows.values():
        if r["location"] not in cc.ELIGIBLE:
            continue
        r["gtf_region"] = cc.gtf_region(gtf, r["chrom"], r["pos"], r["pos"])
        if r["gtf_region"] not in cc.GATE_EXCLUDED_REGIONS:
            keep.append(r)
    print(f"  ... passing the regulatory-core filters: {len(keep):,}")
    return pd.DataFrame(keep)


def to_pairs(df, genes, y):
    """eQTL pair columns (see data/eQTL_v2/pairs.tsv) for build_strings."""
    out = []
    for r in df.itertuples():
        g = genes.get(str(r.gene).split("|")[0])
        chrom = cc.ucsc_chrom(r.chrom)
        if g is None or g[0] != chrom:
            continue
        gchrom, gs, ge, strand, gid = g
        tss = gs if strand == "+" else ge - 1
        v0 = r.pos - 1
        span = max(v0 + 1 + REGION_FLANK, tss + 1 + TSS_FLANK) - min(v0 - REGION_FLANK, tss - TSS_FLANK)
        out.append({"region_id": f"{chrom}_{r.pos}_{r.ref}_{r.alt}_b38", "gene_id": gid,
                    "gene_chrom": gchrom, "gene_start": gs, "gene_end": ge,
                    "region_chrom": chrom, "region_start": v0, "region_end": r.pos,
                    "gene_strand": strand, "allele1": r.ref, "allele2": r.alt, "y": y,
                    "clinvar_id": r.id, "gene": r.gene, "stars": r.stars, "clnsig": r.clnsig,
                    "location": r.location, "gtf_region": r.gtf_region,
                    "dist": abs(v0 - tss), "span": span})
    p = pd.DataFrame(out)
    return p[p["span"] <= SEQ_LEN].reset_index(drop=True)


def match(pos, neg, seed=0):
    rng = np.random.default_rng(seed)
    neg = neg.copy()
    neg["used"] = False
    ld = lambda d: np.log10(np.asarray(d, float) + 1)
    picks, levels = [], []
    for i in rng.permutation(len(pos)):
        p = pos.iloc[i]
        free = ~neg["used"]
        for level, m in ((1, free & (neg["gene"] == p["gene"]) & (neg["location"] == p["location"])),
                         (2, free & (neg["gene"] == p["gene"])),
                         (3, free & (neg["location"] == p["location"]))):
            if m.any():
                cand = neg[m]
                j = cand.index[np.argmin(np.abs(ld(cand["dist"]) - ld(p["dist"])))]
                neg.at[j, "used"] = True
                picks.append((i, j))
                levels.append(level)
                break
    pi, ni = zip(*picks)
    P = pos.iloc[list(pi)].copy()
    N = neg.loc[list(ni)].drop(columns="used").copy()
    P["match_level"] = N["match_level"] = levels
    P["pair_id"] = N["pair_id"] = np.arange(len(P))
    return pd.concat([P, N], ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--census", default="census_1star/clinvar_filtered.tsv")
    ap.add_argument("--vcf", default="clinvar.vcf.gz")
    ap.add_argument("--summary", default="variant_summary.txt.gz")
    ap.add_argument("--gtf", default="gencode.v50.annotation.gtf.gz")
    ap.add_argument("--min-stars", type=int, default=1)
    ap.add_argument("--out", default="data/clinical")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    c = pd.read_csv(args.census, sep="\t", low_memory=False)
    core = c[c["location"].isin(cc.ELIGIBLE) & ~c["gtf_region"].isin(cc.GATE_EXCLUDED_REGIONS)]
    print(f"pathogenic regulatory core: {len(core):,}")
    core = core[core["clnvc"] == "single_nucleotide_variant"]
    print(f"  ... SNVs: {len(core):,}")

    print("loading GENCODE ...")
    gtf = cc.load_gtf(args.gtf)
    genes = load_genes(args.gtf)
    ben = benign_pool(args, gtf)

    pos = to_pairs(core, genes, 1)
    neg = to_pairs(ben, genes, 0)
    print(f"mapped to a GENCODE gene TSS, span <= 450 kb: {len(pos):,} pathogenic, {len(neg):,} benign")
    pairs = match(pos, neg)
    pairs["matched"] = 1
    lv = pairs[pairs.y == 1]["match_level"].value_counts().sort_index()
    print(f"matched pairs: {int((pairs.y == 1).sum()):,}  "
          f"(level 1 same gene+location {lv.get(1, 0)}, 2 same gene {lv.get(2, 0)}, "
          f"3 same location {lv.get(3, 0)})")
    for y, g in pairs.groupby("y"):
        print(f"  y={y}: median distance to TSS {g['dist'].median():,.0f} bp, "
              f"median span {g['span'].median():,.0f} bp, 2+ star {int((g['stars'] >= 2).sum())}")
    path = os.path.join(args.out, "pairs.tsv")
    pairs.to_csv(path, sep="\t", index=False)
    print(f"-> {path}")


if __name__ == "__main__":
    main()
