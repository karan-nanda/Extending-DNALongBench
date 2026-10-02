#!/usr/bin/env python3
"""
Precision check for the frozen HyenaDNA result (matrix 11.5).

The main embeddings ran under bf16 autocast, and the SNP moves the pooled embedding
by only ~1e-4 relative, below bf16 per-element resolution. This re-embeds a random
subset of matched pairs in full fp32 (no autocast, TF32 off) and asks:

  1. How far bf16 is from fp32 on the same sequence (rounding noise).
  2. Whether the bf16 allele difference e_alt - e_ref points the same way as the fp32
     one (cosine). High cosine = bf16 kept the real SNP signal.
  3. Whether a diff_only probe on the fp32 differences still scores ~0.5 (chrom folds,
     same subset, compared with the bf16 differences of the same pairs).

Pairs longer than --max-len are skipped so fp32 fits in memory. Resumable: progress is
saved to <out> every --save-every pairs.

Usage:  python -X utf8 analysis/hyena_fp32_check.py --n 400
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from hyena_memcheck import backbone, load_hyena  # noqa: E402  (also patches fftconv to fp32)
from eqtl_v2_cnn import RC, SEQ_LEN, EQTLDataset  # noqa: E402
from hyena_embed import KEYS, TOK  # noqa: E402
from hyena_probe import N_FOLDS, fit  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402


@torch.inference_mode()
def embed_fp32(model, seq, device):
    ids = torch.from_numpy(TOK[np.frombuffer(seq.encode(), np.uint8)]).unsqueeze(0).to(device)
    h = backbone(model.backbone, ids, False)
    return h.float().mean(1).squeeze(0).cpu().numpy()


def rel(a, b):
    return np.linalg.norm(a - b, axis=1) / np.linalg.norm(b, axis=1)


def cosine(a, b):
    return (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))


def report(sub, bf, fp, fold, C):
    y = sub["y"].values
    print(f"\n{len(y)} pairs ({int(y.sum())} positive), fp32 vs bf16\n")
    for o in ("fwd", "rev"):
        r_bf, a_bf, r_fp, a_fp = bf[f"ref_{o}"], bf[f"alt_{o}"], fp[f"ref_{o}"], fp[f"alt_{o}"]
        noise = rel(r_bf, r_fp)
        shift_fp, shift_bf = rel(a_fp, r_fp), rel(a_bf, r_bf)
        cos = cosine(a_bf - r_bf, a_fp - r_fp)
        print(f"{o}")
        print(f"  bf16 rounding  ||ref_bf16 - ref_fp32|| / ||ref_fp32||   median {np.median(noise):.2e}")
        print(f"  SNP shift fp32 ||alt - ref|| / ||ref||                 median {np.median(shift_fp):.2e}"
              f"   90th pct {np.quantile(shift_fp, .9):.2e}")
        print(f"  SNP shift bf16                                        median {np.median(shift_bf):.2e}")
        print(f"  cosine(diff_bf16, diff_fp32)                          median {np.median(cos):.3f}"
              f"   10th pct {np.quantile(cos, .1):.3f}")
    print(f"\ndiff_only probe, chrom folds on this subset (logistic C={C}), mean ± SD")
    for o in ("fwd", "rev"):
        for name, e in (("bf16", bf), ("fp32", fp)):
            d = e[f"alt_{o}"] - e[f"ref_{o}"]
            aucs = [roc_auc_score(y[fold == k], fit(d[fold != k], y[fold != k], C)
                                  .predict_proba(d[fold == k])[:, 1]) for k in range(N_FOLDS)]
            print(f"  {o} {name}  {np.mean(aucs):.3f} ± {np.std(aucs, ddof=1):.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v2", default="data/eQTL_v2")
    ap.add_argument("--fasta", default="data/eQTL/seqs/hg38.fa")
    ap.add_argument("--model", default="data/models/hyenadna-medium-450k-seqlen")
    ap.add_argument("--emb", default="data/eQTL_v2/hyena_emb/matched_none/embeddings.npz")
    ap.add_argument("--out", default="data/eQTL_v2/hyena_emb/matched_none/fp32_check.npz")
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--max-len", type=int, default=131072)
    ap.add_argument("--save-every", type=int, default=50)
    ap.add_argument("--mem-fraction", type=float, default=0.6)
    ap.add_argument("--C", type=float, default=0.1)
    args = ap.parse_args()

    z = np.load(args.emb, allow_pickle=True)
    pairs = pd.read_csv(os.path.join(args.v2, "pairs.tsv"), sep="\t")
    pairs = pairs[pairs["matched"] == 1].reset_index(drop=True)
    assert (pairs["region_id"].values == z["region_id"]).all(), "embeddings not in pairs.tsv order"
    ok = np.flatnonzero(z["length"] <= args.max_len)
    idx = np.sort(np.random.default_rng(0).choice(ok, size=min(args.n, len(ok)), replace=False))
    sub = pairs.iloc[idx].reset_index(drop=True)
    bf = {k: z[k][idx] for k in KEYS}

    fp = {k: np.full_like(bf[k], np.nan) for k in KEYS}
    if os.path.exists(args.out):
        prev = np.load(args.out)
        if np.array_equal(prev["idx"], idx):
            fp = {k: prev[k] for k in KEYS}
    todo = np.flatnonzero(np.isnan(fp["ref_fwd"][:, 0]))
    print(f"{len(idx)} pairs sampled (length <= {args.max_len:,} of {len(ok):,} eligible), "
          f"{len(todo)} to embed in fp32", flush=True)

    if len(todo):
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        dev = torch.device("cuda")
        torch.cuda.set_per_process_memory_fraction(args.mem_fraction, 0)
        model, _ = load_hyena(args.model, dev)
        model.eval().float()
        ds = EQTLDataset(sub, args.fasta, "refalt")
        t0 = time.time()
        for n, i in enumerate(todo, 1):
            r = sub.iloc[i]
            ref, alt, _ = ds.build_strings(r, r.allele2)
            ref, alt = ref[:SEQ_LEN], alt[:SEQ_LEN]
            torch.cuda.empty_cache()
            fp["ref_fwd"][i] = embed_fp32(model, ref, dev)
            fp["alt_fwd"][i] = embed_fp32(model, alt, dev)
            fp["ref_rev"][i] = embed_fp32(model, ref.translate(RC)[::-1], dev)
            fp["alt_rev"][i] = embed_fp32(model, alt.translate(RC)[::-1], dev)
            if n % args.save_every == 0 or n == len(todo):
                np.savez(args.out, idx=idx, **fp)
                print(f"{n}/{len(todo)} pairs  {(time.time() - t0) / 60:.1f} min  "
                      f"peak GPU reserved {torch.cuda.max_memory_reserved() / 2**30:.1f} GB", flush=True)

    report(sub, bf, fp, sub["fold_chrom"].values, args.C)


if __name__ == "__main__":
    main()
