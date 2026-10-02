#!/usr/bin/env python3
"""
Frozen HyenaDNA-medium-450k embeddings for eQTLP-v2 pairs.

For each pair: the mean-pooled last-layer hidden state (the representation the
paper's Methods feed to its classifier) for ref and alt, in two reading orders:

  fwd  the published orientation (parse_eQTL). For most pairs the variant sits
       ~500 bp from the END, so a causal model lets the allele touch only the
       last few hundred positions.
  rev  the reverse complement of fwd: the variant comes first, and every
       downstream position can see it.

Inputs are unpadded by default. --pad published right-pads with N to 450 kb as
the published loader does (fwd only in that case).

Resumable: pairs are processed in shards and finished shards are skipped.

Usage
-----
  python -X utf8 analysis/hyena_embed.py --version matched --limit 20     # smoke test
  python -X utf8 analysis/hyena_embed.py --version matched                # full run
  python -X utf8 analysis/hyena_embed.py --version matched --merge        # -> embeddings.npz
"""
import argparse
import glob
import math
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

TOK = np.full(256, 11, np.int64)          # HyenaDNA char tokenizer: A C G T N -> 7..11
for _i, _ch in enumerate("ACGTN"):
    TOK[ord(_ch)] = 7 + _i
KEYS = ("ref_fwd", "alt_fwd", "ref_rev", "alt_rev")


@torch.inference_mode()
def embed(model, seq, device):
    ids = torch.from_numpy(TOK[np.frombuffer(seq.encode(), np.uint8)]).unsqueeze(0).to(device)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        h = backbone(model.backbone, ids, False)
    return h.float().mean(1).squeeze(0).cpu().numpy()


def merge(out):
    shards = sorted(glob.glob(os.path.join(out, "shard_*.npz")))
    parts = [dict(np.load(s, allow_pickle=True)) for s in shards]
    merged = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    path = os.path.join(out, "embeddings.npz")
    np.savez(path, **merged)
    print(f"merged {len(shards)} shards, {len(merged['y']):,} pairs -> {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v2", default="data/eQTL_v2")
    ap.add_argument("--fasta", default="data/eQTL/seqs/hg38.fa")
    ap.add_argument("--model", default="data/models/hyenadna-medium-450k-seqlen")
    ap.add_argument("--version", choices=("matched", "all"), default="matched")
    ap.add_argument("--pad", choices=("none", "published"), default="none")
    ap.add_argument("--shard", type=int, default=250)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="data/eQTL_v2/hyena_emb")
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--mem-fraction", type=float, default=0.5,
                    help="allocator cap as a fraction of VRAM; leave room for other apps")
    ap.add_argument("--max-shards", type=int, default=0,
                    help="stop after this many shards (exit code 3 if more remain); "
                         "run_hyena_embed.ps1 relaunches a fresh process")
    args = ap.parse_args()

    out = os.path.join(args.out, f"{args.version}_{args.pad}" + (f"_lim{args.limit}" if args.limit else ""))
    if args.merge:
        merge(out)
        return
    os.makedirs(out, exist_ok=True)

    pairs = pd.read_csv(os.path.join(args.v2, "pairs.tsv"), sep="\t")
    if args.version == "matched":
        pairs = pairs[pairs["matched"] == 1]
    pairs = pairs.reset_index(drop=True)
    if args.limit:
        pairs = pairs.sample(args.limit, random_state=0).reset_index(drop=True)
    both = args.pad == "none"

    dev = torch.device("cuda")
    # Windows can spill GPU memory into system RAM ("sysmem fallback") when the card
    # is oversubscribed; with other apps on the GPU that slowed runs 2x and got them
    # killed for low RAM. A 450k sequence needs ~5.1 GB with the chunked FFT.
    torch.cuda.set_per_process_memory_fraction(args.mem_fraction, 0)
    model, _ = load_hyena(args.model, dev)
    model.eval()
    ds = EQTLDataset(pairs, args.fasta, "refalt")

    n_shards = math.ceil(len(pairs) / args.shard)
    todo = [k for k in range(n_shards) if not os.path.exists(os.path.join(out, f"shard_{k:04d}.npz"))]
    print(f"{len(pairs):,} pairs, {n_shards} shards, {len(todo)} to do -> {out}", flush=True)
    t0, done = time.time(), 0
    for k in todo[:args.max_shards or None]:
        torch.cuda.reset_peak_memory_stats()
        rows = range(k * args.shard, min((k + 1) * args.shard, len(pairs)))
        emb = {key: [] for key in KEYS}
        meta = {"region_id": [], "gene_id": [], "y": [], "length": [], "var_pos": []}
        for i in rows:
            r = pairs.iloc[i]
            ref, alt, pos = ds.build_strings(r, r.allele2)
            ref, alt = ref[:SEQ_LEN], alt[:SEQ_LEN]
            if args.pad == "published":
                ref, alt = ref.ljust(SEQ_LEN, "N"), alt.ljust(SEQ_LEN, "N")
            torch.cuda.empty_cache()        # variable lengths fragment the cache; keep it small
            emb["ref_fwd"].append(embed(model, ref, dev))
            emb["alt_fwd"].append(embed(model, alt, dev))
            if both:
                emb["ref_rev"].append(embed(model, ref.translate(RC)[::-1], dev))
                emb["alt_rev"].append(embed(model, alt.translate(RC)[::-1], dev))
            for key, v in (("region_id", r.region_id), ("gene_id", r.gene_id), ("y", int(r.y)),
                           ("length", len(ref)), ("var_pos", pos)):
                meta[key].append(v)
        np.savez(os.path.join(out, f"shard_{k:04d}.npz"),
                 **{key: np.stack(v).astype(np.float32) for key, v in emb.items() if v},
                 **{key: np.array(v) for key, v in meta.items()})
        done += len(rows)
        rate = (time.time() - t0) / done
        left = sum(min((j + 1) * args.shard, len(pairs)) - j * args.shard for j in todo[todo.index(k) + 1:])
        print(f"shard {k + 1}/{n_shards}  {done:,} pairs in {(time.time() - t0) / 60:.1f} min  "
              f"({rate:.2f} s/pair)  ETA {left * rate / 60:.0f} min  "
              f"peak GPU reserved {torch.cuda.max_memory_reserved() / 2**30:.1f} GB", flush=True)
    remaining = [k for k in range(n_shards) if not os.path.exists(os.path.join(out, f"shard_{k:04d}.npz"))]
    if remaining:
        print(f"{len(remaining)} shards remain", flush=True)
        sys.exit(3)
    merge(out)


if __name__ == "__main__":
    main()
