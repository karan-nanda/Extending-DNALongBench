#!/usr/bin/env python3
"""
Frozen Caduceus-Ph embeddings for eQTLP-v2 pairs (the Caduceus analogue of hyena_embed.py).

Runs inside WSL2 (mamba_ssm has no Windows build); environment from caduceus_env_setup.sh.
For each pair: the mean-pooled last-layer hidden state for ref and alt, in the published
orientation (fwd) and its reverse complement (rev). This is the representation the
released CaduceusEQTL head consumes (mean over positions, ref and alt concatenated).
Caduceus is bidirectional, so unlike HyenaDNA every position can see the variant in
either orientation; Ph is not RC-equivariant, so fwd and rev still differ.

Output has the same layout as hyena_embed.py, so hyena_probe.py reads it unchanged.
Resumable: pairs are processed in shards and finished shards are skipped.

Usage (from the project root, inside WSL):
  ~/cad/bin/python analysis/caduceus_embed.py --limit 20            # smoke test
  ~/cad/bin/python analysis/caduceus_embed.py                       # full run
  ~/cad/bin/python analysis/caduceus_embed.py --merge               # -> embeddings.npz
  ~/cad/bin/python analysis/caduceus_embed.py --fp32-check          # precision check subset

The main run uses --bf16: at 450 kb, fp32 does not fit in 7.2 GB (bf16 peaks at 4.4 GB).
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
from transformers import AutoModel

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from eqtl_v2_cnn import RC, SEQ_LEN, EQTLDataset  # noqa: E402

TOK = np.full(256, 11, np.int64)          # Caduceus char tokenizer: A C G T N -> 7..11 (as HyenaDNA)
for _i, _ch in enumerate("ACGTN"):
    TOK[ord(_ch)] = 7 + _i
KEYS = ("ref_fwd", "alt_fwd", "ref_rev", "alt_rev")


def merge(out):
    shards = sorted(glob.glob(os.path.join(out, "shard_*.npz")))
    parts = [dict(np.load(s, allow_pickle=True)) for s in shards]
    merged = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    path = os.path.join(out, "embeddings.npz")
    np.savez(path, **merged)
    print(f"merged {len(shards)} shards, {len(merged['y']):,} pairs -> {path}")


@torch.inference_mode()
def embed(model, seq, device, amp):
    ids = torch.from_numpy(TOK[np.frombuffer(seq.encode(), np.uint8)]).unsqueeze(0).to(device)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
        h = model(input_ids=ids).last_hidden_state
    return h.float().mean(1).squeeze(0).cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v2", default="data/eQTL_v2")
    ap.add_argument("--fasta", default="data/eQTL/seqs/hg38.fa")
    ap.add_argument("--model", default="data/models/caduceus-ph_seqlen-131k_d_model-256_n_layer-16")
    ap.add_argument("--version", choices=("matched", "all"), default="matched")
    ap.add_argument("--shard", type=int, default=250)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="data/eQTL_v2/caduceus_emb")
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--bf16", action="store_true", help="bf16 autocast (default: full fp32)")
    ap.add_argument("--mem-fraction", type=float, default=0.6)
    ap.add_argument("--fp32-check", action="store_true",
                    help="after a bf16 run: re-embed hyena_fp32_check.py's 400 pairs (<=131 kb) in fp32")
    ap.add_argument("--memtest", action="store_true",
                    help="embed one random full-length (450 kb) sequence and report time and memory")
    args = ap.parse_args()

    out = os.path.join(args.out, f"{args.version}_none" + (f"_lim{args.limit}" if args.limit else ""))
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

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    dev = torch.device("cuda")
    torch.cuda.set_per_process_memory_fraction(args.mem_fraction, 0)
    model = AutoModel.from_pretrained(args.model, trust_remote_code=True).to(dev).eval()
    ds = EQTLDataset(pairs, args.fasta, "refalt")

    if args.fp32_check:
        # Same 400 pairs as hyena_fp32_check.py (same sampling rule on the same pair order),
        # re-embedded in full fp32. Compare on Windows with:
        #   python -X utf8 analysis/hyena_fp32_check.py --emb <out>/embeddings.npz --out <out>/fp32_check.npz
        z = np.load(os.path.join(out, "embeddings.npz"), allow_pickle=True)
        assert (pairs["region_id"].values == z["region_id"]).all()
        ok = np.flatnonzero(z["length"] <= 131072)
        idx = np.sort(np.random.default_rng(0).choice(ok, size=min(400, len(ok)), replace=False))
        fp = {key: [] for key in KEYS}
        t0 = time.time()
        for n, i in enumerate(idx, 1):
            r = pairs.iloc[i]
            ref, alt, _ = ds.build_strings(r, r.allele2)
            for o, (a, b) in (("fwd", (ref, alt)), ("rev", (ref.translate(RC)[::-1], alt.translate(RC)[::-1]))):
                fp[f"ref_{o}"].append(embed(model, a, dev, False))
                fp[f"alt_{o}"].append(embed(model, b, dev, False))
            if n % 100 == 0:
                print(f"fp32 {n}/{len(idx)}  {(time.time() - t0) / 60:.1f} min", flush=True)
        np.savez(os.path.join(out, "fp32_check.npz"), idx=idx,
                 **{key: np.stack(v).astype(np.float32) for key, v in fp.items()})
        print(f"saved {os.path.join(out, 'fp32_check.npz')}")
        return

    if args.memtest:
        ref = "".join(np.random.default_rng(0).choice(list("ACGT"), SEQ_LEN))  # worst-case length
        for name, s in (("fwd", ref), ("rev", ref.translate(RC)[::-1])):
            torch.cuda.reset_peak_memory_stats()
            t = time.time()
            e = embed(model, s, dev, args.bf16)
            torch.cuda.synchronize()
            print(f"{name}: length {len(s):,}  {time.time() - t:.1f} s  "
                  f"peak GPU {torch.cuda.max_memory_allocated() / 2**30:.2f} GB  finite {np.isfinite(e).all()}")
        return

    n_shards = math.ceil(len(pairs) / args.shard)
    todo = [k for k in range(n_shards) if not os.path.exists(os.path.join(out, f"shard_{k:04d}.npz"))]
    print(f"{len(pairs):,} pairs, {n_shards} shards, {len(todo)} to do -> {out}  "
          f"({'bf16' if args.bf16 else 'fp32'})", flush=True)
    t0, done = time.time(), 0
    for k in todo:
        torch.cuda.reset_peak_memory_stats()
        rows = range(k * args.shard, min((k + 1) * args.shard, len(pairs)))
        emb = {key: [] for key in KEYS}
        meta = {"region_id": [], "gene_id": [], "y": [], "length": [], "var_pos": []}
        for i in rows:
            r = pairs.iloc[i]
            ref, alt, pos = ds.build_strings(r, r.allele2)
            ref, alt = ref[:SEQ_LEN], alt[:SEQ_LEN]
            emb["ref_fwd"].append(embed(model, ref, dev, args.bf16))
            emb["alt_fwd"].append(embed(model, alt, dev, args.bf16))
            emb["ref_rev"].append(embed(model, ref.translate(RC)[::-1], dev, args.bf16))
            emb["alt_rev"].append(embed(model, alt.translate(RC)[::-1], dev, args.bf16))
            for key, v in (("region_id", r.region_id), ("gene_id", r.gene_id), ("y", int(r.y)),
                           ("length", len(ref)), ("var_pos", pos)):
                meta[key].append(v)
        np.savez(os.path.join(out, f"shard_{k:04d}.npz"),
                 **{key: np.stack(v).astype(np.float32) for key, v in emb.items()},
                 **{key: np.array(v) for key, v in meta.items()})
        done += len(rows)
        rate = (time.time() - t0) / done
        left = sum(min((j + 1) * args.shard, len(pairs)) - j * args.shard for j in todo[todo.index(k) + 1:])
        print(f"shard {k + 1}/{n_shards}  {done:,} pairs in {(time.time() - t0) / 60:.1f} min  "
              f"({rate:.2f} s/pair)  ETA {left * rate / 60:.0f} min  "
              f"peak GPU {torch.cuda.max_memory_allocated() / 2**30:.1f} GB", flush=True)
    merge(out)


if __name__ == "__main__":
    main()
