#!/usr/bin/env python3
"""
Robustness check R6: does the allele show up anywhere in the model, if we stop
averaging over the whole sequence?

For every eQTLP-v2 matched pair, embeds three sequences -- ref, alt, and alt_rand
(the variant replaced by a random *different* base, a control for "the diff vector
just encodes local context") -- and records, for every layer's residual stream plus
the final normed output, four poolings:
    mean   mean over all positions (the benchmark's readout)
    w1k    mean over the variant +-1,000 bp
    w64    mean over the variant +-64 bp
    pos    the hidden state at the variant position

Orientation: HyenaDNA is causal, so it runs in `rev` (variant first; positions after
the variant are the ones that can see it). Caduceus is bidirectional and runs in `fwd`.
bf16 autocast, unpadded, as the main runs.

  Windows (HyenaDNA):  python -X utf8 analysis/variant_window_embed.py --model hyena
  WSL (Caduceus):      /root/cad/bin/python analysis/variant_window_embed.py --model caduceus
Resumable by shard; merge happens at the end. Output: data/eQTL_v2/window_emb/<model>/.
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
from eqtl_v2_cnn import RC, SEQ_LEN, EQTLDataset  # noqa: E402

TOK = np.full(256, 11, np.int64)
for _i, _ch in enumerate("ACGTN"):
    TOK[ord(_ch)] = 7 + _i
POOLS = ("mean", "w1k", "w64", "pos")
SEQS = ("ref", "alt", "alt_rand")


def pooled(h, v):
    """h: (L, d) tensor on GPU (bf16 ok); v: variant index. -> (4, d) float32.

    Sums in float32 without materialising a float32 copy of the whole sequence: a
    450k x 256 fp32 copy is 460 MB per layer, which is what made this run out of memory.
    """
    L = h.shape[0]
    win = lambda lo, hi: h[max(0, lo):min(L, hi)].sum(0, dtype=torch.float32) / (min(L, hi) - max(0, lo))
    return torch.stack([h.sum(0, dtype=torch.float32) / L,
                        win(v - 1000, v + 1001),
                        win(v - 64, v + 65),
                        h[v].float()])


class Hyena:
    orient = "rev"

    def __init__(self, dev):
        import hyena_memcheck as hm  # patches fftconv to fp32
        from hyena_memcheck import load_hyena
        hm.FFT_CHUNK = 32          # smaller FFT slices: 64 peaks ~1 GB per call and fragments
        self.model, _ = load_hyena("data/models/hyenadna-medium-450k-seqlen", dev)
        self.model.eval()
        self.n_layers = len(self.model.backbone.layers) + 1

    @torch.inference_mode()
    def __call__(self, ids, v):
        bb, out = self.model.backbone, []
        with torch.autocast("cuda", dtype=torch.bfloat16):
            h, res = bb.embeddings(ids), None
            for layer in bb.layers:
                h, res = layer(h, res)
                out.append(pooled((h + res)[0], v))
            res = bb.drop_f(h) + res
            out.append(pooled(bb.ln_f(res.to(bb.ln_f.weight.dtype))[0], v))
        return torch.stack(out)


class Caduceus:
    orient = "fwd"

    def __init__(self, dev):
        from transformers import AutoModel
        self.model = AutoModel.from_pretrained(
            "data/models/caduceus-ph_seqlen-131k_d_model-256_n_layer-16", trust_remote_code=True).to(dev).eval()
        self.layers = self.model.backbone.layers
        self.n_layers = len(self.layers) + 1
        self._buf, self._v = [], 0
        for layer in self.layers:
            layer.register_forward_hook(self._hook)

    def _hook(self, module, inputs, output):
        h = output[0] + output[1] if output[1] is not None else output[0]
        self._buf.append(pooled(h[0], self._v))     # pool here: keeping 16 full layers OOMs

    @torch.inference_mode()
    def __call__(self, ids, v):
        self._buf, self._v = [], v
        with torch.autocast("cuda", dtype=torch.bfloat16):
            last = self.model(input_ids=ids).last_hidden_state
        out = torch.stack(self._buf + [pooled(last[0], v)])
        self._buf = []
        return out


def merge(out):
    shards = sorted(glob.glob(os.path.join(out, "shard_*.npz")))
    parts = [dict(np.load(s, allow_pickle=True)) for s in shards]
    merged = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    np.savez(os.path.join(out, "window_emb.npz"), **merged)
    print(f"merged {len(shards)} shards, {len(merged['y']):,} pairs -> {out}/window_emb.npz")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=("hyena", "caduceus"), required=True)
    ap.add_argument("--v2", default="data/eQTL_v2")
    ap.add_argument("--fasta", default="data/eQTL/seqs/hg38.fa")
    ap.add_argument("--shard", type=int, default=250)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-shards", type=int, default=0)
    ap.add_argument("--mem-fraction", type=float, default=0.6)
    args = ap.parse_args()

    out = os.path.join(args.v2, "window_emb", args.model + (f"_lim{args.limit}" if args.limit else ""))
    os.makedirs(out, exist_ok=True)
    pairs = pd.read_csv(os.path.join(args.v2, "pairs.tsv"), sep="\t")
    pairs = pairs[pairs["matched"] == 1].reset_index(drop=True)
    if args.limit:
        pairs = pairs.head(args.limit)

    dev = torch.device("cuda")
    torch.cuda.set_per_process_memory_fraction(args.mem_fraction, 0)
    model = (Hyena if args.model == "hyena" else Caduceus)(dev)
    ds = EQTLDataset(pairs, args.fasta, "refalt")

    n_shards = math.ceil(len(pairs) / args.shard)
    todo = [k for k in range(n_shards) if not os.path.exists(os.path.join(out, f"shard_{k:04d}.npz"))]
    print(f"{args.model} ({model.orient}, {model.n_layers} layers): {len(pairs):,} pairs, "
          f"{len(todo)} of {n_shards} shards to do -> {out}", flush=True)
    t0, done = time.time(), 0
    for k in todo[:args.max_shards or None]:
        rows = range(k * args.shard, min((k + 1) * args.shard, len(pairs)))
        emb = {s: [] for s in SEQS}
        meta = {"region_id": [], "y": [], "var_idx": [], "length": [], "alt_rand_base": []}
        for i in rows:
            r = pairs.iloc[i]
            ref, alt, pos = ds.build_strings(r, r.allele2)
            ref_base = ref[pos]
            rng = np.random.default_rng(i)
            other = [b for b in "ACGT" if b not in (ref_base.upper(), alt[pos].upper())]
            rb = other[rng.integers(len(other))]
            alt_rand = ref[:pos] + rb + ref[pos + 1:]
            L = min(len(ref), SEQ_LEN)
            seqs = {"ref": ref[:SEQ_LEN], "alt": alt[:SEQ_LEN], "alt_rand": alt_rand[:SEQ_LEN]}
            v = pos
            if model.orient == "rev":
                seqs = {s: x.translate(RC)[::-1] for s, x in seqs.items()}
                v = L - 1 - pos
            for s, x in seqs.items():
                torch.cuda.empty_cache()   # lengths vary per pair; keep the cache from fragmenting
                ids = torch.from_numpy(TOK[np.frombuffer(x.encode(), np.uint8)]).unsqueeze(0).to(dev)
                emb[s].append(model(ids, v).cpu().numpy().astype(np.float16))
            for key, val in (("region_id", r.region_id), ("y", int(r.y)), ("var_idx", v),
                             ("length", L), ("alt_rand_base", rb)):
                meta[key].append(val)
        np.savez(os.path.join(out, f"shard_{k:04d}.npz"),
                 **{s: np.stack(v) for s, v in emb.items()}, **{m: np.array(v) for m, v in meta.items()})
        done += len(rows)
        rate = (time.time() - t0) / done
        print(f"shard {k + 1}/{n_shards}  {done:,} pairs  {(time.time() - t0) / 60:.1f} min  "
              f"({rate:.2f} s/pair)  peak GPU {torch.cuda.max_memory_allocated() / 2**30:.1f} GB", flush=True)
    remaining = [k for k in range(n_shards) if not os.path.exists(os.path.join(out, f"shard_{k:04d}.npz"))]
    if remaining:
        print(f"{len(remaining)} shards remain", flush=True)
        sys.exit(3)
    merge(out)


if __name__ == "__main__":
    main()
