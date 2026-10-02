#!/usr/bin/env python3
"""
Can HyenaDNA-medium-450k run on this GPU at the eQTLP input length?

Loads the pretrained weights into the repo's standalone (flash-attention-free)
model and measures time and peak memory at --len tokens for:
  1. inference, one sequence
  2. one training step, one sequence, no activation checkpointing
  3. one training step, one sequence, per-block activation checkpointing
  4. one training step, ref + alt in one graph (the published eQTL head), checkpointing

The long FFT convolution is done in fp32: cuFFT has no bf16 path, and under
autocast the repo's fftconv would hand it bf16 tensors.

Usage:  python -X utf8 analysis/hyena_memcheck.py [--len 450000]
"""
import argparse
import json
import os
import re
import sys
import time

import torch
import torch.nn as nn
import torch.utils.checkpoint as ckpt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "DNALONGBENCH", "experiments", "HyenaDNA",
                                "HyenaDNA_ETGP_CMP_eQTLP"))
import standalone_hyenadna as sh  # noqa: E402


FFT_CHUNK = 64      # channels per FFT; the long conv is depthwise, so chunks are independent


def fftconv_fp32(u, k, D):
    """The repo's fftconv, in fp32 and chunked over channels. At 450k tokens one
    256-channel complex spectrum is ~0.9 GB; chunking keeps only a quarter of each
    alive at once."""
    seqlen = u.shape[-1]
    n = 2 * seqlen
    if u.dim() > 3:     # multi-head layout; not used by the eQTL models, keep the original path
        k_f = (torch.fft.rfft(k.float(), n=n) / n).unsqueeze(1)
        y = torch.fft.irfft(torch.fft.rfft(u.float(), n=n) * k_f, n=n, norm="forward")[..., :seqlen]
        return (y + u.float() * D.float().unsqueeze(-1)).to(u.dtype)
    ys = []
    for c0 in range(0, u.shape[-2], FFT_CHUNK):
        c1 = c0 + FFT_CHUNK
        uc = u[..., c0:c1, :].float()
        k_f = torch.fft.rfft(k[..., c0:c1, :].float(), n=n) / n
        y = torch.fft.irfft(torch.fft.rfft(uc, n=n) * k_f, n=n, norm="forward")[..., :seqlen]
        ys.append((y + uc * D[c0:c1].float().unsqueeze(-1)).to(u.dtype))
    return torch.cat(ys, dim=-2)


sh.fftconv = fftconv_fp32      # HyenaFilter.forward looks fftconv up at call time


def load_hyena(model_dir, device):
    with open(os.path.join(model_dir, "config.json")) as fh:
        cfg = json.load(fh)
    model = sh.HyenaDNAModel(**cfg, use_head=False)
    ckpt_sd = torch.load(os.path.join(model_dir, "weights.ckpt"), map_location="cpu",
                         weights_only=False)["state_dict"]
    sd = model.state_dict()
    for k in sd:
        src = "model." + k
        if cfg.get("checkpoint_mixer"):          # same key surgery as huggingface.load_weights
            src = re.sub(r"\.mixer", ".mixer.layer", src)
            src = re.sub(r"\.mlp", ".mlp.layer", src)
        sd[k] = ckpt_sd[src]
    model.load_state_dict(sd)
    return model.to(device), cfg


def backbone(bb, ids, use_ckpt):
    h, res = bb.embeddings(ids), None
    for layer in bb.layers:
        if use_ckpt:
            h, res = ckpt.checkpoint(layer, h, res, use_reentrant=False)
        else:
            h, res = layer(h, res)
    res = bb.drop_f(h) + res
    return bb.ln_f(res.to(bb.ln_f.weight.dtype))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="data/models/hyenadna-medium-450k-seqlen")
    ap.add_argument("--len", type=int, default=450_000)
    ap.add_argument("--mem-fraction", type=float, default=0.92,
                    help="hard cap on the caching allocator, as a fraction of VRAM")
    args = ap.parse_args()
    dev = torch.device("cuda")
    # On Windows the NVIDIA driver can spill into system RAM instead of raising OOM
    # ("sysmem fallback"), which turns an over-budget run into hours of thrashing.
    # Capping the allocator makes it fail fast with OutOfMemoryError instead.
    torch.cuda.set_per_process_memory_fraction(args.mem_fraction, 0)

    model, cfg = load_hyena(args.model, dev)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"loaded {args.model}: {n_params / 1e6:.2f}M params, l_max {cfg['layer']['l_max']}")
    bb = model.backbone
    head = nn.Linear(2 * cfg["d_model"], 2).to(dev)
    opt = torch.optim.AdamW(list(model.parameters()) + list(head.parameters()), lr=6e-4)
    ids = torch.randint(7, 11, (1, args.len), device=dev)          # A/C/G/T token ids
    ids_alt = ids.clone()
    ids_alt[0, args.len // 2] = 7 + (ids[0, args.len // 2] - 6) % 4
    y = torch.tensor([1], device=dev)
    amp = dict(device_type="cuda", dtype=torch.bfloat16)

    def infer():
        model.eval()
        with torch.no_grad(), torch.autocast(**amp):
            backbone(bb, ids, False).mean(1)

    def train_one(use_ckpt):
        def f():
            model.train()
            with torch.autocast(**amp):
                e = backbone(bb, ids, use_ckpt).mean(1)
                loss = nn.functional.cross_entropy(head(torch.cat([e, e], -1)).float(), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        return f

    def train_pair():
        model.train()
        with torch.autocast(**amp):
            e_ref = backbone(bb, ids, True).mean(1)
            e_alt = backbone(bb, ids_alt, True).mean(1)
            loss = nn.functional.cross_entropy(head(torch.cat([e_ref, e_alt], -1)).float(), y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    tests = [("inference, 1 sequence", infer),
             ("train step, 1 seq, no checkpointing", train_one(False)),
             ("train step, 1 seq, block checkpointing", train_one(True)),
             ("train step, ref+alt one graph, checkpointing", train_pair)]
    print(f"\nseq len {args.len:,}, bf16 autocast, fp32 FFT, allocator cap "
          f"{args.mem_fraction:.0%} of VRAM; each test run twice, second timed", flush=True)
    for label, fn in tests:
        print(f"  {label:<46} running...", flush=True)
        try:
            fn()
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            t = time.time()
            fn()
            torch.cuda.synchronize()
            print(f"  {label:<46} {time.time() - t:7.2f} s   peak "
                  f"{torch.cuda.max_memory_allocated() / 2**30:5.2f} GB", flush=True)
        except torch.cuda.OutOfMemoryError:
            print(f"  {label:<46}     OOM (over the {args.mem_fraction:.0%} cap)", flush=True)
            opt.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
