#!/usr/bin/env python3
"""
Does the published eQTLP CNN train at all?

SimpleCNN(task='eQTLP') in experiments/CNN/cnn.py ends in fc -> ReLU, so both
logits can be clamped to 0: softmax 0.5 for every input and zero gradient to
every weight. BatchNorm running statistics still update in train mode, so
eval-mode predictions drift even when nothing is learned.

Runs in fp32, as the published train.py does:
  1. across init seeds, on fixed real batches in train mode: share of samples
     whose two logits are both exactly 0, and the total gradient norm
  2. one seed trained with the published optimiser (AdamW lr 0.005 wd 0.01,
     clip 1.0, batch 2): dead share and loss per window, and how far the weights
     move relative to what weight decay alone would do

Usage:  python -X utf8 analysis/cnn_dead_head_check.py
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from eqtl_v2_cnn import EQTLDataset, Published  # noqa: E402


def loader(pairs, fasta, n, bs, seed, shuffle=False):
    ds = EQTLDataset(pairs.sample(n, random_state=seed), fasta, "published")
    return torch.utils.data.DataLoader(ds, batch_size=bs, shuffle=shuffle, drop_last=True,
                                       num_workers=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v2", default="data/eQTL_v2")
    ap.add_argument("--fasta", default="data/eQTL/seqs/hg38.fa")
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--probe-batches", type=int, default=8)
    ap.add_argument("--train-steps", type=int, default=150)
    ap.add_argument("--train-seeds", default="0", help="comma-separated init seeds to train")
    ap.add_argument("--published-tissue", default=None,
                    help="train on the PUBLISHED split's train rows of this tissue "
                         "(data/eQTL/targets/<tissue>.data.tsv) instead of eQTLP-v2; "
                         "--train-steps 0 means one full epoch")
    args = ap.parse_args()
    dev = torch.device("cuda")

    pairs = pd.read_csv(os.path.join(args.v2, "pairs.tsv"), sep="\t")
    pairs = pairs[(pairs.matched == 1) & (pairs.fold_chrom >= 2)]      # fold-0 split's train part
    if args.published_tissue:
        t = pd.read_csv(os.path.join("data/eQTL/targets", f"{args.published_tissue}.data.tsv"), sep="	")
        t = t[t["subset"] == "train"].copy()
        t["y"] = (t["target"] == "positive").astype(int)
        tss = np.where(t["gene_strand"] == "+", t["gene_start"], t["gene_end"] - 1)
        span = (np.maximum(t["region_end"] + 500, tss + 3001) - np.minimum(t["region_start"] - 500, tss - 3000))
        pairs = t[span <= 450_000].reset_index(drop=True)                  # parse_eQTL's cutoff
        print(f"published split, {args.published_tissue}: {len(pairs)} train rows "
              f"({int(pairs.y.sum())} positive)")
        if args.train_steps == 0:
            args.train_steps = len(pairs) // 2                             # one epoch at batch 2
    bs = 2
    probe = [(r.float(), y) for r, _, y in loader(pairs, args.fasta, args.probe_batches * bs, bs, 123)]

    print("1. INIT SEEDS (train mode, fp32, fixed batches)")
    print(f"   {'seed':>4} {'dead_train':>11} {'grad_norm':>11} {'dead_eval':>10} {'eval_logit_sd':>14}")
    for seed in range(args.seeds):
        torch.manual_seed(seed)
        m = Published().to(dev).train()
        dead, gn = [], []
        for ref, y in probe:
            m.zero_grad(set_to_none=True)
            out = m(ref.to(dev), None)
            nn.functional.cross_entropy(out, y.to(dev)).backward()
            dead.append((out == 0).all(1).float().mean().item())
            gn.append(torch.sqrt(sum((p.grad ** 2).sum() for p in m.parameters()
                                     if p.grad is not None)).item())
        m.eval()
        with torch.no_grad():
            oe = torch.cat([m(r.to(dev), None) for r, _ in probe])
        print(f"   {seed:>4} {np.mean(dead):>11.2f} {np.mean(gn):>11.2e} "
              f"{(oe == 0).all(1).float().mean().item():>10.2f} {oe.std().item():>14.3e}")

    for seed in [int(s) for s in args.train_seeds.split(",")]:
        print(f"\n2. TRAINING seed {seed}, published optimiser, {args.train_steps} steps")
        torch.manual_seed(seed)
        m = Published().to(dev).train()
        w0 = {n: p.detach().clone() for n, p in m.named_parameters() if p.dim() > 1}
        opt = torch.optim.AdamW(m.parameters(), lr=0.005, weight_decay=0.01)
        dl = loader(pairs, args.fasta, args.train_steps * bs, bs, 7, shuffle=True)
        win, dead, losses = max(25, args.train_steps // 12), [], []
        for step, (ref, _, y) in enumerate(dl, 1):
            out = m(ref.float().to(dev), None)
            loss = nn.functional.cross_entropy(out, y.to(dev))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step()
            dead.append((out == 0).all(1).float().mean().item())
            losses.append(loss.item())
            if step % win == 0:
                print(f"   steps {step - win + 1:>4}-{step:<4} dead {np.mean(dead[-win:]):.2f}  "
                      f"loss {np.mean(losses[-win:]):.4f}  min/max loss {min(losses[-win:]):.4f}/"
                      f"{max(losses[-win:]):.4f}")
        decay_only = 1 - (1 - 0.005 * 0.01) ** len(losses)
        print(f"   relative weight change ||W - W0|| / ||W0||  (weight decay alone: {decay_only:.4f})")
        for n, p in m.named_parameters():
            if n in w0:
                print(f"     {n:<32} {((p - w0[n]).norm() / w0[n].norm()).item():.4f}")
        m.eval()
        with torch.no_grad():
            oe = torch.cat([m(r.to(dev), None) for r, _ in probe])
        print(f"   after training, eval mode on probe batches: dead "
              f"{(oe == 0).all(1).float().mean().item():.2f}, logit sd {oe.std().item():.3e}")


if __name__ == "__main__":
    main()
