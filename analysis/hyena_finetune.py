#!/usr/bin/env python3
"""
End-to-end fine-tuning of HyenaDNA-medium-450k on eQTLP-v2.

This answers the one reviewer question the frozen probes cannot: would training
the backbone make the model use the allele? The interface is the benchmark's own
-- embed ref and alt, mean-pool each, concatenate, classify -- so a positive
result here would be a result for the published protocol, not a new task.

At the end of every epoch the same fitted model scores the test fold twice:

  refalt     [e_ref, e_alt]     the real pair
  ref_copy   [e_ref, e_ref]     the alternate allele replaced by the reference

Delta = AUROC(refalt) - AUROC(ref_copy) is the number the paper reports. If
fine-tuning teaches the model to read the allele, Delta must move away from 0.

Memory: the backbone is gradient-checkpointed and run in bf16 autocast, one pair
per step (two forwards, ref and alt) with gradient accumulation. 450 kb does not
fit in 12 GB; --maxlen 131072 does. Sequences longer than --maxlen keep their
LAST maxlen characters, which in the published orientation retains the variant
(it sits ~500 bp from the end) plus the nearest context toward the TSS.

Usage
-----
  python -X utf8 analysis/hyena_finetune.py --fold 0 --limit 8 --epochs 1   # smoke test
  python -X utf8 analysis/hyena_finetune.py --fold 0 --epochs 2             # one fold
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from hyena_memcheck import backbone, load_hyena  # noqa: E402  (also patches fftconv to fp32)
from eqtl_v2_cnn import EQTLDataset  # noqa: E402

MODEL_DIR = "data/models/hyenadna-medium-450k-seqlen"
OUT_DIR = "analysis/results/finetune"

TOK = np.full(256, 11, np.int64)          # HyenaDNA char tokenizer: A C G T N -> 7..11
for _i, _ch in enumerate("ACGTN"):
    TOK[ord(_ch)] = 7 + _i


class PairSeqs(torch.utils.data.Dataset):
    """Oriented ref/alt token ids for one pair, truncated to maxlen from the left."""

    def __init__(self, pairs, fasta_path, maxlen):
        self.inner = EQTLDataset(pairs, fasta_path, mode="refalt")
        self.p = self.inner.p
        self.maxlen = maxlen

    def __len__(self):
        return len(self.p)

    def __getitem__(self, i):
        r = self.p.iloc[i]
        ref, alt, _pos = self.inner.build_strings(r, r.allele2)
        ref, alt = ref[-self.maxlen:], alt[-self.maxlen:]
        enc = lambda s: torch.from_numpy(TOK[np.frombuffer(s.encode(), np.uint8)])
        return enc(ref), enc(alt), int(r.y)


class Classifier(nn.Module):
    """The benchmark's interface: [mean(e_ref), mean(e_alt)] -> 2 logits."""

    def __init__(self, hyena, d_model):
        super().__init__()
        self.hyena = hyena
        self.head = nn.Linear(2 * d_model, 2)

    def embed(self, ids, use_ckpt):
        return backbone(self.hyena.backbone, ids, use_ckpt).float().mean(1)

    def forward(self, ref_ids, alt_ids, use_ckpt=True, ref_copy=False):
        e_ref = self.embed(ref_ids, use_ckpt)
        e_alt = e_ref if ref_copy else self.embed(alt_ids, use_ckpt)
        return self.head(torch.cat([e_ref, e_alt], dim=-1))


def evaluate(model, ds, idx, device, maxlen):
    """Test-fold AUROC for refalt and for ref_copy, from one pass of the model."""
    model.eval()
    scores = {"refalt": [], "ref_copy": []}
    y = []
    with torch.inference_mode():
        for i in idx:
            ref, alt, label = ds[i]
            ref = ref.unsqueeze(0).to(device)
            alt = alt.unsqueeze(0).to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                e_ref = model.embed(ref, False)
                e_alt = model.embed(alt, False)
            for key, vec in (("refalt", e_alt), ("ref_copy", e_ref)):
                logit = model.head(torch.cat([e_ref, vec], dim=-1)).float()
                scores[key].append(torch.softmax(logit, -1)[0, 1].item())
            y.append(label)
    y = np.array(y)
    out = {k: float(roc_auc_score(y, np.array(v))) for k, v in scores.items()}
    out["delta"] = out["refalt"] - out["ref_copy"]
    out["n"], out["n_pos"] = int(len(y)), int(y.sum())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--maxlen", type=int, default=131072)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--head-lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0, help="smoke test: cap train/test size")
    ap.add_argument("--fasta", default="data/eQTL/seqs/hg38.fa")
    ap.add_argument("--out", default=OUT_DIR)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda"

    pairs = pd.read_csv("data/eQTL_v2/pairs.tsv", sep="\t")
    mp = pairs[pairs["matched"] == 1].reset_index(drop=True)
    ds = PairSeqs(mp, args.fasta, args.maxlen)

    fold = mp["fold_chrom"].values
    tr_idx = np.flatnonzero(fold != args.fold)
    te_idx = np.flatnonzero(fold == args.fold)
    if args.limit:
        tr_idx, te_idx = tr_idx[:args.limit], te_idx[:args.limit]
    rng = np.random.default_rng(args.seed)

    hyena, cfg = load_hyena(MODEL_DIR, device)
    model = Classifier(hyena, cfg["d_model"]).to(device)
    opt = torch.optim.AdamW(
        [{"params": model.hyena.parameters(), "lr": args.lr},
         {"params": model.head.parameters(), "lr": args.head_lr}], weight_decay=args.wd)
    lossf = nn.CrossEntropyLoss()

    # a smoke test must not write where the real run's result goes
    tag = f"fold{args.fold}_len{args.maxlen}" + (f"_limit{args.limit}" if args.limit else "")
    hist, t_start = [], time.time()
    print(f"[{tag}] train {len(tr_idx)}  test {len(te_idx)}  maxlen {args.maxlen}", flush=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        order = rng.permutation(tr_idx)
        opt.zero_grad(set_to_none=True)
        run_loss, t0 = 0.0, time.time()
        for step, i in enumerate(order, 1):
            ref, alt, label = ds[i]
            ref = ref.unsqueeze(0).to(device)
            alt = alt.unsqueeze(0).to(device)
            yt = torch.tensor([label], device=device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(ref, alt, use_ckpt=True)
            loss = lossf(logits.float(), yt) / args.accum
            loss.backward()
            run_loss += loss.item() * args.accum
            if step % args.accum == 0 or step == len(order):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                opt.zero_grad(set_to_none=True)
            every = max(1, min(200, len(order) // 5))
            if step % every == 0:
                peak = torch.cuda.max_memory_allocated() / 2**30
                print(f"  e{epoch} {step}/{len(order)}  loss {run_loss / step:.4f}  "
                      f"{(time.time() - t0) / step:.2f}s/pair  peak {peak:.2f} GB", flush=True)

        res = evaluate(model, ds, te_idx, device, args.maxlen)
        res.update(epoch=epoch, train_loss=run_loss / len(order),
                   epoch_sec=time.time() - t0)
        hist.append(res)
        print(f"[{tag}] epoch {epoch}: refalt {res['refalt']:.4f}  "
              f"ref_copy {res['ref_copy']:.4f}  delta {res['delta']:+.5f}  "
              f"({res['epoch_sec'] / 3600:.2f} h)", flush=True)
        with open(os.path.join(args.out, f"{tag}.json"), "w") as fh:
            json.dump({"args": vars(args), "history": hist,
                       "total_sec": time.time() - t_start}, fh, indent=1)

    print(f"[{tag}] done in {(time.time() - t_start) / 3600:.2f} h", flush=True)


if __name__ == "__main__":
    main()
