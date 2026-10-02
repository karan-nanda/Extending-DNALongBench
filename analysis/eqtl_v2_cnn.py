#!/usr/bin/env python3
"""
CNN on eQTLP-v2 splits, with the published harness's input construction.

Modes
-----
  published      the repo's SimpleCNN on x_ref only (what Table 7's CNN column is)
  refalt         siamese SimpleCNN encoder on ref and alt, global max-pooled,
                 LayerNorm, concatenated -> linear head (the FM protocol in the
                 paper's Methods)
  ref_only       same model, alt embedding replaced by a copy of the ref embedding
  random_allele  same model, alt allele replaced by a random base != ref
                 (fixed per pair)

Input is the variant->TSS span exactly as dnalongbench.utils.parse_eQTL builds
it (+-3 kb TSS flank, +-500 bp variant flank, reverse-complemented when the gene
lies downstream, right-padded with N to 450 kb, N one-hot = 0.25 as in
kipoiseq), except that positive eQTLs are NOT masked -- that mask is label
information.

Training defaults follow experiments/CNN/train.py: AdamW lr 0.005 wd 0.01,
batch 2, 5 epochs, grad-norm clip 1.0, cross-entropy. Added: fp16 autocast
(memory), best-epoch selection on validation AUROC.

Usage
-----
  python -X utf8 analysis/eqtl_v2_cnn.py --version matched --scheme chrom --fold 0 --mode refalt
  python -X utf8 analysis/eqtl_v2_cnn.py --mode refalt --limit 64 --epochs 1      # smoke test
"""
import argparse
import copy
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "DNALONGBENCH", "experiments", "CNN"))
sys.path.insert(0, HERE)
from cnn import SimpleCNN  # noqa: E402  (the published model definition)
from eqtl_v2_baselines import Fasta  # noqa: E402

SEQ_LEN = 450_000
TSS_FLANK, REGION_FLANK = 3000, 500
N_FOLDS = 5
MODES = ("published", "refalt", "ref_only", "random_allele")

LUT = np.zeros((256, 4), np.float16)
for _i, _b in enumerate("ACGT"):
    LUT[ord(_b), _i] = LUT[ord(_b.lower()), _i] = 1
LUT[ord("N")] = LUT[ord("n")] = 0.25
RC = str.maketrans("ACGTN", "TGCAN")


class EQTLDataset(torch.utils.data.Dataset):
    def __init__(self, pairs, fasta_path, mode, seed=0):
        self.p = pairs.reset_index(drop=True)
        self.fasta_path, self.fasta, self.mode = fasta_path, None, mode
        rng = np.random.default_rng(seed)
        self.rand_alt = [rng.choice([b for b in "ACGT" if b != a.upper()]) for a in self.p["allele1"]]

    def __len__(self):
        return len(self.p)

    def build_strings(self, r, alt_allele):
        """Oriented ref / alt strings as parse_eQTL builds them, before N padding,
        plus the variant's index in that reading order."""
        if self.fasta is None:          # opened lazily so DataLoader workers each get one
            self.fasta = Fasta(self.fasta_path)
        tss = r.gene_start if r.gene_strand == "+" else r.gene_end - 1
        t0, t1 = tss - TSS_FLANK, tss + 1 + TSS_FLANK
        r0, r1 = r.region_start - REGION_FLANK, r.region_end + REGION_FLANK
        s0, s1 = min(t0, r0), max(t1, r1)
        ref = self.fasta.fetch(r.region_chrom, s0, s1).upper()
        vs, ve = r.region_start - s0, r.region_end - s0
        assert ref[vs:ve] == r.allele1, (r.region_id, ref[vs:ve])
        alt = None if alt_allele is None else ref[:vs] + alt_allele + ref[ve:]
        pos = vs
        if r.gene_start > r1:           # same flip rule as parse_eQTL
            ref = ref.translate(RC)[::-1]
            alt = None if alt is None else alt.translate(RC)[::-1]
            pos = len(ref) - ve
        return ref, alt, pos

    def build(self, r, alt_allele):
        ref, alt, _ = self.build_strings(r, alt_allele)
        enc = lambda s: LUT[np.frombuffer(s[:SEQ_LEN].ljust(SEQ_LEN, "N").encode(), np.uint8)]
        return enc(ref), (None if alt is None else enc(alt))

    def __getitem__(self, i):
        r = self.p.iloc[i]
        if self.mode in ("published", "ref_only"):
            alt_allele = None
        elif self.mode == "random_allele":
            alt_allele = self.rand_alt[i]
        else:
            alt_allele = r.allele2
        ref, alt = self.build(r, alt_allele)
        alt = torch.empty(0) if alt is None else torch.from_numpy(alt)
        return torch.from_numpy(ref), alt, int(r.y)


class Siamese(nn.Module):
    def __init__(self, mode):
        super().__init__()
        self.mode = mode
        self.encoder = SimpleCNN(channels=[128, 64, 32], input_length=SEQ_LEN, task="eQTLP").layers
        # max over 450 kb of ReLU outputs is unbounded; without this the head's
        # logits blow up (train loss 2-7 at lr 1e-4..5e-3 in smoke tests)
        self.norm = nn.LayerNorm(32)
        self.head = nn.Linear(64, 2)

    def embed(self, x):
        return self.norm(self.encoder(x.transpose(1, 2)).amax(dim=2))

    def forward(self, ref, alt):
        e_ref = self.embed(ref)
        e_alt = e_ref if alt is None else self.embed(alt)
        return self.head(torch.cat([e_ref, e_alt], -1))


class Published(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = SimpleCNN(channels=[128, 64, 32], input_length=SEQ_LEN, task="eQTLP")

    def forward(self, ref, alt):
        out = self.net(ref)             # SimpleCNN squeezes away a batch of 1
        return out.unsqueeze(0) if out.dim() == 1 else out


class TriDataset(EQTLDataset):
    """ref, true alt and random-allele alt for the same pair."""

    def __getitem__(self, i):
        r = self.p.iloc[i]
        ref, alt = self.build(r, r.allele2)
        _, rnd = self.build(r, self.rand_alt[i])
        return torch.from_numpy(ref), torch.from_numpy(alt), torch.from_numpy(rnd), int(r.y)


@torch.no_grad()
def test_time_ablation(model, pairs, args, device):
    """The plan's two-line edit on the SAME trained weights: score each test pair
    with its true alt, with the alt embedding replaced by a copy of the ref
    embedding, and with a random alt allele."""
    torch.backends.cudnn.benchmark = False       # identical kernels for ref and alt passes
    torch.backends.cudnn.deterministic = True
    dl = torch.utils.data.DataLoader(TriDataset(pairs, args.fasta, "refalt", args.seed),
                                     batch_size=args.bs, num_workers=args.workers)
    model.eval()
    ys, scores, max_diff = [], {"alt": [], "ref_copy": [], "random_allele": []}, []
    for ref, alt, rnd, y in dl:
        with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            e_ref = model.embed(ref.to(device).float())
            e_alt = model.embed(alt.to(device).float())
            e_rnd = model.embed(rnd.to(device).float())
            for k, e in (("alt", e_alt), ("ref_copy", e_ref), ("random_allele", e_rnd)):
                logits = model.head(torch.cat([e_ref, e], -1)).float()
                scores[k].append(torch.softmax(logits, -1)[:, 1].cpu().numpy())
        max_diff.append((e_alt.float() - e_ref.float()).abs().amax(1).cpu().numpy())
        ys.append(y.numpy())
    y = np.concatenate(ys)
    scores = {k: np.concatenate(v) for k, v in scores.items()}
    max_diff = np.concatenate(max_diff)
    res = {k: float(roc_auc_score(y, v)) for k, v in scores.items()}
    res["frac_embedding_changed"] = float((max_diff > 1e-3).mean())
    res["median_max_abs_diff"] = float(np.median(max_diff))
    return res, y, scores


def run_epoch(model, loader, device, opt=None, scaler=None, max_steps=None):
    train = opt is not None
    model.train(train)
    ce = nn.CrossEntropyLoss()
    ys, ss, total, n = [], [], 0.0, 0
    with torch.set_grad_enabled(train):
        for step, (ref, alt, y) in enumerate(loader):
            if max_steps and step >= max_steps:
                break
            ref = ref.to(device, non_blocking=True).float()
            alt = alt.to(device, non_blocking=True).float() if alt.numel() else None
            y = y.to(device)
            with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                logits = model(ref, alt).float()
            loss = ce(logits, y)
            if train:
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
            total += loss.item()
            n += 1
            ys.append(y.cpu().numpy())
            ss.append(torch.softmax(logits.detach(), -1)[:, 1].cpu().numpy())
    y, s = np.concatenate(ys), np.concatenate(ss)
    two = len(np.unique(y)) == 2
    return {"loss": total / max(n, 1),
            "auroc": roc_auc_score(y, s) if two else float("nan"),
            "auprc": average_precision_score(y, s) if two else float("nan")}, y, s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v2", default="data/eQTL_v2")
    ap.add_argument("--fasta", default="data/eQTL/seqs/hg38.fa")
    ap.add_argument("--version", choices=("matched", "all"), default="matched")
    ap.add_argument("--scheme", choices=("random", "gene", "chrom"), default="chrom")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--mode", choices=MODES, default="refalt")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--bs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=0.005)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0, help="subsample each split (smoke test)")
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--out", default="analysis/results/cnn")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pairs = pd.read_csv(os.path.join(args.v2, "pairs.tsv"), sep="\t")
    if args.version == "matched":
        pairs = pairs[pairs["matched"] == 1]
    f = pairs[f"fold_{args.scheme}"]
    val_fold = (args.fold + 1) % N_FOLDS
    parts = {"train": pairs[(f != args.fold) & (f != val_fold)],
             "valid": pairs[f == val_fold], "test": pairs[f == args.fold]}
    if args.limit:
        parts = {k: v.sample(min(len(v), args.limit), random_state=args.seed) for k, v in parts.items()}
    print("sizes: " + ", ".join(f"{k}={len(v)} (pos {int(v.y.sum())})" for k, v in parts.items()))

    def loader(split, shuffle):
        ds = EQTLDataset(parts[split], args.fasta, args.mode, args.seed)
        return torch.utils.data.DataLoader(ds, batch_size=args.bs, shuffle=shuffle, drop_last=shuffle,
                                           num_workers=args.workers, pin_memory=device.type == "cuda",
                                           persistent_workers=args.workers > 0)

    train_dl, valid_dl, test_dl = loader("train", True), loader("valid", False), loader("test", False)
    model = (Published() if args.mode == "published" else Siamese(args.mode)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")

    name = (f"{args.version}_{args.scheme}_f{args.fold}_{args.mode}_s{args.seed}"
            + (f"_lim{args.limit}" if args.limit else ""))
    os.makedirs(args.out, exist_ok=True)
    best, best_state, history = -1.0, None, []
    for ep in range(1, args.epochs + 1):
        t = time.time()
        tr, _, _ = run_epoch(model, train_dl, device, opt, scaler, args.max_steps or None)
        va, _, _ = run_epoch(model, valid_dl, device, max_steps=args.max_steps or None)
        dt = time.time() - t
        mem = torch.cuda.max_memory_allocated() / 2**30 if device.type == "cuda" else 0
        history.append({"epoch": ep, "train": tr, "valid": va, "sec": dt})
        print(f"[{name}] epoch {ep}  train loss {tr['loss']:.4f} auroc {tr['auroc']:.3f}  |  "
              f"valid auroc {va['auroc']:.3f}  |  {dt / 60:.1f} min, peak {mem:.1f} GB", flush=True)
        score = va["auroc"] if not np.isnan(va["auroc"]) else -va["loss"]
        if score > best:
            best, best_state = score, copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    torch.save(best_state, os.path.join(args.out, f"{name}.pt"))
    te, y, s = run_epoch(model, test_dl, device, max_steps=args.max_steps or None)
    print(f"[{name}] TEST auroc {te['auroc']:.3f}  auprc {te['auprc']:.3f}  (best valid {best:.3f})")
    tp = parts["test"].iloc[:len(y)]
    preds = pd.DataFrame({"region_id": tp["region_id"].values, "gene_id": tp["gene_id"].values,
                          "y": y, "score": s})
    ablation = None
    if args.mode == "refalt":
        ablation, ya, sa = test_time_ablation(model, tp, args, device)
        assert (ya == y).all(), "ablation pairs out of order"
        for k, v in sa.items():
            preds[f"score_{k}"] = v
        print(f"[{name}] TEST-TIME ABLATION (same weights): "
              + "  ".join(f"{k} {v:.4f}" for k, v in ablation.items()))
    preds.to_csv(os.path.join(args.out, f"{name}.preds.tsv"), sep="\t", index=False)
    with open(os.path.join(args.out, f"{name}.json"), "w") as fh:
        json.dump({"args": vars(args), "history": history, "test": te, "best_valid": best,
                   "ablation": ablation}, fh, indent=1)


if __name__ == "__main__":
    main()
