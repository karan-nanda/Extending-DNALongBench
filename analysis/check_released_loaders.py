#!/usr/bin/env python3
"""
Robustness check R5: run the RELEASED eQTL dataloaders (HyenaDNA and Caduceus, as
shipped in DNALONGBENCH/experiments) on real records and report what they yield.

For each repo, imports its own eqtl_dataset.EQTLseqDataSet and calls its own __iter__
on records built exactly like parse_eQTL (variant..TSS span, N-padded to 450 kb;
built here with eqtl_v2_cnn.build_strings because parse_eQTL needs the tabix mask).
Also runs the repo's parse_eQTL itself when its dependencies import.

Runs inside WSL (kipoiseq, pytabix, pyfaidx in /root/cad):
  wsl -d Ubuntu-24.04 -u root --cd "/mnt/d/Extending DNALongBench" -- /root/cad/bin/python analysis/check_released_loaders.py
"""
import importlib.util
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from eqtl_v2_cnn import SEQ_LEN, EQTLDataset  # noqa: E402

REPOS = {
    "HyenaDNA": "DNALONGBENCH/experiments/HyenaDNA/HyenaDNA_ETGP_CMP_eQTLP/src/dataloaders/datasets/eqtl_dataset.py",
    "Caduceus": "DNALONGBENCH/experiments/Caduceus/Caduceus_CMP_eQTLP_ETGP/src/dataloaders/datasets/eqtl_dataset.py",
}
VOCAB_ACGT = (7, 8, 9, 10)   # A C G T in the pretrained HyenaDNA / Caduceus char tokenizers


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    t = pd.read_csv("data/eQTL/targets/Whole_Blood.data.tsv", sep="\t")
    rows = t[t["subset"] == "test"].head(3)
    ds = EQTLDataset(rows.reset_index(drop=True), "data/eQTL/seqs/hg38.fa", "refalt")
    records = []
    for r in rows.itertuples():
        ref, alt, _ = ds.build_strings(r, r.allele2)
        records.append((ref[:SEQ_LEN].ljust(SEQ_LEN, "N"), alt[:SEQ_LEN].ljust(SEQ_LEN, "N"), r.target))

    for repo, path in REPOS.items():
        mod = load_module(f"eqtl_{repo}", path)
        obj = mod.EQTLseqDataSet.__new__(mod.EQTLseqDataSet)   # skip __init__ (needs config + tabix)
        obj.dataset = records
        print(f"\n{repo}: {path}")
        for i, (s_ref, s_alt, target) in enumerate(obj.__iter__()):
            print(f"  record {i}: input length {len(records[i][0]):,} -> yielded ref shape {s_ref.shape}, "
                  f"alt shape {s_alt.shape}, target {target}")
        # the same code path without the slice: what token ids would the model get?
        oh = mod.one_hot_encode(records[0][0])
        ids = np.argmax(oh, axis=-1)
        print(f"  without the slice: one-hot {oh.shape}, argmax ids used {sorted(set(ids.tolist()))} "
              f"(pretrained vocab A/C/G/T = {VOCAB_ACGT}; N padding -> id {int(ids[-1])})")


if __name__ == "__main__":
    main()
