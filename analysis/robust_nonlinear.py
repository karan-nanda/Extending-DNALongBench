#!/usr/bin/env python3
"""
Robustness check R4b: the nonlinear side of "no allele use" (frozen FM embeddings,
eQTLP-v2 matched pairs, fixed chromosome folds and gene folds).

  mlp_refalt / mlp_ref_copy   an MLP head on [e_ref, e_alt] (the released CaduceusEQTL head
                              is an MLP, not a linear layer), scored with the true alt and
                              with alt := ref. Delta = the allele's contribution to that head.
  mlp_diff                    MLP on e_alt - e_ref, with a label-permutation null
                              (labels shuffled within training folds, N_PERM times) to say
                              whether ~0.53 is above chance.

Usage:  python -X utf8 analysis/robust_nonlinear.py
"""
import os

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

N_FOLDS, N_PERM = 5, 20
EMB = {"HyenaDNA": "data/eQTL_v2/hyena_emb/matched_none/embeddings.npz",
       "Caduceus": "data/eQTL_v2/caduceus_emb/matched_none/embeddings.npz"}


def mlp(seed):
    return make_pipeline(StandardScaler(), MLPClassifier((256, 64), alpha=1e-3, max_iter=300,
                                                         early_stopping=True, random_state=seed))


def fold_mean(y, s, fold):
    return float(np.mean([roc_auc_score(y[fold == k], s[fold == k]) for k in range(N_FOLDS)]))


def main():
    pairs = pd.read_csv("data/eQTL_v2/pairs.tsv", sep="\t")
    mp = pairs[pairs["matched"] == 1].reset_index(drop=True)
    y = mp["y"].values
    rows = []
    for model, path in EMB.items():
        z = np.load(path, allow_pickle=True)
        assert (z["region_id"] == mp["region_id"].values).all()
        er = np.hstack([z["ref_fwd"], z["ref_rev"]])
        ea = np.hstack([z["alt_fwd"], z["alt_rev"]])
        d = ea - er
        for scheme in ("chrom", "gene"):
            fold = mp[f"fold_{scheme}"].values
            s_ra, s_copy, s_diff = (np.zeros(len(y)) for _ in range(3))
            for k in range(N_FOLDS):
                te, tr = fold == k, fold != k
                m = mlp(0).fit(np.hstack([er, ea])[tr], y[tr])
                s_ra[te] = m.predict_proba(np.hstack([er, ea])[te])[:, 1]
                s_copy[te] = m.predict_proba(np.hstack([er, er])[te])[:, 1]
                s_diff[te] = mlp(0).fit(d[tr], y[tr]).predict_proba(d[te])[:, 1]
            null = []
            rng = np.random.default_rng(0)
            for p in range(N_PERM):
                s_null = np.zeros(len(y))
                for k in range(N_FOLDS):
                    te, tr = fold == k, fold != k
                    y_perm = rng.permutation(y[tr])
                    s_null[te] = mlp(p).fit(d[tr], y_perm).predict_proba(d[te])[:, 1]
                null.append(fold_mean(y, s_null, fold))
            obs = fold_mean(y, s_diff, fold)
            row = {"model": model, "scheme": scheme,
                   "mlp_refalt": fold_mean(y, s_ra, fold), "mlp_ref_copy": fold_mean(y, s_copy, fold),
                   "mlp_diff": obs, "perm_null_mean": np.mean(null), "perm_null_max": np.max(null),
                   "perm_p": (1 + sum(n >= obs for n in null)) / (1 + N_PERM)}
            row["mlp_delta"] = row["mlp_refalt"] - row["mlp_ref_copy"]
            rows.append(row)
            print("  ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()),
                  flush=True)
    os.makedirs("analysis/results/robustness", exist_ok=True)
    pd.DataFrame(rows).to_csv("analysis/results/robustness/R4b_nonlinear.tsv", sep="\t", index=False)


if __name__ == "__main__":
    main()
