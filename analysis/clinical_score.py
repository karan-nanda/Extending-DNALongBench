#!/usr/bin/env python3
"""
Score the clinical evaluation set (build_clinical_eval.py) with frozen models.
Eval-only: nothing here is fitted on clinical labels.

Scores per model (HyenaDNA, Caduceus) and reading order:
  shift      ||e_alt - e_ref|| / ||e_ref||: does the SNP move the pooled embedding more
             for pathogenic than for matched benign variants? (zero-shot)
  eqtl_probe the linear probe from hyena_probe.py, fitted on ALL eQTLP-v2 matched pairs,
             applied to clinical [e_ref, e_alt]  (transfer)
  eqtl_allele the same probe's score for the true alt minus its score with alt := ref;
             the part of the transfer score that comes from the allele
Baseline: -distance to TSS (matched away by construction, so ~0.5 is the sanity check).

AUROC over all variants, with a 95% CI from resampling matched pairs (1,000 draws), and
the paired win rate (pathogenic scores above its own matched benign). Strata: all,
2+ star positives, same-gene same-location matches, and location class.

Usage:  python -X utf8 analysis/clinical_score.py
"""
import argparse
import os

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from hyena_probe import fit

MODELS = {"HyenaDNA": "hyena_emb", "Caduceus": "caduceus_emb"}


def load(path, pairs=None):
    z = np.load(path, allow_pickle=True)
    if pairs is not None:
        assert (z["region_id"] == pairs["region_id"].values).all(), path
    return z


def scores(z_eqtl, z_clin, C):
    out = {}
    for o in ("fwd", "rev"):
        r, a = z_clin[f"ref_{o}"], z_clin[f"alt_{o}"]
        out[f"shift_{o}"] = np.linalg.norm(a - r, axis=1) / np.linalg.norm(r, axis=1)
    X = lambda z, alt: np.hstack([z["ref_fwd"], z[f"{alt}_fwd"], z["ref_rev"], z[f"{alt}_rev"]])
    m = fit(X(z_eqtl, "alt"), z_eqtl["y"], C)
    s_alt = m.predict_proba(X(z_clin, "alt"))[:, 1]
    s_ref = m.predict_proba(X(z_clin, "ref"))[:, 1]
    out["eqtl_probe"], out["eqtl_allele"] = s_alt, s_alt - s_ref
    return out


def auroc_ci(y, s, pair, n_boot=1000, seed=0):
    rng = np.random.default_rng(seed)
    ids = np.unique(pair)
    rows = {p: np.flatnonzero(pair == p) for p in ids}
    boot = []
    for _ in range(n_boot):
        idx = np.concatenate([rows[p] for p in rng.choice(ids, len(ids))])
        boot.append(roc_auc_score(y[idx], s[idx]))
    return roc_auc_score(y, s), *np.quantile(boot, [0.025, 0.975])


def win_rate(y, s, pair):
    d = pd.DataFrame({"y": y, "s": s, "pair": pair}).pivot_table(index="pair", columns="y", values="s")
    return float(((d[1] > d[0]) + 0.5 * (d[1] == d[0])).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clinical", default="data/clinical")
    ap.add_argument("--eqtl", default="data/eQTL_v2")
    ap.add_argument("--C", type=float, default=0.1)
    args = ap.parse_args()

    pairs = pd.read_csv(os.path.join(args.clinical, "pairs.tsv"), sep="\t")
    y, pair = pairs["y"].values, pairs["pair_id"].values
    pos = pairs[pairs.y == 1].set_index("pair_id")
    strata = {"all": np.ones(len(pairs), bool),
              "2+ star pathogenic": pairs["pair_id"].map(pos["stars"] >= 2).values,
              "same gene + location": pairs["match_level"].values == 1}
    for loc in sorted(pos["location"].unique()):
        strata[f"location: {loc}"] = pairs["pair_id"].map(pos["location"] == loc).values

    all_scores = {("baseline", "-distance to TSS"): -pairs["dist"].values.astype(float)}
    for model, sub in MODELS.items():
        f_clin = os.path.join(args.clinical, sub, "matched_none", "embeddings.npz")
        if not os.path.exists(f_clin):
            print(f"skipping {model}: {f_clin} not found")
            continue
        z_clin = load(f_clin, pairs)
        z_eqtl = load(os.path.join(args.eqtl, sub, "matched_none", "embeddings.npz"))
        for name, s in scores(z_eqtl, z_clin, args.C).items():
            all_scores[(model, name)] = s

    rows = []
    for sname, mask in strata.items():
        n = int(mask.sum() // 2)
        if n < 20:
            continue
        for (model, name), s in all_scores.items():
            auc, lo, hi = auroc_ci(y[mask], s[mask], pair[mask])
            rows.append({"stratum": sname, "pairs": n, "model": model, "score": name,
                         "auroc": auc, "ci_lo": lo, "ci_hi": hi,
                         "win_rate": win_rate(y[mask], s[mask], pair[mask])})
    r = pd.DataFrame(rows)
    out = os.path.join(args.clinical, "clinical_scores.tsv")
    r.to_csv(out, sep="\t", index=False)
    pd.set_option("display.width", 160)
    for sname, g in r.groupby("stratum", sort=False):
        print(f"\n{sname}  ({g['pairs'].iloc[0]} pairs)")
        for t in g.itertuples():
            print(f"  {t.model:<9} {t.score:<17} AUROC {t.auroc:.3f} [{t.ci_lo:.3f}, {t.ci_hi:.3f}]"
                  f"   paired win {t.win_rate:.3f}")
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
