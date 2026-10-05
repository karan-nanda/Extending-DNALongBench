# Reproducing the paper

Every claim in the paper maps to one command below. Each command rewrites a file that is
already committed, so **`git diff` after a run is the check**: no diff (or a difference only
in the last floating-point digit) means the result reproduced.

Commands are written for Linux. Everything was developed on Windows 11 with WSL2 Ubuntu 24.04
and an RTX 3060 (12 GB), and the CPU and loader tiers were rerun from a fresh clone on that
Ubuntu. Times are from that machine. On Windows, use `python -X utf8` (the default cp1252
codec cannot read the ClinVar and GTF files).

## Setup

```bash
git clone https://github.com/karan-nanda/Extending-DNALongBench.git
cd Extending-DNALongBench

python3.12 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt

bash fetch_data.sh              # DNALONGBENCH @ 2c4060a + eQTL data from Dataverse (~1 GB download, ~3 GB hg38 unpacked)
```

`fetch_data.sh` pins every input (DNALONGBENCH commit, Dataverse file ids, Hugging Face
revisions, ClinVar release) and checks each file's MD5. Add `--models` for the pretrained
weights, `--embeddings` for our frozen embeddings, `--clinvar` for the ClinVar/GENCODE
inputs, or `--all`.

Caduceus (and the released-loader check, which needs `pytabix`) uses a second environment
with `mamba-ssm`. Prebuilt wheels are used, so no `nvcc` is needed, but `pytabix` compiles
against zlib (`apt install build-essential zlib1g-dev`):

```bash
bash analysis/caduceus_env_setup.sh          # venv at ~/cad; set CAD_ENV to change
```

## Tier 1: CPU only, ~3 minutes

| Claim (paper section) | Command | Expected | Writes |
|---|---|---|---|
| Distance to TSS alone beats every published model on the released split (§2, Table 1) | `python analysis/eqtl_leakage.py --root data/eQTL` | mean test AUROC 0.751; every tissue in Table 1 | `analysis/results/eqtl_leakage.tsv` |
| Same, recounted independently; tiny unstratified tests; every test gene in train (§2, R1–R2) | `python analysis/robust_published.py` | `column 0.663  coords 0.752  coords+cutoff 0.750`; 11–34 test positives per tissue | `analysis/results/robustness/R1_published_split.tsv` |
| Rebuild eQTLP-v2 (distance-matched; random / gene / chromosome folds) (§3) | `python analysis/build_eqtl_v2.py` | 5,810 matched pairs (2,905 positive); `pairs.tsv` byte-identical | `data/eQTL_v2/pairs.tsv`, `build_report.txt` |
| Shortcuts: distance 0.79 → 0.50 under matching, gene prior 0.70 → 0.44 under chromosome holdout, published mask count leaks the label; composition floor ≈ 0.60 (§3, Table 3) | `python analysis/eqtl_v2_baselines.py` | matched/chrom `dist+comp+allele` 0.590; `dist+mask_count` 0.78 → 0.44 | `analysis/results/eqtl_v2_baselines.tsv` |
| Floor is stable over 7 matchings × 10 chromosome partitions (R3) | `python analysis/robust_eqtl_v2.py --part R3` | 0.594 ± 0.006 | `analysis/results/robustness/R3_floor.tsv` |

## Tier 2: released-code defects (§5), minutes, GPU optional

| Claim | Command | Expected |
|---|---|---|
| The released HyenaDNA and Caduceus eQTL loaders yield empty inputs; without the slice the token ids are wrong (R5) | `bash analysis/check_released_loaders.sh` | `ref shape (0,), alt shape (0,)` for both repos; ids `[0, 1, 2, 3]` vs pretrained vocab `(7, 8, 9, 10)` |
| The published eQTLP CNN (`fc → ReLU` head) is dead at initialisation for some seeds and stays dead under the published optimiser (R7) | `python analysis/cnn_dead_head_check.py --published-tissue Whole_Blood --seeds 10 --train-seeds 0 --train-steps 50` (45 s) | seeds 0 and 3 `dead_train 1.00`, `grad_norm 0.00e+00`; seed 0 loss stays at exactly 0.6931 |
| Same, one full epoch for 10 seeds on two tissues (R7, as reported) | add `--train-seeds 0,1,2,3,4,5,6,7,8,9 --train-steps 0`, and repeat with `--published-tissue Nerve_Tibial` (hours) | 2/10 dead at init; 5/10 (Whole_Blood) and 6/10 (Nerve_Tibial) constant after one epoch. Logs: `analysis/results/robustness/R7_collapse_published_*.log` |

File and line references for each defect are in Appendix E.

## Tier 3: probes on our frozen embeddings, CPU only, ~20 min plus R6

`bash fetch_data.sh --embeddings` downloads the embeddings (~950 MB) from the
[`data-v1` release](https://github.com/karan-nanda/Extending-DNALongBench/releases/tag/data-v1).
To compute them yourself instead, see Tier 4.

Probe results do not reproduce bit for bit across machines, because BLAS threading changes
floating-point summation order. On a fresh Linux rerun of R4 and R4b, the logistic AUROCs moved
by at most 0.0003 and the MLP AUROCs by up to 0.006 (0.015 for the `|e_alt − e_ref|` MLP).
Δ stayed at zero and every FM stayed below the floor. The one borderline number is the
HyenaDNA chromosome-fold permutation test in R4b: p = 0.095 in the committed run and
p = 0.048 on the rerun. With 20 permutations, 0.048 is the smallest possible p, and the
paper reports the range.

| Claim | Command | Expected | Writes |
|---|---|---|---|
| Frozen FMs fall below the floor, and replacing alt by ref changes nothing (§4, Table 4; R4) | `python analysis/robust_eqtl_v2.py --part R4` (~6 min) | max \|Δ\| ≤ 0.0006; HyenaDNA 0.556, Caduceus 0.564 on chrom folds; both below the floor in 10/10 partitions | `analysis/results/robustness/R4_probes.tsv` |
| A nonlinear (MLP) head on [e_ref, e_alt] gives the allele zero weight; permutation null on the difference (R4b) | `python analysis/robust_nonlinear.py` (~11 min) | MLP head Δ = 0.0000 in all 4 settings; difference probe above its permutation null (p = 0.048) | `analysis/results/robustness/R4b_nonlinear.tsv` |
| Every layer × readout (whole / ±1 kb / ±64 bp / variant position), with a random-allele control; Figure 2 (§4, R6) | `python analysis/variant_window_probe.py` | max \|Δ\| 0.013 (HyenaDNA), 0.0165 (Caduceus); one real cell: Caduceus final layer ±1 kb, 0.580 vs 0.545 control | `analysis/results/robustness/R6_variant_window.tsv` |
| Figure 1 | `python analysis/figure1_data.py && python analysis/figure1_plot.py && python analysis/figure1_plot.py --width 6.5` | per-fold refalt = ref_copy | `analysis/results/robustness/F1_per_fold.tsv`, `paper/figures/figure1*.pdf` |
| Clinical set: all scores at chance (§6, R9) | `python analysis/clinical_score.py` | every AUROC 0.47–0.51 | `data/clinical/clinical_scores.tsv` |

## Tier 4: GPU (times on one RTX 3060, 12 GB)

```bash
bash fetch_data.sh --models
```

| Step | Command | Time |
|---|---|---|
| HyenaDNA-450k frozen embeddings (eQTLP-v2) | `python analysis/hyena_embed.py --version matched` then `--merge` | ~3 h (1.9 s/pair) |
| Caduceus-Ph frozen embeddings | `~/cad/bin/python analysis/caduceus_embed.py --bf16` then `--merge` | not recorded |
| Linear probes on them (§4) | `python analysis/hyena_probe.py --emb data/eQTL_v2/hyena_emb/matched_none/embeddings.npz` (same script for `caduceus_emb`) | minutes |
| bf16 vs fp32 precision check | `python analysis/hyena_fp32_check.py --n 400`; `~/cad/bin/python analysis/caduceus_embed.py --fp32-check` | not recorded |
| Variant-window embeddings (R6) | `python analysis/variant_window_embed.py --model hyena`; `~/cad/bin/python analysis/variant_window_embed.py --model caduceus` | ~4.5 h + ~1.5 h |
| CNN refalt with test-time ref_copy ablation, 5 chromosome folds × 3 seeds (R8) | `for f in 0 1 2 3 4; do for s in 0 1 2; do python analysis/eqtl_v2_cnn.py --version matched --scheme chrom --fold $f --mode refalt --seed $s --lr 0.001 --out analysis/results/cnn_ablation; done; done` | ~25 min per run (5 epochs) |
| End-to-end HyenaDNA fine-tuning, 3 folds × 2 epochs at 131 kb (§4, Table 5) | `bash analysis/run_finetune.sh` | ~33 h (2.5 s/pair, 6.5 GB peak) |
| Clinical embeddings | `python analysis/hyena_embed.py --v2 data/clinical` (and `caduceus_embed.py`) | not recorded |

Embedding scripts are resumable: they write shards and skip finished ones.
HyenaDNA at 450 kb fits in 12 GB for inference only; fine-tuning uses `--maxlen 131072`.

## Rebuilding the ClinVar census and clinical set

```bash
bash fetch_data.sh --clinvar        # clinvar_20260905.vcf.gz, its matching variant_summary, GENCODE v50
python build_clinvar_census.py --vcf clinvar.vcf.gz --summary variant_summary.txt.gz \
    --gtf gencode.v50.annotation.gtf.gz --outdir census/ [--min-stars 1 --outdir census_1star/]
python analysis/build_clinical_eval.py
```

## Paper

```bash
python analysis/make_appendix_tables.py      # regenerates every generated table from the result files
cd paper && latexmk -pdf recomb.tex && latexmk -pdf appendix.tex
```
