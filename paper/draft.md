# Long-context DNA models do not use the allele: an audit of the DNALongBench eQTL task

> **Historical draft, kept for provenance — not the paper.** The paper is `paper/body.tex` (PDFs: `recomb.pdf`, `workshop.pdf`, `appendix.pdf`). Numbers and claim wording here predate the robustness pass and the fine-tuning run; where they differ, the PDFs are correct.

*Working notes, 2026-09-19. **Superseded for claim wording by `workshop_draft.md` and matrix §14 (robustness).** Numbers trace to `dnalongbench_coverage_matrix.md` (§) and `analysis/`. [TODO] marks open items.*

---

## Abstract

DNALongBench evaluates long-context DNA models on five tasks, and eQTL prediction (eQTLP) is the only one that involves a genetic variant. On the released data, a baseline that ranks variants by distance to the transcription start site (TSS), with no training, reaches a mean test AUROC of 0.751 across the nine tissues. That beats the task-specific expert model (0.681) and every foundation model (0.51–0.57). The gap has three causes. Negatives lie about ten times farther from the TSS than positives, and the input padding encodes that distance. The test sets hold 11–34 positives per tissue. And every test gene also appears in training. We rebuild the task as eQTLP-v2: tissues are pooled, negatives are distance-matched, and whole chromosomes are held out, giving 2,905 positive/negative pairs and 570–665 test positives per fold. Hand-crafted local sequence features then score 0.599. Next, we ask whether models use the allele at all, by scoring each pair a second time with the alternate allele replaced by the reference. For a CNN trained end to end, and for linear probes on frozen HyenaDNA and Caduceus embeddings, the swap changes AUROC by at most 0.0004. A probe trained only on the alt − ref embedding difference scores at chance, and all three models score below the hand-feature baseline (0.51–0.56). The alleles barely register: a single-nucleotide change moves the pooled embedding by about 10⁻⁴ of its norm, in both reading orientations and in full fp32. The released eQTL code for HyenaDNA and Caduceus cannot reproduce the published numbers as shipped: the dataloader slices every input to length zero. Finally, on an evaluation-only set of 652 ClinVar pathogenic regulatory variants, each matched to a benign variant in the same gene and location class, [TODO: clinical result]. We release eQTLP-v2, the ablation harness, and the clinical set.

---

## 1. Introduction

- Long-context DNA foundation models (HyenaDNA, Caduceus, Evo) are motivated by distal regulation: enhancers and eQTLs acting tens to hundreds of kilobases from their target gene.
- DNALongBench (Cheng et al., *Nat Commun* 2025) is the standard long-range benchmark. Of its five tasks, eQTLP is the only one that asks what a *variant* does. The other four score reference sequence.
- For eQTLP, the benchmark's own interface embeds the reference and alternate sequences, averages and concatenates them, and classifies. A model can therefore score well without ever distinguishing the two alleles, if the label is predictable from *where* the variant is.
- **We test that directly.** Our contributions:
  1. **Audit of the published protocol:** a zero-training distance baseline beats all published models; the test sets are too small to rank models; and there is complete gene overlap between train and test (§3).
  2. **eQTLP-v2:** a distance-matched, chromosome-held-out version, with a CPU floor that any learned model must beat (§4).
  3. **An allele ablation, applied across architectures:** a CNN, HyenaDNA and Caduceus all score identically with and without the alternate allele (§5).
  4. **Reproducibility findings:** the released eQTL pipelines do not run as shipped, and the published CNN recipe often collapses (§6).
  5. **An evaluation-only clinical set** of matched pathogenic and benign regulatory variants from ClinVar (§7).

**Related work to position against:**
- the Genomics Long-Range Benchmark (Kao et al. 2024), which includes an OMIM pathogenic non-coding variant task;
- BEND;
- work showing that sequence-to-expression models mispredict the direction of eQTL effects (Sasse et al. 2023; Huang et al. 2023);
- shortcut learning and benchmark leakage in genomics.

Citations are in `references.bib`; the workshop draft carries the in-text keys.

---

## 2. The eQTLP task as published

- **Data:** 9 GTEx tissues. Positives are SuSiE fine-mapped eQTLs; negatives are other variant–gene pairs within 450 kb of the TSS.
- **Input:** the sequence spanning the variant and the TSS (±3 kb around the TSS, ±500 bp around the variant), right-padded with N to 450 kb. It is embedded once with the reference allele and once with the alternate.
- **Split:** "randomly split … using a stratified sampling approach with an 8:1:1 ratio", per tissue.
- **Reported (Table 7):** mean test AUROC of 0.681 for the expert model (Enformer), 0.528 for the CNN, 0.514 for HyenaDNA, and 0.566 and 0.538 for Caduceus-Ph and Caduceus-PS.

---

## 3. Audit of the published protocol (§9)

### 3.1 Distance to the TSS beats every model

Scoring each test variant by −distance to the TSS, with no training, gives AUROC 0.688–0.828 per tissue (mean 0.751). It matches or beats the expert model in 7 of 9 tissues and every foundation model in every tissue.

**Table 1.** Test AUROC on the published split. The distance baseline is our own; the other columns are Table 7.

| Tissue | Distance (95% CI) | Expert | CNN | HyenaDNA | Cad-Ph | Cad-PS | Test pos / neg |
|---|---|---|---|---|---|---|---|
| Adipose_Subcutaneous | 0.715 (0.55–0.87) | 0.736 | 0.551 | 0.513 | 0.541 | 0.519 | 16 / 316 |
| Artery_Tibial | 0.744 (0.64–0.84) | 0.741 | 0.576 | 0.479 | 0.547 | 0.536 | 27 / 542 |
| Cells_Cultured_fibroblasts | 0.688 (0.54–0.82) | 0.639 | 0.547 | 0.584 | 0.597 | 0.549 | 11 / 350 |
| Muscle_Skeletal | 0.798 (0.68–0.90) | 0.621 | 0.502 | 0.487 | 0.538 | 0.523 | 16 / 397 |
| Nerve_Tibial | 0.818 (0.76–0.88) | 0.683 | 0.516 | 0.511 | 0.588 | 0.552 | 34 / 609 |
| Skin_Not_Sun_Exposed | 0.828 (0.73–0.91) | 0.710 | 0.499 | 0.471 | 0.586 | 0.529 | 17 / 456 |
| Skin_Sun_Exposed | 0.694 (0.60–0.78) | 0.700 | 0.499 | 0.544 | 0.574 | 0.541 | 26 / 607 |
| Thyroid | 0.699 (0.58–0.81) | 0.612 | 0.487 | 0.529 | 0.527 | 0.547 | 26 / 524 |
| Whole_Blood | 0.776 (0.66–0.88) | 0.689 | 0.577 | 0.512 | 0.594 | 0.542 | 17 / 305 |
| **Mean** | **0.751** | 0.681 | 0.528 | 0.514 | 0.566 | 0.538 | |

- **Why distance works:** the median distance to the TSS is 8–33 kb for positives and 88–145 kb for negatives. And because the input is right-padded to 450 kb, the number of N tokens in every input encodes that distance.

### 3.2 The test sets cannot rank models

- **Too few positives:** 11–34 test positives per tissue. For an AUROC near 0.55, the 95% CI half-width is ±0.10–0.18.
- **No published model result is distinguishable from 0.5** in any tissue, and no two models are distinguishable from each other.
- **The split is not stratified:** positive rates are 9–21% in train, 30–43% in validation and 3–5% in test, and the realised proportions are about 60:25:15.

### 3.3 Train and test share genes and loci

- **Genes:** 100% of test pairs have their gene in training.
- **Loci:** 82–92% of test pairs have a training variant within 100 kb.
- **Variants:** 0–13% reuse the exact variant, paired with a different gene.

---

## 4. eQTLP-v2 (§10)

### 4.1 Construction

- **Pooling:** the 9 tissue tables give 25,600 unique variant–gene pairs, 3,254 of them positive.
- **Conflicting labels:** 111 pairs that are positive in one tissue and negative in another are dropped.
- **Distance matching:** each positive is matched to one negative on log₁₀ distance to the TSS, on the same side of the TSS, with caliper 0.05. This gives 2,905 positive/negative pairs. After matching, the distance-only AUROC is 0.495.
- **Folds:** three 5-fold schemes are fixed on the full pool: random (label-stratified), gene holdout, and chromosome holdout (the primary scheme).
- **Test size:** 570–665 test positives per fold.
- **No label mask:** the published loader masks the positions of all positive eQTLs between the variant and the TSS. That mask is label information, so v2 drops it.

### 4.2 Baselines

**Table 2.** Mean per-fold test AUROC ± SD. Gradient-boosted trees; the features are described in §10.2.

| Model | all / random | matched / gene | **matched / chrom** |
|---|---|---|---|
| distance (no training) | 0.790 ± .008 | 0.488 ± .039 | 0.495 ± .054 |
| gene prior (memorisation) | 0.703 ± .002 | 0.500 | 0.500 |
| composition (GC, CpG, repeats, dinucleotides) | 0.744 ± .006 | 0.608 ± .009 | 0.593 ± .018 |
| distance + composition + allele | 0.835 ± .006 | 0.604 ± .006 | **0.599 ± .031** |

- The random split rewards memorisation. A gene's training-set positive rate alone scores 0.70.
- On the honest configuration, local composition gives a floor of about 0.60, and allele features add nothing (Δ ≤ 0.01).

---

## 5. Models do not use the allele (§10.4, §11.5, §12)

### 5.1 Protocol

- **Probe:** each model sees the reference and alternate sequences separately. The last-layer states are mean-pooled, and [e_ref, e_alt] goes to a classifier, exactly as in the benchmark's Methods.
- **`ref_copy`:** at test time, the *same fitted classifier* scores [e_ref, e_ref]. If AUROC is unchanged, the classifier gets nothing from the alternate allele.
- **`diff_only`:** a classifier trained only on e_alt − e_ref. It measures whether the pooled embedding carries usable allele information at all.
- **Models:**
  - **CNN:** the benchmark's encoder, trained end to end, with the logit ReLU repaired (§6.2).
  - **HyenaDNA-medium-450k** (causal) and **Caduceus-Ph-131k** (bidirectional): frozen, with a logistic-regression probe.
- **Inputs:** unpadded. Each sequence is embedded in the published orientation (`fwd`) and in its reverse complement (`rev`), which for HyenaDNA puts the variant first.
- **Split:** matched pairs with chromosome holdout.

### 5.2 Results

**Table 3.** Mean per-fold test AUROC ± SD, matched pairs, chromosome holdout.

| Model | `refalt` | `ref_copy` | Δ | `ref_only` | `diff_only` |
|---|---|---|---|---|---|
| CNN (trained) | 0.511 ± .030 | 0.512 | −0.0002 | 0.521 ± .028 | — |
| HyenaDNA, fwd + rev (frozen) | 0.556 ± .031 | 0.556 ± .030 | −0.0001 | 0.560 ± .031 | 0.495 ± .021 |
| Caduceus-Ph, fwd + rev (frozen) | 0.564 ± .021 | 0.564 ± .021 | 0.0000 | 0.567 ± .019 | 0.507 ± .016 |
| CPU floor (Table 2) | 0.599 ± .031 | | | | |

- **Swapping the allele doesn't matter.** Replacing the alternate allele with the reference changes AUROC by at most 0.0004 on any fold, for any model and orientation.
- **The allele information isn't there to find.** `diff_only` is at chance for both foundation models.
- **All three models are below the floor.** They score below a gradient-boosted model on local sequence composition.

### 5.3 The SNP barely moves the embedding

- **Size of the shift:** ‖e_alt − e_ref‖ / ‖e_ref‖ has a median of 1.5–2.0 × 10⁻⁴ for HyenaDNA and 0.7 × 10⁻⁴ for Caduceus.
- **For the CNN** (global max-pool, 7 bp receptive field), the SNP changes the pooled embedding in only 0.4–1.7% of pairs.
- **Orientation isn't the explanation.** HyenaDNA is causal, and in the published orientation 71% of matched pairs have only the 500 bp flank after the variant (§11.4). Putting the variant first raises the shift only slightly and leaves AUROC unchanged. Caduceus, which sees the variant from both sides, behaves the same way.
- **Precision isn't the explanation.** On 400 pairs re-embedded in fp32, the SNP shift is the same size. The bf16 and fp32 difference vectors agree (median cosine 0.89–0.95), because ref and alt share their rounding. `diff_only` stays at chance.

### 5.4 Scope of the claim

- These are **frozen** foundation-model features with a linear probe, not fine-tuned models. The published FM numbers come from fine-tuning. Two points bound the gap:
  1. The CNN was trained end to end and shows the same null.
  2. A fine-tuned model would have to amplify an input-dependent shift of about 10⁻⁴ in a representation that was never trained to separate alleles.
- [TODO: decide whether to add one fine-tuning run, e.g. HyenaDNA at ≤131 kb on one fold, as a reviewer-anticipation experiment.]

---

## 6. Reproducibility of the released code (§9.4, §10.4, §11.2, §12.1)

### 6.1 The released FM eQTL loaders produce empty inputs

- **Empty inputs:** in both the HyenaDNA and Caduceus eQTL datasets, each input is padded to 450,000 characters and then sliced with `[2028500:-2028500]`, which returns an empty array.
- **Wrong token ids:** even without the slice, `argmax` over the one-hot gives token ids 0–3. In the pretrained vocabularies those are special tokens; A, C, G and T are 7–10.
- **Config vs checkpoints:** the configs set `n_layer: 2`, while the published checkpoints have 8 (HyenaDNA) and 16 (Caduceus). The pretrained-weights path is left unset.
- **HyenaDNA forward pass:** `EQTLModel.forward` references an undefined variable.
- **Consequence:** Table 7's FM columns cannot have come from this code unchanged.

### 6.2 The published CNN recipe often collapses

- **Why:** the eQTLP head ends in `fc → ReLU` on the logits. When both logits clamp to zero, the prediction is constant and every weight gets zero gradient.
- **At initialisation:** 2 of 10 initialisations are dead from the start.
- **During training:** all 3 live seeds we trained collapsed within one epoch.
- **In the training script:** no seed is set, and the script evaluates the final-epoch model rather than the best checkpoint.
- **Separately,** the CNN training script feeds only the reference sequence, so the published CNN is a ref-only model.

---

## 7. Clinical evaluation set (§8, [TODO §13])

- **Pathogenic set:** ClinVar (release 2026-09-06) pathogenic or likely pathogenic SNVs with at least 1 star. They must be non-coding by HGVS location, more than 50 bp from a splice site, not overlapping CDS in any GENCODE v50 isoform, and not in a small structural RNA. That leaves 1,169 variants; 696 are SNVs, and 652 map to a gene TSS within 450 kb.
- **Benign set:** benign or likely benign SNVs passing the same filters (112,775).
- **Matching:** each pathogenic variant gets one benign match, preferring the same gene and location class (555 pairs), then the same gene (77), then the same location class (20), with the closest distance to the TSS. Median distance is 26.7 kb vs 26.6 kb.
- **Scoring (evaluation only, no clinical labels used for fitting):**
  - the embedding shift ‖e_alt − e_ref‖;
  - transfer of the eQTLP-v2 probe;
  - that probe's allele-only component (alt minus ref-copy);
  - a distance baseline.
- **Results:** [TODO: results table from `analysis/clinical_score.py`, by stratum: all, 2★, same gene and location, and location class.]
- **Why the set is evaluation-only:** no ClinVar-derived regulatory set reaches 1,500 positives once near-splice, CDS-overlapping and structural-RNA variants are removed (409 at 2★). Almost all pathogenic SVs that fit the context overlap coding exons (126 of 5,498 at ≥1★ do not).

---

## 8. Discussion

- **What eQTLP measures:** as published, eQTLP mostly measures distance to the TSS and gene identity. Once those are removed, local sequence composition carries the remaining signal. None of the evaluated models reads the allele.
- **Recommendations for variant benchmarks:**
  - match negatives on distance (and on gene where possible);
  - hold out whole chromosomes or genes;
  - make test sets big enough for ±0.05 CIs;
  - report a ref-copy ablation next to every variant result;
  - publish a hand-feature floor.
- **Why the SNP barely registers:** mean-pooling over tens of kilobases dilutes a one-base change, and pretraining on the reference genome gives no incentive to separate alleles. Variant-aware readouts (for example, the embedding at the variant position, or a local window) are the obvious next test. [TODO: consider a variant-centred pooling probe; it is cheap with the existing embedding code.]
- **Limitations:**
  - one causal and one bidirectional FM, both small (HyenaDNA-medium-450k 6.55 M parameters, Caduceus-Ph-131k 7.73 M);
  - no Evo 2 or GENERator (compute);
  - frozen probes for the FMs;
  - the clinical set is small and concentrated in a few genes (33 genes cover half of the 1★ core).

---

## Methods (to expand)

- **Data:**
  - DNALongBench eQTL tables: Dataverse doi:10.7910/DVN/YUP2G5.
  - hg38.
  - GENCODE v50.
  - ClinVar 2026-09-06.
- **Embeddings:**
  - bf16 autocast; the FFT convolution runs in fp32 for HyenaDNA.
  - Unpadded inputs; mean over positions of the last hidden layer.
  - Scripts: `hyena_embed.py`, `caduceus_embed.py`.
- **Probe:** standardised logistic regression, C = 0.1, fit per fold (`hyena_probe.py`).
- **CNN:** `eqtl_v2_cnn.py`, AdamW, lr 1e-3.
- **Hardware:** one RTX 3060 (12 GB). Caduceus runs under WSL2 with mamba-ssm 1.2.0.post1.

## Artefacts

- **Data:** `data/eQTL_v2/pairs.tsv` and `data/clinical/pairs.tsv`.
- **Code:** all `analysis/*.py`.
- [TODO: package as a pip-installable loader matching `load_data(root, task_name, subset, batch_size)`.]
