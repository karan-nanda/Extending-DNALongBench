# DNALongBench Coverage Audit

**Source:** Cheng et al., *Nat Commun* 16:10108 (2025), doi:10.1038/s41467-025-65077-4
**Code:** github.com/ma-compbio/DNALONGBENCH (BSD-3) · Zenodo 10.5281/zenodo.17179568
**Audit date:** Week 1, this project
**Purpose:** establish, from the paper's own Methods, exactly which axes the benchmark covers — and locate a defensible gap.

---

## 1. Task-by-task specification

| | **ETGP** | **CMP** | **RSAP** | **TISP** | **eQTLP** |
|---|---|---|---|---|---|
| Full name | Enhancer–target gene prediction | 3D chromatin contact map | Regulatory sequence activity | Transcription initiation signal | eQTL prediction |
| Task type | Binary classification | 2D regression | Multitask regression | Multitask regression | Binary classification |
| Input length | ≤450 kb (pair within 450 kb of TSS) | 1 Mb | 196,608 bp | 100 kb sliding, center 50 kb scored | ≤450 kb (variant→promoter + 3 kb past TSS) |
| Output granularity | Sequence-wide (1 label) | 448×448 @ 2 kb bins | 896 bins @ 128 bp | Base-pair resolution | Sequence-wide (1 label) |
| Label source | CRISPRi screens (Fulco '19, Gasperini '19, Schraivogel '20) | Hi-C / Micro-C (Akita + 4DN) | ENCODE/FANTOM tracks via Enformer | CAGE, RAMPAGE, GRO-cap, PRO-cap via Puffin | GTEx, SuSiE fine-mapped, via Enformer |
| Biological contexts | K562 only | 9 cell lines: HFF, H1-hESC, GM12878, IMR-90, HCT116, HAP1, HeLa, HepG2, K562 | 5,313 human + 1,643 mouse tracks | 5 assays | 9 GTEx tissues (top 9 of 48 by variant count) |
| Species | Human | Human | **Human + mouse** | Human | Human |
| Dataset size | — | 7,008 / 419 / 413 | 38,171 human (34,021/2,213/1,937); 33,521 mouse | chr8+chr9 held out | — |
| Split strategy | — | Non-overlapping virtual contigs, random 8:1:1 | Enformer splits | **Chromosome holdout (chr8, chr9)** | **Random stratified 8:1:1** |
| Metric | AUROC, AUPRC | SCC, PCC | PCC | PCC | AUROC |
| Expert model | ABC | Akita | Enformer | Puffin-D | Enformer |
| Best reported | — | SCC 0.233 (Akita) | — | 0.733 (Puffin-D) vs 0.132 HyenaDNA | — |

**Models evaluated across all five:** 3-layer CNN · task-specific expert · HyenaDNA (medium-450k) · Caduceus-Ph · Caduceus-PS.
Transformer FMs (DNABERT-1/2, Nucleotide Transformer) explicitly excluded — quadratic attention infeasible at these lengths.

---

## 2. Coverage matrix — what is and isn't tested

### 2.1 Variant-level coverage

| Axis | Covered? | Detail |
|---|---|---|
| Tasks that involve a variant at all | **1 of 5** | eQTLP only. ETGP, CMP, RSAP, TISP all score *reference* sequence. |
| SNVs | Yes | eQTLP: paper states "Positive SNPs were identified using SuSiE" |
| Insertions / deletions | **No** | No indel anywhere in the suite |
| Structural variants (>50 bp), CNVs, inversions | **No** | — |
| Repeat expansions | **No** | — |
| Coding variants | **No** | Entire benchmark is non-coding-facing |
| Splice-region variants | **No** | No splicing task exists |

### 2.2 Label-semantics coverage

| Label type | Covered? |
|---|---|
| Molecular assay readout (expression, contact freq., initiation, accessibility) | Yes — all five tasks |
| Statistical genetics (fine-mapped association) | Yes — eQTLP |
| **Clinical / disease causality** | **No — zero tasks** |
| Functional assay of variant effect (MPRA, saturation mutagenesis) | No |

### 2.3 Species and ancestry

| Axis | Covered? |
|---|---|
| Human (hg38) | Yes — all five |
| Mouse | RSAP only |
| Other species | No |
| Non-European ancestry | Not addressed; GTEx donor pool is predominantly European |

### 2.4 Structural-variation × 3D genome

The benchmark contains a 1 Mb contact-map task **and** a variant task, but never crosses them. No task asks what a variant does to chromatin architecture. This is the single largest structural hole, and it sits precisely where the benchmark's stated purpose — long-range dependency — is most clinically consequential.

---

## 3. Positioning against prior benchmarks

The paper's own Table 1 compares against Genomic Benchmarks, NT Benchmark, GUE, BEND, and LRB. Two of these matter for us:

- **BEND** — long-range tasks are enhancer annotation and gene finding. Both are element classification on reference sequence. No clinical variants.
- **Genomics Long-Range Benchmark (LRB, Kao et al. 2024)** — adapted from Enformer; three datasets on expression prediction and variant effects on expression. **LRB includes an OMIM variant-effect task on pathogenic Mendelian non-coding variants.** DNALongBench's own eQTLP follows LRB-compatible splits.

**Consequence for our framing:** "clinical variants are untested" is false at the field level — LRB tests OMIM pathogenic non-coding SNVs. A reviewer will raise this. The claim that survives is narrower and stronger:

> No existing long-range DNA benchmark tests whether a model can predict a variant's effect on **3D regulatory architecture**, and none tests **non-SNV clinical variants** of any kind.

---

## 4. Candidate gaps, ranked

### G1 — Variant effects on 3D genome organization *(recommended)*
**Claim:** DNALongBench predicts contact maps from reference sequence only; it never asks what a variant does to that map. LRB and BEND have no structural task at all.
**Clinical grounding:** TAD-boundary disruption and enhancer hijacking cause Mendelian disease (limb malformations at the EPHA4 locus; enhancer hijacking in medulloblastoma; several documented CTCF-boundary deletions).
**Why it's the right gap:** it exercises exactly the capability the benchmark exists to measure, at exactly the length scale it claims to cover, and the failure mode is interpretable.
**Cost:** highest. Needs pathogenic SVs with matched Hi-C or a validated proxy label (predicted boundary-insulation delta). Label noise is the main risk.

### G2 — Pathogenic indels and small SVs *(recommended as companion or fallback)*
**Claim:** every long-range genomic FM benchmark is SNV-only. ClinVar contains thousands of pathogenic deletions, duplications, and insertions.
**Why it's the right gap:** indels break the fixed-length-window assumption every one of these models relies on; the models were never designed for length-changing edits. Cheap to build, and the negative result is close to guaranteed and genuinely informative.
**Cost:** low. This is the de-risked option.

### G3 — Deep-intronic and non-coding pathogenic SNVs by disease panel
**Cost:** low, but overlaps LRB-OMIM substantially. Weakest standalone contribution. Useful only as a *calibration arm* — a category where LRB numbers exist, so you can show your harness reproduces known behaviour before reporting on G1/G2.

**Recommended shape:** G2 as the core task (guaranteed deliverable), G1 as the headline task (higher risk, higher value), G3 as a calibration arm. All three share one loader and one evaluation protocol.

---

## 5. Two exploitable weaknesses in the original protocol

These are not criticisms to lead with, but they are levers.

**5.1 The eQTLP split is random, not chromosomal.** The paper states the eQTL dataset was "randomly split into training, validation, and test sets using a stratified sampling approach with an 8:1:1 ratio." Meanwhile CMP uses virtual contigs and TISP holds out whole chromosomes. A random split over variant–gene pairs lets the same locus appear in train and test. Our task should use chromosome *and* gene holdout, and we can report both splits to quantify what the difference is worth. If it's large, that is a finding about the original benchmark, delivered constructively.

**5.2 The eQTL model interface is already variant-aware — and it's trivially ablatable.** From Methods: for eQTLP they "extracted last-layer hidden representations from both the reference and allele sequences, averaged and concatenated them, and applied a binary classification layer."

This is the hook for the whole paper. That interface means:
- Our task drops into the existing harness with near-zero architectural change. Same embed-ref, embed-alt, concat, classify.
- The **ref-only ablation** is a two-line edit: replace the alt embedding with a copy of the ref embedding. If AUROC barely moves, the classifier is reading *genomic location*, not *allele identity*. Given that these models are pretrained on the reference genome and never optimized for allele discrimination, this is the predicted result — and it would apply to the original eQTLP task too.

Run the ablation on **their** eQTL task first, during the Week 2–4 reproduction step. If it fires there, the paper writes itself.

---

## 6. Hard constraints inherited from the harness

- **450 kb ceiling.** HyenaDNA medium-450k caps context at 450 kb; this is why ETGP and eQTLP both filter to 450 kb of the TSS. Our task must respect the same ceiling or lose direct comparability.
- **Input format is BED.** The paper notes BED "allows flexible adjustment of the flanking context without requiring reprocessing." Our dataset ships as BED + a variant table, same convention.
- **Loader signature.** `load_data(root, task_name, subset, batch_size)` → train/valid/test loaders. Match it.
- **No transformer FMs.** Don't add DNABERT or NT; the original excluded them for compute reasons and adding them breaks comparability.

---

## 7. Week 1–2 gate

| Item | Status |
|---|---|
| Coverage matrix complete | **Done** |
| Gap identified and defended against LRB | **Done — G1/G2, see §3–4** |
| ClinVar positives counted after filtering | **Done** — `build_clinvar_census.py`, see §8 |
| Go/no-go threshold | ≥1,500 usable positives after 2★ + non-coding + splice-exclusion filters |
| **Gate result** | **FAIL** — 409 at 2★, 1,169 at 1★ (regulatory core, §8.1) |
| Reproduction environment built | Pending — Week 2 |

---

## 8. ClinVar census results

**Inputs:** ClinVar GRCh38 VCF + `variant_summary.txt.gz` (release of 2026-09-06), GENCODE v50 comprehensive annotation.
**Command:** `python -X utf8 build_clinvar_census.py --vcf clinvar.vcf.gz --summary variant_summary.txt.gz --gtf gencode.v50.annotation.gtf.gz --outdir census/ [--min-stars 1]`
**Outputs:** `census/` (2★) and `census_1star/` — `report.txt`, `clinvar_filtered.tsv`, `clinvar_sv.tsv`.

### 8.1 Small variants (≤50 bp): the gate fails

The first version of the script filtered on ClinVar's MC consequence field only and reported 1,683 non-coding positives (PASS). That count was wrong: ClinVar only uses `splice_donor/acceptor_variant` for ±1/±2, so `c.X+5G>A` and `c.X-12A>G` come through as `intron_variant`. 77% of those 1,683 were within 50 bp of a splice site.

| Filter step | 2★+ | 1★+ |
|---|---|---|
| MC non-coding or unannotated | 2,242 | 5,933 |
| …not coding, not ≤50 bp from a splice site (HGVS offset) | 599 | 1,848 |
| …not overlapping CDS in any GENCODE isoform | 510 | 1,399 |
| …not in a small structural-RNA exon (RMRP, RNU4ATAC, RNU4-2, RNU2-2, TERC) | 409 | 1,169 |
| **Regulatory core (gate count)** | **409** | **1,169** |
| Strict core (also drops deep-intronic, mostly pseudoexon/splicing) | 211 | 636 |

- Widening the disease panel cannot fix this; the census already covers all of ClinVar.
- Even at 1★ the core is below 1,500, and 1★ means single-submitter labels that need a manual audit.
- **Gene concentration is high:** at 1★, 33 genes cover 50% of the core and the top 20 cover 44%. RMRP promoter variants alone are ~15% (182). Gene-level holdout is mandatory, not optional.
- **Deep-intronic variants are counted in the gate** but most act by activating cryptic exons, which is still a splicing mechanism. Treat the strict core as the conservative number.

**Consequence for G3:** underpowered as a standalone task. It can still serve as a small calibration arm (~400–1,200 positives) if paired with matched negatives. The benign pool is large in every location class (e.g. 19k deep-intronic, 3k 5′UTR/promoter at 2★).

### 8.2 Structural variants (>50 bp)

The VCF omits most CNVs, so SV counts come from `variant_summary`.

| Pathogenic SVs that fit ≤450 kb | ≥0★ | ≥1★ | ≥2★ |
|---|---|---|---|
| All | 7,054 | 5,498 | 329 |
| **Missing every coding exon (any isoform)** | **212** | **126** | **2** |

- Almost every pathogenic SV that fits the context overlaps a coding exon. A model can score those by asking "did it hit an exon"; they don't test long-range regulation.
- The 2★ rule eliminates SVs entirely: CNV submissions are rarely multi-submitter.
- At ≥1★, the 126 non-coding SVs are mostly deletions: 52 of 51 bp–1 kb, 29 of 1–10 kb, 17 of 10–100 kb, and 3 of 100–450 kb.

**Consequence for G1 (3D architecture):** ClinVar cannot supply the positives. It needs curated sources: DECIPHER, dbVar clinical, and literature catalogues of TAD-boundary/enhancer-hijacking cases (EPHA4, SOX9, LMNB1, SHH-ZRS, etc.). Expect low hundreds of cases with heavy locus clustering, so frame it as a curated challenge set rather than a trainable task.

**Consequence for G2 (indels/SVs):** viable only if coding-exon-overlapping SVs are allowed, which changes the question from long-range regulation to variant length handling. If G2 stays, report CDS-overlapping and non-CDS SVs as separate strata.

### 8.3 Decision needed before Week 2

The ≥1,500 threshold assumed a trainable fine-tuning task. No ClinVar-derived long-range set reaches it. Options:
1. **Evaluation-only benchmark.** Zero-shot / frozen-embedding scoring of the ~1,169 (1★) regulatory core and the curated SV set, with matched negatives and no fine-tuning. The threshold then drops to what a stratified AUROC CI needs, a few hundred per stratum.
2. **Add non-ClinVar positives** (HGMD regulatory, literature curation) to reach 1,500. This costs weeks and brings licensing issues (HGMD).
3. **Lead with the protocol findings** (§5: ref-only ablation, random vs chromosome/gene split on eQTLP) and keep the clinical arm as a small evaluation-only set.

**Decision (2026-09-10): option 3, with option 1 as the clinical arm.**
- **Headline:** protocol findings on the published eQTLP task. These are (a) the ref-only and shuffled-alt ablations, and (b) random 8:1:1 vs chromosome holdout vs gene holdout.
- **Clinical arm:** evaluation-only, with no fine-tuning on clinical labels. It covers the 1★ regulatory core (1,169, with the 2★ subset of 409 reported separately) and the non-CDS SVs (126 at ≥1★), scored with the eQTLP-fine-tuned heads and frozen embeddings, against matched benign negatives. Results are reported per stratum with CIs.
- **Week 2–4 gate is unchanged:** reproduce the published eQTLP AUROC first.

---

## 9. eQTLP protocol audit (no GPU needed)

**Data:** Dataverse doi:10.7910/DVN/YUP2G5, original TSVs (`?format=original`; the Dataverse-ingested `.tab` copies do not match the config filenames).
**Script:** `python -X utf8 analysis/eqtl_leakage.py --root data/eQTL` → `analysis/results/eqtl_leakage.tsv`.
**Published numbers:** Table 7 of the Nat Commun paper (PMC12627797).

### 9.1 Variant-to-TSS distance alone beats every published model

| Tissue | Distance only (95% CI) | Expert | CNN | HyenaDNA | Cad-Ph | Cad-PS | Test pos / neg |
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

- Negatives are not distance-matched. Median distance to the TSS is 8–33 kb for positives vs 88–145 kb for negatives.
- The loader leaks this distance into the input: it right-pads the variant→TSS sequence with N to 450 kb, so the padding length encodes distance.
- The distance baseline needs no training: it scores each test variant by −distance.

### 9.2 Test sets are too small to rank models

- **The split is not stratified.** Positive rates differ by split: train 9–21%, valid 30–43%, test 3–5%. Actual proportions are about 60:25:15, not the stated 8:1:1.
- **Test positives are 11–34 per tissue.** The 95% CI half-width for an AUROC near 0.55 is ±0.10–0.18 (Hanley–McNeil).
- **No foundation-model or CNN result in Table 7 is distinguishable from 0.5 in any tissue,** and no two models are distinguishable from each other.

### 9.3 Every test gene is also a training gene

- 100% of test rows have their gene in train.
- 82–92% of test rows have a train variant within the same 100 kb window.
- 0–13% of test rows reuse the exact variant, paired with a different gene.
- Chromosome or gene holdout will remove this; the redesigned splits must also match negatives on distance.

### 9.4 Code-level observations

- **The CNN baseline never sees the alt allele.** `experiments/CNN/train.py:48,118` feeds only `batch['x_ref']`. Its published eQTLP numbers are already a ref-only model.
- **HyenaDNA `EQTLModel.forward` does not run as shipped.** `experiments/HyenaDNA/.../long_conv_lm.py:627` references an undefined `hidden_states`. It also concatenates full per-position hidden states rather than the averaged representations the Methods describe.

### 9.5 Revised Week 2–4 plan

1. **Redesigned eQTLP splits:** chromosome holdout and gene holdout, with distance-matched negatives and tests large enough for ±0.05 CIs (pool tissues or use `combined_data.tsv`). Add baselines for distance only, distance + composition, and the CNN in ref-only and ref+alt modes.
2. **HyenaDNA-450k reproduction** on the published split, using `standalone_hyenadna.py` plus HF weights, which avoids flash-attention on Windows. Then the ref-only and shuffled-alt ablations on both old and new splits.
3. **Caduceus** needs `mamba_ssm`, which means WSL2 or a cloud GPU. Defer it until HyenaDNA runs.

---

## 10. eQTLP-v2: redesigned splits and CPU baselines (step 1)

**Scripts:**
- `analysis/build_eqtl_v2.py` → `data/eQTL_v2/pairs.tsv`, `build_report.txt`
- `analysis/eqtl_v2_baselines.py` → `analysis/results/eqtl_v2_baselines.tsv`

### 10.1 Construction

- **Pooled pairs:** the 9 tissue tables give 25,600 unique (variant, gene) pairs, 3,254 of them positive. 111 pairs were positive in one tissue and negative in another; they are dropped.
- **Distance matching (`matched`):** 1:1 nearest neighbour on log10 distance to the TSS, within the same side (upstream/downstream), caliper 0.05. This gives 5,810 pairs (2,905 positive).
  - 349 positives go unmatched, mostly ones within ~2 kb of the TSS, where negatives are scarcer than positives.
  - Distance-only AUROC drops from 0.790 to 0.500. The median distance becomes 21.2 kb for both classes.
- **Folds:** three 5-fold schemes, fixed on the full pool:
  - `random`: label-stratified.
  - `gene`: whole gene groups held out. Genes sharing a candidate variant are merged into one group.
  - `chrom`: whole chromosomes held out, balanced on positives.
- **Test size:** 570–665 test positives per fold, vs 11–34 per tissue in the published split.
- **No label masking:** the published loader masks all positive eQTLs between the variant and the TSS. The "blacklist" is exactly the 8,501 positives, so the mask is label information. v2 drops it.

| Train→test overlap (share of test pairs) | Same gene | Same variant | Same 100 kb locus |
|---|---|---|---|
| random | 1.000 | 0.280 | 0.967 |
| gene | 0.000 | 0.000 | 0.148 |
| chrom | 0.000 | 0.000 | 0.000 |

**Caveat: one gene group is very large.** It holds 5,664 pairs (22%) across 7 chr17 genes, with only 20 positives. In the `all` version this makes fold prevalence range from 6% to 18%; `matched` folds are balanced.

The primary metric is therefore the **mean of per-fold AUROCs**. Pooled out-of-fold AUROC mixes fold-level calibration: a score that is constant within each fold lands below 0.5 purely from prevalence differences.

### 10.2 Baselines: mean per-fold AUROC ± SD

| Model | all / random | all / gene | all / chrom | matched / random | matched / gene | **matched / chrom** |
|---|---|---|---|---|---|---|
| distance (no training) | 0.790 ± .008 | 0.768 ± .034 | 0.770 ± .055 | 0.500 ± .015 | 0.488 ± .039 | 0.495 ± .054 |
| gene_prior (memorisation) | 0.703 ± .002 | 0.500 | 0.500 | 0.707 ± .009 | 0.500 | 0.500 |
| composition (no distance) | 0.744 ± .006 | 0.669 ± .011 | 0.668 ± .028 | 0.665 ± .013 | 0.608 ± .009 | 0.593 ± .018 |
| dist + comp | 0.833 ± .008 | 0.779 ± .031 | 0.784 ± .049 | 0.670 ± .017 | 0.610 ± .005 | 0.590 ± .029 |
| dist + comp + allele | 0.835 ± .006 | 0.780 ± .029 | 0.783 ± .047 | 0.670 ± .018 | 0.604 ± .006 | **0.599 ± .031** |
| dist + mask_count | 0.811 ± .007 | 0.767 ± .035 | 0.768 ± .047 | 0.529 ± .014 | 0.463 ± .014 | 0.444 ± .012 |

Composition features are GC, CpG observed/expected and repeat fraction in ±50 bp / 500 bp / 5 kb windows, plus dinucleotides in ±500 bp. Allele features are the ref/alt base, transition vs transversion, and CpG site. All learned baselines use gradient-boosted trees.

### 10.3 Findings

1. **Distance explains the unmatched task.** With no training it reaches 0.77–0.79, and it goes to chance once negatives are distance-matched.
2. **The random split rewards memorisation.** A gene's training-fold positive rate alone scores 0.70 on the random split and exactly 0.50 under gene or chromosome holdout. Composition features lose 0.06–0.08 moving from random to holdout.
3. **The honest floor is ~0.59–0.61** (matched, gene or chrom holdout), and **allele identity adds nothing** (Δ ≤ 0.01, within one fold SD). A model must beat ~0.60 to show it learned anything beyond local sequence context. It must also beat its own ref-only ablation to show it uses the allele.
4. **Published masking leaks a little and unstably.** Counting masked positives adds +0.02 over distance on the random split. It is informative but its direction flips under matched holdout (0.44–0.46).
5. **The published CNN cannot use context.** `experiments/CNN/cnn.py` is three kernel-3 convolutions and a global max-pool: a 7 bp receptive field that is blind to padding length. That is consistent with its chance-level Table 7 numbers.

**Primary benchmark configuration:** `matched / chrom`, with `matched / gene` as secondary. `all / random` is reported only to show the published-protocol inflation.

### 10.4 CNN runs (GPU)

`analysis/eqtl_v2_cnn.py` has four modes:
- `published`: the repo CNN, ref only.
- `refalt`: a siamese encoder with concatenated embeddings, the FM protocol from the paper's Methods.
- `ref_only`: the same model with the alt embedding replaced by a copy of the ref embedding.
- `random_allele`: the same model with the alt allele swapped for a random base.

**Finding: the published eQTLP CNN often never trains.** `SimpleCNN(task='eQTLP')` ends in `fc → ReLU`. Both logits can clamp to 0, which gives a constant 0.5 prediction and zero gradient to every weight. Evidence from `analysis/cnn_dead_head_check.py`, run in fp32 on real training batches as published:
- **Init seeds 0–9:**
  - 2 of 10 are fully dead in train mode (gradient norm exactly 0), and 1 is 31% dead.
  - 3 of 10 give constant logits in eval mode at initialisation.
- **Seed 0, trained 150 steps with the published optimiser** (AdamW lr 0.005, wd 0.01, batch 2):
  - Loss is exactly 0.6931 on every step.
  - Every weight moves by exactly the weight-decay-only amount (0.0075).
  - Only BatchNorm running statistics change, so eval-mode predictions drift without anything being learned.
- **Seeds that start alive collapse within one epoch.** I trained seeds 1, 2 and 4 for 1,800 steps, one epoch of the matched/chrom fold-0 training pairs:
  - Seeds 2 and 4 were fully dead within the first 150 steps. Seed 1 died between steps 600 and 900.
  - After collapse, loss is exactly 0.6931 on every step and eval-mode logits are constant (SD 0). A test set scored at that point gets AUROC exactly 0.5.
  - The weights did move before collapse (relative change 0.23–1.64, vs 0.086 from decay alone), so these seeds learned briefly and then died.
  - Log: `analysis/results/cnn_dead_head_epoch.log`.
- **Tally:** 2 of 10 initialisations are dead from the start, and all 3 live seeds I trained for a full epoch collapsed.
- **The published training script adds to the fragility:**
  - `experiments/CNN/train.py` sets no seed.
  - It saves `best_model.pt` by validation AUROC but evaluates the *final-epoch* model on test, without reloading the best checkpoint (lines 80–93).
- **What this does and does not show about Table 7:** its CNN values (0.487–0.577) are not exactly 0.5, so the published runs were not all fully collapsed when tested. The finding is that the recipe is unreliable, not that Table 7 is wrong.
  - This test used v2 training pairs, not the original per-tissue tables. The mechanism is architectural (a ReLU on the logits), but a run on the published split would tie it to Table 7 directly.

**Consequence for our runs:** a `published`-mode sweep would measure collapse, not the task. `ref_only` is the working equivalent of the published CNN: the same encoder on the ref sequence only, with the logit ReLU replaced by LayerNorm and a linear head. It doubles as the repaired CNN baseline.

**Sweep 1: separately trained models** (matched/chrom, lr 0.001, seed 0, `analysis/results/cnn/`)

| Test AUROC | f0 | f1 | f2 | f3 | f4 | Mean ± SD |
|---|---|---|---|---|---|---|
| `ref_only` (repaired CNN) | 0.540 | 0.548 | 0.509 | 0.478 | 0.528 | 0.521 ± 0.028 |
| `refalt` | 0.482 | 0.562 | 0.502 | 0.580 | 0.607 | 0.547 ± 0.053 |
| CPU floor (dist + comp + allele) | | | | | | 0.599 ± 0.031 |

- **Both CNNs sit below the CPU floor,** except `refalt` on fold 4 (0.607).
- **The paired difference (`refalt` − `ref_only`) flips sign across folds:** −0.059, +0.014, −0.007, +0.102, +0.079 (mean +0.026 ± 0.058).
- **This comparison cannot establish allele use.** The two arms are separately trained models, and the per-fold CIs cover test-pair sampling only, not training variance. Swings of ±0.1 between folds are what training noise between two models looks like.

**Sweep 2: test-time ablation on the same weights** (`analysis/results/cnn_ablation/`, weights saved as `.pt`)

This is the plan's ablation: each trained `refalt` model scores the same test pairs three ways.

| Test AUROC, matched/chrom | f0 | f1 | f2 | f3 | f4 | Mean ± SD |
|---|---|---|---|---|---|---|
| True alt | 0.469 | 0.544 | 0.536 | 0.511 | 0.498 | 0.511 ± 0.030 |
| Alt → copy of ref | 0.469 | 0.544 | 0.536 | 0.511 | 0.499 | 0.512 |
| Random alt allele | 0.469 | 0.544 | 0.536 | 0.511 | 0.499 | 0.512 |
| Pairs whose pooled embedding changes | 0.7% | 0.7% | 1.7% | 0.4% | 0.4% | |

- **True alt − ref copy is −0.0002 ± 0.0003.** Every fold's 95% CI includes zero, and the largest single-fold gap is 0.0007.
- **The allele reaches the classifier in about 1% of pairs.** The SNP changes the max-pooled embedding in only 0.4–1.7% of test pairs. With a 7 bp receptive field and a global max over 450 kb, one base almost never moves the maximum.
- **The CNN is at chance on this split** (0.511), well below the CPU floor of 0.599.
- **Training variance is about 0.04.** The same configuration trained in sweep 1 scored 0.547 ± 0.053, which confirms that sweep 1's `refalt` − `ref_only` gaps (±0.1) were training noise, not allele use.

**Conclusion (CNN):** the ablation pipeline works end to end, and for this CNN it shows the answer is architectural. It reads position, not allele, and cannot do otherwise. This is a harness check, not a headline. The open question is the same test on HyenaDNA and Caduceus, whose mean-pooled hidden states can in principle carry the allele.

---

## 11. Step 2: HyenaDNA-medium-450k

### 11.1 Setup on this machine

- **Weights:** `LongSafari/hyenadna-medium-450k-seqlen` (`weights.ckpt`): 8 layers, d_model 256, l_max 450,002. Stored in `data/models/`.
- **Model code:** the repo's `standalone_hyenadna.py`, plain PyTorch with no flash-attention.
- **One patch:** the long FFT convolution runs in fp32, because cuFFT has no bf16 path.
- **Added packages:** `einops` (needed by the standalone model) and `omegaconf` (needed to unpickle the Lightning checkpoint).
- **Memory and speed check:** `analysis/hyena_memcheck.py`. Results in 11.3.

### 11.2 The released HyenaDNA eQTL pipeline, as shipped

I read this code but did not run it; the full pipeline needs hydra, Lightning and flash-attention.

- **The dataloader empties the input.**
  - `parse_eQTL` pads every sequence to 450,000 characters (`src/dataloaders/datasets/eqtl_dataset.py:275–282`).
  - `__iter__` then slices the one-hot with `[2028500:-2028500]` (lines 104 and 106). On a 450,000-long array that is empty; I confirmed it returns shape `(0, 4)`.
  - The released dataloader therefore cannot have produced Table 7 unchanged. The slice looks like a leftover from a different input length.
- **The token ids don't match the pretrained vocabulary.** Even with the slice removed:
  - `np.argmax` of the one-hot gives 0–3 for A/C/G/T.
  - The pretrained character tokenizer maps A/C/G/T/N to 7–11; ids 0–3 are `[CLS]`/`[SEP]`/`[BOS]`/`[MASK]`.
  - N (0.25 in every column) becomes 0, the same as A.
- **The model's forward pass is broken and differs from the Methods.**
  - `EQTLModel.forward` references an undefined `hidden_states` (`long_conv_lm.py:627`).
  - It applies the head per position to the concatenated full hidden states, rather than averaging them as the Methods describe.
- **The experiment config** (`configs/experiment/hg38/eqtl_benchmark.yaml`):
  - `n_layer: 2`, while the pretrained model has 8.
  - `l_max: 450000`, while the pretrained model uses 450002.
  - lr 5e-6, weight decay 0.1, 30 epochs, batch 1, fp16, grad clip 1.0, cosine warmup, seed 2222.
  - The data module hardcodes `tissue = "Whole_Blood"`.
- **Consequence:** Table 7's HyenaDNA column cannot be reproduced from the released code as-is. Our reproduction reimplements the protocol the Methods describe: average the last-layer hidden states for ref and alt, concatenate, and apply a binary classifier. It uses the config's hyperparameters where they are unambiguous.
  - It is worth asking the authors which commit and layer count produced Table 7.

### 11.3 Memory and speed on the RTX 3060 (12 GB)

Measured with `analysis/hyena_memcheck.py`: bf16 autocast, fp32 FFT, allocator capped at 92% of VRAM (11.3 GB).

The cap matters. Without it, the Windows driver spills into system RAM instead of raising out-of-memory, and an over-budget run thrashes for hours.

| Input length | Inference, 1 sequence | Train step, 1 sequence (checkpointed) | Train step, ref + alt (checkpointed) |
|---|---|---|---|
| 450,000 | 3.2 s, 8.7 GB | out of memory | out of memory |
| 262,144 | 1.7 s, 5.1 GB | out of memory | out of memory |
| 131,072 | 0.9 s, 2.6 GB | 3.0 s, 5.2 GB | 6.1 s, 6.8 GB |

Without activation checkpointing, training is out of memory even at 131k.

**Real input lengths** (the variant→TSS span with flanks, before N padding; `data/eQTL_v2/pairs.tsv`):
- **matched:** median 24.7 kb. 87.2% are ≤131k and 95.6% ≤262k. On average, real bases are 13% of the 450k padded input.
- **all:** median 99 kb. Positives are much shorter than negatives (88.5% vs 53.1% ≤131k), which is the distance leak seen through padding length.

**With the FFT chunked over channels** (64 at a time; `FFT_CHUNK` in `hyena_memcheck.py`). The long convolution is depthwise, so the chunks are independent and the output is identical:

| Input length | Inference, 1 sequence | Train step, 1 sequence (checkpointed) | Train step, ref + alt (checkpointed) |
|---|---|---|---|
| 450,000 | 3.1 s, 5.1 GB | out of memory | out of memory |
| 262,144 | 1.6 s, 3.0 GB | 5.8 s, 9.6 GB | out of memory |

**Consequences:**
- **450k fine-tuning:** published-style full fine-tuning still does not fit on this GPU, even with chunking.
- **Where training fits:**
  - Ref+alt in one graph fits only at ≤131k.
  - One sequence fits at 262k, so 262k fine-tuning would need ref and alt backpropagated separately.
- **Time at 131k:** one epoch of one matched/chrom fold (~3,645 pairs at 6.1 s) takes ~6 h, so a 5-fold sweep is not practical locally.
- **Frozen embeddings fit at every length.** Unpadded, in both orientations, the matched set runs at ~1.9 s per pair (~3 h for 5,810 pairs).

### 11.4 In the published orientation, a causal model barely sees the variant

HyenaDNA is causal: a position's hidden state depends only on earlier positions, so the allele can change only the states downstream of the variant in reading order. `parse_eQTL` orients each span so that it usually *ends* at the variant.

| Where the variant sits in reading order | All pairs | Matched |
|---|---|---|
| Only the 500 bp flank comes after it | 86.4% | 71.2% |
| Variant in the first half of the real sequence | 10.8% | 20.5% |
| Share of the 450k padded input after it (median) | 80.1% | 96.0% |

- **Unpadded,** the ref and alt mean-pooled embeddings of most pairs can differ only through the last ~500 positions, about 2% of a 25 kb span.
- **With the published right-padding,** most positions come after the variant, but they are N tokens. In the released code they are also emptied or mis-tokenised (11.2).
- **Bidirectional models (Caduceus) do not have this problem.**
- **For the unmatched set this interacts with labels:** positives have the variant in the first half 25% of the time vs 9% for negatives. So the orientation itself differs by class.

**Design of our frozen-embedding pass** (`analysis/hyena_embed.py`, `analysis/hyena_probe.py`): embed each pair unpadded in the published orientation (`fwd`), and in its reverse complement (`rev`) so the variant comes first. Report the ablation for both.

### 11.5 Frozen embeddings + linear probe: HyenaDNA does not use the allele

**Run:** 2026-09-18. All 5,810 matched pairs, unpadded, in both reading orders (`data/eQTL_v2/hyena_emb/matched_none/embeddings.npz`). Logistic regression (C = 0.1, standardised) on mean-pooled last-layer states, matched/chrom 5-fold. Per-fold results in `probe_chrom.tsv` next to the embeddings.

| Test AUROC, matched/chrom | `refalt` | `ref_copy` | `refalt` − `ref_copy` | `ref_only` | `diff_only` |
|---|---|---|---|---|---|
| `fwd` (published orientation) | 0.551 ± .029 | 0.551 ± .029 | −0.0001 ± .0001 | 0.553 ± .028 | 0.502 ± .011 |
| `rev` (variant first) | 0.556 ± .033 | 0.556 ± .033 | +0.0001 ± .0001 | 0.561 ± .033 | 0.497 ± .019 |
| `fwd+rev` | 0.556 ± .031 | 0.556 ± .030 | −0.0001 ± .0002 | 0.560 ± .031 | 0.495 ± .021 |
| CPU floor (dist + comp + allele, 10.2) | | | | | 0.599 ± .031 |

Per-fold `refalt`, `fwd`: 0.571, 0.530, 0.570, 0.574, 0.511. `rev`: 0.577, 0.511, 0.598, 0.552, 0.542.

**How far the SNP moves the pooled embedding** (‖e_alt − e_ref‖ / ‖e_ref‖): median 1.5e-4 (`fwd`) and 2.0e-4 (`rev`), 90th percentile ≤ 6e-4. It is never exactly zero.

**Findings:**
1. **The allele contributes nothing.** Swapping alt for a copy of ref at test time changes AUROC by at most 0.0003 on any fold, in either orientation. This is the same null as the CNN (10.4, Δ −0.0002).
2. **The allele signal isn't there to find.** A probe trained only on e_alt − e_ref scores 0.50 in every orientation. So the null is not a probe that ignored a usable feature.
3. **Reading order doesn't rescue it.** Putting the variant first (`rev`), so every downstream position can see it, moves the SNP's effect on the pooled embedding only from 1.5e-4 to 2.0e-4, and AUROC is unchanged. The orientation problem in 11.4 is real, but it isn't why HyenaDNA fails here.
4. **What HyenaDNA does learn is context, and less of it than hand features.** `ref_only` ≈ `refalt` ≈ 0.55–0.56, below the CPU floor of 0.599 (GC, CpG and repeat content in fixed windows).
5. **The CNN and HyenaDNA now fail the same way on the same split.** Neither reads the variant, and both sit at or below a gradient-boosted baseline on local sequence composition.

**Caveats:**
- **This is frozen HyenaDNA with a linear head, not fine-tuned.** Fine-tuning could in principle amplify a 1e-4 shift, but it has to start from a representation where the SNP barely registers.
- **Numerical precision: checked, not the cause** (`analysis/hyena_fp32_check.py`, 2026-09-18). 400 random matched pairs of ≤131 kb were re-embedded in full fp32 (no autocast, TF32 off), in both orientations, and compared with their bf16 embeddings.

| | `fwd` | `rev` |
|---|---|---|
| bf16 rounding error, ‖ref_bf16 − ref_fp32‖ / ‖ref_fp32‖ (median) | 2.6e-3 | 2.6e-3 |
| SNP shift in fp32, ‖alt − ref‖ / ‖ref‖ (median) | 1.7e-4 | 1.9e-4 |
| SNP shift in bf16 (median) | 2.0e-4 | 2.2e-4 |
| cosine(bf16 difference, fp32 difference): median, 10th pct | 0.89, 0.58 | 0.90, 0.68 |
| `diff_only` AUROC on these 400 pairs: bf16 | 0.509 ± .035 | 0.490 ± .078 |
| `diff_only` AUROC on these 400 pairs: fp32 | 0.496 ± .042 | 0.499 ± .050 |

  - **bf16 rounding is 15× larger than the SNP's effect, yet it cancels in the difference.** Ref and alt share every computation before the variant, so they share the same rounding, and the bf16 alt − ref difference points the same way as the fp32 one (median cosine ≈ 0.9).
  - **The SNP's effect is just as small in fp32** (~1.7e-4), and **`diff_only` stays at chance in fp32.** The null in 11.5 is a property of the model, not of reduced precision.
  - **Limits:** the check covers 87% of matched pairs (≤131 kb), and 400 pairs gives a noisy probe (fold SD 0.04–0.05), but it is centred on 0.50 in both precisions.

## 12. Step 3: Caduceus-Ph (frozen embeddings + linear probe)

### 12.1 Setup

- **Environment:** WSL2 Ubuntu 24.04, Python 3.10, pinned to the repo's `caduceus_env.yml`: torch 2.2.0 (cu121), mamba-ssm 1.2.0.post1, causal-conv1d 1.2.0.post2. The two CUDA packages come from their GitHub release wheels, because their `setup.py` needs `nvcc` even just to pick a prebuilt wheel. Rebuild with `analysis/caduceus_env_setup.sh`.
- **Weights:** `kuleshov-group/caduceus-ph_seqlen-131k_d_model-256_n_layer-16` (bidirectional, not RC-equivariant, pretrained at 131 kb). Stored in `data/models/`.
- **Memory:** at 450 kb, bf16 inference peaks at 4.4 GB (2.5 s per sequence). fp32 runs out of memory under the 7.2 GB cap, because Mamba's fused scan keeps its intermediates even at inference. The main pass is bf16; 12.3 checks fp32.
- **Script:** `analysis/caduceus_embed.py`. It mirrors `hyena_embed.py` (unpadded, `fwd` and `rev`, mean-pooled last layer, the same char vocabulary A/C/G/T/N → 7–11), and `hyena_probe.py` reads its output unchanged.

**The released Caduceus eQTL code has the same defects as HyenaDNA's (11.2):**
- `src/dataloaders/datasets/eqtl_dataset.py:104–107` slices each padded 450 kb one-hot with `[2028500:-2028500]` (empty), then takes `np.argmax`, which gives ids 0–3. The pretrained vocabulary puts A/C/G/T at 7–10, and 0–3 are special tokens.
- `configs/model/caduceus.yaml` sets `n_layer: 2`, while the only public Ph checkpoint at this width has 16 layers. `pretrained_model_path` is left as `???` in `configs/experiment/hg38/eqtl_benchmark.yaml`, so which weights produced Table 7 is unstated.
- The head (`CaduceusEQTL`, `caduceus/modeling_caduceus.py:678`) matches the Methods: mean-pool ref and alt, concatenate, then an MLP. Our linear probe on the same pooled features is the frozen version of it.

### 12.2 Result: Caduceus does not use the allele either

All 5,810 matched pairs, matched/chrom 5-fold, logistic C = 0.1 (`data/eQTL_v2/caduceus_emb/matched_none/probe_chrom.tsv`).

| Test AUROC, matched/chrom | `refalt` | `ref_copy` | `refalt` − `ref_copy` | `ref_only` | `diff_only` |
|---|---|---|---|---|---|
| `fwd` | 0.563 ± .015 | 0.563 ± .015 | −0.0001 ± .0002 | 0.565 ± .011 | 0.498 ± .017 |
| `rev` | 0.554 ± .011 | 0.554 ± .011 | −0.0001 ± .0001 | 0.560 ± .011 | 0.503 ± .009 |
| `fwd+rev` | 0.564 ± .021 | 0.564 ± .021 | 0.0000 ± .0002 | 0.567 ± .019 | 0.507 ± .016 |
| HyenaDNA `fwd+rev` (11.5) | 0.556 ± .031 | 0.556 ± .030 | −0.0001 ± .0002 | 0.560 ± .031 | 0.495 ± .021 |
| CPU floor (dist + comp + allele, 10.2) | | | | | 0.599 ± .031 |

Per-fold `refalt`, `fwd`: 0.554, 0.568, 0.554, 0.586, 0.551.

**How far the SNP moves the pooled embedding:** median 6.8e-5 in both orientations, 90th percentile 2.4e-4. That is about half of HyenaDNA's shift, even though every Caduceus position can see the variant.

**Findings:**
1. **No allele use.** Swapping alt for a copy of ref changes AUROC by at most 0.0004 on any fold. `diff_only` is at chance (0.50–0.51).
2. **Bidirectionality doesn't help.** The orientation problem that limits causal HyenaDNA (11.4) doesn't apply to Caduceus, yet the result is the same. The null isn't an artefact of reading order.
3. **Context only, and below the hand-feature floor.** `ref_only` ≈ `refalt` ≈ 0.55–0.57, below 0.599.
4. **All three architectures fail the same way.** The CNN (10.4), HyenaDNA (11.5) and Caduceus score the same with the variant as with a copy of the reference, on the same split.

### 12.3 Precision check (fp32)

The same 400 pairs of ≤131 kb as 11.5, re-embedded in full fp32 (`caduceus_embed.py --fp32-check`, compared with `hyena_fp32_check.py`).

| | `fwd` | `rev` |
|---|---|---|
| bf16 rounding error (median) | 4.5e-4 | 4.4e-4 |
| SNP shift, fp32 (median) | 7.8e-5 | 7.3e-5 |
| SNP shift, bf16 (median) | 8.9e-5 | 8.1e-5 |
| cosine(bf16 difference, fp32 difference): median, 10th pct | 0.95, 0.65 | 0.94, 0.63 |
| `diff_only` on these 400 pairs: bf16 / fp32 | 0.500 / 0.514 | 0.552 / 0.543 (± .06) |

- **bf16 keeps the SNP's direction** (cosine ≈ 0.95), and the SNP's effect is just as small in fp32.
- **`diff_only` agrees between precisions on every subset.** The `rev` subset's 0.55 ± 0.06 is the same in bf16 and fp32, so it is subset sampling noise (~80 test pairs per fold), not a precision effect. On all 5,810 pairs `rev` `diff_only` is 0.503 ± 0.009.

## 13. Clinical evaluation set (eval-only)

### 13.1 Construction

**Script:** `analysis/build_clinical_eval.py` → `data/clinical/pairs.tsv`.

- **Positives:** the 1★ regulatory core (8.1), SNVs only: 696 of 1,169. The other 473 are small indels, not yet handled by the embedding pipeline. 652 of the SNVs map to a GENCODE gene TSS (via ClinVar's gene symbol) within a 450 kb span.
- **Negative pool:** benign or likely benign SNVs, ≥1★, re-derived from the VCF, variant_summary and GENCODE with the same filters. 447,012 are non-coding or unannotated; 112,775 pass the core filters; 107,703 map to a TSS.
- **Matching (1:1):** same gene and same location class (555 pairs), then same gene (77), then same location class (20), each time choosing the closest |log distance to the TSS|.
  - Overall median distance is 26.7 kb (pathogenic) vs 26.6 kb (benign).
  - **The residual imbalance is in the promoter stratum:** the pathogenic median is 170 bp vs 450 bp benign, and the pathogenic variant is closer in 67% of pairs.
- **Inputs:** the same sequence construction as eQTLP (`build_strings`). Both models are embedded unpadded, `fwd` and `rev`, bf16.
- **Composition:** 301 genes. The largest are NF1 (38 pairs), HBB (32) and ATM (18). 285 pairs have a 2★+ pathogenic.

### 13.2 Scores (nothing fitted on clinical labels)

**Script:** `analysis/clinical_score.py` → `data/clinical/clinical_scores.tsv`.

- **Scores:**
  - `shift`: the relative embedding shift.
  - `eqtl_probe`: the logistic probe fitted on all eQTLP-v2 matched pairs (fwd+rev), applied to the clinical pairs.
  - `eqtl_allele`: the probe's score for the true alt minus its score for a ref copy.
- **Reporting:** AUROC with a 95% CI from resampling matched pairs, plus the paired win rate.

| AUROC [95% CI] | All (652) | 2★+ (285) | Same gene + location (555) | 5′UTR/promoter (146) | Deep intron (399) |
|---|---|---|---|---|---|
| −distance to TSS | 0.510 [.505, .516] | 0.518 [.511, .527] | 0.503 [.498, .508] | **0.612** [.575, .653] | 0.501 [.497, .505] |
| HyenaDNA shift, `rev` | 0.506 [.497, .516] | 0.500 [.485, .513] | 0.504 [.495, .513] | 0.565 [.506, .620] | 0.505 [.492, .517] |
| HyenaDNA eQTL probe | 0.503 [.496, .511] | 0.502 [.491, .514] | 0.495 [.490, .499] | 0.526 [.499, .554] | 0.496 [.490, .503] |
| HyenaDNA allele component | 0.502 [.469, .532] | 0.523 [.478, .568] | 0.494 [.462, .529] | 0.543 [.475, .614] | 0.483 [.445, .521] |
| Caduceus shift, `rev` | 0.500 [.488, .512] | 0.504 [.486, .521] | 0.498 [.484, .512] | 0.554 [.503, .601] | 0.490 [.469, .508] |
| Caduceus eQTL probe | 0.507 [.500, .514] | 0.509 [.499, .520] | 0.499 [.496, .502] | 0.539 [.515, .565] | 0.501 [.495, .507] |
| Caduceus allele component | 0.489 [.462, .519] | 0.504 [.459, .548] | 0.485 [.453, .516] | 0.460 [.399, .522] | 0.491 [.452, .532] |

`fwd` results are in the TSV and are within ±0.02 of `rev`. The small strata (3′UTR 34, genomic-only 31, intron-mid 42) are in the TSV.

### 13.3 Findings

1. **No model separates pathogenic from matched benign regulatory SNVs.** Every model score has a CI covering 0.5 or lies within ±0.01 of it, in every well-populated stratum. That holds for the zero-shot shift, the transferred eQTL probe, and its allele component.
2. **The only signal is residual distance, in the promoter stratum.** Distance alone scores 0.61 there. The models' 0.53–0.57 in that stratum is below what distance gives and is consistent with the same leak. It is not evidence of allele reading.
3. **Small-stratum outliers are expected by chance.** Caduceus's allele component is 0.65 [0.52, 0.78] on 31 genomic-only pairs, while its shift is 0.37 on the same pairs. That is one of about 50 stratum × score tests, so it is not significant after correction.
4. **Why the CIs are narrow for the probe and shift scores:** most pairs share a gene, and hence nearly the same sequence context, so the two members of a pair get almost the same score. The CI is narrow because the score cannot separate within a pair, not because the estimate is precise. The paired win rates (0.47–0.52) show the same thing.

**Takeaway:** the eQTL result carries over to clinical labels. Frozen long-context embeddings don't register which base is present, whether the labels are GTEx fine-mapping or ClinVar pathogenicity.

## 14. Robustness of the paper's claims

Each claim was re-tested for a single-run artefact. Results are in `analysis/results/robustness/`. The GPU checks (R6–R8) run from `analysis/run_robustness_gpu.ps1`.

| # | Claim | Check | Result | Status |
|---|---|---|---|---|
| R1 | Distance alone beats all published models (0.751) | Independent recount from the released tables (`robust_published.py`) | 0.752 with distance from coordinates; 0.750 with parse_eQTL's 450 kb cutoff. **The released `distance_to_tss` column gives 0.663** (see note) | Holds, with a precise definition of distance |
| R2 | Tiny tests, unstratified split, gene overlap | Same recount | 11–34 test positives; positive rate 3–5% in test vs 9–21% in train; 100% of test genes in train; exact variant reuse 0–13% | Holds |
| R3 | eQTLP-v2 floor ≈ 0.60 | 7 matchings (seeds 0–4, calipers 0.02/0.05/0.1) × 10 random chromosome partitions, varying the GBT seed (`robust_eqtl_v2.py --part R3`) | dist+comp+allele 0.594 ± 0.006 (0.582–0.606); composition 0.596 ± 0.004; distance 0.493 ± 0.004 | Holds |
| R4 | FM `refalt` = `ref_copy`; FMs below the floor | C ∈ {0.001–10}; fixed chrom, gene, random and 10 random chrom partitions; paired bootstrap Δ CI (`--part R4`) | max \|Δ\| 0.0006 over every setting; Δ 95% CI [−0.0002, 0.0002]; both FMs below the floor in 10/10 random partitions (HyenaDNA 0.563, Caduceus 0.582 vs 0.595) | Holds |
| R4 | (new) FMs beat the floor only on the random split | Same | Random split: HyenaDNA 0.687, Caduceus 0.696 vs floor 0.670. Chrom/gene: below | New supporting finding |
| R4 | `diff_only` at chance | Nonlinear probes (GBT, MLP) on e_alt − e_ref and \|e_alt − e_ref\| | **Linear 0.49–0.51, but nonlinear 0.519–0.530 (HyenaDNA) and 0.511–0.547 (Caduceus).** Allele identity alone (base, Ts/Tv, CpG; GBT) scores 0.49–0.51 | **Claim must be narrowed** (R4b, R6) |
| R5 | Released FM loaders yield empty inputs with wrong token ids | Ran each repo's own `EQTLseqDataSet.__iter__` on real 450 kb records (`check_released_loaders.py`, WSL) | Both repos yield shape `(0,)` for ref and alt; without the slice the ids are {0,1,2,3}, and N padding → 0 (= A) | Holds; now demonstrated, not argued |
| R9 | Clinical null | Drop NF1/HBB/ATM; drop the top 10 genes; one pair per gene | Every score 0.47–0.51 in every subset | Holds |

**Note on R1:** the released `distance_to_tss` column agrees with the coordinates within 1 bp for 86% of rows. It differs by more than 1 kb for 13%, **all of them negatives** (16% of negatives). For those rows it apparently refers to another gene's TSS. The input the model sees encodes the coordinate distance (the variant→TSS span and its N padding), so the paper defines distance from the coordinates and reports the column discrepancy as a further data issue. With the column, distance alone scores 0.663: still above every FM (≤ 0.566), but below the expert (0.681).

| R4b | The allele isn't used by a nonlinear head either | MLP (256-64) on [e_ref, e_alt] with the `ref_copy` swap; MLP on e_alt − e_ref with a 20-permutation label null, chrom and gene folds (`robust_nonlinear.py`) | **MLP head Δ = 0.0000** in all 4 settings (HyenaDNA 0.546/0.572, Caduceus 0.574/0.598 on chrom/gene). **MLP on the difference is above the permutation null:** Caduceus 0.547 chrom and 0.538 gene; HyenaDNA 0.530 gene (all p = 0.048, the minimum for 20 permutations; null max ≤ 0.518); HyenaDNA 0.522 chrom (p = 0.095) | **Narrowed claim:** pooled embeddings carry a weak, nonlinearly decodable allele trace (0.52–0.55), but a classifier on the benchmark's [e_ref, e_alt] input, linear or MLP, gives it zero weight |

| R7 | The published CNN recipe collapses | 10 init seeds, each trained one full epoch, on the **published** split of two tissues (`cnn_dead_head_check.py --published-tissue`) | **2/10 seeds dead at initialisation** in both tissues (gradient norm exactly 0). After one epoch, **5/10 (Whole_Blood) and 6/10 (Nerve_Tibial) seeds give a constant prediction** (loss exactly 0.6931, eval logit SD 0, AUROC exactly 0.5). Logs: `R7_collapse_published_*.log` | Holds, and now on the published split rather than v2 |
| R8 | The CNN's `refalt` − `ref_copy` ≈ 0 | 3 seeds × 5 chrom folds, test-time ablation on each trained model (`R8_cnn_seeds.tsv`) | max \|Δ\| 0.0017 over 15 runs, mean −0.0002; random-allele swap likewise. Mean test 0.517 ± 0.040, below the floor. Seed-to-seed SD within a fold 0.03–0.06 | Holds; also confirms sweep-1's ±0.1 gaps were training noise |

| R6 | Does the allele register anywhere, if we stop averaging? | Every layer's residual stream, pooled 4 ways (whole sequence / ±1 kb / ±64 bp / the variant's own position), for ref, alt and **alt_rand** (a base change that did not occur), 5,810 matched pairs, chrom folds (`variant_window_embed.py`, `variant_window_probe.py`) | **HyenaDNA (rev, 9 layers):** the R4b trace is *not* allele-specific. Nonlinear diff probe 0.518–0.556 across layers/poolings, but the **alt_rand control matches or beats it** (mean `diff_gbt − rand_gbt` = −0.006). Swapping the allele still changes nothing: max \|Δ\| 0.013, and at the variant position Δ is −0.006 to −0.013, i.e. `ref_copy` scores *higher*. Local pooling only adds context: ±64 bp gives `refalt` ≤ 0.59 with `ref_only` equal or better; the best cell overall is 0.602 ≈ the floor | **Restores the strong claim:** no allele use at any layer or pooling, including at the variant's own position; R4b's 0.52–0.55 is local context |

| R6 | (Caduceus half) | Same, 17 layers × 4 poolings | Benchmark readout: Δ ≈ 0 everywhere (max \|Δ\| 0.0165, always in favour of `ref_copy`). Nonlinear diff probe vs the alt_rand control: mean +0.004 over 68 cells, true > rand in 42/68. **One cell is real:** final layer, ±1 kb → true 0.580 vs rand 0.545, Δ +0.034, bootstrap CI [+0.017, +0.051], and it replicates on the gene (+0.023) and random (+0.025) folds | **Caduceus keeps a weak allele-specific trace near the variant; the benchmark's readout discards it** |

**Shift by pooling (median ‖alt − ref‖/‖ref‖):** whole sequence 0.0001–0.0009, ±64 bp 0.007–0.067, variant position 0.37–1.12. Both models represent the substituted base locally; mean-pooling over tens of kilobases dilutes it by ~1,000×.

### 14.1 What R6 changes about the claim

1. **The benchmark's readout never uses the allele.** Across 104 layer × pooling cells and both models, swapping alt for ref changes the probe's AUROC by at most 0.017, and the sign is consistently *against* the true allele (`ref_copy` scores the same or higher). This includes pooling at the variant's own position, where the hidden state changes by 37–112%.
2. **HyenaDNA carries no allele-specific information**: its nonlinear diff probe is matched or beaten by the alt_rand control (mean −0.006, true > rand in 9/36 cells), and its best cell does not replicate across fold schemes (+0.009 chrom, +0.006 gene, −0.003 random).
3. **Caduceus carries a little, in one place.** With a variant-centred readout (final layer, ±1 kb) a gradient-boosted probe on the allele difference reaches 0.575–0.580 vs 0.545–0.553 for a base change that never happened, on all three fold schemes. This is still below the local-composition floor (0.599), and no [e_ref, e_alt] head recovers it.
4. **Consequence for the paper:** the constructive recommendation is a *variant-centred readout*, not just a bigger model. The benchmark's mean-pooled interface throws away what little allele signal the bidirectional model has.
