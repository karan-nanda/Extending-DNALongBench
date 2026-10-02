# Long-context DNA models do not use the allele: an audit of DNALongBench's eQTL task

***Superseded by `paper/main.tex`, which is now the source of truth.** This file is kept as the readable prose version; `main.tex` carries the trims made to fit the 4-page budget (§2 distance column, §4 trace bullet and Scope, §5 CNN collapse, §6 clinical bullets, §7 prior work, figure caption), so edit `main.tex` and back-port here, not the other way round. The full notes are in `paper/draft.md`, and the appendix is taken from there.*

## Abstract

eQTL prediction (eQTLP) is the only variant task in DNALongBench [@cheng2025dnalongbench], a standard benchmark for long-context DNA models. Ranking variants by distance to the transcription start site (TSS), with no training, reaches a mean test AUROC of 0.751, beating the expert model (0.681) and every foundation model (FM; 0.51–0.57). The test sets hold 11–34 positives per tissue, and every test gene also appears in training. We rebuild the task with distance-matched negatives and chromosome holdout, where hand-crafted local sequence features set a floor of 0.599. Scoring each pair again with the alternate allele replaced by the reference shows that no model uses the allele: for a CNN trained end to end and for frozen HyenaDNA and Caduceus, the swap changes AUROC by at most 0.0017, and all three fall below the floor. A nonlinear probe on the allele difference alone recovers only a weak trace (0.52–0.55). A single-nucleotide change moves the pooled FM embedding by about 10⁻⁴ of its norm. The released FM eQTL code also slices every input to length zero. On 652 ClinVar pathogenic regulatory SNVs, each matched to a benign SNV in the same gene and location class, neither the embedding shift nor the transferred eQTL probe separates the two classes (AUROC 0.49–0.51). We release the rebuilt task, the ablation harness, and the matched ClinVar evaluation set.

## 1 Introduction

Long-context DNA models [@nguyen2023hyenadna; @schiff2024caduceus; @brixi2026evo2] are motivated by distal regulation, meaning variants that act on genes tens to hundreds of kilobases away. They inherit their architectures from long-convolution and state-space sequence models [@poli2023hyena; @gu2023mamba], and are benchmarked against expert models such as Enformer [@avsec2021enformer], Akita [@fudenberg2020akita] and Puffin [@dudnyk2024puffin]. DNALongBench [@cheng2025dnalongbench] evaluates HyenaDNA [@nguyen2023hyenadna] and Caduceus [@schiff2024caduceus] at up to 450 kb. Of its five tasks, only eQTLP involves a variant. Its model interface embeds the reference and alternate sequences, averages each, concatenates them, and classifies. A model can therefore succeed without distinguishing the alleles at all, if *where* the variant sits predicts the label.

We show that this is what happens. The published split rewards distance and gene identity. Once both are removed, three architectures score the same with and without the alternate allele. We contribute:

1. an audit of the published protocol;
2. **eQTLP-v2**, a distance-matched, chromosome-held-out version with a published floor;
3. a **ref-copy ablation** that any variant benchmark can report;
4. reproducibility findings on the released code;
5. a matched, evaluation-only ClinVar regulatory set.

## 2 The published protocol rewards distance and memorisation

Positives are SuSiE [@wang2020susie] fine-mapped GTEx [@gtex2020] eQTLs in 9 tissues. Negatives are other variant–gene pairs within 450 kb. Each input spans the variant and the TSS and is right-padded with N to 450 kb. The split is a per-tissue random 8:1:1.

- **Distance beats every model.** Positives lie a median 8–33 kb from the TSS, negatives 88–145 kb. Scoring by −distance (|variant − TSS| from the released coordinates, which is what the padded input encodes) gives a test AUROC of 0.688–0.828 per tissue, with a mean of 0.751 (Table 1). That beats the expert model in 7 of 9 tissues and every FM in every tissue. The padding also hands the distance to the model, since the number of N tokens encodes it. The released `distance_to_tss` column disagrees with the coordinates by more than 1 kb for 16% of negatives and no positives. With that column, distance scores 0.663: still above every FM, but not above the expert.
- **The test sets cannot rank models.** They hold 11–34 positives per tissue, which gives AUROC CI half-widths of ±0.10–0.18 [@hanley1982auc]. No published FM result is distinguishable from 0.5. The split is also not stratified: positive rates are 9–21% in train and 3–5% in test.
- **Train and test overlap.** Every test gene appears in training, and 82–92% of test pairs have a training variant within 100 kb.

**Table 1.** Published test AUROC (Table 7 of the benchmark paper), mean over 9 tissues, next to our zero-training baseline.

| Distance only | Expert (Enformer) | CNN | HyenaDNA | Caduceus-Ph | Caduceus-PS |
|---|---|---|---|---|---|
| **0.751** | 0.681 | 0.528 | 0.514 | 0.566 | 0.538 |

Per-tissue values with CIs are in Appendix A.

## 3 eQTLP-v2

- **Pairs:** we pool the tissues into 25,600 unique pairs and drop 111 pairs with conflicting labels.
- **Distance matching:** each positive is matched to one negative on log distance to the TSS, on the same side of the TSS. This gives 2,905 pairs, and the distance-only AUROC falls to 0.495.
- **Folds:** 5-fold, with whole chromosomes held out, giving 570–665 test positives per fold.
- **No label mask:** we drop the published label-derived mask, which covers the positions of the positive eQTLs.

On this configuration, gradient-boosted trees on local composition (GC, CpG, repeats and dinucleotides in windows up to 5 kb) score **0.599 ± 0.031**. Adding allele features changes this by at most 0.01. On the random split, a gene's training-set positive rate alone scores 0.70. That is the memorisation the published split allows.

## 4 No model uses the allele

**Protocol.** Each model embeds the reference and alternate sequences. The last-layer states are mean-pooled, and [e_ref, e_alt] is classified, as in the benchmark's Methods. We report three quantities:
- **`refalt`:** the test AUROC.
- **`ref_copy`:** the same fitted classifier, scoring [e_ref, e_ref].
- **`diff_only`:** a classifier trained only on e_alt − e_ref.

The models:
- **CNN:** the benchmark's encoder, trained end to end, with its collapsing logit ReLU repaired (§5).
- **HyenaDNA-medium-450k** [@nguyen2023hyenadna] (causal) and **Caduceus-Ph-131k** [@schiff2024caduceus] (bidirectional): frozen, with a logistic-regression probe [@pedregosa2011sklearn].

Inputs are unpadded. For each FM we embed both the published orientation and its reverse complement.

**Table 2.** eQTLP-v2, matched pairs, chromosome holdout. Mean per-fold test AUROC ± SD.

| Model | `refalt` | `ref_copy` | Δ | `diff_only` |
|---|---|---|---|---|
| CNN (trained, 3 seeds × 5 folds) | 0.517 ± .040 | 0.517 | −0.0002 | — |
| HyenaDNA (frozen) | 0.556 ± .031 | 0.556 ± .030 | −0.0001 | 0.495 ± .021 |
| Caduceus-Ph (frozen) | 0.564 ± .021 | 0.564 ± .021 | 0.0000 | 0.507 ± .016 |
| Local-composition floor | 0.599 ± .031 | | | |

- **The allele changes nothing** (Figure 1A). Replacing it with the reference changes AUROC by at most 0.0017 for the trained CNN (15 runs) and 0.0006 for the FM probes, across probe strengths (C = 0.001–10), heads (linear, MLP), orientations, and 13 fold partitions (chromosome, gene and random). The paired 95% CI is [−0.0002, 0.0002].
- **Only a weak trace of the allele is there to find.** A linear `diff_only` probe scores at chance (0.49–0.51). An MLP on the difference beats a label-permutation null (Caduceus 0.54–0.55, HyenaDNA 0.52–0.53), but that is below the floor, and a classifier on [e_ref, e_alt] gives it no weight: an MLP head, like the released Caduceus head, also has Δ = 0.0000. Embedding every layer with four readouts (whole sequence, ±1 kb, ±64 bp, the variant position) and a control in which the variant is replaced by a base that did **not** occur shows what that trace is: for HyenaDNA the control matches it (mean −0.006 over 36 cells), so it is local context; for Caduceus one readout survives the control, final layer ±1 kb, at 0.575–0.580 vs 0.545–0.553, replicated on all three fold schemes. Even there, swapping the allele in the [e_ref, e_alt] head changes nothing (max |Δ| 0.017 over 104 cells, always favouring `ref_copy`), and the variant-position hidden state changes by 37–112% without moving the score.
- **All three models score below the hand-feature floor** (Figure 1A), in 10 of 10 random chromosome partitions. Only on the random split, which allows locus memorisation, do the FMs pass it (0.69–0.70 vs 0.67).

**The SNP barely registers** (Figure 1B). The median relative shift ‖e_alt − e_ref‖/‖e_ref‖ is 1.5–2.0 × 10⁻⁴ for HyenaDNA and 0.7 × 10⁻⁴ for Caduceus. The CNN's max-pooled embedding changes in only 0.4–1.7% of pairs. Two explanations for this fail:
- **Orientation:** HyenaDNA is causal, and in the published orientation most variants sit in the last 500 bp. But reversing the input leaves AUROC unchanged, and bidirectional Caduceus behaves identically.
- **Precision:** re-embedding 400 pairs in fp32 gives the same shift (bf16–fp32 difference-vector cosine 0.89–0.95), and `diff_only` stays at chance.
What remains is pooling. The SNP moves the variant-position hidden state by 37–112%, the ±64 bp mean by 0.7–6.1%, and the whole-sequence mean by 0.006–0.09%: mean-pooling over tens of kilobases dilutes the substitution ~1,000×.

![Figure 1](figures/figure1.pdf)

**Figure 1.** **(A)** eQTLP-v2, matched pairs, chromosome holdout: per-fold test AUROC for each model, scored normally (`refalt`, filled) and with the reference copied over the alternate allele (`ref_copy`, open). Each fold's pair is joined by a line, and every line is vertical: the swap moves no score. Points are 5 folds (5 folds × 3 seeds for the CNN), and the short bar is the mean. The dashed line is the local-composition floor fitted on these same folds (0.593); §3 quotes 0.599 ± 0.031, its mean over 7 matchings × 10 chromosome partitions. No model reaches it. **(B)** Relative shift ‖e_alt − e_ref‖ / ‖e_ref‖ of the frozen FM representation at four readouts, from the variant position out to a mean over the whole input. Markers are the median over layers, bands the min–max. Mean-pooling dilutes a one-base substitution by about 1,000×.

**Scope.** The FMs are frozen, whereas the published FM numbers come from fine-tuning. However, the end-to-end CNN shows the same null. And fine-tuning would have to amplify a 10⁻⁴ input-dependent shift that pretraining on a single reference genome never rewarded. [TODO: optionally, a variant-centred pooling probe, or one short fine-tuning run.]

## 5 The released code does not reproduce the published numbers

- **The FM eQTL loaders produce empty inputs.** Both the HyenaDNA and Caduceus loaders pad each input to 450,000 positions and then slice it with `[2028500:-2028500]`, which leaves an empty array.
- **The token ids are wrong.** Without the slice, `argmax` over the one-hot gives ids 0–3, which are special tokens in the pretrained vocabularies.
- **The configs don't match the checkpoints.** They specify 2 layers, while the checkpoints have 8 and 16, and the pretrained-weights path is unset.
- **The CNN's training often collapses.** Its eQTLP head ends in a ReLU on the logits, so both logits can clamp to zero: the prediction is constant and every weight gets zero gradient. On the **published** split, 2 of 10 initialisations are dead before the first step, and after one epoch 5 of 10 (Whole_Blood) and 6 of 10 (Nerve_Tibial) seeds output a constant score, i.e. AUROC exactly 0.5. The training script sets no seed and evaluates the final-epoch model rather than the best checkpoint. The CNN is also fed only the reference sequence.

## 6 Clinical evaluation set

- **Pathogenic set:** ClinVar [@landrum2025clinvar] pathogenic or likely-pathogenic regulatory SNVs with at least 1 star, annotated against GENCODE [@gencode2025]. They must be non-coding, more than 50 bp from a splice site, off every CDS, and outside small structural RNAs. 652 of them map to a gene TSS within 450 kb.
- **Benign set:** each pathogenic variant is matched to a benign SNV that passes the same filters, preferring the same gene and location class (555 of 652 pairs), with matched distance to the TSS.
- **Evaluation only:** no model is fitted on clinical labels. We score the embedding shift, the transferred eQTLP-v2 probe, and that probe's allele-only component.
- **Results:** all three are at chance (AUROC 0.489–0.507 overall, 0.49–0.52 on the 285 pairs with a 2★+ pathogenic).
- **Where signal does appear, it's distance.** The only stratum above chance is 5′UTR/promoter (0.53–0.57). There, matching left pathogenic variants closer to the TSS (median 170 bp vs 450 bp), and distance alone scores 0.61.
- **Takeaway:** the eQTL null carries over from fine-mapping labels to pathogenicity labels. The set is evaluation-only because, after the splice, CDS and structural-RNA filters, ClinVar leaves too few regulatory positives to train on (409 at 2★).

## 6.1 Relation to prior work

Two long-range DNA benchmarks are adjacent to ours. The Genomics Long-Range Benchmark [@kao2024lrb] includes an OMIM pathogenic non-coding variant task, and BEND [@marin2024bend] evaluates element annotation on reference sequence. Neither reports an allele ablation, which is the diagnostic we argue for: without it, a variant task can be passed by locating the variant rather than reading it. Our result also extends, to a benchmark setting, the finding that sequence-to-expression models explain individual expression variation poorly and often mispredict the direction of cis-regulatory variant effects [@sasse2023benchmarking; @huang2023personal]. The failure mode is shortcut learning [@geirhos2020shortcut]: the protocol, not the architecture, decides what a model must learn.

## 7 Recommendations

Variant benchmarks for long-context models should:
- match negatives on distance to the TSS (and on gene where possible);
- hold out chromosomes;
- size test sets for ±0.05 CIs;
- publish a hand-feature floor;
- report a **ref-copy ablation** next to every variant score;
- read the variant out **locally** (a window around the variant), not as a mean over the whole input: Caduceus's only allele-specific signal lives in a ±1 kb window and is lost to mean-pooling.

On DNALongBench's eQTL task, the current models learn where a variant is, not what it is.

## Appendix (from `paper/draft.md`)

- A: per-tissue Table 1 with CIs; test-set sizes; overlap statistics.
- B: eQTLP-v2 construction and the full baseline grid (random, gene and chromosome holdout; all vs matched).
- C: per-fold ablation results; results by orientation; fp32 check.
- D: the CNN collapse analysis.
- E: the released-code defects, with file and line references.
- F: clinical-set construction and results by stratum.


## References

All keys resolve in `paper/references.bib` (verified 2026-09-20).
