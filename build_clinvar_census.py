#!/usr/bin/env python3
"""
Week 1-2 gate: ClinVar census for a DNALongBench extension.

Answers four questions before any modelling work starts:
  Q1  How many pathogenic non-coding variants carry LONG-RANGE signal?     (gate: >= 1500)
      i.e. not coding, not within --splice-window bp of a splice site.
  Q2  How many pathogenic indels / SVs exist, by type and span?            (gap G2)
  Q3  How many of those SVs miss every coding exon?                        (gap G1)
  Q4  How concentrated are the Q1 positives by gene?                       (split-design risk)

Usage
-----
  # one-time downloads
  curl -LO https://ftp.ncbi.nlm.nih.gov/pub/clinvar/vcf_GRCh38/clinvar.vcf.gz
  curl -LO https://ftp.ncbi.nlm.nih.gov/pub/clinvar/tab_delimited/variant_summary.txt.gz
  curl -LO https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/latest_release/gencode.v50.annotation.gtf.gz

  python build_clinvar_census.py --vcf clinvar.vcf.gz \
      --summary variant_summary.txt.gz --gtf gencode.v50.annotation.gtf.gz --outdir census/

  --summary and --gtf are optional but without them Q1 over-counts badly: ClinVar
  labels c.123+5G>A an intron_variant, so near-splice variants pass the MC filter.
  --summary supplies the c./n. HGVS name that carries the intron offset, plus the
  CNVs the VCF omits. --gtf supplies exon/CDS coordinates for the SV overlap check.

Outputs
-------
  census/clinvar_filtered.tsv      one row per non-coding / unannotated small variant
  census/clinvar_sv.tsv            one row per pathogenic SV > 50 bp (needs --summary)
  census/report.txt                the numbers you take to the Week 2 gate meeting
"""

import argparse
import bisect
import gzip
import os
import re
from collections import Counter, defaultdict

# ---------------------------------------------------------------------------
# Filter definitions. Edit these, not the code below.
# ---------------------------------------------------------------------------

# CLNSIG is "/"-joined primary terms, then "|"-joined secondary ones
# (e.g. Pathogenic/Likely_pathogenic|risk_factor). A record is pathogenic if
# every primary term is P/LP (low penetrance included) or a risk-allele term.
PATHOGENIC_TERMS = {
    "Pathogenic",
    "Likely_pathogenic",
    "Pathogenic,_low_penetrance",
    "Likely_pathogenic,_low_penetrance",
}
RISK_TERMS = {
    "Likely_risk_allele",
    "Established_risk_allele",
    "Uncertain_risk_allele",
}
BENIGN_TERMS = {
    "Benign",
    "Likely_benign",
}

# ClinVar review status -> star rating
STARS = {
    "practice_guideline": 4,
    "reviewed_by_expert_panel": 3,
    "criteria_provided,_multiple_submitters,_no_conflicts": 2,
    "criteria_provided,_single_submitter": 1,
    "criteria_provided,_conflicting_classifications": 1,
    "criteria_provided,_conflicting_interpretations": 1,
    "no_assertion_criteria_provided": 0,
    "no_classification_provided": 0,
    "no_classifications_from_unflagged_records": 0,
    "no_classification_for_the_single_variant": 0,
    "no_interpretation_for_the_single_variant": 0,
}

# Sequence Ontology terms in ClinVar's MC field, bucketed.
NONCODING_SO = {
    "intron_variant",
    "5_prime_UTR_variant",
    "3_prime_UTR_variant",
    "upstream_transcript_variant",
    "downstream_transcript_variant",
    "genic_upstream_transcript_variant",
    "genic_downstream_transcript_variant",
    "non-coding_transcript_variant",
    "regulatory_region_variant",
    "TF_binding_site_variant",
}

# Kept separate: splice variants are non-coding by position but the signal is
# local (a few bp), not long-range. Including them silently inflates AUROC and
# makes the task solvable without any long context. Reported, not counted.
# Note ClinVar only uses these for +1/+2 and -1/-2; +5 or -12 comes through as
# intron_variant, which is what --splice-window catches.
SPLICE_SO = {
    "splice_acceptor_variant",
    "splice_donor_variant",
    "splice_site_variant",
}

CODING_SO = {
    "missense_variant",
    "synonymous_variant",
    "nonsense",
    "stop_gained",
    "stop_lost",
    "start_lost",
    "frameshift_variant",
    "inframe_deletion",
    "inframe_insertion",
    "inframe_indel",
    "initiator_codon_variant",
}

# CLNVC values that are not single-nucleotide substitutions
NON_SNV_TYPES = {
    "Deletion",
    "Duplication",
    "Insertion",
    "Indel",
    "Inversion",
    "Microsatellite",
    "Tandem_duplication",
    "Complex",
}

# HGVS-derived locations (needs --summary). ELIGIBLE counts toward the gate;
# STRICT drops deep-intronic too, since most pathogenic deep-intronic variants
# act by creating a cryptic exon -- still a splicing mechanism.
LOCATIONS = ["coding", "near_splice", "intron_mid", "deep_intron",
             "5UTR/promoter", "3UTR", "ncRNA_exonic", "genomic_only", "other"]
ELIGIBLE = {"intron_mid", "deep_intron", "5UTR/promoter", "3UTR",
            "ncRNA_exonic", "genomic_only"}
STRICT = {"5UTR/promoter", "3UTR", "ncRNA_exonic", "genomic_only"}

# Exons of these GENCODE gene types are excluded from the gate: a variant in
# RMRP, RNU4ATAC (inside a CLASP1 intron), RNU4-2 or TERC breaks the RNA
# itself -- local signal, and ClinVar often names it after the host gene.
SMALL_NCRNA_TYPES = {
    "snRNA", "snoRNA", "scaRNA", "misc_RNA", "rRNA", "rRNA_pseudogene",
    "ribozyme", "sRNA", "scRNA", "vault_RNA", "miRNA", "Mt_rRNA", "Mt_tRNA",
}
# Structural RNAs GENCODE types as lncRNA. Note RMRP *promoter* variants
# (cartilage-hair hypoplasia) fall outside the gene and stay in the gate.
SMALL_NCRNA_GENES = {"TERC"}

GTF_REGIONS = ["overlaps_CDS", "small_ncRNA", "exon_noncoding",
               "genic_nonexonic", "intergenic"]
GATE_EXCLUDED_REGIONS = {"overlaps_CDS", "small_ncRNA"}
SPAN_BANDS = ["51bp-1kb", "1-10kb", "10-100kb", "100-450kb", "450kb-1Mb", ">1Mb"]

MIN_STARS = 2
GATE_NONCODING = 1500
SPLICE_WINDOW = 50
MAX_SPAN = 450_000      # HyenaDNA medium-450k context ceiling


def parse_info(info_str):
    """ClinVar INFO field -> dict. Flags map to True."""
    out = {}
    for field in info_str.split(";"):
        if "=" in field:
            k, v = field.split("=", 1)
            out[k] = v
        else:
            out[field] = True
    return out


def primary_terms(sig):
    if not sig or sig is True:
        return []
    return sig.split("|")[0].split("/")


def is_pathogenic(sig):
    terms = primary_terms(sig)
    return (any(t in PATHOGENIC_TERMS for t in terms)
            and all(t in PATHOGENIC_TERMS or t in RISK_TERMS for t in terms))


def is_benign(sig):
    terms = primary_terms(sig)
    return bool(terms) and all(t in BENIGN_TERMS for t in terms)


def summary_norm(s):
    """variant_summary spells fields with spaces and '; ' -- map to VCF style."""
    return s.replace("; ", "|").replace(" ", "_")


def so_terms(info):
    """Extract the set of SO consequence names from the MC field."""
    mc = info.get("MC")
    if not mc or mc is True:
        return set()
    terms = set()
    for entry in mc.split(","):
        if "|" in entry:
            terms.add(entry.split("|", 1)[1])
    return terms


def gene_symbol(info):
    gi = info.get("GENEINFO")
    if not gi or gi is True:
        return "NA"
    # GENEINFO=SYMBOL:GeneID|SYMBOL2:GeneID2 -- take the first
    return gi.split("|")[0].split(":")[0]


def classify(terms):
    """Bucket a variant by its consequence terms. Coding wins over non-coding
    when both are present, since a variant annotated on any transcript as
    protein-altering is not a clean non-coding example."""
    if terms & CODING_SO:
        return "coding"
    if terms & SPLICE_SO:
        return "splice"
    if terms & NONCODING_SO:
        return "noncoding"
    if not terms:
        return "unannotated"
    return "other"


HGVS_POS = re.compile(r":([cn])\.([^A-Za-z=\[]+)")
HGVS_END = re.compile(r"^([-*]?)(\d+)([+-]\d+)?$")


def locate(name, window):
    """Location of a variant from its ClinVar HGVS name, e.g.
    NM_000267.3(NF1):c.1527+5G>A -> ('near_splice', 5).
    Returns (location, intron offset or None)."""
    if not name:
        return "other", None
    if "(p." in name and "(p.=)" not in name:
        return "coding", None
    if ":g." in name and not re.search(r":[cn]\.", name):
        return "genomic_only", None     # no transcript HGVS -> intergenic / distal
    m = HGVS_POS.search(name)
    if not m:
        return "other", None
    kind, pos = m.groups()
    if "(" in pos or "?" in pos:
        return "other", None            # uncertain breakpoints
    offs, exonic = [], []
    for ep in pos.split("_"):
        e = HGVS_END.match(ep)
        if not e:
            return "other", None
        if e.group(3):
            offs.append(abs(int(e.group(3))))
        else:
            exonic.append(e.group(1) or kind)
    if exonic:                          # at least one endpoint in an exon
        return {"-": "5UTR/promoter", "*": "3UTR",
                "n": "ncRNA_exonic"}.get(exonic[0], "coding"), None
    d = min(offs)
    if d <= window:
        return "near_splice", d
    return ("intron_mid" if d <= 100 else "deep_intron"), d


def span_band(s):
    for cut, band in ((1e3, "51bp-1kb"), (1e4, "1-10kb"), (1e5, "10-100kb"),
                      (4.5e5, "100-450kb"), (1e6, "450kb-1Mb")):
        if s <= cut:
            return band
    return ">1Mb"


def ucsc_chrom(c):
    return "chrM" if c in ("MT", "M") else (c if c.startswith("chr") else "chr" + c)


# ---------------------------------------------------------------------------
# GTF interval index
# ---------------------------------------------------------------------------

def _merge(intervals):
    intervals.sort()
    starts, ends = [], []
    for s, e in intervals:
        if ends and s <= ends[-1] + 1:
            ends[-1] = max(ends[-1], e)
        else:
            starts.append(s)
            ends.append(e)
    return starts, ends


def load_gtf(path):
    """{feature: {chrom: (starts, ends)}} for CDS, small_ncRNA, exon, gene --
    all transcripts, so a variant counts as coding if it hits a CDS in ANY
    isoform."""
    raw = {"CDS": defaultdict(list), "small_ncRNA": defaultdict(list),
           "exon": defaultdict(list), "gene": defaultdict(list)}
    gene_type = re.compile(r'gene_type "([^"]+)"')
    gene_name = re.compile(r'gene_name "([^"]+)"')
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            f = line.split("\t", 8)
            if f[2] not in raw:
                continue
            iv = (int(f[3]), int(f[4]))
            raw[f[2]][f[0]].append(iv)
            if f[2] == "exon":
                t, n = gene_type.search(f[8]), gene_name.search(f[8])
                if ((t and t.group(1) in SMALL_NCRNA_TYPES)
                        or (n and n.group(1) in SMALL_NCRNA_GENES)):
                    raw["small_ncRNA"][f[0]].append(iv)
    return {feat: {c: _merge(iv) for c, iv in d.items()}
            for feat, d in raw.items()}


def _overlaps(index, chrom, s, e):
    t = index.get(chrom)
    if not t:
        return False
    starts, ends = t
    i = bisect.bisect_right(starts, e) - 1
    return i >= 0 and ends[i] >= s


def gtf_region(gtf, chrom, s, e):
    chrom = ucsc_chrom(chrom)
    if _overlaps(gtf["CDS"], chrom, s, e):
        return "overlaps_CDS"
    if _overlaps(gtf["small_ncRNA"], chrom, s, e):
        return "small_ncRNA"
    if _overlaps(gtf["exon"], chrom, s, e):
        return "exon_noncoding"
    if _overlaps(gtf["gene"], chrom, s, e):
        return "genic_nonexonic"
    return "intergenic"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vcf", required=True, help="clinvar.vcf.gz (GRCh38)")
    ap.add_argument("--summary", help="variant_summary.txt.gz (enables Q1 splice filter + Q2 SVs)")
    ap.add_argument("--gtf", help="GENCODE annotation GTF (enables Q3 exon overlap)")
    ap.add_argument("--outdir", default="census")
    ap.add_argument("--min-stars", type=int, default=MIN_STARS)
    ap.add_argument("--splice-window", type=int, default=SPLICE_WINDOW,
                    help="intronic variants within this many bp of an exon are near-splice")
    ap.add_argument("--max-span", type=int, default=MAX_SPAN,
                    help="SVs longer than this cannot fit the model context")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    kept = {}                                # id -> row (noncoding + unannotated, small)
    counts = Counter()
    bucket_by_type = defaultdict(Counter)   # bucket -> CLNVC -> n
    benign_bucket = Counter()
    benign_ids = {}                          # id -> bucket (noncoding/unannotated benign)

    opener = gzip.open if args.vcf.endswith(".gz") else open
    with opener(args.vcf, "rt", encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            f = line.rstrip("\n").split("\t")
            if len(f) < 8:
                continue
            chrom, pos, vid, ref, alt = f[0], f[1], f[2], f[3], f[4]
            info = parse_info(f[7])

            counts["total_records"] += 1

            sig = info.get("CLNSIG", "")
            rev = info.get("CLNREVSTAT", "")
            stars = STARS.get(rev, 0)
            vc = info.get("CLNVC", "NA")
            terms = so_terms(info)
            bucket = classify(terms)

            if is_benign(sig) and stars >= args.min_stars:
                benign_bucket[bucket] += 1
                if bucket in ("noncoding", "unannotated"):
                    benign_ids[vid] = bucket

            if not is_pathogenic(sig):
                continue
            counts["pathogenic_any_star"] += 1

            if stars < args.min_stars:
                counts["pathogenic_below_star_cutoff"] += 1
                continue
            counts[f"pathogenic_{args.min_stars}star"] += 1

            bucket_by_type[bucket][vc] += 1
            counts[f"bucket_{bucket}"] += 1

            alt_len = max((len(a) for a in alt.split(",")), default=1)
            if bucket in ("noncoding", "unannotated") and max(len(ref), alt_len) <= 51:
                dn = info.get("CLNDN", "NA")
                kept[vid] = {
                    "chrom": chrom, "pos": pos, "id": vid,
                    "ref": ref, "alt": alt,
                    "clnsig": sig, "stars": stars, "clnvc": vc,
                    "gene": gene_symbol(info), "bucket": bucket,
                    "consequence": ";".join(sorted(terms)) or "NA",
                    "disease": dn if dn is not True else "NA",
                    "hgvs_name": "NA", "location": "NA",
                    "intron_offset": "", "gtf_region": "NA",
                }

    # ---- variant_summary: HGVS names + SVs ------------------------------------
    svs = {}
    benign_loc = Counter()
    if args.summary:
        opener = gzip.open if args.summary.endswith(".gz") else open
        with opener(args.summary, "rt", encoding="utf-8") as fh:
            hdr = fh.readline().lstrip("#").rstrip("\n").split("\t")
            ix = {h: i for i, h in enumerate(hdr)}
            for line in fh:
                f = line.rstrip("\n").split("\t")
                if f[ix["Assembly"]] != "GRCh38":
                    continue
                vid, name = f[ix["VariationID"]], f[ix["Name"]]
                if vid in kept:
                    loc, off = locate(name, args.splice_window)
                    kept[vid].update(hgvs_name=name, location=loc,
                                     intron_offset="" if off is None else off)
                if vid in benign_ids:
                    benign_loc[locate(name, args.splice_window)[0]] += 1
                    del benign_ids[vid]
                sig = summary_norm(f[ix["ClinicalSignificance"]])
                if not is_pathogenic(sig) or vid in svs:
                    continue
                try:
                    start, stop = int(f[ix["Start"]]), int(f[ix["Stop"]])
                except ValueError:
                    continue
                span = stop - start + 1
                if span <= 50 or start < 1:
                    continue
                svs[vid] = {
                    "id": vid, "type": f[ix["Type"]], "chrom": f[ix["Chromosome"]],
                    "start": start, "stop": stop, "span": span,
                    "stars": STARS.get(summary_norm(f[ix["ReviewStatus"]]), 0),
                    "clnsig": sig, "gene": f[ix["GeneSymbol"]],
                    "gtf_region": "NA", "name": name,
                    "phenotype": f[ix["PhenotypeList"]],
                }
        for r in kept.values():
            if r["location"] == "NA":
                r["location"] = "other"      # no GRCh38 row in variant_summary

    # ---- GTF overlap ------------------------------------------------------------
    if args.gtf:
        gtf = load_gtf(args.gtf)
        for r in kept.values():
            p = int(r["pos"])
            r["gtf_region"] = gtf_region(gtf, r["chrom"], p, p + len(r["ref"]) - 1)
        for r in svs.values():
            r["gtf_region"] = gtf_region(gtf, r["chrom"], r["start"], r["stop"])

    # ---- write variant tables ---------------------------------------------------
    tsv = os.path.join(args.outdir, "clinvar_filtered.tsv")
    cols = ["chrom", "pos", "id", "ref", "alt", "clnsig", "stars", "clnvc",
            "gene", "bucket", "consequence", "location", "intron_offset",
            "gtf_region", "hgvs_name", "disease"]
    with open(tsv, "w", encoding="utf-8") as out:
        out.write("\t".join(cols) + "\n")
        for r in kept.values():
            out.write("\t".join(str(r[c]) for c in cols) + "\n")

    sv_tsv = os.path.join(args.outdir, "clinvar_sv.tsv")
    if svs:
        sv_cols = ["chrom", "start", "stop", "span", "id", "type", "clnsig",
                   "stars", "gene", "gtf_region", "name", "phenotype"]
        with open(sv_tsv, "w", encoding="utf-8") as out:
            out.write("\t".join(sv_cols) + "\n")
            for r in sorted(svs.values(), key=lambda r: (r["chrom"], r["start"])):
                out.write("\t".join(str(r[c]) for c in sv_cols) + "\n")

    # ---- the gate set -----------------------------------------------------------
    if args.summary:
        eligible = [r for r in kept.values() if r["location"] in ELIGIBLE]
        gate = [r for r in eligible if r["gtf_region"] not in GATE_EXCLUDED_REGIONS]
    else:
        gate = [r for r in kept.values() if r["bucket"] == "noncoding"]
    n_gate = len(gate)
    genes = Counter(r["gene"] for r in gate)
    diseases = Counter(r["disease"].split("|")[0] for r in gate)

    top = genes.most_common(20)
    cum = 0
    genes_to_half = 0
    for _, c in genes.most_common():
        cum += c
        genes_to_half += 1
        if n_gate and cum >= n_gate / 2:
            break

    nc_indels = sum(v for k, v in bucket_by_type["noncoding"].items()
                    if k in NON_SNV_TYPES)
    all_indels = sum(sum(v for k, v in d.items() if k in NON_SNV_TYPES)
                     for d in bucket_by_type.values())

    # ---- report --------------------------------------------------------------
    L = []
    A = L.append
    A("=" * 72)
    A(f"ClinVar census  |  star cutoff: {args.min_stars}+  |  source: {args.vcf}")
    A("=" * 72)
    A("")
    A("FILTER CASCADE")
    A(f"  total VCF records                  {counts['total_records']:>9,}")
    A(f"  pathogenic / likely pathogenic     {counts['pathogenic_any_star']:>9,}")
    A(f"  ...dropped, below {args.min_stars} stars           "
      f"{counts['pathogenic_below_star_cutoff']:>9,}")
    A(f"  ...surviving                       "
      f"{counts[f'pathogenic_{args.min_stars}star']:>9,}")
    A("")
    A("CONSEQUENCE BUCKETS (pathogenic, star-filtered, from ClinVar MC field)")
    for b in ("coding", "splice", "noncoding", "other", "unannotated"):
        A(f"  {b:<14} {counts.get('bucket_' + b, 0):>9,}")
    A("")

    A("Q1  GATE -- long-range-eligible non-coding positives")
    if args.summary:
        A(f"      candidates: MC non-coding or unannotated, <= 50 bp   {len(kept):>7,}")
        A(f"      located by HGVS name (splice window {args.splice_window} bp):")
        loc_n = Counter(r["location"] for r in kept.values())
        for loc in LOCATIONS:
            tag = "strict" if loc in STRICT else "eligible" if loc in ELIGIBLE else "dropped"
            A(f"        {loc:<16} {loc_n[loc]:>7,}   {tag}")
        A(f"      eligible by HGVS name          {len(eligible):>9,}")
        if args.gtf:
            reg = Counter(r["gtf_region"] for r in eligible)
            A("      GENCODE cross-check of the eligible set:")
            for g in GTF_REGIONS:
                tag = "dropped" if g in GATE_EXCLUDED_REGIONS else "kept"
                A(f"        {g:<16} {reg[g]:>7,}   {tag}")
            A("      -> overlaps_CDS = coding in an isoform ClinVar did not name;")
            A("         small_ncRNA = breaks the RNA itself (RMRP, RNU4ATAC, RNU4-2...).")
        n_strict = sum(1 for r in gate if r["location"] in STRICT)
        A(f"      regulatory core (gate count)   {n_gate:>9,}   (threshold {GATE_NONCODING:,})")
        A(f"      strict core (no deep-intronic) {n_strict:>9,}")
        A(f"      {'PASS' if n_gate >= GATE_NONCODING else 'FAIL'}"
          + ("" if n_gate >= GATE_NONCODING else
             " -- widening the disease panel cannot fix this (it is all of ClinVar);"
             + ("\n      try --min-stars 1 (with a manual audit) or add non-ClinVar sources."
                if args.min_stars > 1 else
                "\n      add non-ClinVar sources (HGMD regulatory, literature curation).")))
    else:
        A(f"      {n_gate:,}   (threshold {GATE_NONCODING:,})  -- MC field only")
        A("      WARNING: near-splice intronic variants (c.X+5, c.X-12) are NOT excluded.")
        A("      Pass --summary for the HGVS splice-distance filter.")
    A(f"      note: {counts.get('bucket_splice', 0):,} canonical splice variants excluded on purpose;")
    A( "      they are non-coding but the signal is local, not long-range.")
    A("")

    A("Q2  GAP G2 -- non-SNV clinical variants")
    A(f"      VCF, non-coding indels/SVs     {nc_indels:>9,}")
    A(f"      VCF, all buckets, indels/SVs   {all_indels:>9,}")
    A( "      DNALongBench tests ZERO of these. Any count here is the gap.")
    A("")
    A("      breakdown by variant class (VCF, non-coding only):")
    for k, v in sorted(bucket_by_type["noncoding"].items(),
                       key=lambda x: -x[1]):
        A(f"        {k:<26} {v:>8,}")
    if svs:
        A("")
        A("      SVs > 50 bp from variant_summary (the VCF omits most CNVs):")
        types = Counter(r["type"] for r in svs.values())
        for k in (0, 1, 2):
            A(f"      stars >= {k}")
            A("        " + f"{'type':<20}" + "".join(f"{b:>10}" for b in SPAN_BANDS) + f"{'total':>8}")
            for t, _ in types.most_common():
                row = Counter(span_band(r["span"]) for r in svs.values()
                              if r["type"] == t and r["stars"] >= k)
                if sum(row.values()):
                    A("        " + f"{t[:19]:<20}"
                      + "".join(f"{row[b]:>10,}" for b in SPAN_BANDS)
                      + f"{sum(row.values()):>8,}")
            fits = sum(1 for r in svs.values()
                       if r["stars"] >= k and r["span"] <= args.max_span)
            A(f"        fits <= {args.max_span:,} bp: {fits:,}")
    A("")

    if svs and args.gtf:
        A("Q3  GAP G1 -- SVs that fit the context AND miss every coding exon")
        A(f"      (span 51 bp - {args.max_span:,} bp; CDS from any GENCODE isoform)")
        fit = [r for r in svs.values() if r["span"] <= args.max_span]
        A("        " + f"{'region':<18}" + "".join(f"{'>=' + str(k) + '*':>9}" for k in (0, 1, 2)))
        for g in GTF_REGIONS:
            A("        " + f"{g:<18}" + "".join(
                f"{sum(1 for r in fit if r['gtf_region'] == g and r['stars'] >= k):>9,}"
                for k in (0, 1, 2)))
        nc_sv = [r for r in fit if r["gtf_region"] not in GATE_EXCLUDED_REGIONS
                 and r["stars"] >= 1]
        A(f"      non-CDS SVs at >= 1 star, by type x span ({len(nc_sv):,}):")
        for t, _ in Counter(r["type"] for r in nc_sv).most_common():
            row = Counter(span_band(r["span"]) for r in nc_sv if r["type"] == t)
            A(f"        {t[:19]:<20}" + "".join(
                f"{b}:{row[b]:<6,}" for b in SPAN_BANDS[:4] if row[b]))
        A( "      -> these are the only SVs whose effect must go through regulation;")
        A( "         anything overlapping CDS can be scored by 'did it hit an exon'.")
        A("")

    A("Q4  SPLIT RISK -- gene concentration of the Q1 gate set")
    A(f"      distinct genes                 {len(genes):>9,}")
    A(f"      genes covering 50% of positives{genes_to_half:>9,}")
    if n_gate:
        A(f"      top-20 genes share             "
          f"{sum(c for _, c in top) / n_gate:>8.1%}")
    A( "      -> if this is concentrated, a chromosome split still leaks.")
    A( "         Use gene-level holdout and report both.")
    A("")
    A("      top 20 genes:")
    for g, c in top:
        A(f"        {g:<18} {c:>6,}")
    A("")

    A("NEGATIVE POOL (benign / likely benign, star-filtered)")
    for b, c in benign_bucket.most_common():
        A(f"  {b:<14} {c:>9,}")
    if benign_loc:
        A("  non-coding/unannotated benign, located by HGVS name:")
        for loc in LOCATIONS:
            if benign_loc[loc]:
                A(f"    {loc:<16} {benign_loc[loc]:>9,}")
    A( "  -> raw counts only. Matching on gene, distance-to-TSS, region class")
    A( "     and allele frequency happens in Week 4-7, not here.")
    A("")
    A("TOP DISEASES (Q1 gate set)")
    for d, c in diseases.most_common(15):
        A(f"  {c:>6,}  {d[:60]}")
    A("")
    A(f"variant table written to: {tsv}")
    if svs:
        A(f"SV table written to:      {sv_tsv}")

    report = "\n".join(L)
    with open(os.path.join(args.outdir, "report.txt"), "w", encoding="utf-8") as out:
        out.write(report + "\n")
    print(report)


if __name__ == "__main__":
    main()
