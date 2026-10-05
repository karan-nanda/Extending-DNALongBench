#!/usr/bin/env bash
# Fetches everything the repo does not ship, pinned and checksummed, into the paths the
# analysis scripts expect. Run from the repo root. Safe to rerun: verified files are skipped.
#
#   bash fetch_data.sh                  # benchmark: DNALONGBENCH clone + eQTL data (~1 GB, hg38 ~3 GB unpacked)
#   bash fetch_data.sh --models         # + HyenaDNA / Caduceus weights (~200 MB)
#   bash fetch_data.sh --embeddings     # + our frozen-model embeddings (~950 MB): every probe result, no GPU
#   bash fetch_data.sh --clinvar        # + ClinVar / GENCODE inputs to rebuild census/ and data/clinical/ (~750 MB)
#   bash fetch_data.sh --all
#
# Needs: bash, curl, git, gzip, and md5sum (Linux) or md5 (macOS).
set -euo pipefail
cd "$(dirname "$0")"

MODELS=0; EMB=0; CLINVAR=0
for a in "$@"; do
  case "$a" in
    --models) MODELS=1 ;;
    --embeddings) EMB=1 ;;
    --clinvar) CLINVAR=1 ;;
    --all) MODELS=1; EMB=1; CLINVAR=1 ;;
    *) echo "unknown option: $a" >&2; exit 2 ;;
  esac
done

md5_of() { if command -v md5sum >/dev/null; then md5sum "$1" | cut -d' ' -f1; else md5 -q "$1"; fi; }

# get URL DEST MD5 -- download unless DEST already exists with the right checksum
get() {
  local url=$1 dest=$2 md5=$3
  if [ -f "$dest" ] && [ "$(md5_of "$dest")" = "$md5" ]; then echo "ok    $dest"; return; fi
  mkdir -p "$(dirname "$dest")"
  echo "fetch $dest"
  curl -fL --retry 3 -o "$dest.part" "$url"
  local got; got=$(md5_of "$dest.part")
  if [ "$got" != "$md5" ]; then
    echo "CHECKSUM MISMATCH for $dest: expected $md5, got $got" >&2; exit 1
  fi
  mv "$dest.part" "$dest"
}

# ---------------------------------------------------------------- DNALONGBENCH (audited commit)
DLB_COMMIT=2c4060a933fe9e7768d151fcd3c8216ebe0d7d26
if [ ! -d DNALONGBENCH/.git ]; then
  # core.longpaths: the vendored flash-attention tree exceeds Windows MAX_PATH
  git -c core.longpaths=true clone https://github.com/ma-compbio/DNALONGBENCH.git DNALONGBENCH
fi
git -C DNALONGBENCH -c advice.detachedHead=false checkout -q "$DLB_COMMIT"
echo "ok    DNALONGBENCH @ ${DLB_COMMIT:0:7}"

# ---------------------------------------------------------------- eQTL data (Dataverse doi:10.7910/DVN/YUP2G5, v1.0)
# Tables are fetched as the ORIGINAL .tsv: Dataverse's ingested .tab copies do not
# match the filenames in the configs.
DV=https://dataverse.harvard.edu/api/access/datafile
while read -r id path md5; do
  case "$path" in *.tsv) q="?format=original" ;; *) q="" ;; esac
  get "$DV/$id$q" "data/$path" "$md5"
done <<'EOF'
10443586 eQTL/targets/Adipose_Subcutaneous.data.tsv 836a398d4b4b55b0f5ca5085c106b153
10443580 eQTL/targets/Artery_Tibial.data.tsv 893dcf155f6921ef11079485630ef00a
10443585 eQTL/targets/Cells_Cultured_fibroblasts.data.tsv 2bc845dbb34e9f50f976da8c2649e794
10443582 eQTL/targets/Muscle_Skeletal.data.tsv 031f5ccc47c6a31ca698f89aee5ded75
10443583 eQTL/targets/Nerve_Tibial.data.tsv 92163584123266a11ad14dc746e69f0e
10443581 eQTL/targets/Skin_Not_Sun_Exposed_Suprapubic.data.tsv 2f7572458c073e3947b4709e5d40d41b
10443587 eQTL/targets/Skin_Sun_Exposed_Lower_leg.data.tsv 6e9aa92d761c623dd38b0f9a5775660a
10443588 eQTL/targets/Thyroid.data.tsv cdc5f73f2921d588c01bec8da5ee90c7
10443579 eQTL/targets/Whole_Blood.data.tsv 045ddb02c269eacb55751064c7378938
10443584 eQTL/targets/combined_data.tsv 902f34d3ea9b7464117946598430667a
10443603 eQTL/blacklist/Adipose_Subcutaneous.blacklist.bed.gz 85e990556826c5c7844e0d1522776d04
10443604 eQTL/blacklist/Adipose_Subcutaneous.blacklist.bed.gz.tbi a2d26eb5ae24e310e33c74e098aee123
10443602 eQTL/blacklist/Artery_Tibial.blacklist.bed.gz 6fe6ff40a8363f6f666b41256c40a934
10443606 eQTL/blacklist/Artery_Tibial.blacklist.bed.gz.tbi ca3d3306613c3063a695b035dcff5f28
10443589 eQTL/blacklist/Cells_Cultured_fibroblasts.blacklist.bed.gz fb14dd8399cdd6ce2d71883677ba3511
10443594 eQTL/blacklist/Cells_Cultured_fibroblasts.blacklist.bed.gz.tbi 97917cc20cdbe7e6592830729b58604e
10443597 eQTL/blacklist/combined_blacklist.bed.gz de1a9abc93fc1d29f5d1f356017c9e2b
10443596 eQTL/blacklist/combined_blacklist.bed.gz.tbi 1826e4982f86c0820b68ead3704a1d26
10443605 eQTL/blacklist/Muscle_Skeletal.blacklist.bed.gz 5cff47eb01928354ee31c0d9be1db4fb
10443593 eQTL/blacklist/Muscle_Skeletal.blacklist.bed.gz.tbi ca60325c6e12723502b2082b27b50c06
10443591 eQTL/blacklist/Nerve_Tibial.blacklist.bed.gz 1ad8da102778c1e994e42b07f7e5a4e9
10443608 eQTL/blacklist/Nerve_Tibial.blacklist.bed.gz.tbi 64bf2214c54a785e5aaf7c2606b82a1d
10443595 eQTL/blacklist/Skin_Not_Sun_Exposed_Suprapubic.blacklist.bed.gz 77da03283e5707f43942d38bd747fa46
10443590 eQTL/blacklist/Skin_Not_Sun_Exposed_Suprapubic.blacklist.bed.gz.tbi 4b59259b93f44967d3744e20293e6131
10443600 eQTL/blacklist/Skin_Sun_Exposed_Lower_leg.blacklist.bed.gz 3d8022abe35466e72f0dbf9cf11ab4a6
10443607 eQTL/blacklist/Skin_Sun_Exposed_Lower_leg.blacklist.bed.gz.tbi 3bce31d2c43b21a2a37bf9c72b6ee989
10443592 eQTL/blacklist/Thyroid.blacklist.bed.gz bddfbf4c7c0c2b2951ea0a575db9eb5c
10443598 eQTL/blacklist/Thyroid.blacklist.bed.gz.tbi 1f734b065310ead6bb7b6a6b71277c96
10443599 eQTL/blacklist/Whole_Blood.blacklist.bed.gz bba0ce8ede12216ccb64ddb0593078fc
10443601 eQTL/blacklist/Whole_Blood.blacklist.bed.gz.tbi 4c02d13060a310040344689a977d4aa4
11593674 eQTL/config/gtex_hg38.Adipose_Subcutaneous.config 9820b3eb1c74e6cde928526a60c02b97
11593671 eQTL/config/gtex_hg38.Artery_Tibial.config 4dd840d6f54882feace192e707e73853
11593680 eQTL/config/gtex_hg38.Cells_Cultured_fibroblasts.config 370c5de3c93a46b35521daad70940e34
11593675 eQTL/config/gtex_hg38.config aeeac082482df32b027711d8b6e428ee
11593673 eQTL/config/gtex_hg38.Muscle_Skeletal.config 2a066cff5817029a45ce3655c8ae5b63
11593679 eQTL/config/gtex_hg38.Nerve_Tibial.config ae56c3372c2771f444e85a5f4214b3a3
11593676 eQTL/config/gtex_hg38.Skin_Not_Sun_Exposed_Suprapubic.config b42e9d92083ea648859b62d257e75725
11593672 eQTL/config/gtex_hg38.Skin_Sun_Exposed_Lower_leg.config 16d5d68a130a32e2ee53f447988d7171
11593677 eQTL/config/gtex_hg38.Thyroid.config ef91e58ba663f8b299e0b587bff802c3
11593678 eQTL/config/gtex_hg38.Whole_Blood.config 038bc5c584d363505457643e3b0048ae
10443609 eQTL/seqs/hg38.fa.fai 1954682fe412d4c5f8b93ad8e4083181
10443685 eQTL/seqs/hg38.fa.gz 40a7a6ec0779abd27fab533ba4b0d09b
EOF
# The scripts read the uncompressed FASTA (random access via the .fai).
if [ ! -s data/eQTL/seqs/hg38.fa ]; then
  echo "unpack data/eQTL/seqs/hg38.fa"
  gzip -dc data/eQTL/seqs/hg38.fa.gz > data/eQTL/seqs/hg38.fa.part
  mv data/eQTL/seqs/hg38.fa.part data/eQTL/seqs/hg38.fa
fi

# ---------------------------------------------------------------- pretrained weights (pinned HF revisions)
if [ $MODELS = 1 ]; then
  HF=https://huggingface.co
  R=LongSafari/hyenadna-medium-450k-seqlen; REV=820fe013ec58c0f87f8caba21af0c48fdc8d0ddd
  D=data/models/hyenadna-medium-450k-seqlen
  get "$HF/$R/resolve/$REV/config.json"  "$D/config.json"  af284a56cc5e29271339207e468f1871
  get "$HF/$R/resolve/$REV/weights.ckpt" "$D/weights.ckpt" e7770e3e8c0891791655dcfb64edb358
  R=kuleshov-group/caduceus-ph_seqlen-131k_d_model-256_n_layer-16; REV=b0477522ac5d044ad03578aa724ec8e4bdbd405b
  D=data/models/caduceus-ph_seqlen-131k_d_model-256_n_layer-16
  while read -r f md5; do get "$HF/$R/resolve/$REV/$f" "$D/$f" "$md5"; done <<'EOF'
README.md dd9538cfc0f2cc30c10047febee5f7df
config.json 87ea65c94c5db6bffef9393c115787ff
configuration_caduceus.py b0e22ddce15455e59888070e9e84c78c
model.safetensors 8142b9719889a8f270ceadc763792f7c
modeling_caduceus.py 2dbe4714a5af84be2c35de7015282c6d
modeling_rcps.py 7c609428ac2ecc40d6319981edf2269f
special_tokens_map.json 2f24a0b35ba01cb787c49e1e14304c65
tokenization_caduceus.py ecdbbe6eebcf3281c2020d30a1834ceb
tokenizer_config.json 33301714d7cc4fa4a47dd7ed206a4fb2
EOF
fi

# ---------------------------------------------------------------- our embeddings (GitHub release)
REL=https://github.com/karan-nanda/Extending-DNALongBench/releases/download/data-v1
if [ $EMB = 1 ]; then
  get "$REL/pooled_embeddings.tar.gz" downloads/pooled_embeddings.tar.gz 19ea3fe233bc8a0e495fb2b84ad225c3
  tar -xzf downloads/pooled_embeddings.tar.gz            # -> data/{eQTL_v2,clinical}/*_emb/
  get "$REL/window_emb_hyena.npz"    data/eQTL_v2/window_emb/hyena/window_emb.npz    ac995f0507bc7af054d1e9dafdd92a49
  get "$REL/window_emb_caduceus.npz" data/eQTL_v2/window_emb/caduceus/window_emb.npz f34dd7da385f8756b10a3ba3b92b1f92
fi

# ---------------------------------------------------------------- ClinVar / GENCODE (exact releases used)
if [ $CLINVAR = 1 ]; then
  get https://ftp.ncbi.nlm.nih.gov/pub/clinvar/vcf_GRCh38/archive_2.0/2026/clinvar_20260905.vcf.gz \
      clinvar.vcf.gz ece04fe2ee72db1dd988d8b188df34b9
  # NCBI archives variant_summary only monthly; the weekly file used here is mirrored on our release.
  get "$REL/variant_summary.txt.gz" variant_summary.txt.gz 17ec6902042d1c5d9f87c2d64eb95a69
  get https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_50/gencode.v50.annotation.gtf.gz \
      gencode.v50.annotation.gtf.gz 2d273848c6068682fabea72fae4896a9
fi

echo "done"
