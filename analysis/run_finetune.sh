#!/bin/sh
# Three folds x 2 epochs of end-to-end HyenaDNA fine-tuning on eQTLP-v2.
# ~3.1 h/epoch measured (2.27 s/pair train, 4648 train pairs, plus eval), so ~19 h total.
set -e
for f in 0 1 2; do
  python -X utf8 analysis/hyena_finetune.py --fold "$f" --epochs 2 --maxlen 131072
done
echo "ALL FOLDS DONE"
