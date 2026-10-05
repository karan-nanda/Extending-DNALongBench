#!/bin/bash
# Runs check_released_loaders.py (robustness check R5) in the Caduceus venv, which also
# holds the released loaders' dependencies (kipoiseq, pytabix, pyfaidx).
# Build that venv first with analysis/caduceus_env_setup.sh. From the repo root:
#   bash analysis/check_released_loaders.sh             # venv at ~/cad
#   CAD_ENV=/path/to/venv bash analysis/check_released_loaders.sh
# On Windows, inside WSL2:
#   wsl -d Ubuntu-24.04 -u root --cd "/mnt/d/Extending DNALongBench" -- bash analysis/check_released_loaders.sh
set -euo pipefail
CAD_ENV=${CAD_ENV:-$HOME/cad}
"$CAD_ENV/bin/python" -W ignore analysis/check_released_loaders.py
