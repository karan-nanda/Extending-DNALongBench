#!/bin/bash
# Installs the released loaders' dependencies into the WSL Caduceus venv, then runs
# check_released_loaders.py. Run from Windows:
#   wsl -d Ubuntu-24.04 -u root --cd "/mnt/d/Extending DNALongBench" -- bash analysis/check_released_loaders.sh
set -euo pipefail
DEBIAN_FRONTEND=noninteractive apt-get install -y -qq zlib1g-dev > /dev/null
VIRTUAL_ENV=/root/cad /root/.local/bin/uv pip install -q kipoiseq pytabix pyfaidx tqdm
/root/cad/bin/python -W ignore analysis/check_released_loaders.py
