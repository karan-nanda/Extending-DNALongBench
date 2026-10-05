#!/bin/bash
# Builds the Caduceus environment on Linux x86_64 with an NVIDIA GPU (tested on Ubuntu 24.04
# under WSL2), pinned to the versions in
# DNALONGBENCH/experiments/Caduceus/Caduceus_CMP_eQTLP_ETGP/caduceus_env.yml
# (torch 2.2.0, mamba-ssm 1.2.0.post1, causal-conv1d 1.2.0.post2), on Python 3.10.
# Safe to rerun. No compiler or nvcc needed: mamba-ssm and causal-conv1d come as prebuilt wheels.
#
#   bash analysis/caduceus_env_setup.sh            # venv at ~/cad
#   CAD_ENV=/path/to/venv bash analysis/caduceus_env_setup.sh
#
# On Windows, run it inside WSL2:
#   wsl -d Ubuntu-24.04 -u root -- bash "/mnt/d/Extending DNALongBench/analysis/caduceus_env_setup.sh"
set -euo pipefail
CAD_ENV=${CAD_ENV:-$HOME/cad}

UV=$(command -v uv || echo ~/.local/bin/uv)
if [ ! -x "$UV" ]; then
    curl -LsSf https://astral.sh/uv/install.sh | sh > /dev/null
    UV=~/.local/bin/uv
fi
export VIRTUAL_ENV=$CAD_ENV
[ -x "$CAD_ENV/bin/python" ] || $UV venv -q --python 3.10 "$CAD_ENV"

$UV pip install -q torch==2.2.0 --index-url https://download.pytorch.org/whl/cu121
$UV pip install -q "numpy<2" packaging ninja wheel setuptools transformers==4.38.1 \
    pyfaidx pandas scikit-learn einops kipoiseq pytabix tqdm
# Prebuilt wheels straight from the GitHub releases: their setup.py needs nvcc just to
# pick a wheel, and torch 2.2 pip wheels use the pre-cxx11 ABI.
REL=https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.2.0.post2
$UV pip install "$REL/causal_conv1d-1.2.0.post2+cu122torch2.2cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"
REL=https://github.com/state-spaces/mamba/releases/download/v1.2.0.post1
$UV pip install "$REL/mamba_ssm-1.2.0.post1+cu122torch2.2cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"

"$CAD_ENV/bin/python" - <<'EOF'
import torch, causal_conv1d, mamba_ssm
from mamba_ssm.modules.mamba_simple import Mamba, Block
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), "| mamba_ssm", mamba_ssm.__version__)
EOF
