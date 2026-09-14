#!/usr/bin/env bash
# Creates the dedicated conda env for the clearer-voice endpoint.
# Safe to re-run; skips env creation if it already exists.
set -euo pipefail

ENV_NAME="${ENV_NAME:-clearvoice_endpoint}"
eval "$(conda shell.bash hook)"

if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    echo "[setup] creating conda env $ENV_NAME (python 3.10)"
    conda create -y -n "$ENV_NAME" python=3.10
else
    echo "[setup] env $ENV_NAME already exists"
fi

conda activate "$ENV_NAME"
python -m pip install --upgrade pip wheel

# numpy<2 first: demucs 4.x predates the numpy 2 ABI break.
python -m pip install "numpy<2"

echo "[setup] installing clearvoice + demucs + gradio"
python -m pip install clearvoice demucs gradio soundfile librosa scipy

# Pin torch LAST so nothing above silently upgrades it. cu121 covers sm_75 (Quadro RTX 6000).
echo "[setup] pinning torch 2.5.1 / torchaudio 2.5.1 (cu121)"
python -m pip install torch==2.5.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121

python - <<'PY'
import torch, torchaudio
print(f"[setup] torch {torch.__version__} torchaudio {torchaudio.__version__} cuda={torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"[setup] device: {torch.cuda.get_device_name(0)}")
PY
echo "[setup] done"
