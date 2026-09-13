#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
eval "$(conda shell.bash hook)"
conda activate "${ENV_NAME:-clearvoice_endpoint}"
exec python app.py
