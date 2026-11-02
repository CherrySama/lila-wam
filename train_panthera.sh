#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
export PYTHONDONTWRITEBYTECODE=1
export TZ=Asia/Shanghai
exec .venv/bin/python -u train.py \
    --config configs/panthera_dual.yaml \
    --save_dir checkpoints_vla/panthera "$@"
