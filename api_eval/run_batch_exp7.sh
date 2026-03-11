#!/usr/bin/env bash
set -euo pipefail

# Datasets and percents are configured in run_batch_exp7.py:
# DATASET_PERCENT_MAP = {"DatasetA": 100.0, ...}
# Optional: pass dataset keys to run a subset:
# ./run_batch_exp7.sh DatasetA DatasetC

python "$(dirname "$0")/run_batch_exp7.py" \
  "$@" \
  --reasoning-modes true false
