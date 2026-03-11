#!/usr/bin/env bash
set -euo pipefail

# Smoke test for pipeline health:
# - exactly 1 sample per dataset
# - keeps default model list and reasoning modes (true/false)
# - use --datasets to run subset if needed
#
# Examples:
# ./run_smoke_test_exp7.sh
# ./run_smoke_test_exp7.sh --datasets "BoKelvin/SLAKE" "flaviagiammarino/vqa-rad"

python "$(dirname "$0")/run_batch_exp7.py" \
  "$@" \
  --smoke-test-one-sample \
  --reasoning-modes true false
