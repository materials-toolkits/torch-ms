#!/usr/bin/env bash
# Paired sampling of MatterGen, coupled and independent Langevin step: run.sh FIRST_BATCH NUM_BATCHES
set -euo pipefail
export PYTHONPATH=src
mkdir -p results
echo "host: $(hostname)  commit: ${GIT_SHA:-unknown}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
python experiments/mattergen-langevin/sample.py "results/langevin_$1.npz" 64 "$2" "$1"
echo "done: batches $1 to $(( $1 + $2 - 1 ))"
