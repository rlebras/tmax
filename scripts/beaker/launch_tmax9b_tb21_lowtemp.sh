#!/usr/bin/env bash
#
# tmax-9b-tb21-64k (terminal-bench@2.1) via Vanillux2Agent, with LOWER
# SAMPLING TEMPERATURE ONLY (ablation arm): temperature 0.2 / top_p 0.9 vs
# the 0.7 / 0.95 baseline. No other changes vs
# launch_tmax9b_tb21_replicate.sh.
#
# Pinned to commit 4c1beb60c5558140890e9f155075ac5abb3b886f on low_temp.
#
# Run from the root of the tmax repo checkout.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

./beaker_configs/launch_eval.sh allenai/tmax-9b \
    --revision main \
    --name tmax-9b-tb21-64k \
    --gpus 1 --tp 1 --dp 1 \
    --tool-call-parser qwen3_xml \
    --mirror-url jupiter-cs-aus-137.reviz.ai2.in:5000 \
    --model-provider openai \
    --language-model-only \
    --max-model-len 65536 \
    --dataset-path /weka/oe-adapt-default/shashankg/datasets/terminal-bench-2-1 \
    --agent Vanillux2Agent:Vanillux2Agent \
    --n-attempts 5 \
    --job-name tmax-9b-tb21-64k-vanillux2-lowtemp-k5 \
    --cluster ai2/jupiter \
    --priority high \
    --budget ai2/oe-omai \
    --repo-ref 4c1beb60c5558140890e9f155075ac5abb3b886f
