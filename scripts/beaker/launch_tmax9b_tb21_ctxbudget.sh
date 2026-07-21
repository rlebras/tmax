#!/usr/bin/env bash
#
# tmax-9b-tb21-64k (terminal-bench@2.1) via Vanillux2Agent, with CONTEXT
# MANAGEMENT + OVERFLOW RECOVERY + WALL-CLOCK DISCIPLINE and the self-test
# gate DISABLED (ablation arm). Pins the same commit as
# launch_tmax9b_tb21_selftest.sh, so the only delta vs that arm is the gate,
# and the delta vs launch_tmax9b_tb21_replicate.sh is the full budget stack.
# Targets the two measured infrastructure loss buckets on the 444-run
# baseline: 38% context-overflow early stops and 7.4% wall-clock kills.
#
# Pinned to commit 01776cdb08a7cf265ad2ae73395d15ecdeeada26 on self_test_v3.
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
    --agent-kwarg enable_self_test_gate=false \
    --agent-kwarg max_context_tokens=65536 \
    --n-attempts 5 \
    --job-name tmax-9b-tb21-64k-vanillux2-ctxbudget-k5 \
    --cluster ai2/jupiter \
    --priority high \
    --budget ai2/oe-omai \
    --repo-ref 01776cdb08a7cf265ad2ae73395d15ecdeeada26
