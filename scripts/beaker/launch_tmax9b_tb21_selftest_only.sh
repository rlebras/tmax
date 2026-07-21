#!/usr/bin/env bash
#
# tmax-9b-tb21-64k (terminal-bench@2.1) via Vanillux2Agent, with the
# SELF-TEST GATE ONLY — no context management, no edit tools (ablation arm).
# The tool contract is plain bash, identical to
# launch_tmax9b_tb21_replicate.sh's baseline; the only delta is the gate
# (agent-check convention + criteria file + isolated checks + bounded submit
# rejection), so this A/B isolates self-testing from the compaction work on
# self_test_v3 (launch_tmax9b_tb21_selftest.sh carries both).
#
# Pinned to commit 2f83c5db06fcdd2f2707805093d866b29e3b6244 on self_test_only.
# The gate is on by default; to A/B against it disabled on the SAME commit:
#   --agent-kwarg enable_self_test_gate=false
#
# Analyze each arm's trials with (from any checkout of self_test_v3):
#   uv run python scripts/vanillux2_self_test_report.py --glob '<trials>/*'
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
    --job-name tmax-9b-tb21-64k-vanillux2-selftest-only-k5 \
    --cluster ai2/jupiter \
    --priority high \
    --budget ai2/oe-omai \
    --repo-ref 2f83c5db06fcdd2f2707805093d866b29e3b6244
