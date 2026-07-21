#!/usr/bin/env bash
#
# tmax-9b-tb21-64k (terminal-bench@2.1) via Vanillux2Agent, WITH the
# context-management harness AND the self-test submission gate
# (declare_criteria/run_check tools, isolated checks, submit rejected until
# every declared criterion is covered) — both on by default in Vanillux2Agent.
#
# Pinned to commit 01776cdb on self_test_v3, which adds context-overflow
# recovery and wall-clock discipline on top of the gate (see the budget
# sections of docs/vanillux2_context_management.md). The gate and reactive
# overflow recovery are on by default; max_context_tokens below matches
# --max-model-len so the PROACTIVE budget check is active too. Otherwise
# identical to launch_tmax9b_tb21_ctxmgmt.sh (context management, no gate)
# and launch_tmax9b_tb21_replicate.sh (pre-context-management baseline), so
# all three runs are directly comparable. See docs/vanillux2_self_test_gate.md
# "Measurement" for what this run is meant to produce: the finished-wrong /
# step-exhaustion / stopped-early outcome taxonomy,
# submitted_with_failing_checks rate, and self-test adoption / added
# steps-tokens vs. the baselines — computed per arm with
#   uv run python scripts/vanillux2_self_test_report.py --glob '<trials>/*'
#
# To A/B against the gate disabled on the SAME commit, add:
#   --agent-kwarg enable_self_test_gate=false
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
    --agent-kwarg max_context_tokens=65536 \
    --n-attempts 5 \
    --job-name tmax-9b-tb21-64k-vanillux2-selftest-v3-k5 \
    --cluster ai2/jupiter \
    --priority high \
    --budget ai2/oe-omai \
    --repo-ref 01776cdb
