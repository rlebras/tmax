#!/usr/bin/env bash
#
# tmax-9b-tb21-64k (terminal-bench@2.1) via Vanillux2Agent, WITH the
# context-management harness AND the self-test submission gate
# (declare_criteria/run_check, isolated checks, submit rejected until
# min_criteria are covered) — both on by default in Vanillux2Agent.
#
# Pinned to commit 7679ea2290b6a140c5d088249fe324ec3658b099 on self_test_gate.
# No extra flags are needed to enable the gate; this is otherwise identical
# to launch_tmax9b_tb21_ctxmgmt.sh (context-management, no gate) and
# launch_tmax9b_tb21_replicate.sh (pre-context-management baseline), so all
# three runs are directly comparable. See
# docs/vanillux2_self_test_gate.md for the mechanism, and its "Not yet
# measured" section for what this run is meant to produce: the
# finished-wrong / step-exhaustion / context-run-out outcome taxonomy,
# submitted_with_failing_checks rate, and self-test adoption / added
# steps-tokens, vs. the ctxmgmt-only baseline above.
#
# To A/B against the gate disabled without a separate pinned commit, add:
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
    --n-attempts 5 \
    --job-name tmax-9b-tb21-64k-vanillux2-selftest-k5 \
    --cluster ai2/jupiter \
    --priority high \
    --budget ai2/oe-omai \
    --repo-ref 7679ea2290b6a140c5d088249fe324ec3658b099
