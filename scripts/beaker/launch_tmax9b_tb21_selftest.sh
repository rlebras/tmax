#!/usr/bin/env bash
#
# tmax-9b-tb21-64k (terminal-bench@2.1) via Vanillux2Agent, WITH the
# standalone self-test submission gate (Vanillux2Agent/self_test.py):
# acceptance-criteria declaration, harness-intercepted `agent-check`,
# isolated verification, and a submit gate bounded by max_gate_rejections.
# Off by default in Vanillux2Agent, so it's explicitly enabled below.
#
# This is otherwise identical to launch_tmax9b_tb21_replicate.sh (the
# baseline, no gate) so the two runs are directly comparable: outcome
# taxonomy (finished-wrong / step-exhaustion / context-run-out / PASS),
# submitted_with_failing_checks rate, and median added steps/tokens.
#
# --repo-ref pins to the self_testing_v2 branch — push it to origin before
# running (launch_eval.sh needs the ref to exist on the remote to clone it
# in the Beaker job).
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
    --agent-kwarg enable_self_test_gate=true \
    --n-attempts 5 \
    --job-name tmax-9b-tb21-64k-vanillux2-selftest-k5 \
    --cluster ai2/jupiter \
    --priority high \
    --budget ai2/oe-omai \
    --repo-ref self_testing_v2
