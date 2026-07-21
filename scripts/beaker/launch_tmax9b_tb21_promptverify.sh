#!/usr/bin/env bash
#
# tmax-9b-tb21-64k (terminal-bench@2.1) via Vanillux2Agent, with the
# PROMPT-ONLY VERIFICATION ADDENDUM (ablation arm): the self-test gate's
# teaching with no enforcement. No context management, no gate, no other
# changes — the only delta vs launch_tmax9b_tb21_replicate.sh; compare also
# against launch_tmax9b_tb21_selftest_only.sh to isolate enforcement.
#
# Pinned to commit 54c0f8dfbb5240fe726a3a86a0b7bba632fe2457 on prompt_self_verify.
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
    --job-name tmax-9b-tb21-64k-vanillux2-promptverify-k5 \
    --cluster ai2/jupiter \
    --priority high \
    --budget ai2/oe-omai \
    --repo-ref 54c0f8dfbb5240fe726a3a86a0b7bba632fe2457
