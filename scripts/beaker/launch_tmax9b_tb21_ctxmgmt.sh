#!/usr/bin/env bash
#
# tmax-9b-tb21-64k (terminal-bench@2.1) via Vanillux2Agent, WITH the
# context-management harness (stub stale file-writes, truncate+spill tool
# output, str_replace/insert/create/apply_edits/read tools) — on by default
# in Vanillux2Agent.
#
# Pinned to commit 66e2e2f6 on context_management_fixes: the FIXED version.
# The first context-management commit (6069c3d4, context_management branch)
# regressed pass@1/pass@5 (27.6/43.8 -> 26.7/40.4) for four reasons — dead
# edit tool (bash-only prompt), ARG_MAX crashes, embedded-null-byte crashes,
# and a JSON-safety gap — all fixed on this branch; see
# docs/vanillux2_context_management.md's "Regression fixes" section.
# No extra flags are needed to enable context management; this is otherwise
# identical to launch_tmax9b_tb21_replicate.sh (the pre-context-management
# baseline run), so all three runs are directly comparable.
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
    --job-name tmax-9b-tb21-64k-vanillux2-ctxmgmt-fix-k5 \
    --cluster ai2/jupiter \
    --priority high \
    --budget ai2/oe-omai \
    --repo-ref 66e2e2f6b09bf855fa1591c0171a792a378885b3
