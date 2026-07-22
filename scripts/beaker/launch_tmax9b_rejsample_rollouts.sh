#!/usr/bin/env bash
#
# STEP 1 of the rejection-sampling SFT loop (docs/rejection_sampling_sft.md),
# wired as a one-command gantry job that reuses the eval's vLLM + harbor +
# Vanillux2Agent orchestration (beaker_configs/launch_eval.sh) — but pointed at
# the TRAINING corpus, not terminal-bench.
#
# What it does: serve tmax-9b via vLLM in-job and run the SAME Vanillux2Agent
# harness the eval uses over `tmax/TMax-15K-Harbor` (the published 15k training
# corpus, decontaminated vs terminal-bench), at high k for coverage. Output is
# the usual per-task trial layout under jobs/<job>/<task>/{result.json,
# agent/trajectory.json} — exactly what rl_data.rejection_sample_sft.py reads.
#
# CONTAMINATION GUARD: dataset is the TMax-15K training corpus, never
# terminal-bench. Do not repoint DATASET at a terminal-bench dataset.
#
# Two-step usage (smoke first, it de-risks dataset resolution + wiring cheaply):
#   # smoke: 10 tasks x 2 attempts
#   N_TASKS=10 N_ATTEMPTS=2 PRIORITY=normal bash scripts/beaker/launch_tmax9b_rejsample_rollouts.sh
#   # full first round: 2000 tasks x 16 attempts (override N_TASKS=0-ish to widen)
#   bash scripts/beaker/launch_tmax9b_rejsample_rollouts.sh
#
# Then harvest (local, focusing the flaky band):
#   uv run python -m rl_data.rejection_sample_sft <fetched jobs/ dir> \
#       --out rl_data/output/rejsample_sft.jsonl --max-solve-rate 0.8 \
#       --push-to-hub <you>/tmax-rejsample-sft
#
# NOTE on the corpus ref: TMax-15K-Harbor is a published Harbor registry
# dataset, so `--dataset tmax/TMax-15K-Harbor@latest` should resolve in-job. If
# the job's harbor build can't resolve the registry ref (as happened for
# terminal-bench-2-1), set DATASET_PATH to a weka copy and pass
# `--dataset-path "$DATASET_PATH"` instead of `--dataset`.
#
# Run from the root of the tmax repo checkout.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

# ---- rejection-sampling rollout knobs (env-overridable) ----
DATASET="${DATASET:-tmax/TMax-15K-Harbor@latest}"   # the TRAINING corpus
N_ATTEMPTS="${N_ATTEMPTS:-16}"                        # high k = more harvestable wins
N_TASKS="${N_TASKS:-2000}"                            # first-round subset; raise to widen
PRIORITY="${PRIORITY:-high}"
TEMPERATURE="${TEMPERATURE:-1.0}"                     # widen the attempt distribution
JOB_NAME="${JOB_NAME:-tmax-9b-rejsample-rollouts-k${N_ATTEMPTS}}"
REPO_REF="${REPO_REF:-__PINNED_SHA__}"

./beaker_configs/launch_eval.sh allenai/tmax-9b \
    --revision main \
    --name tmax-9b-rejsample \
    --gpus 1 --tp 1 --dp 1 \
    --tool-call-parser qwen3_xml \
    --mirror-url jupiter-cs-aus-137.reviz.ai2.in:5000 \
    --model-provider openai \
    --language-model-only \
    --max-model-len 65536 \
    --dataset "$DATASET" \
    --n-attempts "$N_ATTEMPTS" \
    --n-tasks "$N_TASKS" \
    --agent Vanillux2Agent:Vanillux2Agent \
    --agent-kwarg "temperature=$TEMPERATURE" \
    --job-name "$JOB_NAME" \
    --cluster ai2/jupiter \
    --priority "$PRIORITY" \
    --budget ai2/oe-omai \
    --repo-ref "$REPO_REF"
