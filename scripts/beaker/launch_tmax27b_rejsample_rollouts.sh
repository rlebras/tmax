#!/usr/bin/env bash
#
# TEACHER rollouts for the 27B->9B rejection-sampling DISTILLATION track
# (docs/rejection_sampling_sft.md). Same one-command shape as the 9B rollout
# launcher, but serves allenai/tmax-27b (pass@1 44.9% vs the 9B's 29.0%) and
# uses the SAME Vanillux2Agent bash harness, so the harvested trajectories are
# format-compatible with the 9B's SFT data.
#
# 27B serving config comes from the TB2.0 27B eval (scripts/beaker/
# eval_runs_2026-07-08.md): TP=2, qwen3_xml tool parser, --language-model-only,
# no reasoning parser (same Qwen3.5 arch as the 9B).
#
# WHY k=4 (not 1, not 16): the 27B's pass@1 is high, but pass@k > pass@1 —
# extra samples recover the harder tasks it solves only sometimes (exactly the
# ones the 9B most needs), and give a pool to pick a clean demonstration from.
# k=1 under-covers those; k=16 wastes compute on the easy tasks. See the doc's
# "how many teacher rollouts" section.
#
# CONTAMINATION GUARD: dataset is the TMax-15K TRAINING corpus, never
# terminal-bench.
#
# ADAPTIVE TOP-UP (follow-up, cheaper coverage): after this pass, re-run ONLY
# on tasks with zero teacher successes so far, at higher k, to recover the
# hard tail — pass those task ids via harbor -i / a --dataset-path subset.
#
# Usage:
#   # smoke first
#   N_TASKS=10 N_ATTEMPTS=2 PRIORITY=normal bash scripts/beaker/launch_tmax27b_rejsample_rollouts.sh
#   # first teacher round
#   bash scripts/beaker/launch_tmax27b_rejsample_rollouts.sh
#
# Then harvest in GAP mode against the 9B's rollouts (student):
#   uv run python -m rl_data.rejection_sample_sft <27b jobs/ dir> \
#       --student-rollouts <9b jobs/ dir> --max-student-solve-rate 0.8 \
#       --out rl_data/output/distill_sft.jsonl --push-to-hub <you>/tmax-distill-sft
#
# Run from the root of the tmax repo checkout.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

DATASET="${DATASET:-tmax/TMax-15K-Harbor@latest}"   # the TRAINING corpus
N_ATTEMPTS="${N_ATTEMPTS:-4}"                         # teacher k (see header)
N_TASKS="${N_TASKS:-2000}"                            # first-round subset; raise to widen
PRIORITY="${PRIORITY:-high}"
TEMPERATURE="${TEMPERATURE:-0.7}"                     # moderate; teacher is reliable, no need to over-diversify
JOB_NAME="${JOB_NAME:-tmax-27b-rejsample-teacher-k${N_ATTEMPTS}}"
GPUS="${GPUS:-2}"
TP="${TP:-2}"
REPO_REF="${REPO_REF:-358ffb91f7512463e64de1ccff3dd24be679c599}"

./beaker_configs/launch_eval.sh allenai/tmax-27b \
    --revision main \
    --name tmax-27b-rejsample \
    --gpus "$GPUS" --tp "$TP" --dp 1 \
    --tool-call-parser qwen3_xml \
    --mirror-url jupiter-cs-aus-137.reviz.ai2.in:5000 \
    --model-provider openai \
    --language-model-only \
    --max-model-len "${MAX_MODEL_LEN:-32768}" \
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
