#!/bin/bash
#SBATCH --job-name=rejsample-rollouts
#SBATCH --output=logs/rejsample_rollouts_%j.out
#SBATCH --error=logs/rejsample_rollouts_%j.err
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h200:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=960G
#
# STEP 1 of the rejection-sampling SFT loop (see docs/rejection_sampling_sft.md).
#
# Generate MANY rollouts of the CURRENT checkpoint (tmax-9b) over the rl_data
# TRAINING corpus, so step 2 (rl_data/rejection_sample_sft.py) can harvest the
# passing ones as SFT data. Two choices here are deliberate and specific to
# rejection sampling (vs the apples-to-apples pass@k comparison scripts):
#
#   * MODEL is the checkpoint we want to IMPROVE (tmax-9b), not a reference
#     model — we are distilling the model's own successes back into itself.
#   * NUM_SOLUTIONS and SOLUTION_TEMPERATURE are turned UP (16 rollouts,
#     temp 1.0): more diverse attempts per task = more tasks with >=1 passing
#     trajectory to harvest. Coverage of the reachable-solution set is what
#     matters here, not a calibrated pass@k estimate.
#
# CONTAMINATION GUARD: TASKS_DIR MUST be the rl_data generated corpus (already
# decontaminated vs terminal-bench). Never point this at terminal-bench tasks
# — training on the eval set is leakage (rejection_sample_sft.py also refuses
# eval-sourced inputs downstream).

set -euo pipefail

# ---- rejection-sampling parameters ----
TASKS_DIR="${TASKS_DIR:-rl_data/output/tasks_skill_tax_20260401_10k}"

export LAUNCH_VLLM="${LAUNCH_VLLM:-1}"
# The checkpoint being improved. Override VLLM_MODEL to a local path / another
# revision to bootstrap from a different starting point.
export VLLM_MODEL="${VLLM_MODEL:-allenai/tmax-9b}"
MODEL="${MODEL:-hosted_vllm/${VLLM_MODEL}}"
export VLLM_MAX_LEN="${VLLM_MAX_LEN:-65536}"

NUM_SOLUTIONS="${NUM_SOLUTIONS:-16}"       # more samples/task -> more harvestable wins
SOLUTION_TEMPERATURE="${SOLUTION_TEMPERATURE:-1.0}"  # widen the attempt distribution
MAX_ACTIONS="${MAX_ACTIONS:-64}"           # match the eval harness budget
MAX_TOKENS="${MAX_TOKENS:-65536}"
NUM_TASKS="${NUM_TASKS:-999999}"
START_AT="${START_AT:-0}"
SAMPLE_SIZE="${SAMPLE_SIZE:-0}"            # 0 = full corpus (rejection sampling wants breadth)
SAMPLE_SEED="${SAMPLE_SEED:-0}"
COMMAND_TIMEOUT="${COMMAND_TIMEOUT:-600}"
SHELL_INIT_TIMEOUT="${SHELL_INIT_TIMEOUT:-240}"
SHELL_INIT_ATTEMPTS="${SHELL_INIT_ATTEMPTS:-3}"
BUILD_WORKERS="${BUILD_WORKERS:-4}"
BUILD_RETRIES="${BUILD_RETRIES:-3}"
BASE_SIFS_DIR="${BASE_SIFS_DIR:-rl_data/containers}"
WORKERS="${WORKERS:-24}"
NUM_POOL_WORKERS="${NUM_POOL_WORKERS:-16}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
COMPARISON_DIR="$PROJECT_ROOT/rl_data/scripts/comparison"
cd "$PROJECT_ROOT"
mkdir -p logs

# In-job vLLM bring-up (no-op unless LAUNCH_VLLM=1) — same helper the
# comparison rollouts use; auto-picks the qwen3 tool-call/reasoning parsers.
# shellcheck source=../comparison/_vllm_local.sh
source "$COMPARISON_DIR/_vllm_local.sh"
_vllm_start_local

export APPTAINER_DOCKER_USERNAME="${APPTAINER_DOCKER_USERNAME:?Set APPTAINER_DOCKER_USERNAME before running}"
export APPTAINER_DOCKER_PASSWORD="${APPTAINER_DOCKER_PASSWORD:?Set APPTAINER_DOCKER_PASSWORD before running}"

EXTRA_ARGS=()
[ "$SAMPLE_SIZE" -gt 0 ] && EXTRA_ARGS+=(--sample-size "$SAMPLE_SIZE" --sample-seed "$SAMPLE_SEED")
[ -n "$BASE_SIFS_DIR" ] && EXTRA_ARGS+=(--base-sifs-dir "$BASE_SIFS_DIR")

uv run python -m rl_data.generate_solutions \
    --tasks-dir "$TASKS_DIR" \
    --model "$MODEL" \
    --harness vanillux \
    --num-solutions "$NUM_SOLUTIONS" \
    --max-actions "$MAX_ACTIONS" \
    --max-tokens "$MAX_TOKENS" \
    --num-tasks "$NUM_TASKS" \
    --start-at "$START_AT" \
    --workers "$WORKERS" \
    --num-pool-workers "$NUM_POOL_WORKERS" \
    --solution-temperature "$SOLUTION_TEMPERATURE" \
    --command-timeout "$COMMAND_TIMEOUT" \
    --shell-init-timeout "$SHELL_INIT_TIMEOUT" \
    --shell-init-attempts "$SHELL_INIT_ATTEMPTS" \
    --build-workers "$BUILD_WORKERS" \
    --build-retries "$BUILD_RETRIES" \
    --verbose \
    "${EXTRA_ARGS[@]}"

echo
echo "Rollouts written under $TASKS_DIR/*/solutions/. Next:"
echo "  uv run python -m rl_data.rejection_sample_sft $TASKS_DIR \\"
echo "      --out rl_data/output/rejsample_sft.jsonl --push-to-hub <you>/tmax-rejsample-sft"
