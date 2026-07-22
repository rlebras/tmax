# Rejection-sampling SFT (STaR) for tmax terminal agents

*Source:* [`rl_data/rejection_sample_sft.py`](../rl_data/rejection_sample_sft.py) ·
rollout launch [`run_rejection_sampling_rollouts.sh`](../rl_data/scripts/generate_solutions/run_rejection_sampling_rollouts.sh) ·
SFT launch [`sft_qwen35_9b_rejsample.sh`](../training/open-instruct/scripts/tmax/SFT/sft_qwen35_9b_rejsample.sh)

## Why

An 11-arm harness sweep on terminal-bench 2.1 moved tmax-9b's pass@1 essentially
nowhere (27–30% across every configuration): in-context steering, self-test
gating, budget nudges, and context management were all neutral-to-negative
against a matched baseline (pass@1 29.0%, pass@5 49.4%). The lever is not how
the agent is prompted turn-to-turn.

The exploitable signal is the **pass@1 → pass@5 gap**: the model already solves
~44/89 tasks *sometimes* but only ~26 on any single try. Twenty points of
"known-achievable" capability are lost to variance. Rejection-sampling SFT
(a.k.a. STaR / rejection fine-tuning) attacks exactly this: sample many
rollouts, keep the ones that *pass their own verifier*, and fine-tune the model
on its own successes — turning occasionally-reachable behavior into the
default, converting pass@k capability into pass@1 reliability, with no human
labels.

## The loop

```
[1] generate   run_rejection_sampling_rollouts.sh
    tmax-9b (vLLM) × the rl_data TRAINING corpus, k=16 rollouts/task, temp 1.0
        │  writes  <task>/solutions/<model>_vanillux_summary.json
        ▼          (each = {results: [{reward, messages, ...}]})
[2] harvest    rl_data/rejection_sample_sft.py
    keep reward==1 · drop format-errors · require a clean submit · dedup ·
    cap per task · length-cap  →  SFT jsonl ({messages, tools, dataset, id})
        │  (optionally --push-to-hub <you>/tmax-rejsample-sft)
        ▼
[3] train      sft_qwen35_9b_rejsample.sh
    open_instruct/finetune.py on the harvested data (base hamishivi/Qwen3.5-9B)
        ▼
[4] eval       scripts/beaker/launch_tmax9b_tb21_replicate.sh (terminal-bench, HELD OUT)
```

Iterate: the SFTed checkpoint becomes the model in step 1 of the next round.

## Contamination guard (the one thing you must not get wrong)

The training signal **must** come from the rl_data generated corpus, which the
pipeline [decontaminates against terminal-bench](rl_data.md) (n-gram overlap).
Harvesting the terminal-bench *eval* trajectories and then re-evaluating on
terminal-bench is training on the test set — it inflates the number without
improving the model. `rejection_sample_sft.py` refuses inputs whose path looks
like a terminal-bench eval (`--allow-eval-source` overrides, only ever
legitimate for building unit-test fixtures). Keep step 1 pointed at the
training corpus.

## Commands

```bash
# [1] rollouts (cluster; serves tmax-9b via vLLM over the training corpus)
sbatch rl_data/scripts/generate_solutions/run_rejection_sampling_rollouts.sh
#   knobs (env): VLLM_MODEL, NUM_SOLUTIONS (default 16), SOLUTION_TEMPERATURE
#   (1.0), TASKS_DIR (the rl_data corpus), SAMPLE_SIZE (0 = full).

# [2] harvest + convert  (local, no GPU)
uv run python -m rl_data.rejection_sample_sft \
    rl_data/output/tasks_skill_tax_20260401_10k \
    --out rl_data/output/rejsample_sft.jsonl \
    --max-per-task 4 --max-tokens 32000 \
    --push-to-hub <you>/tmax-rejsample-sft
#   prints a harvest report: passing / kept / dropped-{format,submit,dup,cap,len}.

# [3] SFT  (cluster)
DATASET=<you>/tmax-rejsample-sft \
    bash training/open-instruct/scripts/tmax/SFT/sft_qwen35_9b_rejsample.sh

# [4] eval on terminal-bench, compare to the 44/89 (49.4%) baseline
bash scripts/beaker/launch_tmax9b_tb21_replicate.sh   # point --revision at the SFTed ckpt
```

## Harvest filtering (`rejection_sample_sft.py`)

- **`reward == 1` only** — the verifier is the label; nothing else is kept.
- **drop format-errors** — a trajectory with any malformed / no-tool-call
  assistant turn is discarded (training on the failure mode teaches it).
- **require a clean submit** — the final action must issue
  `COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT` (a genuine completion, not a
  coincidental stop).
- **dedup** on the assistant action-sequence, so identical rollouts collapse.
- **per-task cap** (`--max-per-task`, default 4), keeping the **shortest**
  passing trajectories — cheapest correct demonstrations, least flailing — so
  easy high-yield tasks don't dominate the mixture.
- **length cap** (`--max-tokens`, default 32000 ≈ SFT `max_seq_length`).

Output rows are `{messages, tools, dataset, id}`: the sanitized message log
(harness/provider-internal fields stripped, `reasoning_content` preserved for
the interleaved-reasoning chat template) plus the bash `tools` column
open-instruct renders against.

## What is validated vs. what needs the cluster

The harvest/convert tool is unit-tested
([`tests/rejection_sampling/`](../tests/rejection_sampling)) and validated
end-to-end on real production trajectories. Steps 1 and 3 are GPU cluster jobs
(vLLM rollout generation; multi-node SFT) — launch them as usual.

## Honest expectations

Rejection-sampling SFT reliably lifts pass@1 toward the *current* pass@5 ceiling
but cannot exceed what the model can already do at all — the ~45 terminal-bench
tasks that never solve at any k are a capability gap that needs RL or new
training data, not distillation of existing successes. Realistic target: move
pass@1 from ~29% up toward the high-30s/40s as the reliable-solve set grows,
with diminishing returns across rounds. Measure every round against the matched
`replicate` baseline, and re-establish the noise floor (±~5 tasks on pass@5)
with a couple of baseline replicates before declaring a win.
