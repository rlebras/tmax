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

## Which data (canonical corpus)

Roll out on the tmax **training** corpus, which the pipeline decontaminates
against terminal-bench — never on terminal-bench itself. The published corpus:

- **`tmax/TMax-15K-Harbor`** — 15k self-contained tasks (10k legacy + 5k
  intricate multimodal), each a Harbor environment with a programmatic
  verifier. Roll out with `harbor run -d`, the **same harness path as the
  eval**, so harvested trajectories match the deployment distribution. This is
  the recommended source.
- **`allenai/tmax-15k-open-instruct`** — the same corpus in open-instruct
  format (what the RL scripts consume). Useful if you prefer the RL data path.

The repo already warm-starts SFT this exact way (rl_data README: "roll out 8
trajectories per environment … the successful (pass) trajectories form the SFT
corpus"); this is that loop, made re-runnable and focused on the payoff band.

## The loop

```
[1] generate   run_rejection_sampling_rollouts.sh  (or `harbor run -d tmax/TMax-15K-Harbor`)
    tmax-9b (vLLM) × the training corpus, k=16 rollouts/task, temp 1.0
        │  harbor path:  jobs/<job>/<task>/{result.json, agent/trajectory.json}
        │  rl_data path: <task>/solutions/<model>_vanillux_summary.json
        ▼
[2] harvest    rl_data/rejection_sample_sft.py   (reads BOTH layouts)
    per-task solve-rate focus (--max-solve-rate) · keep reward==1 · drop
    format-errors · require a clean submit · dedup · cap per task · length-cap
        │  →  SFT jsonl ({messages, tools, dataset, id})  (optionally --push-to-hub)
        ▼
[3] train      sft_qwen35_9b_rejsample.sh
    open_instruct/finetune.py on the harvested data (base hamishivi/Qwen3.5-9B)
        ▼
[4] eval       scripts/beaker/launch_tmax9b_tb21_replicate.sh (terminal-bench, HELD OUT)
```

Iterate: the SFTed checkpoint becomes the model in step 1 of the next round.

## Focus on the instances where rejection sampling is relevant

Rejection-sampling SFT only helps on tasks the model solves **sometimes but
not reliably** (`0 < solve_rate < 1`). Always-solved tasks teach nothing (and,
being easy, flood the mixture with short redundant trajectories); never-solved
tasks have nothing to harvest. The harvester computes each task's solve-rate
across its attempts, prints the distribution, and `--max-solve-rate` drops
already-reliable tasks.

To make this concrete, harvesting the tmax-9b terminal-bench baseline as a
*format* example (89 tasks) showed the payoff band clearly: 44 tasks solvable,
but 12 already solved on every attempt (redundant) and 16 barely solved
(`<0.25` — the highest-value targets). Only ~32 tasks are in the band where
rejection sampling actually moves pass@1. `--max-solve-rate 0.8` keeps exactly
those. At rollout time, the complement is to spend MORE samples on the flaky
tasks (they are where each extra rollout has the best chance of yielding a new
harvestable success), rather than uniformly across the corpus.

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

# [2] harvest + convert  (local, no GPU) — reads harbor jobs/ dirs OR rl_data summaries
uv run python -m rl_data.rejection_sample_sft \
    jobs/<rollout-job>              # or rl_data/output/<corpus> \
    --out rl_data/output/rejsample_sft.jsonl \
    --max-per-task 4 --max-tokens 32000 --max-solve-rate 0.8 \
    --push-to-hub <you>/tmax-rejsample-sft
#   prints a harvest report + the per-task solve-rate distribution
#   (passing / kept / dropped-{format,submit,dup,cap,len,solve_rate}).

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

## Teacher distillation (27B → 9B) — the higher-value variant

Self-rejection-sampling can only reinforce what the 9B already reaches; it
can't add capability. Distilling from a stronger teacher can. On terminal-bench
the **27B's pass@1 is 44.9% vs the 9B's 29.0%** — a large, real edge — so
importing the 27B's verified successes is the higher-ceiling move (realistic
9B target: mid-to-high 30s, bounded below 44.9% by the capacity gap).

The selection rule **inverts** relative to self mode. There you drop
already-reliable tasks (`--max-solve-rate`). Here you keep tasks by the
**teacher−student gap** — where the *student* is weak, including tasks the 9B
*never* solves (those are the pure capability imports self-sampling can't
touch). Rejection filtering still applies to the teacher (keep only its
verifier-passing trajectories — don't distill the 27B's errors).

```bash
# [1a] student solve-rates: roll out the 9B on the corpus (cheap)
bash scripts/beaker/launch_tmax9b_rejsample_rollouts.sh          # N_ATTEMPTS>=4

# [1b] teacher successes: roll out the 27B on the same corpus (k=4, TP=2)
bash scripts/beaker/launch_tmax27b_rejsample_rollouts.sh

# [2] harvest in GAP mode: teacher inputs + student rollouts
uv run python -m rl_data.rejection_sample_sft <27b jobs/ dir> \
    --student-rollouts <9b jobs/ dir> \
    --max-student-solve-rate 0.8 \   # keep tasks the 9B solves <=80% (incl. never)
    --out rl_data/output/distill_sft.jsonl --push-to-hub <you>/tmax-distill-sft
```

**How many teacher rollouts?** Not k=1 (even at 44.9% pass@1): pass@k > pass@1,
so extra samples recover the harder tasks the 27B solves only sometimes — the
ones the 9B most needs — and give a pool to pick a clean demonstration from.
Not k=16 either (wasteful on a reliable teacher). Use k≈4, or the adaptive
two-pass version: cheap k=2 everywhere, then top up k≈6–8 *only* on the tasks
still unsolved (spends the expensive 27B compute where coverage is incomplete,
which is also where the 9B is weakest). Keep 1–3 demos per task regardless.

## Honest expectations

Rejection-sampling SFT reliably lifts pass@1 toward the *current* pass@5 ceiling
but cannot exceed what the model can already do at all — the ~45 terminal-bench
tasks that never solve at any k are a capability gap that needs RL or new
training data, not distillation of existing successes. Realistic target: move
pass@1 from ~29% up toward the high-30s/40s as the reliable-solve set grows,
with diminishing returns across rounds. Measure every round against the matched
`replicate` baseline, and re-establish the noise floor (±~5 tasks on pass@5)
with a couple of baseline replicates before declaring a win.
