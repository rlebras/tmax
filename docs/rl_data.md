# `rl_data/` — Terminal-environment generation for RL/SFT training

This package **synthesizes brand-new Terminal/CLI tasks** (interactive sandbox
environments) and the machinery to **solve, score, analyze, decontaminate, and
publish** them. The output is a corpus of self-contained, verifiable terminal
tasks used as training data for agentic RL (and SFT) — each task is a
containerized scenario plus an automated pass/fail verifier.

> **One-line mental model:** an LLM (Gemini by default) invents a task + its
> verifier; we bake it into an Apptainer container; we run *other* LLM agents
> against it to measure difficulty (pass@k); then we filter, dedup vs. eval
> benchmarks, and upload to Hugging Face.

The LLM backend for *generation* is **Gemini 3.1 Pro** (`DEFAULT_MODEL` in
[`__init__.py`](../rl_data/__init__.py)), driven through `litellm`. The agents that
*solve* tasks can be any model (Gemini, or a locally-served Qwen via vLLM).

---

## Table of contents

1. [The end-to-end pipeline](#1-the-end-to-end-pipeline)
2. [Anatomy of a generated task](#2-anatomy-of-a-generated-task)
3. [Stage 1 — generate tasks (`generate_tasks.py`)](#3-stage-1--generate-tasks)
4. [Stage 2 — generate solutions / pass@k (`generate_solutions.py`)](#4-stage-2--generate-solutions--passk)
5. [Stage 3 — analyze (`analyze.py`, `classify_difficulty.py`)](#5-stage-3--analyze)
6. [Stage 4 — combine, decontaminate, upload](#6-stage-4--combine-decontaminate-upload)
7. [The container base images (`containers/`)](#7-the-container-base-images)
8. [The `comparison/` branch — benchmarking vs external datasets](#8-the-comparison-branch)
9. [Key concepts & glossary](#9-key-concepts--glossary)
10. [Which script do I run? (cheat sheet)](#10-which-script-do-i-run-cheat-sheet)
11. [End-to-end runbook (full command sequence)](#11-end-to-end-runbook-full-command-sequence)
12. [Cost estimation & smoke tests](#12-cost-estimation--smoke-tests)
13. [Gotchas & operational notes](#13-gotchas--operational-notes)

---

## 1. The end-to-end pipeline

```
generate_tasks.py  ──►  task_*/  (task.json, container.def, test_*.py, fixtures/)
       │                   │
       │                   ▼  containers/base_*.sif   (shared runtime substrate)
generate_solutions.py ──►  task_*/solutions/<MODEL>[_<harness>]_summary.json   (pass@k)
       │
       ├─► analyze.py             (tables + pie/bar/pass@k plots)
       ├─► classify_difficulty.py (Frontier / Advanced+ / Advanced / Core / easy tiers)
       ├─► convert_to_harbor.py   (export to Harbor format for the eval that feeds the above)
       ├─► combine_corpora.py     (legacy + v2  →  one symlink view; balanced-SFT or union-RL)
       ├─► decontamination/cli.py (n-gram overlap vs terminal-bench / openthoughts-tblite)
       └─► upload_to_hf.py        (raw tree + parquet; container.sif excluded)

         comparison/   ── side branch: ingest external datasets, classify into our
                          taxonomy, solve with the same harness, emit head-to-head report
```

Each stage **checkpoints to disk** and is resumable; stages are decoupled so you
can re-run analysis/upload without regenerating tasks.

---

## 2. Anatomy of a generated task

Every task is a directory `task_<idx>_<hash>/`:

| File | Meaning |
|---|---|
| `task.json` | Metadata: `description` (what the agent is told), `truth` (hidden ground truth), taxonomy axes (`domain`, `skill_type`, `primitive_skills`, `task_complexity`, `command_complexity`, `scenario`, `language`), and the **v2 axes** `verifier_kind`, `fixture_kind`, `corpus_kind`, `base_image`. |
| `container.def` | Apptainer recipe that builds the task's starting filesystem state. |
| `container.sif` | Built image (excluded from upload; rebuilt from `.def` at train time). |
| `setup.sh` | The **task-specific delta** of `%post` — lets the task run on a *shared* base SIF instead of a 200 MB+ per-task SIF (see §7). |
| `test_initial_state.py` | pytest verifying the **starting** state was set up correctly (precondition gate; checks no output files). |
| `test_final_state.py` | pytest verifying the task is **solved** — this is the **reward signal**. Its exit code is the binary pass/fail. |
| `fixtures/` | (v2 only) materialized non-text artifacts — images, audio, video, stripped binaries, vendored packages, multi-service compose — baked into the SIF via a `%files` section. |
| `solutions/<MODEL>[_<harness>]_summary.json` | Written by Stage 2: per-attempt message threads, rewards, and the `pass_at_k` map. |
| `task_summary.txt` | Human-readable dump of the task. |

### `truth` vs the two test files (the verification core)

- **`truth`** — privileged, hidden ground truth declared by the template LLM
  inside `<truth>…</truth>`. **Never shown to the solving agent.** It tells the
  *verifier author* the exact expected end state (file contents, metric targets,
  ports, oracle algorithm, fixture answer).
- **`test_initial_state.py`** — confirms setup produced the right *starting*
  state. Run by `env.run_initial_tests()` as a gate.
- **`test_final_state.py`** — the reward function. Run by
  `env.run_final_tests()` inside the container after the agent stops; pytest's
  exit code is the reward.

> **Known soft spot:** `truth` and `test_final_state.py` are both LLM-generated
> text and are not externally re-validated by executing a reference solution.
> The prompts mitigate this by instructing the verifier author to *re-derive*
> expected values in stdlib rather than trust opaque literals (see the docstring
> at the top of [`generate_tasks.py`](../rl_data/generate_tasks.py)).

---

## 3. Stage 1 — generate tasks

**Module:** [`generate_tasks.py`](../rl_data/generate_tasks.py) → orchestrates the
[`generator/`](../rl_data/generator/) package. **Launchers:**
[`scripts/generate_tasks/`](../rl_data/scripts/generate_tasks/).

Four sub-stages, checkpointed in two phases:

```
1. task templates      generator/task_template_gen.py     → {description, truth, + metadata}
2. initial-state test  generator/initial_state_test_gen.py → test_initial_state.py
3. final-state test    generator/completion_test_gen.py    → test_final_state.py
   ── Phase-1 checkpoint: _intermediates.jsonl (stages 1–3 are LLM-only, fast) ──
4. container def        generator/apptainer_def_gen.py      → container.def + build/test .sif
   (+ fixtures)         generator/fixture_gen.py            → fixtures/ + %files injection
   ── Phase-2 checkpoint: _stage4_done.jsonl (CPU-heavy build/test; streaming saves) ──
```

If `_intermediates.jsonl` already exists, stages 1–3 are skipped on restart;
completed stage-4 items are skipped via `_stage4_done.jsonl`.

### The taxonomy & sampler ([`generator/task_template_gen.py`](../rl_data/generator/task_template_gen.py))

A task is a sampled point in a multi-axis design space:

- **`SKILL_TAXONOMY`** — a 3-level tree: **9 domains** (`security`,
  `software_engineering`, `file_operations`, `data_querying`, `data_science`,
  `debugging`, `scientific_computing`, `data_processing`,
  `system_administration`) → skill types → 3–5 primitive skills.
- **`TASK_COMPLEXITY`** — `short` / `moderate` / `complex` (legacy) + **`intricate`**
  (v2; "30–60 commands" — calibrated toward Terminal-Bench 2.0's ~40-turn mean).
- **`COMMAND_COMPLEXITY`** — `bash-only` / `bash and code` / `bash, code, and system services`.
- **`DOMAIN_SCENARIOS`** (personas), **`REAL_SOFTWARE_ANCHORS`** (concrete
  buggy-scenario seeds, injected ~35% of the time), **`TASK_LANGUAGES`**
  (weighted: Python 0.35, C/Bash 0.15, …).
- **v2 axes:**
  - **`verifier_kind`** — `exact_text` (legacy) + `metric_threshold`,
    `adversarial_corpus`, `fuzz_equivalence`, `multi_protocol`.
  - **`fixture_kind`** — `text_only` (legacy) + `image`, `audio`, `video`,
    `stripped_binary`, `vendored_package`, `multi_service_compose`.
  - **`corpus_kind`** — `legacy` / `sft_v2` / `rl_v2`.

**The "bucket-upweight" sampler** controls how aggressively v2 axes fire,
parameterized by `corpus_kind` via per-axis multipliers `M`:

| corpus_kind | task_complexity | verifier_kind | fixture_kind |
|---|---|---|---|
| `legacy` | (off — always legacy values) | (off) | (off) |
| `sft_v2` | M=2.0 | M=2.0 | M=2.0 |
| `rl_v2` | M=3.0 | M=∞ | M=∞ |

`M` upweights the "new" bucket relative to the single legacy value:
`P(new) = M/(M+1)`. `M=∞` means the legacy value is never emitted. The intent:
concatenating a pure-legacy 10k corpus with a v2 5k corpus yields a roughly
balanced 15k. Any task that ends up non-legacy on *any* axis (or is `intricate`)
is a **v2 task** and gets `base_image="intricate"` — a routing hint consumed at
solve time (§7).

### Tests, defs, fixtures

- [`initial_state_test_gen.py`](../rl_data/generator/initial_state_test_gen.py) /
  [`completion_test_gen.py`](../rl_data/generator/completion_test_gen.py) — batched LLM
  calls; each result is `compile()`-checked and dropped on failure. For v2,
  the final-test prompt relaxes "stdlib only" to a per-`verifier_kind` import
  allow-list (numpy/scipy/PIL/torch/bs4/requests/…) that `base_intricate.sif`
  preinstalls.
- [`apptainer_def_gen.py`](../rl_data/generator/apptainer_def_gen.py) — generates
  `container.def`, builds the `.sif`, runs `test_initial_state.py` inside it, and
  **self-corrects** by re-prompting failures with the build/test error
  (`max_def_retries`). Crucially, `parse_def_to_delta()` strips the standard
  base preamble from `%post`, leaving only the task-specific delta, saved as
  `setup.sh`.
- [`fixture_gen.py`](../rl_data/generator/fixture_gen.py) — for non-text `fixture_kind`,
  deterministically materializes real artifact **bytes on the host** (so no
  internet is needed at solve time), writes a hidden-answer sidecar, and returns
  `(host_path, container_path)` pairs that
  [`container_def_patch.py`](../rl_data/generator/container_def_patch.py) injects as a
  `%files` block. Determinism comes from a sha256-based seed
  (`fixture_seed_for_task`).

### The runtime environment ([`generator/env.py`](../rl_data/generator/env.py))

`InteractiveContainerEnvironment` runs one task inside Apptainer over a
**persistent PTY login shell** (so `cd`/`export` persist across commands). Key
pieces:

- **`exec(command)`** — the agent's action primitive; wraps each command so the
  shell emits a unique `{marker}:{exit_code}`, strips ANSI, and recovers a hung
  shell (double Ctrl-C / restart).
- **`_resolve_runtime_sif()`** — the SIF-selection core. Uses the per-task
  `.sif` if present; otherwise, with `base_sifs_dir` set, reads `task.json` and
  routes by `base_image` (**`intricate` → `base_intricate.sif`**, taking
  precedence over `domain`), then applies the `setup.sh` delta on top of the
  shared base. This is the ~200 MB/task disk saving that makes large corpora
  practical.
- **`run_initial_tests()` / `run_final_tests()`** — write the test files onto a
  bind-mounted writable `/home/user` and run `pytest -q` inside the container.

### Solver harnesses (used by Stage 2)

- [`sample_solutions.py`](../rl_data/generator/sample_solutions.py) — the **legacy `bash`
  harness**: native tool-calling agent (single `bash` tool, same system prompt
  as tmax SFT), `max_actions=16`, computes pass@k.
- [`vanillux_solver.py`](../rl_data/generator/vanillux_solver.py) +
  [`vanillux_prompts.yaml`](../rl_data/generator/vanillux_prompts.yaml) — the **`vanillux`
  harness**: a mini-swe-agent-style bash agent with `max_actions=64`,
  head/tail observation truncation, and explicit format-error recovery. Same
  sandbox and same summary schema (plus `"harness": "vanillux"`).

---

## 4. Stage 2 — generate solutions / pass@k

**Module:** [`generate_solutions.py`](../rl_data/generate_solutions.py). **Launchers:**
[`scripts/generate_solutions/`](../rl_data/scripts/generate_solutions/).

Runs an LLM agent against each task **N times** (`--num-solutions`), grades each
attempt with `test_final_state.py`, and writes a per-task
`solutions/<MODEL>[_<harness>][_thinking]_summary.json` containing the message
threads, per-attempt reward, and the `pass_at_k` map (for every k in `1..N`).
This is how task **difficulty and quality** are measured.

**Concurrency model (important):**
- `--workers` = number of **tasks** processed in parallel.
- `--num-pool-workers` = number of **solution attempts within a task** in
  parallel (≥ `--num-solutions`).
- Total live containers ≈ `workers × num_solutions` — size to your CPU/RAM.

**`--base-sifs-dir rl_data/containers` is critical:** it uses the pre-built base
SIFs + per-task `setup.sh` deltas instead of building a fresh SIF per task
(which would thunder-herd Docker Hub / apt; the driver also `apptainer pull`s each
unique base image once via `_prepull_base_images` to dodge Docker Hub's
unauthenticated rate limit). The module's *default* `--command-timeout` is 30s,
but the launcher scripts raise it to ~600s because v2 `setup.sh` (apt + pip +
compile) under parallelism routinely exceeded the old caps.

**Harness / mode selection** affects the summary filename (via
`_summary_basename`) so all four configs coexist in one corpus without
clobbering each other:
- `bash` → `<MODEL>_summary.json`
- `bash + --thinking` → `<MODEL>_thinking_summary.json`
- `vanillux` → `<MODEL>_vanillux_summary.json`
- `vanillux + --thinking` → `<MODEL>_vanillux_thinking_summary.json`

`--thinking` is a **naming knob only** — the actual reasoning-trace switch lives
in `LITELLM_EXTRA_BODY_JSON` (`{"chat_template_kwargs": {"enable_thinking": true}}`).
Runs resume by skipping any task that already has the target summary unless
`--force-rerun` is passed.

**Teachers:** Gemini-3-flash (via API) or a **locally-served Qwen via vLLM**.
For the latter, start the server with
[`launch_vllm.sh`](../rl_data/scripts/generate_solutions/launch_vllm.sh) (or the inline
`_vllm_local.sh` sourced by the skill_tax scripts) and pre-pull weights with
[`predownload_model.sh`](../rl_data/scripts/predownload_model.sh).

---

## 5. Stage 3 — analyze

### `analyze.py` — corpus stats & plots

**Module:** [`analyze.py`](../rl_data/analyze.py). **Launcher:**
[`scripts/analyze/run_analyze.sh`](../rl_data/scripts/analyze/run_analyze.sh).

Scans a solved task dir and emits (to `<tasks-dir>/analysis/`):
- **stdout tables** — counts, mean pass@1 + a pass@k ladder (default `4, 8`),
  avg turns, token totals (real API `usage` when available, else word-count
  estimate), reasoning tokens, and estimated solution + task-gen cost; plus
  aggregate breakdowns by domain / complexity / command_complexity (and the v2
  axes when they vary).
- **distribution pies** — `dist_domain.png`, `dist_task_complexity.png`, … and
  v2 variants.
- **quality bars + curves** — `quality_pass{1,4,8}_by_*.png`,
  `quality_pass_at_k.png`, and `quality_num_success_distribution.png` (an
  X-of-N histogram that visualizes the easy/hard split: red bar at 0/N, green at
  N/N).

It is harness-aware (vanillux summaries plot to an isolated subtree) and reads
the full pass@k ladder (`pass_at_k_full`) per task.

### `classify_difficulty.py` — eval-time difficulty tiers

**Script:** [`scripts/analyze/classify_difficulty.py`](../rl_data/scripts/analyze/classify_difficulty.py).

Computes the **Frontier / Advanced+ / Advanced / Core / easy** tiers from
**Harbor evaluation jobs** (i.e. *after* you run an agent against the Harbor
export — not from the generation-time summaries), using max/min accuracy across
one or two models:

| Tier | Rule (`min`/`max` accuracy) | Target share |
|---|---|---|
| `frontier` | `max_acc < 0.40` | 10–20% |
| `advanced_plus` | `min_acc < 0.40` (excl. frontier) | 30–40% |
| `advanced` | `0.40 ≤ min_acc < 0.60` | 20–30% |
| `core` | `0.60 ≤ min_acc < 0.80` | 20–30% |
| `easy` | `min_acc ≥ 0.80` | (excluded) |

Emits `difficulty_report.{json,md}` with per-domain / per-skill-type breakdowns.

> ⚠️ **Two difficulty vocabularies** coexist — don't conflate them:
> generation-time **complexity** (`short`/`moderate`/`complex`/`intricate`, in
> `task.json`) vs. eval-time **tiers** (`Frontier`/…/`easy`, from
> `classify_difficulty.py`).

### Other analysis helpers

- [`convert_to_harbor.py`](../rl_data/scripts/analyze/convert_to_harbor.py) — export the
  Apptainer task format to a **Harbor local dataset** (`instruction.md`,
  `task.toml`, `environment/Dockerfile`, `tests/`). This is what you eval to
  feed `classify_difficulty.py`. Handles both legacy (self-contained) and
  intricate (inlines/`FROM`s the intricate base, COPYs fixtures) tasks.
- [`peak_context.py`](../rl_data/scripts/analyze/peak_context.py) — computes peak-context
  token stats from trajectories (justified raising `VLLM_MAX_LEN` to 128K).

---

## 6. Stage 4 — combine, decontaminate, upload

### Combine — `combine_corpora.py`

**Script:** [`scripts/combine/combine_corpora.py`](../rl_data/scripts/combine/combine_corpora.py).
Merges a legacy and a v2 corpus into one `--out-dir` of **symlinks** (drop-in
root for analyze/upload), plus a `_combine_manifest.json`.

- **`balanced`** (SFT-oriented) — keep all non-intricate tasks from both, then
  down-sample v2 `intricate` to hit `--total`.
- **`union`** (RL-oriented) — symlink *every* task from both; fails fast on a
  basename collision.

### Decontaminate — `decontamination/`

**Module:** [`decontamination/cli.py`](../rl_data/decontamination/cli.py). **Launcher:**
[`scripts/decontamination/run_decontamination.sh`](../rl_data/scripts/decontamination/run_decontamination.sh).

Word-level **n-gram overlap** (default `n = 13, 8`) between generated task
**descriptions** and **eval-benchmark instructions** (`terminal-bench@2.0` and
`openthoughts-tblite@2.0`, downloaded via `harbor`). Reports, per
(benchmark, dataset, n), the fraction of generated tasks containing ≥1 matching
n-gram. Outputs `decontamination_table.md`, `decontamination_data.csv`,
`report.json`.

### Upload — `upload_to_hf.py`

**Module:** [`upload_to_hf.py`](../rl_data/upload_to_hf.py). **Launchers:**
[`scripts/upload/upload_data_to_hf.sh`](../rl_data/scripts/upload/upload_data_to_hf.sh) and
`upload_data_to_hf_verified.sh`.

Pushes the raw `task_*` tree + an `analysis/` dir + a consolidated
`data/train-00000-of-00001.parquet` (for the HF Dataset Viewer). **`container.sif`
is always excluded** (rebuilt from `.def` at train time). Modes: `compact`
(zip + parquet, fastest, default), `fast`, default (`upload_large_folder`,
resumable). `--verified-only` keeps only tasks with non-zero `pass_at_k`.

### Repair scripts — `scripts/repair/`

Fix v2 **fixtures** that were only written as a placeholder sentinel because the
task-gen host lacked a tool: `repair_stripped_binary_fixtures.py` (missing
`gcc`) and `repair_video_fixtures.py` (missing `ffmpeg`). They re-materialize
the fixture with the same deterministic seed and rewrite the `%files` block. The
`run_*_in_sif.sh` wrappers borrow `gcc`/`ffmpeg` from `base_intricate.sif` while
running the orchestration on the host. **Rebuild the `.sif` afterward.**

---

## 7. The container base images

**Dir:** [`containers/`](../rl_data/containers/). **Builder:**
[`containers/build_bases.sh`](../rl_data/containers/build_bases.sh).

Ten Apptainer `.def` files (all `Bootstrap: docker / From: ubuntu:22.04`):

- **9 per-domain bases** (`base_<domain>.def`) — each ships a common core
  (python3/pip, coreutils, curl/git, jq, sudo, pytest, a non-root `user`) plus
  domain tools (e.g. `software_engineering` adds gcc/g++/cmake/node/go/rust;
  `security` adds nmap/john/hashcat; `debugging` adds gdb/valgrind/strace).
- **`base_intricate.def`** — the v2 shared base (~3–4 GB). Pre-bakes the union of
  the SE toolchain + multimedia/RE tools (ffmpeg, tesseract, upx, binutils) +
  a broad Python stack (numpy/scipy/sklearn/pandas, Pillow/imageio, biopython,
  bs4/lxml, **CPU-only torch**) + the apt packages that v2 `setup.sh` files most
  often re-installed. Motivation: ~26% of agent commands were timing out
  re-running heavy `apt install` per (task, solution); baking them in turns those
  into ~1s no-ops.

`build_bases.sh` builds the 9 per-domain SIFs and self-verifies each (deleting on
failure). **`base_intricate.sif` is built separately** (the v2 task-gen scripts
and the smoke test do
`apptainer build base_intricate.sif base_intricate.def`).

At solve time, `env._resolve_runtime_sif()` picks the base SIF and layers the
per-task `setup.sh` delta — so you build ~10 base images once, not one per task.

---

## 8. The `comparison/` branch

**Dir:** [`comparison/`](../rl_data/comparison/). **Reference:**
[`scripts/comparison/COMPARISON.md`](../rl_data/scripts/comparison/COMPARISON.md).

Compares **our** corpus (the reference, displayed as "TMaxx (ours)") head-to-head
against external terminal-task datasets. Four stages:

1. **Ingest** ([`comparison/adapters/`](../rl_data/comparison/adapters/)) — each adapter
   pulls one external dataset and normalizes it into our canonical task layout
   (`task.json` + `container.def` + `test_*.py`):
   - `endless_terminals` (`obiwan96/endless-terminals`), `openthoughts_tb`
     (`open-thoughts/OpenThoughts-TB-dev`), `openthoughts_agent_rl`
     (`OpenThoughts-Agent-v1-RL`), `termigen` (`ucsb-mlsec/terminal-bench-env`,
     GitHub sparse-clone), `terminaltraj` (`m-a-p/TerminalTraj-5k`), `r2e_gym`
     (`hamishivi/agent-task-r2e-gym`), and `skill_tax` (our own, identity).
   - Run via `python -m rl_data.comparison.adapters.<name>` (flags:
     `--cache-dir --dst --limit --workers --skip-download --revision`).
2. **Classify** ([`taxonomy_classifier.py`](../rl_data/comparison/taxonomy_classifier.py))
   — an LLM maps each external *task* onto our 4-axis taxonomy, writing
   `classified_*` fields (native fields untouched). Separately,
   [`command_taxonomy.py`](../rl_data/comparison/command_taxonomy.py) is a *rule-based*
   classifier of bash one-liners in solution traces (16 categories) — i.e. *what
   the agent did*, vs. the LLM classifier's *what the task is*.
3. **Solve** — same `generate_solutions` harness as our own corpus.
4. **Compare** (`python -m rl_data.comparison.cli`) — six analysis modules
   (`difficulty`, `command_mix`, `composition`, `diversity`, `realism`,
   `verifier`) emit a paper-ready `main/` (figures + `summary_table.md` +
   `paper_snippets.md`) and a deep-dive `appendix/`, plus `report.json`.
   Figure aesthetics live in [`styles.py`](../rl_data/comparison/styles.py);
   [`preview_palettes.py`](../rl_data/comparison/preview_palettes.py) re-renders the
   headline composition figure under different palettes without rerunning the
   pipeline.

The one-shot driver is
[`scripts/comparison/run_comparison.sh`](../rl_data/scripts/comparison/run_comparison.sh)
(ingest → classify → solve → compare).

---

## 9. Key concepts & glossary

| Term | Meaning |
|---|---|
| **task** | One containerized terminal scenario + its verifier. |
| **`truth`** | Hidden ground truth (never shown to the solver); tells the verifier the expected end state. |
| **`test_initial_state.py`** | Precondition gate — was setup correct? |
| **`test_final_state.py`** | Reward function — is the task solved? pytest exit code = reward. |
| **pass@k** | Fraction of tasks an agent solves within k attempts; the primary quality/difficulty metric. |
| **harness** | The solver agent loop: `bash` (legacy, 16 actions) or `vanillux` (mini-swe-agent style, 64 actions). |
| **corpus_kind** | `legacy` (pre-v2, byte-identical), `sft_v2`, `rl_v2` — controls the bucket-upweight sampler. |
| **verifier_kind / fixture_kind** | v2 axes: *how* the task is checked / *what artifact* it ships. |
| **base_image** | Routing hint; `intricate` → run on the shared `base_intricate.sif`. |
| **setup.sh** | Per-task `%post` delta applied on top of a shared base SIF. |
| **complexity** (gen-time) | `short`/`moderate`/`complex`/`intricate` (in `task.json`). |
| **tier** (eval-time) | `Frontier`/`Advanced+`/`Advanced`/`Core`/`easy` (from `classify_difficulty.py`). |

---

## 10. Which script do I run? (cheat sheet)

**Generate tasks** — [`scripts/generate_tasks/`](../rl_data/scripts/generate_tasks/):

| Script | Tasks | corpus_kind | Use for |
|---|---|---|---|
| `run_generate_tasks.sh` | 10 | legacy | toy / template |
| `run_generate_tasks_1k.sh` | ~1k | legacy | small dev corpus |
| **`run_generate_tasks_10k.sh`** | 10k | legacy | **production 10k corpus** (`tasks_skill_tax_20260401_10k`) |
| `run_generate_tasks_sft_v2_1k.sh` | ~1k | sft_v2 | v2 SFT (needs `base_intricate.sif`) |
| `run_generate_tasks_rl_v2_5k.sh` | ~5k | rl_v2 | v2 RL corpus (needs gcc+ffmpeg on host for fixtures) |

**Generate solutions** — [`scripts/generate_solutions/`](../rl_data/scripts/generate_solutions/):

| Script | Corpus | Teacher | Harness | Use for |
|---|---|---|---|---|
| `run_generate_solutions.sh` | toy | gemini-3-flash | bash | smoke |
| **`run_generate_solutions_10k_gemini.sh`** | 10k | gemini-3-flash | bash | **production difficulty baseline** |
| `run_generate_solutions_skill_tax_10k.sh` | 10k | local Qwen3.5-9B (vLLM) | bash | apples-to-apples vs Gemini |
| `..._skill_tax_combined_2.5k.sh` | combined 2.5k | local Qwen (vLLM) | vanillux | **SFT trajectory production** |
| `..._skill_tax_combined_2.5k_thinking.sh` | combined 2.5k | Qwen | vanillux + thinking | SFT with `<think>` traces |
| `..._vanillux_*smoke.sh` | v2 2k | Qwen / Gemini | vanillux | A/B smoke tests |
| `launch_vllm.sh` | — | — | — | start a local vLLM server |

**Then:** `run_analyze.sh` → (`convert_to_harbor.py` + Harbor eval →
`classify_difficulty.py`) → `combine_corpora.py` → `run_decontamination.sh` →
`upload_data_to_hf.sh`.

> **Reference release corpora:** the production legacy corpus is
> `tasks_skill_tax_20260401_10k`; the RL release is the `union` of legacy-10k +
> v2-5k (`tasks_skill_tax_combined_20260506_legacy10k_new5k`), the default HF
> upload target.

---

## 11. End-to-end runbook (full command sequence)

This is the concrete, ordered sequence to produce a full release corpus. All
commands run from the **repo root** (each script `cd`s there anyway). Scripts are
launched with `bash …` for interactive runs, or `sbatch …` on Slurm (the
`#SBATCH` headers are live). Paths below match the defaults baked into the
launchers — override the `*_DIR` / `MODEL` env vars at the top of each script (or
edit them) to point at your own corpus.

> **Two tracks.** The **legacy 10k** track (steps 0–3, 6–9) is the standalone
> production corpus. The **v2** track adds an intricate-task corpus (step 4) and
> merges it with legacy (step 5) to form the RL release. Do the legacy track
> first; layer v2 on top only if you want the combined release.

```bash
# ── 0. ONE-TIME: build the base container images (the runtime substrate) ──
bash rl_data/containers/build_bases.sh                 # 9 per-domain base_*.sif
apptainer build rl_data/containers/base_intricate.sif \
                 rl_data/containers/base_intricate.def  # v2 shared base (only needed for v2)

# ── 1. (optional) project cost before committing GPU/$$ ──
bash rl_data/scripts/analyze/estimate_cost.sh

# ── 2. GENERATE TASKS — legacy 10k corpus ──
#     out: rl_data/output/tasks_skill_tax_20260401_10k/  (task_*/ dirs + checkpoints)
bash rl_data/scripts/generate_tasks/run_generate_tasks_10k.sh

# ── 3. GENERATE SOLUTIONS — measure pass@k (difficulty) on the 10k ──
#     writes solutions/<MODEL>_summary.json into each task dir
bash rl_data/scripts/generate_solutions/run_generate_solutions_10k_gemini.sh
#     (start a local vLLM server first if solving with a local model:
#      bash rl_data/scripts/generate_solutions/launch_vllm.sh)

# ── 4. (v2 track) GENERATE the intricate v2 corpus + its solutions ──
#     out: rl_data/output/tasks_skill_tax_v2_20260506_5k/  (requests 5500 → ~5k survive)
bash rl_data/scripts/generate_tasks/run_generate_tasks_rl_v2_5k.sh
#     If the gen host lacked gcc/ffmpeg, repair the sentinel fixtures (pass the v2
#     corpus dir; tools are borrowed from base_intricate.sif), then rebuild SIFs & solve:
bash rl_data/scripts/repair/run_repair_stripped_binary_in_sif.sh \
    --corpus-dir rl_data/output/tasks_skill_tax_v2_20260506_5k
bash rl_data/scripts/repair/run_repair_video_fixtures_in_sif.sh \
    --corpus-dir rl_data/output/tasks_skill_tax_v2_20260506_5k
bash rl_data/scripts/generate_solutions/run_generate_solutions_skill_tax_combined_2.5k.sh

# ── 5. (v2 track) COMBINE legacy + v2 into the RL release (union of all tasks) ──
uv run python -m rl_data.scripts.combine.combine_corpora \
    --mode union \
    --legacy-dir rl_data/output/tasks_skill_tax_20260401_10k \
    --v2-dir     rl_data/output/tasks_skill_tax_v2_20260506_5k \
    --out-dir    rl_data/output/tasks_skill_tax_combined_20260506_legacy10k_new5k

# ── 6. ANALYZE — corpus stats + difficulty/quality plots (edit TASKS_DIR inside) ──
bash rl_data/scripts/analyze/run_analyze.sh

# ── 7. (optional) DIFFICULTY TIERS via a real Harbor eval ──
uv run python rl_data/scripts/analyze/convert_to_harbor.py \
    --src rl_data/output/<corpus> --dst rl_data/output/<corpus>_harbor
#   …run your Harbor agent eval against <corpus>_harbor to get job result.json…
uv run python rl_data/scripts/analyze/classify_difficulty.py --job <harbor_job_dir>

# ── 8. DECONTAMINATE — n-gram overlap vs terminal-bench / openthoughts-tblite ──
bash rl_data/scripts/decontamination/run_decontamination.sh

# ── 9. UPLOAD to Hugging Face (container.sif excluded; parquet built for the viewer) ──
bash rl_data/scripts/upload/upload_data_to_hf.sh                 # all tasks
bash rl_data/scripts/upload/upload_data_to_hf_verified.sh        # only pass@k > 0
```

**Dependencies / ordering rules:**

- Step **0 must precede** steps 3, 4, 8 (solving and `convert_to_harbor` need the
  base SIFs; decontamination needs `harbor`). It is one-time — skip on reruns.
- Steps **2 → 3** and **4 → 5** are hard sequential (solutions need tasks; combine
  needs both corpora). Within each, generation is **resumable** — re-running picks
  up from the on-disk checkpoints / skips tasks that already have a summary.
- Step **6 (analyze)**, step **8 (decontaminate)**, and step **9 (upload)** are
  independent of each other and can run in any order once solutions exist.
- Step **7** is a separate eval loop (export → run an external Harbor eval →
  classify), not a pure-Python step; it's optional unless you need the
  Frontier/Advanced/Core tiers.

**Smaller/dev variants:** swap the `_10k` launchers for `_1k` (tasks) and
`_1k_gemini` (solutions) for a fast end-to-end dry run, or run
`bash rl_data/scripts/data_gen_v2_smoke_test.sh` to validate the whole v2 path on
5 tasks before committing to the 5k run.

**The `comparison/` branch is a side quest**, not part of this sequence — its own
one-shot driver is `bash rl_data/scripts/comparison/run_comparison.sh` (ingest →
classify → solve → compare), see §8.

---

## 12. Cost estimation & smoke tests

- [`estimate_cost.py`](../rl_data/estimate_cost.py) /
  [`scripts/analyze/estimate_cost.sh`](../rl_data/scripts/analyze/estimate_cost.sh) —
  projects LLM cost *before* a run: models the 4-step task-gen stage (with
  per-step pass rates) + the solution stage
  (`num_solutions × avg_turns` calls), printing per-component and per-task cost
  plus a scaling table (10 → 10 000 tasks).
- [`scripts/data_gen_v2_smoke_test.sh`](../rl_data/scripts/data_gen_v2_smoke_test.sh) — the
  pre-flight for the whole v2 pipeline: (A) asserts the v2 sampler
  distributions, (B) materializes every fixture kind, (C) builds + verifies
  `base_intricate.sif`, (D) runs a 5-task `sft_v2` end-to-end and checks the v2
  fields propagate through stage 4.

---

## 13. Gotchas & operational notes

- **Generation LLM is Gemini**, not Claude — `DEFAULT_MODEL` in `__init__.py`.
  Concurrency, retries, and `LITELLM_EXTRA_BODY_JSON` (for vLLM
  `chat_template_kwargs` like `enable_thinking`) are handled there.
- **`--base-sifs-dir rl_data/containers`** must be set when solving large
  corpora — otherwise every task rebuilds a full SIF and you thunder-herd Docker
  Hub / apt.
- **Fixtures are host-generated** at task-gen time. If the host lacks
  `gcc`/`ffmpeg`, you get sentinel files — fix with the `scripts/repair/`
  scripts (which borrow the tools from `base_intricate.sif`), then rebuild the
  `.sif`.
- **Combining then renaming a source corpus** leaves dangling symlinks that
  `generate_solutions` silently skips — re-run `combine_corpora.py --force`.
- **`container.sif` is never uploaded** — RL training rebuilds from
  `container.def`.
- **Two timeouts to know:** `--command-timeout` (per agent command, ~600s for
  v2 setup) and the delta-setup timeout (1800s) for applying `setup.sh`.
- **Verifier reliability** is the pipeline's known soft spot — see §2.
```
