# Running Evals

This document is the end-to-end guide to running agentic terminal/coding-task
evaluations in `tmax`. It covers the mental model, the launch paths
(Beaker and local/direct), the available datasets and agents, how models are
served and selected, where results land, and how to analyze them.

If you only want the Beaker-against-vLLM recipe, see
[`scripts/beaker/README.md`](../scripts/beaker/README.md). This document is the
superset.

---

## 1. Mental model

Every eval in this repo is the same shape, regardless of how it is launched:

```
            ┌─────────────────────────────────────────────────────┐
            │                   harbor run                         │
            │                                                      │
  dataset ──┼─▶ for each task:                                     │
  (tasks)   │     1. spin up an isolated SANDBOX (the --env)       │
            │     2. an AGENT drives a MODEL, issuing shell        │
            │        commands inside the sandbox                   │
            │     3. a VERIFIER runs the task's tests → REWARD     │
            │                                                      │
            └───────────────────────────┬─────────────────────────┘
                                         │
                                   jobs/<job-name>/   (per-trial results)
```

**harbor** is the test harness (the runner behind Terminal-Bench). `tmax`
depends on it as a package
([`pyproject.toml`](../pyproject.toml)) and adds custom **agents**, **launch
scripts**, and **analysis tooling** on top. We never modify harbor's source in
git — instead we patch its installed package at runtime where needed (see
[§8 Troubleshooting](#8-resuming-cleaning-and-troubleshooting)).

### Vocabulary

| Term | Meaning |
|---|---|
| **task** | One benchmark problem: a Dockerfile/environment + instruction + verifier tests. |
| **trial** | One `(task, attempt)` pair. Lives at `jobs/<job>/<trial_name>/`. |
| **attempt / `-k`** | How many independent trials to run per task (for pass@k). |
| **agent** | The scaffold that turns a model into a tool-using loop (e.g. `Vanillux2Agent`). |
| **environment / sandbox** | Where the agent's shell commands actually execute (`--env docker` or `--env daytona`). |
| **model** | The LLM being graded. Served either by a self-hosted vLLM or a hosted API. |
| **verifier** | Task-supplied tests that produce a `reward` (typically 0.0 / 1.0). |
| **job** | One full run over a dataset, named with `--job-name`, written to `jobs/<job-name>/`. |

### The two independent axes

The single most important thing to understand is that **"where the agent runs"**
and **"where the model runs"** are *independent* choices. Almost all confusion
about these scripts comes from conflating them.

1. **Sandbox backend** — harbor's `--env` flag:
   - `--env docker`: containers via `docker compose` on the local machine.
     In our Beaker jobs this is **podman** masquerading as Docker (see below).
   - `--env daytona`: each task runs in a fresh, fully-managed **cloud
     sandbox** from [Daytona](https://www.daytona.io/). Requires
     `DAYTONA_API_KEY`. No local container runtime needed.

2. **Model serving** — chosen by the `--model` string (a
   [litellm](https://docs.litellm.ai/) identifier):
   - **Self-hosted vLLM**: `--model hosted_vllm/<served-name>` plus
     `--agent-kwarg api_base=http://host:port/v1`. Used to grade your own
     trained checkpoints.
   - **Hosted API**: `--model anthropic/claude-sonnet-4-...`,
     `--model openai/gpt-4o`, `--model gemini/gemini-3-flash-preview`, etc.
     Requires the matching `*_API_KEY`.

Any sandbox can be paired with any model source. The launch scripts just bundle
common combinations.

---

## 2. Choosing a launch path

There are two ways to launch a run. Pick based on **where you are** and **what
you're evaluating**.

| Path | Script(s) | Sandbox | Typical model | Use when |
|---|---|---|---|---|
| **Beaker** | [`beaker_configs/launch_eval.sh`](../beaker_configs/launch_eval.sh) | podman (in-job) | self-hosted vLLM (your checkpoint) | Iterating on a checkpoint at AI2; you want model + sandboxes in one GPU job. |
| **Local / direct** | [`beaker_configs/run_eval_local.sh`](../beaker_configs/run_eval_local.sh) or `uv run harbor run ...` | local Docker (default) **or** Daytona | API or vLLM | Quick smoke tests on a dev VM with a real Docker daemon, or any direct harbor run. |

A useful rule of thumb that mirrors how this repo is actually used day to day:

- **Dev loop on Beaker** → Beaker + podman + in-job vLLM. Fast, self-contained,
  but the podman path carries a stack of compatibility patches and is therefore
  somewhat brittle.
- **Dev loop on a VM** → `run_eval_local.sh` against a real Docker daemon, or a
  direct `uv run harbor run`. Good for quickly smoke-testing an agent + model
  before committing to a full Beaker job.
- **Daytona** → clean, isolated cloud sandboxes (harbor `--env daytona`), useful
  when you have no local container runtime; requires a `DAYTONA_API_KEY`.

---

## 3. Path A — Beaker against a local vLLM

This is the canonical way to evaluate **your own trained checkpoint**: a single
Beaker task allocates N GPUs, serves the model with vLLM on `localhost`, brings
up podman + harbor, and runs a dataset against it. Results land on weka.

### Quickstart

```bash
./beaker_configs/launch_eval.sh allenai/open_instruct_dev \
    --revision sft_qwen3_4b_tmax_4node \
    --name sft-4b \
    --dataset terminal-bench@2.0
```

This submits an 8-GPU Gantry job that runs
[`scripts/beaker/run_eval_in_job.sh`](../scripts/beaker/run_eval_in_job.sh)
inside the container. Results end up at:

```
/weka/oe-adapt-default/$USER/tmax-eval/<job-name>/jobs/<job-name>/
```

> The quickstart uses the **default `Vanillux2Agent`** (see [§6](#6-agents)). A
> verified small-scale run on Qwen3.5-4B — the sample dataset on one GPU — looks
> like:
>
> ```bash
> DOCKER_PAT_SECRET=<user>_DOCKER_PAT ./beaker_configs/launch_eval.sh Qwen/Qwen3.5-4B \
>     --name qwen35-4b-vanillux2 \
>     --agent Vanillux2Agent:Vanillux2Agent \
>     --tool-call-parser qwen3_xml \
>     --model-provider openai \
>     --gpus 1 --dataset terminal-bench-sample@2.0 --max-model-len 32768 \
>     --workspace ai2/oe-agents
> ```
>
> `launch_eval.sh` references an `HF_TOKEN` and a `*_DOCKER_PAT` secret in the
> target workspace; set `DOCKER_PAT_SECRET` to your own (e.g.
> `<user>_DOCKER_PAT`). Keep a run small with `--n-tasks N` (first N tasks)
> and/or the `terminal-bench-sample@2.0` dataset. Verify the launched spec
> with `beaker experiment spec <EXP_ID> --format json`.

### What the in-job script does (in order)

From [`run_eval_in_job.sh`](../scripts/beaker/run_eval_in_job.sh):

1. Optionally `git clone` the tmax repo at a SHA (or use the Gantry-provided
   checkout).
2. `apt-get install` podman + helpers; install the Docker Compose v2 CLI plugin
   (harbor shells out to `docker compose`, which talks to podman's socket).
3. Write `/etc/containers/containers.conf` (host netns/ipc/uts, `userns=auto:size=65536`, crun, cgroups disabled).
4. `uv sync`, then **patch harbor's installed package** for podman compat
   (host networking in the compose file, world-writable bind-mount dirs, drop
   `--rmi all` from compose-down to avoid Docker Hub rate limits).
5. `source scripts/setup_podman_harbor.sh`: `mknod /dev/net/tun`, create the
   aardvark-dns dir, start `podman system service` on `/tmp/podman.sock`, export
   `DOCKER_HOST`.
6. Authenticate to Docker Hub for task-image pulls (from the `DOCKER_PAT` beaker
   secret): both `docker login` **and `podman login`**. The podman login is
   essential — harbor pulls via the podman socket, and podman reads creds from
   `containers/auth.json`, *not* `~/.docker/config.json`, so a `docker login`
   alone leaves pulls anonymous → the shared-IP unauthenticated rate cap
   (`toomanyrequests`) once jobs co-locate. If `--mirror-url` is set,
   `setup_dockerio_mirror` also points podman at a docker.io pull-through cache
   (co-located jobs then pull over the local network; authenticated docker.io is
   the fallback if the mirror is down).
7. Launch vLLM in the background (`uvx vllm==$VLLM_VERSION serve ...`) and poll
   `/v1/models` for up to 30 min.
8. `uv run harbor run --env docker --model hosted_vllm/$SERVED_MODEL_NAME --agent-kwarg api_base=...`.
9. `scripts/compute_stats.py` → `stats.txt` + `metrics.json`.
10. Copy `jobs/$JOB_NAME/` to `$RESULTS_DIR` on weka.

### Key flags

`./beaker_configs/launch_eval.sh <model_path> [options]` — full list with
`--help`. Most-used:

| Flag | Default | Notes |
|---|---|---|
| `<model_path>` | (required) | HF id or a weka path the image can read. |
| `--revision REV` | `main` | vLLM `--revision` + `--tokenizer-revision`. |
| `--name NAME` | `basename(model_path)` | vLLM `--served-model-name`; drives `JOB_NAME`. |
| `--gpus N` / `--tp N` / `--dp N` | `8` / gpus / `1` | GPU + parallelism. |
| `--dataset DS` | `terminal-bench@2.0` | See [§5 Datasets](#5-datasets). |
| `--dataset-path DIR` | unset | Run a local dataset dir via harbor `--path` (overrides `--dataset`). For off-registry sets like [TerminalBench 2.1](#off-registry-datasets-eg-terminalbench-21); dir must be under a mounted weka fs. |
| `--mirror-url HOST:PORT` | unset | docker.io pull-through mirror(s) for task-image pulls; avoids Docker Hub rate limits under co-located jobs. Falls back to authenticated docker.io if down. |
| `--agent IMPORT_PATH` | `Vanillux2Agent:Vanillux2Agent` | `module:Class` (e.g. `Vanillux2Agent:Vanillux2Agent`) **or** a harbor built-in name (`mini-swe-agent`, `swe-agent`, `terminus-2`, `oracle`). See [§6](#6-agents). |
| `--model-provider PROV` | per agent type | litellm provider prefix: `hosted_vllm` for `swe-agent`, `openai` otherwise. Use `openai` for Vanillux2Agent / built-ins. |
| `--n-concurrent N` | `8` | Parallel trials. |
| `--n-attempts N` | `1` | harbor `-k`. |
| `--n-tasks N` | all | Run only the first N tasks of the dataset (harbor `--n-tasks`). |
| `--harbor-env ENV` | `docker` | Harbor backend; `daytona` installs `harbor[daytona]` and wires `DAYTONA_API_KEY`. |
| `--override-cpus/-gpus/-memory-mb/-storage-mb`, `--*-timeout-multiplier` | unset | Pass-through Harbor resource/timeout overrides (`HARBOR_OVERRIDE_*` / `HARBOR_*_TIMEOUT_MULTIPLIER`). |
| `--docker-image IMG` / `--image IMG` | `hamishivi/tmax-eval-interactive` | Public Docker image vs. a Beaker image/ID (each clears the other). |
| `--max-model-len LEN` | unset | vLLM context length. Pass `32768` for Qwen3.5 so the 262k default KV cache fits one GPU and vLLM actually starts. |
| `--tool-call-parser P` | `hermes` | vLLM tool parser. **Use `qwen3_xml` for Qwen3.5 with structured-tool agents** (Vanillux2Agent); `hermes` silently drops its tool-calls. |
| `--reasoning-parser P` | none | vLLM reasoning parser. Use `qwen3` for Qwen3 reasoning models so `<think>…</think>` is split out of the visible response. |
| `--cluster` / `--workspace` / `--priority` / `--budget` | see script | Beaker placement. |
| `--results-dir DIR` | `/results` (Gantry → weka) | Where to copy `jobs/`. |
| `--repo-ref REF` | current HEAD SHA | **Must be pushed** to the remote. |

> ⚠️ Gantry submits a git SHA, not your working tree. Commit and push first, or
> pass `--repo-ref`. The script warns if the SHA isn't on a remote branch.

### Running the podman path locally

If you're on a node that already has podman, you can skip Beaker:

```bash
source scripts/setup_podman_harbor.sh    # runtime fixes + DOCKER_HOST
uv run harbor run --dataset terminal-bench@2.0 --agent oracle --env docker
```

Note: the sourced script does **not** apply the harbor source patches — those
must be reapplied after every `uv sync` (see
[`scripts/beaker/README.md`](../scripts/beaker/README.md#harbor-source-patches)).

### Mirroring the Beaker pipeline locally with real Docker (`run_eval_local.sh`)

[`beaker_configs/run_eval_local.sh`](../beaker_configs/run_eval_local.sh) is a
local smoke-test mirror of `run_eval_in_job.sh` for a dev VM that has a **real
Docker daemon** (not podman, not Daytona). It serves vLLM on one GPU, applies
the `network_mode: host` patch, and runs harbor `--env docker`. Defaults to
`mini-swe-agent` + `Qwen/Qwen3.5-4B`, 2 tasks.

```bash
./beaker_configs/run_eval_local.sh Qwen/Qwen3.5-4B --n-concurrent 1 --task fix-git
```

Two realities differ from Beaker (where podman runs *inside* the job container,
so `localhost` and the verifier writes both just work):

- **Sibling-container networking.** With the host Docker daemon, harbor's task
  containers are *siblings* of your dev-VM container; `network_mode: host` puts
  them on the **real host** netns, so `localhost:$VLLM_PORT` does **not** reach a
  vLLM running inside your container. For SWE-style **in-container** agents, point
  the agent at your container's docker-bridge IP (`hostname -i`, e.g.
  `172.17.0.4`) — `run_eval_local.sh` does this automatically. `Vanillux2Agent`
  runs host-side, so `localhost` works for it.
- **Verifier patches → rewards.** Producing `reward.txt` needs harbor's
  verifier/oracle/paths chmod patches (which `run_eval_in_job.sh` applies but
  `run_eval_local.sh` does not by default). Without them the agent still runs to
  completion but the trial ends in `RewardFileNotFoundError` (empty `verifier/`
  dir). Apply those three patches manually to confirm rewards locally, or just
  run on Beaker.

Prereqs: a running Docker daemon **and** the `docker compose` v2 CLI plugin
(harbor shells out to `docker compose`; on AI2 dev VMs
[`beaker-utils/interactive/set_dev_vm.sh`](https://github.com/shatu/beaker-utils)
installs it). `run_eval_local.sh` installs the plugin if it's missing.

**Docker Hub auth (local).** Task images come from Docker Hub, so the script
authenticates the same way as the Beaker path: it resolves a PAT from
`$DOCKER_PAT` (else reads the `DOCKER_PAT_SECRET` beaker secret via the beaker
CLI, default `shashankg_DOCKER_PAT`), runs `docker login -u $DOCKERHUB_USERNAME`
(default `shashankg209`), and **hard-aborts on failure — no anonymous fallback**.
It also neutralizes a broken `credsStore` (e.g. the VS Code dev-containers
helper, which otherwise makes `docker login` fail to persist and turns every
pull into `unauthorized`). To use a different account, set `DOCKERHUB_USERNAME`
+ `DOCKER_PAT` (or `DOCKER_PAT_SECRET`).

---

## 4. Local / direct harbor runs

Beyond the `run_eval_local.sh` mirror documented in [§3](#3-path-a--beaker-against-a-local-vllm),
you can invoke `uv run harbor run` directly from any machine with the relevant
sandbox backend. This is the path used for ad-hoc smoke tests and for evaluating
locally-generated RL datasets.

A direct invocation looks like:

```bash
# API model against Daytona cloud sandboxes
export DAYTONA_API_KEY=... ANTHROPIC_API_KEY=...
uv run harbor run \
    --dataset terminal-bench@2.0 \
    --agent mini-swe-agent \
    --model anthropic/claude-sonnet-4-5 \
    --env daytona --n-concurrent 8

# Self-hosted vLLM checkpoint against a local Docker daemon
source scripts/setup_podman_harbor.sh    # if using podman; skip for real Docker
uv run harbor run \
    --dataset terminal-bench-sample@2.0 \
    --agent Vanillux2Agent:Vanillux2Agent \
    --model openai/my-checkpoint \
    --agent-kwarg api_base=http://localhost:8008/v1 \
    --env docker
```

The `--env` flag selects the sandbox backend (`docker` / `daytona`) and the
`--model` prefix selects the model source (see [§7 Models](#7-models-outputs-and-result-analysis)).
To restart only the trials that failed with a transient Daytona error, use
`harbor jobs resume --job-path jobs/<job-name> --filter-error-type DaytonaError`.

### Evaluating on a generated RL dataset

[`scripts/run_rldata_claude.sh`](../scripts/run_rldata_claude.sh) (and
`run_rldata_claude_test.sh` for a 10-task subset) run the `terminus-2` agent
over a locally-generated task set. Convert tasks to harbor format first:

```bash
uv run python rl_data/scripts/analyze/convert_to_harbor.py \
    --src rl_data/output/tasks_skill_tax_20260401_10k \
    --dst rl_data/output/tasks_skill_tax_20260401_10k_harbor
```

then `bash scripts/run_rldata_claude.sh` (or `_test.sh` for a 10-task subset;
the test script supports `ENV=docker` if you have a local daemon). Note these
use harbor's `--path <dir>` (a local dataset) rather than `--dataset <name>`.

---

## 5. Datasets

Selected with `--dataset <name>@<version>` (downloaded/cached by harbor) or
`--path <dir>` (a local harbor-format dataset).

| Dataset | What it is |
|---|---|
| `terminal-bench@2.0` | Terminal-Bench 2.0 — the primary terminal-task suite (~89+ tasks). |
| `terminal-bench-sample@2.0` | Tiny sample slice for smoke tests. |
| `terminal-bench-pro@1.0` | Harder Terminal-Bench Pro tasks. |
| `openthoughts-tblite@2.0` | OpenThoughts "TB-lite" lightweight terminal tasks. |
| `swebench-verified@1.0` | SWE-Bench Verified (real GitHub issue fixes). |
| `terminal-bench-2-1` (via `--dataset-path`) | TerminalBench 2.1 — 89 revised tasks; off the pinned registry, run from a local dir (see [Off-registry datasets](#off-registry-datasets-eg-terminalbench-21)). |
| local `--path` | Your own converted task set (e.g. generated RL data). |

Restrict to specific tasks with repeated `--task-name <task>` (e.g. to pin a
fixed subset like the seeds in
[`scripts/swebench100_tasks.txt`](../scripts/swebench100_tasks.txt)).

> **Where dataset names come from.** Slugs aren't hardcoded in harbor — they
> resolve against a remote registry
> (`https://raw.githubusercontent.com/laude-institute/harbor/main/registry.json`,
> see `harbor/models/registry.py`). The catalog is much larger than the table
> above (aider-polyglot, livecodebench, gaia, swebenchpro, …); browse it at
> [hub.harborframework.com](https://hub.harborframework.com) or by reading
> `registry.json`. The table only lists the datasets `tmax`'s scripts actually
> use; all five were confirmed present in the registry. Because the set is
> registry-driven it can change upstream — the `@version` pin guards against
> content drift but not against a dataset being renamed or removed.
>
> To list every `name version` in the live registry yourself:
>
> ```bash
> curl -fsSL "https://raw.githubusercontent.com/laude-institute/harbor/main/registry.json" \
>   | python3 -c "import sys, json; [print(x['name'], x.get('version')) for x in json.load(sys.stdin)['datasets']]"
> ```
>
> Confirm the registry URL harbor actually uses (in case it changes):
>
> ```bash
> uv run python -c "from harbor import constants; print(constants.DEFAULT_REGISTRY_URL)"
> ```

### Off-registry datasets (e.g. TerminalBench 2.1)

The pinned harbor (0.6.6) registry only exposes `terminal-bench@2.0`.
**TerminalBench 2.1** (`terminal-bench/terminal-bench-2-1`, 89 revised tasks)
exists only on the *current* hub ([hub.harborframework.com](https://hub.harborframework.com)),
which the pinned harbor can't resolve — but its tasks use the same
`task.toml`+`environment/` format and prebuilt `alexgshaw/*` images as 2.0, so our
harbor runs them unchanged from a **local directory**. No harbor upgrade required.

1. **Download once** with a newer harbor as the resolver (the ref is namespaced;
   a bare `terminal-bench-2-1` 404s). Put it on a weka fs the Beaker jobs mount —
   `oe-adapt-default` is mounted by default:

   ```bash
   uvx harbor==0.18.0 datasets download terminal-bench/terminal-bench-2-1 \
     -o /weka/oe-adapt-default/shashankg/datasets --export
   # -> /weka/oe-adapt-default/shashankg/datasets/terminal-bench-2-1/<task>/  (89 dirs)
   ```

2. **Run it** with `--dataset-path <dir>` — a launcher flag on `launch_eval.sh`,
   `run_eval_in_job.sh`, and `run_eval_local.sh` that makes harbor use `--path`
   instead of `--dataset`. Everything else (agent, parser, provider,
   `--language-model-only`, mirror, k, context) is identical to a 2.0 run:

   ```bash
   ./beaker_configs/launch_eval.sh allenai/tmax-4b \
     --dataset-path /weka/oe-adapt-default/shashankg/datasets/terminal-bench-2-1 \
     --agent Vanillux2Agent:Vanillux2Agent --model-provider openai \
     --tool-call-parser qwen3_xml --language-model-only \
     --gpus 1 --max-model-len 65536 --n-attempts 5 --cluster ai2/jupiter \
     --workspace ai2/oe-agents
   ```

   The directory must live under a weka mount the job has (`launch_eval.sh`
   mounts `oe-adapt-default`). To move any existing eval onto 2.1, just swap
   `--dataset terminal-bench@2.0` for the `--dataset-path` above.

---

## 6. Agents

Agents live in top-level packages and are selected either by harbor's built-in
name (`--agent <name>`) or by import path (`--agent-import-path module:Class`).
Custom agents take `--agent-kwarg key=value` flags. For a deeper per-harness
breakdown (prompts, tool surfaces, the works-on-0.6.6 × provider × parser compat
matrix), see [Agent harnesses](agent_harnesses.md).

| Agent | Selector | Summary |
|---|---|---|
| **Vanillux2Agent** (default) | `Vanillux2Agent:Vanillux2Agent` | The `launch_eval.sh` default. Direct-LiteLLM port of the `rl_data` vanillux solver: same prompts/tool schema/truncation as the RL-data generator, but driven through harbor's environment. Uses **structured** litellm tool-calls → needs the right `--tool-call-parser` for the model and the `openai/` provider (see below). Runs **host-side** (only bash execs enter the sandbox). |
| **mini-swe-agent** | `mini-swe-agent` (built-in) | Harbor's lightweight SWE agent, installed inside the sandbox. Parses `bash` code blocks from plain text, so it is **independent of the tool-call parser**. Use the `openai/` provider. |
| **swe-agent** | `swe-agent` (built-in) | Upstream SWE-agent (Yang et al. 2024) inside the sandbox: bash + view/edit/submit tools. Uses `hosted_vllm/` + `hermes`. |
| **terminus-2** | `terminus-2` (built-in) | Harbor's built-in agent, used by the RL-data scripts. Uses the `openai/` provider; `qwen3_xml` is a safe parser default. |
| **oracle** | `oracle` (built-in) | Runs the task's reference solution; for sanity-checking infra (should score ~1.0). |

> **Tool-call parser & provider must match the agent.**
>
> | Agent kind | `--model-provider` | `--tool-call-parser` (Qwen3.5) |
> |---|---|---|
> | **Vanillux2Agent** (repo custom litellm agent, structured tool-calls) | `openai` | **`qwen3_xml`** |
> | `mini-swe-agent` (text bash blocks) | `openai` | any (`hermes` fine — parser-independent) |
> | `terminus-2` (built-in) | `openai` | `qwen3_xml` (safe default) |
> | `swe-agent` (SWE-agent in sandbox) | `hosted_vllm` | `hermes` |
>
> Qwen3.5 emits `<function=name><parameter=…>` XML; with the default `hermes`
> parser those tool-calls are **silently dropped**, so a structured-tool agent
> like **Vanillux2Agent** loops on "Format error" and gives up with ~0 useful
> steps. `qwen_xml` is **not** a valid parser name — use `qwen3_xml` (valid
> names include `hermes, qwen3_coder, qwen3_xml, …`). For provider: built-in
> agents and `Vanillux2Agent` use `openai/<served-name>` (+ `OPENAI_API_BASE` /
> `OPENAI_API_KEY=dummy`) on the Beaker vLLM path — that's the verified combo
> and what `launch_eval.sh` defaults via `--model-provider`. `swe-agent`
> instead addresses the vLLM as `hosted_vllm/<name>`. **Do not set
> `MSWEA_API_KEY`** for built-in agents — harbor's mini-swe-agent then forwards
> only that, skips `OPENAI_API_KEY`, and litellm reports "Missing credentials".

> The choice of agent matters for fairness — keep the harness fixed when
> comparing models, since different agents use different step/call limits and
> tool surfaces.

---

## 7. Models, outputs, and result analysis

### How `--model` is interpreted

`--model` is a litellm identifier. The prefix selects the provider and the
required key:

| `--model` prefix | Provider | Required env |
|---|---|---|
| `hosted_vllm/<name>` | self-hosted vLLM (also pass `--agent-kwarg api_base=...`) | — (or `OPENAI_API_KEY=dummy`) |
| `anthropic/...` | Anthropic API | `ANTHROPIC_API_KEY` |
| `openai/...` | OpenAI API | `OPENAI_API_KEY` |
| `gemini/...` | Google AI Studio | `GEMINI_API_KEY` |

For self-hosted vLLM, harbor's SWE-agent adapter also needs `OPENAI_BASE_URL`
set (litellm convention) — the in-job script exports both `OPENAI_API_BASE`
and `OPENAI_BASE_URL`.

> A self-hosted vLLM can be addressed two ways, and which one depends on the
> **agent**, not the server: SWE-agent agents use `hosted_vllm/<served-name>`,
> while built-in agents and `Vanillux2Agent` use `openai/<served-name>` (the
> installed harbor's litellm has no usable `hosted_vllm` path). `launch_eval.sh`
> picks the prefix from the agent type; override with `--model-provider`. See
> the parser/provider table in [§6](#6-agents).

### Output layout

A finished job writes `jobs/<job-name>/`:

```
jobs/<job-name>/
├── result.json                 # job-level: trial count, reward dist, errors
├── config.json                 # the resolved harbor config (presence = resumable)
├── stats.txt                   # compute_stats.py human summary  (Beaker path)
├── metrics.json                # compute_stats.py structured metrics (Beaker path)
└── <task>__<rand>/             # one dir per trial
    ├── result.json             # task_name, verifier_result.rewards.reward, exception_info
    ├── agent/oracle.txt        # agent stdout
    ├── verifier/test-stdout.txt
    ├── verifier/reward.txt
    ├── exception.txt
    └── trial.log
```

### `compute_stats.py` — aggregate reward + pass@k

```bash
uv run python scripts/compute_stats.py jobs/<job-name>
uv run python scripts/compute_stats.py jobs/<job-name> --per-task
uv run python scripts/compute_stats.py jobs/<job-name> --json-output metrics.json
```

Reports mean reward ± std/SEM (treating each attempt index as an independent
run over the task set) and an unbiased **pass@k** for `k ∈ {1, min, max}`
attempts. See [`scripts/compute_stats.py`](../scripts/compute_stats.py).

### `compare_smoke_harnesses.py` — bash vs vanillux

Compares two harnesses on the same task set (intersection only), reporting
pass@1/pass@k, per-task wins/losses, exit-reason breakdowns, context-size
proxies, and token usage. See
[`scripts/analysis/compare_smoke_harnesses.py`](../scripts/analysis/compare_smoke_harnesses.py).

---

## 8. Resuming, cleaning, and troubleshooting

### Resume

If `jobs/$JOB_NAME/` exists with a `config.json`, you can resume with
`harbor jobs resume --job-path jobs/$JOB_NAME --filter-error-type DaytonaError`
(re-running only trials that failed with a transient Daytona error). The
`run_rldata_*.sh` wrappers do this automatically. To start fresh, change
`JOB_NAME` or delete the dir.

### Clean errored trials

To re-run errored trials selectively, delete them so harbor re-creates them:

```bash
./scripts/clean_errors.sh jobs/<job-name>                  # all errored trials
./scripts/clean_errors.sh jobs/<job-name> AgentTimeoutError # only this type
```

### Troubleshooting (podman/Beaker path)

The full failure-mode table lives in
[`scripts/beaker/README.md`](../scripts/beaker/README.md#where-to-look-when-something-goes-wrong).
Highlights:

| Symptom | Look at |
|---|---|
| `unknown shorthand flag: 'p' in -p` on `docker compose down` | Image lacks the compose plugin; the script auto-installs it — check network reachability. |
| `mknod` permission denied | Cluster doesn't grant `CAP_MKNOD`; use an image with `/dev/net/tun` pre-created. |
| vLLM never ready | `/tmp/vllm.log` (the in-job script tails it on failure). |
| All trials `RewardFileNotFoundError` | Harbor patches didn't apply — confirm the chmod patch in `harbor/agents/oracle.py`. |
| `setgroups 65534` | userns too small — confirm `auto:size=65536` in `containers.conf`. |
| Reward 0 across the board | Likely a real model/agent issue; inspect one trial's `agent/oracle.txt` + `verifier/test-stdout.txt`. |
| `toomanyrequests` from Docker Hub | Set the `DOCKER_PAT` secret (authenticated pull cap); `--rmi all` is already dropped. |
| Agent loops on "Format error", ~0 progress | Wrong tool parser for the model — pass `--tool-call-parser qwen3_xml` for Qwen3.5 + structured-tool agents (e.g. Vanillux2Agent) ([§6](#6-agents)). |
| litellm `Missing credentials` / `set OPENAI_API_KEY` | `MSWEA_API_KEY` is set (skips `OPENAI_API_KEY`), or wrong provider — use `--model-provider openai`, `OPENAI_API_KEY=dummy`, and unset `MSWEA_API_KEY`. |
| litellm `Connection error` (local real-Docker) | Agent used `localhost` but is a sibling container — use the dev-VM container's bridge IP (`run_eval_local.sh` handles it). |
| `'compose' is not a docker command` / `unknown flag: --project-name` (local) | Docker Compose v2 plugin missing — install it (`set_dev_vm.sh` / `run_eval_local.sh` do). |

### Common Daytona-path gotchas

- **Missing key**: `--env daytona` needs `DAYTONA_API_KEY`; the model needs its
  own `*_API_KEY`. Validate these up front.
- **Home-quota blowups**: harbor caches tasks under `~/.cache/harbor`; on
  quota-constrained machines symlink it to scratch (set `HARBOR_CACHE_TARGET`
  if your scratch path differs).

---

## 9. Prerequisites summary

| Path | Needs |
|---|---|
| Beaker | Beaker access + workspace; `HF_TOKEN` secret; `DOCKER_PAT` secret (recommended); weka mount; a **pushed** git SHA. |
| Local | A Daytona account/key (for `--env daytona`) **or** a local Docker/podman daemon + the `docker compose` v2 CLI plugin (for `--env docker` / `run_eval_local.sh`); the relevant model `*_API_KEY` (or `OPENAI_API_KEY=dummy` for self-hosted vLLM). |

All paths run through `uv` (`uv sync` / `uv run`), Python ≥ 3.12.
