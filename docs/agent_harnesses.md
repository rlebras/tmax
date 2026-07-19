# Agent Harnesses Compared

This document compares the agent harnesses available in `tmax` for driving a
model through terminal/coding tasks (used both for eval and for RL/SFT data
generation). It explains what each one is, how its agent loop works, and—most
usefully—how they differ from each other and from a textbook **ReAct** agent.

For *how to launch* these agents (CLI flags, providers, parsers, datasets) see
[`running_evals.md`](running_evals.md) §7. This doc is about *what they are and
why they differ*.

---

!!! note "Removed agents (master cleanup)"
    `VanilluxAgent`, `TassieAgent`, and `TassumAgent` were **removed** by the
    master cleanup commit (`06b9621`, "Remove obsolete sft/ pipeline, old eval
    launchers, and legacy agents"). They were Gemini-era / structured-tool
    harnesses (a SWE-agent wrapper plus two save/restore litellm agents) that
    have been superseded by `Vanillux2Agent`. Their directories no longer exist
    in the repo, so any older references to them elsewhere are **historical
    only** and do not describe a live agent. The remaining repo-custom agent is
    `Vanillux2Agent`; everything else below is a harbor built-in.

---

## TL;DR

Almost every agent here is a variant of the same idea: a **ReAct loop with one
`bash` tool**. The model thinks (THOUGHT), emits an action (a shell command),
sees the observation (truncated stdout/stderr), and repeats until it runs
`echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`. What actually differs between them
is the *operational envelope* around that loop:

- **Where the loop runs** — host-side (Python driving `environment.exec()`) vs.
  inside the sandbox (an installed agent binary).
- **How actions are parsed** — native structured tool-calls vs. text/XML/JSON
  blocks vs. tmux keystrokes.
- **How shell state is kept** — stateless subshells, a save/restore wrapper, or
  a real persistent tmux session.
- **How context is managed** — let it grow until OOM, or proactively summarize.
- **How rich the tool surface is** — bash-only vs. bash+view/edit/submit.

| Agent | Kind | Runs | Tools | Action format | State | Context mgmt |
|---|---|---|---|---|---|---|
| **Vanillux2Agent** (default) | standalone `BaseAgent` | host-side | bash | structured tool-call | save/restore wrapper | none |
| **mini-swe-agent** | built-in installed agent | in sandbox | bash | text ` ```bash ``` ` block | persistent shell | none |
| **terminus-2** | built-in `BaseAgent` | host-side | tmux keystrokes | JSON or XML plain | **persistent tmux** | proactive summarization |
| **swe-agent** | built-in installed agent | in sandbox | bash + view/edit/submit | SWE-agent internal | persistent (in-sandbox) | history processors |
| **oracle** | built-in | n/a | n/a (reference solution) | n/a | n/a | n/a |

`Vanillux2Agent` is the **default agent in `launch_eval.sh`** (`AGENT_IMPORT_PATH=Vanillux2Agent:Vanillux2Agent`).

---

## The ReAct baseline (reference point)

A "simple ReAct-style agent" (Yao et al. 2022) is:

1. A system prompt that asks the model to alternate **Thought → Action →
   Observation**.
2. A **typed action vocabulary** (e.g. `search[query]`, `lookup[term]`,
   `finish[answer]`), parsed out of free-form text with a regex.
3. A loop that runs the parsed action, appends the observation, and repeats.
4. Termination via a dedicated `finish[]` action or a step cap.
5. Minimal-to-no error handling, output truncation, or context management.

Every agent below keeps the Thought→Action→Observation spine but changes one or
more of: the action *channel*, the action *vocabulary*, the *state* model, the
*context* model, and the *operational hardening*. The recurring theme in `tmax`
is the **mini-swe-agent thesis**: collapse the action vocabulary to a single
`bash` tool and put all the structure into the *prompt*, not into tool
machinery.

---

## The two architectural families

### A. Host-side litellm agents (`BaseAgent`)
`Vanillux2Agent` is the surviving member. The Python agent loop runs on the
**host**; only the bash commands cross into the sandbox via
`environment.exec()`. It calls the model directly through litellm with
**structured (OpenAI-style) tool-calls**. On Qwen3.5 it therefore requires
`--tool-call-parser qwen3_xml` (with the default `hermes`, structured tool-calls
are silently dropped and the agent loops on "Format error"); on Qwen3 use
`hermes` + `--reasoning-parser qwen3`. Provider is `openai/` — the installed
harbor's litellm has no usable `hosted_vllm/` path for this agent.

### B. In-sandbox installed agents
`mini-swe-agent` and `swe-agent`. The agent binary is installed *inside* the
task container and runs there; harbor converts its native trajectory to ATIF for
reporting. `tmax` does not control their loop directly—it configures them via
flags. These are harbor built-ins, not repo code.

### C. terminus-2 (built-in, host-side, tmux)
Harbor's own built-in agent. Host-side like family A, but instead of a `bash`
tool it drives a **persistent tmux session** with raw keystrokes, and parses
the model's output as a plain **JSON or XML** document rather than a tool-call.

---

## Per-agent detail

### Vanillux2Agent — mini-swe-agent prompts, host-side (launch_eval.sh default)
*Source:* `Vanillux2Agent/agent.py` · prompts vendored in
`rl_data/generator/vanillux_prompts.yaml`

The only repo-custom agent left, and the **default agent in `launch_eval.sh`**.
A host-side litellm port of the `rl_data` vanillux solver: a clean Harbor
`BaseAgent` implementing a single-`bash`-tool ReAct loop with **native
structured tool-calls** (it reads the `command` argument out of
`response.tool_calls`).

- **Prompts are the vendored mini-swe-agent v2.2 templates** (system + instance
  with the explicit 5-step "Recommended Workflow" and command examples).
- **Action channel:** native structured tool-call—no text parsing. On Qwen3.5
  this means `--tool-call-parser qwen3_xml` is mandatory (Qwen3.5 emits
  `<function=..><parameter=..>` XML that `qwen3_xml` decodes back into the
  structured call); `hermes` silently drops the call.
- **Budgets tuned to mini-swe-agent:** `max_steps=64`, `max_tokens=16384`.
- Explicit **format-error recovery loop**: a malformed turn is fed back as an
  error message (up to `max_format_errors=64`) so the model self-corrects rather
  than wasting a step.
- **State:** save/restore persistent shell—each command is wrapped to restore
  `cwd`+`env` from `/tmp/.vanillux2/` before running and re-save after, so
  `cd`/`export` persist across turns without a long-lived shell process.
- **Observation:** each tool result truncated to ~10 000 chars (head/tail with an
  "elided" marker).
- **Termination:** `echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT` submit marker,
  `max_steps`, optional cost limit, no-tool-call, or context-window overflow.
- **Hardening:** exponential-backoff retry; auth / not-found / context-window /
  unsupported-param / permission errors abort immediately.
- **Context management:** *none*—history grows until the context window is
  exceeded, then the loop stops and submits whatever exists.

This is the recommended agent on harbor 0.6.6 when you want the mini-swe-agent
recipe but a host-side loop you control. See the prior writeup in
[`running_evals.md`](running_evals.md) for the full Vanillux2 vs ReAct
breakdown.

### mini-swe-agent — the upstream minimalist
*Source:* harbor built-in (`harbor/agents/installed/mini_swe_agent.py`); prompts
also vendored into `rl_data/generator/vanillux_prompts.yaml`

The reference implementation of the "prompts > tools" philosophy: a **single
bash tool**, no editor/submit/review machinery, ~74 % on SWE-bench-verified
beating SWE-agent's own ~65 %. Installed inside the sandbox.

- **Action channel:** upstream parses a ` ```bash … ``` ` **text block** out of
  free-form output—so it is **independent of the tool-call parser** (any parser,
  even `hermes`, is fine). This is the key practical difference from the `tmax`
  litellm agents, which vendored these prompts but switched to native
  tool-calls. Provider `openai/`.
- Persistent shell, head/tail truncation, format-error re-prompt, submit-marker
  termination—all the patterns `Vanillux2Agent` copied.
- **Gotcha:** don't set `MSWEA_API_KEY`; mini-swe-agent forwards only that and
  skips `OPENAI_API_KEY`, breaking litellm credentials.

### terminus-2 — harbor's tmux agent
*Source:* harbor built-in (`harbor/agents/terminus_2/`)

The structurally most different agent here. Host-side, but instead of a `bash`
tool it operates a **persistent tmux session** and the model emits a plain
**JSON or XML** document (pluggable parser) with fields `analysis`, `plan`,
`commands`, `task_complete`. Each command is `keystrokes` + a `duration`,
letting it handle interactive/long-running TUI programs (send `C-c`, wait N
seconds, etc.)—something none of the bash-tool agents can do cleanly. Provider
`openai/`; `qwen3_xml` is the safe parser on Qwen3.5.

- **State:** a real long-lived tmux session—the strongest persistence model
  (genuine interactive shell, not a save/restore wrapper).
- **Context:** built-in proactive summarization (threshold default 8000), with
  optional linear-history segmentation.
- **Termination:** `task_complete: true` in the JSON, cost limit, or
  context-window overflow; `max_turns` defaults effectively unlimited.
- Supports extended thinking (`max_thinking_tokens`) and full ATIF trajectories.

Used by the RL-data generation scripts (`run_rldata_claude.sh`) to produce
training traces; those traces assume a persistent shell, which is why the
inference harnesses also use persistent state.

### swe-agent — upstream SWE-agent
*Source:* harbor built-in (installed agent)

The full upstream agent (Yang et al. 2024) installed in the sandbox: bash +
view/edit/submit tools, its own history processors, internal string parsing. The
dedicated `str_replace_editor` is the main measured quality lift on long,
edit-heavy tasks. Uses `hosted_vllm/` + `hermes`.

### oracle — reference-solution runner
*Source:* harbor built-in

Not a model-driven loop at all: harbor's `oracle` agent runs each task's
reference solution. Useful as an upper-bound / task-sanity check (confirms the
task + verifier are well-formed) rather than as a model harness.

---

## Dimension-by-dimension comparison

### 1. Action channel (how the model's action is parsed)
- **Native structured tool-call** (Vanillux2): robust, but ties you to a
  model+parser combo. On Qwen3.5 you *must* use `qwen3_xml`.
- **Text bash block** (mini-swe-agent): parser-independent, simplest to serve.
- **JSON/XML plain document** (terminus-2): richest action schema (keystrokes +
  duration + plan + completion flag), supports interactive programs.
- **SWE-agent internal** (swe-agent): multi-tool, opaque to `tmax`.
- **vs ReAct:** ReAct parses a typed `Action: verb[arg]` line with a regex. All
  of these move the action onto a more structured or more general channel.

### 2. Tool surface
- **bash-only:** Vanillux2, mini-swe-agent, terminus-2 (as keystrokes). Editing
  is done via `sed`/heredoc.
- **bash + view/edit/submit:** swe-agent. The dedicated editor is the main
  measured quality lift on long tasks.
- **vs ReAct:** ReAct typically has 3–10 typed tools; the `tmax` trend is the
  opposite—*fewer* tools, richer prompts.

### 3. Shell-state model
- **Save/restore wrapper:** Vanillux2 persistent mode—snapshot `pwd`+`export -p`
  to `/tmp/.vanillux2/` around each command. State persists without a live shell
  process.
- **Genuine persistent session:** mini-swe-agent (login shell), swe-agent
  (in-sandbox), terminus-2 (tmux). terminus-2's tmux is the only one that
  supports true interactivity.
- **vs ReAct:** ReAct actions are usually stateless function calls; persistence
  here is a deliberate addition for coding tasks.

### 4. Context management
- **None (grow until OOM):** Vanillux2, mini-swe-agent, swe-agent (modulo its
  history processors).
- **Proactive summarization:** terminus-2 (built-in).
- **vs ReAct:** vanilla ReAct has no context management; this is the single
  biggest "production" addition for long-horizon tasks.

### 5. Observation truncation
The `tmax` litellm agent (Vanillux2) truncates per-observation to ~10 k chars
(head 5 k + tail 5 k + hint). ReAct usually hard-cuts or drops. Keeping both head
and tail preserves e.g. the error line at the end of a stack trace.

### 6. Termination
- **Shell submit sentinel** (`echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`):
  Vanillux2, mini-swe-agent. "Done" travels through the same bash channel as
  everything else—no dedicated finish action.
- **Completion flag:** terminus-2 (`task_complete: true`), swe-agent (submit
  tool).
- Plus universal caps: step/turn limit, cost limit, context overflow.
- **vs ReAct:** ReAct uses a dedicated `finish[]` action.

### 7. Budgets (defaults; launch scripts override)
- Vanillux2: `max_steps=64`, `max_tokens=16384`.
- terminus-2: effectively unlimited turns; bounded by cost.
- **Fairness note:** call-limit vs step-limit agents are *not* directly
  comparable; keep the harness fixed when comparing models.

### 8. Error/retry hardening
The host-side litellm agent (Vanillux2) uses exponential backoff (5 attempts,
2→32 s) with an abort-immediately set (auth/not-found/context-window/
unsupported/permission), plus an explicit format-error recovery loop;
mini-swe-agent re-prompts on malformed turns. ReAct typically has minimal or no
retry logic.

---

## Practical compatibility (harbor 0.6.6)

Harbor is still pinned at **0.6.6**.

| Agent | Works on 0.6.6? | `--model-provider` | `--tool-call-parser` (Qwen3.5) |
|---|---|---|---|
| Vanillux2Agent (default) | yes | `openai` | **`qwen3_xml`** |
| mini-swe-agent | yes | `openai` | any (parser-independent) |
| terminus-2 | yes | `openai` | `qwen3_xml` (safe) |
| swe-agent | yes | `hosted_vllm` | `hermes` |

Key traps (from [`running_evals.md`](running_evals.md) §7):

- **Structured-tool agents** (Vanillux2, terminus-2) on Qwen3.5 **must** use
  `qwen3_xml`; `hermes` silently drops their structured tool-calls → 0 useful
  steps. `qwen_xml` is not a valid name. On **Qwen3** use `hermes` +
  `--reasoning-parser qwen3` (Qwen3 emits Hermes-style tool calls plus `<think>`
  blocks).
- `mini-swe-agent` is **parser-agnostic** (it reads a text ` ```bash``` ` block),
  so the tool-call parser choice doesn't matter for it.
- Don't set `MSWEA_API_KEY` for built-in agents (breaks litellm credentials).

---

## Which to use

- **Default eval / fast & simple:** `Vanillux2Agent` (the `launch_eval.sh`
  default—works on 0.6.6, mini-swe-agent prompts, host-side loop you control).
- **Long-horizon / interactive / TUI / long-running-program tasks:** `terminus-2`
  (tmux + built-in summarization).
- **Best raw quality on edit-heavy SWE tasks:** `swe-agent`—the dedicated editor
  matters.
- **Parser-agnostic serving:** `mini-swe-agent` (text bash blocks, any parser).
- **Task / verifier sanity check:** `oracle` (runs the reference solution).

---

*Sources: `Vanillux2Agent/agent.py`, `rl_data/generator/vanillux_solver.py`,
`rl_data/generator/vanillux_prompts.yaml`, harbor `agents/terminus_2/` and
`agents/installed/mini_swe_agent.py`, `launch_eval.sh`, `docs/running_evals.md`
§7, and the run-terminal-eval skill compat matrix.*
