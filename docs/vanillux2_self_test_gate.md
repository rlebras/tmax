# Vanillux2Agent self-test gate

*Source:* [`Vanillux2Agent/self_test.py`](../Vanillux2Agent/self_test.py) ·
wired into [`Vanillux2Agent/agent.py`](../Vanillux2Agent/agent.py) · builds on
[`vanillux2_context_management.md`](vanillux2_context_management.md)'s
compaction/edit-tool harness

## Why

A failure analysis of 445 runs found that of runs that finish and submit a
wrong solution, **67% end by explicitly declaring success** ("all tests
pass", "ready to submit") on a solution the held-out grader rejects. Only
**22% of runs self-test at all**, and those tests are usually **circular or
trivial** — one agent printed "PASS" on a sampler whose output had `std=0`;
another "verified" against a filter it had itself neutered, apparently
checking it against a known-good backup left over from an earlier iteration
rather than the file it actually submitted. The grader is held out (the
agent never sees `/tests`).

This is a pure harness change — no model or training changes. It gates the
`COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT` sentinel on isolated self-test
coverage, and it composes with the context-management harness:
criteria/checks are file-backed state (disk is the source of truth, never
inlined into history at full size), and every check result goes through the
same compact-result discipline as everything else in this harness.

## Two model-facing interfaces, one core

Which interface the model gets follows the tool contract
(`enable_edit_tools`):

* **Tools** (multi-tool contract, the default): `declare_criteria` and
  `run_check` are registered alongside bash/edit tools, described in the
  `self_test_*_addendum` keys of `vanillux_prompts.yaml`. This is the
  primary interface — a live 445-trial A/B of an earlier
  bash-convention-only variant found **12% of trials botched the
  convention's syntax** and 15 trials probed for the (non-existent) binary
  with `which agent-check` and then gave up on the mechanism; a real tool
  schema removes both failure modes at the source.
* **Bash convention** (bash-only contract, keeps the RL-data-compatible
  single-`bash` schema untouched): the model writes the criteria file
  itself and runs `agent-check <id> -- <command>`, which the harness
  intercepts before it ever reaches the shell — as the whole bash call, or
  as its own line inside a larger call (setup on other lines runs normally),
  or several checks back to back. A line starting with the `agent-check`
  token that matches neither form gets a corrective syntax hint instead of a
  shell "command not found", and existence probes are answered directly —
  all three hardening behaviors come from failure modes observed live in
  that A/B. The interception also stays active in tool mode, so a model
  that reaches for the convention anyway still gets real checks.

## Mechanism 1 — acceptance-criteria declaration

Criteria are `{id, description, how_to_check}` triples — the observable
properties the task actually asks for — plus the list of `deliverables`
(the file(s) that make up the solution). They live in a harness-owned file,
`<state_dir>/criteria.json`: the `declare_criteria` tool validates and
merges into it (re-declaring updates by id); in bash-only mode the model
writes the file directly and the harness re-reads it fresh whenever it
needs the current declaration. Either way the declaration is never inlined
into the model-facing history — the tool's compact ack (`"declared N
criteria: ..."`) is all the model sees back. The prompt addenda tell the
model to declare early and to turn any example input/output from the task
statement into criteria first.

## Mechanism 2 — structured, compact checks

`run_check(criterion_id, command, expected)` — or `agent-check <id> --
<command>` — runs `command` twice (see Mechanism 3). A check passes iff the
command **exits 0**, or, when the tool form supplies a static `expected`
string, iff its trimmed output equals it. The model only ever sees
`PASS`/`FAIL` plus, on failure, the first failing line — never full output —
so a check whose command dumps hundreds of lines costs the same handful of
tokens as one that prints nothing. Every check is recorded to
`<state_dir>/checks.json` (`criterion_id -> [{command, expected,
session_pass, isolated_pass, first_failure_line, circular, trivial,
check_category, ...}]`) so coverage, the anti-weakening heuristics, and the
offline report can look back at history.

## Mechanism 3 — isolation (what makes the gate hard to game)

Every check runs **twice**: once in-session (the persistent bash shell the
model has been using), and once against a **fresh copy of only the declared
deliverables**, mirrored at their real absolute paths under a scratch root
(`<isolation_root>/<criterion_id>/...`), invoked as a fresh process with no
persistent cwd/env chaining (`agent.py`'s `isolated_exec` deliberately
bypasses `_wrap_command`). Only the isolated result counts toward coverage.
A check that passes in-session but fails in isolation is surfaced as an
explicit warning — the direct counter to the "neutered filter, checked
against a leftover known-good backup" failure mode: the backup is real, on
disk, and genuinely passes in the session shell, but it was never declared
as a deliverable, so it simply isn't there in isolation.

The deliverable mirror is batched into a single exec round trip per check
(`cp -a`, so directory deliverables and permission bits copy faithfully;
missing deliverables are reported to the model rather than silently
skipped).

**Known scope limitation, stated plainly:** this harness has no primitive
for spinning up a second container or namespace — `BaseAgent` only has
`environment.exec`/`upload_file`/`download_file`. So isolation is "fresh
temp dir + fresh process", not a fresh container, and a check command that
references a deliverable by **absolute** path escapes the sandbox (it
resolves against the real filesystem, not the isolated copy) — only
cwd-relative references are actually isolated. Real task-solving commands
overwhelmingly use relative paths, so this covers the realistic case, but
it is not a hard security boundary and isn't claimed to be one.
`self_test_isolation_mode` is a config axis reserved for a stronger mode if
the environment layer ever exposes one.

## Mechanism 4 — the submission gate

`agent.py` intercepts the submit sentinel. An **explicit** submit
(`echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT` as the bash call) is gated
*before* it executes — a rejected attempt costs no execution, and its tool
response is the compact nudge listing what's unmet. An **implicit** trip
(command output that merely happens to contain the marker text) is gated
after the fact so it can't bypass the gate. Submit is allowed only once:

- criteria have been declared, **and**
- **every** declared criterion has >=1 check that passed in isolation
  (circular or existence-only checks — Mechanism 5 — do not count), **and**
- at least `min_criteria` criteria are covered that way (so declaring one
  token criterion isn't enough).

Requiring *every* declared criterion (not just `min_criteria` of them)
closes the declare-many-check-few loophole; the criteria file is editable,
and pruning a criterion whose check failed is audited and warned about
rather than blocked (Mechanism 5).

Rejections are bounded by `max_gate_rejections`: once exhausted, submit is
allowed regardless of coverage and `submitted_with_failing_checks=True` is
recorded in `context.metadata` and `self_test.json`. **The bound applies
uniformly** — whether the model never declared criteria at all or just
can't get its checks passing — so either failure mode converts to a
flagged, analyzable submit rather than burning the entire step budget (see
"Risk" below).

## Mechanism 5 — anti-circularity + honesty heuristics

Best-effort, harness-side, and **non-blocking** (a self-test can
legitimately be wrong — these warn and/or exclude from coverage, never
hard-stop the run):

- **Circularity** (`circularity_flags`): no-op commands (`true`/`exit 0`);
  no recognizable assertion; a bare `==`/`!=` comparison nothing enforces
  (e.g. `python -c "print(a == b)"` — always exits 0); always-zero tails
  (`cmd && echo PASS || echo FAIL`, `|| true`, `; echo $?`); swallowed
  exceptions (`except: pass`) and `set +e`; comparing a program's output to
  itself (`diff <(prog) <(prog)`); an `expected` containing `$(...)`/
  backticks (it must be a fixed literal — the harness compares text, not
  shell). The `==`-blind and always-zero-tail patterns were each observed
  letting a wrong deliverable through in a live A/B before being added.
  With a static `expected`, the harness's own output comparison is the
  assertion, so the exit-code-blindness family is skipped.
- **External-script inspection**: a check that just delegates to
  `python3 verify.py` hides its (lack of) assertions from command-level
  heuristics, so when the outer command has no assertion of its own the
  referenced script's *content* is read and vetted instead.
- **Trivial existence** (`test -f x` / `ls x` alone): proves the file
  exists, not that it behaves — flagged and excluded from coverage, with a
  nudge to verify content/behavior.
- **Check weakening** (`weakening_warning`): if a criterion's most recent
  check failed in isolation and the next check for it changes the command
  or expected value, a warning fires ("confirm the criterion still holds,
  don't just loosen the check"). Deliberately coarse — "fixed the check"
  vs "weakened the check" isn't decidable from text alone.
- **Criteria-edit audit** (`criteria_edit_warnings`): every criteria-file
  add/change/removal is recorded to an audit trail; removing or rewording a
  criterion whose latest check failed additionally warns (deleting a
  failing requirement is the cheapest way to loosen the gate).
- **Observability**: every check gets a `check_category` — `behavioral`,
  `external_script`, `trivial_existence`, or `circular` — recorded purely
  so the offline report can see how substantive a run's checks were.

## Config flags

All on `Vanillux2Agent.__init__`:

| Flag | Default | Effect |
|---|---|---|
| `enable_self_test_gate` | `True` | Gate the submit sentinel on isolated self-test coverage (tools or agent-check convention per `enable_edit_tools`). |
| `min_criteria` | `2` | At least this many criteria must be declared and covered (and every declared criterion must be covered). |
| `max_gate_rejections` | `3` | After this many rejected submit attempts, allow submit anyway and record `submitted_with_failing_checks`. |
| `self_test_check_timeout` | `60` | Per-check timeout (seconds) for the isolated run. |
| `self_test_isolation_mode` | `"tempdir"` | Only mode implemented (see Mechanism 3's scope limitation); reserved for a future stronger-isolation mode. |

`context.metadata["self_test"]` reports `criteria_declared`,
`criteria_covered`, `checks_run`, `checks_by_category`,
`session_isolation_discrepancies`, `gate_rejections`, and
`submitted_with_failing_checks`; `<logs_dir>/self_test.json` additionally
records per-criterion coverage, the full per-check history, and the
criteria-edit audit trail.

## Risk this is designed to manage

A hard gate can convert "finished-wrong" into "step-exhaustion" or
"context run-out" if the model keeps failing its own checks — the
context-management change's own history shows failures merely *relocating*
under a naive harness change. Three things bound that risk here:
`max_gate_rejections` (Mechanism 4) forces a flagged submit rather than an
infinite retry loop; compact pass/fail-only check results (Mechanism 2)
keep verification token-cheap regardless of how much output a check's
command produces; and the two per-check executions (session + isolated,
with the isolation mirror batched into one round trip) are subprocess
calls, not extra LLM calls, so the gate's marginal cost lands on
wall-clock, not on the context or step budget. Whether that bounding
works is an empirical question — see Measurement.

## Measurement

`scripts/vanillux2_self_test_report.py` computes everything the A/B needs
from finished harbor trial directories: the outcome taxonomy, gate
adoption/coverage stats, check categories, `submitted_with_failing_checks`
rate, and step/token overhead.

Baseline (the failure-analysis population itself — 444 readable
`tmax-9b` terminal-bench trials under
`evaluation_assets/daytona/tmax-9b/terminal-bench-2-0/`, no gate):

```
submitted_right    105 (23.6%)      median steps            28
submitted_wrong    110 (24.8%)      median prompt tokens    503,654
step_exhausted      23 ( 5.2%)      median completion tokens 21,311
stopped_early      206 (46.4%)      (33 of stopped_early: AgentTimeoutError)
```

The claims this change makes are testable against exactly these numbers:
`submitted_wrong` should drop, those runs should convert to
`submitted_right` **rather than** to `step_exhausted`/`stopped_early`, the
`submitted_with_failing_checks` rate reports how often the bounded gate
had to stand down, and the medians bound the added step/token cost.

**Not yet measured, stated plainly:** unlike the context-management report
(which replays fixed trajectories offline), the gate changes what the agent
*does*, so producing the A/B requires live reruns:
`scripts/beaker/launch_tmax9b_tb21_ctxmgmt.sh` (context management, no
gate) vs `scripts/beaker/launch_tmax9b_tb21_selftest.sh` (this gate; add
`--agent-kwarg enable_self_test_gate=false` to A/B against the same commit).
Run the report script once per arm on the resulting trial directories.

## Tests

[`tests/vanillux2/test_self_test.py`](../tests/vanillux2/test_self_test.py)
covers: agent-check parsing (whole-command, line-embedded, multi-check,
malformed, existence probes), criteria-file parsing/merging and the
`declare_criteria` tool, every circularity/triviality heuristic (including
the exit-code-blind patterns, external-script inspection, and
expected-string rules), weakening + criteria-edit-audit warnings, the
neutered-filter / leftover-state fixtures failing in isolation while
passing in-session, directory deliverables, missing-deliverable reporting,
compact-result-only output, gate acceptance/rejection across coverage
shapes (including the every-criterion rule and circular/trivial exclusion),
the uniform rejection bound, and `dispatch()` never raising.

[`tests/vanillux2/test_agent_self_test_gate.py`](../tests/vanillux2/test_agent_self_test_gate.py)
drives the actual `Vanillux2Agent.run()` loop end to end (scripted fake
model, local-filesystem fake environment — no docker, no live LLM) in both
contract modes: a premature submit is rejected pre-execution with a compact
tool-response nudge and the run continues; a fully-covered submit finishes;
a model that never declares criteria terminates once `max_gate_rejections`
is exhausted with `submitted_with_failing_checks=True`; leftover-state
checks are caught through the real loop; the agent-check convention,
syntax hints, and existence-probe answers work over real bash; the gate-off
configuration leaves submit untouched; and each mode's prompt addendum
describes only its own interface.

Both files reuse `tests/vanillux2/conftest.py`'s fixtures: `ops.exec_fn`
for the in-session side and `isolated_exec_fn` (per-call `cwd`, matching
`self_test.py`'s isolation contract) for the isolated side — real local
subprocesses, not a hand-rolled fake shell. Run with:

```
uv run pytest tests/vanillux2
```
