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

This is a pure harness change — no model, prompt, or training changes. It
adds two tools (`declare_criteria`, `run_check`) and gates the
`COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT` sentinel on them; it composes with
the context-management harness (criteria/checks are file-backed state, never
inlined into history at full size, and `run_check`'s result goes through the
same compact-result discipline as everything else in this harness).

## Mechanism 1 — acceptance-criteria declaration

`declare_criteria` lets the model declare a list of `{id, description,
how_to_check}` triples — the observable properties the task actually asks
for — plus, optionally, which file(s) are its deliverable. The system/
instance prompt addenda (`self_test_system_addendum` /
`self_test_instance_addendum` in `vanillux_prompts.yaml`, appended to
whichever base template `enable_edit_tools` selects) tell the model to call
this early and to turn any example input/output in the task statement into
criteria first. State is persisted to `<state_dir>/criteria.json` — never
inlined into the model-facing history; the tool call's compact confirmation
(`"declared N criteria: ..."`) is all the model sees back.

## Mechanism 2 — `run_check`: a structured, compact check

`run_check(criterion_id, command, expected)` runs `command` and compares its
output to `expected` (omit `expected` to just require exit code 0). The
model only ever sees `PASS`/`FAIL` plus, on failure, the first failing line
— never full output — so a check on a command that dumps hundreds of lines
of noise costs the same handful of tokens as one that prints nothing. Every
check is recorded to `<state_dir>/checks.json` (`criterion_id -> [{command,
session_pass, isolated_pass, first_failure_line, circular, ...}, ...]`) so
coverage and the anti-weakening heuristic (Mechanism 5) can look back at
history.

## Mechanism 3 — isolation (what makes the gate hard to game)

Every check runs **twice**: once in-session (the persistent bash shell the
model has been using), and once against a **fresh copy of only the declared
deliverables**, mirrored at their real absolute paths under a scratch root
(`<isolation_root>/<criterion_id>/...`), invoked as a bare subprocess with no
persistent cwd/env chaining (`agent.py`'s `isolated_exec` deliberately
bypasses `_wrap_command` — see that function for the persistent-shell
wrapper ordinary bash calls go through but isolated checks do not). Only the
isolated result counts toward coverage. A check that passes in-session but
fails in isolation is surfaced as an explicit warning — this is the direct
counter to the "neutered filter, checked against a leftover known-good
backup" failure mode: the backup is real, on disk, and genuinely produces a
passing result in the session shell, but it was never declared as a
deliverable, so it simply isn't there in isolation.

**Known scope limitation, stated plainly:** this harness has no primitive
for spinning up a second container or namespace — `BaseAgent` only has
`environment.exec`/`upload_file`/`download_file`. So isolation here is "fresh
temp dir + fresh process", exactly the minimum the mechanism calls for, not a
fresh container. A check command that references a deliverable by
**absolute** path escapes the sandbox (it resolves against the real
filesystem, not the isolated copy) — only cwd-relative references are
actually isolated. Real task-solving bash commands overwhelmingly use
relative paths (`cd /app && python3 -m pytest test_x.py`), so this covers the
realistic case, but it is not a hard security boundary and isn't claimed to
be one.

## Mechanism 4 — the submission gate

`agent.py` intercepts the `COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT` sentinel
(the bash `echo` still runs normally; only the loop's finish decision is
gated). Submit is allowed only once:

- at least one criterion has been declared, **and**
- at least `min_criteria` declared criteria each have >=1 check that passed
  in isolation (a circular/trivial check, Mechanism 5, does not count).

Otherwise the submit is rejected: the loop does **not** break, and a compact
nudge listing what's unmet is appended as a `user`-role message (there's no
tool_call_id to reuse — the bash `echo` call already got its own tool
response). Rejections are bounded by `max_gate_rejections`: once exhausted,
submit is allowed regardless of coverage, and
`submitted_with_failing_checks=True` is recorded in `context.metadata` and
`self_test.json`. **This bound applies uniformly** — it fires whether the
model never called `declare_criteria` at all or just hasn't gotten enough
checks passing yet — so a model stuck on either failure mode still converts
to a flagged submit rather than burning the entire step budget (see
"Risk" below).

## Mechanism 5 — anti-circularity + honesty heuristics

Best-effort, harness-side, and **non-blocking** (a self-test can legitimately
be wrong — these only warn):

- **Circularity/triviality** (`_circularity_flags`): a no-op command
  (`true`/`:`/`exit 0`) with nothing to fail; no `expected` value and no
  recognizable assertion in `command`; an `expected` that's itself computed
  at check time (contains a command substitution) rather than a fixed
  known-answer; or a command that diffs/compares a program's output against
  itself (`diff <(prog) <(prog)`, `[ "$(prog)" = "$(prog)" ]`). Flagged
  checks are excluded from coverage even if they "pass".
- **Weakening** (`_weakening_warning`): if the most recent check recorded
  for a criterion failed in isolation, and the next `run_check` call for
  that same criterion changes the command or expected value, a warning
  fires ("confirm the criterion still holds, don't just loosen the check").
  This is a coarse proxy — genuinely distinguishing "fixed the check" from
  "weakened the check" isn't decidable from the command text alone — so it
  warns on every edit-after-failure rather than trying to be clever about
  which edits are suspicious.

## Config flags

All on `Vanillux2Agent.__init__`:

| Flag | Default | Effect |
|---|---|---|
| `enable_self_test_gate` | `True` | Register `declare_criteria`/`run_check` and gate the submit sentinel on them. |
| `min_criteria` | `2` | Minimum declared criteria that must have a passing isolated check before submit is allowed. |
| `max_gate_rejections` | `3` | After this many rejected submit attempts, allow submit anyway and record `submitted_with_failing_checks`. |
| `self_test_isolation_mode` | `"tempdir"` | Only mode implemented (see Mechanism 3's scope limitation); reserved for a future stronger-isolation mode. |

`context.metadata["self_test"]` reports `criteria_declared`, `checks_run`,
`gate_rejections`, and `submitted_with_failing_checks` for the run;
`<logs_dir>/self_test.json` additionally records per-criterion coverage and a
compact per-check history (not full command/output — that's on disk under
`state_dir`/`isolation_root` inside the task container, not in the harness's
own logs).

## Risk this is designed to manage

A hard gate can convert "finished-wrong" into "step-exhaustion" or
"context-run-out" if the model keeps failing its own checks — context
management's own history shows failures merely *relocating* under a naive
change. Three things bound that risk here: `max_gate_rejections` (Mechanism
4) forces a flagged submit rather than an infinite retry loop; `run_check`'s
compact pass/fail-only result (Mechanism 2) keeps verification token-cheap
regardless of how much output a check's command produces; and the two
per-check executions (session + isolated) are real subprocess calls, not
extra LLM calls, so the gate's cost shows up in wall-clock/compute, not in
context or step budget.

**Not yet measured, stated plainly:** unlike the context-management report
(which replays already-completed trajectories offline), evaluating this
gate's actual effect on the finished-wrong / step-exhaustion / context-run-out
taxonomy requires a **live rerun** against a real model and environment —
the gate changes what the agent *does*, not just how a fixed trajectory gets
represented. That rerun (and the resulting outcome-taxonomy, adoption-rate,
and step/token-overhead numbers) hasn't been executed as part of this change;
see the top-level task report for what's needed to produce it.

## Tests

[`tests/vanillux2/test_self_test.py`](../tests/vanillux2/test_self_test.py)
covers: `declare_criteria` validation, the gate rejecting with no
criteria/insufficient coverage and accepting once `min_criteria` are covered,
the rejection bound applying uniformly (including with zero criteria
declared), the neutered-filter/leftover-backup fixture failing in isolation
while passing in-session, a genuinely broken deliverable failing in both,
missing-deliverable reporting, circular/trivial checks being flagged and
excluded from coverage (both via `_circularity_flags` directly and
end-to-end through `run_check`), compact-result-only output on both large
and small outputs, the weakening warning firing only on a changed check
after a failure, `dispatch()`'s error handling, and criteria/checks
persistence to disk.

[`tests/vanillux2/test_agent_self_test_gate.py`](../tests/vanillux2/test_agent_self_test_gate.py)
drives the actual `Vanillux2Agent.run()` loop end to end (a scripted fake
model via a monkeypatched `litellm.completion`, a local-filesystem fake
`BaseEnvironment` — no docker, no live LLM) to verify the wiring itself: a
premature submit is rejected with a `user`-role nudge appended to the
trajectory and the run continues; a second, fully-covered submit is
accepted; and a model that never calls `declare_criteria` at all still
terminates once `max_gate_rejections` is exhausted, with
`submitted_with_failing_checks=True` recorded.

Both files reuse `tests/vanillux2/conftest.py`'s fixtures: `ops.exec_fn` for
the in-session side, and a new `isolated_exec_fn` fixture (per-call `cwd`,
matching `self_test.py`'s isolation contract, vs. `ops.exec_fn`'s fixed cwd)
for the isolated side — both are real local subprocesses, not a hand-rolled
fake shell. Run with:

```
uv run pytest tests/vanillux2
```
