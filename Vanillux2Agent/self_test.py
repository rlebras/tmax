"""Mechanisms 1-5: acceptance-criteria declaration + an isolated self-test gate.

Grounded in a 445-run failure analysis: of runs that finish and submit a
wrong solution, 67% explicitly declare success on a solution the held-out
grader rejects, and only 22% self-test at all (usually circularly — e.g.
asserting a program's output equals itself). This module gives the model two
tools, ``declare_criteria`` and ``run_check``, and the machinery
``agent.py`` uses to gate the submit sentinel on them.

Architecture
------------
* **Criteria** (Mechanism 1) are a harness-tracked list of ``{id,
  description, how_to_check}`` triples, persisted to
  ``<state_dir>/criteria.json`` — never inlined into the model-facing
  history (the tool call's compact confirmation is all the model sees).
* **Checks** (Mechanism 2) run a model-supplied command and compare it to an
  expected value, returning only PASS/FAIL plus the first failing line —
  never full output. Every check is recorded to ``<state_dir>/checks.json``.
* **Isolation** (Mechanism 3) is what makes the gate hard to game: every
  check also runs against a FRESH copy of only the declared deliverable
  files, mirrored at their real absolute paths under a scratch root, invoked
  as a bare subprocess (no persistent cwd/env chaining — see
  ``agent.py``'s ``_wrap_command``, which ordinary bash calls go through but
  isolated checks deliberately do not). Only the isolated result counts
  toward coverage; a check that passes in-session but fails in isolation is
  surfaced as a discrepancy (the neutered-deliverable / leftover-scratch-file
  failure mode from the analysis above).

  Known scope limitation: this harness has no primitive for spinning up a
  second container/namespace, so isolation is "fresh temp dir + fresh
  process", not a fresh container. A check command that references a
  deliverable by *absolute* path escapes the sandbox (it hits the real
  filesystem, not the isolated copy) — only cwd-relative references are
  actually isolated. Documented rather than silently assumed away.
* **The gate** (Mechanism 4) is enforced by ``agent.py`` around the submit
  sentinel: allowed only once >= ``min_criteria`` declared criteria each have
  >=1 passing isolated check, bounded by ``max_gate_rejections`` so a model
  stuck failing its own checks converts to a flagged
  ``submitted_with_failing_checks`` submit rather than burning the entire
  step budget.
* **Anti-circularity + honesty heuristics** (Mechanism 5) are best-effort,
  non-blocking: a circular/trivial check is flagged and excluded from
  coverage; changing a check right after it failed in isolation is flagged
  as a possible weakening (warn, don't block — a self-test can legitimately
  need fixing).
"""

from __future__ import annotations

import json
import re
import shlex
from dataclasses import asdict, dataclass, field
from typing import Any, Awaitable, Callable

from Vanillux2Agent.container_ops import ContainerOps

SELF_TEST_TOOL_NAMES = frozenset({"declare_criteria", "run_check"})

SELF_TEST_TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "declare_criteria",
            "description": (
                "Declare the testable acceptance criteria for this task, and (optionally) "
                "which file(s) are your deliverable. Call this once, early — most of these "
                "criteria need a passing run_check before you can submit. If the task "
                "statement includes example input/output, turn those into criteria first."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "criteria": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "id": {
                                    "type": "string",
                                    "description": "Short, stable identifier, e.g. 'handles_empty_input'.",
                                },
                                "description": {
                                    "type": "string",
                                    "description": "The observable property and its expected outcome.",
                                },
                                "how_to_check": {
                                    "type": "string",
                                    "description": (
                                        "How you plan to verify this. Informational — the actual "
                                        "verification is a separate run_check call."
                                    ),
                                },
                            },
                            "required": ["id", "description", "how_to_check"],
                        },
                    },
                    "deliverables": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "Paths to the file(s) that make up your solution. run_check copies "
                            "only these into an isolated sandbox before verifying a criterion — "
                            "anything not listed here (scratch files, test helpers) will NOT be "
                            "present there."
                        ),
                    },
                },
                "required": ["criteria"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_check",
            "description": (
                "Verify one declared criterion: run `command` and compare its output to "
                "`expected` (omit `expected` to just require exit code 0). The check also runs "
                "in an ISOLATED copy of your declared deliverables, not your live scratch state "
                "— a check that only passes because of leftover files or a weakened deliverable "
                "will fail there. Returns a compact pass/fail result, never full output. Use an "
                "independent oracle (a known answer, an invariant, a second method) — a check "
                "that just re-runs the program and compares it to itself does not count."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "criterion_id": {"type": "string"},
                    "command": {"type": "string"},
                    "expected": {
                        "type": "string",
                        "description": "Exact expected stdout+stderr (trimmed). Omit to just require exit code 0.",
                    },
                },
                "required": ["criterion_id", "command"],
            },
        },
    },
]


@dataclass
class Criterion:
    id: str
    description: str
    how_to_check: str


@dataclass
class CheckRecord:
    criterion_id: str
    command: str
    expected: str | None
    session_pass: bool
    isolated_pass: bool
    first_failure_line: str | None
    circular: bool
    circular_reason: str | None
    step: int


@dataclass
class SelfTestConfig:
    min_criteria: int = 2
    max_gate_rejections: int = 3
    state_dir: str = "/tmp/.vanillux2/self_test"
    isolation_root: str = "/tmp/.vanillux2/self_test/isolated"
    check_timeout_sec: int = 60


@dataclass
class SelfTestState:
    criteria: dict[str, Criterion] = field(default_factory=dict)
    deliverables: list[str] = field(default_factory=list)
    checks: dict[str, list[CheckRecord]] = field(default_factory=dict)
    gate_rejections: int = 0


@dataclass
class GateResult:
    allowed: bool
    reasons: list[str]
    forced: bool = False  # True only when allowed because rejections were exhausted


IsolatedExecFn = Callable[[str, str], Awaitable[Any]]


# ---------------------------------------------------------------------------
# declare_criteria
# ---------------------------------------------------------------------------


def _declare_criteria(args: dict, state: SelfTestState) -> str:
    raw_criteria = args.get("criteria") or []
    added: list[str] = []
    for c in raw_criteria:
        if not isinstance(c, dict):
            continue
        cid = str(c.get("id") or "").strip()
        description = str(c.get("description") or "").strip()
        how_to_check = str(c.get("how_to_check") or "").strip()
        if not cid or not description:
            continue
        state.criteria[cid] = Criterion(id=cid, description=description, how_to_check=how_to_check)
        added.append(cid)

    for d in args.get("deliverables") or []:
        d = str(d).strip()
        if d and d not in state.deliverables:
            state.deliverables.append(d)

    if not added:
        return "declare_criteria: no valid criteria given — each needs a non-empty id, description, and how_to_check"

    return (
        f"declared {len(added)} criteria: {', '.join(added)} "
        f"({len(state.deliverables)} deliverable file(s) tracked)"
    )


# ---------------------------------------------------------------------------
# Mechanism 5 — anti-circularity + weakening heuristics
# ---------------------------------------------------------------------------

_TRIVIAL_COMMAND_RE = re.compile(r"^\s*(true|:|exit\s+0)\s*$")
_ASSERTION_HINTS = ("assert", "==", "!=", "diff", "grep", "pytest", "unittest", "[[", "[ ", "cmp ", "test ")
_SELF_DIFF_RE = re.compile(r"diff\s+<\((?P<a>.+?)\)\s+<\((?P<b>.+?)\)")
_SELF_EQ_RE = re.compile(r"\[\[?\s*[\"']?\$\((?P<a>.+?)\)[\"']?\s*(?:==|=)\s*[\"']?\$\((?P<b>.+?)\)[\"']?\s*\]\]?")


def _circularity_flags(command: str, expected: str | None) -> list[str]:
    """Best-effort detection of trivially/circularly passing checks (heuristic, not a proof)."""
    flags: list[str] = []
    cmd = command or ""
    exp = (expected or "").strip()

    if _TRIVIAL_COMMAND_RE.match(cmd):
        flags.append("command is a no-op (`true`/`:`/`exit 0`) with no real assertion")
    elif not exp and not any(h in cmd for h in _ASSERTION_HINTS):
        flags.append("no `expected` value and no recognizable assertion in `command`")

    if exp and re.search(r"\$\(|`", exp):
        flags.append("`expected` is itself computed at check time, not a fixed known-answer")

    for rx in (_SELF_DIFF_RE, _SELF_EQ_RE):
        m = rx.search(cmd)
        if m and m.group("a").strip() and m.group("a").strip() == m.group("b").strip():
            flags.append("compares the program's output to itself — no independent oracle")
            break

    return flags


def _weakening_warning(criterion_id: str, command: str, expected: str | None, state: SelfTestState) -> str | None:
    history = state.checks.get(criterion_id) or []
    if not history:
        return None
    last = history[-1]
    if last.isolated_pass:
        return None
    if command == last.command and (expected or "") == (last.expected or ""):
        return None
    return (
        f"check {criterion_id!r} was changed after its last isolated run FAILED — "
        "confirm the criterion still holds, don't just loosen the check"
    )


# ---------------------------------------------------------------------------
# Mechanism 3 — isolation
# ---------------------------------------------------------------------------

_UNSAFE_ID_RE = re.compile(r"[^A-Za-z0-9_.-]")


def _safe_dir_name(criterion_id: str) -> str:
    """Sanitize a model-supplied criterion_id before it's used as a path component."""
    safe = _UNSAFE_ID_RE.sub("_", criterion_id).strip(".") or "check"
    return safe


async def _prepare_isolated_root(
    ops: ContainerOps, config: SelfTestConfig, state: SelfTestState, criterion_id: str
) -> tuple[str, list[str]]:
    """Build a fresh directory containing only the declared deliverables, mirrored at
    their real absolute paths. Returns ``(isolated_cwd, missing_deliverable_paths)``.
    """
    root = f"{config.isolation_root}/{_safe_dir_name(criterion_id)}"
    await ops.exec_fn(f"rm -rf {shlex.quote(root)} && mkdir -p {shlex.quote(root)}")

    cwd_result = await ops.exec_fn("pwd")
    cwd = (cwd_result.stdout or "/").strip() or "/"
    isolated_cwd = root if cwd == "/" else f"{root}{cwd}"
    await ops.exec_fn(f"mkdir -p {shlex.quote(isolated_cwd)}")

    missing: list[str] = []
    for path in state.deliverables:
        abs_path = path if path.startswith("/") else f"{cwd.rstrip('/')}/{path}"
        dest = f"{root}{abs_path}"
        dest_dir = dest.rsplit("/", 1)[0]
        mkdir_res = await ops.exec_fn(f"mkdir -p {shlex.quote(dest_dir)}")
        if mkdir_res.return_code != 0:
            missing.append(path)
            continue
        try:
            data = await ops.download_bytes(abs_path)
        except Exception:
            missing.append(path)
            continue
        await ops.upload_bytes(data, dest)
        await ops.exec_fn(f"chmod +x {shlex.quote(dest)} 2>/dev/null || true")

    return isolated_cwd, missing


# ---------------------------------------------------------------------------
# run_check
# ---------------------------------------------------------------------------


def _combine_output(result: Any) -> str:
    out = getattr(result, "stdout", None) or ""
    err = getattr(result, "stderr", None) or ""
    if err:
        out += f"\n{err}" if out else err
    return out.rstrip()


def _judge(output: str, result: Any, expected: str | None) -> bool:
    if expected is not None and expected.strip():
        return output.strip() == expected.strip()
    return getattr(result, "return_code", 1) == 0


def _first_failure_line(output: str) -> str | None:
    for line in output.splitlines():
        line = line.strip()
        if line:
            return line[:300]
    return None


async def _persist(ops: ContainerOps, config: SelfTestConfig, state: SelfTestState) -> None:
    criteria_doc = {
        "criteria": [asdict(c) for c in state.criteria.values()],
        "deliverables": state.deliverables,
    }
    checks_doc = {cid: [asdict(r) for r in records] for cid, records in state.checks.items()}
    await ops.exec_fn(f"mkdir -p {shlex.quote(config.state_dir)}")
    await ops.upload_bytes(json.dumps(criteria_doc, indent=2).encode("utf-8"), f"{config.state_dir}/criteria.json")
    await ops.upload_bytes(json.dumps(checks_doc, indent=2).encode("utf-8"), f"{config.state_dir}/checks.json")


async def _run_check(
    args: dict,
    state: SelfTestState,
    ops: ContainerOps,
    isolated_exec: IsolatedExecFn,
    config: SelfTestConfig,
    step: int,
) -> str:
    criterion_id = str(args.get("criterion_id") or "").strip()
    command = str(args.get("command") or "").strip()
    expected_raw = args.get("expected")
    expected = str(expected_raw) if expected_raw is not None else None

    if not criterion_id:
        return "run_check: criterion_id is required"
    if criterion_id not in state.criteria:
        return f"run_check: unknown criterion_id {criterion_id!r} — call declare_criteria first"
    if not command:
        return f"run_check {criterion_id}: command is required"

    circular_flags = _circularity_flags(command, expected)
    weaken_warning = _weakening_warning(criterion_id, command, expected, state)

    session_result = await ops.exec_fn(command)
    session_output = _combine_output(session_result)
    session_pass = _judge(session_output, session_result, expected)

    isolated_cwd, missing = await _prepare_isolated_root(ops, config, state, criterion_id)
    isolated_result = await isolated_exec(command, isolated_cwd)
    isolated_output = _combine_output(isolated_result)
    isolated_pass = _judge(isolated_output, isolated_result, expected)

    record = CheckRecord(
        criterion_id=criterion_id,
        command=command,
        expected=expected,
        session_pass=session_pass,
        isolated_pass=isolated_pass,
        first_failure_line=None if isolated_pass else _first_failure_line(isolated_output),
        circular=bool(circular_flags),
        circular_reason="; ".join(circular_flags) if circular_flags else None,
        step=step,
    )
    state.checks.setdefault(criterion_id, []).append(record)
    await _persist(ops, config, state)

    lines = [f"check {criterion_id}: {'PASS' if isolated_pass else 'FAIL'} (isolated)"]
    if not isolated_pass:
        lines.append(f"first failing line: {record.first_failure_line or '(no output)'}")
    if session_pass and not isolated_pass:
        lines.append(
            "WARNING: passed in your session but FAILED in isolation — likely relying on "
            "leftover state or a weakened deliverable, not your declared files"
        )
    if missing:
        lines.append(f"WARNING: {len(missing)} declared deliverable(s) not found on disk: {', '.join(missing)}")
    if circular_flags:
        lines.append(f"WARNING: check looks circular/trivial ({circular_flags[0]}) — not counted toward coverage")
    if weaken_warning:
        lines.append(f"WARNING: {weaken_warning}")
    return "\n".join(lines)


async def dispatch(
    name: str,
    args: dict,
    state: SelfTestState,
    ops: ContainerOps,
    isolated_exec: IsolatedExecFn,
    config: SelfTestConfig,
    step: int,
) -> str:
    """Route a self-test tool call; never raises — always returns model-facing text."""
    try:
        if name == "declare_criteria":
            result = _declare_criteria(args, state)
            await _persist(ops, config, state)
            return result
        if name == "run_check":
            return await _run_check(args, state, ops, isolated_exec, config, step)
        return f"error: unknown self-test tool '{name}'"
    except Exception as exc:  # defensive: a bad call must not crash the run
        return f"error: {type(exc).__name__}: {exc}"


# ---------------------------------------------------------------------------
# Mechanism 4 — the submission gate
# ---------------------------------------------------------------------------


def coverage(state: SelfTestState) -> dict[str, bool]:
    """Which declared criteria have >=1 non-circular, isolated-passing check."""
    return {
        cid: any(r.isolated_pass and not r.circular for r in (state.checks.get(cid) or []))
        for cid in state.criteria
    }


def evaluate_gate(state: SelfTestState, config: SelfTestConfig) -> GateResult:
    covered_ids: list[str] = []
    uncovered_ids: list[str] = []
    if state.criteria:
        cov = coverage(state)
        covered_ids = [cid for cid, ok in cov.items() if ok]
        uncovered_ids = [cid for cid, ok in cov.items() if not ok]
        if len(covered_ids) >= config.min_criteria:
            return GateResult(True, [])

    # The rejection bound is a universal safety valve — it applies whether
    # the model never called declare_criteria at all or just hasn't gotten
    # enough checks passing yet, so a model stuck on either failure mode
    # still converts to a flagged submit rather than burning the whole step
    # budget (see self_test.py's module docstring / the task's risk section).
    if state.gate_rejections >= config.max_gate_rejections:
        return GateResult(
            True, [f"proceeding after {state.gate_rejections} rejected submit attempts"], forced=True
        )

    if not state.criteria:
        return GateResult(False, ["no acceptance criteria declared — call declare_criteria first"])

    reasons = [f"only {len(covered_ids)}/{config.min_criteria} required criteria have a passing isolated check"]
    if uncovered_ids:
        reasons.append(f"unmet: {', '.join(uncovered_ids)}")
    return GateResult(False, reasons)


def gate_nudge(gate: GateResult) -> str:
    return (
        "Submit rejected — " + "; ".join(gate.reasons) + ". "
        "Keep working, then call run_check again once fixed, or declare_criteria if you haven't yet."
    )
