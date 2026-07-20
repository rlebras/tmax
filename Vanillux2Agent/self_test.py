"""Mechanisms 1-5: acceptance-criteria declaration + an isolated self-test gate.

Grounded in a 445-run failure analysis: of runs that finish and submit a
wrong solution, 67% explicitly declare success on a solution the held-out
grader rejects, and only 22% self-test at all (usually circularly — e.g.
asserting a program's output equals itself). This module gives the model a
harness-intercepted bash convention (no new tools, no change to the
bash-only tool contract) plus the machinery ``agent.py`` uses to gate the
submit sentinel on it.

Architecture
------------
* **Criteria** (Mechanism 1) are declared by the agent itself, as an ordinary
  bash write of a JSON file to ``<state_dir>/criteria.json`` (never a new
  tool call) — ``{criteria: [{id, description, how_to_check}], deliverables:
  [path, ...]}``. The harness re-reads this file fresh whenever it needs the
  current declaration; it is never inlined into the model-facing history.
* **Checks** (Mechanism 2) are declared via the bash convention
  ``agent-check <id> -- <command>``, intercepted before it ever reaches the
  shell (there is no real ``agent-check`` binary) — either as the whole bash
  call, or as one line within a larger multi-line call that also does setup
  on other lines (``extract_agent_check_line``; the "remainder" runs
  normally first). A line starting with the ``agent-check`` token that
  matches neither form gets a corrective ``AGENT_CHECK_SYNTAX_HINT`` instead
  of silently hitting the real shell as "command not found" (a real failure
  mode observed in a 445-trial A/B: 12% of trials tried the convention and
  got a shell error because they'd bundled it into a larger call the
  original whole-command-only matcher couldn't see). ``command`` runs once
  and the check passes iff it exits 0 — the standard bash idiom (``test``,
  ``diff``, ``grep -q``, ``pytest``, ...) already encodes the comparison, so
  no separate "expected value" protocol is needed. Only PASS/FAIL plus the
  first failing line is ever returned — never full output.
* **Isolation** (Mechanism 3) is what makes the gate hard to game: every
  check also runs against a FRESH directory containing only the declared
  deliverable files, copied to their real absolute paths, invoked as a bare
  ``environment.exec(..., cwd=isolated_root)`` — no persistent env/cwd
  chaining (see ``agent.py``'s ``_wrap_command``, which ordinary bash calls
  go through but isolated checks deliberately bypass). Only the isolated
  result counts toward coverage; a check that passes in-session but fails in
  isolation is surfaced as a discrepancy (the neutered-deliverable /
  leftover-scratch-file failure mode from the analysis above).

  Known scope limitation: this harness has no primitive for spinning up a
  second container/namespace (the Docker environment only exposes ``docker
  compose exec``, no container ID), so isolation is "fresh temp dir + fresh
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
  coverage; changing a check's command right after it failed in isolation is
  flagged as a possible weakening (warn, don't block — a self-test can
  legitimately need fixing).
"""

from __future__ import annotations

import json
import re
import shlex
from dataclasses import asdict, dataclass, field
from typing import Any, Awaitable, Callable, Protocol

SELF_TEST_STATE_SUBDIR = "self_test"


class ExecFn(Protocol):
    async def __call__(self, command: str) -> Any: ...


@dataclass
class Criterion:
    id: str
    description: str
    how_to_check: str = ""


@dataclass
class CheckRecord:
    criterion_id: str
    command: str
    session_pass: bool
    isolated_pass: bool
    first_failure_line: str | None
    circular: bool
    circular_reason: str | None
    missing_deliverables: list[str]
    step: int


@dataclass
class SelfTestConfig:
    min_criteria: int = 2
    max_gate_rejections: int = 3
    check_timeout_sec: int = 60
    state_dir: str = "/tmp/.vanillux2/self_test"
    isolation_mode: str = "tmpdir"  # only mode currently implemented; see module docstring

    @property
    def criteria_path(self) -> str:
        return f"{self.state_dir}/criteria.json"

    @property
    def isolation_root(self) -> str:
        return f"{self.state_dir}/isolated"


@dataclass
class SelfTestState:
    checks: dict[str, list[CheckRecord]] = field(default_factory=dict)
    gate_rejections: int = 0
    submitted_with_failing_checks: bool = False


@dataclass
class GateResult:
    allowed: bool
    reasons: list[str]
    forced: bool = False  # True only when allowed because rejections were exhausted


IsolatedExecFn = Callable[[str, str], Awaitable[Any]]


# ---------------------------------------------------------------------------
# agent-check convention parsing
#
# Two supported forms:
#  1. Whole-command: the ENTIRE bash call is one agent-check invocation. Its
#     inner command may itself be multi-line (e.g. a heredoc/python -c block).
#  2. Line-embedded: `agent-check <id> -- <command>` appears as ONE LINE
#     within a larger multi-line bash call that also does setup on other
#     lines (very common in practice — models naturally bundle `mkdir ...`
#     or comments alongside the check). The inner command is limited to that
#     single line; everything else in the call ("the remainder") is run
#     normally by the caller before the check executes.
# A line that starts with the `agent-check` token but matches neither form
# (missing ` -- `, empty id/command, ...) is almost certainly a botched
# attempt rather than a comment or string literal — `looks_like_malformed_
# agent_check` flags it so the caller can return a corrective hint instead of
# letting the literal (non-existent) binary hit the real shell.
# ---------------------------------------------------------------------------

AGENT_CHECK_RE = re.compile(r"^\s*agent-check\s+(?P<id>\S+)\s+--\s+(?P<cmd>.+)$", re.DOTALL)
# Same-line whitespace only (`[ \t]+`, not `\s+`) between tokens — unlike the
# whole-command form above, this must NOT be able to span a newline, or a
# check with no command on its own line would silently swallow the next
# line's command as its own (a real bug caught by extract_agent_check_line's
# own tests).
AGENT_CHECK_LINE_RE = re.compile(r"(?m)^[ \t]*agent-check[ \t]+(?P<id>\S+)[ \t]+--[ \t]+(?P<cmd>.+?)[ \t]*$")
AGENT_CHECK_ATTEMPT_RE = re.compile(r"(?m)^[ \t]*agent-check\b")


def parse_agent_check(command: str) -> tuple[str, str] | None:
    """Return ``(criterion_id, inner_command)`` if the ENTIRE ``command`` is
    one ``agent-check <id> -- <command>`` invocation, else ``None``."""
    m = AGENT_CHECK_RE.match(command)
    if not m:
        return None
    cmd = m.group("cmd").strip()
    if not cmd:
        return None
    return m.group("id"), cmd


def extract_agent_check_line(command: str) -> tuple[str, str, str] | None:
    """Find an ``agent-check <id> -- <command>`` invocation embedded as one
    line within a larger multi-line ``command``. Returns ``(criterion_id,
    inner_command, remainder)`` where ``remainder`` is ``command`` with that
    line removed (run normally by the caller, e.g. setup on other lines) —
    or ``None`` if no such line is present.
    """
    m = AGENT_CHECK_LINE_RE.search(command)
    if not m:
        return None
    cmd = m.group("cmd").strip()
    if not cmd:
        return None
    start, end = m.span()
    if end < len(command) and command[end] == "\n":
        end += 1  # swallow the line's own trailing newline, not just its content
    remainder = (command[:start] + command[end:]).strip()
    return m.group("id"), cmd, remainder


def looks_like_malformed_agent_check(command: str) -> bool:
    """True if ``command`` contains a line that starts with the literal
    ``agent-check`` token but matches neither supported form above — a
    near-certain botched attempt (comments/string literals don't start a
    shell line with this exact token), not a false positive worth chasing.
    """
    if parse_agent_check(command) or extract_agent_check_line(command):
        return False
    return bool(AGENT_CHECK_ATTEMPT_RE.search(command))


AGENT_CHECK_SYNTAX_HINT = (
    "agent-check: syntax not recognized — nothing was run. It must be the "
    "exact form `agent-check <id> -- <command>` (id has no spaces; ` -- ` "
    "separates it from <command>), either as the whole bash call or as its "
    "own line within a larger call (setup on other lines still runs "
    "normally). Example combined with setup:\n"
    "  mkdir -p /app/out\n"
    "  agent-check my_id -- test -f /app/out/result.txt"
)


# ---------------------------------------------------------------------------
# Mechanism 1 — reading the agent's declared criteria file
# ---------------------------------------------------------------------------


def parse_criteria_doc(raw: str) -> tuple[dict[str, Criterion], list[str]]:
    """Parse the criteria.json contents. Never raises — malformed/missing
    input yields an empty declaration."""
    try:
        doc = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return {}, []
    if not isinstance(doc, dict):
        return {}, []

    criteria: dict[str, Criterion] = {}
    for c in doc.get("criteria") or []:
        if not isinstance(c, dict):
            continue
        cid = str(c.get("id") or "").strip()
        description = str(c.get("description") or "").strip()
        if not cid or not description:
            continue
        criteria[cid] = Criterion(
            id=cid, description=description, how_to_check=str(c.get("how_to_check") or "").strip()
        )

    deliverables = [str(d).strip() for d in (doc.get("deliverables") or []) if str(d).strip()]
    return criteria, deliverables


async def load_criteria(exec_fn: ExecFn, config: SelfTestConfig) -> tuple[dict[str, Criterion], list[str]]:
    result = await exec_fn(f"cat {shlex.quote(config.criteria_path)} 2>/dev/null")
    return parse_criteria_doc(getattr(result, "stdout", None) or "")


# ---------------------------------------------------------------------------
# Mechanism 5 — anti-circularity + weakening heuristics
# ---------------------------------------------------------------------------

_TRIVIAL_COMMAND_RE = re.compile(r"^\s*(true|:|exit\s+0)\s*$")
_ASSERTION_HINTS = ("assert", "==", "!=", "diff", "grep", "pytest", "unittest", "[[", "[ ", "cmp ", "test ")
_SELF_DIFF_RE = re.compile(r"diff\s+<\((?P<a>.+?)\)\s+<\((?P<b>.+?)\)")
_SELF_EQ_RE = re.compile(r"\[\[?\s*[\"']?\$\((?P<a>.+?)\)[\"']?\s*(?:==|=)\s*[\"']?\$\((?P<b>.+?)\)[\"']?\s*\]\]?")


def circularity_flags(command: str) -> list[str]:
    """Best-effort detection of trivially/circularly passing checks (heuristic, not a proof)."""
    flags: list[str] = []
    cmd = command or ""

    if _TRIVIAL_COMMAND_RE.match(cmd):
        flags.append("command is a no-op (`true`/`:`/`exit 0`) with no real assertion")
    elif not any(h in cmd for h in _ASSERTION_HINTS):
        flags.append("no recognizable assertion in `command` (no test/diff/grep/assert/pytest/...)")

    for rx in (_SELF_DIFF_RE, _SELF_EQ_RE):
        m = rx.search(cmd)
        if m and m.group("a").strip() and m.group("a").strip() == m.group("b").strip():
            flags.append("compares the program's output to itself — no independent oracle")
            break

    return flags


def weakening_warning(criterion_id: str, command: str, state: SelfTestState) -> str | None:
    history = state.checks.get(criterion_id) or []
    if not history:
        return None
    last = history[-1]
    if last.isolated_pass:
        return None
    if command == last.command:
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
    return _UNSAFE_ID_RE.sub("_", criterion_id).strip(".") or "check"


_MISSING_MARKER = "__SELF_TEST_MISSING__:"


async def prepare_isolated_root(
    exec_fn: ExecFn, config: SelfTestConfig, deliverables: list[str], cwd: str, criterion_id: str
) -> tuple[str, list[str]]:
    """Build a fresh directory containing only the declared deliverables, mirrored at
    their real absolute paths. Returns ``(isolated_cwd, missing_deliverable_paths)``.

    Batched into a SINGLE ``exec_fn`` round trip (one per check was costing
    2 + 2*len(deliverables) container round trips — a real chunk of the
    latency/wall-clock overhead the gate adds per check) — each deliverable
    copy is wrapped in ``|| echo MISSING:...`` so one failure doesn't abort
    the rest, and missing ones are recovered by scanning stdout afterward.
    """
    root = f"{config.isolation_root}/{_safe_dir_name(criterion_id)}"
    cwd = cwd or "/"
    isolated_cwd = root if cwd == "/" else f"{root}{cwd}"

    parts = [
        f"rm -rf {shlex.quote(root)}",
        f"mkdir -p {shlex.quote(isolated_cwd)}",
    ]
    for path in deliverables:
        abs_path = path if path.startswith("/") else f"{cwd.rstrip('/')}/{path}"
        dest = f"{root}{abs_path}"
        dest_dir = dest.rsplit("/", 1)[0]
        marker = shlex.quote(f"{_MISSING_MARKER}{path}")
        parts.append(
            f"(mkdir -p {shlex.quote(dest_dir)} && cp -a {shlex.quote(abs_path)} {shlex.quote(dest)}) "
            f"|| echo {marker}"
        )
    result = await exec_fn("\n".join(parts))
    output = getattr(result, "stdout", None) or ""
    missing = [
        line[len(_MISSING_MARKER):] for line in output.splitlines() if line.startswith(_MISSING_MARKER)
    ]
    return isolated_cwd, missing


# ---------------------------------------------------------------------------
# agent-check execution
# ---------------------------------------------------------------------------


def _combine_output(result: Any) -> str:
    out = getattr(result, "stdout", None) or ""
    err = getattr(result, "stderr", None) or ""
    if err:
        out += f"\n{err}" if out else err
    return out.rstrip()


def _first_failure_line(output: str) -> str | None:
    for line in output.splitlines():
        line = line.strip()
        if line:
            return line[:300]
    return None


async def persist_state(exec_fn: ExecFn, config: SelfTestConfig, state: SelfTestState) -> None:
    doc = {
        "checks": {cid: [asdict(r) for r in records] for cid, records in state.checks.items()},
        "gate_rejections": state.gate_rejections,
        "submitted_with_failing_checks": state.submitted_with_failing_checks,
    }
    script = (
        f"mkdir -p {shlex.quote(config.state_dir)} && "
        f"cat > {shlex.quote(config.state_dir)}/checks.json <<'__SELF_TEST_STATE__'\n"
        f"{json.dumps(doc, indent=2)}\n__SELF_TEST_STATE__"
    )
    await exec_fn(script)


async def run_agent_check(
    criterion_id: str,
    command: str,
    *,
    criteria: dict[str, Criterion],
    deliverables: list[str],
    state: SelfTestState,
    config: SelfTestConfig,
    session_exec: ExecFn,
    isolated_exec: IsolatedExecFn,
    persistent_cwd: str,
    step: int,
) -> str:
    """Run one ``agent-check`` invocation; never raises — always returns
    model-facing text."""
    if criterion_id not in criteria:
        return f"agent-check: unknown criterion_id {criterion_id!r} — declare it in {config.criteria_path} first"
    if not command:
        return f"agent-check {criterion_id}: command is required after `--`"

    flags = circularity_flags(command)
    weaken_warning = weakening_warning(criterion_id, command, state)

    session_result = await session_exec(command)
    session_pass = getattr(session_result, "return_code", 1) == 0

    isolated_cwd, missing = await prepare_isolated_root(
        session_exec, config, deliverables, persistent_cwd, criterion_id
    )
    isolated_result = await isolated_exec(command, isolated_cwd)
    isolated_output = _combine_output(isolated_result)
    isolated_pass = getattr(isolated_result, "return_code", 1) == 0

    record = CheckRecord(
        criterion_id=criterion_id,
        command=command,
        session_pass=session_pass,
        isolated_pass=isolated_pass,
        first_failure_line=None if isolated_pass else _first_failure_line(isolated_output),
        circular=bool(flags),
        circular_reason="; ".join(flags) if flags else None,
        missing_deliverables=missing,
        step=step,
    )
    state.checks.setdefault(criterion_id, []).append(record)
    await persist_state(session_exec, config, state)

    lines = [f"agent-check {criterion_id}: {'PASS' if isolated_pass else 'FAIL'} (isolated)"]
    if not isolated_pass:
        lines.append(f"first failing line: {record.first_failure_line or '(no output)'}")
    if session_pass and not isolated_pass:
        lines.append(
            "WARNING: passed in your session but FAILED in isolation — likely relying on "
            "leftover state or a weakened deliverable, not your declared files"
        )
    if missing:
        lines.append(f"WARNING: {len(missing)} declared deliverable(s) not found on disk: {', '.join(missing)}")
    if flags:
        lines.append(f"WARNING: check looks circular/trivial ({flags[0]}) — not counted toward coverage")
    if weaken_warning:
        lines.append(f"WARNING: {weaken_warning}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Mechanism 4 — the submission gate
# ---------------------------------------------------------------------------


def coverage(criteria: dict[str, Criterion], state: SelfTestState) -> dict[str, bool]:
    """Which declared criteria have >=1 non-circular, isolated-passing check."""
    return {
        cid: any(r.isolated_pass and not r.circular for r in (state.checks.get(cid) or []))
        for cid in criteria
    }


def evaluate_gate(criteria: dict[str, Criterion], state: SelfTestState, config: SelfTestConfig) -> GateResult:
    covered_ids: list[str] = []
    uncovered_ids: list[str] = []
    if criteria:
        cov = coverage(criteria, state)
        covered_ids = [cid for cid, ok in cov.items() if ok]
        uncovered_ids = [cid for cid, ok in cov.items() if not ok]
        if len(covered_ids) >= config.min_criteria:
            return GateResult(True, [])

    # The rejection bound is a universal safety valve — it applies whether the
    # model never declared criteria at all or just hasn't gotten enough checks
    # passing yet, so a model stuck on either failure mode still converts to a
    # flagged submit rather than burning the whole step budget.
    if state.gate_rejections >= config.max_gate_rejections:
        return GateResult(
            True, [f"proceeding after {state.gate_rejections} rejected submit attempts"], forced=True
        )

    if not criteria:
        return GateResult(False, [f"no acceptance criteria declared — write them to {config.criteria_path}"])

    reasons = [f"only {len(covered_ids)}/{config.min_criteria} required criteria have a passing isolated check"]
    if uncovered_ids:
        reasons.append(f"unmet: {', '.join(uncovered_ids)}")
    return GateResult(False, reasons)


def gate_nudge(gate: GateResult) -> str:
    return (
        "Submit rejected — " + "; ".join(gate.reasons) + ". "
        "Keep working, then run `agent-check <id> -- <command>` again once fixed."
    )
