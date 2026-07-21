"""Mechanisms 1-5: acceptance-criteria declaration + an isolated self-test gate.

Grounded in a 445-run failure analysis: of runs that finish and submit a
wrong solution, 67% explicitly declare success on a solution the held-out
grader rejects, and only 22% self-test at all (usually circularly — e.g.
asserting a program's output equals itself). This module gives the model a
way to declare acceptance criteria and verify them, plus the machinery
``agent.py`` uses to gate the submit sentinel on that verification.

Two model-facing interfaces, one shared core, chosen by config
(``agent.py``'s ``enable_edit_tools``):

* **Tools** (multi-tool contract, the default): ``declare_criteria`` and
  ``run_check(criterion_id, command, expected)`` are registered alongside
  bash/edit tools. This is the primary interface — a live 445-trial A/B of
  the earlier bash-convention-only variant found 12% of trials botched the
  convention's syntax and 15 trials probed for the (non-existent)
  ``agent-check`` binary and gave up; a real tool schema removes both
  failure modes at the source.
* **Bash convention** (bash-only contract, RL-data compatible): the model
  writes the criteria file itself and runs ``agent-check <id> -- <command>``,
  which the harness intercepts before it ever reaches the shell — either as
  the whole bash call, or as its own line within a larger call (setup on
  other lines runs normally), or several checks back to back. A line that
  starts with the ``agent-check`` token but matches neither form gets a
  corrective hint instead of hitting the real shell as "command not found",
  and existence probes (``which agent-check``) are answered directly.

Architecture
------------
* **Criteria** (Mechanism 1) live in a harness-owned file,
  ``<state_dir>/criteria.json`` — ``{criteria: [{id, description,
  how_to_check}], deliverables: [path, ...]}``. The ``declare_criteria``
  tool validates and merges into that file; in bash-only mode the model
  writes it directly. Either way the harness re-reads the file fresh
  whenever it needs the current declaration and never inlines it into the
  model-facing history (disk is the source of truth; the tool's compact
  confirmation is all the model sees).
* **Checks** (Mechanism 2) run a model-supplied command; the check passes
  iff the command exits 0, or — when the tool form supplies a static
  ``expected`` string — iff the command's trimmed output equals it. Only
  PASS/FAIL plus the first failing line is ever returned, never full
  output. Every check is recorded to ``<state_dir>/checks.json``.
* **Isolation** (Mechanism 3) is what makes the gate hard to game: every
  check also runs against a FRESH directory containing only the declared
  deliverable files, copied to their real absolute paths, invoked as a
  fresh process with no persistent env/cwd chaining (see ``agent.py``'s
  ``_wrap_command``, which ordinary bash calls go through but isolated
  checks deliberately bypass). Only the isolated result counts toward
  coverage; a check that passes in-session but fails in isolation is
  surfaced as a discrepancy (the neutered-deliverable / leftover-scratch-
  file failure mode from the analysis above).

  Known scope limitation: this harness has no primitive for spinning up a
  second container/namespace (the Docker environment only exposes ``docker
  compose exec``, no container ID), so isolation is "fresh temp dir + fresh
  process", not a fresh container. A check command that references a
  deliverable by *absolute* path escapes the sandbox (it hits the real
  filesystem, not the isolated copy) — only cwd-relative references are
  actually isolated. Documented rather than silently assumed away.
* **The gate** (Mechanism 4) is enforced by ``agent.py`` around the submit
  sentinel: allowed only once criteria are declared, EVERY declared
  criterion has >=1 non-circular, non-trivial check that passed in
  isolation, and at least ``min_criteria`` criteria are covered that way.
  Bounded by ``max_gate_rejections`` so a model stuck failing its own
  checks converts to a flagged ``submitted_with_failing_checks`` submit
  rather than burning the entire step budget.
* **Anti-circularity + honesty heuristics** (Mechanism 5) are best-effort,
  non-blocking: a circular or existence-only check is flagged and excluded
  from coverage; changing a check right after it failed in isolation, or
  removing/rewording a criterion whose latest check failed, is flagged as a
  possible weakening (warn, don't block — a self-test can legitimately be
  wrong). Two exit-code-blind patterns found live in a 445-trial A/B (a
  ``python -c "print(a == b)"`` whose comparison never reaches an
  ``assert``/``sys.exit``, and ``cmd && echo PASS || echo FAIL``, which
  always exits 0 because ``echo`` never fails) let a wrong deliverable
  through the gate several times before being caught here — both are
  flagged, as is a check that mostly delegates to an external script whose
  own content looks circular.
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
                "Declare the testable acceptance criteria for this task, and which "
                "file(s) are your deliverable. Call this once, early — every declared "
                "criterion needs a passing run_check before you can submit. If the task "
                "statement includes example input/output, turn those into criteria "
                "first. Calling it again merges/updates by criterion id."
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
                "`expected` (omit `expected` to require exit code 0 instead — the "
                "standard bash idiom: test/diff/grep -q/assert/pytest). The check also "
                "runs in an ISOLATED copy of your declared deliverables, not your live "
                "scratch state — a check that only passes because of leftover files or a "
                "weakened deliverable will fail there. Returns a compact pass/fail "
                "result, never full output. Use an independent oracle — a known answer "
                "(e.g. from the task's example I/O), an invariant, or a second method; a "
                "check that re-runs the program and compares it to itself, or one that "
                "can't exit non-zero, does not count toward coverage."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "criterion_id": {"type": "string"},
                    "command": {"type": "string"},
                    "expected": {
                        "type": "string",
                        "description": (
                            "Exact expected stdout+stderr (trimmed), as a FIXED literal "
                            "value. Omit to just require exit code 0."
                        ),
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
    how_to_check: str = ""


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
    trivial: bool
    missing_deliverables: list[str]
    step: int
    # Observability only — never affects pass/fail. Lets a later A/B analysis
    # see at a glance how substantive the checks in a run actually were:
    # "circular" (flagged by circularity_flags, incl. an external script it
    # delegates to), "trivial_existence" (bare `test -f`/`ls`, nothing
    # behavioral), "external_script" (delegates to a file we inspected but
    # didn't flag), or "behavioral" (everything else).
    check_category: str = "behavioral"


@dataclass
class SelfTestConfig:
    min_criteria: int = 2
    max_gate_rejections: int = 3
    check_timeout_sec: int = 60
    state_dir: str = "/tmp/.vanillux2/self_test"
    isolation_mode: str = "tempdir"  # only mode currently implemented; see module docstring

    @property
    def criteria_path(self) -> str:
        return f"{self.state_dir}/criteria.json"

    @property
    def checks_path(self) -> str:
        return f"{self.state_dir}/checks.json"

    @property
    def isolation_root(self) -> str:
        return f"{self.state_dir}/isolated"


@dataclass
class SelfTestState:
    checks: dict[str, list[CheckRecord]] = field(default_factory=dict)
    gate_rejections: int = 0
    submitted_with_failing_checks: bool = False
    # Last-seen view of the criteria FILE (id -> description), refreshed on
    # every load — the baseline the criteria-edit audit diffs against.
    criteria_snapshot: dict[str, str] = field(default_factory=dict)
    # Append-only audit trail of criteria edits (Mechanism 5): one small dict
    # per added/changed/removed criterion, persisted with the checks.
    audit: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class GateResult:
    allowed: bool
    reasons: list[str]
    forced: bool = False  # True only when allowed because rejections were exhausted


IsolatedExecFn = Callable[[str, str], Awaitable[Any]]


# ---------------------------------------------------------------------------
# agent-check convention parsing (bash-only contract)
#
# Two supported forms:
#  1. Whole-command: the ENTIRE bash call is one agent-check invocation. Its
#     inner command may itself be multi-line (e.g. a heredoc/python -c block).
#  2. Line-embedded: `agent-check <id> -- <command>` appears as its OWN LINE
#     within a larger multi-line bash call that also does setup on other
#     lines, or checks several criteria back to back (both are very common
#     in practice — models naturally bundle `mkdir ...`/comments alongside
#     checks, and often verify multiple criteria in one action). Each inner
#     command is limited to its own line; everything else in the call ("the
#     remainder") is run normally by the caller before the checks execute.
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
# line's command as its own (a real bug caught by extract_agent_check_lines's
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


def extract_agent_check_lines(command: str) -> tuple[list[tuple[str, str]], str] | None:
    """Find ALL ``agent-check <id> -- <command>`` invocations embedded as
    their own lines within a larger multi-line ``command`` — models commonly
    check several criteria back to back in one bash call, not just one.
    Returns ``(checks, remainder)`` where ``checks`` is the ``(criterion_id,
    inner_command)`` pairs in the order they appeared, and ``remainder`` is
    ``command`` with all matched lines removed (run normally by the caller,
    e.g. setup on other lines) — or ``None`` if no such line is present.
    """
    checks: list[tuple[str, str]] = []
    spans: list[tuple[int, int]] = []
    for m in AGENT_CHECK_LINE_RE.finditer(command):
        cmd = m.group("cmd").strip()
        if not cmd:
            continue
        checks.append((m.group("id"), cmd))
        spans.append(m.span())
    if not checks:
        return None

    remainder = command
    for start, end in sorted(spans, reverse=True):
        if end < len(remainder) and remainder[end] == "\n":
            end += 1  # swallow the line's own trailing newline, not just its content
        remainder = remainder[:start] + remainder[end:]
    return checks, remainder.strip()


def looks_like_malformed_agent_check(command: str) -> bool:
    """True if ``command`` contains a line that starts with the literal
    ``agent-check`` token but matches neither supported form above — a
    near-certain botched attempt (comments/string literals don't start a
    shell line with this exact token), not a false positive worth chasing.
    """
    if parse_agent_check(command) or extract_agent_check_lines(command):
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

# In a 445-trial A/B, 15 trials probed for the binary this way — almost
# always followed by the model concluding agent-check "isn't available" and
# giving up on the mechanism (self-simulating fake PASS/FAIL text instead).
# `which`/`type`/`command -v` genuinely can't find it (it's intercepted, not
# on PATH), so answer the probe directly instead of letting that cascade start.
EXISTENCE_PROBE_RE = re.compile(r"\b(?:which|type|command\s+-v)\s+agent-check\b")

AGENT_CHECK_EXISTENCE_HINT = (
    "agent-check is a harness convention intercepted from the bash command "
    "stream — it is NOT a real binary on PATH, so `which`/`type`/`command -v` "
    "will always report it missing. That's expected; just run it directly: "
    "`agent-check <id> -- <command>`."
)


def looks_like_existence_probe(command: str) -> bool:
    """True if ``command`` checks whether ``agent-check`` exists as a real
    binary (``which``/``type``/``command -v``) rather than using it."""
    return bool(EXISTENCE_PROBE_RE.search(command))


# ---------------------------------------------------------------------------
# Mechanism 1 — the harness-owned criteria file
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


async def load_criteria(ops: ContainerOps, config: SelfTestConfig) -> tuple[dict[str, Criterion], list[str]]:
    result = await ops.exec_fn(f"cat {shlex.quote(config.criteria_path)} 2>/dev/null")
    return parse_criteria_doc(getattr(result, "stdout", None) or "")


async def _write_criteria(
    ops: ContainerOps, config: SelfTestConfig, criteria: dict[str, Criterion], deliverables: list[str]
) -> None:
    doc = {"criteria": [asdict(c) for c in criteria.values()], "deliverables": deliverables}
    await ops.exec_fn(f"mkdir -p {shlex.quote(config.state_dir)}")
    await ops.upload_bytes(json.dumps(doc, indent=2).encode("utf-8"), config.criteria_path)


async def declare_criteria(
    args: dict, *, ops: ContainerOps, config: SelfTestConfig, state: SelfTestState, step: int
) -> str:
    """The ``declare_criteria`` tool: validate + merge into the harness-owned
    criteria file (by id; re-declaring updates). Returns a compact ack — the
    full declaration lives on disk, never in the model-facing history."""
    criteria, deliverables = await load_criteria(ops, config)

    added: list[str] = []
    for c in args.get("criteria") or []:
        if not isinstance(c, dict):
            continue
        cid = str(c.get("id") or "").strip()
        description = str(c.get("description") or "").strip()
        if not cid or not description:
            continue
        criteria[cid] = Criterion(
            id=cid, description=description, how_to_check=str(c.get("how_to_check") or "").strip()
        )
        added.append(cid)

    for d in args.get("deliverables") or []:
        d = str(d).strip()
        if d and d not in deliverables:
            deliverables.append(d)

    if not added:
        return (
            "declare_criteria: no valid criteria given — each needs a non-empty id, "
            "description, and how_to_check"
        )

    edit_warnings = criteria_edit_warnings(state, criteria, step)
    await _write_criteria(ops, config, criteria, deliverables)

    lines = [
        f"declared {len(added)} criteria: {', '.join(added)} "
        f"({len(criteria)} total, {len(deliverables)} deliverable file(s) tracked)"
    ]
    lines.extend(f"WARNING: {w}" for w in edit_warnings)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Mechanism 5 — anti-circularity + honesty heuristics
# ---------------------------------------------------------------------------

_TRIVIAL_COMMAND_RE = re.compile(r"^\s*(true|:|exit\s+0)\s*$")
# Constructs whose OWN exit status genuinely reflects success/failure. `==`/
# `!=` deliberately excluded: a bare comparison (e.g. inside `print(...)`, or
# as an unused expression like `print(x) == y`) doesn't affect the exit code
# by itself, so treating it as sufficient evidence of "a real assertion" (as
# an earlier version of this heuristic did) let checks that can never fail
# regardless of the underlying condition pass a 445-trial A/B undetected.
# `sys.exit`/`raise`/`os._exit`/bare `exit(` count too — a script that acts
# on a comparison via one of these (rather than the literal word `assert`)
# is just as real an assertion; omitting them would flag legitimate
# `if x != y: sys.exit(1)`-style scripts as circular.
_REAL_ASSERTION_HINTS = (
    "assert", "diff", "grep", "pytest", "unittest", "[[", "[ ", "cmp ", "test ",
    "sys.exit", "os._exit", "raise ", "exit(",
)


def _has_real_assertion(cmd: str) -> bool:
    return any(h in cmd for h in _REAL_ASSERTION_HINTS)


_SELF_DIFF_RE = re.compile(r"diff\s+<\((?P<a>.+?)\)\s+<\((?P<b>.+?)\)")
_SELF_EQ_RE = re.compile(r"\[\[?\s*[\"']?\$\((?P<a>.+?)\)[\"']?\s*(?:==|=)\s*[\"']?\$\((?P<b>.+?)\)[\"']?\s*\]\]?")
# Shell idioms whose tail always exits 0 regardless of what ran before it —
# each one seen (or a direct variant of one seen) in a real trace that let a
# wrong deliverable through the gate undetected.
_ALWAYS_ZERO_TAIL_PATTERNS = (
    (
        re.compile(r"\|\|\s*echo\b[^|&;]*$"),
        "ends in `|| echo ...` (or `cmd && echo PASS || echo FAIL`) — `echo` always "
        "succeeds, so the check's exit code is always 0 regardless of what came before it",
    ),
    (
        re.compile(r"(?:\|\||;)\s*true\s*$"),
        "ends in `|| true` / `; true` — `true` always succeeds, masking any real "
        "failure that happened before it",
    ),
    (
        re.compile(r"echo\s+[\"']?\$\?[\"']?\s*$"),
        "ends in `echo $?` — printing the exit code isn't the same as acting on it; "
        "`echo` itself always succeeds regardless of what `$?` held",
    ),
)
_EXCEPT_PASS_RE = re.compile(r"except\b[^:]*:\s*pass\b")
_SET_PLUS_E_RE = re.compile(r"\bset\s+\+e\b")
_RUNTIME_EXPECTED_RE = re.compile(r"\$\(|`")


def circularity_flags(command: str, expected: str | None = None) -> list[str]:
    """Best-effort detection of trivially/circularly passing checks (heuristic, not a proof).

    With a static ``expected`` string the harness itself compares the
    command's output to a fixed value, so the exit-code-blindness family of
    flags (no assertion, always-zero tails, swallowed exceptions) doesn't
    apply — the comparison IS the assertion. Self-comparison still does, and
    an ``expected`` containing `$(...)`/backticks signals the model *meant*
    to compute the oracle from the program under test (the harness compares
    literally, so such a check also honestly fails — but the intent is the
    thing worth correcting).
    """
    flags: list[str] = []
    cmd = command or ""
    exp = (expected or "").strip()

    if exp:
        if _RUNTIME_EXPECTED_RE.search(exp):
            flags.append(
                "`expected` contains `$(...)`/backticks — it must be a FIXED literal "
                "value (the harness compares text, not shell), derived independently "
                "of the program under test"
            )
    else:
        has_real_assertion = _has_real_assertion(cmd)
        if _TRIVIAL_COMMAND_RE.match(cmd):
            flags.append("command is a no-op (`true`/`:`/`exit 0`) with no real assertion")
        elif not has_real_assertion:
            if "==" in cmd or "!=" in cmd:
                flags.append(
                    "uses `==`/`!=` but nothing (assert/test/diff/grep/pytest/sys.exit/...) "
                    "enforces it — a bare comparison (e.g. inside `print(...)`, or as an unused "
                    "expression) doesn't make the process exit non-zero on a mismatch"
                )
            else:
                flags.append("no recognizable assertion in `command` (no test/diff/grep/assert/pytest/...)")

        stripped = cmd.rstrip()
        for rx, msg in _ALWAYS_ZERO_TAIL_PATTERNS:
            if rx.search(stripped):
                flags.append(msg)
                break

        if _EXCEPT_PASS_RE.search(cmd):
            flags.append(
                "catches exceptions with a bare `except: pass` — a real failure is silently "
                "swallowed instead of propagating as a non-zero exit"
            )

        if _SET_PLUS_E_RE.search(cmd):
            flags.append(
                "uses `set +e`, which disables exit-on-error for the rest of the script — "
                "make sure the final exit code still reflects the real outcome, not just "
                "whatever command happened to run last"
            )

    for rx in (_SELF_DIFF_RE, _SELF_EQ_RE):
        m = rx.search(cmd)
        if m and m.group("a").strip() and m.group("a").strip() == m.group("b").strip():
            flags.append("compares the program's output to itself — no independent oracle")
            break

    return flags


# A check that mostly delegates to an external script file (`python3
# /tmp/verify.py`, `./check.sh`) hides whatever circularity lives in that
# file from every heuristic above, which only ever sees the invocation
# command. Best-effort: extract the referenced path and, if readable, run
# the same heuristic against its contents too.
_SCRIPT_REF_RE = re.compile(
    r"(?:python3?|bash|sh|node|ruby|perl)\s+([^\s;&|]+\.(?:py|sh|js|rb|pl))\b"
    r"|(?:^|[;&|]\s*)(\.{0,2}/[^\s;&|]+\.(?:py|sh))\b"
)


def extract_script_reference(command: str) -> str | None:
    """Best-effort: the external script file a check command primarily
    delegates to, if any (``None`` for inline one-liners)."""
    m = _SCRIPT_REF_RE.search(command or "")
    if not m:
        return None
    return m.group(1) or m.group(2)


# A bare existence probe proves a file is on disk, not that it behaves — the
# held-out grader almost always checks content/behavior, so an existence-only
# check covering a criterion is false confidence (Mechanism 4c: only
# non-trivial criteria count toward the gate).
_TRIVIAL_EXISTENCE_RE = re.compile(r"^\s*(?:test\s+-[ef]\s+\S+|\[\s+-[ef]\s+\S+\s*\]|ls\s+\S+)\s*$")


def is_trivial_existence_check(command: str) -> bool:
    return bool(_TRIVIAL_EXISTENCE_RE.match((command or "").strip()))


TRIVIAL_EXISTENCE_WARNING = (
    "existence-only check (`test -f`/`ls`) — not counted toward coverage; "
    "verify the file's CONTENT or the program's BEHAVIOR instead"
)


def classify_check(command: str, flags: list[str], trivial: bool) -> str:
    """Coarse, observability-only bucket for a check — never affects
    pass/fail. See ``CheckRecord.check_category``."""
    cmd = command or ""
    if flags:
        return "circular"
    if trivial:
        return "trivial_existence"
    if not _has_real_assertion(cmd) and extract_script_reference(cmd):
        return "external_script"
    return "behavioral"


def weakening_warning(
    criterion_id: str, command: str, expected: str | None, state: SelfTestState
) -> str | None:
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


def criteria_edit_warnings(
    state: SelfTestState, criteria: dict[str, Criterion], step: int
) -> list[str]:
    """Diff the freshly-loaded criteria against the last-seen snapshot: audit
    every add/change/removal, and warn when a criterion whose latest isolated
    check FAILED is removed or reworded (the criteria-level analogue of
    ``weakening_warning`` — deleting a failing requirement is the cheapest
    way to loosen the gate). Updates the snapshot as a side effect."""
    warnings: list[str] = []

    def _latest_failed(cid: str) -> bool:
        history = state.checks.get(cid) or []
        return bool(history) and not history[-1].isolated_pass

    for cid, old_description in state.criteria_snapshot.items():
        if cid not in criteria:
            state.audit.append({"event": "criterion_removed", "id": cid, "step": step})
            if _latest_failed(cid):
                warnings.append(
                    f"criterion {cid!r} was REMOVED after its last check FAILED — "
                    "confirm the requirement really doesn't apply, don't just drop it"
                )
        elif criteria[cid].description != old_description:
            state.audit.append({"event": "criterion_changed", "id": cid, "step": step})
            if _latest_failed(cid):
                warnings.append(
                    f"criterion {cid!r} was reworded after its last check FAILED — "
                    "confirm the requirement still holds, don't just loosen it"
                )
    for cid in criteria:
        if cid not in state.criteria_snapshot:
            state.audit.append({"event": "criterion_added", "id": cid, "step": step})

    state.criteria_snapshot = {cid: c.description for cid, c in criteria.items()}
    return warnings


# ---------------------------------------------------------------------------
# Mechanism 3 — isolation
# ---------------------------------------------------------------------------

_UNSAFE_ID_RE = re.compile(r"[^A-Za-z0-9_.-]")


def _safe_dir_name(criterion_id: str) -> str:
    """Sanitize a model-supplied criterion_id before it's used as a path component."""
    return _UNSAFE_ID_RE.sub("_", criterion_id).strip(".") or "check"


_MISSING_MARKER = "__SELF_TEST_MISSING__:"


async def prepare_isolated_root(
    ops: ContainerOps, config: SelfTestConfig, deliverables: list[str], cwd: str, criterion_id: str
) -> tuple[str, list[str]]:
    """Build a fresh directory containing only the declared deliverables, mirrored at
    their real absolute paths. Returns ``(isolated_cwd, missing_deliverable_paths)``.

    Batched into a SINGLE ``exec_fn`` round trip (one per deliverable was
    costing 2 + 2*len(deliverables) container round trips — a real chunk of
    the latency/wall-clock overhead the gate adds per check) — each
    deliverable copy is wrapped in ``|| echo MISSING:...`` so one failure
    doesn't abort the rest, and missing ones are recovered by scanning stdout
    afterward. ``cp -a`` (not byte upload/download) so directory deliverables
    and permission bits mirror faithfully.
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
    result = await ops.exec_fn("\n".join(parts))
    output = getattr(result, "stdout", None) or ""
    missing = [
        line[len(_MISSING_MARKER):] for line in output.splitlines() if line.startswith(_MISSING_MARKER)
    ]
    return isolated_cwd, missing


# ---------------------------------------------------------------------------
# Mechanism 2 — check execution (shared by the run_check tool and the
# agent-check bash convention)
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


async def persist_state(ops: ContainerOps, config: SelfTestConfig, state: SelfTestState) -> None:
    doc = {
        "checks": {cid: [asdict(r) for r in records] for cid, records in state.checks.items()},
        "gate_rejections": state.gate_rejections,
        "submitted_with_failing_checks": state.submitted_with_failing_checks,
        "audit": state.audit,
    }
    await ops.exec_fn(f"mkdir -p {shlex.quote(config.state_dir)}")
    await ops.upload_bytes(json.dumps(doc, indent=2).encode("utf-8"), config.checks_path)


async def run_check(
    criterion_id: str,
    command: str,
    expected: str | None,
    *,
    state: SelfTestState,
    config: SelfTestConfig,
    ops: ContainerOps,
    isolated_exec: IsolatedExecFn,
    step: int,
) -> str:
    """Run one check (tool or agent-check form); never raises — always
    returns compact model-facing text."""
    criteria, deliverables = await load_criteria(ops, config)
    edit_warnings = criteria_edit_warnings(state, criteria, step)

    if not criterion_id:
        return "run_check: criterion_id is required"
    if criterion_id not in criteria:
        known = ", ".join(criteria) if criteria else "none declared yet"
        return (
            f"run_check: unknown criterion_id {criterion_id!r} (declared: {known}) — "
            "declare it first"
        )
    if not command:
        return f"run_check {criterion_id}: command is required"

    cwd_result = await ops.exec_fn("pwd")
    persistent_cwd = (getattr(cwd_result, "stdout", None) or "/").strip() or "/"

    # Only worth reading the referenced file when the OUTER command has no
    # assertion mechanism of its own (and no harness-side `expected`
    # comparison) — otherwise the script is just a data source for an
    # already-legitimate outer check (e.g. `test "$(python3 solution.py)" =
    # "42"`), and reading it risks flagging perfectly good checks over
    # unrelated content in a script invoked only for its stdout.
    script_path = None
    if not (expected or "").strip() and not _has_real_assertion(command):
        script_path = extract_script_reference(command)
    script_content = ""
    if script_path:
        abs_script_path = (
            script_path if script_path.startswith("/") else f"{persistent_cwd.rstrip('/')}/{script_path}"
        )
        script_result = await ops.exec_fn(f"cat {shlex.quote(abs_script_path)} 2>/dev/null")
        script_content = getattr(script_result, "stdout", None) or ""

    if script_content.strip():
        # The outer command has no assertion of its own (gated above), so
        # the SCRIPT's own circularity is what actually determines whether
        # a real assertion backs this check — it supersedes, rather than
        # adds to, the outer command's "no recognizable assertion" verdict
        # (which would otherwise fire on every `python3 script.py`-only
        # invocation regardless of what that script actually does).
        flags = [
            f"external script {script_path!r} looks circular: {f}" for f in circularity_flags(script_content)
        ]
    else:
        flags = circularity_flags(command, expected)
    trivial = is_trivial_existence_check(command)
    weaken_warning = weakening_warning(criterion_id, command, expected, state)

    session_result = await ops.exec_fn(command)
    session_output = _combine_output(session_result)
    session_pass = _judge(session_output, session_result, expected)

    isolated_cwd, missing = await prepare_isolated_root(
        ops, config, deliverables, persistent_cwd, criterion_id
    )
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
        circular=bool(flags),
        circular_reason="; ".join(flags) if flags else None,
        trivial=trivial,
        missing_deliverables=missing,
        step=step,
        check_category=classify_check(command, flags, trivial),
    )
    state.checks.setdefault(criterion_id, []).append(record)
    await persist_state(ops, config, state)

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
    if flags:
        lines.append(f"WARNING: check looks circular/trivial ({flags[0]}) — not counted toward coverage")
    elif trivial:
        lines.append(f"WARNING: {TRIVIAL_EXISTENCE_WARNING}")
    if weaken_warning:
        lines.append(f"WARNING: {weaken_warning}")
    lines.extend(f"WARNING: {w}" for w in edit_warnings)
    return "\n".join(lines)


async def dispatch(
    name: str,
    args: dict,
    *,
    state: SelfTestState,
    ops: ContainerOps,
    isolated_exec: IsolatedExecFn,
    config: SelfTestConfig,
    step: int,
) -> str:
    """Route a self-test tool call; never raises — always returns model-facing text."""
    try:
        if name == "declare_criteria":
            return await declare_criteria(args, ops=ops, config=config, state=state, step=step)
        if name == "run_check":
            expected_raw = args.get("expected")
            return await run_check(
                str(args.get("criterion_id") or "").strip(),
                str(args.get("command") or "").strip(),
                str(expected_raw) if expected_raw is not None else None,
                state=state,
                config=config,
                ops=ops,
                isolated_exec=isolated_exec,
                step=step,
            )
        return f"error: unknown self-test tool '{name}'"
    except Exception as exc:  # defensive: a bad call must not crash the run
        return f"error: {type(exc).__name__}: {exc}"


# ---------------------------------------------------------------------------
# Mechanism 4 — the submission gate
# ---------------------------------------------------------------------------


def _counts(record: CheckRecord) -> bool:
    return record.isolated_pass and not record.circular and not record.trivial


def coverage(criteria: dict[str, Criterion], state: SelfTestState) -> dict[str, bool]:
    """Which declared criteria have >=1 non-circular, non-trivial check that
    passed in isolation."""
    return {cid: any(_counts(r) for r in (state.checks.get(cid) or [])) for cid in criteria}


def evaluate_gate(
    criteria: dict[str, Criterion], state: SelfTestState, config: SelfTestConfig
) -> GateResult:
    covered_ids: list[str] = []
    uncovered_ids: list[str] = []
    if criteria:
        cov = coverage(criteria, state)
        covered_ids = [cid for cid, ok in cov.items() if ok]
        uncovered_ids = [cid for cid, ok in cov.items() if not ok]
        if not uncovered_ids and len(covered_ids) >= config.min_criteria:
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
        return GateResult(False, ["no acceptance criteria declared"])

    reasons: list[str] = []
    if uncovered_ids:
        reasons.append(
            f"checks failed or missing in isolation for: {', '.join(uncovered_ids)}"
        )
    if len(criteria) < config.min_criteria:
        reasons.append(
            f"only {len(criteria)} criteria declared; at least {config.min_criteria} "
            "non-trivial criteria must be declared and covered"
        )
    if not reasons:  # all declared covered, but fewer than min_criteria declared
        reasons.append(
            f"only {len(covered_ids)}/{config.min_criteria} required criteria are covered"
        )
    return GateResult(False, reasons)


def gate_nudge(gate: GateResult, config: SelfTestConfig, tools_enabled: bool) -> str:
    if tools_enabled:
        how = (
            "Declare criteria with the declare_criteria tool, then get each one a "
            "passing run_check, then submit again."
        )
    else:
        how = (
            f"Write your criteria to {config.criteria_path}, then get each one a "
            "passing `agent-check <id> -- <command>`, then submit again."
        )
    return "Submit rejected — " + "; ".join(gate.reasons) + ". " + how
