"""Context-management compaction for Vanillux2Agent's model-facing history.

``agent.py`` keeps an append-only, never-mutated raw event log (the same
``messages`` list it has always kept, now holding full/untruncated tool
output — see ``_RAW_OUTPUT_SAFETY_CAP_CHARS`` in agent.py for the one
disk-safety exception). Before every LLM call, :func:`build_model_messages`
rebuilds a *compacted* model-facing view from that raw log from scratch. It
never mutates the raw log; the rebuild is a pure function of the raw log plus
config, except for the one side effect of writing spill files to disk (see
"Determinism" below).

Two independent passes:

* **Stubbing** (Feature 1): a file-write bash command (heredoc/``tee``/
  redirect) or Feature-3 edit-tool call has its *request* payload (the bash
  command string, or the edit tool's bulky arguments) replaced with a short
  stub once it is superseded by a later write to the same path, or once it
  falls outside the trailing ``write_recency_keep``-turn window. Concretely: a
  write is kept in full only if it is *both* the latest write to its path
  *and* within the last ``write_recency_keep`` turns; every other write to a
  path that was ever written is eventually stubbed. This is a deliberate
  reading of the spec's two stubbing conditions ("superseded" / "older than
  K turns") as ANDed into one "keep-full" criterion, since that is what
  actually bounds context growth on a long run — see the docstring on
  Vanillux2Agent for the caveat.
* **Truncation** (Feature 2): any tool-role message whose content exceeds
  ``max_tool_output_tokens`` gets head/tail-line truncated, with the elided
  middle spilled verbatim to disk.

Determinism / idempotency: spill file paths are a pure function of the raw
message's index (not a random uuid), so rebuilding from the same raw log
twice always produces byte-identical output and always resolves to the same
spill path. ``spilled_indices`` is a cache the agent instance owns across
steps purely to avoid re-writing an unchanged spill file every step; it does
not affect what the rebuild produces.
"""

from __future__ import annotations

import json
import logging
import re
import shlex
from dataclasses import dataclass, replace
from typing import Any

import litellm

from Vanillux2Agent.container_ops import ContainerOps

EDIT_TOOL_NAMES = {"str_replace", "insert", "create", "apply_edits"}


@dataclass
class CompactionConfig:
    stub_file_writes: bool = True
    write_recency_keep: int = 2
    max_tool_output_tokens: int = 2000
    head_lines: int = 40
    tail_lines: int = 40
    spill_dir: str = "/tmp/harness_spill"
    # Overflow-recovery levers (normally None/off — see escalate_config and
    # agent.py's escalation ladder): when stale_output_keep_turns is set,
    # tool outputs older than that many assistant turns from the end are
    # ALWAYS truncated, down to the tiny stale head/tail budgets, regardless
    # of max_tool_output_tokens. The recent window keeps its normal budgets
    # so the model can still see what it just did.
    stale_output_keep_turns: int | None = None
    stale_output_head_lines: int = 3
    stale_output_tail_lines: int = 2


@dataclass
class CompactionStats:
    stubbed_writes: int = 0
    truncated_outputs: int = 0


# The escalation ladder is deliberately short: level 1 should already fit a
# 64-step run comfortably, level 2 is scorched-earth. Past that the loop
# gives up the same way it always did (see agent.py).
MAX_COMPACTION_LEVEL = 2


def escalate_config(config: CompactionConfig, level: int) -> CompactionConfig:
    """Derive the harsher compaction used by the overflow-recovery ladder.

    Pure and deterministic (a rebuilt history at a given level is stable
    across retries and steps). Level 0 is the configured behavior; level 1
    shrinks per-message output budgets, stubs writes sooner, and
    hard-truncates tool outputs older than a trailing window; level 2 keeps
    only a skeleton of everything but the last couple of turns.
    """
    if level <= 0:
        return config
    if level == 1:
        return replace(
            config,
            max_tool_output_tokens=max(250, config.max_tool_output_tokens // 4),
            head_lines=min(config.head_lines, 10),
            tail_lines=min(config.tail_lines, 10),
            write_recency_keep=min(config.write_recency_keep, 1),
            stale_output_keep_turns=8,
        )
    return replace(
        config,
        max_tool_output_tokens=120,
        head_lines=4,
        tail_lines=4,
        write_recency_keep=0,
        stale_output_keep_turns=2,
        stale_output_head_lines=2,
        stale_output_tail_lines=1,
    )


def count_tokens(text: str, model: str | None) -> int:
    """Token count via the model's own tokenizer where litellm can resolve one."""
    if not text:
        return 0
    if model:
        try:
            return litellm.token_counter(model=model, text=text)
        except Exception:
            pass
    return max(1, len(text) // 4)


def estimate_messages_tokens(messages: list[dict], model: str | None) -> int:
    """Best-effort token estimate of a full request payload — content AND
    tool-call arguments (the latter dominate on write-heavy runs: measured
    on real trajectories, heredoc arguments were most of the context), plus
    a small per-message chat-template overhead. This is an estimate, not a
    guarantee — callers must keep a reactive overflow path as the backstop.
    """
    try:
        if model:
            return litellm.token_counter(model=model, messages=messages)
    except Exception:
        pass
    total = 0
    for m in messages:
        total += 4
        total += len(str(m.get("content") or "")) // 4
        for tc in m.get("tool_calls") or []:
            total += len(str((tc.get("function") or {}).get("arguments") or "")) // 4
            total += 8
    return total


def _safe_json_loads(raw: Any) -> dict | None:
    if raw is None:
        return None
    if isinstance(raw, dict):
        return raw
    try:
        return json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        return None


# ---------------------------------------------------------------------------
# Feature 1 — file-write detection + stubbing
# ---------------------------------------------------------------------------

# ``cat``/``tee`` heredocs, in either token order mini-swe-agent's own prompt
# examples use (`cat > f << 'EOF' ...` and `cat <<'EOF' > f ...`).
_HEREDOC_REDIRECT_FIRST = re.compile(
    r"\b(?:cat|tee)\b[^\n<>|;&]*?>{1,2}\s*(?P<path>[^\s<>|;&]+)[^\n]*?"
    r"<<-?\s*(?P<quote>['\"]?)(?P<delim>\w+)(?P=quote)\s*\n"
    r"(?P<body>.*?)\n[ \t]*(?P=delim)\b",
    re.DOTALL,
)
_HEREDOC_MARKER_FIRST = re.compile(
    r"\b(?:cat|tee)\b[^\n<>|;&]*?<<-?\s*(?P<quote>['\"]?)(?P<delim>\w+)(?P=quote)[^\n]*?"
    r">{1,2}\s*(?P<path>[^\s<>|;&]+)\s*\n"
    r"(?P<body>.*?)\n[ \t]*(?P=delim)\b",
    re.DOTALL,
)
# ``tee PATH`` (no heredoc — content typically arrives via a pipe).
_TEE_RE = re.compile(r"\btee\b(?:\s+-\w+)*\s+(?P<path>(?!/dev/)[^\s<>|;&]+)")
# Plain ``> PATH`` / ``>> PATH``, excluding fd redirects (``2>&1``, ``>&2``),
# process substitution (``>(...)``), and /dev sinks.
_REDIRECT_RE = re.compile(r"(?<![\d&])>{1,2}(?!\(|&)\s*(?P<path>(?!/dev/)[^\s<>|;&()]+)")


def detect_bash_write_targets(command: str) -> list[tuple[str, int | None]]:
    """Best-effort detection of file-mutating constructs in a bash command.

    Returns ``[(path, approx_line_count_or_None), ...]``. Known limitations
    (acceptable for a heuristic harness-level detector, not a shell parser):
    doesn't understand ``sed -i``, Python ``open()``, or ``[[ a > b ]]``
    string comparisons (the latter can false-positive as a redirect).
    """
    targets: list[tuple[str, int | None]] = []
    seen_paths: set[str] = set()
    body_spans: list[tuple[int, int]] = []

    for rx in (_HEREDOC_REDIRECT_FIRST, _HEREDOC_MARKER_FIRST):
        for m in rx.finditer(command):
            path = m.group("path")
            body = m.group("body")
            lines = body.count("\n") + 1 if body else 0
            targets.append((path, lines))
            seen_paths.add(path)
            body_spans.append(m.span("body"))

    # Scrub heredoc bodies before scanning for tee/redirect: a body line like
    # `print(2 > 1)` or `foo > bar.txt` is file *content*, not a shell
    # redirect, and treating it as a write target would create a phantom
    # write event (harmless to correctness but it delays stubbing of the real
    # write by keeping bogus "latest write to that path" entries alive).
    scrubbed = command
    for start, end in sorted(body_spans, reverse=True):
        scrubbed = scrubbed[:start] + (" " * (end - start)) + scrubbed[end:]

    for m in _TEE_RE.finditer(scrubbed):
        path = m.group("path")
        if path not in seen_paths:
            targets.append((path, None))
            seen_paths.add(path)

    for m in _REDIRECT_RE.finditer(scrubbed):
        path = m.group("path")
        if path not in seen_paths:
            targets.append((path, None))
            seen_paths.add(path)

    return targets


def _edit_tool_write_targets(name: str, args: dict) -> list[tuple[str, int | None]]:
    path = args.get("path")
    if not path:
        return []
    if name == "create":
        content = args.get("content") or ""
        return [(path, content.count("\n") + 1 if content else 0)]
    if name == "insert":
        text = args.get("text") or ""
        return [(path, text.count("\n") + 1 if text else 0)]
    if name == "str_replace":
        new_string = args.get("new_string") or ""
        return [(path, new_string.count("\n") + 1 if new_string else 0)]
    if name == "apply_edits":
        edits = args.get("edits") or []
        if not edits:
            return []
        total = sum((e.get("new_string") or "").count("\n") + 1 for e in edits)
        return [(path, total)]
    return []


@dataclass(frozen=True)
class _WriteEvent:
    turn_index: int  # index into raw_messages of the assistant message
    call_index: int  # index into that message's tool_calls
    path: str
    lines: int | None


def _iter_write_events(raw_messages: list[dict]) -> list[_WriteEvent]:
    events: list[_WriteEvent] = []
    for turn_index, msg in enumerate(raw_messages):
        if msg.get("role") != "assistant":
            continue
        for call_index, tc in enumerate(msg.get("tool_calls") or []):
            func = tc.get("function") or {}
            name = func.get("name")
            args = _safe_json_loads(func.get("arguments"))
            if args is None:
                continue
            if name == "bash":
                targets = detect_bash_write_targets(args.get("command") or "")
            elif name in EDIT_TOOL_NAMES:
                targets = _edit_tool_write_targets(name, args)
            else:
                targets = []
            for path, lines in targets:
                events.append(_WriteEvent(turn_index, call_index, path, lines))
    return events


def _turn_numbers(raw_messages: list[dict]) -> list[int]:
    """One "turn" per assistant message; tool/system/user rows share it."""
    turns: list[int] = []
    n = 0
    for msg in raw_messages:
        if msg.get("role") == "assistant":
            n += 1
        turns.append(n)
    return turns


def _format_write_stub(writes: list[tuple[str, int | None]]) -> str:
    parts = [
        f"{lines} lines to {path}" if lines is not None else f"to {path}"
        for path, lines in writes
    ]
    return f"[wrote {', '.join(parts)} — current content on disk; use read to inspect]"


def _stub_tool_call(tc: dict, stub_text: str) -> dict:
    func = dict(tc.get("function") or {})
    if func.get("name") == "bash":
        new_args = json.dumps({"command": stub_text})
    else:
        args = _safe_json_loads(func.get("arguments")) or {}
        new_args = json.dumps({"path": args.get("path"), "note": stub_text})
    func["arguments"] = new_args
    new_tc = dict(tc)
    new_tc["function"] = func
    return new_tc


def _maybe_stub_assistant_message(
    msg: dict,
    idx: int,
    turn_no: int,
    current_turn: int,
    config: CompactionConfig,
    call_write_info: dict[tuple[int, int], list[tuple[str, int | None]]],
    last_turn_for_path: dict[str, int],
    stats: CompactionStats,
    logger: logging.Logger | None,
) -> dict:
    tool_calls = msg.get("tool_calls")
    if not tool_calls:
        return msg

    new_calls = list(tool_calls)
    changed = False
    for call_index, tc in enumerate(tool_calls):
        writes = call_write_info.get((idx, call_index))
        if not writes:
            continue

        keep_full = any(
            last_turn_for_path.get(path) == turn_no
            and (current_turn - turn_no) <= config.write_recency_keep
            for path, _lines in writes
        )
        if keep_full:
            continue

        stub_text = _format_write_stub(writes)
        new_calls[call_index] = _stub_tool_call(tc, stub_text)
        changed = True
        stats.stubbed_writes += 1
        if logger:
            logger.info("context_management: stubbed write turn=%s %s", turn_no, stub_text)

    if not changed:
        return msg
    new_msg = dict(msg)
    new_msg["tool_calls"] = new_calls
    return new_msg


# ---------------------------------------------------------------------------
# Feature 2 — tool-output truncation with spill-to-disk
# ---------------------------------------------------------------------------

_EXIT_CODE_RE = re.compile(r"\n\n\(exit_code=(-?\d+)\)\s*$")


def _spill_path(config: CompactionConfig, raw_index: int) -> str:
    # Deterministic (not a random uuid) so rebuilding the same raw log twice
    # is idempotent — see module docstring.
    return f"{config.spill_dir.rstrip('/')}/turn_{raw_index:05d}.txt"


async def _write_spill(ops: ContainerOps, path: str, content: str) -> bool:
    # Content moves via upload_bytes (docker cp), never through exec_fn's
    # command argv — a truncated tool output can be large or genuinely
    # binary (e.g. a command that dumped raw bytes to stdout), and embedding
    # either into a heredoc command string risks ARG_MAX or "embedded null
    # byte" (see edit_tools.py's module docstring for the full mechanism).
    # Only the tiny, fixed-size `mkdir -p` still goes through exec_fn.
    directory = path.rsplit("/", 1)[0]
    mkdir_result = await ops.exec_fn(f"mkdir -p {shlex.quote(directory)}")
    if mkdir_result.return_code != 0:
        return False
    try:
        await ops.upload_bytes(content.encode("utf-8"), path)
    except Exception:
        return False
    return True


async def _maybe_truncate_tool_message(
    msg: dict,
    idx: int,
    config: CompactionConfig,
    model: str | None,
    ops: ContainerOps,
    spilled_indices: set[int],
    logger: logging.Logger | None,
    force: bool = False,
    head_lines: int | None = None,
    tail_lines: int | None = None,
) -> tuple[dict, bool]:
    """Head/tail-truncate one tool message if it's over budget.

    ``force`` (the stale-output path — see CompactionConfig) truncates
    regardless of the token cap; ``head_lines``/``tail_lines`` override the
    configured keeps for that path.
    """
    content = msg.get("content")
    if not isinstance(content, str) or not content:
        return msg, False

    head_n = config.head_lines if head_lines is None else head_lines
    tail_n = config.tail_lines if tail_lines is None else tail_lines

    if not force and count_tokens(content, model) <= config.max_tool_output_tokens:
        return msg, False

    m = _EXIT_CODE_RE.search(content)
    body, suffix = (content[: m.start()], content[m.start() :]) if m else (content, "")
    lines = body.split("\n")

    if len(lines) <= head_n + tail_n:
        # Token-heavy but line-sparse (e.g. one huge minified line) — the
        # head/tail-by-lines strategy has nothing to elide, so fall back to a
        # char-budget head/tail split. Without this, a single-line dump could
        # defeat the whole context budget (this bucket exists in real runs).
        char_budget = max(400, (head_n + tail_n) * 160)
        if not force and len(body) <= config.max_tool_output_tokens * 4:
            return msg, False
        if len(body) <= char_budget:
            return msg, False
        spill_path = _spill_path(config, idx)
        if idx not in spilled_indices:
            if not await _write_spill(ops, spill_path, body):
                if logger:
                    logger.warning(
                        "context_management: spill write failed for %s; keeping full output", spill_path
                    )
                return msg, False
            spilled_indices.add(idx)
        half = char_budget // 2
        n_elided = len(body) - 2 * half
        new_body = (
            f"{body[:half]}\n\n[truncated {n_elided} chars — full output at {spill_path}; "
            f"use read to inspect]\n\n{body[-half:]}"
        )
        if logger:
            logger.info(
                "context_management: char-truncated line-sparse tool output, elided=%s spill=%s",
                n_elided,
                spill_path,
            )
        new_msg = dict(msg)
        new_msg["content"] = new_body + suffix
        return new_msg, True

    spill_path = _spill_path(config, idx)
    if idx not in spilled_indices:
        ok = await _write_spill(ops, spill_path, body)
        if not ok:
            if logger:
                logger.warning("context_management: spill write failed for %s; keeping full output", spill_path)
            return msg, False
        spilled_indices.add(idx)

    elided = len(lines) - head_n - tail_n
    head = "\n".join(lines[:head_n])
    tail = "\n".join(lines[-tail_n:])
    new_body = (
        f"{head}\n\n[truncated {elided} lines — full output at {spill_path}; use read to inspect]\n\n{tail}"
    )
    if logger:
        logger.info("context_management: truncated tool output, elided=%s spill=%s", elided, spill_path)

    new_msg = dict(msg)
    new_msg["content"] = new_body + suffix
    return new_msg, True


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def _ensure_json_safe(msg: dict, logger: logging.Logger | None) -> dict:
    """Guarantee *msg* round-trips through json.loads/json.dumps before it's
    ever sent back to the API.

    Compaction itself only ever parses tool-call arguments and re-serializes
    via json.dumps (see _stub_tool_call) — it never regex/substring-edits a
    serialized string. But a message can arrive in the raw log already
    malformed (e.g. a truncated/malformed tool-call from an upstream
    parser), and nothing previously validated that before re-sending it on a
    later turn. This is a defense-in-depth guard, not a fix targeted at one
    specific origin: any tool_calls[].function.arguments that doesn't parse
    gets replaced with a minimal valid stub, preserving the same
    tool_call_id so pairing with the following tool-response message stays
    intact; any message that still doesn't serialize as a whole has its
    content dropped rather than being shipped broken.
    """
    tool_calls = msg.get("tool_calls")
    if tool_calls:
        fixed_calls = []
        changed = False
        for tc in tool_calls:
            func = tc.get("function") or {}
            args_raw = func.get("arguments")
            ok = True
            try:
                json.loads(args_raw) if isinstance(args_raw, str) else json.dumps(args_raw)
            except (TypeError, ValueError):
                ok = False
            if ok:
                fixed_calls.append(tc)
                continue
            changed = True
            if logger:
                logger.warning(
                    "context_management: repairing malformed tool_call arguments (id=%s)", tc.get("id")
                )
            new_func = dict(func)
            new_func["arguments"] = json.dumps({"error": "malformed arguments; dropped during history rebuild"})
            fixed_calls.append({**tc, "function": new_func})
        if changed:
            msg = {**msg, "tool_calls": fixed_calls}

    try:
        json.dumps(msg)
        return msg
    except (TypeError, ValueError):
        if logger:
            logger.warning("context_management: message not JSON-serializable; dropping content")
        safe = dict(msg)
        safe["content"] = "[content dropped: not JSON-serializable]"
        return safe


async def build_model_messages(
    raw_messages: list[dict],
    *,
    config: CompactionConfig,
    model: str | None,
    ops: ContainerOps,
    spilled_indices: set[int],
    stats: CompactionStats,
    logger: logging.Logger | None = None,
) -> list[dict]:
    """Rebuild the model-facing message list from the raw event log.

    Pure given (raw_messages, config, model) except for the spill side
    effect on disk — see module docstring on why that doesn't break
    determinism. Never mutates ``raw_messages`` or any of its elements.
    """
    turns = _turn_numbers(raw_messages)
    current_turn = turns[-1] if turns else 0

    call_write_info: dict[tuple[int, int], list[tuple[str, int | None]]] = {}
    last_turn_for_path: dict[str, int] = {}
    if config.stub_file_writes:
        for event in _iter_write_events(raw_messages):
            call_write_info.setdefault((event.turn_index, event.call_index), []).append(
                (event.path, event.lines)
            )
            turn_no = turns[event.turn_index]
            if turn_no > last_turn_for_path.get(event.path, -1):
                last_turn_for_path[event.path] = turn_no

    out: list[dict] = []
    for idx, msg in enumerate(raw_messages):
        role = msg.get("role")
        if role == "assistant" and config.stub_file_writes and msg.get("tool_calls"):
            msg = _maybe_stub_assistant_message(
                msg, idx, turns[idx], current_turn, config, call_write_info, last_turn_for_path, stats, logger
            )
        elif role == "tool":
            force = False
            head_override = tail_override = None
            if config.stale_output_keep_turns is not None:
                age = current_turn - turns[idx]
                if age > config.stale_output_keep_turns:
                    force = True
                    head_override = config.stale_output_head_lines
                    tail_override = config.stale_output_tail_lines
            msg, did_truncate = await _maybe_truncate_tool_message(
                msg,
                idx,
                config,
                model,
                ops,
                spilled_indices,
                logger,
                force=force,
                head_lines=head_override,
                tail_lines=tail_override,
            )
            if did_truncate:
                stats.truncated_outputs += 1
        out.append(_ensure_json_safe(msg, logger))
    return out
