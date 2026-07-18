"""Feature-3 edit tools: str_replace / insert / create / apply_edits / read.

Registered alongside ``bash`` so the model has a small-write path that never
puts a whole file's text back into the conversation. These execute host-side
(matching Vanillux2Agent's architecture): file content is fetched and mutated
in Python, then written back atomically via the same ``exec_fn`` the agent
uses for bash (so cwd/env state stays consistent — see ``_wrap_command`` in
agent.py). No file content ever needs to be capable of round-tripping through
the model — that's the whole point.

Note: the model (tmax-9b) was SFT/RL-trained exclusively on the single-tool
bash harness, and the vendored system prompt says "calling a tool other than
bash... will cause your response to be rejected." We do not change that
prompt text (out of scope — no model/prompt changes). These tools are
therefore mostly future-facing / defense-in-depth: nothing here stops the
model from using them if it does, but there's no expectation it spontaneously
will without a training update that teaches it to.
"""

from __future__ import annotations

import difflib
import json
import re
import shlex
from typing import Any, Awaitable, Callable

from rl_data.generator.sample_solutions import SUBMIT_MARKER

ExecFn = Callable[[str], Awaitable[Any]]

EDIT_TOOL_NAMES = {"str_replace", "insert", "create", "apply_edits"}
ALL_TOOL_NAMES = EDIT_TOOL_NAMES | {"bash", "read"}

EDIT_TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "read",
            "description": (
                "Read a line-numbered range of a file on disk (also works on "
                "spill files referenced by truncated tool output). Output is "
                "capped; read large files in windows."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path to read."},
                    "start": {"type": "integer", "description": "1-indexed start line (default 1)."},
                    "end": {"type": "integer", "description": "1-indexed end line (default: start+199)."},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "str_replace",
            "description": (
                "Replace exactly one occurrence of old_string with new_string in "
                "path. Whitespace-tolerant match. Fails (no write) if old_string "
                "matches zero or more than one location."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "old_string": {"type": "string"},
                    "new_string": {"type": "string"},
                },
                "required": ["path", "old_string", "new_string"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "insert",
            "description": "Insert text after the given 1-indexed line (0 = start of file).",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "line": {"type": "integer"},
                    "text": {"type": "string"},
                },
                "required": ["path", "line", "text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create",
            "description": "Create (or overwrite) a file with the given content.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "apply_edits",
            "description": (
                "Apply multiple str_replace-style edits to one file atomically: "
                "if any edit fails to match uniquely, none are applied."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "edits": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "old_string": {"type": "string"},
                                "new_string": {"type": "string"},
                            },
                            "required": ["old_string", "new_string"],
                        },
                    },
                },
                "required": ["path", "edits"],
            },
        },
    },
]


def _safe_json_loads(raw: Any) -> dict | None:
    if raw is None:
        return None
    if isinstance(raw, dict):
        return raw
    try:
        return json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        return None


def extract_action(response_msg: dict) -> dict[str, Any]:
    """Parse a tool-calling response message into an action dict.

    ``type`` is one of ``"no_tool_call"`` (format error — no tool call, or a
    tool name we don't recognize), ``"done"`` (bash command containing the
    submit marker), or ``"tool"`` (bash or any Feature-3 tool).
    """
    tool_calls = response_msg.get("tool_calls")
    if not tool_calls:
        return {"type": "no_tool_call", "name": None, "args": None, "tool_call_id": None}

    tc = tool_calls[0]
    func = tc.get("function") or {}
    name = func.get("name", "")
    tool_call_id = tc.get("id")

    if name not in ALL_TOOL_NAMES:
        return {"type": "no_tool_call", "name": name, "args": None, "tool_call_id": tool_call_id}

    args = _safe_json_loads(func.get("arguments", "{}"))
    if args is None:
        return {"type": "no_tool_call", "name": name, "args": None, "tool_call_id": tool_call_id}

    if name == "bash":
        command = (args.get("command") or "").strip()
        args = {"command": command}
        if SUBMIT_MARKER in command:
            return {"type": "done", "name": "bash", "args": args, "tool_call_id": tool_call_id}

    return {"type": "tool", "name": name, "args": args, "tool_call_id": tool_call_id}


# ---------------------------------------------------------------------------
# Fuzzy whitespace-tolerant matching
# ---------------------------------------------------------------------------


def _build_fuzzy_regex(old_string: str) -> re.Pattern[str]:
    """Match old_string tolerant of any whitespace differences between tokens."""
    tokens = old_string.split()
    if not tokens:
        return re.compile(re.escape(old_string))
    return re.compile(r"\s+".join(re.escape(t) for t in tokens))


def _nearest_lines_hint(content: str, old_string: str, context: int = 3) -> str:
    content_lines = content.splitlines()
    if not content_lines:
        return "(file is empty)"
    query_lines = [line for line in old_string.splitlines() if line.strip()]
    query = query_lines[0].strip() if query_lines else old_string.strip()[:80]
    if not query:
        return "(old_string is blank)"

    best_idx, best_ratio = 0, -1.0
    for i, line in enumerate(content_lines):
        ratio = difflib.SequenceMatcher(None, query, line.strip()).ratio()
        if ratio > best_ratio:
            best_idx, best_ratio = i, ratio

    start = max(0, best_idx - context)
    end = min(len(content_lines), best_idx + context + 1)
    numbered = "\n".join(f"{i + 1:6d}\t{content_lines[i]}" for i in range(start, end))
    return f"nearest lines (around L{best_idx + 1}):\n{numbered}"


# ---------------------------------------------------------------------------
# Disk I/O helpers (all via exec_fn, so cwd/env stays consistent with bash)
# ---------------------------------------------------------------------------


async def _read_file(exec_fn: ExecFn, path: str) -> str:
    result = await exec_fn(f"cat -- {shlex.quote(path)}")
    if result.return_code != 0:
        raise FileNotFoundError(f"{path}: {(result.stderr or result.stdout or '').strip() or 'not found'}")
    return result.stdout or ""


async def _write_file_atomic(exec_fn: ExecFn, path: str, content: str) -> None:
    tmp = f"{path}.vanillux2_tmp"
    if not content:
        script = f": > {shlex.quote(tmp)} && mv -f -- {shlex.quote(tmp)} {shlex.quote(path)}"
    else:
        delim = "VANILLUX2_EDIT_EOF"
        body = content if content.endswith("\n") else content + "\n"
        while re.search(rf"(?m)^{re.escape(delim)}$", body):
            delim += "_X"
        script = (
            f"cat > {shlex.quote(tmp)} << '{delim}'\n"
            f"{body}{delim}\n"
            f"mv -f -- {shlex.quote(tmp)} {shlex.quote(path)}"
        )
    result = await exec_fn(script)
    if result.return_code != 0:
        await exec_fn(f"rm -f -- {shlex.quote(tmp)}")
        raise OSError(f"failed to write {path}: {(result.stderr or result.stdout or '').strip()}")


# ---------------------------------------------------------------------------
# Tool implementations — each returns a compact string, never raw file text
# ---------------------------------------------------------------------------


async def read_file_range(
    exec_fn: ExecFn, path: str, start: int | None, end: int | None, max_lines: int = 200
) -> str:
    wc = await exec_fn(f"wc -l -- {shlex.quote(path)}")
    if wc.return_code != 0:
        raise FileNotFoundError(f"{path}: {(wc.stderr or wc.stdout or '').strip() or 'not found'}")
    total_lines = int((wc.stdout or "0").split()[0] or 0)
    if total_lines == 0:
        return "(file is empty)"

    start = max(1, start or 1)
    if end is None:
        end = start + max_lines - 1
    end = min(end, start + max_lines - 1, total_lines)
    if start > total_lines or start > end:
        return f"(file has {total_lines} lines; requested range [{start}, {end}] is out of bounds)"

    result = await exec_fn(f"sed -n '{start},{end}p' -- {shlex.quote(path)} | nl -ba -v{start} -w6 -s $'\\t'")
    body = (result.stdout or "").rstrip("\n")
    remaining = total_lines - end
    suffix = "" if remaining <= 0 else f"\n... ({remaining} more lines; pass start={end + 1} to continue)"
    return f"{body}{suffix}"


async def str_replace(exec_fn: ExecFn, path: str, old_string: str, new_string: str) -> str:
    content = await _read_file(exec_fn, path)
    matches = list(_build_fuzzy_regex(old_string).finditer(content))

    if len(matches) == 0:
        hint = _nearest_lines_hint(content, old_string)
        return f"str_replace {path}: 0 matches for old_string.\n{hint}"
    if len(matches) > 1:
        return (
            f"str_replace {path}: {len(matches)} matches for old_string; "
            "provide a longer, unique anchor (more surrounding context)."
        )

    m = matches[0]
    matched_text = m.group(0)
    removed = matched_text.count("\n") + 1
    added = new_string.count("\n") + 1
    line_no = content[: m.start()].count("\n") + 1
    new_content = content[: m.start()] + new_string + content[m.end() :]
    await _write_file_atomic(exec_fn, path, new_content)
    return f"str_replace {path}: -{removed}/+{added} lines @L{line_no} (exit_code=0)"


async def insert(exec_fn: ExecFn, path: str, line: int, text: str) -> str:
    content = await _read_file(exec_fn, path)
    lines = content.splitlines(keepends=True)
    if line < 0 or line > len(lines):
        return f"insert {path}: line {line} out of range (file has {len(lines)} lines)"

    insert_text = text if text.endswith("\n") else text + "\n"
    new_content = "".join(lines[:line] + [insert_text] + lines[line:])
    await _write_file_atomic(exec_fn, path, new_content)
    added = insert_text.count("\n")
    return f"insert {path}: +{added} lines @L{line} (exit_code=0)"


async def create(exec_fn: ExecFn, path: str, content: str) -> str:
    await _write_file_atomic(exec_fn, path, content)
    lines = len(content.splitlines()) if content else 0
    return f"create {path}: {lines} lines (exit_code=0)"


async def apply_edits(exec_fn: ExecFn, path: str, edits: list[dict]) -> str:
    if not edits:
        return f"apply_edits {path}: no edits given"

    working = await _read_file(exec_fn, path)
    for i, edit in enumerate(edits):
        old_string = edit.get("old_string", "")
        new_string = edit.get("new_string", "")
        matches = list(_build_fuzzy_regex(old_string).finditer(working))

        if len(matches) == 0:
            hint = _nearest_lines_hint(working, old_string)
            return f"apply_edits {path}: edit {i} — 0 matches for old_string; aborted, disk unchanged.\n{hint}"
        if len(matches) > 1:
            return (
                f"apply_edits {path}: edit {i} — {len(matches)} matches for old_string; "
                "aborted, disk unchanged. Provide a longer, unique anchor."
            )

        m = matches[0]
        working = working[: m.start()] + new_string + working[m.end() :]

    await _write_file_atomic(exec_fn, path, working)
    return f"apply_edits {path}: {len(edits)} edits applied (exit_code=0)"


async def dispatch(name: str, args: dict, exec_fn: ExecFn) -> str:
    """Route a Feature-3 tool call; never raises — always returns model-facing text."""
    try:
        if name == "read":
            return await read_file_range(exec_fn, args["path"], args.get("start"), args.get("end"))
        if name == "str_replace":
            return await str_replace(exec_fn, args["path"], args["old_string"], args["new_string"])
        if name == "insert":
            return await insert(exec_fn, args["path"], int(args["line"]), args["text"])
        if name == "create":
            return await create(exec_fn, args["path"], args.get("content", ""))
        if name == "apply_edits":
            return await apply_edits(exec_fn, args["path"], args.get("edits", []))
        return f"error: unknown tool '{name}'"
    except KeyError as exc:
        return f"error: missing required argument {exc}"
    except FileNotFoundError as exc:
        return f"error: {exc}"
    except Exception as exc:  # defensive: a bad edit call must not crash the run
        return f"error: {type(exc).__name__}: {exc}"
