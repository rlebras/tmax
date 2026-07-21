import copy
import json
from pathlib import Path

import context_management as cm  # noqa: E402 (sys.path set up by conftest)


def _assistant_bash(call_id: str, command: str) -> dict:
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {"name": "bash", "arguments": json.dumps({"command": command})},
            }
        ],
    }


def _assistant_edit(call_id: str, name: str, args: dict) -> dict:
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {"id": call_id, "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}
        ],
    }


def _tool(call_id: str, content: str) -> dict:
    return {"role": "tool", "tool_call_id": call_id, "content": content}


# ---------------------------------------------------------------------------
# detect_bash_write_targets — heredoc / tee / redirect detection
# ---------------------------------------------------------------------------


def test_detect_heredoc_redirect_first():
    cmd = "cat > file.py << 'EOF'\nimport os\nprint(1)\nEOF"
    assert cm.detect_bash_write_targets(cmd) == [("file.py", 2)]


def test_detect_heredoc_marker_first():
    # mini-swe-agent's own prompt example order: `cat <<'EOF' > newfile.py ...`
    cmd = "cat <<'EOF' > newfile.py\nimport numpy as np\nhello = 1\nprint(hello)\nEOF"
    assert cm.detect_bash_write_targets(cmd) == [("newfile.py", 3)]


def test_detect_tee_via_pipe():
    cmd = "echo hi | tee output.log"
    assert cm.detect_bash_write_targets(cmd) == [("output.log", None)]


def test_detect_plain_redirect():
    assert cm.detect_bash_write_targets("echo hi > out.txt") == [("out.txt", None)]


def test_detect_append_redirect():
    assert cm.detect_bash_write_targets("echo more >> out.txt") == [("out.txt", None)]


def test_no_false_positive_fd_redirect():
    assert cm.detect_bash_write_targets("some_cmd 2>&1 | grep foo") == []


def test_no_false_positive_dev_null():
    assert cm.detect_bash_write_targets("noisy_cmd > /dev/null 2>&1") == []


def test_no_false_positive_plain_read_command():
    assert cm.detect_bash_write_targets("cat file.py | grep foo") == []


def test_detect_multiple_writes_in_one_command():
    cmd = "cat > a.py << 'EOF'\nx\nEOF\ncat > b.py << 'EOF'\ny\nz\nEOF"
    targets = cm.detect_bash_write_targets(cmd)
    assert ("a.py", 1) in targets
    assert ("b.py", 2) in targets


# ---------------------------------------------------------------------------
# stubbing: recency + supersession
# ---------------------------------------------------------------------------


async def test_recent_sole_write_stays_unstubbed(ops):
    raw = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "task"},
        _assistant_bash("c1", "cat > f.py << 'EOF'\nprint(1)\nEOF"),
        _tool("c1", "(no output)\n\n(exit_code=0)"),
    ]
    config = cm.CompactionConfig(write_recency_keep=2)
    stats = cm.CompactionStats()
    out = await cm.build_model_messages(
        raw, config=config, model=None, ops=ops, spilled_indices=set(), stats=stats
    )
    args = json.loads(out[2]["tool_calls"][0]["function"]["arguments"])
    assert args["command"] == "cat > f.py << 'EOF'\nprint(1)\nEOF"
    assert stats.stubbed_writes == 0


async def test_stale_sole_write_gets_stubbed_past_recency_window(ops):
    raw = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "task"},
        _assistant_bash("c1", "cat > f.py << 'EOF'\nprint(1)\nEOF"),
        _tool("c1", "(no output)\n\n(exit_code=0)"),
        _assistant_bash("c2", "ls"),
        _tool("c2", "f.py\n\n(exit_code=0)"),
        _assistant_bash("c3", "ls -la"),
        _tool("c3", "total 0\n\n(exit_code=0)"),
        _assistant_bash("c4", "pwd"),
        _tool("c4", "/\n\n(exit_code=0)"),
    ]
    config = cm.CompactionConfig(write_recency_keep=2)
    stats = cm.CompactionStats()
    out = await cm.build_model_messages(
        raw, config=config, model=None, ops=ops, spilled_indices=set(), stats=stats
    )
    args = json.loads(out[2]["tool_calls"][0]["function"]["arguments"])
    assert "wrote" in args["command"] and "f.py" in args["command"]
    assert "print(1)" not in args["command"]
    assert stats.stubbed_writes == 1
    # the exit code / tool result for the write itself is untouched
    assert out[3]["content"] == "(no output)\n\n(exit_code=0)"


async def test_superseded_write_is_stubbed_even_if_recent(ops):
    raw = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "task"},
        _assistant_bash("c1", "cat > f.py << 'EOF'\nv1\nEOF"),
        _tool("c1", "(no output)\n\n(exit_code=0)"),
        _assistant_bash("c2", "cat > f.py << 'EOF'\nv2\nEOF"),
        _tool("c2", "(no output)\n\n(exit_code=0)"),
    ]
    config = cm.CompactionConfig(write_recency_keep=2)
    stats = cm.CompactionStats()
    out = await cm.build_model_messages(
        raw, config=config, model=None, ops=ops, spilled_indices=set(), stats=stats
    )
    turn1_args = json.loads(out[2]["tool_calls"][0]["function"]["arguments"])
    turn2_args = json.loads(out[4]["tool_calls"][0]["function"]["arguments"])
    assert "wrote" in turn1_args["command"]  # superseded -> stubbed despite being only 1 turn old
    assert turn2_args["command"] == "cat > f.py << 'EOF'\nv2\nEOF"  # latest write kept in full
    assert stats.stubbed_writes == 1


async def test_edit_tool_write_gets_stubbed_and_keeps_path(ops):
    raw = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "task"},
        _assistant_edit("c1", "create", {"path": "f.py", "content": "line1\nline2\n"}),
        _tool("c1", "create f.py: 2 lines (exit_code=0)"),
        _assistant_bash("c2", "ls"),
        _tool("c2", "f.py\n\n(exit_code=0)"),
        _assistant_bash("c3", "ls -la"),
        _tool("c3", "total 0\n\n(exit_code=0)"),
        _assistant_bash("c4", "pwd"),
        _tool("c4", "/\n\n(exit_code=0)"),
    ]
    config = cm.CompactionConfig(write_recency_keep=2)
    stats = cm.CompactionStats()
    out = await cm.build_model_messages(
        raw, config=config, model=None, ops=ops, spilled_indices=set(), stats=stats
    )
    stubbed = json.loads(out[2]["tool_calls"][0]["function"]["arguments"])
    assert stubbed["path"] == "f.py"
    assert "wrote" in stubbed["note"]
    assert "content" not in stubbed  # bulky payload dropped


async def test_stubbing_disabled_leaves_history_untouched(ops):
    raw = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "task"},
        _assistant_bash("c1", "cat > f.py << 'EOF'\nprint(1)\nEOF"),
        _tool("c1", "(no output)\n\n(exit_code=0)"),
        _assistant_bash("c2", "ls"),
        _tool("c2", "f.py\n\n(exit_code=0)"),
        _assistant_bash("c3", "ls -la"),
        _tool("c3", "total 0\n\n(exit_code=0)"),
        _assistant_bash("c4", "pwd"),
        _tool("c4", "/\n\n(exit_code=0)"),
    ]
    config = cm.CompactionConfig(stub_file_writes=False)
    stats = cm.CompactionStats()
    out = await cm.build_model_messages(
        raw, config=config, model=None, ops=ops, spilled_indices=set(), stats=stats
    )
    args = json.loads(out[2]["tool_calls"][0]["function"]["arguments"])
    assert args["command"] == "cat > f.py << 'EOF'\nprint(1)\nEOF"
    assert stats.stubbed_writes == 0


# ---------------------------------------------------------------------------
# truncation + spill
# ---------------------------------------------------------------------------


async def test_truncate_preserves_exit_code_and_tail_and_spills_full_output(ops, workdir):
    lines = [f"line{i}" for i in range(1, 201)]
    body = "\n".join(lines)
    content = f"{body}\n\n(exit_code=0)"
    raw = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "task"},
        _assistant_bash("c1", "big_output_cmd"),
        _tool("c1", content),
    ]
    config = cm.CompactionConfig(
        max_tool_output_tokens=10, head_lines=5, tail_lines=5, spill_dir=str(workdir / "spill")
    )
    stats = cm.CompactionStats()
    out = await cm.build_model_messages(
        raw, config=config, model=None, ops=ops, spilled_indices=set(), stats=stats
    )
    tool_content = out[3]["content"]
    assert "(exit_code=0)" in tool_content
    assert "line196" in tool_content and "line200" in tool_content  # tail kept verbatim
    assert "line1\n" in tool_content or tool_content.startswith("line1")  # head kept
    assert "line100" not in tool_content  # middle elided
    assert "truncated" in tool_content
    assert stats.truncated_outputs == 1

    spill_path = cm._spill_path(config, 3)  # idx 3 = the tool message
    assert spill_path in tool_content
    assert body in Path(spill_path).read_text()


async def test_short_tool_output_is_not_truncated(ops):
    raw = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "task"},
        _assistant_bash("c1", "echo hi"),
        _tool("c1", "hi\n\n(exit_code=0)"),
    ]
    config = cm.CompactionConfig(max_tool_output_tokens=2000)
    stats = cm.CompactionStats()
    out = await cm.build_model_messages(
        raw, config=config, model=None, ops=ops, spilled_indices=set(), stats=stats
    )
    assert out[3]["content"] == "hi\n\n(exit_code=0)"
    assert stats.truncated_outputs == 0


# ---------------------------------------------------------------------------
# Regression: Fix 2/3 — spilling large/binary-looking tool output must not crash
# ---------------------------------------------------------------------------


async def test_spill_multi_megabyte_output_does_not_use_exec_argv(ops, workdir):
    lines = [f"line{i}: " + "x" * 200 for i in range(1, 20000)]
    body = "\n".join(lines)
    assert len(body) > 3_000_000
    content = f"{body}\n\n(exit_code=0)"
    raw = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "task"},
        _assistant_bash("c1", "big_output_cmd"),
        _tool("c1", content),
    ]
    config = cm.CompactionConfig(max_tool_output_tokens=10, spill_dir=str(workdir / "spill"))
    stats = cm.CompactionStats()
    out = await cm.build_model_messages(
        raw, config=config, model=None, ops=ops, spilled_indices=set(), stats=stats
    )
    assert stats.truncated_outputs == 1
    spill_path = cm._spill_path(config, 3)
    assert Path(spill_path).stat().st_size > 3_000_000


async def test_spill_output_containing_null_byte_does_not_crash(ops, workdir):
    # Enough lines that head/tail truncation actually fires (needs more than
    # head_lines+tail_lines=80 lines); the null byte sits in the elided
    # middle, so it only ever needs to survive the *spill* write, not
    # display.
    lines = [f"line{i}" for i in range(1, 101)]
    lines[50] = "line51\x00\x01binary-ish-content-here"
    body = "\n".join(lines)
    content = f"{body}\n\n(exit_code=0)"
    raw = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "task"},
        _assistant_bash("c1", "weird_cmd"),
        _tool("c1", content),
    ]
    config = cm.CompactionConfig(max_tool_output_tokens=10, spill_dir=str(workdir / "spill"))
    stats = cm.CompactionStats()
    out = await cm.build_model_messages(
        raw, config=config, model=None, ops=ops, spilled_indices=set(), stats=stats
    )
    assert stats.truncated_outputs == 1
    spill_path = cm._spill_path(config, 3)
    assert "\x00" in Path(spill_path).read_text(errors="replace")


# ---------------------------------------------------------------------------
# Regression: Fix 4 — every rebuilt message must round-trip through JSON
# ---------------------------------------------------------------------------


def test_ensure_json_safe_repairs_malformed_arguments():
    msg = {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": "call_broken",
                "type": "function",
                "function": {"name": "bash", "arguments": '{"command": "echo \\"unterminated'},
            }
        ],
    }
    fixed = cm._ensure_json_safe(msg, logger=None)
    # round-trips cleanly now
    json.dumps(fixed)
    args = json.loads(fixed["tool_calls"][0]["function"]["arguments"])
    assert "error" in args
    # tool_call_id preserved so the paired tool-response message still lines up
    assert fixed["tool_calls"][0]["id"] == "call_broken"


def test_ensure_json_safe_leaves_valid_message_untouched():
    msg = {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {"id": "c1", "type": "function", "function": {"name": "bash", "arguments": '{"command": "ls"}'}}
        ],
    }
    fixed = cm._ensure_json_safe(msg, logger=None)
    assert fixed == msg


async def test_build_model_messages_repairs_malformed_raw_arguments_end_to_end(ops):
    # Simulates a message that arrived in the raw log already malformed
    # (e.g. an upstream tool-call-parser truncation artifact) — compaction
    # itself never introduces this, but every message it emits must still be
    # guaranteed JSON-safe regardless of what it inherited.
    raw = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "task"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_broken",
                    "type": "function",
                    "function": {"name": "bash", "arguments": '{"command": "cat > f.py << \'EOF\'\\nunterm'},
                }
            ],
        },
        _tool("call_broken", "(no output)\n\n(exit_code=0)"),
    ]
    config = cm.CompactionConfig()
    stats = cm.CompactionStats()
    out = await cm.build_model_messages(
        raw, config=config, model=None, ops=ops, spilled_indices=set(), stats=stats
    )
    for m in out:
        json.dumps(m)  # every message must round-trip
        for tc in m.get("tool_calls") or []:
            json.loads(tc["function"]["arguments"])  # arguments must independently parse


async def test_rebuilt_messages_always_json_dumpable_with_adversarial_content(ops):
    adversarial = 'quotes " and \\ backslashes and \n newlines and \x00 null and emoji 🎉'
    raw = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "task"},
        _assistant_bash("c1", f"cat > f.py << 'EOF'\n{adversarial}\nEOF"),
        _tool("c1", f"{adversarial}\n\n(exit_code=0)"),
    ]
    config = cm.CompactionConfig(write_recency_keep=0)  # force stubbing on this very turn
    stats = cm.CompactionStats()
    out = await cm.build_model_messages(
        raw, config=config, model=None, ops=ops, spilled_indices=set(), stats=stats
    )
    for m in out:
        json.dumps(m)


# ---------------------------------------------------------------------------
# determinism / idempotency / no mutation of the raw log
# ---------------------------------------------------------------------------


async def test_rebuild_is_deterministic_and_never_mutates_raw_log(ops, workdir):
    body = "\n".join(f"line{i}" for i in range(1, 201))
    raw = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "task"},
        _assistant_bash("c1", "cat > f.py << 'EOF'\n" + body + "\nEOF"),
        _tool("c1", f"{body}\n\n(exit_code=0)"),
        _assistant_bash("c2", "ls"),
        _tool("c2", "f.py\n\n(exit_code=0)"),
        _assistant_bash("c3", "cat f.py"),
        _tool("c3", "checked\n\n(exit_code=0)"),
    ]
    raw_snapshot = copy.deepcopy(raw)
    config = cm.CompactionConfig(write_recency_keep=1, max_tool_output_tokens=10, spill_dir=str(workdir / "spill"))

    out1 = await cm.build_model_messages(
        raw, config=config, model=None, ops=ops, spilled_indices=set(), stats=cm.CompactionStats()
    )
    out2 = await cm.build_model_messages(
        raw, config=config, model=None, ops=ops, spilled_indices=set(), stats=cm.CompactionStats()
    )

    assert out1 == out2  # rebuilding from the same raw log twice is byte-identical
    assert raw == raw_snapshot  # the raw log itself was never mutated


# ---------------------------------------------------------------------------
# Char-fallback truncation — a token-heavy but line-sparse output (one huge
# minified line) must still be bounded, not kept in full.
# ---------------------------------------------------------------------------


async def test_line_sparse_giant_output_gets_char_truncated(ops, workdir):
    body = "x" * 60_000  # one line, no newlines to head/tail by
    content = f"{body}\n\n(exit_code=0)"
    raw = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "task"},
        _assistant_bash("c1", "dump_minified"),
        _tool("c1", content),
    ]
    config = cm.CompactionConfig(
        max_tool_output_tokens=2000, head_lines=5, tail_lines=5, spill_dir=str(workdir / "spill")
    )
    stats = cm.CompactionStats()
    out = await cm.build_model_messages(
        raw, config=config, model=None, ops=ops, spilled_indices=set(), stats=stats
    )
    tool_content = out[3]["content"]
    assert len(tool_content) < 5_000  # bounded, nowhere near 60k
    assert "truncated" in tool_content and "chars" in tool_content
    assert "(exit_code=0)" in tool_content
    assert stats.truncated_outputs == 1
    spill_path = cm._spill_path(config, 3)
    assert body in Path(spill_path).read_text()  # full body re-fetchable from disk


async def test_line_sparse_output_under_cap_kept(ops):
    content = ("y" * 400) + "\n\n(exit_code=0)"
    raw = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "task"},
        _assistant_bash("c1", "small"),
        _tool("c1", content),
    ]
    config = cm.CompactionConfig(max_tool_output_tokens=2000)
    stats = cm.CompactionStats()
    out = await cm.build_model_messages(
        raw, config=config, model=None, ops=ops, spilled_indices=set(), stats=stats
    )
    assert out[3]["content"] == content


# ---------------------------------------------------------------------------
# Stale-output hard truncation (overflow-recovery lever)
# ---------------------------------------------------------------------------


async def test_stale_outputs_forced_down_recent_kept(ops, workdir):
    old_output = "\n".join(f"old{i}" for i in range(1, 31)) + "\n\n(exit_code=0)"
    new_output = "\n".join(f"new{i}" for i in range(1, 31)) + "\n\n(exit_code=0)"
    raw = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "task"},
        _assistant_bash("c1", "step one"),
        _tool("c1", old_output),
        _assistant_bash("c2", "step two"),
        _tool("c2", "mid\n\n(exit_code=0)"),
        _assistant_bash("c3", "step three"),
        _tool("c3", new_output),
    ]
    # Both 30-line outputs are far below the token cap — only staleness
    # (older than 1 turn from the end) forces the first one down.
    config = cm.CompactionConfig(
        max_tool_output_tokens=2000,
        stale_output_keep_turns=1,
        stale_output_head_lines=3,
        stale_output_tail_lines=2,
        spill_dir=str(workdir / "spill"),
    )
    stats = cm.CompactionStats()
    out = await cm.build_model_messages(
        raw, config=config, model=None, ops=ops, spilled_indices=set(), stats=stats
    )
    assert "truncated" in out[3]["content"] and "old15" not in out[3]["content"]
    assert "old1" in out[3]["content"] and "old30" in out[3]["content"]  # tiny head/tail kept
    assert out[7]["content"] == new_output  # the latest turn keeps its full output
    assert cm._spill_path(config, 3) in out[3]["content"]


async def test_stale_lever_off_by_default(ops):
    output = "\n".join(f"l{i}" for i in range(1, 31)) + "\n\n(exit_code=0)"
    raw = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "task"},
        _assistant_bash("c1", "one"),
        _tool("c1", output),
        _assistant_bash("c2", "two"),
        _tool("c2", "ok\n\n(exit_code=0)"),
    ]
    config = cm.CompactionConfig(max_tool_output_tokens=2000)
    stats = cm.CompactionStats()
    out = await cm.build_model_messages(
        raw, config=config, model=None, ops=ops, spilled_indices=set(), stats=stats
    )
    assert out[3]["content"] == output


# ---------------------------------------------------------------------------
# Heredoc bodies are file CONTENT — redirects inside them are not writes.
# ---------------------------------------------------------------------------


def test_heredoc_body_redirects_are_not_phantom_writes():
    cmd = "cat > real.sh << 'EOF'\necho hi > phantom.txt\ndata | tee ghost.log\nEOF"
    assert cm.detect_bash_write_targets(cmd) == [("real.sh", 2)]


def test_redirect_after_heredoc_still_detected():
    cmd = "cat > real.txt << 'EOF'\nbody\nEOF\necho done > after.txt"
    targets = cm.detect_bash_write_targets(cmd)
    assert ("real.txt", 1) in targets
    assert ("after.txt", None) in targets


# ---------------------------------------------------------------------------
# escalate_config — the overflow-recovery ladder
# ---------------------------------------------------------------------------


def test_escalate_config_level_zero_is_identity():
    config = cm.CompactionConfig()
    assert cm.escalate_config(config, 0) is config


def test_escalate_config_levels_are_monotonically_harsher():
    base = cm.CompactionConfig()
    l1 = cm.escalate_config(base, 1)
    l2 = cm.escalate_config(base, 2)
    assert l1.max_tool_output_tokens < base.max_tool_output_tokens
    assert l2.max_tool_output_tokens < l1.max_tool_output_tokens
    assert l1.stale_output_keep_turns is not None
    assert l2.stale_output_keep_turns < l1.stale_output_keep_turns
    assert l2.write_recency_keep == 0
    # levels beyond MAX clamp to the harshest config
    assert cm.escalate_config(base, 5) == l2
    # base config is never mutated
    assert base.max_tool_output_tokens == 2000 and base.stale_output_keep_turns is None


# ---------------------------------------------------------------------------
# estimate_messages_tokens — must see tool-call arguments (measured: heredoc
# arguments, not tool output, dominated real overflowing trajectories).
# ---------------------------------------------------------------------------


def test_estimate_counts_tool_call_arguments():
    small = [{"role": "user", "content": "hi"}]
    with_args = small + [_assistant_bash("c1", "cat > f << 'EOF'\n" + "x" * 8000 + "\nEOF")]
    assert cm.estimate_messages_tokens(with_args, None) > cm.estimate_messages_tokens(small, None) + 1500
