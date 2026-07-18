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


async def test_recent_sole_write_stays_unstubbed(exec_fn):
    raw = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "task"},
        _assistant_bash("c1", "cat > f.py << 'EOF'\nprint(1)\nEOF"),
        _tool("c1", "(no output)\n\n(exit_code=0)"),
    ]
    config = cm.CompactionConfig(write_recency_keep=2)
    stats = cm.CompactionStats()
    out = await cm.build_model_messages(
        raw, config=config, model=None, exec_fn=exec_fn, spilled_indices=set(), stats=stats
    )
    args = json.loads(out[2]["tool_calls"][0]["function"]["arguments"])
    assert args["command"] == "cat > f.py << 'EOF'\nprint(1)\nEOF"
    assert stats.stubbed_writes == 0


async def test_stale_sole_write_gets_stubbed_past_recency_window(exec_fn):
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
        raw, config=config, model=None, exec_fn=exec_fn, spilled_indices=set(), stats=stats
    )
    args = json.loads(out[2]["tool_calls"][0]["function"]["arguments"])
    assert "wrote" in args["command"] and "f.py" in args["command"]
    assert "print(1)" not in args["command"]
    assert stats.stubbed_writes == 1
    # the exit code / tool result for the write itself is untouched
    assert out[3]["content"] == "(no output)\n\n(exit_code=0)"


async def test_superseded_write_is_stubbed_even_if_recent(exec_fn):
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
        raw, config=config, model=None, exec_fn=exec_fn, spilled_indices=set(), stats=stats
    )
    turn1_args = json.loads(out[2]["tool_calls"][0]["function"]["arguments"])
    turn2_args = json.loads(out[4]["tool_calls"][0]["function"]["arguments"])
    assert "wrote" in turn1_args["command"]  # superseded -> stubbed despite being only 1 turn old
    assert turn2_args["command"] == "cat > f.py << 'EOF'\nv2\nEOF"  # latest write kept in full
    assert stats.stubbed_writes == 1


async def test_edit_tool_write_gets_stubbed_and_keeps_path(exec_fn):
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
        raw, config=config, model=None, exec_fn=exec_fn, spilled_indices=set(), stats=stats
    )
    stubbed = json.loads(out[2]["tool_calls"][0]["function"]["arguments"])
    assert stubbed["path"] == "f.py"
    assert "wrote" in stubbed["note"]
    assert "content" not in stubbed  # bulky payload dropped


async def test_stubbing_disabled_leaves_history_untouched(exec_fn):
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
        raw, config=config, model=None, exec_fn=exec_fn, spilled_indices=set(), stats=stats
    )
    args = json.loads(out[2]["tool_calls"][0]["function"]["arguments"])
    assert args["command"] == "cat > f.py << 'EOF'\nprint(1)\nEOF"
    assert stats.stubbed_writes == 0


# ---------------------------------------------------------------------------
# truncation + spill
# ---------------------------------------------------------------------------


async def test_truncate_preserves_exit_code_and_tail_and_spills_full_output(exec_fn, workdir):
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
        raw, config=config, model=None, exec_fn=exec_fn, spilled_indices=set(), stats=stats
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


async def test_short_tool_output_is_not_truncated(exec_fn):
    raw = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "task"},
        _assistant_bash("c1", "echo hi"),
        _tool("c1", "hi\n\n(exit_code=0)"),
    ]
    config = cm.CompactionConfig(max_tool_output_tokens=2000)
    stats = cm.CompactionStats()
    out = await cm.build_model_messages(
        raw, config=config, model=None, exec_fn=exec_fn, spilled_indices=set(), stats=stats
    )
    assert out[3]["content"] == "hi\n\n(exit_code=0)"
    assert stats.truncated_outputs == 0


# ---------------------------------------------------------------------------
# determinism / idempotency / no mutation of the raw log
# ---------------------------------------------------------------------------


async def test_rebuild_is_deterministic_and_never_mutates_raw_log(exec_fn, workdir):
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
        raw, config=config, model=None, exec_fn=exec_fn, spilled_indices=set(), stats=cm.CompactionStats()
    )
    out2 = await cm.build_model_messages(
        raw, config=config, model=None, exec_fn=exec_fn, spilled_indices=set(), stats=cm.CompactionStats()
    )

    assert out1 == out2  # rebuilding from the same raw log twice is byte-identical
    assert raw == raw_snapshot  # the raw log itself was never mutated
