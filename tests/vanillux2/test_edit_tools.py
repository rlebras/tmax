import edit_tools  # noqa: E402 (sys.path set up by conftest)
import pytest
from rl_data.generator.sample_solutions import SUBMIT_MARKER


# ---------------------------------------------------------------------------
# extract_action
# ---------------------------------------------------------------------------


def test_extract_action_bash():
    msg = {"tool_calls": [{"id": "c1", "function": {"name": "bash", "arguments": '{"command": "ls"}'}}]}
    action = edit_tools.extract_action(msg)
    assert action == {"type": "tool", "name": "bash", "args": {"command": "ls"}, "tool_call_id": "c1"}


def test_extract_action_submit_marker():
    msg = {
        "tool_calls": [
            {
                "id": "c2",
                "function": {"name": "bash", "arguments": f'{{"command": "echo {SUBMIT_MARKER}"}}'},
            }
        ]
    }
    action = edit_tools.extract_action(msg)
    assert action["type"] == "done"
    assert action["name"] == "bash"


def test_extract_action_edit_tool():
    msg = {
        "tool_calls": [
            {
                "id": "c3",
                "function": {
                    "name": "str_replace",
                    "arguments": '{"path": "a.py", "old_string": "x", "new_string": "y"}',
                },
            }
        ]
    }
    action = edit_tools.extract_action(msg)
    assert action["type"] == "tool"
    assert action["name"] == "str_replace"
    assert action["args"]["path"] == "a.py"


def test_extract_action_no_tool_call():
    assert edit_tools.extract_action({"content": "just talking"})["type"] == "no_tool_call"


def test_extract_action_unknown_tool_name():
    msg = {"tool_calls": [{"id": "c4", "function": {"name": "frobnicate", "arguments": "{}"}}]}
    action = edit_tools.extract_action(msg)
    assert action["type"] == "no_tool_call"


def test_extract_action_malformed_arguments():
    msg = {"tool_calls": [{"id": "c5", "function": {"name": "bash", "arguments": "{not json"}}]}
    action = edit_tools.extract_action(msg)
    assert action["type"] == "no_tool_call"


# ---------------------------------------------------------------------------
# create / read
# ---------------------------------------------------------------------------


async def test_create_and_read_back(ops, workdir):
    path = str(workdir / "hello.py")
    result = await edit_tools.create(ops, path, "print('hi')\n")
    assert "create" in result and "(exit_code=0)" in result
    assert (workdir / "hello.py").read_text() == "print('hi')\n"


async def test_create_empty_file(ops, workdir):
    path = str(workdir / "empty.txt")
    await edit_tools.create(ops, path, "")
    assert (workdir / "empty.txt").read_text() == ""


async def test_create_overwrites_existing(ops, workdir):
    p = workdir / "a.py"
    p.write_text("old\n")
    await edit_tools.create(ops, str(p), "new\n")
    assert p.read_text() == "new\n"


async def test_create_resolves_relative_path(ops, workdir):
    # exec_fn's cwd IS workdir, so a bare relative path should resolve there.
    await edit_tools.create(ops, "relative.py", "x = 1\n")
    assert (workdir / "relative.py").read_text() == "x = 1\n"


async def test_read_file_range_window(ops, workdir):
    p = workdir / "a.py"
    p.write_text("\n".join(f"line{i}" for i in range(1, 11)) + "\n")
    result = await edit_tools.read_file_range(ops, str(p), 2, 4)
    assert "line2" in result and "line4" in result
    assert "line1" not in result and "line5" not in result
    assert "more lines" in result  # 6 remaining after line 4


async def test_read_file_range_defaults_to_whole_small_file(ops, workdir):
    p = workdir / "a.py"
    p.write_text("line1\nline2\n")
    result = await edit_tools.read_file_range(ops, str(p), None, None)
    assert "line1" in result and "line2" in result
    assert "more lines" not in result


async def test_read_missing_file_raises(ops, workdir):
    with pytest.raises(FileNotFoundError):
        await edit_tools.read_file_range(ops, str(workdir / "nope.txt"), None, None)


# ---------------------------------------------------------------------------
# str_replace: 0 / 1 / 2+ matches, whitespace tolerance
# ---------------------------------------------------------------------------


async def test_str_replace_single_match(ops, workdir):
    p = workdir / "a.py"
    p.write_text("def f():\n    return 1\n")
    result = await edit_tools.str_replace(ops, str(p), "return 1", "return 2")
    assert "-1/+1" in result and "(exit_code=0)" in result
    assert p.read_text() == "def f():\n    return 2\n"


async def test_str_replace_is_whitespace_tolerant(ops, workdir):
    p = workdir / "a.py"
    p.write_text("def f():\n        return   1\n")  # irregular spacing
    result = await edit_tools.str_replace(ops, str(p), "return 1", "return 2")
    assert "(exit_code=0)" in result
    assert "return 2" in p.read_text()


async def test_str_replace_zero_matches_returns_nearest_lines_not_whole_file(ops, workdir):
    p = workdir / "a.py"
    original = "\n".join(f"line{i}" for i in range(1, 51)) + "\n"
    p.write_text(original)
    result = await edit_tools.str_replace(ops, str(p), "line23_typo", "replacement")
    assert "0 matches" in result
    assert "nearest lines" in result
    # never dumps the whole file
    assert result.count("line") < 50
    assert p.read_text() == original  # untouched


async def test_str_replace_multiple_matches_rejected(ops, workdir):
    p = workdir / "a.py"
    original = "x = 1\nx = 1\n"
    p.write_text(original)
    result = await edit_tools.str_replace(ops, str(p), "x = 1", "x = 2")
    assert "2 matches" in result
    assert "unique anchor" in result
    assert p.read_text() == original  # untouched


async def test_str_replace_missing_file(ops, workdir):
    with pytest.raises(FileNotFoundError):
        await edit_tools.str_replace(ops, str(workdir / "nope.py"), "a", "b")


# ---------------------------------------------------------------------------
# insert
# ---------------------------------------------------------------------------


async def test_insert_mid_file(ops, workdir):
    p = workdir / "a.py"
    p.write_text("line1\nline2\n")
    result = await edit_tools.insert(ops, str(p), 1, "inserted")
    assert "@L1" in result
    assert p.read_text() == "line1\ninserted\nline2\n"


async def test_insert_at_start(ops, workdir):
    p = workdir / "a.py"
    p.write_text("line1\n")
    await edit_tools.insert(ops, str(p), 0, "line0")
    assert p.read_text() == "line0\nline1\n"


async def test_insert_out_of_range(ops, workdir):
    p = workdir / "a.py"
    p.write_text("line1\n")
    result = await edit_tools.insert(ops, str(p), 5, "x")
    assert "out of range" in result
    assert p.read_text() == "line1\n"  # untouched


# ---------------------------------------------------------------------------
# apply_edits: all-or-nothing
# ---------------------------------------------------------------------------


async def test_apply_edits_all_succeed(ops, workdir):
    p = workdir / "a.py"
    p.write_text("a = 1\nb = 2\n")
    edits = [
        {"old_string": "a = 1", "new_string": "a = 10"},
        {"old_string": "b = 2", "new_string": "b = 20"},
    ]
    result = await edit_tools.apply_edits(ops, str(p), edits)
    assert "2 edits applied" in result
    assert p.read_text() == "a = 10\nb = 20\n"


async def test_apply_edits_atomic_on_partial_failure(ops, workdir):
    p = workdir / "a.py"
    original = "a = 1\nb = 2\n"
    p.write_text(original)
    edits = [
        {"old_string": "a = 1", "new_string": "a = 10"},  # would succeed alone
        {"old_string": "NOPE", "new_string": "x"},  # fails
    ]
    result = await edit_tools.apply_edits(ops, str(p), edits)
    assert "aborted, disk unchanged" in result
    assert p.read_text() == original  # edit 0 was NOT partially applied


async def test_apply_edits_no_edits(ops, workdir):
    p = workdir / "a.py"
    p.write_text("x\n")
    result = await edit_tools.apply_edits(ops, str(p), [])
    assert "no edits given" in result


# ---------------------------------------------------------------------------
# atomicity of the underlying write primitive
# ---------------------------------------------------------------------------


async def test_write_file_atomic_failure_is_safe(ops, workdir):
    missing_dir_path = workdir / "no_such_dir" / "a.py"
    with pytest.raises(OSError):
        await edit_tools._write_file_atomic(ops, str(missing_dir_path), "content")
    assert not (workdir / "no_such_dir").exists()


# ---------------------------------------------------------------------------
# Regression: Fix 2 (ARG_MAX) — large content must never touch exec argv
# ---------------------------------------------------------------------------


async def test_str_replace_on_multi_megabyte_file_does_not_use_exec_argv(ops, workdir):
    # >2MB, comfortably past typical ARG_MAX (~2MB total argv+envp on Linux).
    # Before the fix this content was embedded in a heredoc `command` string
    # passed through exec_fn -> environment.exec() -> docker argv.
    big_line = "x" * 1000
    original = "\n".join(big_line for _ in range(1, 2200)) + "\nTARGET_LINE\n" + big_line + "\n"
    p = workdir / "big.py"
    p.write_text(original)
    assert p.stat().st_size > 2_000_000

    result = await edit_tools.str_replace(ops, str(p), "TARGET_LINE", "REPLACED_LINE")

    assert "(exit_code=0)" in result
    assert "REPLACED_LINE" in p.read_text()
    assert p.stat().st_size > 2_000_000  # still a multi-MB file, wrote fine


async def test_create_multi_megabyte_file(ops, workdir):
    content = "y" * 3_000_000
    p = workdir / "huge.txt"
    result = await edit_tools.create(ops, str(p), content)
    assert "(exit_code=0)" in result
    assert p.stat().st_size == 3_000_000


# ---------------------------------------------------------------------------
# Regression: Fix 3 (embedded null byte / binary files)
# ---------------------------------------------------------------------------


def test_looks_binary_detects_null_byte():
    assert edit_tools._looks_binary(b"hello\x00world") is True


def test_looks_binary_detects_invalid_utf8():
    assert edit_tools._looks_binary(b"\xff\xfe\x00\x01garbage") is True


def test_looks_binary_false_for_plain_text():
    assert edit_tools._looks_binary(b"hello world\nline two\n") is False


async def test_str_replace_refuses_binary_file_without_crashing(ops, workdir):
    p = workdir / "binary.dat"
    p.write_bytes(b"\x00\x01\x02binary\xffcontent\x00more")
    result = await edit_tools.str_replace(ops, str(p), "binary", "text")
    assert "binary" in result.lower()
    assert p.read_bytes() == b"\x00\x01\x02binary\xffcontent\x00more"  # untouched


async def test_insert_refuses_binary_file_without_crashing(ops, workdir):
    p = workdir / "binary.dat"
    p.write_bytes(b"\x00\x01\x02\x03")
    result = await edit_tools.insert(ops, str(p), 0, "text")
    assert "binary" in result.lower()


async def test_apply_edits_refuses_binary_file_without_crashing(ops, workdir):
    p = workdir / "binary.dat"
    p.write_bytes(b"\x00\x01\x02\x03")
    result = await edit_tools.apply_edits(ops, str(p), [{"old_string": "a", "new_string": "b"}])
    assert "binary" in result.lower()


async def test_read_binary_file_returns_safe_summary_not_crash(ops, workdir):
    p = workdir / "binary.dat"
    data = b"\x00\x01\x02\xff\xfe" + b"more binary data here"
    p.write_bytes(data)
    result = await edit_tools.read_file_range(ops, str(p), None, None)
    assert "binary file" in result
    assert str(len(data)) in result


async def test_create_file_containing_embedded_null_byte_via_json_escape(ops, workdir):
    # A model CAN emit a literal NUL through a valid JSON unicode escape;
    # json.loads decodes that to a real \x00 in the Python str. create()
    # must not crash even though its own content now contains one.
    p = workdir / "has_null.bin"
    content_with_null = "before\x00after\n"
    result = await edit_tools.create(ops, str(p), content_with_null)
    assert "(exit_code=0)" in result
    assert p.read_bytes() == content_with_null.encode("utf-8")


async def test_dispatch_str_replace_on_binary_file_returns_error_not_crash(ops, workdir):
    p = workdir / "binary.dat"
    p.write_bytes(b"\x00\x01\x02binary")
    result = await edit_tools.dispatch(
        "str_replace", {"path": str(p), "old_string": "a", "new_string": "b"}, ops
    )
    assert "binary" in result.lower()


# ---------------------------------------------------------------------------
# dispatch: never raises, always returns model-facing text
# ---------------------------------------------------------------------------


async def test_dispatch_missing_argument(ops):
    result = await edit_tools.dispatch("create", {}, ops)
    assert result.startswith("error: missing required argument")


async def test_dispatch_unknown_tool(ops):
    result = await edit_tools.dispatch("frobnicate", {}, ops)
    assert "unknown tool" in result


async def test_dispatch_missing_file_becomes_error_string(ops, workdir):
    result = await edit_tools.dispatch(
        "str_replace",
        {"path": str(workdir / "nope.py"), "old_string": "a", "new_string": "b"},
        ops,
    )
    assert result.startswith("error:")


async def test_dispatch_create_roundtrip(ops, workdir):
    path = str(workdir / "created.py")
    result = await edit_tools.dispatch("create", {"path": path, "content": "x = 1\n"}, ops)
    assert "(exit_code=0)" in result
    assert (workdir / "created.py").read_text() == "x = 1\n"


# ---------------------------------------------------------------------------
# End-to-end: model -> extract_action -> dispatch -> disk -> compact result
# ---------------------------------------------------------------------------


async def test_str_replace_end_to_end_model_to_disk(ops, workdir):
    import json

    p = workdir / "app.py"
    p.write_text("def greet():\n    return 'hello'\n")

    # A model-shaped response message calling str_replace.
    response_msg = {
        "role": "assistant",
        "content": "THOUGHT: fix the greeting",
        "tool_calls": [
            {
                "id": "call_abc",
                "type": "function",
                "function": {
                    "name": "str_replace",
                    "arguments": json.dumps(
                        {
                            "path": str(p),
                            "old_string": "return 'hello'",
                            "new_string": "return 'hi'",
                        }
                    ),
                },
            }
        ],
    }

    action = edit_tools.extract_action(response_msg)
    assert action["type"] == "tool"
    assert action["name"] == "str_replace"

    result = await edit_tools.dispatch(action["name"], action["args"], ops)

    assert "(exit_code=0)" in result
    assert "str_replace" in result
    assert p.read_text() == "def greet():\n    return 'hi'\n"
    # the compact result never contains the full file text
    assert "def greet" not in result
