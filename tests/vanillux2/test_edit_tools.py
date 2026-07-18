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


async def test_create_and_read_back(exec_fn, workdir):
    path = str(workdir / "hello.py")
    result = await edit_tools.create(exec_fn, path, "print('hi')\n")
    assert "create" in result and "(exit_code=0)" in result
    assert (workdir / "hello.py").read_text() == "print('hi')\n"


async def test_create_empty_file(exec_fn, workdir):
    path = str(workdir / "empty.txt")
    await edit_tools.create(exec_fn, path, "")
    assert (workdir / "empty.txt").read_text() == ""


async def test_create_overwrites_existing(exec_fn, workdir):
    p = workdir / "a.py"
    p.write_text("old\n")
    await edit_tools.create(exec_fn, str(p), "new\n")
    assert p.read_text() == "new\n"


async def test_read_file_range_window(exec_fn, workdir):
    p = workdir / "a.py"
    p.write_text("\n".join(f"line{i}" for i in range(1, 11)) + "\n")
    result = await edit_tools.read_file_range(exec_fn, str(p), 2, 4)
    assert "line2" in result and "line4" in result
    assert "line1" not in result and "line5" not in result
    assert "more lines" in result  # 6 remaining after line 4


async def test_read_file_range_defaults_to_whole_small_file(exec_fn, workdir):
    p = workdir / "a.py"
    p.write_text("line1\nline2\n")
    result = await edit_tools.read_file_range(exec_fn, str(p), None, None)
    assert "line1" in result and "line2" in result
    assert "more lines" not in result


async def test_read_missing_file_raises(exec_fn, workdir):
    with pytest.raises(FileNotFoundError):
        await edit_tools.read_file_range(exec_fn, str(workdir / "nope.txt"), None, None)


# ---------------------------------------------------------------------------
# str_replace: 0 / 1 / 2+ matches, whitespace tolerance
# ---------------------------------------------------------------------------


async def test_str_replace_single_match(exec_fn, workdir):
    p = workdir / "a.py"
    p.write_text("def f():\n    return 1\n")
    result = await edit_tools.str_replace(exec_fn, str(p), "return 1", "return 2")
    assert "-1/+1" in result and "(exit_code=0)" in result
    assert p.read_text() == "def f():\n    return 2\n"


async def test_str_replace_is_whitespace_tolerant(exec_fn, workdir):
    p = workdir / "a.py"
    p.write_text("def f():\n        return   1\n")  # irregular spacing
    result = await edit_tools.str_replace(exec_fn, str(p), "return 1", "return 2")
    assert "(exit_code=0)" in result
    assert "return 2" in p.read_text()


async def test_str_replace_zero_matches_returns_nearest_lines_not_whole_file(exec_fn, workdir):
    p = workdir / "a.py"
    original = "\n".join(f"line{i}" for i in range(1, 51)) + "\n"
    p.write_text(original)
    result = await edit_tools.str_replace(exec_fn, str(p), "line23_typo", "replacement")
    assert "0 matches" in result
    assert "nearest lines" in result
    # never dumps the whole file
    assert result.count("line") < 50
    assert p.read_text() == original  # untouched


async def test_str_replace_multiple_matches_rejected(exec_fn, workdir):
    p = workdir / "a.py"
    original = "x = 1\nx = 1\n"
    p.write_text(original)
    result = await edit_tools.str_replace(exec_fn, str(p), "x = 1", "x = 2")
    assert "2 matches" in result
    assert "unique anchor" in result
    assert p.read_text() == original  # untouched


async def test_str_replace_missing_file(exec_fn, workdir):
    with pytest.raises(FileNotFoundError):
        await edit_tools.str_replace(exec_fn, str(workdir / "nope.py"), "a", "b")


# ---------------------------------------------------------------------------
# insert
# ---------------------------------------------------------------------------


async def test_insert_mid_file(exec_fn, workdir):
    p = workdir / "a.py"
    p.write_text("line1\nline2\n")
    result = await edit_tools.insert(exec_fn, str(p), 1, "inserted")
    assert "@L1" in result
    assert p.read_text() == "line1\ninserted\nline2\n"


async def test_insert_at_start(exec_fn, workdir):
    p = workdir / "a.py"
    p.write_text("line1\n")
    await edit_tools.insert(exec_fn, str(p), 0, "line0")
    assert p.read_text() == "line0\nline1\n"


async def test_insert_out_of_range(exec_fn, workdir):
    p = workdir / "a.py"
    p.write_text("line1\n")
    result = await edit_tools.insert(exec_fn, str(p), 5, "x")
    assert "out of range" in result
    assert p.read_text() == "line1\n"  # untouched


# ---------------------------------------------------------------------------
# apply_edits: all-or-nothing
# ---------------------------------------------------------------------------


async def test_apply_edits_all_succeed(exec_fn, workdir):
    p = workdir / "a.py"
    p.write_text("a = 1\nb = 2\n")
    edits = [
        {"old_string": "a = 1", "new_string": "a = 10"},
        {"old_string": "b = 2", "new_string": "b = 20"},
    ]
    result = await edit_tools.apply_edits(exec_fn, str(p), edits)
    assert "2 edits applied" in result
    assert p.read_text() == "a = 10\nb = 20\n"


async def test_apply_edits_atomic_on_partial_failure(exec_fn, workdir):
    p = workdir / "a.py"
    original = "a = 1\nb = 2\n"
    p.write_text(original)
    edits = [
        {"old_string": "a = 1", "new_string": "a = 10"},  # would succeed alone
        {"old_string": "NOPE", "new_string": "x"},  # fails
    ]
    result = await edit_tools.apply_edits(exec_fn, str(p), edits)
    assert "aborted, disk unchanged" in result
    assert p.read_text() == original  # edit 0 was NOT partially applied


async def test_apply_edits_no_edits(exec_fn, workdir):
    p = workdir / "a.py"
    p.write_text("x\n")
    result = await edit_tools.apply_edits(exec_fn, str(p), [])
    assert "no edits given" in result


# ---------------------------------------------------------------------------
# atomicity of the underlying write primitive
# ---------------------------------------------------------------------------


async def test_write_file_atomic_failure_is_safe(exec_fn, workdir):
    missing_dir_path = workdir / "no_such_dir" / "a.py"
    with pytest.raises(OSError):
        await edit_tools._write_file_atomic(exec_fn, str(missing_dir_path), "content")
    assert not (workdir / "no_such_dir").exists()


# ---------------------------------------------------------------------------
# dispatch: never raises, always returns model-facing text
# ---------------------------------------------------------------------------


async def test_dispatch_missing_argument(exec_fn):
    result = await edit_tools.dispatch("create", {}, exec_fn)
    assert result.startswith("error: missing required argument")


async def test_dispatch_unknown_tool(exec_fn):
    result = await edit_tools.dispatch("frobnicate", {}, exec_fn)
    assert "unknown tool" in result


async def test_dispatch_missing_file_becomes_error_string(exec_fn, workdir):
    result = await edit_tools.dispatch(
        "str_replace",
        {"path": str(workdir / "nope.py"), "old_string": "a", "new_string": "b"},
        exec_fn,
    )
    assert result.startswith("error:")


async def test_dispatch_create_roundtrip(exec_fn, workdir):
    path = str(workdir / "created.py")
    result = await edit_tools.dispatch("create", {"path": path, "content": "x = 1\n"}, exec_fn)
    assert "(exit_code=0)" in result
    assert (workdir / "created.py").read_text() == "x = 1\n"
