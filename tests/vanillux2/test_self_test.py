"""Unit tests for Vanillux2Agent/self_test.py — Mechanisms 1-5, standalone.

These exercise self_test.py directly (it has no ``harbor`` import) against a
``FakeEnvironment`` (see conftest.py) so no Docker/Harbor runtime is needed.
"""

from __future__ import annotations

import json

import pytest

from Vanillux2Agent import self_test as st


# ---------------------------------------------------------------------------
# agent-check parsing
# ---------------------------------------------------------------------------


def test_parse_agent_check_matches_convention():
    parsed = st.parse_agent_check("agent-check handles_empty -- pytest -q tests/test_foo.py")
    assert parsed == ("handles_empty", "pytest -q tests/test_foo.py")


def test_parse_agent_check_ignores_ordinary_commands():
    assert st.parse_agent_check("pytest -q tests/test_foo.py") is None
    assert st.parse_agent_check("echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT") is None


def test_parse_agent_check_multiline_command():
    command = "agent-check crit1 -- python3 -c \"\nimport sys\nsys.exit(0)\n\""
    parsed = st.parse_agent_check(command)
    assert parsed is not None
    cid, inner = parsed
    assert cid == "crit1"
    assert "sys.exit(0)" in inner


def test_parse_agent_check_rejects_empty_command():
    assert st.parse_agent_check("agent-check crit1 -- ") is None


def test_extract_agent_check_line_finds_embedded_invocation():
    command = "mkdir -p /app/out\nagent-check c1 -- test -f /app/out/result.txt\necho done"
    found = st.extract_agent_check_line(command)
    assert found is not None
    cid, inner, remainder = found
    assert cid == "c1"
    assert inner == "test -f /app/out/result.txt"
    assert remainder == "mkdir -p /app/out\necho done"


def test_extract_agent_check_line_no_match_for_ordinary_multiline_command():
    command = "mkdir -p /app/out\necho done"
    assert st.extract_agent_check_line(command) is None


def test_extract_agent_check_line_rejects_empty_command():
    assert st.extract_agent_check_line("mkdir -p /app\nagent-check c1 -- \necho done") is None


def test_looks_like_malformed_agent_check_true_for_botched_attempts():
    assert st.looks_like_malformed_agent_check("agent-check c1") is True
    assert st.looks_like_malformed_agent_check("agent-check c1 test -f out.txt") is True
    assert st.looks_like_malformed_agent_check("agent-check c1 -- ") is True


def test_looks_like_malformed_agent_check_false_for_valid_forms():
    assert st.looks_like_malformed_agent_check("agent-check c1 -- pytest -q") is False
    assert st.looks_like_malformed_agent_check("mkdir -p x\nagent-check c1 -- pytest -q") is False


def test_looks_like_malformed_agent_check_false_for_unrelated_commands():
    assert st.looks_like_malformed_agent_check("echo 'the agent-check convention...'") is False
    assert st.looks_like_malformed_agent_check("# agent-check is a harness convention") is False


# ---------------------------------------------------------------------------
# criteria file parsing
# ---------------------------------------------------------------------------


def test_parse_criteria_doc_valid():
    raw = json.dumps(
        {
            "criteria": [{"id": "c1", "description": "d1", "how_to_check": "h1"}],
            "deliverables": ["solution.py"],
        }
    )
    criteria, deliverables = st.parse_criteria_doc(raw)
    assert set(criteria) == {"c1"}
    assert criteria["c1"].description == "d1"
    assert deliverables == ["solution.py"]


@pytest.mark.parametrize("raw", ["", "not json", "[]", "null", "{}"])
def test_parse_criteria_doc_tolerates_malformed_input(raw):
    criteria, deliverables = st.parse_criteria_doc(raw)
    assert criteria == {}
    assert deliverables == []


def test_parse_criteria_doc_skips_incomplete_entries():
    raw = json.dumps({"criteria": [{"id": "no_description"}, {"description": "no_id"}, "not_a_dict"]})
    criteria, _ = st.parse_criteria_doc(raw)
    assert criteria == {}


@pytest.mark.asyncio
async def test_load_criteria_missing_file_is_empty(session_exec):
    config = st.SelfTestConfig(state_dir="/tmp/does-not-exist-self-test")
    criteria, deliverables = await st.load_criteria(session_exec, config)
    assert criteria == {}
    assert deliverables == []


# ---------------------------------------------------------------------------
# Mechanism 5 — circularity + weakening heuristics
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("command", ["true", ":", "exit 0", "  true  "])
def test_circularity_flags_trivial_commands(command):
    flags = st.circularity_flags(command)
    assert flags and "no-op" in flags[0]


@pytest.mark.parametrize(
    "command",
    [
        "pytest -q tests/test_foo.py",
        "diff out.txt expected.txt",
        "grep -q PASS out.txt",
        "[[ $(cat out.txt) == 'expected' ]]",
        "test -f out.txt",
    ],
)
def test_circularity_flags_real_assertions_not_flagged(command):
    assert st.circularity_flags(command) == []


def test_circularity_flags_self_diff_is_flagged():
    flags = st.circularity_flags("diff <(python3 solve.py) <(python3 solve.py)")
    assert any("independent oracle" in f for f in flags)


def test_circularity_flags_self_eq_is_flagged():
    flags = st.circularity_flags('[[ "$(python3 solve.py)" == "$(python3 solve.py)" ]]')
    assert any("independent oracle" in f for f in flags)


def test_circularity_flags_no_assertion_hint():
    flags = st.circularity_flags("python3 solve.py > /dev/null")
    assert flags and "no recognizable assertion" in flags[0]


def test_weakening_warning_no_history():
    state = st.SelfTestState()
    assert st.weakening_warning("c1", "pytest -q", state) is None


def test_weakening_warning_last_passed():
    state = st.SelfTestState()
    state.checks["c1"] = [
        st.CheckRecord("c1", "pytest -q", True, True, None, False, None, [], step=1)
    ]
    assert st.weakening_warning("c1", "pytest -q -k different", state) is None


def test_weakening_warning_same_command_after_failure():
    state = st.SelfTestState()
    state.checks["c1"] = [
        st.CheckRecord("c1", "pytest -q", True, False, "AssertionError", False, None, [], step=1)
    ]
    assert st.weakening_warning("c1", "pytest -q", state) is None


def test_weakening_warning_changed_command_after_failure():
    state = st.SelfTestState()
    state.checks["c1"] = [
        st.CheckRecord("c1", "pytest -q", True, False, "AssertionError", False, None, [], step=1)
    ]
    warning = st.weakening_warning("c1", "true", state)
    assert warning is not None
    assert "changed after its last isolated run FAILED" in warning


# ---------------------------------------------------------------------------
# Mechanism 3 — isolation, incl. the neutered-deliverable / leftover-state fixture
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_isolation_excludes_undeclared_leftover_files(fake_env, session_exec, isolated_exec):
    """A check that passes only because of a leftover scratch file in the live
    session must FAIL in isolation, since only declared deliverables are mirrored.
    """
    config = st.SelfTestConfig(state_dir=str(fake_env.root / ".agent" / "self_test"))
    (fake_env.root / "answer.txt").write_text("42")  # leftover scratch file, NOT declared
    (fake_env.root / "solution.py").write_text(
        "import pathlib\nprint(pathlib.Path('answer.txt').read_text())\n"
    )

    check_cmd = "test \"$(python3 solution.py)\" = \"42\""
    session_result = await session_exec(check_cmd)
    assert session_result.return_code == 0  # passes live: answer.txt is right there

    isolated_cwd, missing = await st.prepare_isolated_root(
        session_exec, config, deliverables=["solution.py"], cwd=str(fake_env.root), criterion_id="c1"
    )
    assert missing == []
    isolated_result = await isolated_exec(check_cmd, isolated_cwd)
    assert isolated_result.return_code != 0  # fails isolated: answer.txt was never mirrored


@pytest.mark.asyncio
async def test_isolation_catches_neutered_deliverable(fake_env, session_exec, isolated_exec):
    """A deliverable that was weakened after the live check ran (e.g. a filter
    the agent quietly neutered back) must be caught by isolation using the
    on-disk state at check time, not any earlier in-memory assumption.
    """
    config = st.SelfTestConfig(state_dir=str(fake_env.root / ".agent" / "self_test"))
    filt = fake_env.root / "filter.py"
    filt.write_text("import sys\nprint('BLOCKED' if 'bad' in sys.argv[1] else 'OK')\n")

    check_cmd = 'test "$(python3 filter.py bad_input)" = "BLOCKED"'
    isolated_cwd, missing = await st.prepare_isolated_root(
        session_exec, config, deliverables=["filter.py"], cwd=str(fake_env.root), criterion_id="c2"
    )
    assert missing == []
    first_result = await isolated_exec(check_cmd, isolated_cwd)
    assert first_result.return_code == 0

    # Agent neuters the filter (always prints OK) after the fact.
    filt.write_text("print('OK')\n")
    isolated_cwd, _ = await st.prepare_isolated_root(
        session_exec, config, deliverables=["filter.py"], cwd=str(fake_env.root), criterion_id="c2"
    )
    second_result = await isolated_exec(check_cmd, isolated_cwd)
    assert second_result.return_code != 0  # isolation mirrors the CURRENT (neutered) file


@pytest.mark.asyncio
async def test_isolation_reports_missing_deliverable(fake_env, session_exec):
    config = st.SelfTestConfig(state_dir=str(fake_env.root / ".agent" / "self_test"))
    _, missing = await st.prepare_isolated_root(
        session_exec, config, deliverables=["nope.py"], cwd=str(fake_env.root), criterion_id="c3"
    )
    assert missing == ["nope.py"]


# ---------------------------------------------------------------------------
# agent-check end to end (compact output, discrepancy, circular)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_agent_check_unknown_criterion(fake_env, session_exec, isolated_exec):
    config = st.SelfTestConfig(state_dir=str(fake_env.root / ".agent" / "self_test"))
    state = st.SelfTestState()
    result = await st.run_agent_check(
        "nope",
        "true",
        criteria={},
        deliverables=[],
        state=state,
        config=config,
        session_exec=session_exec,
        isolated_exec=isolated_exec,
        persistent_cwd=str(fake_env.root),
        step=1,
    )
    assert "unknown criterion_id" in result
    assert state.checks == {}


@pytest.mark.asyncio
async def test_run_agent_check_compact_pass(fake_env, session_exec, isolated_exec):
    config = st.SelfTestConfig(state_dir=str(fake_env.root / ".agent" / "self_test"))
    (fake_env.root / "solution.py").write_text("print(42)\n")
    state = st.SelfTestState()
    criteria = {"c1": st.Criterion(id="c1", description="prints 42")}

    result = await st.run_agent_check(
        "c1",
        'test "$(python3 solution.py)" = "42"',
        criteria=criteria,
        deliverables=["solution.py"],
        state=state,
        config=config,
        session_exec=session_exec,
        isolated_exec=isolated_exec,
        persistent_cwd=str(fake_env.root),
        step=1,
    )
    assert "PASS (isolated)" in result
    assert "42" not in result  # never echoes full output back
    record = state.checks["c1"][0]
    assert record.isolated_pass is True
    assert record.circular is False


@pytest.mark.asyncio
async def test_run_agent_check_returns_first_failure_line_only(fake_env, session_exec, isolated_exec):
    config = st.SelfTestConfig(state_dir=str(fake_env.root / ".agent" / "self_test"))
    (fake_env.root / "solution.py").write_text(
        "print('line one of a lot of output')\nprint('line two')\nraise SystemExit(1)\n"
    )
    state = st.SelfTestState()
    criteria = {"c1": st.Criterion(id="c1", description="exits 0")}

    result = await st.run_agent_check(
        "c1",
        "python3 solution.py",
        criteria=criteria,
        deliverables=["solution.py"],
        state=state,
        config=config,
        session_exec=session_exec,
        isolated_exec=isolated_exec,
        persistent_cwd=str(fake_env.root),
        step=1,
    )
    assert "FAIL (isolated)" in result
    assert "first failing line: line one of a lot of output" in result
    assert "line two" not in result


@pytest.mark.asyncio
async def test_run_agent_check_flags_circular(fake_env, session_exec, isolated_exec):
    config = st.SelfTestConfig(state_dir=str(fake_env.root / ".agent" / "self_test"))
    state = st.SelfTestState()
    criteria = {"c1": st.Criterion(id="c1", description="whatever")}

    result = await st.run_agent_check(
        "c1",
        "true",
        criteria=criteria,
        deliverables=[],
        state=state,
        config=config,
        session_exec=session_exec,
        isolated_exec=isolated_exec,
        persistent_cwd=str(fake_env.root),
        step=1,
    )
    assert "circular/trivial" in result
    assert state.checks["c1"][0].circular is True


@pytest.mark.asyncio
async def test_run_agent_check_flags_session_isolation_discrepancy(fake_env, session_exec, isolated_exec):
    config = st.SelfTestConfig(state_dir=str(fake_env.root / ".agent" / "self_test"))
    (fake_env.root / "answer.txt").write_text("42")
    (fake_env.root / "solution.py").write_text(
        "import pathlib\nprint(pathlib.Path('answer.txt').read_text())\n"
    )
    state = st.SelfTestState()
    criteria = {"c1": st.Criterion(id="c1", description="prints 42")}

    result = await st.run_agent_check(
        "c1",
        'test "$(python3 solution.py)" = "42"',
        criteria=criteria,
        deliverables=["solution.py"],  # answer.txt deliberately NOT declared
        state=state,
        config=config,
        session_exec=session_exec,
        isolated_exec=isolated_exec,
        persistent_cwd=str(fake_env.root),
        step=1,
    )
    assert "FAIL (isolated)" in result
    assert "passed in your session but FAILED in isolation" in result


# ---------------------------------------------------------------------------
# Mechanism 4 — the gate
# ---------------------------------------------------------------------------


def test_evaluate_gate_no_criteria_rejected():
    config = st.SelfTestConfig(min_criteria=2, max_gate_rejections=3)
    gate = st.evaluate_gate({}, st.SelfTestState(), config)
    assert gate.allowed is False
    assert not gate.forced
    assert "no acceptance criteria declared" in gate.reasons[0]


def test_evaluate_gate_insufficient_coverage_rejected():
    config = st.SelfTestConfig(min_criteria=2, max_gate_rejections=3)
    criteria = {"c1": st.Criterion("c1", "d1"), "c2": st.Criterion("c2", "d2")}
    state = st.SelfTestState()
    state.checks["c1"] = [st.CheckRecord("c1", "pytest -q", True, True, None, False, None, [], step=1)]
    gate = st.evaluate_gate(criteria, state, config)
    assert gate.allowed is False
    assert "unmet: c2" in gate.reasons[-1]


def test_evaluate_gate_circular_pass_does_not_count():
    config = st.SelfTestConfig(min_criteria=1, max_gate_rejections=3)
    criteria = {"c1": st.Criterion("c1", "d1")}
    state = st.SelfTestState()
    state.checks["c1"] = [st.CheckRecord("c1", "true", True, True, None, circular=True, circular_reason="x", missing_deliverables=[], step=1)]
    gate = st.evaluate_gate(criteria, state, config)
    assert gate.allowed is False


def test_evaluate_gate_accepts_once_covered():
    config = st.SelfTestConfig(min_criteria=1, max_gate_rejections=3)
    criteria = {"c1": st.Criterion("c1", "d1")}
    state = st.SelfTestState()
    state.checks["c1"] = [st.CheckRecord("c1", "pytest -q", True, True, None, False, None, [], step=1)]
    gate = st.evaluate_gate(criteria, state, config)
    assert gate.allowed is True
    assert gate.reasons == []
    assert gate.forced is False


def test_evaluate_gate_forced_after_max_rejections():
    config = st.SelfTestConfig(min_criteria=2, max_gate_rejections=2)
    state = st.SelfTestState(gate_rejections=2)
    gate = st.evaluate_gate({}, state, config)
    assert gate.allowed is True
    assert gate.forced is True
    assert "rejected submit attempts" in gate.reasons[0]


def test_gate_nudge_lists_reasons():
    gate = st.GateResult(False, ["no acceptance criteria declared"])
    nudge = st.gate_nudge(gate)
    assert "Submit rejected" in nudge
    assert "no acceptance criteria declared" in nudge
    assert "agent-check" in nudge
