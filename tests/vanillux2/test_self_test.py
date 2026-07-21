"""Tests for self_test.py — acceptance criteria + the isolated check gate.

See tests/vanillux2/conftest.py for the ``ops``/``isolated_exec_fn``/``workdir``
fixtures: ``ops.exec_fn`` runs real bash with a fixed cwd (the "session"),
``isolated_exec_fn`` runs real bash with a per-call cwd and no shared state
(the "isolation" side) — our test "container" is the local filesystem, so
these exercise the real commands/paths self_test.py generates end to end.

Criteria live in the harness-owned criteria.json (see self_test.py's module
docstring); tests write it either through the ``declare_criteria`` tool path
or directly to disk (the bash-convention path — the model writing the file
itself is exactly what production bash-only mode does).
"""

import json

import pytest

import self_test  # noqa: E402 (sys.path set up by conftest)


def _config(workdir, **overrides):
    kwargs = dict(
        min_criteria=2,
        max_gate_rejections=3,
        state_dir=str(workdir / "state"),
    )
    kwargs.update(overrides)
    return self_test.SelfTestConfig(**kwargs)


def _write_criteria(workdir, config, *criteria, deliverables=None):
    """The bash-convention declaration path: the criteria file on disk."""
    (workdir / "state").mkdir(exist_ok=True)
    doc = {
        "criteria": [
            {"id": cid, "description": desc, "how_to_check": "n/a"} for cid, desc in criteria
        ],
        "deliverables": deliverables or [],
    }
    (workdir / "state" / "criteria.json").write_text(json.dumps(doc))


def _criteria(*ids):
    return {cid: self_test.Criterion(cid, f"description of {cid}") for cid in ids}


def _record(cid, **overrides):
    kwargs = dict(
        criterion_id=cid,
        command="test 1 = 1",
        expected=None,
        session_pass=True,
        isolated_pass=True,
        first_failure_line=None,
        circular=False,
        circular_reason=None,
        trivial=False,
        missing_deliverables=[],
        step=1,
    )
    kwargs.update(overrides)
    return self_test.CheckRecord(**kwargs)


async def _run_check(ops, isolated_exec_fn, config, state, cid, command, expected=None, step=1):
    return await self_test.run_check(
        cid,
        command,
        expected,
        state=state,
        config=config,
        ops=ops,
        isolated_exec=isolated_exec_fn,
        step=step,
    )


# ---------------------------------------------------------------------------
# agent-check convention parsing (bash-only contract)
# ---------------------------------------------------------------------------


def test_parse_agent_check_matches_convention():
    assert self_test.parse_agent_check("agent-check c1 -- test -f out.txt") == ("c1", "test -f out.txt")


def test_parse_agent_check_ignores_ordinary_commands():
    assert self_test.parse_agent_check("ls -la && echo done") is None
    assert self_test.parse_agent_check("echo 'agent-check is neat'") is None


def test_parse_agent_check_multiline_command():
    cid, cmd = self_test.parse_agent_check("agent-check c1 -- python3 - <<'EOF'\nassert 1 == 1\nEOF")
    assert cid == "c1"
    assert "assert 1 == 1" in cmd


def test_parse_agent_check_rejects_empty_command():
    assert self_test.parse_agent_check("agent-check c1 --  ") is None


def test_extract_agent_check_lines_finds_embedded_invocation():
    command = "mkdir -p /app/out\nagent-check c1 -- test -f /app/out/result.txt"
    checks, remainder = self_test.extract_agent_check_lines(command)
    assert checks == [("c1", "test -f /app/out/result.txt")]
    assert remainder == "mkdir -p /app/out"


def test_extract_agent_check_lines_finds_multiple_invocations():
    command = (
        "echo setup\n"
        "agent-check c1 -- test -f a.txt\n"
        "agent-check c2 -- grep -q done a.txt\n"
        "echo teardown"
    )
    checks, remainder = self_test.extract_agent_check_lines(command)
    assert checks == [("c1", "test -f a.txt"), ("c2", "grep -q done a.txt")]
    assert remainder == "echo setup\necho teardown"


def test_extract_agent_check_lines_no_match_for_ordinary_multiline_command():
    assert self_test.extract_agent_check_lines("echo one\necho two") is None


def test_extract_agent_check_lines_does_not_swallow_next_line():
    # A check with no command on its own line must NOT absorb the next line's
    # command as its own (the same-line-whitespace-only regex guards this).
    command = "agent-check c1 --\necho innocent"
    assert self_test.extract_agent_check_lines(command) is None


def test_looks_like_malformed_agent_check_true_for_botched_attempts():
    assert self_test.looks_like_malformed_agent_check("agent-check c1 test -f x") is True
    assert self_test.looks_like_malformed_agent_check("agent-check -- test -f x") is True
    assert self_test.looks_like_malformed_agent_check("agent-check c1 --") is True


def test_looks_like_malformed_agent_check_false_for_valid_forms():
    assert self_test.looks_like_malformed_agent_check("agent-check c1 -- test -f x") is False
    assert (
        self_test.looks_like_malformed_agent_check("echo setup\nagent-check c1 -- test -f x")
        is False
    )


def test_looks_like_malformed_agent_check_false_for_unrelated_commands():
    assert self_test.looks_like_malformed_agent_check("echo agent-check is a convention") is False
    assert self_test.looks_like_malformed_agent_check("ls -la") is False


def test_looks_like_existence_probe_true_for_which_type_command_v():
    assert self_test.looks_like_existence_probe("which agent-check") is True
    assert self_test.looks_like_existence_probe("type agent-check") is True
    assert self_test.looks_like_existence_probe("command -v agent-check") is True


def test_looks_like_existence_probe_false_for_unrelated_commands():
    assert self_test.looks_like_existence_probe("which python3") is False
    assert self_test.looks_like_existence_probe("agent-check c1 -- test -f x") is False


# ---------------------------------------------------------------------------
# Mechanism 1 — the criteria file
# ---------------------------------------------------------------------------


def test_parse_criteria_doc_valid():
    criteria, deliverables = self_test.parse_criteria_doc(
        json.dumps(
            {
                "criteria": [{"id": "c1", "description": "d1", "how_to_check": "h1"}],
                "deliverables": ["solve.py"],
            }
        )
    )
    assert set(criteria) == {"c1"}
    assert criteria["c1"].how_to_check == "h1"
    assert deliverables == ["solve.py"]


@pytest.mark.parametrize("raw", ["", "not json", "[1, 2]", '{"criteria": "nope"}', "null"])
def test_parse_criteria_doc_tolerates_malformed_input(raw):
    criteria, deliverables = self_test.parse_criteria_doc(raw)
    assert criteria == {}
    assert deliverables == []


def test_parse_criteria_doc_skips_incomplete_entries():
    criteria, _ = self_test.parse_criteria_doc(
        json.dumps({"criteria": [{"id": "", "description": "no id"}, {"id": "ok", "description": "d"}, "junk"]})
    )
    assert set(criteria) == {"ok"}


async def test_load_criteria_missing_file_is_empty(ops, workdir):
    criteria, deliverables = await self_test.load_criteria(ops, _config(workdir))
    assert criteria == {}
    assert deliverables == []


async def test_declare_criteria_tool_persists_and_acks_compactly(ops, workdir):
    config = _config(workdir)
    state = self_test.SelfTestState()
    result = await self_test.declare_criteria(
        {
            "criteria": [
                {"id": "c1", "description": "d1", "how_to_check": "h1"},
                {"id": "c2", "description": "d2", "how_to_check": "h2"},
            ],
            "deliverables": ["solve.py"],
        },
        ops=ops,
        config=config,
        state=state,
        step=1,
    )
    assert "declared 2 criteria" in result
    assert len(result) < 300  # compact ack, not the full declaration
    criteria, deliverables = await self_test.load_criteria(ops, config)
    assert set(criteria) == {"c1", "c2"}
    assert deliverables == ["solve.py"]


async def test_declare_criteria_tool_merges_on_redeclare(ops, workdir):
    config = _config(workdir)
    state = self_test.SelfTestState()
    await self_test.declare_criteria(
        {"criteria": [{"id": "c1", "description": "d1", "how_to_check": "h"}]},
        ops=ops, config=config, state=state, step=1,
    )
    await self_test.declare_criteria(
        {"criteria": [{"id": "c2", "description": "d2", "how_to_check": "h"}], "deliverables": ["a.py"]},
        ops=ops, config=config, state=state, step=2,
    )
    criteria, deliverables = await self_test.load_criteria(ops, config)
    assert set(criteria) == {"c1", "c2"}
    assert deliverables == ["a.py"]


async def test_declare_criteria_rejects_empty_list(ops, workdir):
    result = await self_test.declare_criteria(
        {"criteria": []}, ops=ops, config=_config(workdir), state=self_test.SelfTestState(), step=1
    )
    assert "no valid criteria" in result


# ---------------------------------------------------------------------------
# Mechanism 5 — anti-circularity heuristics
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("command", ["true", "  :  ", "exit 0"])
def test_circularity_flags_trivial_commands(command):
    assert self_test.circularity_flags(command)


@pytest.mark.parametrize(
    "command",
    [
        "diff expected.txt actual.txt",
        "grep -q 'all good' out.log",
        "python3 -m pytest -q tests/",
        'python3 -c "assert solve(2) == 4"',
        '[ "$(./solve input)" = "42" ]',
        "cmp -s a.bin b.bin",
    ],
)
def test_circularity_flags_real_assertions_not_flagged(command):
    assert self_test.circularity_flags(command) == []


def test_circularity_flags_self_diff_is_flagged():
    flags = self_test.circularity_flags("diff <(python3 prog.py) <(python3 prog.py)")
    assert any("independent oracle" in f for f in flags)


def test_circularity_flags_self_eq_is_flagged():
    flags = self_test.circularity_flags('[ "$(./prog.sh)" = "$(./prog.sh)" ]')
    assert any("independent oracle" in f for f in flags)


def test_circularity_flags_no_assertion_hint():
    flags = self_test.circularity_flags("python3 script.py")
    assert any("no recognizable assertion" in f for f in flags)


def test_circularity_flags_print_comparison_with_no_exit_mechanism():
    # The exit-code-blind pattern that slipped through a live A/B: the
    # comparison happens but nothing turns a mismatch into a non-zero exit.
    flags = self_test.circularity_flags('python3 -c "print(solve(2) == 4)"')
    assert any("`==`/`!=`" in f for f in flags)


def test_circularity_flags_comparison_with_assert_not_flagged():
    assert self_test.circularity_flags('python3 -c "assert solve(2) == 4"') == []


def test_circularity_flags_trailing_echo_fallback():
    flags = self_test.circularity_flags("python3 t.py && echo PASS || echo FAIL")
    assert any("echo" in f and "always" in f for f in flags)


def test_circularity_flags_trailing_true_fallback():
    flags = self_test.circularity_flags("pytest -q || true")
    assert any("`|| true`" in f for f in flags)


def test_circularity_flags_real_test_without_fallback_not_flagged():
    assert self_test.circularity_flags("pytest -q tests/") == []


def test_circularity_flags_trailing_echo_exit_code():
    flags = self_test.circularity_flags("python3 t.py; echo $?")
    assert any("echo $?" in f or "exit code" in f for f in flags)


def test_circularity_flags_except_pass():
    flags = self_test.circularity_flags('python3 -c "\ntry:\n    check()\nexcept Exception: pass"')
    assert any("except" in f for f in flags)


def test_circularity_flags_set_plus_e():
    flags = self_test.circularity_flags("set +e\npytest -q\ntest -f out.txt")
    assert any("set +e" in f for f in flags)


@pytest.mark.parametrize(
    "command",
    [
        'python3 -c "import sys; sys.exit(0 if solve(2) == 4 else 1)"',
        'python3 -c "\nif solve(2) != 4:\n    raise SystemExit(1)"',
    ],
)
def test_circularity_flags_sys_exit_raise_count_as_real_assertion(command):
    assert self_test.circularity_flags(command) == []


def test_circularity_flags_fixed_expected_is_a_real_assertion():
    # With a static expected value the harness itself does the comparison, so
    # "no recognizable assertion" / always-zero-tail flags must not fire.
    assert self_test.circularity_flags("python3 script.py", "42") == []
    assert self_test.circularity_flags("python3 script.py || true", "42") == []


def test_circularity_flags_runtime_computed_expected():
    flags = self_test.circularity_flags("python3 script.py", "$(python3 script.py)")
    assert any("FIXED literal" in f for f in flags)


def test_circularity_flags_self_comparison_still_flagged_with_expected():
    flags = self_test.circularity_flags("diff <(python3 p.py) <(python3 p.py)", "ok")
    assert any("independent oracle" in f for f in flags)


def test_extract_script_reference_python_invocation():
    assert self_test.extract_script_reference("python3 /tmp/verify.py --strict") == "/tmp/verify.py"


def test_extract_script_reference_bare_path():
    assert self_test.extract_script_reference("./check.sh") == "./check.sh"


def test_extract_script_reference_none_for_inline_command():
    assert self_test.extract_script_reference('python3 -c "assert 1 == 1"') is None


@pytest.mark.parametrize(
    "command,expected_trivial",
    [
        ("test -f out.txt", True),
        ("[ -e out.txt ]", True),
        ("ls out.txt", True),
        ("test -f out.txt && grep -q done out.txt", False),
        ("grep -q done out.txt", False),
    ],
)
def test_is_trivial_existence_check(command, expected_trivial):
    assert self_test.is_trivial_existence_check(command) is expected_trivial


def test_classify_check_buckets():
    assert self_test.classify_check("true", ["a flag"], False) == "circular"
    assert self_test.classify_check("test -f out.txt", [], True) == "trivial_existence"
    assert self_test.classify_check("python3 verify.py", [], False) == "external_script"
    assert self_test.classify_check("pytest -q", [], False) == "behavioral"


# ---------------------------------------------------------------------------
# Mechanism 5 — weakening + criteria-edit audit
# ---------------------------------------------------------------------------


def test_weakening_warning_fires_after_a_failure_with_a_changed_check():
    state = self_test.SelfTestState()
    state.checks["c1"] = [_record("c1", command="python3 x.py", expected="42", session_pass=False, isolated_pass=False)]
    warning = self_test.weakening_warning("c1", "python3 x.py", "", state)
    assert warning is not None and "c1" in warning


def test_weakening_warning_fires_when_only_expected_changes():
    state = self_test.SelfTestState()
    state.checks["c1"] = [_record("c1", command="python3 x.py", expected="42", isolated_pass=False)]
    assert self_test.weakening_warning("c1", "python3 x.py", "43", state) is not None


def test_no_weakening_warning_when_unchanged_after_failure():
    state = self_test.SelfTestState()
    state.checks["c1"] = [_record("c1", command="python3 x.py", expected="42", isolated_pass=False)]
    assert self_test.weakening_warning("c1", "python3 x.py", "42", state) is None


def test_no_weakening_warning_after_a_pass():
    state = self_test.SelfTestState()
    state.checks["c1"] = [_record("c1", command="python3 x.py", expected="42", isolated_pass=True)]
    assert self_test.weakening_warning("c1", "python3 x.py", "anything else", state) is None


def test_no_weakening_warning_on_first_check():
    assert self_test.weakening_warning("c1", "cmd", "42", self_test.SelfTestState()) is None


def test_criteria_edit_warning_on_removal_after_failed_check():
    state = self_test.SelfTestState()
    state.criteria_snapshot = {"c1": "d1", "c2": "d2"}
    state.checks["c1"] = [_record("c1", isolated_pass=False)]
    warnings = self_test.criteria_edit_warnings(state, _criteria("c2"), step=3)
    assert any("REMOVED" in w and "c1" in w for w in warnings)
    assert any(e["event"] == "criterion_removed" and e["id"] == "c1" for e in state.audit)
    assert set(state.criteria_snapshot) == {"c2"}  # snapshot refreshed


def test_criteria_removal_without_failure_audits_but_does_not_warn():
    state = self_test.SelfTestState()
    state.criteria_snapshot = {"c1": "d1", "c2": "d2"}
    warnings = self_test.criteria_edit_warnings(state, _criteria("c2"), step=3)
    assert warnings == []
    assert any(e["event"] == "criterion_removed" for e in state.audit)


def test_criteria_edit_warning_on_reword_after_failed_check():
    state = self_test.SelfTestState()
    state.criteria_snapshot = {"c1": "must reject bad input"}
    state.checks["c1"] = [_record("c1", isolated_pass=False)]
    reworded = {"c1": self_test.Criterion("c1", "should mostly reject bad input")}
    warnings = self_test.criteria_edit_warnings(state, reworded, step=3)
    assert any("reworded" in w for w in warnings)


async def test_declare_criteria_surfaces_reword_after_failure(ops, workdir):
    config = _config(workdir)
    state = self_test.SelfTestState()
    await self_test.declare_criteria(
        {"criteria": [{"id": "c1", "description": "rejects bad input", "how_to_check": "h"}]},
        ops=ops, config=config, state=state, step=1,
    )
    state.checks["c1"] = [_record("c1", isolated_pass=False)]
    result = await self_test.declare_criteria(
        {"criteria": [{"id": "c1", "description": "accepts most input", "how_to_check": "h"}]},
        ops=ops, config=config, state=state, step=2,
    )
    assert "reworded" in result


# ---------------------------------------------------------------------------
# Mechanism 3 — isolation catches leftover state / a neutered deliverable
# (deliverable requirement #2, the "neutered filter" fixture).
# ---------------------------------------------------------------------------


async def test_neutered_filter_with_leftover_backup_fails_in_isolation(ops, isolated_exec_fn, workdir):
    """Reproduces the failure analysis's "verified against a filter it had
    itself neutered" case: the *declared* deliverable (filter.py) has been
    weakened to accept everything, but a known-good backup was left behind
    in scratch/ from an earlier iteration and never declared. A check that
    happens to run against the backup looks like it passes — but the backup
    isn't part of what gets submitted, so isolation (which copies only
    declared deliverables) must fail it.
    """
    (workdir / "scratch").mkdir()
    (workdir / "scratch" / "filter.py").write_text("def is_valid(x):\n    return x != 'bad'\n")
    (workdir / "filter.py").write_text("def is_valid(x):\n    return True\n")  # neutered

    config = _config(workdir)
    state = self_test.SelfTestState()
    _write_criteria(
        workdir, config, ("rejects_bad_input", "filter.is_valid('bad') is False"),
        deliverables=["filter.py"],
    )

    command = "cd scratch && python3 -c \"import filter; print(filter.is_valid('bad'))\""
    result = await _run_check(
        ops, isolated_exec_fn, config, state, "rejects_bad_input", command, expected="False"
    )

    record = state.checks["rejects_bad_input"][-1]
    assert record.session_pass is True  # scratch/filter.py (leftover, correct) answers False
    assert record.isolated_pass is False  # scratch/ was never declared, so it isn't copied
    assert "FAILED in isolation" in result
    criteria, _ = await self_test.load_criteria(ops, config)
    assert self_test.coverage(criteria, state)["rejects_bad_input"] is False


async def test_declared_deliverable_itself_broken_fails_in_both(ops, isolated_exec_fn, workdir):
    """The simpler half of the same failure mode: no leftover backup at all,
    just a genuinely broken/neutered deliverable. Isolation can't be gamed
    by session state here because there is none to exploit — both runs
    honestly fail.
    """
    (workdir / "filter.py").write_text("def is_valid(x):\n    return True\n")

    config = _config(workdir)
    state = self_test.SelfTestState()
    _write_criteria(
        workdir, config, ("rejects_bad_input", "filter.is_valid('bad') is False"),
        deliverables=["filter.py"],
    )

    command = "python3 -c \"import filter; print(filter.is_valid('bad'))\""
    await _run_check(ops, isolated_exec_fn, config, state, "rejects_bad_input", command, expected="False")
    record = state.checks["rejects_bad_input"][-1]
    assert record.session_pass is False
    assert record.isolated_pass is False


async def test_leftover_undeclared_output_file_fails_in_isolation(ops, isolated_exec_fn, workdir):
    # The other leftover-state shape: the check reads an output file that
    # exists in the session only as a stale artifact of an earlier run, and
    # was never declared as a deliverable.
    (workdir / "out.txt").write_text("42\n")
    config = _config(workdir)
    state = self_test.SelfTestState()
    _write_criteria(workdir, config, ("right_answer", "out.txt holds 42"), deliverables=["solve.py"])
    (workdir / "solve.py").write_text("print(42)\n")

    result = await _run_check(
        ops, isolated_exec_fn, config, state, "right_answer", 'test "$(cat out.txt)" = "42"'
    )
    record = state.checks["right_answer"][-1]
    assert record.session_pass is True
    assert record.isolated_pass is False
    assert "FAILED in isolation" in result


async def test_missing_deliverable_is_reported(ops, isolated_exec_fn, workdir):
    config = _config(workdir)
    state = self_test.SelfTestState()
    _write_criteria(workdir, config, ("c1", "d"), deliverables=["does_not_exist.py"])
    result = await _run_check(ops, isolated_exec_fn, config, state, "c1", "test 1 = 1")
    assert "not found on disk" in result
    assert state.checks["c1"][-1].missing_deliverables == ["does_not_exist.py"]


async def test_directory_deliverable_is_mirrored(ops, isolated_exec_fn, workdir):
    # cp -a (not per-file byte copies) so a directory deliverable works.
    pkg = workdir / "pkg"
    pkg.mkdir()
    (pkg / "mod.py").write_text("VALUE = 7\n")
    config = _config(workdir)
    state = self_test.SelfTestState()
    _write_criteria(workdir, config, ("pkg_value", "pkg.mod.VALUE == 7"), deliverables=["pkg"])
    result = await _run_check(
        ops, isolated_exec_fn, config, state, "pkg_value",
        'python3 -c "from pkg import mod; assert mod.VALUE == 7"',
    )
    assert state.checks["pkg_value"][-1].isolated_pass is True
    assert "PASS" in result


# ---------------------------------------------------------------------------
# Mechanism 2 — compact result only, never full output (deliverable
# requirement #4), and run_check end-to-end behavior.
# ---------------------------------------------------------------------------


async def test_run_check_result_is_compact_not_full_output(ops, isolated_exec_fn, workdir):
    config = _config(workdir)
    state = self_test.SelfTestState()
    _write_criteria(workdir, config, ("c1", "d"))

    noisy = 'for i in $(seq 1 500); do echo "line $i FAILURE_MARKER_$i"; done; exit 1'
    result = await _run_check(ops, isolated_exec_fn, config, state, "c1", noisy, expected="nope")

    assert len(result) < 1000  # nowhere near the ~500-line/~10KB raw output
    assert "line 1 FAILURE_MARKER_1" in result  # first failing line kept
    assert "line 500" not in result  # rest is not


async def test_run_check_reports_first_failure_line(ops, isolated_exec_fn, workdir):
    config = _config(workdir)
    state = self_test.SelfTestState()
    _write_criteria(workdir, config, ("c1", "d"))
    result = await _run_check(ops, isolated_exec_fn, config, state, "c1", "echo boom && exit 1")
    assert "boom" in result
    assert "first failing line" in result


async def test_run_check_unknown_criterion_lists_declared(ops, isolated_exec_fn, workdir):
    config = _config(workdir)
    state = self_test.SelfTestState()
    _write_criteria(workdir, config, ("c1", "d"))
    result = await _run_check(ops, isolated_exec_fn, config, state, "nope", "test 1 = 1")
    assert "unknown criterion_id" in result
    assert "c1" in result
    assert not state.checks


async def test_run_check_flags_circular_and_excludes_from_coverage(ops, isolated_exec_fn, workdir):
    config = _config(workdir)
    state = self_test.SelfTestState()
    _write_criteria(workdir, config, ("c1", "d"))

    result = await _run_check(ops, isolated_exec_fn, config, state, "c1", "true")

    record = state.checks["c1"][-1]
    assert record.isolated_pass is True  # `true` genuinely exits 0...
    assert record.circular is True  # ...but it's a no-op, so it doesn't count
    assert record.check_category == "circular"
    assert "circular" in result.lower()
    criteria, _ = await self_test.load_criteria(ops, config)
    assert self_test.coverage(criteria, state)["c1"] is False


async def test_run_check_flags_trivial_existence_and_excludes_from_coverage(
    ops, isolated_exec_fn, workdir
):
    (workdir / "out.txt").write_text("data\n")
    config = _config(workdir)
    state = self_test.SelfTestState()
    _write_criteria(workdir, config, ("c1", "output exists"), deliverables=["out.txt"])

    result = await _run_check(ops, isolated_exec_fn, config, state, "c1", "test -f out.txt")

    record = state.checks["c1"][-1]
    assert record.isolated_pass is True
    assert record.circular is False
    assert record.trivial is True
    assert record.check_category == "trivial_existence"
    assert "existence-only" in result
    criteria, _ = await self_test.load_criteria(ops, config)
    assert self_test.coverage(criteria, state)["c1"] is False


async def test_run_check_inspects_external_script_content(ops, isolated_exec_fn, workdir):
    # A check that delegates to a script hides its (lack of) assertions from
    # the command-level heuristic — the script's content is what gets vetted.
    (workdir / "verify.py").write_text("result = compute() == 42\nprint(result)\n")
    config = _config(workdir)
    state = self_test.SelfTestState()
    _write_criteria(workdir, config, ("c1", "d"))
    result = await _run_check(ops, isolated_exec_fn, config, state, "c1", "python3 verify.py")
    assert "external script" in result
    assert state.checks["c1"][-1].circular is True


async def test_run_check_external_script_with_real_assertions_not_flagged(
    ops, isolated_exec_fn, workdir
):
    (workdir / "verify.py").write_text("assert compute() == 42\n")
    config = _config(workdir)
    state = self_test.SelfTestState()
    _write_criteria(workdir, config, ("c1", "d"))
    await _run_check(ops, isolated_exec_fn, config, state, "c1", "python3 verify.py")
    record = state.checks["c1"][-1]
    assert record.circular is False
    assert record.check_category == "external_script"


async def test_run_check_surfaces_weakening_warning_end_to_end(ops, isolated_exec_fn, workdir):
    config = _config(workdir)
    state = self_test.SelfTestState()
    _write_criteria(workdir, config, ("c1", "d"))

    await _run_check(ops, isolated_exec_fn, config, state, "c1", "echo wrong", expected="right")
    result = await _run_check(
        ops, isolated_exec_fn, config, state, "c1", "echo wrong", expected="wrong", step=2
    )
    assert "changed after its last isolated run FAILED" in result


async def test_run_check_surfaces_criteria_file_edits(ops, isolated_exec_fn, workdir):
    # Bash-convention path: the model rewrites criteria.json dropping a
    # criterion whose check failed — the next check surfaces the removal.
    config = _config(workdir)
    state = self_test.SelfTestState()
    _write_criteria(workdir, config, ("c1", "d1"), ("c2", "d2"))
    await _run_check(ops, isolated_exec_fn, config, state, "c1", "test 1 = 2")  # fails
    _write_criteria(workdir, config, ("c2", "d2"))  # c1 quietly dropped
    result = await _run_check(ops, isolated_exec_fn, config, state, "c2", "test 1 = 1", step=2)
    assert "REMOVED" in result


async def test_run_check_persists_state_to_disk(ops, isolated_exec_fn, workdir):
    config = _config(workdir)
    state = self_test.SelfTestState()
    _write_criteria(workdir, config, ("c1", "d"))
    await _run_check(ops, isolated_exec_fn, config, state, "c1", 'test "$(echo hi)" = "hi"')

    checks_doc = json.loads((workdir / "state" / "checks.json").read_text())
    assert checks_doc["checks"]["c1"][0]["isolated_pass"] is True
    assert "audit" in checks_doc


# ---------------------------------------------------------------------------
# dispatch(): never raises, always returns model-facing text
# ---------------------------------------------------------------------------


async def _dispatch(ops, isolated_exec_fn, config, state, name, args):
    return await self_test.dispatch(
        name, args, state=state, ops=ops, isolated_exec=isolated_exec_fn, config=config, step=1
    )


async def test_dispatch_declare_and_run_check(ops, isolated_exec_fn, workdir):
    config = _config(workdir)
    state = self_test.SelfTestState()
    result = await _dispatch(
        ops, isolated_exec_fn, config, state, "declare_criteria",
        {"criteria": [{"id": "c1", "description": "d", "how_to_check": "h"}]},
    )
    assert "declared 1 criteria" in result
    result = await _dispatch(
        ops, isolated_exec_fn, config, state, "run_check",
        {"criterion_id": "c1", "command": "echo hi", "expected": "hi"},
    )
    assert "PASS" in result
    assert state.checks["c1"][-1].isolated_pass is True


async def test_dispatch_unknown_tool_name(ops, isolated_exec_fn, workdir):
    result = await _dispatch(
        ops, isolated_exec_fn, _config(workdir), self_test.SelfTestState(), "frobnicate", {}
    )
    assert "unknown self-test tool" in result


async def test_dispatch_run_check_missing_command(ops, isolated_exec_fn, workdir):
    config = _config(workdir)
    state = self_test.SelfTestState()
    _write_criteria(workdir, config, ("c1", "d"))
    result = await _dispatch(
        ops, isolated_exec_fn, config, state, "run_check", {"criterion_id": "c1", "command": ""}
    )
    assert "command is required" in result


async def test_dispatch_never_raises(workdir, isolated_exec_fn):
    # A broken ops (exec explodes) must surface as model-facing text, not a
    # crash of the agent loop.
    from container_ops import ContainerOps

    async def broken_exec(command):
        raise RuntimeError("container is gone")

    async def broken_upload(content, path):
        raise RuntimeError("container is gone")

    async def broken_download(path):
        raise RuntimeError("container is gone")

    broken_ops = ContainerOps(broken_exec, broken_upload, broken_download)
    result = await self_test.dispatch(
        "run_check",
        {"criterion_id": "c1", "command": "true"},
        state=self_test.SelfTestState(),
        ops=broken_ops,
        isolated_exec=isolated_exec_fn,
        config=_config(workdir),
        step=1,
    )
    assert "error" in result.lower()


# ---------------------------------------------------------------------------
# Mechanism 4 — the gate (deliverable requirement #1: rejected without a
# passing isolated check, accepted once coverage is complete)
# ---------------------------------------------------------------------------


def test_gate_rejects_with_no_criteria_declared(workdir):
    gate = self_test.evaluate_gate({}, self_test.SelfTestState(), _config(workdir))
    assert gate.allowed is False
    assert "no acceptance criteria" in gate.reasons[0]


def test_gate_rejects_uncovered_criterion_even_past_min(workdir):
    # v3 tightening: EVERY declared criterion must be covered, not just
    # min_criteria of them — otherwise declaring many and checking few games
    # the gate.
    state = self_test.SelfTestState()
    state.checks["a"] = [_record("a")]
    state.checks["b"] = [_record("b")]
    gate = self_test.evaluate_gate(_criteria("a", "b", "c"), state, _config(workdir, min_criteria=2))
    assert gate.allowed is False
    assert any("c" in r for r in gate.reasons)


def test_gate_rejects_when_fewer_than_min_criteria_declared(workdir):
    state = self_test.SelfTestState()
    state.checks["a"] = [_record("a")]
    gate = self_test.evaluate_gate(_criteria("a"), state, _config(workdir, min_criteria=2))
    assert gate.allowed is False


def test_gate_circular_pass_does_not_count(workdir):
    state = self_test.SelfTestState()
    state.checks["a"] = [_record("a", circular=True)]
    state.checks["b"] = [_record("b")]
    gate = self_test.evaluate_gate(_criteria("a", "b"), state, _config(workdir))
    assert gate.allowed is False
    assert any("a" in r for r in gate.reasons)


def test_gate_trivial_pass_does_not_count(workdir):
    state = self_test.SelfTestState()
    state.checks["a"] = [_record("a", trivial=True)]
    state.checks["b"] = [_record("b")]
    gate = self_test.evaluate_gate(_criteria("a", "b"), state, _config(workdir))
    assert gate.allowed is False


def test_gate_failed_isolated_check_does_not_count(workdir):
    state = self_test.SelfTestState()
    state.checks["a"] = [_record("a", isolated_pass=False, session_pass=True)]
    state.checks["b"] = [_record("b")]
    gate = self_test.evaluate_gate(_criteria("a", "b"), state, _config(workdir))
    assert gate.allowed is False


def test_gate_accepts_once_all_covered(workdir):
    state = self_test.SelfTestState()
    state.checks["a"] = [_record("a", isolated_pass=False), _record("a")]  # later retry passed
    state.checks["b"] = [_record("b")]
    gate = self_test.evaluate_gate(_criteria("a", "b"), state, _config(workdir))
    assert gate.allowed is True
    assert gate.forced is False


def test_gate_rejections_are_bounded_even_with_no_criteria(workdir):
    # The rejection bound is a universal safety valve: it must eventually
    # force a submit even if the model never declared criteria at all (the
    # "never even tried" case), not just the "insufficient coverage" case —
    # otherwise the gate could turn a wrong-but-submitted run into a
    # step-exhausted one (see self_test.py's module docstring).
    state = self_test.SelfTestState()
    state.gate_rejections = 2
    gate = self_test.evaluate_gate({}, state, _config(workdir, max_gate_rejections=2))
    assert gate.allowed is True
    assert gate.forced is True


def test_gate_rejections_bounded_with_partial_coverage(workdir):
    state = self_test.SelfTestState()
    state.checks["a"] = [_record("a")]
    state.gate_rejections = 1
    gate = self_test.evaluate_gate(
        _criteria("a", "b"), state, _config(workdir, min_criteria=2, max_gate_rejections=1)
    )
    assert gate.allowed is True
    assert gate.forced is True


def test_gate_nudge_is_compact_and_actionable(workdir):
    gate = self_test.GateResult(False, ["checks failed or missing in isolation for: a, b"])
    config = _config(workdir)
    tool_nudge = self_test.gate_nudge(gate, config, tools_enabled=True)
    assert "Submit rejected" in tool_nudge
    assert "declare_criteria" in tool_nudge
    assert len(tool_nudge) < 400

    bash_nudge = self_test.gate_nudge(gate, config, tools_enabled=False)
    assert "agent-check" in bash_nudge
    assert config.criteria_path in bash_nudge


async def test_gate_end_to_end_reject_then_accept(ops, isolated_exec_fn, workdir):
    config = _config(workdir, min_criteria=2)
    state = self_test.SelfTestState()
    _write_criteria(workdir, config, ("outputs_hello", "prints hello"), ("outputs_world", "prints world"))

    criteria, _ = await self_test.load_criteria(ops, config)
    gate = self_test.evaluate_gate(criteria, state, config)
    assert gate.allowed is False  # no checks run yet

    await _run_check(ops, isolated_exec_fn, config, state, "outputs_hello", "echo hello", expected="hello")
    gate = self_test.evaluate_gate(criteria, state, config)
    assert gate.allowed is False  # only 1/2 covered
    assert "outputs_world" in gate.reasons[0]

    await _run_check(
        ops, isolated_exec_fn, config, state, "outputs_world", "echo world", expected="world", step=2
    )
    gate = self_test.evaluate_gate(criteria, state, config)
    assert gate.allowed is True
    assert gate.forced is False
