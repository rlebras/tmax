"""Tests for self_test.py — the acceptance-criteria + isolated check gate.

See tests/vanillux2/conftest.py for the ``ops``/``isolated_exec_fn``/``workdir``
fixtures: ``ops.exec_fn`` runs real bash with a fixed cwd (the "session"),
``isolated_exec_fn`` runs real bash with a per-call cwd and no shared state
(the "isolation" side) — our test "container" is the local filesystem, so
these exercise the real commands/paths self_test.py generates end to end.
"""

import self_test  # noqa: E402 (sys.path set up by conftest)


def _config(workdir, **overrides):
    kwargs = dict(
        min_criteria=2,
        max_gate_rejections=3,
        state_dir=str(workdir / "state"),
        isolation_root=str(workdir / "isolated"),
    )
    kwargs.update(overrides)
    return self_test.SelfTestConfig(**kwargs)


def _declare(state, *criteria, deliverables=None):
    self_test._declare_criteria(
        {
            "criteria": [
                {"id": cid, "description": desc, "how_to_check": "n/a"} for cid, desc in criteria
            ],
            "deliverables": deliverables or [],
        },
        state,
    )


# ---------------------------------------------------------------------------
# declare_criteria
# ---------------------------------------------------------------------------


def test_declare_criteria_basic():
    state = self_test.SelfTestState()
    result = self_test._declare_criteria(
        {
            "criteria": [
                {"id": "c1", "description": "d1", "how_to_check": "h1"},
                {"id": "c2", "description": "d2", "how_to_check": "h2"},
            ],
            "deliverables": ["solution.py"],
        },
        state,
    )
    assert "declared 2 criteria" in result
    assert set(state.criteria) == {"c1", "c2"}
    assert state.deliverables == ["solution.py"]


def test_declare_criteria_skips_incomplete_entries():
    state = self_test.SelfTestState()
    result = self_test._declare_criteria(
        {"criteria": [{"id": "", "description": "no id", "how_to_check": "x"}, {"id": "ok", "description": "d", "how_to_check": "h"}]},
        state,
    )
    assert set(state.criteria) == {"ok"}
    assert "declared 1 criteria" in result


def test_declare_criteria_empty_list_is_reported_not_silently_dropped():
    state = self_test.SelfTestState()
    result = self_test._declare_criteria({"criteria": []}, state)
    assert "no valid criteria" in result
    assert not state.criteria


# ---------------------------------------------------------------------------
# Mechanism 4 — gate: rejected without an isolated pass, accepted once one
# exists, and dispatch()/run_check() end to end (deliverable requirement #1).
# ---------------------------------------------------------------------------


async def test_gate_end_to_end_reject_then_accept(ops, isolated_exec_fn, workdir):
    state = self_test.SelfTestState()
    config = _config(workdir, min_criteria=2)
    _declare(state, ("outputs_hello", "prints hello"), ("outputs_world", "prints world"))

    # No checks run yet — rejected, and specifically because of missing
    # criteria coverage (not "no criteria declared").
    gate = self_test.evaluate_gate(state, config)
    assert gate.allowed is False
    assert "0/2" in gate.reasons[0]

    await self_test._run_check(
        {"criterion_id": "outputs_hello", "command": "echo hello", "expected": "hello"},
        state, ops, isolated_exec_fn, config, step=1,
    )
    gate = self_test.evaluate_gate(state, config)
    assert gate.allowed is False  # only 1/2 covered
    assert "outputs_world" in gate.reasons[-1]

    await self_test._run_check(
        {"criterion_id": "outputs_world", "command": "echo world", "expected": "world"},
        state, ops, isolated_exec_fn, config, step=2,
    )
    gate = self_test.evaluate_gate(state, config)
    assert gate.allowed is True
    assert gate.forced is False


def test_gate_rejects_with_no_criteria_declared(workdir):
    state = self_test.SelfTestState()
    config = _config(workdir)
    gate = self_test.evaluate_gate(state, config)
    assert gate.allowed is False
    assert "no acceptance criteria" in gate.reasons[0]


def test_gate_rejections_are_bounded_even_with_no_criteria(workdir):
    # The rejection bound is a universal safety valve: it must eventually
    # force a submit even if the model never called declare_criteria at all
    # (the "never even tried" case), not just the "insufficient coverage"
    # case — otherwise the gate could turn a wrong-but-submitted run into a
    # step-exhausted one (see self_test.py's module docstring).
    state = self_test.SelfTestState()
    config = _config(workdir, max_gate_rejections=2)
    state.gate_rejections = 2
    gate = self_test.evaluate_gate(state, config)
    assert gate.allowed is True
    assert gate.forced is True


def test_gate_rejections_bounded_with_partial_coverage(workdir):
    state = self_test.SelfTestState()
    config = _config(workdir, min_criteria=2, max_gate_rejections=1)
    _declare(state, ("a", "d"), ("b", "d"))
    state.checks["a"] = [
        self_test.CheckRecord("a", "true", None, True, True, None, False, None, step=1)
    ]
    state.gate_rejections = 1
    gate = self_test.evaluate_gate(state, config)
    assert gate.allowed is True
    assert gate.forced is True


def test_gate_nudge_is_compact_and_actionable(workdir):
    gate = self_test.GateResult(False, ["only 0/2 required criteria have a passing isolated check"])
    nudge = self_test.gate_nudge(gate)
    assert "Submit rejected" in nudge
    assert len(nudge) < 300


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

    state = self_test.SelfTestState()
    config = _config(workdir)
    _declare(state, ("rejects_bad_input", "filter.is_valid('bad') is False"), deliverables=["filter.py"])

    command = "cd scratch && python3 -c \"import filter; print(filter.is_valid('bad'))\""
    result = await self_test._run_check(
        {"criterion_id": "rejects_bad_input", "command": command, "expected": "False"},
        state, ops, isolated_exec_fn, config, step=1,
    )

    record = state.checks["rejects_bad_input"][-1]
    assert record.session_pass is True  # scratch/filter.py (leftover, correct) answers False
    assert record.isolated_pass is False  # scratch/ was never declared, so it isn't copied
    assert "FAILED in isolation" in result
    assert self_test.coverage(state)["rejects_bad_input"] is False


async def test_declared_deliverable_itself_broken_fails_in_both(ops, isolated_exec_fn, workdir):
    """The simpler half of the same failure mode: no leftover backup at all,
    just a genuinely broken/neutered deliverable. Isolation can't be gamed
    by session state here because there is none to exploit — both runs
    honestly fail.
    """
    (workdir / "filter.py").write_text("def is_valid(x):\n    return True\n")

    state = self_test.SelfTestState()
    config = _config(workdir)
    _declare(state, ("rejects_bad_input", "filter.is_valid('bad') is False"), deliverables=["filter.py"])

    command = "python3 -c \"import filter; print(filter.is_valid('bad'))\""
    await self_test._run_check(
        {"criterion_id": "rejects_bad_input", "command": command, "expected": "False"},
        state, ops, isolated_exec_fn, config, step=1,
    )
    record = state.checks["rejects_bad_input"][-1]
    assert record.session_pass is False
    assert record.isolated_pass is False


async def test_missing_deliverable_is_reported(ops, isolated_exec_fn, workdir):
    state = self_test.SelfTestState()
    config = _config(workdir)
    _declare(state, ("c1", "d"), deliverables=["does_not_exist.py"])
    result = await self_test._run_check(
        {"criterion_id": "c1", "command": "true"},
        state, ops, isolated_exec_fn, config, step=1,
    )
    assert "not found on disk" in result


# ---------------------------------------------------------------------------
# Mechanism 5 — a trivial/circular check is flagged and excluded from
# coverage (deliverable requirement #3).
# ---------------------------------------------------------------------------


async def test_noop_command_is_flagged_circular_and_excluded_from_coverage(ops, isolated_exec_fn, workdir):
    state = self_test.SelfTestState()
    config = _config(workdir)
    _declare(state, ("c1", "d"))

    result = await self_test._run_check(
        {"criterion_id": "c1", "command": "true"},
        state, ops, isolated_exec_fn, config, step=1,
    )

    record = state.checks["c1"][-1]
    assert record.isolated_pass is True  # `true` genuinely exits 0...
    assert record.circular is True  # ...but it's a no-op, so it doesn't count
    assert self_test.coverage(state)["c1"] is False
    assert "circular" in result.lower()


def test_circularity_flags_no_assertion_no_expected():
    flags = self_test._circularity_flags("python3 script.py", None)
    assert flags


def test_circularity_flags_expected_computed_at_check_time():
    flags = self_test._circularity_flags("python3 script.py", "$(python3 script.py)")
    assert any("computed at check time" in f for f in flags)


def test_circularity_flags_self_diff_no_independent_oracle():
    flags = self_test._circularity_flags("diff <(python3 prog.py) <(python3 prog.py)", None)
    assert any("independent oracle" in f for f in flags)


def test_circularity_flags_self_eq_no_independent_oracle():
    flags = self_test._circularity_flags('[ "$(prog.sh)" = "$(prog.sh)" ]', None)
    assert any("independent oracle" in f for f in flags)


def test_circularity_flags_none_for_a_real_assertion():
    flags = self_test._circularity_flags("python3 -m pytest test_solution.py", None)
    assert flags == []


def test_circularity_flags_none_for_fixed_expected():
    flags = self_test._circularity_flags("python3 script.py", "42")
    assert flags == []


# ---------------------------------------------------------------------------
# Mechanism 2 — compact result only, never full output (deliverable
# requirement #4).
# ---------------------------------------------------------------------------


async def test_run_check_result_is_compact_not_full_output(ops, isolated_exec_fn, workdir):
    state = self_test.SelfTestState()
    config = _config(workdir)
    _declare(state, ("c1", "d"))

    noisy = "for i in $(seq 1 500); do echo \"line $i FAILURE_MARKER_$i\"; done; exit 1"
    result = await self_test._run_check(
        {"criterion_id": "c1", "command": noisy, "expected": "nope"},
        state, ops, isolated_exec_fn, config, step=1,
    )

    assert len(result) < 1000  # nowhere near the ~500-line/~10KB raw output
    assert "line 1 FAILURE_MARKER_1" in result  # first failing line kept
    assert "line 500" not in result  # rest is not


async def test_run_check_reports_first_failure_line(ops, isolated_exec_fn, workdir):
    state = self_test.SelfTestState()
    config = _config(workdir)
    _declare(state, ("c1", "d"))
    result = await self_test._run_check(
        {"criterion_id": "c1", "command": "echo boom && exit 1", "expected": "ok"},
        state, ops, isolated_exec_fn, config, step=1,
    )
    assert "boom" in result
    assert "first failing line" in result


# ---------------------------------------------------------------------------
# Mechanism 5 — weakening a failed check triggers a warning (deliverable
# requirement #5), and the gate rejection bound (already covered above).
# ---------------------------------------------------------------------------


def test_weakening_warning_fires_after_a_failure_with_a_changed_check():
    state = self_test.SelfTestState()
    state.checks["c1"] = [
        self_test.CheckRecord("c1", "python3 x.py", "42", False, False, "wrong", False, None, step=1)
    ]
    warning = self_test._weakening_warning("c1", "python3 x.py", "", state)
    assert warning is not None
    assert "c1" in warning


def test_no_weakening_warning_when_unchanged_after_failure():
    state = self_test.SelfTestState()
    state.checks["c1"] = [
        self_test.CheckRecord("c1", "python3 x.py", "42", False, False, "wrong", False, None, step=1)
    ]
    assert self_test._weakening_warning("c1", "python3 x.py", "42", state) is None


def test_no_weakening_warning_after_a_pass():
    state = self_test.SelfTestState()
    state.checks["c1"] = [
        self_test.CheckRecord("c1", "python3 x.py", "42", True, True, None, False, None, step=1)
    ]
    assert self_test._weakening_warning("c1", "python3 x.py", "anything else", state) is None


def test_no_weakening_warning_on_first_check():
    state = self_test.SelfTestState()
    assert self_test._weakening_warning("c1", "cmd", "42", state) is None


async def test_run_check_surfaces_weakening_warning_end_to_end(ops, isolated_exec_fn, workdir):
    state = self_test.SelfTestState()
    config = _config(workdir)
    _declare(state, ("c1", "d"))

    await self_test._run_check(
        {"criterion_id": "c1", "command": "echo wrong", "expected": "right"},
        state, ops, isolated_exec_fn, config, step=1,
    )
    result = await self_test._run_check(
        {"criterion_id": "c1", "command": "echo wrong", "expected": "wrong"},
        state, ops, isolated_exec_fn, config, step=2,
    )
    assert "changed after its last isolated run FAILED" in result


# ---------------------------------------------------------------------------
# dispatch(): never raises, always returns model-facing text
# ---------------------------------------------------------------------------


async def test_dispatch_declare_criteria(ops, isolated_exec_fn, workdir):
    state = self_test.SelfTestState()
    config = _config(workdir)
    result = await self_test.dispatch(
        "declare_criteria",
        {"criteria": [{"id": "c1", "description": "d", "how_to_check": "h"}]},
        state, ops, isolated_exec_fn, config, step=1,
    )
    assert "declared 1 criteria" in result
    assert "c1" in state.criteria


async def test_dispatch_run_check_unknown_criterion(ops, isolated_exec_fn, workdir):
    state = self_test.SelfTestState()
    config = _config(workdir)
    result = await self_test.dispatch(
        "run_check", {"criterion_id": "nope", "command": "true"}, state, ops, isolated_exec_fn, config, step=1
    )
    assert "unknown criterion_id" in result


async def test_dispatch_unknown_tool_name(ops, isolated_exec_fn, workdir):
    state = self_test.SelfTestState()
    config = _config(workdir)
    result = await self_test.dispatch("frobnicate", {}, state, ops, isolated_exec_fn, config, step=1)
    assert "unknown self-test tool" in result


async def test_dispatch_run_check_missing_command(ops, isolated_exec_fn, workdir):
    state = self_test.SelfTestState()
    config = _config(workdir)
    _declare(state, ("c1", "d"))
    result = await self_test.dispatch(
        "run_check", {"criterion_id": "c1", "command": ""}, state, ops, isolated_exec_fn, config, step=1
    )
    assert "command is required" in result


# ---------------------------------------------------------------------------
# Persistence — criteria/checks are written to disk (harness-owned state,
# never inlined into the model-facing history at full size).
# ---------------------------------------------------------------------------


async def test_declare_criteria_persists_to_disk(ops, isolated_exec_fn, workdir):
    state = self_test.SelfTestState()
    config = _config(workdir)
    await self_test.dispatch(
        "declare_criteria",
        {"criteria": [{"id": "c1", "description": "d", "how_to_check": "h"}], "deliverables": ["a.py"]},
        state, ops, isolated_exec_fn, config, step=1,
    )
    import json

    criteria_doc = json.loads((workdir / "state" / "criteria.json").read_text())
    assert criteria_doc["criteria"][0]["id"] == "c1"
    assert criteria_doc["deliverables"] == ["a.py"]


async def test_run_check_persists_checks_to_disk(ops, isolated_exec_fn, workdir):
    state = self_test.SelfTestState()
    config = _config(workdir)
    _declare(state, ("c1", "d"))
    await self_test._run_check(
        {"criterion_id": "c1", "command": "echo hi", "expected": "hi"},
        state, ops, isolated_exec_fn, config, step=1,
    )
    import json

    checks_doc = json.loads((workdir / "state" / "checks.json").read_text())
    assert checks_doc["c1"][0]["isolated_pass"] is True
