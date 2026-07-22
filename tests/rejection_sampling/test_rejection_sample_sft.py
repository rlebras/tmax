"""Tests for rl_data/rejection_sample_sft.py — the STaR harvest+convert tool.

Covers: reward filtering, format-error dropping, submit requirement, dedup,
per-task cap (shortest-first), length cap, message sanitization to the SFT
schema, and the contamination guard that refuses eval-sourced inputs.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import rl_data.rejection_sample_sft as rs


def _asst(cmd: str, content: str = "THOUGHT: go", tid: str = "c1") -> dict:
    return {
        "role": "assistant",
        "content": content,
        "tool_calls": [
            {"id": tid, "type": "function", "function": {"name": "bash", "arguments": json.dumps({"command": cmd})}}
        ],
    }


def _tool(content: str, tid: str = "c1") -> dict:
    return {"role": "tool", "tool_call_id": tid, "content": content}


def _traj(*cmds: str, submit: bool = True) -> list[dict]:
    msgs = [{"role": "system", "content": "sys"}, {"role": "user", "content": "task"}]
    for i, c in enumerate(cmds):
        msgs.append(_asst(c, tid=f"c{i}"))
        msgs.append(_tool("ok", tid=f"c{i}"))
    if submit:
        msgs.append(_asst(f"echo {rs.SUBMIT_MARKER}", tid="sub"))
        msgs.append(_tool(rs.SUBMIT_MARKER, tid="sub"))
    return msgs


def _rec(task, reward, msgs):
    return rs.Record(task, reward, msgs, "test")


# --- filtering ---------------------------------------------------------------


def test_only_passing_kept():
    recs = [_rec("t1", 1.0, _traj("ls")), _rec("t1", 0.0, _traj("ls", "pwd"))]
    ex = rs.harvest(recs, rs.HarvestConfig())
    assert len(ex) == 1
    assert ex[0]["id"] == "t1"


def test_format_error_dropped():
    bad = _traj("ls")
    bad.insert(2, {"role": "assistant", "content": "no tool call here"})  # malformed action
    st = rs.HarvestStats()
    ex = rs.harvest([_rec("t1", 1.0, bad)], rs.HarvestConfig(), st)
    assert ex == []
    assert st.dropped_format_error == 1


def test_malformed_tool_args_is_format_error():
    m = _traj("ls")
    m[2]["tool_calls"][0]["function"]["arguments"] = "{not json"
    assert rs.has_format_error(m) is True


def test_require_submit():
    st = rs.HarvestStats()
    ex = rs.harvest([_rec("t1", 1.0, _traj("ls", submit=False))], rs.HarvestConfig(), st)
    assert ex == []
    assert st.dropped_no_submit == 1


def test_keep_when_not_requiring_submit():
    cfg = rs.HarvestConfig(require_submit=False)
    ex = rs.harvest([_rec("t1", 1.0, _traj("ls", submit=False))], cfg)
    assert len(ex) == 1


def test_length_cap():
    huge = _traj("x" * 200000)
    st = rs.HarvestStats()
    ex = rs.harvest([_rec("t1", 1.0, huge)], rs.HarvestConfig(max_tokens=1000), st)
    assert ex == []
    assert st.dropped_too_long == 1


def test_min_assistant_turns():
    # a submit-only trajectory has 1 assistant turn; require 2
    st = rs.HarvestStats()
    ex = rs.harvest([_rec("t1", 1.0, _traj(submit=True))], rs.HarvestConfig(min_assistant_turns=2), st)
    assert ex == []
    assert st.dropped_too_short == 1


# --- dedup + per-task cap ----------------------------------------------------


def test_identical_action_sequences_deduped():
    st = rs.HarvestStats()
    recs = [_rec("t1", 1.0, _traj("ls", "pwd")), _rec("t1", 1.0, _traj("ls", "pwd"))]
    ex = rs.harvest(recs, rs.HarvestConfig(), st)
    assert len(ex) == 1
    assert st.dropped_dup == 1


def test_per_task_cap_keeps_shortest():
    recs = [
        _rec("t1", 1.0, _traj("a", "b", "c", "d")),  # long
        _rec("t1", 1.0, _traj("a")),                  # short
        _rec("t1", 1.0, _traj("a", "b")),             # mid
    ]
    st = rs.HarvestStats()
    ex = rs.harvest(recs, rs.HarvestConfig(max_per_task=2), st)
    assert len(ex) == 2
    assert st.dropped_over_cap == 1
    # shortest two kept: the 1-cmd and 2-cmd trajectories
    kept_lens = sorted(len([m for m in e["messages"] if m["role"] == "assistant"]) for e in ex)
    assert kept_lens == [2, 3]  # 1cmd+submit=2 asst, 2cmd+submit=3 asst


def test_multiple_tasks_bucketed_independently():
    recs = [_rec("t1", 1.0, _traj("a")), _rec("t2", 1.0, _traj("b"))]
    ex = rs.harvest(recs, rs.HarvestConfig())
    assert {e["id"] for e in ex} == {"t1", "t2"}


# --- SFT schema --------------------------------------------------------------


def test_sft_example_schema_and_sanitization():
    m = _traj("ls")
    # inject provider junk that must be stripped
    m[2]["provider_specific_fields"] = {"x": 1}
    m[2]["reasoning_content"] = "thinking..."
    ex = rs.harvest([_rec("t1", 1.0, m)], rs.HarvestConfig())[0]
    assert set(ex) == {"messages", "tools", "dataset", "id"}
    assert ex["tools"] == [rs.BASH_TOOL]
    asst = [x for x in ex["messages"] if x["role"] == "assistant"][0]
    assert "provider_specific_fields" not in asst
    assert asst["reasoning_content"] == "thinking..."  # preserved
    tc = asst["tool_calls"][0]
    assert set(tc) == {"id", "type", "function"}
    assert json.loads(tc["function"]["arguments"]) == {"command": "ls"}


# --- contamination guard -----------------------------------------------------


def test_eval_source_refused(tmp_path):
    p = tmp_path / "terminal-bench-2-1" / "run_summary.json"
    p.parent.mkdir(parents=True)
    p.write_text(json.dumps({"results": [{"reward": 1, "messages": _traj("ls")}]}))
    with pytest.raises(ValueError, match="eval-sourced"):
        list(rs.load_records([p]))


def test_terminal_bench_slug_still_refused(tmp_path):
    # terminal-bench eval experiments are named eval-tmax-9b-tb21-*; the tb-slug
    # must still trip the guard.
    p = tmp_path / "eval-tmax-9b-tb21-64k" / "run_summary.json"
    p.parent.mkdir(parents=True)
    p.write_text(json.dumps({"results": [{"reward": 1, "messages": _traj("ls")}]}))
    with pytest.raises(ValueError, match="eval-sourced"):
        list(rs.load_records([p]))


def test_rejsample_rollout_path_allowed(tmp_path):
    # legit training rollouts also run through launch_eval.sh and land in
    # eval-tmax-9b-rejsample-* dirs — the "eval-" prefix must NOT refuse them.
    assert rs._looks_like_eval_source("/x/eval-tmax-9b-rejsample-smoke/task_000355__abc") is False
    p = tmp_path / "eval-tmax-9b-rejsample-smoke" / "run_summary.json"
    p.parent.mkdir(parents=True)
    p.write_text(json.dumps({"results": [{"reward": 1, "messages": _traj("ls")}]}))
    recs = list(rs.load_records([p]))  # must NOT raise
    assert len(recs) == 1


def test_eval_source_allowed_with_override(tmp_path):
    p = tmp_path / "evaluation_assets" / "run_summary.json"
    p.parent.mkdir(parents=True)
    p.write_text(json.dumps({"results": [{"reward": 1, "messages": _traj("ls")}]}))
    recs = list(rs.load_records([p], allow_eval_source=True))
    assert len(recs) == 1 and recs[0].reward == 1.0


# --- loading rl_data summary format -----------------------------------------


def test_load_run_n_solutions_summary(tmp_path):
    summ = {
        "num_runs": 2,
        "results": [
            {"success": True, "reward": 1, "messages": _traj("ls")},
            {"success": False, "reward": 0, "messages": _traj("nope", submit=False)},
        ],
    }
    p = tmp_path / "gen-widget__abc_summary.json"
    p.write_text(json.dumps(summ))
    recs = list(rs.load_records([p]))
    assert len(recs) == 2
    assert [r.reward for r in recs] == [1.0, 0.0]
    ex = rs.harvest(recs, rs.HarvestConfig())
    assert len(ex) == 1  # only the passing one survives


def test_load_jsonl(tmp_path):
    p = tmp_path / "rollouts.jsonl"
    with p.open("w") as f:
        f.write(json.dumps({"task_id": "t1", "reward": 1, "messages": _traj("ls")}) + "\n")
        f.write(json.dumps({"task_id": "t2", "reward": 0, "messages": _traj("x", submit=False)}) + "\n")
    recs = list(rs.load_records([p]))
    assert {r.task_id for r in recs} == {"t1", "t2"}


def _write_harbor_trial(root: Path, task: str, trial: str, reward: float, msgs: list[dict]):
    d = root / f"{task}__{trial}"
    (d / "agent").mkdir(parents=True)
    (d / "result.json").write_text(json.dumps({"verifier_result": {"rewards": {"reward": reward}}}))
    (d / "agent" / "trajectory.json").write_text(json.dumps(msgs))
    return d


# --- harbor rollout ingestion ------------------------------------------------


def test_load_harbor_trials_and_task_grouping(tmp_path):
    root = tmp_path / "jobs" / "tmax15k-rollouts"
    _write_harbor_trial(root, "gen-widget", "aaa", 1.0, _traj("ls"))
    _write_harbor_trial(root, "gen-widget", "bbb", 0.0, _traj("no", submit=False))
    _write_harbor_trial(root, "gen-parser", "ccc", 1.0, _traj("cat f"))
    recs = list(rs.load_records(rs._expand_inputs([str(root)])))
    assert len(recs) == 3
    # trailing __hash stripped so the two gen-widget attempts share a task id
    assert {r.task_id for r in recs} == {"gen-widget", "gen-parser"}
    assert sorted(r.reward for r in recs if r.task_id == "gen-widget") == [0.0, 1.0]


def test_harvest_from_harbor_rollouts(tmp_path):
    root = tmp_path / "jobs" / "rollouts"
    _write_harbor_trial(root, "t1", "a", 1.0, _traj("ls", "cat"))
    _write_harbor_trial(root, "t1", "b", 0.0, _traj("x", submit=False))
    ex = rs.harvest(rs.load_records(rs._expand_inputs([str(root)])), rs.HarvestConfig())
    assert len(ex) == 1 and ex[0]["id"] == "t1"


def test_single_harbor_trial_dir_input(tmp_path):
    root = tmp_path / "run"
    d = _write_harbor_trial(root, "t1", "a", 1.0, _traj("ls"))
    recs = list(rs.load_records(rs._expand_inputs([str(d)])))
    assert len(recs) == 1 and recs[0].task_id == "t1"


# --- relevance band (solve-rate focus) --------------------------------------


def test_always_solved_task_skipped_when_max_solve_rate_below_1():
    # t_easy solved 3/3 (rate 1.0); t_flaky solved 1/3 (rate 0.33)
    recs = [
        _rec("t_easy", 1.0, _traj("a")), _rec("t_easy", 1.0, _traj("a", "b")), _rec("t_easy", 1.0, _traj("a", "b", "c")),
        _rec("t_flaky", 1.0, _traj("z")), _rec("t_flaky", 0.0, _traj("q", submit=False)), _rec("t_flaky", 0.0, _traj("w", submit=False)),
    ]
    st = rs.HarvestStats()
    ex = rs.harvest(recs, rs.HarvestConfig(max_solve_rate=0.8), st)
    ids = {e["id"] for e in ex}
    assert ids == {"t_flaky"}                 # easy (always-solved) dropped
    assert st.tasks_over_solve_rate == 1
    assert st.dropped_over_solve_rate == 3    # its 3 passing trajectories skipped


def test_always_solved_kept_by_default():
    recs = [_rec("t_easy", 1.0, _traj("a")), _rec("t_easy", 1.0, _traj("a", "b"))]
    st = rs.HarvestStats()
    ex = rs.harvest(recs, rs.HarvestConfig(max_solve_rate=1.0), st)
    assert {e["id"] for e in ex} == {"t_easy"}
    assert st.tasks_over_solve_rate == 0


def test_never_solved_task_yields_nothing():
    recs = [_rec("t0", 0.0, _traj("a", submit=False)), _rec("t0", 0.0, _traj("b", submit=False))]
    st = rs.HarvestStats()
    ex = rs.harvest(recs, rs.HarvestConfig(), st)
    assert ex == []
    assert st.tasks_solvable == 0
    assert st.tasks_seen == 1


def test_solve_rate_histogram_counts_solvable_tasks():
    recs = [
        _rec("a", 1.0, _traj("x")), _rec("a", 1.0, _traj("y")),          # 2/2 = 1.0
        _rec("b", 1.0, _traj("x")), _rec("b", 0.0, _traj("y", submit=False)),  # 1/2 = 0.5
        _rec("c", 0.0, _traj("x", submit=False)),                        # 0/1 never solved
    ]
    st = rs.HarvestStats()
    rs.harvest(recs, rs.HarvestConfig(), st)
    assert st.tasks_seen == 3
    assert st.tasks_solvable == 2  # a and b; c excluded
    assert st.solve_rate_hist.get("1.0 (always)") == 1
    assert st.solve_rate_hist.get("[0.5,0.75)") == 1


def test_end_to_end_cli(tmp_path):
    summ = {"results": [{"reward": 1, "messages": _traj("ls", "cat f")}]}
    src = tmp_path / "gen-task__x_summary.json"
    src.write_text(json.dumps(summ))
    out = tmp_path / "sft.jsonl"
    rc = rs.main([str(src), "--out", str(out)])
    assert rc == 0
    rows = [json.loads(l) for l in out.read_text().splitlines()]
    assert len(rows) == 1
    assert rows[0]["dataset"] == "tmax-rejection-sft"
    assert rows[0]["tools"] == [rs.BASH_TOOL]
