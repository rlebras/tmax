"""End-to-end wiring tests: Vanillux2Agent.run()'s submit gate actually blocks
and then unblocks COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT, driven through the
real agent loop with a scripted fake model and a local-filesystem fake
environment (no docker, no live LLM). Unlike test_self_test.py (which drives
self_test.py's functions directly), this exercises the actual interception/
gate wiring added to Vanillux2Agent.run().

This branch (self_test_only) keeps the bash-only tool contract, so
everything goes through the intercepted `agent-check` convention and the
model-written criteria file — there are no declare_criteria/run_check tools
here.

Imports the full ``Vanillux2Agent`` package (not the bare-module trick
tests/vanillux2/conftest.py uses for self_test) since this needs
``agent.py`` itself, which requires harbor.
"""

from __future__ import annotations

import asyncio
import json
import types
from dataclasses import dataclass
from pathlib import Path

import litellm
import pytest

import Vanillux2Agent.agent as agent_module
from Vanillux2Agent.agent import Vanillux2Agent
from Vanillux2Agent import self_test


@dataclass
class FakeExecResult:
    stdout: str | None = None
    stderr: str | None = None
    return_code: int = 0


class FakeEnvironment:
    """Runs real bash locally; upload_file/download_file are plain local
    copies. Commands containing HARNESS_HANG raise harbor's raise-on-timeout
    RuntimeError (see agent.py's _timed_out_result)."""

    async def exec(self, command: str, timeout_sec: float | None = None) -> FakeExecResult:
        if "HARNESS_HANG" in command:
            raise RuntimeError(f"Command timed out after {timeout_sec} seconds")
        proc = await asyncio.create_subprocess_shell(
            command, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        out, err = await proc.communicate()
        return FakeExecResult(out.decode("utf-8", "replace"), err.decode("utf-8", "replace"), proc.returncode or 0)

    async def upload_file(self, local_path: str, remote_path: str) -> None:
        Path(remote_path).write_bytes(Path(local_path).read_bytes())

    async def download_file(self, remote_path: str, local_path: str) -> None:
        Path(local_path).write_bytes(Path(remote_path).read_bytes())


def _bash(call_id: str, command: str) -> dict:
    return {
        "role": "assistant",
        "content": "THOUGHT: proceeding",
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {"name": "bash", "arguments": json.dumps({"command": command})},
            }
        ],
    }


_SUBMIT = "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"


class _FakeMessage:
    def __init__(self, d: dict):
        self._d = d

    def model_dump(self) -> dict:
        return self._d


class _FakeResponse:
    def __init__(self, msg_dict: dict):
        self.choices = [types.SimpleNamespace(message=_FakeMessage(msg_dict))]
        self.usage = None


def _scripted_completion(script: list[dict]):
    calls = {"i": 0}

    def fake_completion(**kwargs):
        i = calls["i"]
        calls["i"] += 1
        return _FakeResponse(script[i])

    return fake_completion


async def _run_agent(tmp_path, monkeypatch, script, **agent_kwargs):
    # Isolate the persistent-shell + self-test state dirs, so tests neither
    # depend on nor clobber whatever a real agent last left in /tmp.
    monkeypatch.setattr(agent_module, "_STATE_DIR", str(tmp_path / "vanillux2_state"))
    monkeypatch.setattr(litellm, "completion", _scripted_completion(script))

    agent = Vanillux2Agent(
        logs_dir=tmp_path / "logs",
        model_name="anthropic/claude-haiku-4-5",
        max_steps=agent_kwargs.pop("max_steps", 10),
        **agent_kwargs,
    )
    env = FakeEnvironment()
    await agent.setup(env)
    context = types.SimpleNamespace(cost_usd=0, n_input_tokens=0, n_output_tokens=0, metadata=None)
    await agent.run("solve it", env, context)
    trajectory = json.loads((tmp_path / "logs" / "trajectory.json").read_text())
    return agent, context, trajectory


def _tool_messages(trajectory):
    return [m for m in trajectory if m.get("role") == "tool"]


def _criteria_write_command(tmp_path, *criteria, deliverables=None) -> str:
    criteria_path = f"{tmp_path}/vanillux2_state/self_test/criteria.json"
    doc = json.dumps(
        {
            "criteria": [
                {"id": cid, "description": desc, "how_to_check": "n/a"} for cid, desc in criteria
            ],
            "deliverables": deliverables or [],
        }
    )
    return (
        f"mkdir -p {tmp_path}/vanillux2_state/self_test && "
        f"cat > {criteria_path} << 'EOF'\n{doc}\nEOF"
    )


async def test_gate_blocks_then_allows_via_agent_check(tmp_path, monkeypatch):
    script = [
        _bash("c1", _criteria_write_command(tmp_path, ("hi", "prints hi"), ("bye", "prints bye"))),
        _bash("c2", _SUBMIT),  # criteria declared but nothing covered -> rejected
        _bash("c3", 'agent-check hi -- test "$(echo hi)" = "hi"'),
        _bash("c4", 'echo checking\nagent-check bye -- test "$(echo bye)" = "bye"'),
        _bash("c5", _SUBMIT),
    ]

    _, context, trajectory = await _run_agent(tmp_path, monkeypatch, script)

    st = context.metadata["self_test"]
    assert st["enabled"] is True
    assert st["criteria_declared"] == 2
    assert st["checks_run"] == 2
    assert st["criteria_covered"] == 2
    assert st["gate_rejections"] == 1
    assert st["submitted_with_failing_checks"] is False

    tool_msgs = _tool_messages(trajectory)
    nudges = [m for m in tool_msgs if "Submit rejected" in (m.get("content") or "")]
    assert len(nudges) == 1
    assert "agent-check" in nudges[0]["content"]
    # The rejected submit was intercepted BEFORE execution: the marker only
    # ever appears once, on the final allowed submit.
    marker_msgs = [m for m in tool_msgs if "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT" in (m.get("content") or "")]
    assert len(marker_msgs) == 1
    # The embedded form ran its remainder ("echo checking") before the check.
    embedded = [m for m in tool_msgs if "check bye: PASS" in (m.get("content") or "")]
    assert len(embedded) == 1 and "checking" in embedded[0]["content"]

    st_log = json.loads((tmp_path / "logs" / "self_test.json").read_text())
    assert st_log["coverage"] == {"hi": True, "bye": True}


async def test_submit_gate_forces_through_after_max_rejections(tmp_path, monkeypatch):
    # No criteria at all — every submit attempt is rejected until
    # max_gate_rejections is exhausted, at which point the run must still
    # terminate (not burn the whole step budget) and record
    # submitted_with_failing_checks=True.
    script = [_bash(f"c{i}", _SUBMIT) for i in range(10)]

    _, context, trajectory = await _run_agent(
        tmp_path, monkeypatch, script, max_gate_rejections=2
    )

    st = context.metadata["self_test"]
    assert st["gate_rejections"] == 2
    assert st["submitted_with_failing_checks"] is True
    assert len([m for m in trajectory if m.get("role") == "assistant"]) == 3


async def test_gate_disabled_leaves_submit_untouched(tmp_path, monkeypatch):
    script = [_bash("c1", _SUBMIT)]
    _, context, trajectory = await _run_agent(
        tmp_path, monkeypatch, script, enable_self_test_gate=False
    )
    assert context.metadata["self_test"]["enabled"] is False
    assert context.metadata["self_test"]["gate_rejections"] == 0
    assert len([m for m in trajectory if m.get("role") == "assistant"]) == 1
    assert not (tmp_path / "logs" / "self_test.json").exists()
    # No self-test prompt text on the disabled arm.
    assert "agent-check" not in trajectory[1]["content"]


async def test_leftover_state_caught_through_agent_loop(tmp_path, monkeypatch):
    # The leftover-state failure mode end to end: the check passes in the
    # session only because of an undeclared leftover file (referenced
    # cwd-relatively via the persistent shell's cwd), so isolation fails it
    # and the gate rejects until rejections are exhausted.
    workdir = tmp_path / "work"
    workdir.mkdir()
    (workdir / "leftover.txt").write_text("42\n")
    script = [
        _bash("c0", f"cd {workdir}"),  # persistent shell: cwd sticks
        _bash("c1", _criteria_write_command(tmp_path, ("a", "answer is 42"), ("b", "has solver"))),
        _bash("c2", 'agent-check a -- test "$(cat leftover.txt)" = "42"'),
        _bash("c3", _SUBMIT),
        _bash("c4", _SUBMIT),
    ]

    _, context, _ = await _run_agent(tmp_path, monkeypatch, script, max_gate_rejections=1)

    st = context.metadata["self_test"]
    assert st["checks_run"] == 1
    assert st["session_isolation_discrepancies"] == 1
    assert st["gate_rejections"] == 1
    assert st["submitted_with_failing_checks"] is True

    st_log = json.loads((tmp_path / "logs" / "self_test.json").read_text())
    record = st_log["checks"]["a"][0]
    assert record["session_pass"] is True
    assert record["isolated_pass"] is False


async def test_existence_probe_is_answered_not_executed(tmp_path, monkeypatch):
    script = [_bash("c1", "which agent-check"), _bash("c2", _SUBMIT)]
    _, _, trajectory = await _run_agent(
        tmp_path, monkeypatch, script, max_gate_rejections=0
    )
    assert _tool_messages(trajectory)[0]["content"] == self_test.AGENT_CHECK_EXISTENCE_HINT


async def test_malformed_agent_check_gets_syntax_hint(tmp_path, monkeypatch):
    script = [_bash("c1", "agent-check hi test 1 = 1"), _bash("c2", _SUBMIT)]  # missing ` -- `
    _, _, trajectory = await _run_agent(
        tmp_path, monkeypatch, script, max_gate_rejections=0
    )
    assert _tool_messages(trajectory)[0]["content"] == self_test.AGENT_CHECK_SYNTAX_HINT


async def test_instance_prompt_describes_convention(tmp_path, monkeypatch):
    script = [_bash("c1", _SUBMIT)]
    _, _, trajectory = await _run_agent(
        tmp_path, monkeypatch, script, max_gate_rejections=0
    )
    instance_msg = trajectory[1]
    assert instance_msg["role"] == "user"
    assert "agent-check" in instance_msg["content"]
    assert "criteria.json" in instance_msg["content"]
    # Tool-mode text from the ctx-mgmt branches must not leak in here.
    assert "declare_criteria" not in instance_msg["content"]
    assert "run_check" not in instance_msg["content"]


async def test_slow_check_command_fails_instead_of_crashing(tmp_path, monkeypatch):
    # harbor raises on exec timeout; with the gate roughly doubling command
    # executions, a hanging check must become a failed check, not a dead run.
    script = [
        _bash("c1", _criteria_write_command(tmp_path, ("a", "d1"), ("b", "d2"))),
        _bash("c2", "agent-check a -- HARNESS_HANG verify"),
        _bash("c3", _SUBMIT),
        _bash("c4", _SUBMIT),
    ]
    _, context, trajectory = await _run_agent(tmp_path, monkeypatch, script, max_gate_rejections=1)

    st = context.metadata["self_test"]
    assert st["checks_run"] == 1
    assert st["submitted_with_failing_checks"] is True  # forced through, run survived
    check_msgs = [m for m in _tool_messages(trajectory) if "check a: FAIL" in (m.get("content") or "")]
    assert len(check_msgs) == 1
    assert "did not finish within" in check_msgs[0]["content"]
