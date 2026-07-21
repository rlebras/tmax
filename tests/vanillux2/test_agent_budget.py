"""End-to-end tests for Vanillux2Agent's budget discipline: context-overflow
recovery (proactive + reactive compaction escalation), command-timeout
handling (a slow command must become a model-visible result, not a run-ending
exception — harbor's docker exec RAISES on timeout), model-requested
`timeout N` deadlines, the wall-clock deadline nudge + gate standdown, and
the one-tool-call-per-turn protocol repair.

Same scripted-fake-model / local-filesystem-environment pattern as
test_agent_self_test_gate.py — no docker, no live LLM.
"""

from __future__ import annotations

import asyncio
import json
import types
from dataclasses import dataclass, field
from pathlib import Path

import litellm
import pytest

import Vanillux2Agent.agent as agent_module
from Vanillux2Agent.agent import Vanillux2Agent


@dataclass
class FakeExecResult:
    stdout: str | None = None
    stderr: str | None = None
    return_code: int = 0


@dataclass
class FakeEnvironment:
    """Runs real bash locally, recording each exec's timeout_sec. Commands
    containing HARNESS_HANG raise harbor's raise-on-timeout RuntimeError."""

    exec_calls: list[tuple[str, float | None]] = field(default_factory=list)

    async def exec(self, command: str, timeout_sec: float | None = None) -> FakeExecResult:
        self.exec_calls.append((command, timeout_sec))
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


def _tool_call(call_id: str, name: str, args: dict) -> dict:
    return {
        "role": "assistant",
        "content": "THOUGHT: proceeding",
        "tool_calls": [
            {"id": call_id, "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}
        ],
    }


def _bash(call_id: str, command: str) -> dict:
    return _tool_call(call_id, "bash", {"command": command})


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


def _scripted_completion(script: list, captured_kwargs: list | None = None):
    """Each script entry is a message dict, or an Exception instance to raise."""
    calls = {"i": 0}

    def fake_completion(**kwargs):
        if captured_kwargs is not None:
            captured_kwargs.append(kwargs)
        i = calls["i"]
        calls["i"] += 1
        entry = script[i]
        if isinstance(entry, Exception):
            raise entry
        return _FakeResponse(entry)

    return fake_completion


async def _run_agent(tmp_path, monkeypatch, script, captured_kwargs=None, **agent_kwargs):
    monkeypatch.setattr(agent_module, "_STATE_DIR", str(tmp_path / "vanillux2_state"))
    monkeypatch.setattr(litellm, "completion", _scripted_completion(script, captured_kwargs))

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
    return agent, env, context, trajectory


def _ctx_overflow() -> Exception:
    return litellm.exceptions.ContextWindowExceededError(
        "prompt is too long", "anthropic/claude-haiku-4-5", "anthropic"
    )


# ---------------------------------------------------------------------------
# Context-overflow recovery
# ---------------------------------------------------------------------------


async def test_overflow_escalates_compaction_and_continues(tmp_path, monkeypatch):
    script = [_ctx_overflow(), _bash("c1", _SUBMIT)]
    _, _, context, trajectory = await _run_agent(
        tmp_path, monkeypatch, script, enable_self_test_gate=False
    )

    compaction = context.metadata["compaction"]
    assert compaction["overflow_recoveries"] == 1
    assert compaction["final_level"] == 1
    # The run went on to finish, instead of ending with nothing.
    tool_msgs = [m for m in trajectory if m.get("role") == "tool"]
    assert any("COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT" in (m.get("content") or "") for m in tool_msgs)


async def test_overflow_at_max_level_still_terminates(tmp_path, monkeypatch):
    script = [_ctx_overflow(), _ctx_overflow(), _ctx_overflow(), _bash("c1", _SUBMIT)]
    _, _, context, trajectory = await _run_agent(
        tmp_path, monkeypatch, script, enable_self_test_gate=False
    )
    compaction = context.metadata["compaction"]
    assert compaction["overflow_recoveries"] == 2  # levels 1 and 2
    assert compaction["final_level"] == 2
    # Third consecutive overflow at max level ends the run the old way.
    assert not any(
        "COMPLETE_TASK" in (m.get("content") or "") for m in trajectory if m.get("role") == "tool"
    )


async def test_proactive_escalation_before_send(tmp_path, monkeypatch):
    # A context budget far smaller than the instance prompt: the proactive
    # check must escalate to max compaction BEFORE the first request, and the
    # run still proceeds (the reactive path stays the backstop).
    script = [_bash("c1", _SUBMIT)]
    _, _, context, trajectory = await _run_agent(
        tmp_path,
        monkeypatch,
        script,
        enable_self_test_gate=False,
        max_tokens=100,
        max_context_tokens=500,
        context_headroom_tokens=100,
    )
    compaction = context.metadata["compaction"]
    assert compaction["proactive_escalations"] == 2
    assert compaction["final_level"] == 2
    assert compaction["overflow_recoveries"] == 0
    tool_msgs = [m for m in trajectory if m.get("role") == "tool"]
    assert any("COMPLETE_TASK" in (m.get("content") or "") for m in tool_msgs)


async def test_no_escalation_when_budget_fits(tmp_path, monkeypatch):
    script = [_bash("c1", _SUBMIT)]
    _, _, context, _ = await _run_agent(
        tmp_path,
        monkeypatch,
        script,
        enable_self_test_gate=False,
        max_tokens=100,
        max_context_tokens=200_000,
    )
    compaction = context.metadata["compaction"]
    assert compaction["proactive_escalations"] == 0
    assert compaction["final_level"] == 0


# ---------------------------------------------------------------------------
# Command timeouts — a slow command is reported, not fatal
# ---------------------------------------------------------------------------


async def test_exec_timeout_becomes_model_visible_result(tmp_path, monkeypatch):
    script = [_bash("c1", "HARNESS_HANG slow_build"), _bash("c2", _SUBMIT)]
    _, _, context, trajectory = await _run_agent(
        tmp_path, monkeypatch, script, enable_self_test_gate=False
    )

    tool_msgs = [m for m in trajectory if m.get("role") == "tool"]
    assert any("did not finish within" in (m.get("content") or "") for m in tool_msgs)
    assert any("exit_code=124" in (m.get("content") or "") for m in tool_msgs)
    # The run survived to submit.
    assert any("COMPLETE_TASK" in (m.get("content") or "") for m in tool_msgs)


async def test_non_timeout_exec_error_still_propagates(tmp_path, monkeypatch):
    class BrokenEnvironment(FakeEnvironment):
        async def exec(self, command: str, timeout_sec: float | None = None) -> FakeExecResult:
            if "BOOM" in command:
                raise RuntimeError("docker daemon is gone")
            return await super().exec(command, timeout_sec)

    monkeypatch.setattr(agent_module, "_STATE_DIR", str(tmp_path / "vanillux2_state"))
    monkeypatch.setattr(litellm, "completion", _scripted_completion([_bash("c1", "BOOM")]))
    agent = Vanillux2Agent(
        logs_dir=tmp_path / "logs", model_name="anthropic/claude-haiku-4-5", enable_self_test_gate=False
    )
    env = BrokenEnvironment()
    await agent.setup(env)
    context = types.SimpleNamespace(cost_usd=0, n_input_tokens=0, n_output_tokens=0, metadata=None)
    with pytest.raises(RuntimeError, match="docker daemon is gone"):
        await agent.run("solve it", env, context)


async def test_model_requested_timeout_raises_exec_deadline(tmp_path, monkeypatch):
    script = [
        _bash("c1", "timeout 240 make -j4"),
        _bash("c2", "timeout 100000 forever"),
        _bash("c3", "ls"),
        _bash("c4", _SUBMIT),
    ]
    _, env, _, _ = await _run_agent(tmp_path, monkeypatch, script, enable_self_test_gate=False)

    def deadline_for(snippet):
        return next(t for cmd, t in env.exec_calls if snippet in cmd)

    assert deadline_for("timeout 240 make") == 270  # requested + 30s margin
    assert deadline_for("timeout 100000 forever") == 600  # capped at max_command_timeout
    assert deadline_for("ls") == 120  # default untouched


def test_requested_timeout_parsing():
    agent = Vanillux2Agent(logs_dir=Path("/tmp/unused"))
    assert agent._requested_timeout_sec("timeout 300 make") == 300
    assert agent._requested_timeout_sec("timeout 30s ./run.sh") == 30
    assert agent._requested_timeout_sec("timeout -k 5 240 cmd") == 240
    assert agent._requested_timeout_sec("timeout 2m pytest") == 120
    assert agent._requested_timeout_sec("cd /app && timeout 90 a; timeout 400 b") == 400
    assert agent._requested_timeout_sec("ls -la") is None
    assert agent._requested_timeout_sec("") is None


# ---------------------------------------------------------------------------
# LLM request timeout
# ---------------------------------------------------------------------------


async def test_llm_timeout_is_bounded_and_configurable(tmp_path, monkeypatch):
    captured: list = []
    script = [_bash("c1", _SUBMIT)]
    await _run_agent(
        tmp_path, monkeypatch, script, captured_kwargs=captured,
        enable_self_test_gate=False, llm_timeout=123,
    )
    assert captured[0]["timeout"] == 123
    assert captured[0]["request_timeout"] == 123


# ---------------------------------------------------------------------------
# One tool call per turn — extras answered, no orphaned tool_call_ids
# ---------------------------------------------------------------------------


async def test_extra_tool_calls_get_ignored_responses(tmp_path, monkeypatch):
    double_call = {
        "role": "assistant",
        "content": "THOUGHT: two at once",
        "tool_calls": [
            {"id": "keep", "type": "function", "function": {"name": "bash", "arguments": json.dumps({"command": "echo one"})}},
            {"id": "orphan", "type": "function", "function": {"name": "bash", "arguments": json.dumps({"command": "echo two"})}},
        ],
    }
    script = [double_call, _bash("c2", _SUBMIT)]
    _, _, _, trajectory = await _run_agent(
        tmp_path, monkeypatch, script, enable_self_test_gate=False
    )

    by_id = {m.get("tool_call_id"): m for m in trajectory if m.get("role") == "tool"}
    assert "keep" in by_id and "one" in by_id["keep"]["content"]
    assert "orphan" in by_id and "ignored" in by_id["orphan"]["content"]
    assert "two" not in by_id["orphan"]["content"]  # the extra call never ran


# ---------------------------------------------------------------------------
# Wall-clock deadline: nudge + gate standdown
# ---------------------------------------------------------------------------


async def test_deadline_nudge_and_gate_standdown(tmp_path, monkeypatch):
    # Budget (60s) already inside the warning window (300s): the very first
    # step gets the finalize-now nudge, and the gate — which would normally
    # reject this no-criteria submit — stands down instead of burning steps.
    script = [_bash("c1", _SUBMIT)]
    _, _, context, trajectory = await _run_agent(
        tmp_path, monkeypatch, script,
        enable_self_test_gate=True, wall_clock_budget_sec=60,
    )

    nudges = [
        m for m in trajectory
        if m.get("role") == "user" and "Wall-clock budget nearly exhausted" in (m.get("content") or "")
    ]
    assert len(nudges) == 1
    assert context.metadata["deadline"]["warned"] is True
    st = context.metadata["self_test"]
    assert st["gate_rejections"] == 0  # stood down, didn't reject
    assert st["submitted_with_failing_checks"] is True  # but the submit is flagged
    tool_msgs = [m for m in trajectory if m.get("role") == "tool"]
    assert any("COMPLETE_TASK" in (m.get("content") or "") for m in tool_msgs)


async def test_no_deadline_machinery_when_unconfigured(tmp_path, monkeypatch):
    script = [_bash("c1", _SUBMIT)]
    _, _, context, trajectory = await _run_agent(
        tmp_path, monkeypatch, script, enable_self_test_gate=False
    )
    assert context.metadata["deadline"] == {"budget_sec": None, "warned": False}
    assert not any(
        "Wall-clock budget" in (m.get("content") or "") for m in trajectory if m.get("role") == "user"
    )
