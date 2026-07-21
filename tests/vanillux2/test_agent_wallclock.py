"""Wall-clock discipline tests for the wallclock_only arm: a command that
outruns its exec deadline becomes a model-visible exit_code=124 result (not a
run-ending exception — harbor's docker exec RAISES on timeout), a model's own
`timeout N` raises the exec deadline (capped), and the per-request LLM
timeout is bounded and configurable.

Scripted fake model + local-filesystem fake environment — no docker, no live
LLM.
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


def _scripted_completion(script: list[dict], captured_kwargs: list | None = None):
    calls = {"i": 0}

    def fake_completion(**kwargs):
        if captured_kwargs is not None:
            captured_kwargs.append(kwargs)
        i = calls["i"]
        calls["i"] += 1
        return _FakeResponse(script[i])

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


def test_requested_timeout_parsing():
    agent = Vanillux2Agent(logs_dir=Path("/tmp/unused"))
    assert agent._requested_timeout_sec("timeout 300 make") == 300
    assert agent._requested_timeout_sec("timeout 30s ./run.sh") == 30
    assert agent._requested_timeout_sec("timeout -k 5 240 cmd") == 240
    assert agent._requested_timeout_sec("timeout 2m pytest") == 120
    assert agent._requested_timeout_sec("cd /app && timeout 90 a; timeout 400 b") == 400
    assert agent._requested_timeout_sec("ls -la") is None


def test_command_deadline_capped():
    agent = Vanillux2Agent(logs_dir=Path("/tmp/unused"))
    assert agent._command_deadline_sec("ls") == 120
    assert agent._command_deadline_sec("timeout 240 make") == 270  # requested + 30s margin
    assert agent._command_deadline_sec("timeout 100000 forever") == 600  # capped


async def test_exec_timeout_becomes_model_visible_result(tmp_path, monkeypatch):
    script = [_bash("c1", "HARNESS_HANG slow_build"), _bash("c2", _SUBMIT)]
    _, _, _, trajectory = await _run_agent(tmp_path, monkeypatch, script)

    tool_msgs = [m for m in trajectory if m.get("role") == "tool"]
    assert any("did not finish within" in (m.get("content") or "") for m in tool_msgs)
    assert any("exit_code=124" in (m.get("content") or "") for m in tool_msgs)
    # The run survived to submit instead of dying on the harbor exception.
    assert any("COMPLETE_TASK" in (m.get("content") or "") for m in tool_msgs)


async def test_model_requested_timeout_raises_exec_deadline(tmp_path, monkeypatch):
    script = [_bash("c1", "timeout 240 make -j4"), _bash("c2", _SUBMIT)]
    _, env, _, _ = await _run_agent(tmp_path, monkeypatch, script)
    deadline = next(t for cmd, t in env.exec_calls if "timeout 240 make" in cmd)
    assert deadline == 270


async def test_non_timeout_exec_error_still_propagates(tmp_path, monkeypatch):
    class BrokenEnvironment(FakeEnvironment):
        async def exec(self, command: str, timeout_sec: float | None = None) -> FakeExecResult:
            if "BOOM" in command:
                raise RuntimeError("docker daemon is gone")
            return await super().exec(command, timeout_sec)

    monkeypatch.setattr(agent_module, "_STATE_DIR", str(tmp_path / "vanillux2_state"))
    monkeypatch.setattr(litellm, "completion", _scripted_completion([_bash("c1", "BOOM")]))
    agent = Vanillux2Agent(logs_dir=tmp_path / "logs", model_name="anthropic/claude-haiku-4-5")
    env = BrokenEnvironment()
    await agent.setup(env)
    context = types.SimpleNamespace(cost_usd=0, n_input_tokens=0, n_output_tokens=0, metadata=None)
    with pytest.raises(RuntimeError, match="docker daemon is gone"):
        await agent.run("solve it", env, context)


async def test_llm_timeout_is_bounded_and_configurable(tmp_path, monkeypatch):
    captured: list = []
    script = [_bash("c1", _SUBMIT)]
    await _run_agent(tmp_path, monkeypatch, script, captured_kwargs=captured, llm_timeout=123)
    assert captured[0]["timeout"] == 123
    assert captured[0]["request_timeout"] == 123
