"""Error-hint tests for the error_hints arm: a nonzero-exit observation
matching a known error signature gets exactly one recovery hint appended,
each signature fires at most once per run, successful commands never get
hints, and the feature can be disabled.

Scripted fake model + local-filesystem fake environment — no docker, no live
LLM.
"""

from __future__ import annotations

import asyncio
import json
import types
from dataclasses import dataclass

import litellm

import Vanillux2Agent.agent as agent_module
from Vanillux2Agent.agent import Vanillux2Agent


@dataclass
class FakeExecResult:
    stdout: str | None = None
    stderr: str | None = None
    return_code: int = 0


class FakeEnvironment:
    async def exec(self, command: str, timeout_sec: float | None = None) -> FakeExecResult:
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


def _scripted_completion(script: list[dict]):
    calls = {"i": 0}

    def fake_completion(**kwargs):
        i = calls["i"]
        calls["i"] += 1
        return _FakeResponse(script[i])

    return fake_completion


async def _run_agent(tmp_path, monkeypatch, script, **agent_kwargs):
    monkeypatch.setattr(agent_module, "_STATE_DIR", str(tmp_path / "vanillux2_state"))
    monkeypatch.setattr(litellm, "completion", _scripted_completion(script))
    agent = Vanillux2Agent(
        logs_dir=tmp_path / "logs", model_name="anthropic/claude-haiku-4-5",
        max_steps=len(script), **agent_kwargs,
    )
    env = FakeEnvironment()
    await agent.setup(env)
    context = types.SimpleNamespace(cost_usd=0, n_input_tokens=0, n_output_tokens=0, metadata=None)
    await agent.run("solve it", env, context)
    trajectory = json.loads((tmp_path / "logs" / "trajectory.json").read_text())
    return context, trajectory


def _tools(trajectory):
    return [m for m in trajectory if m.get("role") == "tool"]


async def test_command_not_found_gets_hint(tmp_path, monkeypatch):
    script = [_bash("c1", "definitely_not_a_real_command"), _bash("c2", _SUBMIT)]
    context, trajectory = await _run_agent(tmp_path, monkeypatch, script)
    first = _tools(trajectory)[0]["content"]
    assert "harness hint" in first
    assert "command was not found" in first
    assert context.metadata["error_hints"]["fired"] == ["cmd_not_found"]


async def test_module_not_found_gets_hint(tmp_path, monkeypatch):
    script = [_bash("c1", 'python3 -c "import nonexistent_module_xyz"'), _bash("c2", _SUBMIT)]
    context, trajectory = await _run_agent(tmp_path, monkeypatch, script)
    first = _tools(trajectory)[0]["content"]
    assert "harness hint" in first
    assert "import failed" in first
    assert context.metadata["error_hints"]["fired"] == ["module_not_found"]


async def test_no_such_file_gets_hint(tmp_path, monkeypatch):
    script = [_bash("c1", "cat /nope/does/not/exist.txt"), _bash("c2", _SUBMIT)]
    context, trajectory = await _run_agent(tmp_path, monkeypatch, script)
    assert "harness hint" in _tools(trajectory)[0]["content"]
    assert context.metadata["error_hints"]["fired"] == ["no_such_file"]


async def test_hint_fires_once_per_signature(tmp_path, monkeypatch):
    script = [_bash(f"c{i}", "still_not_a_command") for i in range(3)] + [_bash("c3", _SUBMIT)]
    context, trajectory = await _run_agent(tmp_path, monkeypatch, script)
    hinted = [m for m in _tools(trajectory) if "harness hint" in (m.get("content") or "")]
    assert len(hinted) == 1  # only the first of the three identical failures
    assert context.metadata["error_hints"]["fired"] == ["cmd_not_found"]


async def test_successful_command_gets_no_hint(tmp_path, monkeypatch):
    # stdout mentions "command not found" but the command SUCCEEDS -> no hint.
    script = [_bash("c1", "echo the phrase command not found appears here"), _bash("c2", _SUBMIT)]
    context, trajectory = await _run_agent(tmp_path, monkeypatch, script)
    assert "harness hint" not in _tools(trajectory)[0]["content"]
    assert context.metadata["error_hints"]["fired"] == []


async def test_disabled_flag(tmp_path, monkeypatch):
    script = [_bash("c1", "definitely_not_a_real_command"), _bash("c2", _SUBMIT)]
    context, trajectory = await _run_agent(tmp_path, monkeypatch, script, enable_error_hints=False)
    assert "harness hint" not in _tools(trajectory)[0]["content"]
    assert context.metadata["error_hints"]["fired"] == []
