"""Pre-submit reflection tests for the submit_reflection arm: the first
submit attempt is intercepted with a task re-read prompt (the sentinel never
runs), the second goes through, implicit marker output is unaffected, and
the feature can be disabled.

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


async def _run_agent(tmp_path, monkeypatch, script, instruction="solve the puzzle in /app", **agent_kwargs):
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
    await agent.run(instruction, env, context)
    trajectory = json.loads((tmp_path / "logs" / "trajectory.json").read_text())
    return context, trajectory


def _tool_messages(trajectory):
    return [m for m in trajectory if m.get("role") == "tool"]


async def test_first_submit_reflects_second_goes_through(tmp_path, monkeypatch):
    script = [_bash("c1", "echo working"), _bash("c2", _SUBMIT), _bash("c3", _SUBMIT)]
    context, trajectory = await _run_agent(tmp_path, monkeypatch, script)

    tool_msgs = _tool_messages(trajectory)
    reflections = [m for m in tool_msgs if "TASK STATEMENT" in (m.get("content") or "")]
    assert len(reflections) == 1
    # The intercepted submit never ran: its tool result is the reflection,
    # and the task text is re-shown verbatim.
    assert "solve the puzzle in /app" in reflections[0]["content"]
    assert "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT" not in reflections[0]["content"]
    # Only the second submit's marker appears, and the run finished on it.
    markers = [m for m in tool_msgs if "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT" in (m.get("content") or "")]
    assert len(markers) == 1
    assert len([m for m in trajectory if m.get("role") == "assistant"]) == 3
    assert context.metadata["submit_reflection"]["triggered"] is True


async def test_reflection_fires_at_most_once(tmp_path, monkeypatch):
    script = [_bash("c1", _SUBMIT), _bash("c2", _SUBMIT)]
    context, trajectory = await _run_agent(tmp_path, monkeypatch, script)
    assert len([m for m in _tool_messages(trajectory) if "TASK STATEMENT" in (m.get("content") or "")]) == 1
    assert len([m for m in trajectory if m.get("role") == "assistant"]) == 2


async def test_long_task_statement_is_excerpted(tmp_path, monkeypatch):
    long_task = "requirement line\n" * 1000  # ~17k chars
    script = [_bash("c1", _SUBMIT), _bash("c2", _SUBMIT)]
    _, trajectory = await _run_agent(
        tmp_path, monkeypatch, script, instruction=long_task, reflection_task_excerpt_chars=500
    )
    reflection = next(m for m in _tool_messages(trajectory) if "TASK STATEMENT" in m["content"])
    assert "task statement truncated" in reflection["content"]
    assert len(reflection["content"]) < 1500


async def test_disabled_flag_lets_first_submit_through(tmp_path, monkeypatch):
    script = [_bash("c1", _SUBMIT)]
    context, trajectory = await _run_agent(
        tmp_path, monkeypatch, script, enable_submit_reflection=False
    )
    assert len([m for m in trajectory if m.get("role") == "assistant"]) == 1
    assert context.metadata["submit_reflection"]["triggered"] is False
