"""Task re-anchoring tests for the task_reanchor arm: a compact verbatim
excerpt of the task statement is re-injected every reanchor_every steps,
long statements are excerpted, and the feature can be disabled.

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


async def _run_agent(tmp_path, monkeypatch, n_steps, instruction="build the widget in /app/widget.py", **agent_kwargs):
    monkeypatch.setattr(agent_module, "_STATE_DIR", str(tmp_path / "vanillux2_state"))
    script = [_bash(f"c{i}", "echo work") for i in range(n_steps)]
    monkeypatch.setattr(litellm, "completion", _scripted_completion(script))

    agent = Vanillux2Agent(
        logs_dir=tmp_path / "logs",
        model_name="anthropic/claude-haiku-4-5",
        max_steps=n_steps,
        **agent_kwargs,
    )
    env = FakeEnvironment()
    await agent.setup(env)
    context = types.SimpleNamespace(cost_usd=0, n_input_tokens=0, n_output_tokens=0, metadata=None)
    await agent.run(instruction, env, context)
    trajectory = json.loads((tmp_path / "logs" / "trajectory.json").read_text())
    return context, trajectory


def _reanchors(trajectory):
    return [
        m for m in trajectory
        if m.get("role") == "user" and "TASK STATEMENT" in (m.get("content") or "")
    ]


async def test_reanchor_fires_on_schedule_with_verbatim_excerpt(tmp_path, monkeypatch):
    context, trajectory = await _run_agent(tmp_path, monkeypatch, n_steps=9, reanchor_every=4)

    reanchors = _reanchors(trajectory)
    assert len(reanchors) == 2  # before steps 5 and 9 (after 4 and 8 completed)
    assert "Reminder after 4 steps" in reanchors[0]["content"]
    assert "Reminder after 8 steps" in reanchors[1]["content"]
    assert "build the widget in /app/widget.py" in reanchors[0]["content"]
    assert context.metadata["task_reanchor"]["sent"] == 2
    # Placement: the first reminder comes right after the 4th tool result.
    fourth_tool = [i for i, m in enumerate(trajectory) if m.get("role") == "tool"][3]
    assert trajectory[fourth_tool + 1] == reanchors[0]


async def test_long_statement_is_excerpted(tmp_path, monkeypatch):
    long_task = "requirement\n" * 2000
    context, trajectory = await _run_agent(
        tmp_path, monkeypatch, n_steps=5, instruction=long_task,
        reanchor_every=4, reanchor_excerpt_chars=300,
    )
    reanchors = _reanchors(trajectory)
    assert len(reanchors) == 1
    assert "excerpt truncated" in reanchors[0]["content"]
    assert len(reanchors[0]["content"]) < 1000


async def test_disabled_with_zero(tmp_path, monkeypatch):
    context, trajectory = await _run_agent(tmp_path, monkeypatch, n_steps=9, reanchor_every=0)
    assert _reanchors(trajectory) == []
    assert context.metadata["task_reanchor"]["sent"] == 0
