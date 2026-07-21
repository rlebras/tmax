"""Step-budget nudge tests for the steps_budget_nudges arm: a one-time
prioritize warning fires when steps_warning_early steps remain, a
finalize-now warning at steps_warning_final, neither fires when disabled or
when the run submits before reaching them.

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
        logs_dir=tmp_path / "logs",
        model_name="anthropic/claude-haiku-4-5",
        **agent_kwargs,
    )
    env = FakeEnvironment()
    await agent.setup(env)
    context = types.SimpleNamespace(cost_usd=0, n_input_tokens=0, n_output_tokens=0, metadata=None)
    await agent.run("solve it", env, context)
    trajectory = json.loads((tmp_path / "logs" / "trajectory.json").read_text())
    return context, trajectory


def _nudges(trajectory):
    return [
        m for m in trajectory
        if m.get("role") == "user" and (m.get("content") or "").startswith("(harness)")
    ]


async def test_early_and_final_warnings_fire_once_at_right_steps(tmp_path, monkeypatch):
    script = [_bash(f"c{i}", "echo work") for i in range(6)]
    context, trajectory = await _run_agent(
        tmp_path, monkeypatch, script,
        max_steps=6, steps_warning_early=4, steps_warning_final=2,
    )

    nudges = _nudges(trajectory)
    assert len(nudges) == 2
    assert "Only 4 steps remain" in nudges[0]["content"]
    assert "FINAL WARNING: 2 steps left" in nudges[1]["content"]
    assert "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT" in nudges[1]["content"]
    assert context.metadata["steps_nudges"]["fired_at_remaining"] == [4, 2]

    # The early warning precedes the assistant turn of step 3 (remaining=4):
    # 2 turns done -> system, user, (a,t)x2, nudge, ...
    assert trajectory[6] == nudges[0]


async def test_no_warnings_when_run_submits_early(tmp_path, monkeypatch):
    script = [_bash("c1", _SUBMIT)]
    context, trajectory = await _run_agent(
        tmp_path, monkeypatch, script,
        max_steps=10, steps_warning_early=4, steps_warning_final=2,
    )
    assert _nudges(trajectory) == []
    assert context.metadata["steps_nudges"]["fired_at_remaining"] == []


async def test_warnings_disabled_with_zero(tmp_path, monkeypatch):
    script = [_bash(f"c{i}", "echo work") for i in range(5)]
    context, trajectory = await _run_agent(
        tmp_path, monkeypatch, script,
        max_steps=5, steps_warning_early=0, steps_warning_final=0,
    )
    assert _nudges(trajectory) == []


async def test_final_warning_only_when_early_disabled(tmp_path, monkeypatch):
    script = [_bash(f"c{i}", "echo work") for i in range(4)]
    context, trajectory = await _run_agent(
        tmp_path, monkeypatch, script,
        max_steps=4, steps_warning_early=0, steps_warning_final=2,
    )
    nudges = _nudges(trajectory)
    assert len(nudges) == 1
    assert "FINAL WARNING" in nudges[0]["content"]
