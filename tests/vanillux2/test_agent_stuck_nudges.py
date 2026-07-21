"""Thrash-detection tests for the stuck_loop_breaker arm: a repeated
identical command triggers a one-time change-approach nudge, a long streak
of failing commands triggers a step-back nudge, success resets the streak,
and total nudges per run are bounded.

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


async def _run_agent(tmp_path, monkeypatch, script, **agent_kwargs):
    monkeypatch.setattr(agent_module, "_STATE_DIR", str(tmp_path / "vanillux2_state"))
    monkeypatch.setattr(litellm, "completion", _scripted_completion(script))

    agent = Vanillux2Agent(
        logs_dir=tmp_path / "logs",
        model_name="anthropic/claude-haiku-4-5",
        max_steps=agent_kwargs.pop("max_steps", len(script)),
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


async def test_repeated_identical_command_triggers_nudge_once(tmp_path, monkeypatch):
    script = [_bash(f"c{i}", "grep -q flag /nonexistent") for i in range(3)] + [
        _bash("c3", "echo moved on")
    ]
    context, trajectory = await _run_agent(tmp_path, monkeypatch, script)

    nudges = _nudges(trajectory)
    assert len(nudges) == 1
    assert "SAME command 3 times" in nudges[0]["content"]
    assert context.metadata["stuck_nudges"]["sent"] == 1
    # The nudge lands right after the third identical command's tool result.
    third_tool_index = [i for i, m in enumerate(trajectory) if m.get("role") == "tool"][2]
    assert trajectory[third_tool_index + 1] == nudges[0]


async def test_failure_streak_triggers_step_back_nudge(tmp_path, monkeypatch):
    # Six DISTINCT failing commands — the repeat detector must not fire, the
    # failure-streak one must.
    script = [_bash(f"c{i}", f"exit {i + 1}") for i in range(6)] + [_bash("c6", "echo ok")]
    context, trajectory = await _run_agent(tmp_path, monkeypatch, script)

    nudges = _nudges(trajectory)
    assert len(nudges) == 1
    assert "6 commands all exited non-zero" in nudges[0]["content"]


async def test_success_resets_failure_streak(tmp_path, monkeypatch):
    # 5 failures, a success, 5 more failures: never 6 consecutive -> no nudge.
    script = (
        [_bash(f"a{i}", f"exit {i + 1}") for i in range(5)]
        + [_bash("ok", "echo fine")]
        + [_bash(f"b{i}", f"exit {i + 1}") for i in range(5)]
    )
    context, trajectory = await _run_agent(tmp_path, monkeypatch, script)
    assert _nudges(trajectory) == []
    assert context.metadata["stuck_nudges"]["sent"] == 0


async def test_total_nudges_are_bounded(tmp_path, monkeypatch):
    # Three separate bursts of the same failing command; cap at 1 nudge.
    script = []
    for burst in range(3):
        script += [_bash(f"c{burst}_{i}", f"burst{burst}; exit 1") for i in range(3)]
    context, trajectory = await _run_agent(
        tmp_path, monkeypatch, script, max_stuck_nudges=1, stuck_failure_streak=0
    )
    assert len(_nudges(trajectory)) == 1
    assert context.metadata["stuck_nudges"]["sent"] == 1


async def test_detectors_disabled_with_zero(tmp_path, monkeypatch):
    script = [_bash(f"c{i}", "exit 1") for i in range(8)]
    context, trajectory = await _run_agent(
        tmp_path, monkeypatch, script, stuck_repeat_threshold=0, stuck_failure_streak=0
    )
    assert _nudges(trajectory) == []
