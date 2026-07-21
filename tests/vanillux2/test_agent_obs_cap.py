"""Observation-cap tests for the obs_cap_tight arm: a long command output is
truncated to head+tail with a re-fetch hint at the configured caps, short
output is untouched, and setting the cap to 0 restores the vanillux default.

Scripted fake model + local-filesystem fake environment — no docker, no live
LLM.
"""

from __future__ import annotations

import asyncio
import json
import types
from dataclasses import dataclass
from pathlib import Path

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
        logs_dir=tmp_path / "logs", model_name="anthropic/claude-haiku-4-5",
        max_steps=len(script), **agent_kwargs,
    )
    env = FakeEnvironment()
    await agent.setup(env)
    context = types.SimpleNamespace(cost_usd=0, n_input_tokens=0, n_output_tokens=0, metadata=None)
    await agent.run("solve it", env, context)
    trajectory = json.loads((tmp_path / "logs" / "trajectory.json").read_text())
    return trajectory


def test_truncate_obs_unit():
    agent = Vanillux2Agent(logs_dir=Path("/tmp/unused"), obs_max_chars=100, obs_head_chars=30, obs_tail_chars=20)
    out = "A" * 30 + "B" * 500 + "C" * 20
    trunc = agent._truncate_obs(out)
    assert trunc.startswith("The output of your last command was too long")
    assert "A" * 30 in trunc and "C" * 20 in trunc
    assert "chars elided" in trunc
    # far shorter than the original 550 chars of body
    assert len(trunc) < 350


def test_truncate_obs_short_output_untouched():
    agent = Vanillux2Agent(logs_dir=Path("/tmp/unused"), obs_max_chars=100)
    assert agent._truncate_obs("short") == "short"


def test_truncate_obs_zero_uses_default():
    agent = Vanillux2Agent(logs_dir=Path("/tmp/unused"), obs_max_chars=0)
    # default vanillux cap is 10000 chars; 500 chars passes through unchanged
    assert agent._truncate_obs("x" * 500) == "x" * 500


async def test_long_command_output_truncated_in_trajectory(tmp_path, monkeypatch):
    big = 'python3 -c "print(\'Z\' * 20000)"'
    script = [_bash("c1", big), _bash("c2", "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT")]
    trajectory = await _run_agent(
        tmp_path, monkeypatch, script, obs_max_chars=3000, obs_head_chars=1500, obs_tail_chars=1500
    )
    tool_msgs = [m for m in trajectory if m.get("role") == "tool"]
    big_result = tool_msgs[0]["content"]
    assert "chars elided" in big_result
    assert len(big_result) < 4000  # nowhere near 20000
    assert "(exit_code=0)" in big_result


async def test_small_command_output_not_truncated(tmp_path, monkeypatch):
    script = [_bash("c1", "echo hello"), _bash("c2", "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT")]
    trajectory = await _run_agent(tmp_path, monkeypatch, script, obs_max_chars=3000)
    first = [m for m in trajectory if m.get("role") == "tool"][0]["content"]
    assert "hello" in first
    assert "elided" not in first
