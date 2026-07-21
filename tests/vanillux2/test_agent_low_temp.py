"""Low-temperature default tests for the low_temp arm: the agent's default
sampling params are lowered (temperature 0.2, top_p 0.9) and those values
actually flow into the litellm completion call.

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


def test_default_sampling_is_low_temp():
    agent = Vanillux2Agent(logs_dir=Path("/tmp/unused"))
    assert agent.temperature == 0.2
    assert agent.top_p == 0.9


async def test_sampling_params_flow_into_completion(tmp_path, monkeypatch):
    captured: list = []

    def fake_completion(**kwargs):
        captured.append(kwargs)
        return _FakeResponse(_bash("c1", "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"))

    monkeypatch.setattr(agent_module, "_STATE_DIR", str(tmp_path / "vanillux2_state"))
    monkeypatch.setattr(litellm, "completion", fake_completion)

    agent = Vanillux2Agent(logs_dir=tmp_path / "logs", model_name="hosted_vllm/tmax-9b")
    env = FakeEnvironment()
    await agent.setup(env)
    context = types.SimpleNamespace(cost_usd=0, n_input_tokens=0, n_output_tokens=0, metadata=None)
    await agent.run("solve it", env, context)

    assert captured[0]["temperature"] == 0.2
    # top_p is only suppressed for anthropic/ + temperature; a served vllm
    # model keeps it.
    assert captured[0]["top_p"] == 0.9


async def test_explicit_override_still_wins(tmp_path, monkeypatch):
    captured: list = []

    def fake_completion(**kwargs):
        captured.append(kwargs)
        return _FakeResponse(_bash("c1", "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"))

    monkeypatch.setattr(agent_module, "_STATE_DIR", str(tmp_path / "vanillux2_state"))
    monkeypatch.setattr(litellm, "completion", fake_completion)

    agent = Vanillux2Agent(
        logs_dir=tmp_path / "logs", model_name="hosted_vllm/tmax-9b", temperature=0.7, top_p=0.95
    )
    env = FakeEnvironment()
    await agent.setup(env)
    context = types.SimpleNamespace(cost_usd=0, n_input_tokens=0, n_output_tokens=0, metadata=None)
    await agent.run("solve it", env, context)

    assert captured[0]["temperature"] == 0.7
    assert captured[0]["top_p"] == 0.95
