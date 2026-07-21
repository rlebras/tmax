"""Prompt-only verification addendum tests for the prompt_self_verify arm:
the addendum is appended to the instance prompt (and only there), carries
the exit-code teaching, and disappears when disabled.

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


async def _run_agent(tmp_path, monkeypatch, **agent_kwargs):
    monkeypatch.setattr(agent_module, "_STATE_DIR", str(tmp_path / "vanillux2_state"))
    monkeypatch.setattr(
        litellm,
        "completion",
        _scripted_completion([_bash("c1", "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT")]),
    )
    agent = Vanillux2Agent(
        logs_dir=tmp_path / "logs", model_name="anthropic/claude-haiku-4-5", **agent_kwargs
    )
    env = FakeEnvironment()
    await agent.setup(env)
    context = types.SimpleNamespace(cost_usd=0, n_input_tokens=0, n_output_tokens=0, metadata=None)
    await agent.run("solve this task", env, context)
    return json.loads((tmp_path / "logs" / "trajectory.json").read_text())


async def test_addendum_appended_to_instance_prompt(tmp_path, monkeypatch):
    trajectory = await _run_agent(tmp_path, monkeypatch)
    instance_msg = trajectory[1]
    assert instance_msg["role"] == "user"
    assert "solve this task" in instance_msg["content"]  # original task kept
    assert "Verify before you submit" in instance_msg["content"]
    # The teaching that matters: exit-code semantics + independent oracles.
    assert "EXITS NON-ZERO" in instance_msg["content"]
    assert "cmd && echo PASS || echo FAIL" in instance_msg["content"]
    assert "independent oracle" in instance_msg["content"]
    assert "fix the SOLUTION, not the check" in instance_msg["content"]
    # Prompt-only: no gate/convention machinery is ever mentioned.
    assert "agent-check" not in instance_msg["content"]
    assert "declare_criteria" not in instance_msg["content"]
    # System prompt untouched.
    assert "Verify before you submit" not in trajectory[0]["content"]


async def test_addendum_absent_when_disabled(tmp_path, monkeypatch):
    trajectory = await _run_agent(tmp_path, monkeypatch, enable_self_verify_prompt=False)
    assert "Verify before you submit" not in trajectory[1]["content"]
