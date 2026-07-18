"""End-to-end wiring test: Vanillux2Agent.run()'s submit gate actually blocks
and then unblocks COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT, driven through the
real agent loop with a scripted fake model and a local-filesystem fake
environment (no docker, no live LLM). Unlike test_self_test.py (which drives
self_test.py's functions directly), this exercises the actual dispatch/gate
wiring added to Vanillux2Agent.run().

Imports the full ``Vanillux2Agent`` package (not the bare-module trick
tests/vanillux2/conftest.py uses for context_management/edit_tools/self_test)
since this needs ``agent.py`` itself, which requires harbor.
"""

from __future__ import annotations

import asyncio
import json
import shutil
import types
from dataclasses import dataclass
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


class FakeEnvironment:
    """Runs real bash locally; upload_file/download_file are plain local copies
    (mirrors tests/vanillux2/conftest.py's ops fixture, but shaped as harbor's
    BaseEnvironment interface: exec/upload_file/download_file)."""

    async def exec(self, command: str, timeout_sec: float | None = None) -> FakeExecResult:
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


@pytest.fixture(autouse=True)
def _clean_self_test_scratch():
    # self_test.py's isolation_root/state_dir default under /tmp/.vanillux2,
    # same fixed location the real agent uses in production (see
    # Vanillux2Agent/self_test.py's SelfTestConfig). Cleaned before/after so
    # this test doesn't accumulate stale scratch state across runs.
    shutil.rmtree("/tmp/.vanillux2", ignore_errors=True)
    yield
    shutil.rmtree("/tmp/.vanillux2", ignore_errors=True)


async def test_submit_gate_blocks_then_allows_end_to_end(tmp_path, monkeypatch):
    # Isolate the persistent-shell state file too, so this test doesn't
    # depend on / clobber whatever the real agent last left in /tmp.
    monkeypatch.setattr(agent_module, "_STATE_DIR", str(tmp_path / "vanillux2_state"))

    script = [
        _tool_call(
            "c1",
            "declare_criteria",
            {
                "criteria": [
                    {"id": "a", "description": "prints hi", "how_to_check": "run it"},
                    {"id": "b", "description": "prints bye", "how_to_check": "run it"},
                ],
                "deliverables": [],
            },
        ),
        _tool_call("c2", "run_check", {"criterion_id": "a", "command": "echo hi", "expected": "hi"}),
        _tool_call("c3", "bash", {"command": "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"}),
        _tool_call("c4", "run_check", {"criterion_id": "b", "command": "echo bye", "expected": "bye"}),
        _tool_call("c5", "bash", {"command": "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"}),
    ]
    monkeypatch.setattr(litellm, "completion", _scripted_completion(script))

    agent = Vanillux2Agent(logs_dir=tmp_path / "logs", model_name="anthropic/claude-haiku-4-5", max_steps=10)
    env = FakeEnvironment()
    await agent.setup(env)
    context = types.SimpleNamespace(cost_usd=0, n_input_tokens=0, n_output_tokens=0, metadata=None)

    await agent.run("solve it", env, context)

    st = context.metadata["self_test"]
    assert st["criteria_declared"] == 2
    assert st["checks_run"] == 2
    assert st["gate_rejections"] == 1  # the first submit attempt (1/2 covered) was rejected
    assert st["submitted_with_failing_checks"] is False

    trajectory = json.loads((tmp_path / "logs" / "trajectory.json").read_text())
    nudges = [m for m in trajectory if m.get("role") == "user" and "Submit rejected" in (m.get("content") or "")]
    assert len(nudges) == 1

    # The run actually stopped at the second submit, not by exhausting steps.
    tool_msgs = [m for m in trajectory if m.get("role") == "tool"]
    assert any("COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT" in (m.get("content") or "") for m in tool_msgs)


async def test_submit_gate_forces_through_after_max_rejections(tmp_path, monkeypatch):
    monkeypatch.setattr(agent_module, "_STATE_DIR", str(tmp_path / "vanillux2_state"))

    # No declare_criteria call at all — every submit attempt is rejected on
    # "no criteria declared" until max_gate_rejections is exhausted, at which
    # point the run must still terminate (not burn the whole step budget)
    # and record submitted_with_failing_checks=True.
    script = [_tool_call(f"c{i}", "bash", {"command": "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"}) for i in range(10)]
    monkeypatch.setattr(litellm, "completion", _scripted_completion(script))

    agent = Vanillux2Agent(
        logs_dir=tmp_path / "logs",
        model_name="anthropic/claude-haiku-4-5",
        max_steps=10,
        max_gate_rejections=2,
    )
    env = FakeEnvironment()
    await agent.setup(env)
    context = types.SimpleNamespace(cost_usd=0, n_input_tokens=0, n_output_tokens=0, metadata=None)

    await agent.run("solve it", env, context)

    st = context.metadata["self_test"]
    assert st["gate_rejections"] == 2
    assert st["submitted_with_failing_checks"] is True
