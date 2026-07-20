"""Integration tests driving Vanillux2Agent.run() end to end against a
FakeEnvironment, scripting the model's turns directly (bypassing litellm) so
no real API/Docker access is needed.

Covers the deliverables from the self-test-gate task:
  - submit REJECTED with no passing isolated check, ACCEPTED once one exists
  - gate rejections are bounded -> forced submit, flagged submitted_with_failing_checks
  - disabling enable_self_test_gate restores baseline behavior exactly
  - the feature works on the baseline agent with no edit tool present
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import Vanillux2Agent.agent as agent_module
from Vanillux2Agent.agent import Vanillux2Agent
from harbor.models.agent.context import AgentContext


def bash_tool_call(command: str, call_id: str) -> list[dict]:
    return [
        {
            "id": call_id,
            "type": "function",
            "function": {"name": "bash", "arguments": json.dumps({"command": command})},
        }
    ]


class FakeMessage:
    def __init__(self, tool_calls=None, content: str = ""):
        self.tool_calls = tool_calls
        self.content = content

    def model_dump(self) -> dict:
        return {"role": "assistant", "content": self.content, "tool_calls": self.tool_calls}


class FakeChoice:
    def __init__(self, message: FakeMessage):
        self.message = message


class FakeResponse:
    def __init__(self, message: FakeMessage):
        self.choices = [FakeChoice(message)]
        self.usage = None


class ScriptedModel:
    """Feeds a fixed sequence of bash commands to the agent loop, one per step."""

    def __init__(self, commands: list[str]):
        self._commands = list(commands)
        self.calls = 0

    async def __call__(self, model: str, messages: list[dict]) -> FakeResponse:
        self.calls += 1
        if not self._commands:
            raise AssertionError("ScriptedModel exhausted — agent requested more turns than scripted")
        command = self._commands.pop(0)
        return FakeResponse(FakeMessage(tool_calls=bash_tool_call(command, f"call_{self.calls}")))


def make_agent(tmp_path: Path, monkeypatch, **kwargs) -> Vanillux2Agent:
    monkeypatch.setattr(agent_module, "_STATE_DIR", str(tmp_path / ".vanillux2"))
    return Vanillux2Agent(
        logs_dir=tmp_path / "logs",
        model_name="fake/model",
        max_steps=kwargs.pop("max_steps", 10),
        **kwargs,
    )


async def run_agent(agent: Vanillux2Agent, environment, commands: list[str]) -> ScriptedModel:
    scripted = ScriptedModel(commands)
    agent._query_with_retry = scripted
    await agent.setup(environment)
    await agent.run("solve the task", environment, AgentContext())
    return scripted


def read_self_test_state(agent: Vanillux2Agent) -> dict:
    return json.loads((agent.logs_dir / "self_test_state.json").read_text())


def read_timing(agent: Vanillux2Agent) -> list[dict]:
    return json.loads((agent.logs_dir / "timing.json").read_text())


# ---------------------------------------------------------------------------
# Submit rejected -> accepted once a check passes
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_submit_rejected_then_accepted(tmp_path, monkeypatch, fake_env):
    agent = make_agent(tmp_path, monkeypatch, enable_self_test_gate=True, min_criteria=1)
    criteria_path = agent.self_test_config.criteria_path

    declare = (
        f"mkdir -p {Path(criteria_path).parent} && cat > {criteria_path} <<'EOF'\n"
        '{"criteria": [{"id": "c1", "description": "prints 42", "how_to_check": "run solution.py"}], '
        '"deliverables": ["solution.py"]}\n'
        "EOF\n"
        "cat > solution.py <<'EOF'\nprint(42)\nEOF"
    )
    early_submit = "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"
    check = 'agent-check c1 -- test "$(python3 solution.py)" = "42"'
    late_submit = "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"

    scripted = await run_agent(agent, fake_env, [declare, early_submit, check, late_submit])
    assert scripted.calls == 4  # nothing extra consumed once the gate finally allows

    timing = read_timing(agent)
    assert any(t.get("gate_rejected") for t in timing)
    assert any(t.get("agent_check") == "c1" for t in timing)

    state = read_self_test_state(agent)
    assert state["gate_rejections"] == 1
    assert state["submitted_with_failing_checks"] is False
    assert state["checks"]["c1"][0]["isolated_pass"] is True

    trajectory = json.loads((agent.logs_dir / "trajectory.json").read_text())
    assert "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT" in trajectory[-1]["content"]


@pytest.mark.asyncio
async def test_submit_rejected_nudge_lists_unmet_criteria(tmp_path, monkeypatch, fake_env):
    agent = make_agent(
        tmp_path, monkeypatch, enable_self_test_gate=True, min_criteria=1, max_gate_rejections=5, max_steps=1
    )
    scripted = await run_agent(agent, fake_env, ["echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"])
    assert scripted.calls == 1

    trajectory = json.loads((agent.logs_dir / "trajectory.json").read_text())
    nudge = trajectory[-1]["content"]
    assert "Submit rejected" in nudge
    assert "no acceptance criteria declared" in nudge
    # the loop must NOT have finished (no break) — trajectory has no submit marker anywhere
    assert not any("COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT" in (m.get("content") or "") for m in trajectory[2:] if m["role"] != "tool" or "Submit rejected" not in m["content"])


# ---------------------------------------------------------------------------
# Gate rejections are bounded
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_gate_rejections_bounded_then_forced_submit(tmp_path, monkeypatch, fake_env):
    agent = make_agent(
        tmp_path, monkeypatch, enable_self_test_gate=True, min_criteria=5, max_gate_rejections=1
    )
    submit = "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"
    scripted = await run_agent(agent, fake_env, [submit, submit])
    assert scripted.calls == 2  # 1st rejected, 2nd forced through — no 3rd turn needed

    state = read_self_test_state(agent)
    assert state["gate_rejections"] == 1
    assert state["submitted_with_failing_checks"] is True

    trajectory = json.loads((agent.logs_dir / "trajectory.json").read_text())
    assert "(exit_code=0)" in trajectory[-1]["content"]
    assert "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT" in trajectory[-1]["content"]


# ---------------------------------------------------------------------------
# Disabling the flag restores baseline behavior exactly
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_disabled_gate_matches_baseline_behavior(tmp_path, monkeypatch, fake_env):
    agent = make_agent(tmp_path, monkeypatch, enable_self_test_gate=False)
    submit = "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"
    scripted = await run_agent(agent, fake_env, [submit])
    assert scripted.calls == 1  # submit honored immediately, no gate interception

    assert not (agent.logs_dir / "self_test_state.json").exists()

    trajectory = json.loads((agent.logs_dir / "trajectory.json").read_text())
    # system, user, assistant, tool = exactly the baseline 4 messages for a 1-turn submit
    assert len(trajectory) == 4
    assert "Self-testing" not in trajectory[1]["content"]  # no addendum injected
    assert "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT" in trajectory[-1]["content"]


@pytest.mark.asyncio
async def test_agent_check_syntax_is_inert_when_gate_disabled(tmp_path, monkeypatch, fake_env):
    """With the gate off, `agent-check ...` is just an ordinary (failing) bash
    command — proves the interception adds no behavior unless explicitly enabled.
    """
    agent = make_agent(tmp_path, monkeypatch, enable_self_test_gate=False)
    scripted = await run_agent(
        agent,
        fake_env,
        ["agent-check c1 -- true", "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"],
    )
    assert scripted.calls == 2
    trajectory = json.loads((agent.logs_dir / "trajectory.json").read_text())
    # the literal "agent-check" command was sent to the shell (and failed: no such binary)
    first_tool_msg = trajectory[3]
    assert first_tool_msg["role"] == "tool"
    assert "exit_code=0" not in first_tool_msg["content"]


# ---------------------------------------------------------------------------
# Robustness fixes: bundled agent-check, malformed syntax, implicit submit
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_bundled_agent_check_runs_setup_then_check(tmp_path, monkeypatch, fake_env):
    """agent-check bundled with setup on other lines: the setup still runs
    (in-session), and the check is still intercepted rather than leaking to
    the real shell as 'command not found'.
    """
    agent = make_agent(tmp_path, monkeypatch, enable_self_test_gate=True, min_criteria=1, max_steps=2)
    criteria_path = agent.self_test_config.criteria_path
    declare = (
        f"mkdir -p {Path(criteria_path).parent} && cat > {criteria_path} <<'EOF'\n"
        '{"criteria": [{"id": "c1", "description": "prints 42", "how_to_check": "run it"}], '
        '"deliverables": ["solution.py"]}\n'
        "EOF"
    )
    bundled = "cat > solution.py << 'EOF'\nprint(42)\nEOF\nagent-check c1 -- test \"$(python3 solution.py)\" = \"42\""

    scripted = await run_agent(agent, fake_env, [declare, bundled])
    assert scripted.calls == 2

    trajectory = json.loads((agent.logs_dir / "trajectory.json").read_text())
    last_tool_msg = trajectory[-1]["content"]
    assert "agent-check c1: PASS (isolated)" in last_tool_msg
    assert "command not found" not in last_tool_msg

    state = read_self_test_state(agent)
    assert state["checks"]["c1"][0]["isolated_pass"] is True


@pytest.mark.asyncio
async def test_malformed_agent_check_returns_hint_without_hitting_shell(tmp_path, monkeypatch, fake_env):
    agent = make_agent(tmp_path, monkeypatch, enable_self_test_gate=True, min_criteria=1, max_steps=1)
    scripted = await run_agent(agent, fake_env, ["agent-check c1 test -f out.txt"])  # missing ` -- `
    assert scripted.calls == 1

    trajectory = json.loads((agent.logs_dir / "trajectory.json").read_text())
    last_tool_msg = trajectory[-1]["content"]
    assert "syntax not recognized" in last_tool_msg
    assert "command not found" not in last_tool_msg

    timing = read_timing(agent)
    assert timing[-1].get("agent_check_malformed") is True


@pytest.mark.asyncio
async def test_implicit_submit_marker_is_gated(tmp_path, monkeypatch, fake_env):
    """If a command's OUTPUT happens to contain the submit marker text (not
    the model's requested command itself), that must still be gated — the
    run should NOT finish, and the nudge should be appended rather than
    replacing the command's real output.
    """
    agent = make_agent(
        tmp_path, monkeypatch, enable_self_test_gate=True, min_criteria=1, max_gate_rejections=5, max_steps=1
    )
    # Built via string concatenation so the literal marker substring is NOT
    # present in the requested command text (action type stays "command",
    # not "done") — only the runtime OUTPUT contains it.
    leak = "python3 -c \"print('COMPLETE_TASK' + '_AND_SUBMIT_FINAL_OUTPUT')\""
    scripted = await run_agent(agent, fake_env, [leak])
    assert scripted.calls == 1  # loop did NOT break — ran out of scripted turns instead

    trajectory = json.loads((agent.logs_dir / "trajectory.json").read_text())
    last_tool_msg = trajectory[-1]["content"]
    assert "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT" in last_tool_msg  # original output preserved
    assert "does not finish the task" in last_tool_msg
    assert "Submit rejected" in last_tool_msg

    state = read_self_test_state(agent)
    assert state["gate_rejections"] == 1


@pytest.mark.asyncio
async def test_implicit_submit_marker_allowed_when_gate_satisfied(tmp_path, monkeypatch, fake_env):
    agent = make_agent(tmp_path, monkeypatch, enable_self_test_gate=False, max_steps=1)
    leak = "python3 -c \"print('COMPLETE_TASK' + '_AND_SUBMIT_FINAL_OUTPUT')\""
    scripted = await run_agent(agent, fake_env, [leak])
    assert scripted.calls == 1

    trajectory = json.loads((agent.logs_dir / "trajectory.json").read_text())
    # with the gate disabled, the pre-existing (baseline) ungated behavior
    # is unchanged: the marker in output ends the run as before.
    assert "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT" in trajectory[-1]["content"]


# ---------------------------------------------------------------------------
# Standalone: no edit tool present, no new model-facing tools
# ---------------------------------------------------------------------------


def test_no_edit_tool_module_present():
    with pytest.raises(ModuleNotFoundError):
        import Vanillux2Agent.edit_tools  # noqa: F401


def test_no_new_tool_schemas_added():
    from rl_data.generator.sample_solutions import TOOL_SCHEMAS

    names = {t["function"]["name"] for t in TOOL_SCHEMAS}
    assert names == {"bash"}
