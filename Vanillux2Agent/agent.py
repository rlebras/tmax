"""Vanillux2Agent - direct LiteLLM agent using the vanillux prompt harness.

This is the Harbor-agent version of ``rl_data.generator.vanillux_solver``:
it uses the same mini-SWE-agent-derived prompts, bash tool schema, submit
marker, format-error recovery, and output truncation, but executes commands
through Harbor's active environment and calls the model directly with LiteLLM.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import litellm
from harbor.agents.base import BaseAgent
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext

from rl_data.generator.sample_solutions import (
    SUBMIT_MARKER,
    TOOL_SCHEMAS,
    _extract_tool_call,
)
from rl_data.generator.vanillux_solver import (
    _format_error_message,
    _render_instance,
    _SYSTEM_TEMPLATE,
    _truncate_observation,
)
from Vanillux2Agent import self_test as st

os.environ.setdefault("OPENAI_API_KEY", "dummy")

logger = logging.getLogger(__name__)

ABORT_EXCEPTIONS = (
    litellm.exceptions.AuthenticationError,
    litellm.exceptions.NotFoundError,
    litellm.exceptions.ContextWindowExceededError,
    litellm.exceptions.UnsupportedParamsError,
    litellm.exceptions.PermissionDeniedError,
)

MAX_RETRIES = 5
RETRY_BASE_DELAY = 2.0
LLM_TIMEOUT_SECONDS = 5 * 60 * 60
LLM_OUTER_TIMEOUT_BUFFER_SECONDS = 30
_STATE_DIR = "/tmp/.vanillux2"
_COMPOSE_PROVIDER_RE = re.compile(
    r"\x1b\[4m>>>> Executing external compose provider "
    r'"[^"]*docker-compose"\. Please see podman-compose\(1\) for how to disable '
    r"this message\. <<<<\n\n\x1b\[0m"
)
_DOCKER_EXEC_ERROR_RE = re.compile(
    r"(?ms)^Error: executing [^\n]*(?:docker-compose|docker compose)"
    r".*?: exit status \d+\s*$"
)

_SELF_TEST_PROMPT_ADDENDUM = """
## Self-testing (required before submit)

Verify your work against the task's own acceptance criteria before submitting:

1. Early on, write testable criteria (turn any example input/output into one)
   as JSON to `{criteria_path}`:
   `{{"criteria": [{{"id": "short_id", "description": "...", "how_to_check": "..."}}], "deliverables": ["path/to/solution/file", ...]}}`
   `deliverables` lists your solution file(s) — only these are copied into
   each isolated check; scratch files/helpers won't be there.
2. Verify each criterion with `agent-check <id> -- <command>` — a harness
   convention, not a real binary, so don't bother checking with `which`/
   `type`/`command -v` first (it will correctly report "not found" — that's
   expected, just run it). It must be on its own LINE (it can share a bash
   call with setup on OTHER lines, but not chained with `&&`/`;` on the SAME
   line). It passes iff `<command>` EXITS non-zero on failure — nothing else
   is checked, so a command that can't fail is worthless regardless of what
   it looks like:
     BAD:  `python3 -c "print(a == b)"` (prints True/False, always exits 0)
     BAD:  `cmd && echo PASS || echo FAIL` (`echo` never fails, always exits 0)
     GOOD: `python3 -c "assert a == b"` — or `test "$a" = "$b"`
   Use an independent oracle, not a comparison of the program to itself:
     - known-answer: `test "$(./solve input.txt)" = "42"` (a fixed value,
       e.g. from the task's own example I/O)
     - differential: `diff <(./solve.sh) <(python3 reference.py)` (two
       independently-derived results)
     - property/invariant: something that must structurally hold (round
       trip, idempotence) rather than one hard-coded value
   It also re-runs `<command>` against an ISOLATED copy of your
   deliverables; only that result counts, so leftover files or a weakened
   deliverable won't fake a pass.
3. Submit needs >= {min_criteria} criteria each with a passing `agent-check`,
   or it's rejected with a compact list of what's unmet.
"""


def _self_test_prompt_addendum(config: "st.SelfTestConfig") -> str:
    return _SELF_TEST_PROMPT_ADDENDUM.format(
        criteria_path=config.criteria_path, min_criteria=config.min_criteria
    )


class Vanillux2Agent(BaseAgent):
    """Bash-tool Harbor agent with vanillux prompts and direct API calls."""

    @staticmethod
    def name() -> str:
        return "vanillux2-agent"

    def version(self) -> str:
        return "0.1.0"

    def __init__(
        self,
        logs_dir: Path,
        model_name: str | None = None,
        max_steps: int = 64,
        temperature: float = 0.7,
        top_p: float | None = 0.95,
        top_k: int | None = None,
        max_tokens: int = 16384,
        cost_limit: float = 0.0,
        api_base: str | None = None,
        command_timeout: int = 120,
        persistent_bash: bool = True,
        max_format_errors: int = 64,
        enable_self_test_gate: bool = False,
        min_criteria: int = 2,
        max_gate_rejections: int = 3,
        self_test_check_timeout: int = 60,
        **kwargs: Any,
    ) -> None:
        super().__init__(logs_dir=logs_dir, model_name=model_name, **kwargs)
        self.max_steps = max_steps
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.max_tokens = max_tokens
        self.cost_limit = cost_limit
        self.api_base = api_base
        self.command_timeout = command_timeout
        self.persistent_bash = persistent_bash
        self.max_format_errors = max_format_errors
        self.enable_self_test_gate = enable_self_test_gate
        self.self_test_config = st.SelfTestConfig(
            min_criteria=min_criteria,
            max_gate_rejections=max_gate_rejections,
            check_timeout_sec=self_test_check_timeout,
            state_dir=f"{_STATE_DIR}/{st.SELF_TEST_STATE_SUBDIR}",
        )
        self.cost: float = 0.0

    async def setup(self, environment: BaseEnvironment) -> None:
        if not self.persistent_bash:
            return
        await environment.exec(
            command=(
                f"mkdir -p {_STATE_DIR} && "
                f"pwd > {_STATE_DIR}/cwd && "
                f"export -p > {_STATE_DIR}/env"
            ),
            timeout_sec=10,
        )

    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        model = self.model_name or "anthropic/claude-haiku-4-5"
        instance_message = _render_instance(instruction.strip())
        if self.enable_self_test_gate:
            instance_message += _self_test_prompt_addendum(self.self_test_config)
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": _SYSTEM_TEMPLATE},
            {"role": "user", "content": instance_message},
        ]

        timing_log: list[dict[str, Any]] = []
        usage_totals = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "reasoning_tokens": 0,
        }
        format_errors = 0
        self_test_state = st.SelfTestState()

        async def session_exec(command: str) -> Any:
            return await self._execute_bash(command, environment)

        async def isolated_exec(command: str, cwd: str) -> Any:
            return await environment.exec(
                command=command, cwd=cwd, timeout_sec=self.self_test_config.check_timeout_sec
            )

        try:
            for step in range(self.max_steps):
                logger.info("Step %s/%s", step + 1, self.max_steps)

                if self.cost_limit > 0 and self.cost >= self.cost_limit:
                    logger.warning(
                        "Cost limit reached: $%.2f >= $%.2f",
                        self.cost,
                        self.cost_limit,
                    )
                    break

                t0 = time.monotonic()
                try:
                    response = await self._query_with_retry(model, messages)
                except litellm.exceptions.ContextWindowExceededError:
                    logger.warning("Context window exceeded; stopping current run")
                    break
                llm_time = time.monotonic() - t0

                self._accumulate_usage(response, usage_totals)
                try:
                    self.cost += litellm.completion_cost(response, model=model)
                except Exception:
                    pass

                msg = response.choices[0].message.model_dump()
                action = _extract_tool_call(msg)
                if action["type"] == "no_tool_call":
                    msg.pop("tool_calls", None)
                    msg["content"] = msg.get("content") or ""
                    messages.append(msg)
                    format_errors += 1
                    self._append_format_error(messages, action.get("tool_call_id"))
                    timing_log.append(
                        {
                            "step": step + 1,
                            "llm_s": round(llm_time, 1),
                            "format_error": True,
                        }
                    )
                    if format_errors >= self.max_format_errors:
                        logger.warning("Stopping after %s format errors", format_errors)
                        break
                    continue

                messages.append(msg)
                format_errors = 0
                command = action.get("command") or ""
                tool_call_id = action.get("tool_call_id") or ""

                checks: list[tuple[str, str]] = []
                remainder = ""
                if self.enable_self_test_gate:
                    whole = st.parse_agent_check(command)
                    if whole is not None:
                        checks = [whole]
                    else:
                        found = st.extract_agent_check_lines(command)
                        if found is not None:
                            checks, remainder = found

                if checks:
                    t1 = time.monotonic()
                    prefix = ""
                    if remainder:
                        if st.looks_like_malformed_agent_check(remainder):
                            # e.g. a well-formed check alongside a botched one
                            # on another line — don't let the botched one hit
                            # the real shell as "command not found".
                            prefix = st.AGENT_CHECK_SYNTAX_HINT + "\n\n"
                        else:
                            pre_result = await self._execute_bash(remainder, environment)
                            prefix = self._format_tool_result(pre_result) + "\n\n"
                    check_outputs = [
                        await self._run_agent_check(
                            criterion_id, inner_command, self_test_state, session_exec, isolated_exec, step + 1
                        )
                        for criterion_id, inner_command in checks
                    ]
                    tool_content = prefix + "\n\n".join(check_outputs)
                    exec_time = time.monotonic() - t1
                    messages.append(
                        {"role": "tool", "tool_call_id": tool_call_id, "content": tool_content}
                    )
                    timing_log.append(
                        {
                            "step": step + 1,
                            "llm_s": round(llm_time, 1),
                            "bash_s": round(exec_time, 1),
                            "agent_check": [cid for cid, _ in checks],
                        }
                    )
                    continue

                if self.enable_self_test_gate and st.looks_like_malformed_agent_check(command):
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call_id,
                            "content": st.AGENT_CHECK_SYNTAX_HINT,
                        }
                    )
                    timing_log.append(
                        {
                            "step": step + 1,
                            "llm_s": round(llm_time, 1),
                            "agent_check_malformed": True,
                        }
                    )
                    continue

                if self.enable_self_test_gate and st.looks_like_existence_probe(command):
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call_id,
                            "content": st.AGENT_CHECK_EXISTENCE_HINT,
                        }
                    )
                    timing_log.append(
                        {
                            "step": step + 1,
                            "llm_s": round(llm_time, 1),
                            "agent_check_existence_probe": True,
                        }
                    )
                    continue

                is_explicit_submit = action["type"] == "done"
                if is_explicit_submit and self.enable_self_test_gate:
                    criteria, _deliverables = await st.load_criteria(session_exec, self.self_test_config)
                    gate = st.evaluate_gate(criteria, self_test_state, self.self_test_config)
                    if not gate.allowed:
                        self_test_state.gate_rejections += 1
                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": tool_call_id,
                                "content": st.gate_nudge(gate),
                            }
                        )
                        timing_log.append(
                            {
                                "step": step + 1,
                                "llm_s": round(llm_time, 1),
                                "gate_rejected": True,
                            }
                        )
                        continue
                    if gate.forced:
                        self_test_state.submitted_with_failing_checks = True

                t1 = time.monotonic()
                result = await self._execute_bash(command, environment)
                exec_time = time.monotonic() - t1

                tool_content = self._format_tool_result(result)

                # A command can trip the submit sentinel by coincidence (its
                # OUTPUT happens to contain the marker text) rather than by
                # the model actually requesting the sentinel command — gate
                # that path too, since it otherwise bypasses the gate
                # entirely. Unlike the explicit path, the command already
                # ran, so a rejection appends the nudge rather than replacing
                # the (already-produced) output, and does not break the loop.
                is_implicit_submit = (not is_explicit_submit) and SUBMIT_MARKER in tool_content
                if is_implicit_submit and self.enable_self_test_gate:
                    criteria, _deliverables = await st.load_criteria(session_exec, self.self_test_config)
                    gate = st.evaluate_gate(criteria, self_test_state, self.self_test_config)
                    if not gate.allowed:
                        self_test_state.gate_rejections += 1
                        tool_content += (
                            "\n\n(NOTE: this output happened to contain the submit marker text, "
                            "but that alone does not finish the task.) " + st.gate_nudge(gate)
                        )
                        is_implicit_submit = False
                    elif gate.forced:
                        self_test_state.submitted_with_failing_checks = True

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call_id,
                        "content": tool_content,
                    }
                )

                timing_log.append(
                    {
                        "step": step + 1,
                        "llm_s": round(llm_time, 1),
                        "bash_s": round(exec_time, 1),
                        "return_code": result.return_code,
                        "cmd": command[:200],
                    }
                )

                if is_explicit_submit or is_implicit_submit:
                    break
        finally:
            self.logs_dir.mkdir(parents=True, exist_ok=True)
            (self.logs_dir / "trajectory.json").write_text(
                json.dumps(messages, indent=2, default=str) + "\n"
            )
            (self.logs_dir / "timing.json").write_text(
                json.dumps(timing_log, indent=2) + "\n"
            )
            (self.logs_dir / "usage.json").write_text(
                json.dumps(
                    {
                        **usage_totals,
                        "cost_usd": self.cost,
                        "max_steps": self.max_steps,
                    },
                    indent=2,
                )
                + "\n"
            )
            context.cost_usd = self.cost
            context.n_input_tokens = usage_totals["prompt_tokens"]
            context.n_output_tokens = usage_totals["completion_tokens"]
            if self.enable_self_test_gate:
                (self.logs_dir / "self_test_state.json").write_text(
                    json.dumps(
                        {
                            "checks": {
                                cid: [asdict(r) for r in records]
                                for cid, records in self_test_state.checks.items()
                            },
                            "gate_rejections": self_test_state.gate_rejections,
                            "submitted_with_failing_checks": self_test_state.submitted_with_failing_checks,
                        },
                        indent=2,
                    )
                    + "\n"
                )

    async def _run_agent_check(
        self,
        criterion_id: str,
        command: str,
        state: "st.SelfTestState",
        session_exec: Any,
        isolated_exec: Any,
        step: int,
    ) -> str:
        criteria, deliverables = await st.load_criteria(session_exec, self.self_test_config)
        cwd_result = await session_exec("pwd")
        persistent_cwd = (getattr(cwd_result, "stdout", None) or "/").strip() or "/"
        return await st.run_agent_check(
            criterion_id,
            command,
            criteria=criteria,
            deliverables=deliverables,
            state=state,
            config=self.self_test_config,
            session_exec=session_exec,
            isolated_exec=isolated_exec,
            persistent_cwd=persistent_cwd,
            step=step,
        )

    async def _query_with_retry(
        self, model: str, messages: list[dict[str, Any]]
    ) -> Any:
        api_base = (
            self.api_base
            or os.environ.get("OPENAI_BASE_URL")
            or os.environ.get("OPENAI_API_BASE")
        )
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                llm_timeout = LLM_TIMEOUT_SECONDS
                completion_kwargs: dict[str, Any] = {
                    "model": model,
                    "messages": messages,
                    "tools": TOOL_SCHEMAS,
                    "max_tokens": self.max_tokens,
                    "api_base": api_base,
                    "timeout": llm_timeout,
                    "request_timeout": llm_timeout,
                    "num_retries": 0,
                }
                if self.temperature is not None:
                    completion_kwargs["temperature"] = self.temperature
                if self.top_p is not None and not (
                    model.startswith("anthropic/") and self.temperature is not None
                ):
                    completion_kwargs["top_p"] = self.top_p
                if self.top_k is not None:
                    completion_kwargs.setdefault("extra_body", {})["top_k"] = self.top_k
                return await asyncio.wait_for(
                    asyncio.to_thread(
                        litellm.completion,
                        **completion_kwargs,
                    ),
                    timeout=llm_timeout + LLM_OUTER_TIMEOUT_BUFFER_SECONDS,
                )
            except ABORT_EXCEPTIONS:
                raise
            except Exception as exc:
                if attempt == MAX_RETRIES:
                    logger.error("Max retries reached: %s: %s", type(exc).__name__, exc)
                    raise
                delay = RETRY_BASE_DELAY * (2 ** (attempt - 1))
                logger.warning(
                    "Retry %s/%s after %s: %s (waiting %.0fs)",
                    attempt,
                    MAX_RETRIES,
                    type(exc).__name__,
                    exc,
                    delay,
                )
                await asyncio.sleep(delay)

    @staticmethod
    def _accumulate_usage(response: Any, usage_totals: dict[str, int]) -> None:
        usage = getattr(response, "usage", None)
        if usage is None:
            return
        for key in usage_totals:
            usage_totals[key] += getattr(usage, key, 0) or 0

    @staticmethod
    def _append_format_error(
        messages: list[dict[str, Any]], tool_call_id: str | None
    ) -> None:
        content = _format_error_message(
            "Your last response did not include a valid `bash` tool call."
        )
        if tool_call_id:
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": content,
                }
            )
            return
        messages.append({"role": "user", "content": content})

    def _wrap_command(self, command: str) -> str:
        if not self.persistent_bash:
            return command
        return (
            f'cd "$(cat {_STATE_DIR}/cwd)" 2>/dev/null || true\n'
            f". {_STATE_DIR}/env 2>/dev/null || true\n"
            f"{command}\n"
            "_vanillux2_ec=$?\n"
            f"pwd > {_STATE_DIR}/cwd\n"
            f"export -p > {_STATE_DIR}/env\n"
            "exit $_vanillux2_ec"
        )

    async def _execute_bash(
        self, command: str, environment: BaseEnvironment
    ) -> Any:
        return await environment.exec(
            command=self._wrap_command(command),
            timeout_sec=self.command_timeout,
        )

    @staticmethod
    def _format_tool_result(result: Any) -> str:
        output = result.stdout or ""
        if result.stderr:
            output += f"\n{result.stderr}" if output else result.stderr
        output = _COMPOSE_PROVIDER_RE.sub("", output)
        output = _DOCKER_EXEC_ERROR_RE.sub("", output).rstrip()
        truncated = _truncate_observation(output) if output else "(no output)"
        return f"{truncated}\n\n(exit_code={result.return_code})"
