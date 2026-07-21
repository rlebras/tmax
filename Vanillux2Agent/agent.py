"""Vanillux2Agent - direct LiteLLM agent using the vanillux prompt harness.

This is the Harbor-agent version of ``rl_data.generator.vanillux_solver``:
it uses the same mini-SWE-agent-derived prompts, bash tool schema, submit
marker, format-error recovery, and output truncation, but executes commands
through Harbor's active environment and calls the model directly with LiteLLM.

This branch (self_test_only) carries the SELF-TEST GATE and nothing else —
no context-management compaction, no edit tools — so an A/B against the
plain replicate baseline isolates the gate's effect. The tool contract stays
bash-only; self-testing rides the intercepted ``agent-check <id> --
<command>`` convention plus a harness-owned criteria file. See
``self_test.py``'s module docstring for the mechanism (criteria declaration,
isolated checks, the bounded submission gate, and the anti-circularity
heuristics) and its documented isolation-scope limitation.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import shlex
import tempfile
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
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
    _SELF_TEST_INSTANCE_ADDENDUM_BASH,
    _SYSTEM_TEMPLATE,
    _truncate_observation,
)

from Vanillux2Agent import self_test
from Vanillux2Agent.container_ops import ContainerOps

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


def _looks_like_exec_timeout(exc: Exception) -> bool:
    return isinstance(exc, asyncio.TimeoutError) or "timed out" in str(exc).lower()


def _timed_out_result(timeout_sec: int) -> Any:
    """Duck-types harbor's ExecResult for a command the harness terminated.

    harbor's docker environment RAISES on exec timeout (RuntimeError
    "Command timed out after Ns") rather than returning a result. The gate
    roughly doubles command executions (every check runs in-session AND in
    isolation), so left unhandled one slow check would end the whole run —
    convert to a model-visible failure instead.
    """
    return SimpleNamespace(
        stdout="",
        stderr=(
            f"(harness) command did not finish within {timeout_sec}s and was terminated; "
            "its effects may be partially applied. Break long-running work into smaller "
            "steps or run it in the background with `nohup ... &` and poll."
        ),
        return_code=124,
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
        enable_self_test_gate: bool = True,
        min_criteria: int = 2,
        max_gate_rejections: int = 3,
        self_test_check_timeout: int = 60,
        self_test_isolation_mode: str = "tempdir",
        **kwargs: Any,
    ) -> None:
        """
        Self-test gate flags (see self_test.py for the mechanism these drive):

        enable_self_test_gate: gate the COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT
            sentinel on isolated self-test coverage, via the intercepted
            `agent-check <id> -- <command>` bash convention (the bash-only
            tool schema stays untouched).
        min_criteria: at least this many declared criteria must be covered by
            a passing isolated check before submit is allowed (and every
            declared criterion must be covered).
        max_gate_rejections: after this many rejected submit attempts, submit
            is allowed anyway (bounds the downside of a model stuck failing
            its own checks) and ``submitted_with_failing_checks`` is recorded
            in ``context.metadata``.
        self_test_check_timeout: per-check timeout for the isolated run.
        self_test_isolation_mode: only "tempdir" (fresh temp dir + fresh
            process, populated with only the declared deliverables) is
            currently implemented; see self_test.py's module docstring for
            why a fresh container isn't available at this layer.
        """
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
        self.cost: float = 0.0
        self.enable_self_test_gate = enable_self_test_gate
        if self_test_isolation_mode != "tempdir":
            raise ValueError(f"unsupported self_test_isolation_mode: {self_test_isolation_mode!r}")
        self._self_test_config = self_test.SelfTestConfig(
            min_criteria=min_criteria,
            max_gate_rejections=max_gate_rejections,
            check_timeout_sec=self_test_check_timeout,
            state_dir=f"{_STATE_DIR}/self_test",
            isolation_mode=self_test_isolation_mode,
        )

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
        rendered_instance = _render_instance(instruction.strip())
        if self.enable_self_test_gate:
            instance_addendum = _SELF_TEST_INSTANCE_ADDENDUM_BASH.replace(
                "{{min_criteria}}", str(self._self_test_config.min_criteria)
            ).replace("{{criteria_path}}", self._self_test_config.criteria_path)
            rendered_instance = f"{rendered_instance}\n{instance_addendum}"
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": _SYSTEM_TEMPLATE},
            {"role": "user", "content": rendered_instance},
        ]

        timing_log: list[dict[str, Any]] = []
        usage_totals = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "reasoning_tokens": 0,
        }
        format_errors = 0

        # Sandbox access for self_test.py (criteria/checks persistence and
        # check execution). Content moves via upload_bytes (docker cp), never
        # through exec argv — see edit_tools.py on the ctx-mgmt branches for
        # the ARG_MAX/null-byte rationale; the same contract applies here.
        def exec_fn(command: str) -> Any:
            return self._execute_bash(command, environment)

        async def upload_bytes(content: bytes, remote_path: str) -> None:
            fd, local_tmp = tempfile.mkstemp(prefix="vanillux2_upload_")
            try:
                with os.fdopen(fd, "wb") as f:
                    f.write(content)
                await environment.upload_file(local_tmp, remote_path)
            finally:
                try:
                    os.unlink(local_tmp)
                except OSError:
                    pass

        async def download_bytes(remote_path: str) -> bytes:
            fd, local_tmp = tempfile.mkstemp(prefix="vanillux2_download_")
            os.close(fd)
            try:
                await environment.download_file(remote_path, local_tmp)
                return Path(local_tmp).read_bytes()
            except Exception as exc:
                raise FileNotFoundError(f"{remote_path}: {exc}") from exc
            finally:
                try:
                    os.unlink(local_tmp)
                except OSError:
                    pass

        ops = ContainerOps(exec_fn=exec_fn, upload_bytes=upload_bytes, download_bytes=download_bytes)

        async def isolated_exec(command: str, cwd: str) -> Any:
            # Deliberately bypasses _wrap_command: no persistent cwd/env
            # sourcing, no state-file updates — a genuinely fresh process per
            # self_test.py's isolation contract (Mechanism 3).
            try:
                return await environment.exec(
                    command=f"cd {shlex.quote(cwd)} && {command}",
                    timeout_sec=self._self_test_config.check_timeout_sec,
                )
            except (asyncio.TimeoutError, RuntimeError) as exc:
                # Same raise-on-timeout contract as _execute_bash: a slow
                # check must count as a failed check, not end the run.
                if not _looks_like_exec_timeout(exc):
                    raise
                return _timed_out_result(self._self_test_config.check_timeout_sec)

        self_test_state = self_test.SelfTestState()

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

                # Self-test interception of the bash stream: agent-check
                # invocations, botched attempts, and existence probes are
                # answered by the harness before they ever reach the shell.
                if self.enable_self_test_gate and action["type"] != "done":
                    if self_test.looks_like_existence_probe(command):
                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": tool_call_id,
                                "content": self_test.AGENT_CHECK_EXISTENCE_HINT,
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

                    checks: list[tuple[str, str]] = []
                    remainder = ""
                    whole = self_test.parse_agent_check(command)
                    if whole is not None:
                        checks = [whole]
                    else:
                        found = self_test.extract_agent_check_lines(command)
                        if found is not None:
                            checks, remainder = found

                    if checks:
                        t1 = time.monotonic()
                        prefix = ""
                        if remainder:
                            if self_test.looks_like_malformed_agent_check(remainder):
                                # e.g. a well-formed check alongside a botched
                                # one on another line — don't let the botched
                                # one hit the real shell as "command not found".
                                prefix = self_test.AGENT_CHECK_SYNTAX_HINT + "\n\n"
                            else:
                                pre_result = await self._execute_bash(remainder, environment)
                                prefix = self._format_tool_result(pre_result) + "\n\n"
                        check_outputs = [
                            await self_test.run_check(
                                criterion_id,
                                inner_command,
                                None,
                                state=self_test_state,
                                config=self._self_test_config,
                                ops=ops,
                                isolated_exec=isolated_exec,
                                step=step + 1,
                            )
                            for criterion_id, inner_command in checks
                        ]
                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": tool_call_id,
                                "content": prefix + "\n\n".join(check_outputs),
                            }
                        )
                        timing_log.append(
                            {
                                "step": step + 1,
                                "llm_s": round(llm_time, 1),
                                "bash_s": round(time.monotonic() - t1, 1),
                                "agent_check": [cid for cid, _ in checks],
                            }
                        )
                        continue

                    if self_test.looks_like_malformed_agent_check(command):
                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": tool_call_id,
                                "content": self_test.AGENT_CHECK_SYNTAX_HINT,
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

                # Explicit submit: gate BEFORE the sentinel command runs, so
                # a rejected attempt costs no execution and the marker never
                # enters the log to trip the finish condition below.
                if action["type"] == "done" and self.enable_self_test_gate:
                    criteria, _deliverables = await self_test.load_criteria(
                        ops, self._self_test_config
                    )
                    gate = self_test.evaluate_gate(
                        criteria, self_test_state, self._self_test_config
                    )
                    if not gate.allowed:
                        self_test_state.gate_rejections += 1
                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": tool_call_id,
                                "content": self_test.gate_nudge(
                                    gate, self._self_test_config, tools_enabled=False
                                ),
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
                # that path too, since it otherwise bypasses the explicit-
                # submit gate above. Unlike that path, the command already
                # ran, so a rejection appends the nudge to the (already-
                # produced) output rather than replacing it.
                is_implicit_submit = action["type"] != "done" and SUBMIT_MARKER in tool_content
                if is_implicit_submit and self.enable_self_test_gate:
                    criteria, _deliverables = await self_test.load_criteria(
                        ops, self._self_test_config
                    )
                    gate = self_test.evaluate_gate(
                        criteria, self_test_state, self._self_test_config
                    )
                    if not gate.allowed:
                        self_test_state.gate_rejections += 1
                        tool_content += (
                            "\n\n(NOTE: this output happened to contain the submit marker text, "
                            "but that alone does not finish the task.) "
                            + self_test.gate_nudge(
                                gate, self._self_test_config, tools_enabled=False
                            )
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

                if action["type"] == "done" or is_implicit_submit:
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

            # Best-effort final read of the criteria file so adoption metrics
            # are accurate even for runs that declared criteria but never
            # reached a check or the gate (the snapshot only refreshes at
            # those points); the environment may already be gone here.
            criteria_view = dict(self_test_state.criteria_snapshot)
            if self.enable_self_test_gate:
                try:
                    final_criteria, _ = await self_test.load_criteria(ops, self._self_test_config)
                    criteria_view = {cid: c.description for cid, c in final_criteria.items()}
                except Exception:
                    pass
            check_records = [r for records in self_test_state.checks.values() for r in records]
            coverage_view = self_test.coverage(
                {cid: self_test.Criterion(cid, desc) for cid, desc in criteria_view.items()},
                self_test_state,
            )
            if self.enable_self_test_gate:
                (self.logs_dir / "self_test.json").write_text(
                    json.dumps(
                        {
                            "enabled": True,
                            "criteria": criteria_view,
                            "coverage": coverage_view,
                            "gate_rejections": self_test_state.gate_rejections,
                            "submitted_with_failing_checks": self_test_state.submitted_with_failing_checks,
                            "checks": {
                                cid: [asdict(r) for r in records]
                                for cid, records in self_test_state.checks.items()
                            },
                            "audit": self_test_state.audit,
                        },
                        indent=2,
                    )
                    + "\n"
                )
            context.metadata = {
                "self_test": {
                    "enabled": self.enable_self_test_gate,
                    "criteria_declared": len(criteria_view),
                    "criteria_covered": sum(coverage_view.values()),
                    "checks_run": len(check_records),
                    "checks_by_category": dict(
                        Counter(r.check_category for r in check_records)
                    ),
                    "session_isolation_discrepancies": sum(
                        1 for r in check_records if r.session_pass and not r.isolated_pass
                    ),
                    "gate_rejections": self_test_state.gate_rejections,
                    "submitted_with_failing_checks": self_test_state.submitted_with_failing_checks,
                },
            }

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
        try:
            return await environment.exec(
                command=self._wrap_command(command),
                timeout_sec=self.command_timeout,
            )
        except (asyncio.TimeoutError, RuntimeError) as exc:
            # See _timed_out_result: harbor raises on exec timeout; anything
            # else is a real environment failure and still propagates.
            if not _looks_like_exec_timeout(exc):
                raise
            logger.warning("Command timed out after %ss: %s", self.command_timeout, command[:120])
            return _timed_out_result(self.command_timeout)

    @staticmethod
    def _format_tool_result(result: Any) -> str:
        output = result.stdout or ""
        if result.stderr:
            output += f"\n{result.stderr}" if output else result.stderr
        output = _COMPOSE_PROVIDER_RE.sub("", output)
        output = _DOCKER_EXEC_ERROR_RE.sub("", output).rstrip()
        truncated = _truncate_observation(output) if output else "(no output)"
        return f"{truncated}\n\n(exit_code={result.return_code})"
