"""Vanillux2Agent - direct LiteLLM agent using the vanillux prompt harness.

This is the Harbor-agent version of ``rl_data.generator.vanillux_solver``:
it uses the same mini-SWE-agent-derived prompts, bash tool schema, submit
marker, and format-error recovery, but executes commands through Harbor's
active environment and calls the model directly with LiteLLM.

Context management: ``messages`` below is the raw, append-only event log
(dumped verbatim to ``trajectory.json``); the model only ever sees a
compacted rebuild of it, produced fresh every step by
``context_management.build_model_messages``. See that module's docstring for
the stubbing/truncation rules, and ``edit_tools.py`` for the optional
str_replace/insert/create/apply_edits/read tools. Config flags for all of
this are documented on ``Vanillux2Agent.__init__``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any

import litellm
from harbor.agents.base import BaseAgent
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext

from rl_data.generator.sample_solutions import SUBMIT_MARKER, TOOL_SCHEMAS
from rl_data.generator.vanillux_solver import (
    _format_error_message,
    _render_instance,
    _SYSTEM_TEMPLATE,
)

from Vanillux2Agent import context_management, edit_tools

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
# Disk/memory safety net for the RAW log only — not context-management
# truncation (that's context_management.py, applied to the model-facing view
# only). Guards against a runaway command dumping gigabytes of output.
_RAW_OUTPUT_SAFETY_CAP_CHARS = 2_000_000


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
        enable_edit_tools: bool = True,
        stub_file_writes: bool = True,
        write_recency_keep: int = 2,
        max_tool_output_tokens: int = 2000,
        head_lines: int = 40,
        tail_lines: int = 40,
        **kwargs: Any,
    ) -> None:
        """
        Context-management flags (see context_management.py for the compaction
        logic these drive):

        enable_edit_tools: register str_replace/insert/create/apply_edits/read
            alongside bash (Feature 3).
        stub_file_writes: replace superseded/stale file-write payloads with a
            short stub in the model-facing history (Feature 1).
        write_recency_keep: a write stays un-stubbed only while it is both the
            latest write to its path and within this many trailing turns.
        max_tool_output_tokens: tool output above this token count is
            head/tail-truncated with the elided middle spilled to disk
            (Feature 2).
        head_lines / tail_lines: how much of a truncated tool output to keep
            verbatim at each end.
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
        self.enable_edit_tools = enable_edit_tools
        self._compaction_config = context_management.CompactionConfig(
            stub_file_writes=stub_file_writes,
            write_recency_keep=write_recency_keep,
            max_tool_output_tokens=max_tool_output_tokens,
            head_lines=head_lines,
            tail_lines=tail_lines,
        )
        self._tool_schemas = TOOL_SCHEMAS + (edit_tools.EDIT_TOOL_SCHEMAS if enable_edit_tools else [])

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
        # The raw event log: append-only, never mutated, dumped verbatim to
        # trajectory.json. The model only ever sees a compacted rebuild of
        # this (see context_management.build_model_messages below).
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": _SYSTEM_TEMPLATE},
            {"role": "user", "content": _render_instance(instruction.strip())},
        ]

        timing_log: list[dict[str, Any]] = []
        usage_totals = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "reasoning_tokens": 0,
        }
        format_errors = 0

        def exec_fn(command: str) -> Any:
            return self._execute_bash(command, environment)

        spilled_indices: set[int] = set()
        compaction_stats = context_management.CompactionStats()

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

                model_messages = await context_management.build_model_messages(
                    messages,
                    config=self._compaction_config,
                    model=model,
                    exec_fn=exec_fn,
                    spilled_indices=spilled_indices,
                    stats=compaction_stats,
                    logger=logger,
                )

                t0 = time.monotonic()
                try:
                    response = await self._query_with_retry(model, model_messages)
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
                action = edit_tools.extract_action(msg)
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
                tool_call_id = action.get("tool_call_id") or ""

                t1 = time.monotonic()
                if action["name"] == "bash":
                    command = action["args"].get("command") or ""
                    result = await self._execute_bash(command, environment)
                    exec_time = time.monotonic() - t1
                    tool_content = self._format_tool_result(result)
                    timing_log.append(
                        {
                            "step": step + 1,
                            "llm_s": round(llm_time, 1),
                            "bash_s": round(exec_time, 1),
                            "return_code": result.return_code,
                            "cmd": command[:200],
                        }
                    )
                else:
                    tool_content = await edit_tools.dispatch(action["name"], action["args"], exec_fn)
                    exec_time = time.monotonic() - t1
                    timing_log.append(
                        {
                            "step": step + 1,
                            "llm_s": round(llm_time, 1),
                            "tool_s": round(exec_time, 1),
                            "tool": action["name"],
                        }
                    )

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call_id,
                        "content": tool_content,
                    }
                )

                if action["type"] == "done" or SUBMIT_MARKER in tool_content:
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
            context.metadata = {
                "compaction": {
                    "stubbed_writes": compaction_stats.stubbed_writes,
                    "truncated_outputs": compaction_stats.truncated_outputs,
                }
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
                    "tools": self._tool_schemas,
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
        # This is the RAW log entry (see context_management.py) — no
        # context-management truncation here, only a generous disk/memory
        # safety net against a runaway command dumping gigabytes of output.
        # The model-facing view is compacted separately, every step, from
        # this full text via context_management.build_model_messages.
        output = result.stdout or ""
        if result.stderr:
            output += f"\n{result.stderr}" if output else result.stderr
        output = _COMPOSE_PROVIDER_RE.sub("", output)
        output = _DOCKER_EXEC_ERROR_RE.sub("", output).rstrip()
        if len(output) > _RAW_OUTPUT_SAFETY_CAP_CHARS:
            half = _RAW_OUTPUT_SAFETY_CAP_CHARS // 2
            n_elided = len(output) - _RAW_OUTPUT_SAFETY_CAP_CHARS
            output = f"{output[:half]}\n\n... [{n_elided} chars elided; raw safety cap] ...\n\n{output[-half:]}"
        if not output:
            output = "(no output)"
        return f"{output}\n\n(exit_code={result.return_code})"
