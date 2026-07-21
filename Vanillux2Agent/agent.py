"""Vanillux2Agent - direct LiteLLM agent using the vanillux prompt harness.

This is the Harbor-agent version of ``rl_data.generator.vanillux_solver``:
it uses the same mini-SWE-agent-derived prompts, bash tool schema, submit
marker, format-error recovery, and output truncation, but executes commands
through Harbor's active environment and calls the model directly with LiteLLM.

This branch (error_hints) appends TARGETED RECOVERY HINTS to tool results
whose output matches a known error signature, and nothing else, so an A/B
against the replicate baseline isolates the effect. The signatures come
from a 444-run failure analysis, counting runs (not occurrences) that
contain each error, fail vs pass: `command not found` 57% vs 39%,
`No such file or directory` 31% vs 20%, `ModuleNotFoundError/ImportError`
25% vs 22%, plus pip/network install failures 4% vs 1%. Each distinct hint
fires at most once per run (a model shouldn't drown in repeated advice),
appended to the observation the model already sees.
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

# Error-signature -> recovery hint. Ordered: the first match on an
# observation wins (so a compound failure gets its most specific hint), and
# each key fires at most once per run. Signatures + rates from the 444-run
# baseline analysis (share of runs containing the error, fail vs pass).
_ERROR_HINTS: list[tuple[str, re.Pattern[str], str]] = [
    (
        "cmd_not_found",
        re.compile(r"command not found|: not found"),
        "A command was not found. Before assuming a tool is missing: check the exact "
        "name/spelling, whether it needs an absolute path or a venv/conda activation, "
        "and whether the project ships it under a different name (`ls`, `which -a`, "
        "`compgen -c | grep`, or read the project's README/Makefile). Install only as a "
        "last resort, and check what package actually provides it first.",
    ),
    (
        "module_not_found",
        re.compile(r"ModuleNotFoundError|No module named|ImportError"),
        "A Python import failed. The module may already be present under a different "
        "interpreter or env (`python3` vs `python`, a venv, `pip list`), importable from "
        "a different working directory, or vendored in the repo. Confirm which "
        "interpreter and cwd you need before installing anything.",
    ),
    (
        "install_fail",
        re.compile(
            r"Could not find a version|No matching distribution|Temporary failure in name resolution"
            r"|Could not resolve host|Network is unreachable|Connection refused",
            re.I,
        ),
        "A download/install failed, likely because this environment has no or limited "
        "network access. Prefer what is already installed or vendored in the repo; do "
        "not build the solution around a package you cannot install.",
    ),
    (
        "no_such_file",
        re.compile(r"No such file or directory"),
        "A path did not exist. Check your current directory (`pwd`) and list the parent "
        "(`ls -la`) — the file may be under a different directory than you assumed, or "
        "you may need to create the parent first. Prefer absolute paths for deliverables.",
    ),
]


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
        enable_error_hints: bool = True,
        **kwargs: Any,
    ) -> None:
        """
        enable_error_hints: append a one-shot recovery hint to a tool result
            whose output matches a known error signature (command-not-found,
            missing module, failed install, missing path). Each signature
            fires at most once per run.
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
        self.enable_error_hints = enable_error_hints
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
        error_hints_fired: set[str] = set()

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

                t1 = time.monotonic()
                result = await self._execute_bash(command, environment)
                exec_time = time.monotonic() - t1

                tool_content = self._format_tool_result(result)

                # Targeted recovery hint: only on a nonzero exit, only the
                # first matching signature, and only once per signature per
                # run. Appended to the observation the model already sees.
                hint_key = None
                if (
                    self.enable_error_hints
                    and result.return_code != 0
                    and SUBMIT_MARKER not in tool_content
                ):
                    for key, rx, hint in _ERROR_HINTS:
                        if key not in error_hints_fired and rx.search(tool_content):
                            error_hints_fired.add(key)
                            hint_key = key
                            tool_content += f"\n\n(harness hint) {hint}"
                            break

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
                        **({"error_hint": hint_key} if hint_key else {}),
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
            context.metadata = {"error_hints": {"fired": sorted(error_hints_fired)}}

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
