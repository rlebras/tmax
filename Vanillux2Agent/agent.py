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

Self-test gate: ``self_test.py`` lets the model declare acceptance criteria
and verify them against an isolated copy of its declared deliverables before
the ``COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`` sentinel is honored — via the
``declare_criteria``/``run_check`` tools when edit tools are on, or the
intercepted ``agent-check <id> -- <command>`` bash convention when the
contract is bash-only. See ``self_test.py``'s module docstring for the full
mechanism and its documented isolation-scope limitation.
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

from rl_data.generator.sample_solutions import SUBMIT_MARKER, TOOL_SCHEMAS
from rl_data.generator.vanillux_solver import (
    _format_error_message,
    _format_error_message_multi_tool,
    _render_instance,
    _render_instance_multi_tool,
    _SELF_TEST_FORMAT_ERROR_ADDENDUM,
    _SELF_TEST_INSTANCE_ADDENDUM,
    _SELF_TEST_INSTANCE_ADDENDUM_BASH,
    _SELF_TEST_SYSTEM_ADDENDUM,
    _SYSTEM_TEMPLATE,
    _SYSTEM_TEMPLATE_MULTI_TOOL,
)

from Vanillux2Agent import context_management, edit_tools, self_test
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
LLM_OUTER_TIMEOUT_BUFFER_SECONDS = 30
_STATE_DIR = "/tmp/.vanillux2"
# Model-requested command timeouts: the prompt tells the model to wrap long
# commands with `timeout N <cmd>`; honor that by raising the exec deadline to
# N plus this margin (bounded by max_command_timeout), so `timeout 300 make`
# isn't killed at the default 120s while the model believes it has 300.
_TIMEOUT_REQUEST_RE = re.compile(
    r"\btimeout\s+(?:(?:-k|--kill-after|-s|--signal)(?:[= ]\S+)\s+|--foreground\s+|--preserve-status\s+)*"
    r"(?P<n>\d+(?:\.\d+)?)(?P<unit>[smhd]?)\b"
)
_TIMEOUT_REQUEST_MARGIN_SECONDS = 30
_TIMEOUT_UNIT_SECONDS = {"": 1, "s": 1, "m": 60, "h": 3600, "d": 86400}
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


def _looks_like_exec_timeout(exc: Exception) -> bool:
    return isinstance(exc, asyncio.TimeoutError) or "timed out" in str(exc).lower()


def _timed_out_result(timeout_sec: int) -> Any:
    """Duck-types harbor's ExecResult for a command the harness terminated."""
    return SimpleNamespace(
        stdout="",
        stderr=(
            f"(harness) command did not finish within {timeout_sec}s and was terminated; "
            "its effects may be partially applied. For long-running work, request more time "
            "with an explicit `timeout N <cmd>` (N in seconds), run it in the background "
            "with `nohup ... &` and poll, or break it into smaller steps."
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
        enable_edit_tools: bool = True,
        stub_file_writes: bool = True,
        write_recency_keep: int = 2,
        max_tool_output_tokens: int = 2000,
        head_lines: int = 40,
        tail_lines: int = 40,
        enable_self_test_gate: bool = True,
        min_criteria: int = 2,
        max_gate_rejections: int = 3,
        self_test_check_timeout: int = 60,
        self_test_isolation_mode: str = "tempdir",
        max_context_tokens: int | None = None,
        context_headroom_tokens: int = 2048,
        llm_timeout: int = 900,
        max_command_timeout: int = 600,
        wall_clock_budget_sec: float | None = None,
        deadline_warning_sec: int = 300,
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

        Self-test gate flags (see self_test.py for the mechanism these drive):

        enable_self_test_gate: gate the COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT
            sentinel on isolated self-test coverage. With edit tools on, the
            model gets real declare_criteria/run_check tools; with the
            bash-only contract it gets the intercepted `agent-check` bash
            convention instead (so the bash-only tool schema stays untouched).
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

        Budget flags (context window + wall clock — measured on the 444-run
        baseline these are the two dominant loss buckets):

        max_context_tokens: the serving context window (e.g. 65536 for the
            standard eval). When set, the model-facing rebuild is measured
            each step and compaction escalates BEFORE a request would
            overflow. When None, only the reactive path below applies.
        context_headroom_tokens: safety margin subtracted (together with
            max_tokens) from max_context_tokens for the proactive check —
            token estimates are approximate, so leave slack. Reactive
            recovery is unconditional: a ContextWindowExceededError from the
            API escalates compaction and retries instead of ending the run
            (up to context_management.MAX_COMPACTION_LEVEL).
        llm_timeout: per-request LLM timeout in seconds (retried with
            backoff). The old effectively-unbounded value let one hung
            request eat an entire agent wall-clock budget.
        max_command_timeout: upper bound on the exec deadline granted when
            the model wraps a command with `timeout N ...` (the exec
            deadline is raised to N + a margin, capped here). A command that
            still overruns is terminated and reported to the model as a
            timeout instead of crashing the run.
        wall_clock_budget_sec: the agent's total wall-clock budget, if the
            launcher knows it (harbor does not expose it to the agent).
            When set, a one-time "finalize and submit now" nudge is injected
            once remaining time drops below deadline_warning_sec, and the
            self-test gate stops rejecting submits (a flagged submit beats
            an AgentTimeoutError kill).
        deadline_warning_sec: how early the deadline nudge/gate-standdown
            fires.
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
        self.max_context_tokens = max_context_tokens
        self.context_headroom_tokens = context_headroom_tokens
        self.llm_timeout = llm_timeout
        self.max_command_timeout = max_command_timeout
        self.wall_clock_budget_sec = wall_clock_budget_sec
        self.deadline_warning_sec = deadline_warning_sec
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
        # Real tools need the multi-tool contract; on the bash-only contract
        # the gate falls back to the intercepted agent-check convention so
        # the RL-data-compatible bash-only schema stays untouched.
        self._self_test_tools_enabled = enable_self_test_gate and enable_edit_tools
        self._tool_schemas = (
            TOOL_SCHEMAS
            + (edit_tools.EDIT_TOOL_SCHEMAS if enable_edit_tools else [])
            + (self_test.SELF_TEST_TOOL_SCHEMAS if self._self_test_tools_enabled else [])
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
        system_template = _SYSTEM_TEMPLATE_MULTI_TOOL if self.enable_edit_tools else _SYSTEM_TEMPLATE
        render_instance = _render_instance_multi_tool if self.enable_edit_tools else _render_instance
        rendered_instance = render_instance(instruction.strip())
        if self.enable_self_test_gate:
            if self._self_test_tools_enabled:
                system_template = f"{system_template}\n{_SELF_TEST_SYSTEM_ADDENDUM}"
                instance_addendum = _SELF_TEST_INSTANCE_ADDENDUM
            else:
                instance_addendum = _SELF_TEST_INSTANCE_ADDENDUM_BASH
            instance_addendum = instance_addendum.replace(
                "{{min_criteria}}", str(self._self_test_config.min_criteria)
            ).replace("{{criteria_path}}", self._self_test_config.criteria_path)
            rendered_instance = f"{rendered_instance}\n{instance_addendum}"
        # The raw event log: append-only, never mutated, dumped verbatim to
        # trajectory.json. The model only ever sees a compacted rebuild of
        # this (see context_management.build_model_messages below).
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_template},
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

        spilled_indices: set[int] = set()
        compaction_stats = context_management.CompactionStats()
        self_test_state = self_test.SelfTestState()

        # Overflow-recovery ladder (see context_management.escalate_config):
        # sticky within a run — once the history has outgrown a level there
        # is no point de-escalating, it would overflow again immediately.
        compaction_level = 0
        overflow_recoveries = 0
        proactive_escalations = 0

        run_deadline = (
            time.monotonic() + self.wall_clock_budget_sec
            if self.wall_clock_budget_sec is not None
            else None
        )
        deadline_warned = False

        def deadline_imminent() -> bool:
            return (
                run_deadline is not None
                and (run_deadline - time.monotonic()) < self.deadline_warning_sec
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

                if run_deadline is not None and not deadline_warned and deadline_imminent():
                    deadline_warned = True
                    remaining = max(0, int(run_deadline - time.monotonic()))
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                f"(harness) Wall-clock budget nearly exhausted (~{remaining}s left). "
                                "Stop exploring: make sure your solution is saved to its "
                                "deliverable files now, then submit with "
                                "`echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`."
                            ),
                        }
                    )

                # Proactive budget enforcement: estimate the rebuilt prompt
                # and escalate compaction BEFORE a request would overflow.
                # Best-effort (token estimates are approximate) — the
                # reactive recovery below is the guaranteed backstop.
                while True:
                    compaction_config = context_management.escalate_config(
                        self._compaction_config, compaction_level
                    )
                    model_messages = await context_management.build_model_messages(
                        messages,
                        config=compaction_config,
                        model=model,
                        ops=ops,
                        spilled_indices=spilled_indices,
                        stats=compaction_stats,
                        logger=logger,
                    )
                    if (
                        self.max_context_tokens is None
                        or compaction_level >= context_management.MAX_COMPACTION_LEVEL
                    ):
                        break
                    prompt_budget = (
                        self.max_context_tokens - self.max_tokens - self.context_headroom_tokens
                    )
                    estimate = context_management.estimate_messages_tokens(model_messages, model)
                    if estimate <= prompt_budget:
                        break
                    compaction_level += 1
                    proactive_escalations += 1
                    logger.warning(
                        "Estimated prompt %s tokens > budget %s; escalating compaction to level %s",
                        estimate,
                        prompt_budget,
                        compaction_level,
                    )

                t0 = time.monotonic()
                try:
                    response = await self._query_with_retry(model, model_messages)
                except litellm.exceptions.ContextWindowExceededError:
                    # Reactive overflow recovery: rebuild under harsher
                    # compaction and keep going (costs this one loop step,
                    # bounded by MAX_COMPACTION_LEVEL) instead of abandoning
                    # the run's remaining steps — on the measured baseline,
                    # unsubmitted runs pass at 1.7% vs 49% for submitted
                    # ones, so giving up here forfeits nearly everything.
                    if compaction_level < context_management.MAX_COMPACTION_LEVEL:
                        compaction_level += 1
                        overflow_recoveries += 1
                        logger.warning(
                            "Context window exceeded; retrying at compaction level %s",
                            compaction_level,
                        )
                        continue
                    logger.warning("Context window exceeded at max compaction; stopping current run")
                    break
                llm_time = time.monotonic() - t0

                self._accumulate_usage(response, usage_totals)
                try:
                    self.cost += litellm.completion_cost(response, model=model)
                except Exception:
                    pass

                msg = response.choices[0].message.model_dump()
                action = edit_tools.extract_action(
                    msg,
                    extra_tool_names=self_test.SELF_TEST_TOOL_NAMES
                    if self._self_test_tools_enabled
                    else frozenset(),
                )
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
                # The contract is one tool call per turn and only the first
                # is executed — answer any extras so no tool_call_id is left
                # unpaired (a strict OpenAI-protocol server rejects a history
                # with orphaned calls on the NEXT request).
                for extra_call in (msg.get("tool_calls") or [])[1:]:
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": extra_call.get("id") or "",
                            "content": (
                                "(ignored: exactly one tool call per turn — "
                                "this extra call was not executed)"
                            ),
                        }
                    )
                is_bash = action["name"] == "bash"
                command = (action["args"].get("command") or "") if is_bash else ""

                # Self-test interception of the bash stream (active in both
                # contract modes when the gate is on): agent-check
                # invocations, botched attempts, and existence probes are
                # answered by the harness before they ever reach the shell.
                if self.enable_self_test_gate and is_bash and action["type"] != "done":
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
                    if not gate.allowed and deadline_imminent():
                        # Deadline standdown: a flagged submit now beats an
                        # AgentTimeoutError kill a few steps later.
                        gate = self_test.GateResult(True, ["deadline imminent"], forced=True)
                    if not gate.allowed:
                        self_test_state.gate_rejections += 1
                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": tool_call_id,
                                "content": self_test.gate_nudge(
                                    gate, self._self_test_config, self._self_test_tools_enabled
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
                if is_bash:
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
                elif self._self_test_tools_enabled and action["name"] in self_test.SELF_TEST_TOOL_NAMES:
                    tool_content = await self_test.dispatch(
                        action["name"],
                        action["args"],
                        state=self_test_state,
                        ops=ops,
                        isolated_exec=isolated_exec,
                        config=self._self_test_config,
                        step=step + 1,
                    )
                    exec_time = time.monotonic() - t1
                    timing_log.append(
                        {
                            "step": step + 1,
                            "llm_s": round(llm_time, 1),
                            "tool_s": round(exec_time, 1),
                            "tool": action["name"],
                        }
                    )
                else:
                    tool_content = await edit_tools.dispatch(action["name"], action["args"], ops)
                    exec_time = time.monotonic() - t1
                    timing_log.append(
                        {
                            "step": step + 1,
                            "llm_s": round(llm_time, 1),
                            "tool_s": round(exec_time, 1),
                            "tool": action["name"],
                        }
                    )

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
                    if not gate.allowed and deadline_imminent():
                        gate = self_test.GateResult(True, ["deadline imminent"], forced=True)
                    if not gate.allowed:
                        self_test_state.gate_rejections += 1
                        tool_content += (
                            "\n\n(NOTE: this output happened to contain the submit marker text, "
                            "but that alone does not finish the task.) "
                            + self_test.gate_nudge(
                                gate, self._self_test_config, self._self_test_tools_enabled
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
            # are accurate even for bash-mode runs that declared criteria but
            # never reached a check or the gate (the snapshot only refreshes
            # at those points); the environment may already be gone here.
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
                "compaction": {
                    "stubbed_writes": compaction_stats.stubbed_writes,
                    "truncated_outputs": compaction_stats.truncated_outputs,
                    "final_level": compaction_level,
                    "overflow_recoveries": overflow_recoveries,
                    "proactive_escalations": proactive_escalations,
                },
                "deadline": {
                    "budget_sec": self.wall_clock_budget_sec,
                    "warned": deadline_warned,
                },
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
                llm_timeout = self.llm_timeout
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

    def _append_format_error(
        self, messages: list[dict[str, Any]], tool_call_id: str | None
    ) -> None:
        if self.enable_edit_tools:
            content = _format_error_message_multi_tool(
                "Your last response did not include a valid tool call."
            )
        else:
            content = _format_error_message(
                "Your last response did not include a valid `bash` tool call."
            )
        if self._self_test_tools_enabled:
            content = f"{content}{_SELF_TEST_FORMAT_ERROR_ADDENDUM}"
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

    def _requested_timeout_sec(self, command: str) -> int | None:
        """Largest `timeout N` the model asked for anywhere in *command*."""
        best: int | None = None
        for m in _TIMEOUT_REQUEST_RE.finditer(command or ""):
            seconds = int(float(m.group("n")) * _TIMEOUT_UNIT_SECONDS[m.group("unit")])
            if best is None or seconds > best:
                best = seconds
        return best

    def _command_deadline_sec(self, command: str) -> int:
        requested = self._requested_timeout_sec(command)
        if requested is None:
            return self.command_timeout
        granted = requested + _TIMEOUT_REQUEST_MARGIN_SECONDS
        cap = max(self.max_command_timeout, self.command_timeout)
        return min(max(self.command_timeout, granted), cap)

    async def _execute_bash(
        self, command: str, environment: BaseEnvironment
    ) -> Any:
        timeout_sec = self._command_deadline_sec(command)
        try:
            return await environment.exec(
                command=self._wrap_command(command),
                timeout_sec=timeout_sec,
            )
        except (asyncio.TimeoutError, RuntimeError) as exc:
            # harbor's docker environment RAISES on exec timeout (RuntimeError
            # "Command timed out after Ns") rather than returning a result —
            # unhandled, one slow command would end the whole run. Convert to
            # a model-visible result instead; anything else is a real
            # environment failure and still propagates.
            if not _looks_like_exec_timeout(exc):
                raise
            logger.warning("Command timed out after %ss: %s", timeout_sec, command[:120])
            return _timed_out_result(timeout_sec)

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
