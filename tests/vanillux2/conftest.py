"""Shared fixtures for the self-test gate tests.

``FakeEnvironment`` duck-types ``harbor.environments.base.BaseEnvironment``'s
``exec()`` (the only method the self-test gate + the baseline bash loop
call) as a real subprocess against a temp directory — our test "container"
*is* the local filesystem, so a remote path is just a real path on disk. This
exercises the actual commands (mkdir/cp/cat/heredocs) the gate generates,
without needing Docker.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path

import pytest


@dataclass
class FakeExecResult:
    """Duck-types harbor.environments.base.ExecResult."""

    stdout: str | None = None
    stderr: str | None = None
    return_code: int = 0


class FakeEnvironment:
    """Runs commands as real subprocesses against a temp dir standing in for a container."""

    def __init__(self, root: Path) -> None:
        self.root = root

    async def exec(
        self,
        command: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_sec: int | None = None,
        user: str | int | None = None,
    ) -> FakeExecResult:
        workdir = cwd or str(self.root)
        Path(workdir).mkdir(parents=True, exist_ok=True)
        proc = await asyncio.create_subprocess_shell(
            command,
            cwd=workdir,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout_b, stderr_b = await asyncio.wait_for(
                proc.communicate(), timeout=timeout_sec or 30
            )
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            return FakeExecResult(stdout="", stderr="timed out", return_code=124)
        return FakeExecResult(
            stdout=stdout_b.decode("utf-8", errors="replace"),
            stderr=stderr_b.decode("utf-8", errors="replace"),
            return_code=proc.returncode or 0,
        )


@pytest.fixture
def fake_env(tmp_path):
    root = tmp_path / "container_root"
    root.mkdir()
    return FakeEnvironment(root)


def make_session_exec(environment: FakeEnvironment):
    async def _exec(command: str):
        return await environment.exec(command)

    return _exec


def make_isolated_exec(environment: FakeEnvironment):
    async def _exec(command: str, cwd: str):
        return await environment.exec(command, cwd=cwd)

    return _exec


@pytest.fixture
def session_exec(fake_env):
    return make_session_exec(fake_env)


@pytest.fixture
def isolated_exec(fake_env):
    return make_isolated_exec(fake_env)
