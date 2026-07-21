"""Test fixtures for context_management.py / edit_tools.py.

These two modules (plus container_ops.py) are imported by *bare* module name
(not ``Vanillux2Agent.context_management``) by inserting ``Vanillux2Agent/``
onto ``sys.path`` below. That's deliberate: going through the
``Vanillux2Agent`` package would execute ``Vanillux2Agent/__init__.py``,
which imports ``agent.py``, which imports ``harbor`` — a heavy dependency
with native extensions that isn't needed to exercise the pure
compaction/edit-tool logic these tests cover, and may not be installed in
every dev environment.

``ops.exec_fn`` runs real bash against a temp directory rather than mocking
the shell. ``ops.upload_bytes``/``download_bytes`` simulate harbor's
``environment.upload_file``/``download_file`` (``docker cp``) as plain local
file writes/reads against that same temp directory — our test "container"
*is* the local filesystem, so a remote path is just a real path on disk.
This exercises the actual commands/paths these modules generate end to end.
"""

from __future__ import annotations

import asyncio
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

_VANILLUX2_DIR = Path(__file__).resolve().parents[2] / "Vanillux2Agent"
if str(_VANILLUX2_DIR) not in sys.path:
    sys.path.insert(0, str(_VANILLUX2_DIR))

from container_ops import ContainerOps  # noqa: E402


@dataclass
class FakeExecResult:
    """Duck-types harbor.environments.base.ExecResult without importing harbor."""

    stdout: str | None = None
    stderr: str | None = None
    return_code: int = 0


def make_exec_fn(cwd: Path):
    async def exec_fn(command: str) -> FakeExecResult:
        proc = await asyncio.create_subprocess_shell(
            command,
            cwd=str(cwd),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout_b, stderr_b = await proc.communicate()
        return FakeExecResult(
            stdout=stdout_b.decode("utf-8", errors="replace"),
            stderr=stderr_b.decode("utf-8", errors="replace"),
            return_code=proc.returncode or 0,
        )

    return exec_fn


def make_ops(cwd: Path) -> ContainerOps:
    exec_fn = make_exec_fn(cwd)

    async def upload_bytes(content: bytes, remote_path: str) -> None:
        # Mirrors `docker cp`: writes exact bytes, does NOT create missing
        # parent directories (Path.write_bytes raises FileNotFoundError for
        # those, same as a real `docker cp` into a nonexistent directory).
        Path(remote_path).write_bytes(content)

    async def download_bytes(remote_path: str) -> bytes:
        p = Path(remote_path)
        if not p.is_file():
            raise FileNotFoundError(remote_path)
        return p.read_bytes()

    return ContainerOps(exec_fn=exec_fn, upload_bytes=upload_bytes, download_bytes=download_bytes)


@pytest.fixture
def workdir(tmp_path):
    return tmp_path


@pytest.fixture
def exec_fn(tmp_path):
    return make_exec_fn(tmp_path)


@pytest.fixture
def ops(tmp_path):
    return make_ops(tmp_path)


def make_isolated_exec():
    """Matches self_test.py's ``IsolatedExecFn`` contract: ``(command, cwd) -> ExecResult``,
    a bare subprocess with no persistent-shell state — unlike ``make_exec_fn``'s cwd, which
    is fixed once, this one takes cwd per call since self_test.py picks a fresh isolated
    directory for every check.
    """

    async def isolated_exec(command: str, cwd: str) -> FakeExecResult:
        proc = await asyncio.create_subprocess_shell(
            command,
            cwd=cwd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout_b, stderr_b = await proc.communicate()
        return FakeExecResult(
            stdout=stdout_b.decode("utf-8", errors="replace"),
            stderr=stderr_b.decode("utf-8", errors="replace"),
            return_code=proc.returncode or 0,
        )

    return isolated_exec


@pytest.fixture
def isolated_exec_fn():
    return make_isolated_exec()
