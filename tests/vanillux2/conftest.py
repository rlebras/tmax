"""Test fixtures for context_management.py / edit_tools.py.

These two modules are imported by *bare* module name (not
``Vanillux2Agent.context_management``) by inserting ``Vanillux2Agent/`` onto
``sys.path`` below. That's deliberate: going through the ``Vanillux2Agent``
package would execute ``Vanillux2Agent/__init__.py``, which imports
``agent.py``, which imports ``harbor`` — a heavy dependency with native
extensions that isn't needed to exercise the pure compaction/edit-tool logic
these tests cover, and may not be installed in every dev environment.

``exec_fn`` runs real bash against a temp directory rather than mocking the
shell, so the tests exercise the actual heredoc/atomic-write/sed-nl commands
these modules generate, not a hand-rolled fake shell.
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


@pytest.fixture
def workdir(tmp_path):
    return tmp_path


@pytest.fixture
def exec_fn(tmp_path):
    return make_exec_fn(tmp_path)
