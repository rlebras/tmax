"""Shared sandbox-access interface for context_management.py and edit_tools.py.

Three operations, backed in agent.py by harbor's ``BaseEnvironment``:

- ``exec_fn(command)`` — run a (small, fixed-size) bash command; backed by
  ``environment.exec()``.
- ``upload_bytes(content, remote_path)`` — write bytes to a path in the
  sandbox; backed by ``environment.upload_file()`` (``docker cp``, not a
  command-line argument — see the module docstring in edit_tools.py for why
  this matters: no ARG_MAX limit, no embedded-null-byte crash, regardless of
  content size or content).
- ``download_bytes(remote_path)`` — read a path's exact bytes back; backed by
  ``environment.download_file()`` (``docker cp`` the other direction — true
  bytes, no lossy text decode).

Kept as a plain dataclass of callables (not importing harbor here) so
context_management.py/edit_tools.py stay testable without harbor installed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Awaitable, Callable

ExecFn = Callable[[str], Awaitable[Any]]
UploadBytesFn = Callable[[bytes, str], Awaitable[None]]
DownloadBytesFn = Callable[[str], Awaitable[bytes]]


@dataclass
class ContainerOps:
    exec_fn: ExecFn
    upload_bytes: UploadBytesFn
    download_bytes: DownloadBytesFn
