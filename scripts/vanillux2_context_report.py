#!/usr/bin/env python3
"""Measure Vanillux2Agent's context-management compaction against real runs.

Replays real ``trajectory.json`` files already on disk under
``evaluation_assets/`` (no live model or environment needed — this is an
offline replay) turn-by-turn through
``context_management.build_model_messages``, and reports peak model-facing
token count before vs. after compaction.

"Before" is what today's harness would have sent at that turn (the raw
prefix, verbatim — matching the harness's behavior prior to this change).
"After" is what ``build_model_messages`` rebuilds from the same prefix.
Both are counted the same way (``litellm.token_counter``), so the
before/after *ratio* is meaningful even though the absolute numbers are an
estimate for a served model litellm doesn't have a exact tokenizer for.

Usage:
    uv run python scripts/vanillux2_context_report.py --limit 30
    uv run python scripts/vanillux2_context_report.py --glob 'evaluation_assets/daytona/tmax-9b/**/trajectory.json'

Caveat: this is a replay, not a re-run. It cannot tell you whether a task
that failed due to context run-out would now pass — that requires a live
rerun against the model/environment. What it does show is how much of the
model-facing context window compaction reclaims on real trajectories, which
is the mechanism the acceptance criteria target.
"""

from __future__ import annotations

import argparse
import asyncio
import glob
import json
import statistics
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "Vanillux2Agent"))
import context_management as cm  # noqa: E402


class _Result:
    def __init__(self, stdout: str, stderr: str, return_code: int) -> None:
        self.stdout = stdout
        self.stderr = stderr
        self.return_code = return_code


def _make_replay_exec_fn(spill_root: Path):
    """Spills get physically written under *spill_root* during replay (no
    live container to write into) so build_model_messages' disk-write side
    effect has somewhere real to go; the spill content itself isn't used by
    this report, only the resulting token counts."""

    async def exec_fn(command: str) -> _Result:
        proc = await asyncio.create_subprocess_shell(
            command,
            cwd=str(spill_root),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        out, err = await proc.communicate()
        return _Result(out.decode("utf-8", errors="replace"), err.decode("utf-8", errors="replace"), proc.returncode or 0)

    return exec_fn


def _count_messages_tokens(messages: list[dict], model: str) -> int:
    import litellm

    try:
        return litellm.token_counter(model=model, messages=messages)
    except Exception:
        return sum(cm.count_tokens(json.dumps(m, default=str), model) for m in messages)


async def replay_trajectory(path: Path, config: cm.CompactionConfig, model: str, spill_root: Path) -> dict:
    raw = json.loads(path.read_text())
    exec_fn = _make_replay_exec_fn(spill_root)

    async def upload_bytes(content: bytes, remote_path: str) -> None:
        # Replay "container" is the local filesystem, rooted at spill_root
        # for relative paths (mirrors tests/vanillux2/conftest.py's ops).
        target = Path(remote_path)
        if not target.is_absolute():
            target = spill_root / target
        target.write_bytes(content)

    async def download_bytes(remote_path: str) -> bytes:
        target = Path(remote_path)
        if not target.is_absolute():
            target = spill_root / target
        return target.read_bytes()

    from container_ops import ContainerOps

    ops = ContainerOps(exec_fn=exec_fn, upload_bytes=upload_bytes, download_bytes=download_bytes)
    spilled_indices: set[int] = set()
    stats = cm.CompactionStats()

    peak_before = 0
    peak_after = 0
    turns = 0

    i = 0
    n = len(raw)
    while i < n:
        if raw[i].get("role") != "assistant":
            i += 1
            continue
        j = i + 1
        while j < n and raw[j].get("role") == "tool":
            j += 1
        prefix = raw[:j]

        before = _count_messages_tokens(prefix, model)
        rebuilt = await cm.build_model_messages(
            prefix,
            config=config,
            model=model,
            ops=ops,
            spilled_indices=spilled_indices,
            stats=stats,
        )
        after = _count_messages_tokens(rebuilt, model)

        peak_before = max(peak_before, before)
        peak_after = max(peak_after, after)
        turns += 1
        i = j

    reward_path = path.parent.parent / "verifier" / "reward.txt"
    reward = reward_path.read_text().strip() if reward_path.is_file() else None

    return {
        "path": str(path.relative_to(REPO_ROOT)),
        "turns": turns,
        "peak_before": peak_before,
        "peak_after": peak_after,
        "reduction_pct": round(100 * (1 - peak_after / peak_before), 1) if peak_before else 0.0,
        "stubbed_writes": stats.stubbed_writes,
        "truncated_outputs": stats.truncated_outputs,
        "reward": reward,
    }


def _select_trajectories(pattern: str, limit: int, select_by: str) -> list[Path]:
    paths = [Path(p) for p in glob.glob(pattern, recursive=True)]
    if select_by == "turns":
        # Most messages first — a proxy for runs that used the full step
        # budget (max_steps=64 -> 130 messages), the likeliest candidates
        # for having hit (or nearly hit) the context ceiling.
        paths.sort(key=lambda p: len(json.loads(p.read_text())), reverse=True)
    else:
        # Largest on-disk trajectory.json first — correlates with more
        # inline file content, the dominant bloat source per the failure
        # analysis.
        paths.sort(key=lambda p: p.stat().st_size, reverse=True)
    return paths[:limit]


async def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--glob",
        default="evaluation_assets/daytona/tmax-9b/terminal-bench-2-0/**/trajectory.json",
        help="glob (relative to repo root) selecting trajectory.json files to replay",
    )
    ap.add_argument("--limit", type=int, default=30, help="max number of trajectories to replay")
    ap.add_argument("--model", default="hosted_vllm/tmax-9b-tb21-64k", help="model id for token counting")
    ap.add_argument("--context-window", type=int, default=65536, help="context window to check peak_after against")
    ap.add_argument(
        "--select-by",
        choices=["size", "turns"],
        default="size",
        help="which trajectories to prioritize: largest on-disk file, or most messages (proxy for step-budget-exhausted runs)",
    )
    args = ap.parse_args()

    paths = _select_trajectories(str(REPO_ROOT / args.glob), args.limit, args.select_by)
    if not paths:
        print(f"no trajectories matched {args.glob!r}", file=sys.stderr)
        sys.exit(1)

    config = cm.CompactionConfig()

    import tempfile

    rows = []
    with tempfile.TemporaryDirectory() as spill_root:
        for path in paths:
            rows.append(await replay_trajectory(path, config, args.model, Path(spill_root)))

    print(f"Replayed {len(rows)} trajectories (model={args.model}, largest-on-disk-first)\n")
    header = f"{'trial':<45} {'turns':>5} {'before':>8} {'after':>8} {'reduce%':>8} {'stub':>5} {'trunc':>5} {'reward':>6}"
    print(header)
    print("-" * len(header))
    for r in rows:
        name = Path(r["path"]).parent.parent.name
        print(
            f"{name:<45} {r['turns']:>5} {r['peak_before']:>8} {r['peak_after']:>8} "
            f"{r['reduction_pct']:>7.1f}% {r['stubbed_writes']:>5} {r['truncated_outputs']:>5} "
            f"{r['reward'] or '-':>6}"
        )

    befores = [r["peak_before"] for r in rows]
    afters = [r["peak_after"] for r in rows]
    reductions = [r["reduction_pct"] for r in rows]
    over_before = sum(1 for b in befores if b > args.context_window)
    over_after = sum(1 for a in afters if a > args.context_window)

    print()
    print("=== Summary ===")
    print(f"peak_before tokens: mean={statistics.mean(befores):.0f} median={statistics.median(befores):.0f} max={max(befores)}")
    print(f"peak_after  tokens: mean={statistics.mean(afters):.0f} median={statistics.median(afters):.0f} max={max(afters)}")
    print(f"mean peak reduction: {statistics.mean(reductions):.1f}%  median: {statistics.median(reductions):.1f}%")
    print(f"runs with peak > {args.context_window} tokens: before={over_before}/{len(rows)}  after={over_after}/{len(rows)}")
    print(f"total stubbed writes: {sum(r['stubbed_writes'] for r in rows)}")
    print(f"total truncated outputs: {sum(r['truncated_outputs'] for r in rows)}")
    print()
    print(
        "Note: this is a replay of already-completed runs, not a re-run — it "
        "cannot confirm a context-run-out failure would now pass. It measures "
        "how much of the model-facing window compaction reclaims on real "
        "trajectories, which is the mechanism the acceptance criteria target."
    )


if __name__ == "__main__":
    asyncio.run(main())
