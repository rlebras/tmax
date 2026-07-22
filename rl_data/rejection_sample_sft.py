"""Rejection-sampling SFT data harvest (STaR-style) for tmax terminal agents.

The loop this feeds:

  1. Serve the current checkpoint (e.g. tmax-9b) and run the vanillux solver
     over the rl_data TRAINING corpus at k rollouts/task (generate_solutions).
  2. --> THIS MODULE: keep only the passing (reward==1) trajectories, clean
     and dedup them, and emit the open-instruct SFT schema (messages + tools).
  3. SFT the base model on the harvested data (open_instruct/finetune.py).
  4. Evaluate on terminal-bench (held out).

Why rejection sampling helps here: on terminal-bench the model already solves
many tasks *sometimes* (pass@5 >> pass@1) — it can produce correct
trajectories but not reliably. Fine-tuning on its own verified successes
turns "occasionally reachable" behavior into the default, converting pass@k
capability into pass@1 reliability without any human labels.

CONTAMINATION GUARD (do not remove)
-----------------------------------
The training signal MUST come from the rl_data generated corpus, which the
pipeline decontaminates against terminal-bench (see
rl_data/decontamination/). Harvesting from the terminal-bench *eval*
trajectories and then re-evaluating on terminal-bench is training on the test
set — it inflates the score without improving the model. ``load_records``
refuses inputs whose path looks like a terminal-bench eval unless the caller
passes ``allow_eval_source=True`` (only ever legitimate for building test
fixtures, never for real training data).

Which tasks are worth harvesting (``--max-solve-rate``)
-------------------------------------------------------
Rejection-sampling SFT only helps on tasks the model solves *sometimes but
not reliably*. A task it already passes on every attempt (solve-rate 1.0) has
nothing to teach — its trajectories are redundant and, being easy, tend to be
short and plentiful, so they crowd the mixture. A task it never solves
(solve-rate 0.0) has nothing to harvest. The payoff band is ``0 < solve_rate
< 1``. The harvester computes each task's solve-rate across all its attempts,
prints the distribution, and ``--max-solve-rate`` drops already-reliable tasks
so training concentrates on the flaky band. (Complement at rollout time: spend
more samples on the flaky tasks — see the rollout launcher.)

Input formats accepted (see ``load_records``):
  * rl_data ``*_summary.json`` from ``run_n_solutions`` — a dict with
    ``results: [{success/reward, messages, ...}]``.
  * a **harbor rollout dir** — ``<root>/<task>/{result.json, agent/trajectory.json}``
    (what ``harbor run -d tmax/TMax-15K-Harbor --agent Vanillux2Agent``
    produces, i.e. the same harness path as the eval, on the canonical
    decontaminated training corpus). One trial dir = one attempt; the task id
    is the dir name with its trailing ``__<hash>`` stripped so the k attempts
    of a task group together for the solve-rate.
  * a JSONL where each line is ``{task_id, reward, messages}``.

Output: a JSONL of ``{messages, tools, dataset, id}`` rows ready for
open-instruct's ``--dataset_mixer_list`` (optionally pushed to the Hub).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator

# The submit sentinel a successful trajectory must end on (kept in sync with
# rl_data.generator.sample_solutions.SUBMIT_MARKER — duplicated here so this
# module has no heavy import just to read a constant).
SUBMIT_MARKER = "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"

# The bash tool definition the vanillux harness exposes; emitted as the SFT
# ``tools`` column so the chat template renders tool calls consistently.
BASH_TOOL = {
    "type": "function",
    "function": {
        "name": "bash",
        "description": "Execute a bash command. Each command runs in a new subshell.",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "The bash command to execute."}
            },
            "required": ["command"],
        },
    },
}

# Path fragments that mark an input as a terminal-bench EVAL artifact rather
# than a training rollout. Harvesting these as SFT data is test-set leakage.
_EVAL_SOURCE_MARKERS = (
    "terminal-bench",
    "terminal_bench",
    "tb21",
    "tb-21",
    "tb20",
    "tb-20",
    "eval-tmax",
    "evaluation_assets",
)


@dataclass
class HarvestConfig:
    max_per_task: int = 4            # cap distinct kept trajectories per task
    max_tokens: int = 32000          # drop trajectories longer than this (~ SFT max_seq_length)
    drop_format_errors: bool = True  # drop trajectories containing a malformed/no-tool-call turn
    require_submit: bool = True      # keep only trajectories that end on the submit sentinel
    min_assistant_turns: int = 1     # drop trivial trajectories with fewer than this many actions
    # Relevance band: keep a task only if 0 < its solve-rate <= max_solve_rate.
    # max_solve_rate=1.0 keeps everything solvable; lower it (e.g. 0.8) to drop
    # already-reliable tasks and concentrate on the flaky payoff band.
    max_solve_rate: float = 1.0
    tools: list = field(default_factory=lambda: [BASH_TOOL])


@dataclass
class HarvestStats:
    files_read: int = 0
    records_seen: int = 0
    passing: int = 0
    kept: int = 0
    dropped_format_error: int = 0
    dropped_no_submit: int = 0
    dropped_too_long: int = 0
    dropped_too_short: int = 0
    dropped_dup: int = 0
    dropped_over_cap: int = 0
    dropped_over_solve_rate: int = 0   # passing trajectories skipped because their task is already reliable
    tasks_seen: int = 0
    tasks_solvable: int = 0            # tasks with >=1 success (harvestable at all)
    tasks_over_solve_rate: int = 0     # solvable tasks skipped by max_solve_rate
    solve_rate_hist: dict = field(default_factory=dict)  # coarse solve-rate buckets over solvable tasks
    per_task_kept: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        d = dict(self.__dict__)
        d["n_tasks_kept"] = len([t for t, c in self.per_task_kept.items() if c])
        return d


@dataclass
class Record:
    task_id: str
    reward: float
    messages: list[dict]
    source: str


# ---------------------------------------------------------------------------
# Loading (with the contamination guard)
# ---------------------------------------------------------------------------


def _looks_like_eval_source(path: str) -> bool:
    low = path.lower()
    return any(m in low for m in _EVAL_SOURCE_MARKERS)


def _reward_of(obj: dict) -> float:
    if "reward" in obj and obj["reward"] is not None:
        try:
            return float(obj["reward"])
        except (TypeError, ValueError):
            return 0.0
    # rl_data run_n_solutions results also carry a boolean ``success``.
    if obj.get("success") is True:
        return 1.0
    return 0.0


def _task_base(name: str) -> str:
    """Strip a harbor trial dir's trailing ``__<hash>`` so the k attempts of a
    task share one task id for the solve-rate."""
    return name.rsplit("__", 1)[0] if "__" in name else name


def _harbor_reward(result: dict) -> float:
    rewards = (result.get("verifier_result") or {}).get("rewards") or {}
    r = rewards.get("reward")
    try:
        return float(r) if r is not None else 0.0
    except (TypeError, ValueError):
        return 0.0


def _record_from_harbor_trial(trial_dir: Path) -> Record | None:
    """One harbor trial dir (``result.json`` + ``agent/trajectory.json``) ->
    one attempt Record. Returns None if either file is missing/malformed."""
    rj = trial_dir / "result.json"
    tj = trial_dir / "agent" / "trajectory.json"
    if not (rj.is_file() and tj.is_file()):
        return None
    try:
        result = json.loads(rj.read_text())
        messages = json.loads(tj.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(messages, list):
        return None
    return Record(_task_base(trial_dir.name), _harbor_reward(result), messages, str(trial_dir))


def _records_from_summary(doc: dict, source: str, task_id: str | None) -> Iterator[Record]:
    results = doc.get("results")
    if not isinstance(results, list):
        return
    tid = task_id or str(doc.get("task_id") or doc.get("task") or source)
    for r in results:
        if not isinstance(r, dict):
            continue
        msgs = r.get("messages")
        if isinstance(msgs, list):
            yield Record(tid, _reward_of(r), msgs, source)


def load_records(
    paths: Iterable[str | Path], *, allow_eval_source: bool = False, stats: HarvestStats | None = None
) -> Iterator[Record]:
    """Yield ``Record``s from rl_data summary JSON or JSONL files.

    Refuses eval-sourced paths unless ``allow_eval_source`` (see the module
    docstring's contamination guard). Never raises on a malformed file — it is
    skipped with a warning to stderr.
    """
    for p in paths:
        p = Path(p)
        if not allow_eval_source and _looks_like_eval_source(str(p)):
            raise ValueError(
                f"refusing eval-sourced input {p!r}: harvesting terminal-bench eval "
                "trajectories as SFT data is test-set leakage. Point this at rl_data "
                "training-corpus solutions, or pass allow_eval_source=True only for "
                "building test fixtures."
            )
        if stats is not None:
            stats.files_read += 1
        # A harbor trial dir (result.json + agent/trajectory.json).
        if p.is_dir():
            rec = _record_from_harbor_trial(p)
            if rec is not None:
                yield rec
            else:
                print(f"warning: {p} is not a harbor trial dir; skipped", file=sys.stderr)
            continue
        try:
            text = p.read_text()
        except OSError as exc:
            print(f"warning: cannot read {p}: {exc}", file=sys.stderr)
            continue
        if p.suffix == ".jsonl":
            for line in text.splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(obj, dict) and isinstance(obj.get("messages"), list):
                    yield Record(
                        str(obj.get("task_id") or obj.get("id") or p.stem),
                        _reward_of(obj),
                        obj["messages"],
                        str(p),
                    )
            continue
        try:
            doc = json.loads(text)
        except json.JSONDecodeError:
            print(f"warning: {p} is not valid JSON; skipped", file=sys.stderr)
            continue
        if isinstance(doc, dict) and "results" in doc:
            yield from _records_from_summary(doc, str(p), None)
        elif isinstance(doc, dict) and isinstance(doc.get("messages"), list):
            yield Record(str(doc.get("task_id") or p.stem), _reward_of(doc), doc["messages"], str(p))


# ---------------------------------------------------------------------------
# Cleaning / filtering
# ---------------------------------------------------------------------------


def _safe_json_loads(raw: Any) -> Any:
    if isinstance(raw, (dict, list)):
        return raw
    try:
        return json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        return None


def _assistant_turns(messages: list[dict]) -> list[dict]:
    return [m for m in messages if m.get("role") == "assistant"]


def has_format_error(messages: list[dict]) -> bool:
    """True if any assistant turn is a malformed action: no tool_calls at all,
    or a tool_call whose arguments don't parse as JSON. These are the
    format-error-recovery turns; training on them teaches the failure."""
    for m in _assistant_turns(messages):
        tcs = m.get("tool_calls")
        if not tcs:
            return True
        for tc in tcs:
            args = ((tc.get("function") or {}).get("arguments"))
            if _safe_json_loads(args) is None:
                return True
    return False


def ends_with_submit(messages: list[dict]) -> bool:
    """True if the trajectory's final action issues the submit sentinel."""
    for m in reversed(messages):
        if m.get("role") == "assistant" and m.get("tool_calls"):
            for tc in m["tool_calls"]:
                args = _safe_json_loads((tc.get("function") or {}).get("arguments")) or {}
                if SUBMIT_MARKER in str(args.get("command", "")):
                    return True
            return False  # last action wasn't a submit
    return False


def estimate_tokens(messages: list[dict]) -> int:
    """Cheap char/4 token estimate over content + tool-call arguments (no
    tokenizer dependency; only used for a length cutoff)."""
    total = 0
    for m in messages:
        total += len(str(m.get("content") or ""))
        for tc in m.get("tool_calls") or []:
            total += len(str((tc.get("function") or {}).get("arguments") or ""))
    return total // 4


_ID_FIELDS_KEEP = ("role", "content", "tool_calls", "tool_call_id", "name", "reasoning_content")


def sanitize_messages(messages: list[dict]) -> list[dict]:
    """Strip harness/provider-internal fields, keeping only what the chat
    template needs. Tool calls are reduced to {id, type, function{name,
    arguments}} with arguments re-serialized to a compact JSON string."""
    out: list[dict] = []
    for m in messages:
        clean = {k: m[k] for k in _ID_FIELDS_KEEP if k in m and m[k] is not None}
        if clean.get("content") is None:
            clean["content"] = ""
        tcs = m.get("tool_calls")
        if tcs:
            new_tcs = []
            for tc in tcs:
                func = tc.get("function") or {}
                args = _safe_json_loads(func.get("arguments"))
                new_tcs.append(
                    {
                        "id": tc.get("id", ""),
                        "type": "function",
                        "function": {
                            "name": func.get("name", ""),
                            "arguments": json.dumps(args if args is not None else {}),
                        },
                    }
                )
            clean["tool_calls"] = new_tcs
        out.append(clean)
    return out


def _dedup_key(task_id: str, messages: list[dict]) -> str:
    # Key on the sequence of assistant commands, so two rollouts that took the
    # same actions collapse even if observations differ slightly.
    actions = []
    for m in _assistant_turns(messages):
        for tc in m.get("tool_calls") or []:
            args = _safe_json_loads((tc.get("function") or {}).get("arguments")) or {}
            actions.append(str(args.get("command", "")))
    blob = task_id + " |#| " + " ; ".join(actions)
    return hashlib.sha1(blob.encode("utf-8", "replace")).hexdigest()


def to_sft_example(task_id: str, messages: list[dict], tools: list) -> dict:
    return {
        "messages": sanitize_messages(messages),
        "tools": tools,
        "dataset": "tmax-rejection-sft",
        "id": task_id,
    }


def _solve_rate_bucket(rate: float) -> str:
    if rate >= 1.0:
        return "1.0 (always)"
    if rate >= 0.75:
        return "[0.75,1.0)"
    if rate >= 0.5:
        return "[0.5,0.75)"
    if rate >= 0.25:
        return "[0.25,0.5)"
    return "(0,0.25)"


def harvest(records: Iterable[Record], config: HarvestConfig, stats: HarvestStats | None = None) -> list[dict]:
    """Filter passing records into cleaned, deduped SFT examples.

    Two passes. First, compute each task's solve-rate across ALL its attempts
    and keep only tasks in the relevant band (``0 < rate <= max_solve_rate``) —
    rejection sampling has nothing to teach on always-solved tasks and nothing
    to harvest on never-solved ones. Second, from the in-band tasks, filter the
    passing trajectories (clean, submit, length), dedup, and cap per task
    (SHORTEST first — cheapest correct demonstrations, least flailing).
    Deterministic given the same records.
    """
    stats = stats or HarvestStats()
    records = list(records)

    # Pass 1 — per-task solve-rate over all attempts.
    totals: dict[str, int] = defaultdict(int)
    successes: dict[str, int] = defaultdict(int)
    for rec in records:
        totals[rec.task_id] += 1
        if rec.reward >= 1.0:
            successes[rec.task_id] += 1
    stats.tasks_seen = len(totals)
    focus: set[str] = set()
    for tid, n in totals.items():
        rate = successes[tid] / n if n else 0.0
        if successes[tid] == 0:
            continue  # never solved: nothing to harvest
        stats.tasks_solvable += 1
        stats.solve_rate_hist[_solve_rate_bucket(rate)] = (
            stats.solve_rate_hist.get(_solve_rate_bucket(rate), 0) + 1
        )
        if rate > config.max_solve_rate:
            stats.tasks_over_solve_rate += 1
            continue  # already reliable: skip the redundant demonstrations
        focus.add(tid)

    # Pass 2 — harvest passing trajectories from in-band tasks only.
    candidates: dict[str, list[tuple[int, str, list[dict]]]] = defaultdict(list)
    seen_keys: set[str] = set()

    for rec in records:
        stats.records_seen += 1
        if rec.reward < 1.0:
            continue
        stats.passing += 1
        if rec.task_id not in focus:
            stats.dropped_over_solve_rate += 1
            continue
        msgs = rec.messages
        if config.drop_format_errors and has_format_error(msgs):
            stats.dropped_format_error += 1
            continue
        if config.require_submit and not ends_with_submit(msgs):
            stats.dropped_no_submit += 1
            continue
        if len(_assistant_turns(msgs)) < config.min_assistant_turns:
            stats.dropped_too_short += 1
            continue
        if estimate_tokens(msgs) > config.max_tokens:
            stats.dropped_too_long += 1
            continue
        key = _dedup_key(rec.task_id, msgs)
        if key in seen_keys:
            stats.dropped_dup += 1
            continue
        seen_keys.add(key)
        candidates[rec.task_id].append((estimate_tokens(msgs), key, msgs))

    examples: list[dict] = []
    for task_id, cands in candidates.items():
        cands.sort(key=lambda c: (c[0], c[1]))  # shortest first, then stable by key
        kept = cands[: config.max_per_task]
        stats.dropped_over_cap += len(cands) - len(kept)
        for _tok, _key, msgs in kept:
            examples.append(to_sft_example(task_id, msgs, config.tools))
        stats.per_task_kept[task_id] = len(kept)
    stats.kept = len(examples)
    return examples


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _expand_inputs(inputs: list[str]) -> list[str]:
    """Expand dirs into concrete inputs: rl_data ``*_summary.json`` / ``*.jsonl``
    files, and harbor trial dirs (each dir holding ``result.json`` +
    ``agent/trajectory.json``). A path that is itself a harbor trial dir is
    kept as-is."""
    out: list[str] = []
    for i in inputs:
        p = Path(i)
        if not p.is_dir():
            out.append(i)
            continue
        if (p / "result.json").is_file() and (p / "agent" / "trajectory.json").is_file():
            out.append(str(p))  # a single trial dir
            continue
        out += [str(q) for q in sorted(p.rglob("*_summary.json"))]
        out += [str(q) for q in sorted(p.rglob("*.jsonl"))]
        # harbor trial dirs nested anywhere under p
        seen = set(out)
        for rj in sorted(p.rglob("result.json")):
            trial = rj.parent
            if (trial / "agent" / "trajectory.json").is_file() and str(trial) not in seen:
                out.append(str(trial))
                seen.add(str(trial))
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("inputs", nargs="+", help="rl_data *_summary.json / JSONL files or dirs to harvest")
    ap.add_argument("--out", required=True, help="output JSONL path for the SFT dataset")
    ap.add_argument("--max-per-task", type=int, default=4)
    ap.add_argument("--max-tokens", type=int, default=32000)
    ap.add_argument("--min-assistant-turns", type=int, default=1)
    ap.add_argument(
        "--max-solve-rate",
        type=float,
        default=1.0,
        help="keep only tasks with solve-rate <= this (0<rate). 1.0 keeps all "
        "solvable tasks; lower (e.g. 0.8) to focus on the flaky payoff band and "
        "drop already-reliable tasks.",
    )
    ap.add_argument("--keep-format-errors", action="store_true", help="do NOT drop malformed-action trajectories")
    ap.add_argument("--no-require-submit", action="store_true", help="keep trajectories not ending on the submit marker")
    ap.add_argument(
        "--allow-eval-source",
        action="store_true",
        help="DANGER: permit terminal-bench eval trajectories as input (test-set leakage; fixtures only)",
    )
    ap.add_argument("--stats-out", default=None, help="optional path to write harvest stats JSON")
    ap.add_argument(
        "--push-to-hub",
        default=None,
        metavar="REPO_ID",
        help="also push the harvested dataset to this HF repo (needs `datasets` + HF auth)",
    )
    ap.add_argument("--hub-private", action="store_true", help="push the hub dataset as private")
    args = ap.parse_args(argv)

    cfg = HarvestConfig(
        max_per_task=args.max_per_task,
        max_tokens=args.max_tokens,
        min_assistant_turns=args.min_assistant_turns,
        max_solve_rate=args.max_solve_rate,
        drop_format_errors=not args.keep_format_errors,
        require_submit=not args.no_require_submit,
    )
    stats = HarvestStats()
    records = load_records(_expand_inputs(args.inputs), allow_eval_source=args.allow_eval_source, stats=stats)
    examples = harvest(records, cfg, stats)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")

    report = stats.as_dict()
    report.pop("per_task_kept", None)
    hist = report.pop("solve_rate_hist", {})
    print(json.dumps(report, indent=2))
    print("\nsolve-rate distribution over solvable tasks (where the harvestable signal is):")
    for bucket in ["(0,0.25)", "[0.25,0.5)", "[0.5,0.75)", "[0.75,1.0)", "1.0 (always)"]:
        if bucket in hist:
            note = "  <- redundant (already reliable)" if bucket == "1.0 (always)" else ""
            print(f"  {bucket:14s} {hist[bucket]}{note}")
    print(f"\nwrote {len(examples)} SFT examples across {report['n_tasks_kept']} tasks to {out}")
    if args.stats_out:
        Path(args.stats_out).write_text(json.dumps(stats.as_dict(), indent=2) + "\n")

    if args.push_to_hub:
        if not examples:
            print("nothing to push (0 examples)", file=sys.stderr)
            return 1
        from datasets import Dataset  # local import: only needed on push

        Dataset.from_list(examples).push_to_hub(args.push_to_hub, private=args.hub_private)
        print(f"pushed {len(examples)} examples to hub: {args.push_to_hub}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
