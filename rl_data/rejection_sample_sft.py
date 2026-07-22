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

Input formats accepted (see ``load_records``):
  * rl_data ``*_summary.json`` from ``run_n_solutions`` — a dict with
    ``results: [{success/reward, messages, ...}]``.
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


def harvest(records: Iterable[Record], config: HarvestConfig, stats: HarvestStats | None = None) -> list[dict]:
    """Filter passing records into cleaned, deduped SFT examples.

    Deterministic: for a per-task cap it keeps the SHORTEST passing
    trajectories (cheapest correct demonstrations, and least likely to include
    flailing), breaking ties by dedup key for stability.
    """
    stats = stats or HarvestStats()
    # bucket candidate (clean, passing) trajectories per task
    candidates: dict[str, list[tuple[int, str, list[dict]]]] = defaultdict(list)
    seen_keys: set[str] = set()

    for rec in records:
        stats.records_seen += 1
        if rec.reward < 1.0:
            continue
        stats.passing += 1
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
    out: list[str] = []
    for i in inputs:
        p = Path(i)
        if p.is_dir():
            out += [str(q) for q in sorted(p.rglob("*_summary.json"))]
            out += [str(q) for q in sorted(p.rglob("*.jsonl"))]
        else:
            out.append(i)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("inputs", nargs="+", help="rl_data *_summary.json / JSONL files or dirs to harvest")
    ap.add_argument("--out", required=True, help="output JSONL path for the SFT dataset")
    ap.add_argument("--max-per-task", type=int, default=4)
    ap.add_argument("--max-tokens", type=int, default=32000)
    ap.add_argument("--min-assistant-turns", type=int, default=1)
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
    print(json.dumps(report, indent=2))
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
