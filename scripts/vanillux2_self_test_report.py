#!/usr/bin/env python3
"""Outcome-taxonomy + self-test-gate report over completed Vanillux2Agent runs.

Walks harbor trial directories already on disk (the same layout
``evaluation_assets/`` uses: ``<trial>/result.json`` +
``<trial>/agent/{trajectory,timing,usage}.json`` and, for gate-enabled runs,
``<trial>/agent/self_test.json``) and reports, per run set:

* the **outcome taxonomy** the self-test gate is meant to shift:
  ``submitted_right`` / ``submitted_wrong`` (finished + verifier reward) /
  ``step_exhausted`` / ``stopped_early`` (context run-out, cost limit,
  format-error stop, or crash — indistinguishable post-hoc without logs, so
  bucketed together and split out by exception where recorded);
* **gate/self-test stats**: adoption (declared criteria, ran >=1 check),
  checks by category (behavioral / external_script / trivial_existence /
  circular), session-vs-isolation discrepancies, gate rejections, and the
  ``submitted_with_failing_checks`` rate;
* **overhead**: self-test-attributable steps (declare/check/gate-rejection
  turns in ``timing.json``) and prompt/completion token totals, so an A/B
  against a no-gate baseline shows what the gate costs.

Usage (run once per arm, then compare):
    uv run python scripts/vanillux2_self_test_report.py \
        --glob 'evaluation_assets/daytona/tmax-9b/terminal-bench-2-0/*'
    uv run python scripts/vanillux2_self_test_report.py \
        --glob '<selftest-run-trials>/*' --json selftest_arm.json

Caveat: this is an offline read of finished runs — producing the A/B itself
requires the live Beaker reruns (see scripts/beaker/launch_tmax9b_tb21_*.sh);
the taxonomy here is only as good as the trial artifacts on disk.
"""

from __future__ import annotations

import argparse
import glob
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

SUBMIT_MARKER = "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"

# timing.json keys that mark a self-test-attributable step (see agent.py):
# an intercepted agent-check bash call, a rejected submit attempt, a
# malformed-attempt/existence-probe hint, or a declare_criteria/run_check
# tool turn.
_SELF_TEST_TIMING_KEYS = (
    "agent_check",
    "gate_rejected",
    "agent_check_malformed",
    "agent_check_existence_probe",
)
_SELF_TEST_TOOLS = {"declare_criteria", "run_check"}


def _load(path: Path):
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def classify_trial(trial_dir: Path) -> dict | None:
    result = _load(trial_dir / "result.json")
    trajectory = _load(trial_dir / "agent" / "trajectory.json")
    usage = _load(trial_dir / "agent" / "usage.json") or {}
    timing = _load(trial_dir / "agent" / "timing.json") or []
    if result is None or trajectory is None:
        return None

    rewards = ((result.get("verifier_result") or {}).get("rewards")) or {}
    reward = rewards.get("reward")
    exception = (result.get("exception_info") or {}).get("exception_type") if result.get("exception_info") else None

    tool_msgs = [m for m in trajectory if m.get("role") == "tool"]
    submitted = any(SUBMIT_MARKER in (m.get("content") or "") for m in tool_msgs)
    n_steps = sum(1 for m in trajectory if m.get("role") == "assistant")
    max_steps = usage.get("max_steps")

    if submitted:
        outcome = "submitted_right" if reward and reward > 0 else "submitted_wrong"
    elif max_steps and n_steps >= max_steps:
        outcome = "step_exhausted"
    else:
        outcome = "stopped_early"

    # Gate stats: prefer the agent's own self_test.json (fullest record);
    # fall back to metadata embedded in result.json for runs whose logs dir
    # wasn't archived.
    self_test = _load(trial_dir / "agent" / "self_test.json")
    if self_test is None:
        meta = ((result.get("agent_result") or {}).get("metadata")) or {}
        self_test = meta.get("self_test")

    st_steps = 0
    for entry in timing:
        if any(k in entry for k in _SELF_TEST_TIMING_KEYS) or entry.get("tool") in _SELF_TEST_TOOLS:
            st_steps += 1

    gate: dict = {"enabled": False}
    if self_test:
        checks = self_test.get("checks") or {}
        records = [r for rs in checks.values() for r in rs] if isinstance(checks, dict) else []
        gate = {
            "enabled": bool(self_test.get("enabled", True)),
            "criteria_declared": len(self_test.get("criteria") or {})
            or self_test.get("criteria_declared", 0),
            "checks_run": len(records) or self_test.get("checks_run", 0),
            "checks_by_category": Counter(
                r.get("check_category", "unknown") for r in records
            )
            if records
            else Counter(self_test.get("checks_by_category") or {}),
            "discrepancies": sum(
                1 for r in records if r.get("session_pass") and not r.get("isolated_pass")
            )
            if records
            else self_test.get("session_isolation_discrepancies", 0),
            "gate_rejections": self_test.get("gate_rejections", 0),
            "submitted_with_failing_checks": bool(
                self_test.get("submitted_with_failing_checks", False)
            ),
        }

    return {
        "trial": trial_dir.name,
        "outcome": outcome,
        "reward": reward,
        "exception": exception,
        "n_steps": n_steps,
        "self_test_steps": st_steps,
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "gate": gate,
    }


def _pct(n: int, total: int) -> str:
    return f"{n:4d} ({100.0 * n / total:5.1f}%)" if total else "   0"


def _median(values):
    values = [v for v in values if v is not None]
    return statistics.median(values) if values else None


def report(trials: list[dict]) -> None:
    total = len(trials)
    print(f"trials: {total}")
    print("\n== outcome taxonomy ==")
    outcomes = Counter(t["outcome"] for t in trials)
    for outcome in ("submitted_right", "submitted_wrong", "step_exhausted", "stopped_early"):
        print(f"  {outcome:16s} {_pct(outcomes.get(outcome, 0), total)}")
    exceptions = Counter(t["exception"] for t in trials if t["exception"])
    if exceptions:
        print(f"  (stopped-early exceptions: {dict(exceptions)})")

    gates = [t["gate"] for t in trials if t["gate"].get("enabled")]
    print(f"\n== self-test gate ({len(gates)}/{total} trials gate-enabled) ==")
    if gates:
        declared = [g for g in gates if g["criteria_declared"] > 0]
        checked = [g for g in gates if g["checks_run"] > 0]
        forced = [g for g in gates if g["submitted_with_failing_checks"]]
        print(f"  declared criteria     {_pct(len(declared), len(gates))}")
        print(f"  ran >=1 check         {_pct(len(checked), len(gates))}")
        print(f"  forced submit (failing checks) {_pct(len(forced), len(gates))}")
        print(f"  median criteria declared: {_median([g['criteria_declared'] for g in gates])}")
        print(f"  median checks run:        {_median([g['checks_run'] for g in gates])}")
        print(f"  median gate rejections:   {_median([g['gate_rejections'] for g in gates])}")
        by_category: Counter = Counter()
        for g in gates:
            by_category.update(g["checks_by_category"])
        print(f"  checks by category:       {dict(by_category)}")
        print(
            "  session-pass/isolated-fail discrepancies: "
            f"{sum(g['discrepancies'] for g in gates)}"
        )

    print("\n== budget ==")
    print(f"  median steps:             {_median([t['n_steps'] for t in trials])}")
    print(f"  median self-test steps:   {_median([t['self_test_steps'] for t in trials])}")
    print(f"  median prompt tokens:     {_median([t['prompt_tokens'] for t in trials])}")
    print(f"  median completion tokens: {_median([t['completion_tokens'] for t in trials])}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--glob",
        required=True,
        help="Glob of harbor trial directories (each containing result.json + agent/)",
    )
    parser.add_argument("--limit", type=int, default=None, help="Cap the number of trials read")
    parser.add_argument("--json", default=None, help="Also dump per-trial rows to this JSON file")
    args = parser.parse_args()

    trial_dirs = sorted(p for p in glob.glob(args.glob) if (Path(p) / "result.json").is_file())
    if args.limit:
        trial_dirs = trial_dirs[: args.limit]
    if not trial_dirs:
        print(f"no trial directories matched {args.glob!r}", file=sys.stderr)
        return 1

    trials = [t for t in (classify_trial(Path(d)) for d in trial_dirs) if t is not None]
    report(trials)

    if args.json:
        rows = [dict(t, gate=dict(t["gate"], checks_by_category=dict(t["gate"].get("checks_by_category", {})))) for t in trials]
        Path(args.json).write_text(json.dumps(rows, indent=2, default=str) + "\n")
        print(f"\nwrote {len(rows)} rows to {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
