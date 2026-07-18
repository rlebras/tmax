# Vanillux2Agent context management

*Source:* [`Vanillux2Agent/context_management.py`](../Vanillux2Agent/context_management.py),
[`Vanillux2Agent/edit_tools.py`](../Vanillux2Agent/edit_tools.py) · wired into
[`Vanillux2Agent/agent.py`](../Vanillux2Agent/agent.py)

## Why

`Vanillux2Agent` runs tmax-9b at a 64k-token context window. A failure
analysis of 445 runs found **context run-out is the #1 failure mode
(29.4% of all runs — more than the pass rate)**: the run gets cut off before
it can submit. Of what fills the window on those runs, **49% is the bash
commands themselves** — the agent repeatedly writes whole files inline via
`cat > f << 'EOF' ... EOF` — **22% is tool output**, and 18% is reasoning.

This is a pure harness change: no model, prompt, or training changes. `bash`
still works exactly as before, including for a model that only ever writes
files via `cat > ... << EOF`.

## Architecture

`agent.py` keeps the same `messages` list it always has: an append-only,
never-mutated raw event log, dumped verbatim to `trajectory.json` at the end
of the run (now holding **full, untruncated** tool output, not the old
10k-char-truncated version — the model-facing view is where compaction now
happens instead).

Before every LLM call, `context_management.build_model_messages` rebuilds a
*compacted* model-facing view from that raw log, from scratch, every step.
It is a pure function of `(raw_messages, config, model)` — it never mutates
the raw log, and rebuilding the same raw log twice produces byte-identical
output (aside from a deterministic disk-spill path, not a random one — see
below). This is what makes compaction safe: nothing is ever lost, only
*re-presented* differently to the model; anything elided is always
reconstructable from disk via the `read` tool (or, for spilled tool output,
the spill file path named in the stub).

## Feature 1 — stub superseded file-writes

A file's full text should never linger in the conversation once it's
superseded or stale — it's on disk. Detected file-mutating actions: heredocs
(`cat`/`tee ... << 'EOF'`, in either token order), `tee PATH`, plain
redirects (`> PATH` / `>> PATH`), and the Feature-3 edit-tool calls below.

**Keep-full rule:** a write is shown in full only if it is *both* the latest
write to its path *and* within the trailing `write_recency_keep` turns.
Every other write to a path that was ever written eventually gets replaced
with a stub: `[wrote N lines to PATH — current content on disk; use read to
inspect]`. The command's existence and its result/exit code are always kept
— only the bulky payload (the bash command string, or an edit tool's
`content`/`text`/`new_string` argument) is replaced.

This is a deliberate reading of the two conditions in the original spec
("superseded by a later write" / "older than the last K turns") as ANDed
into one keep-full criterion — a write that's the *only* write to a path
still eventually gets stubbed once enough newer turns pass, on the
assumption the model can always call `read` if it actually needs to see it
again. That's what actually bounds context growth on a long run.

Known limitations (a heuristic detector, not a shell parser): doesn't
understand `sed -i`, Python `open()`, or `[[ a > b ]]` string comparisons
(the latter can rarely false-positive as a redirect). A bash command that
touches multiple paths is only stubbed if *all* of its write targets are
stub-eligible — otherwise the whole command is left alone (conservative, to
avoid corrupting the display of a compound command).

## Feature 2 — truncate tool output with spill-to-disk

Any tool-role message whose content exceeds `max_tool_output_tokens` (token
count via the model's own tokenizer where `litellm.token_counter` can
resolve one, else `chars // 4`) is head/tail-truncated by lines: the first
`head_lines` and last `tail_lines` are kept verbatim (errors cluster at the
end, so the exit code and tail are never touched), and the full original
text is spilled to `<spill_dir>/turn_NNNNN.txt` before being elided. The
spill path is a deterministic function of the raw message's position (not a
random uuid) so that rebuilding the same raw log twice resolves to the same
path — required for the "rebuild is idempotent" invariant above. A
per-instance `spilled_indices` cache avoids re-writing an unchanged spill
file on every step; it's a write-avoidance optimization only, not something
the rebuild's *output* depends on.

A token-heavy but line-sparse message (e.g. one huge minified JSON line) is
left untouched — the head/tail-by-*lines* strategy has nothing useful to cut
in that case.

## Feature 3 — a real edit tool

`str_replace` / `insert` / `create` / `apply_edits` / `read` are registered
alongside `bash`. All run host-side: content is fetched via `exec_fn`
(`cat`/`wc -l`/`sed -n ... | nl`), matched/mutated in Python, and written
back atomically (temp file + `mv`, so a failed write never leaves a corrupt
file on disk).

- `str_replace(path, old_string, new_string)` — whitespace-tolerant match
  (old_string's tokens joined by `\s+`, so indentation/spacing differences
  don't block a match). 0 matches → an error with the *nearest few lines*
  (never the whole file, via a `difflib`-scored nearest-line search). 2+
  matches → the match count and a request for a longer, unique anchor.
  Exactly 1 → applied atomically, returns a compact record:
  `str_replace PATH: -2/+3 lines @L40 (exit_code=0)`.
- `insert(path, line, text)`, `create(path, content)` — same atomic-write
  primitive.
- `apply_edits(path, edits)` — a batch of `str_replace`-style edits applied
  **all-or-nothing**: if any edit fails to match uniquely, nothing is
  written.
- `read(path, start, end)` — line-numbered, capped window (default 200
  lines), so a read is always anchorable and never dumps an unbounded file.
  Also works on Feature 2's spill files.

**Caveat:** tmax-9b was SFT/RL-trained exclusively on the single-`bash`-tool
harness, and the vendored system prompt says "calling a tool other than
`bash`... will cause your response to be rejected" — that prompt text is
intentionally untouched (no prompt changes), so there's no expectation the
model spontaneously discovers and uses these tools without a training
update that teaches it to. They're registered because the harness needs to
be ready for that, and because nothing about adding them is unsafe or
regresses the bash-only path — but the near-term token savings come almost
entirely from Features 1 and 2, which apply transparently to whatever the
model actually does.

## Config flags

All on `Vanillux2Agent.__init__`, with these defaults:

| Flag | Default | Effect |
|---|---|---|
| `enable_edit_tools` | `True` | Register the Feature-3 tools alongside `bash`. |
| `stub_file_writes` | `True` | Feature 1 on/off. |
| `write_recency_keep` | `2` | Trailing-turn window for the keep-full rule. |
| `max_tool_output_tokens` | `2000` | Feature 2's truncation threshold. |
| `head_lines` / `tail_lines` | `40` / `40` | Lines kept at each end of a truncated tool output. |

`context.metadata["compaction"]` reports `stubbed_writes` and
`truncated_outputs` counts for the run (harbor's `AgentContext.metadata` is a
free-form dict — this doesn't touch harbor's own schema).

## Measured impact

[`scripts/vanillux2_context_report.py`](../scripts/vanillux2_context_report.py)
replays real, already-completed `trajectory.json` files under
`evaluation_assets/daytona/tmax-9b/terminal-bench-2-0/` (444 runs available)
turn-by-turn through `build_model_messages`, comparing the peak model-facing
token count *before* (today's harness: the raw prefix sent verbatim) vs.
*after* (the compacted rebuild), using the same token-counting method for
both so the ratio is meaningful even though the absolute count is an
estimate (`litellm.token_counter` doesn't have an exact tmax-9b tokenizer).

```
uv run python scripts/vanillux2_context_report.py --limit 40                    # largest on-disk trajectories
uv run python scripts/vanillux2_context_report.py --limit 40 --select-by turns  # most turns (step-budget-exhausted proxy)
```

| Selection | n | mean peak_before | mean peak_after | mean reduction |
|---|---|---|---|---|
| Largest on-disk `trajectory.json` | 40 | 48,962 | 31,220 | **35.8%** |
| Most turns (proxy for step-budget-exhausted) | 40 | 36,958 | 31,061 | **15.0%** |

Across both samples, Feature 1 (stubbing) fires far more often than Feature
2 (truncation) — consistent with the failure analysis's 49%-vs-22% split.

**Caveat, stated plainly:** this is a *replay* of already-completed runs, not
a re-run — it cannot confirm that a run which actually failed from context
run-out would now pass; that requires a live rerun against the model and
environment. None of the sampled 444 tmax-9b runs (by either selection
method above) actually reached anywhere near the 64k ceiling in this replay,
so this report demonstrates the *mechanism* reclaims a substantial, real
fraction of context on real trajectories — not a confirmed fix of a specific
overflow case. It's also possible the sample under-represents the runs that
did overflow (they may be shorter/cut off mid-generation rather than
long-and-heavy, which the "largest file" / "most turns" proxies wouldn't
necessarily catch).

## Tests

[`tests/vanillux2/`](../tests/vanillux2/) covers detection (heredoc in both
token orders, `tee`, redirects, false-positive avoidance for `2>&1`/`/dev/null`),
stubbing (recency, supersession, the K-turn boundary, edit-tool payloads),
fuzzy `str_replace` (0/1/2+ matches), atomicity (a failed write never
corrupts the file), truncation (exit code + tail always preserved, spill
round-trips), and determinism (rebuilding the same raw log twice is
byte-identical, and the raw log itself is never mutated). Run with:

```
uv run pytest tests/vanillux2
```

These tests import `context_management`/`edit_tools` directly by file path
rather than through the `Vanillux2Agent` package, so they don't require
`harbor` to be installed (only pure-Python logic plus a real local `bash` for
the `exec_fn` fixture, which runs actual shell commands against a temp
directory rather than a hand-rolled fake shell).
