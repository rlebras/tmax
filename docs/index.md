# tmax docs

Documentation for the **tmax** terminal-agent stack — generating training data,
fine-tuning, and evaluating terminal/CLI agents.

## Guides

- **[RL data generation](rl_data.md)** — the `rl_data/` pipeline that
  *synthesizes* new Terminal/CLI sandbox tasks (containerized scenarios +
  automated verifiers) and solves/scores/decontaminates/uploads them as agentic
  RL/SFT training data.
- **[SFT](sft.md)** — pointer to the SFT (and RL) training stack, which now lives
  in the vendored open-instruct fork under `training/open-instruct/scripts/tmax/`.
  Consumes the solved trajectories produced by `rl_data/`. (The old standalone
  `sft/` pipeline was removed in the master cleanup.)
- **[Running evals](running_evals.md)** — running Terminal-Bench (Harbor) evals
  of a model, locally and on Beaker.

## How the pieces fit

```
rl_data/  ──(generate + solve terminal tasks)──►  solved trajectories
   │                                                     │
   │                                                     ▼
   │              training/open-instruct/  ──(SFT / RL → train)──►  model
   │                                                     │
   └─────────────────────────────────────────────────►  evals (Terminal-Bench / Harbor)
```
