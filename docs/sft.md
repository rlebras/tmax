# SFT

!!! warning "The standalone `sft/` stack was removed"
    The previous top-level `sft/` pipeline (the bespoke
    convert → pre-tokenize → DeepSpeed-Ulysses train stack documented here) was
    **removed in the master cleanup** that vendored a full open-instruct fork into
    the repo. SFT now runs through that fork under
    [`training/open-instruct/`](../training/open-instruct).

## Where SFT lives now

SFT for tmax models is done with the vendored open-instruct training fork. The
tmax-specific entrypoints (both SFT and RL) live under
[`training/open-instruct/scripts/tmax/`](../training/open-instruct/scripts/tmax)
— see its [`README.md`](../training/open-instruct/scripts/tmax/README.md):

- **SFT launch scripts:** [`scripts/tmax/SFT/`](../training/open-instruct/scripts/tmax/SFT)
  (`sft_qwen35_9b_big.sh`, `sft_qwen35_9b_small.sh`, `sft_qwen3_8b_big.sh`,
  `sft_qwen3_8b_small.sh`).
- **RL launch scripts** (alongside SFT): [`scripts/tmax/RL/`](../training/open-instruct/scripts/tmax/RL).
- **Data prep** (convert trajectories → SFT schema, add tool columns, OLMo-core
  conversion): [`scripts/data/`](../training/open-instruct/scripts/data)
  (e.g. `convert_sft_data_for_olmocore.py`, `add_tools_column_tmax_sft.py`,
  `sft/`, `sft_v1_v2/`).

The bespoke `sft/preprocessing/` pipeline (the old `convert.py` / `pipeline.py` /
`harness.py` / `sources.yaml` flow this guide used to document) was removed in the
cleanup; data prep is now the open-instruct-style scripts under `scripts/data/`.

## Upstream of SFT

The training data still comes from the [`rl_data/`](rl_data.md) pipeline —
generate + solve terminal tasks, then feed the successful (pass) trajectories
into the SFT data-prep step above.
