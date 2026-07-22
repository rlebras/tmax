#!/bin/bash
# STEP 3 of the rejection-sampling SFT loop (see docs/rejection_sampling_sft.md).
#
# SFT Qwen3.5-9B on the model's OWN verified successes harvested from the
# rl_data training corpus by rl_data/rejection_sample_sft.py (STaR-style):
# distilling occasionally-reachable behavior into the default, to convert
# terminal-bench pass@5 capability into pass@1 reliability.
#
# This is the 'small' recipe (sft_qwen35_9b_small.sh) with the dataset swapped
# for the rejection-sampled set. Point DATASET at the HF repo you pushed the
# harvest to (rejection_sample_sft.py --push-to-hub <repo>); everything else
# matches the established 9B SFT config so results stay comparable.

BEAKER_IMAGE="${1:-nathanl/open_instruct_auto}"
echo "Using Beaker image: $BEAKER_IMAGE"

# The harvested rejection-sampling dataset (push_to_hub target from step 2).
DATASET="${DATASET:?Set DATASET to the harvested HF repo, e.g. <you>/tmax-rejsample-sft}"

uv run python mason.py \
    --cluster ai2/jupiter \
    --workspace ai2/open-instruct-dev \
    --priority urgent \
    --image "$BEAKER_IMAGE" \
    --pure_docker_mode \
    --preemptible \
    --num_nodes 4 \
    --budget ai2/oe-adapt \
    --gpus 8 \
    -- \
    accelerate launch \
    --mixed_precision bf16 \
    --num_processes 8 \
    --use_deepspeed \
    --deepspeed_config_file configs/ds_configs/stage3_offloading_accelerate.conf \
    --deepspeed_multinode_launcher standard \
    open_instruct/finetune.py \
    --exp_name sft_qwen35_9b_rejsample \
    --model_name_or_path hamishivi/Qwen3.5-9B \
    --tokenizer_name hamishivi/Qwen3.5-9B \
    --use_flash_attn \
    --max_seq_length 32768 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --learning_rate 1e-5 \
    --lr_scheduler_type linear \
    --warmup_ratio 0.03 \
    --weight_decay 0.0 \
    --num_train_epochs 2 \
    --dataset_mixer_list "$DATASET" 1.0 \
    --add_bos \
    --gradient_checkpointing \
    --report_to wandb \
    --with_tracking \
    --logging_steps 1 \
    --seed 42
