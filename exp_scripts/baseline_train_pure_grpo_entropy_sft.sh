#!/bin/bash

source ~/.bashrc

unset ROCR_VISIBLE_DEVICES
unset HIP_VISIBLE_DEVICES

# Set WANDB configuration via environment variables before submitting the job.
#   export WANDB_API_KEY="your_api_key"
#   export WANDB_ENTITY="your_entity"
#   export WANDB_PROJECT="your_project_name"
export WANDB_ENTITY="${WANDB_ENTITY:-EchoRL}"
export WANDB_PROJECT="${WANDB_PROJECT:-EchoRL}"



if ! conda activate echrl; then
echo "[ERROR] Conda environment 'echrl' not found."
conda info --envs || true
exit 1
fi

# Set ROOT to the EchoRL repo root (one level above this script directory)
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"

export PYTHONPATH="$ROOT/echrl:${PYTHONPATH}"

python -m ray stop --force || true

export no_proxy="127.0.0.1,localhost"
export NO_PROXY="127.0.0.1,localhost"

# Set XFormers backend to avoid CUDA errors
export VLLM_ATTENTION_BACKEND=XFORMERS

export MODEL_PATH="${MODEL_PATH:-/path/to/Qwen2.5-Math-7B}"
export DATA_DIR="${DATA_DIR:-$ROOT/data}"
export EXP_NAME="${EXP_NAME:-ECHORL_8192}"


cd $ROOT/echrl/verl/

echo "Using Python: $(which python)"
python -V
python -c "import sys; print('sys.executable =', sys.executable)"
python -c "import torch; print('torch =', torch.__version__)" || echo "[WARN] torch not found in selected Python"


# Train over a single node with entropy-based SFT loss
python3 -m verl.mix_src.main_mix_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$DATA_DIR/openr1.parquet \
    data.val_files=$DATA_DIR/valid.parquet \
    data.train_batch_size=128 \
    data.val_batch_size=512 \
    data.max_prompt_length=1024 \
    data.max_response_length=8192 \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size=64 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=32768 \
    actor_rollout_ref.actor.kl_loss_coef=0.00 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.grad_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.val_temperature=0.6 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.n_val=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.rollout.max_prefix_len=8192 \
    algorithm.kl_ctrl.kl_coef=0.000 \
    actor_rollout_ref.actor.entropy_coeff=0.001 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name="$WANDB_PROJECT" \
    trainer.experiment_name="$EXP_NAME" \
    trainer.default_local_dir="$ROOT/results/checkpoints/$EXP_NAME" \
    +trainer.val_before_train=True \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=5 \
    trainer.test_freq=5 \
    trainer.save_freq_ckpt=True \
    trainer.save_best_ckpt=True \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.use_sft_prefix_reward=False \
    actor_rollout_ref.rollout.prefix_share_across_samples=False \
    actor_rollout_ref.rollout.prefix_strategy=random \
    actor_rollout_ref.rollout.n_prefix=1 \
    actor_rollout_ref.rollout.min_prefix_ratio=0.0 \
    actor_rollout_ref.rollout.max_prefix_ratio=0.0 \
    actor_rollout_ref.rollout.prefix_reward_weight_alpha=1.0 \
    actor_rollout_ref.ref.use_ref=False \
    actor_rollout_ref.actor.use_off_policy_loss=False \
    actor_rollout_ref.actor.off_policy_normalize=False \
    actor_rollout_ref.actor.off_policy_loss_impl=token \
    algorithm.grpo_use_std=False \
    actor_rollout_ref.actor.loss_remove_token_mean=True \
    data.reward_impl_version=3 \
    trainer.max_optim_to_keep=2 \
    data.shuffle=True \
    trainer.default_hdfs_dir=null \
    trainer.total_epochs=30 \
    +actor_rollout_ref.actor.use_entropy_sft_actor=True \
    +actor_rollout_ref.actor.use_entropy_sft=True \
    +actor_rollout_ref.actor.entropy_sft_coef=0.001 \
    +actor_rollout_ref.actor.success_reward_value=1.0 \
    +actor_rollout_ref.actor.newline_token_ids=[198,271] \
    +actor_rollout_ref.actor.entropy_sft_debug=True "${@:1}"

