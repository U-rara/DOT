#!/usr/bin/env bash

# Runner for DOT:
# - Dynamic Outlier Truncation (DOT): apply only on all-correct rollout groups; cutoff
#   T(q)=floor(mean(L)+α·std(L)); truncate only if L_i - T(q) >= m; then recompute task reward.
# - KL-Cov regularization: targeted KL penalty on high-covariance tokens to stabilize exploration.
# - Predictive Dynamic Sampling: estimate effective-group ratio online and do single-round adaptive oversampling.
#
# Default hyperparameters follow the paper's experimental setup:
#   G=32 rollouts/prompt, train_batch_size=128, ppo_mini_batch_size=32 (staleness=4),
#   max_prompt_length=4096, max_response_length=8192, rollout temperature=1.0,
#   DOT α=3, DOT m=32, eval decoding: t=0.6, top_p=0.95, top_k=20.

set -x

ROOT=/path/to/root
WORK_ROOT=${ROOT}/path/to/work/root
PROJ_ROOT=${WORK_ROOT}/projects/dot

# shellcheck disable=SC2164
cd ${WORK_ROOT}

# ================= data/model =================
aime_2024=${ROOT}/data/AIME_2024
aime_2025=${ROOT}/data/aime_2025
amc_23=${ROOT}/data/AMC-23
deepscaler_preview=${ROOT}/data/DeepScaleR-Preview-Dataset
model_path=${ROOT}/model/DeepSeek-R1-Distill-Qwen-1.5B

train_files="['${deepscaler_preview}']"
test_files="['${aime_2025}', '${aime_2024}', '${amc_23}']"

# wandb
project_name=dot
experiment_name=dot-r1_1.5b-8k
default_local_dir=${ROOT}/checkpoint/${experiment_name}


# ================= algorithm =================
adv_estimator=grpo
loss_mode="kl_cov"
loss_agg_mode="seq-mean-token-mean"

clip_ratio_low=0.2
clip_ratio_high=0.2

kl_cov_ratio=0.002
ppo_kl_coef=1.0

enable_filter_groups=True
filter_groups_metric=seq_final_reward
history_window=20
initial_oversample_factor=1.5
min_oversample_factor=1.2
max_oversample_factor=5.0

truncation_enable=True
truncation_success_threshold=1.0
truncation_std_multiplier=3.0
truncation_min_margin=32

max_prompt_length=4096
max_response_length=8192
actor_lr=1e-6

train_batch_size=128
ppo_mini_batch_size=32
n_resp_per_prompt=32
n_resp_per_prompt_val=32

rollout_temperature=1.0

# ================= performance =================
infer_tp=1
train_sp=1
offload=True

actor_max_token_len_per_gpu=$(( (max_prompt_length + max_response_length) * 2 ))
log_prob_max_token_len_per_gpu=$(( actor_max_token_len_per_gpu * 8 ))

python3 -m projects.dot.main_dot \
    algorithm.adv_estimator=${adv_estimator} \
    algorithm.filter_groups.enable=${enable_filter_groups} \
    algorithm.filter_groups.metric=${filter_groups_metric} \
    algorithm.filter_groups.history_window=${history_window} \
    algorithm.filter_groups.initial_oversample_factor=${initial_oversample_factor} \
    algorithm.filter_groups.min_oversample_factor=${min_oversample_factor} \
    algorithm.filter_groups.max_oversample_factor=${max_oversample_factor} \
    algorithm.length_truncation.enable=${truncation_enable} \
    algorithm.length_truncation.success_threshold=${truncation_success_threshold} \
    algorithm.length_truncation.std_multiplier=${truncation_std_multiplier} \
    algorithm.length_truncation.min_truncation_margin=${truncation_min_margin} \
    actor_rollout_ref.actor.policy_loss.loss_mode=${loss_mode} \
    actor_rollout_ref.actor.policy_loss.kl_cov_ratio=${kl_cov_ratio} \
    actor_rollout_ref.actor.policy_loss.ppo_kl_coef=${ppo_kl_coef} \
    data.train_files="${train_files}" \
    data.val_files="${test_files}" \
    data.return_raw_chat=True \
    data.train_batch_size=${train_batch_size} \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.custom_cls.path=${PROJ_ROOT}/dataset.py \
    data.custom_cls.name=CustomRLHFDataset \
    custom_reward_function.path=${PROJ_ROOT}/reward_score/__init__.py \
    custom_reward_function.name=default_compute_score \
    actor_rollout_ref.model.path=${model_path} \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    actor_rollout_ref.actor.optim.lr=${actor_lr} \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode} \
    actor_rollout_ref.actor.ppo_mini_batch_size=${ppo_mini_batch_size} \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${actor_max_token_len_per_gpu} \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=${train_sp} \
    actor_rollout_ref.actor.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${offload} \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${log_prob_max_token_len_per_gpu} \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.temperature=${rollout_temperature} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${infer_tp} \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.9 \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
    actor_rollout_ref.rollout.val_kwargs.top_k=20 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
    actor_rollout_ref.rollout.val_kwargs.n=${n_resp_per_prompt_val} \
    trainer.logger=['console','wandb','tensorboard'] \
    trainer.project_name=${project_name} \
    trainer.experiment_name=${experiment_name} \
    trainer.n_gpus_per_node=${GPU_NUM} \
    trainer.val_before_train=False \
    trainer.log_val_generations=0 \
    trainer.nnodes=${WORLD_SIZE:-1} \
    trainer.save_freq=20 \
    trainer.default_local_dir=${default_local_dir} \
    trainer.test_freq=20 \
    trainer.total_epochs=1000 $@
