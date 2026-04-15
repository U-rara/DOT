import math
import logging
import uuid
from collections import defaultdict, deque

import numpy as np
import torch
from verl import DataProto
from verl.trainer.ppo.ray_trainer import RayPPOTrainer

logger = logging.getLogger(__name__)


class DynamicSamplingStats:
    """Predictive Dynamic Sampling stats (paper Sec. Stabilizing and Scaling RL Training).

    Track the historical ratio of "effective groups" (groups that survive filtering) and
    predict the next-step oversampling factor for single-round sampling.
    """
    
    def __init__(self, history_window=20, initial_oversample_factor=1.5, min_factor=1.2, max_factor=3.0):
        """Initialize dynamic sampling statistics.
        
        Args:
            history_window: Number of recent steps to use for prediction
            initial_oversample_factor: Initial oversample factor (conservative)
            min_factor: Minimum oversample factor (safety lower bound)
            max_factor: Maximum oversample factor (avoid excessive waste)
        """
        self.history_window = history_window
        self.initial_oversample_factor = initial_oversample_factor
        self.min_factor = min_factor
        self.max_factor = max_factor
        
        # Historical pass rates (prompts that pass filter / total prompts sampled)
        self.pass_rate_history = deque(maxlen=history_window)
        
        # Statistics
        self.total_prompts_sampled = 0
        self.total_prompts_passed = 0
        self.total_prompts_used = 0  # Actually used in training (after truncation to train_batch_size)
        self.total_prompts_wasted = 0  # Sampled but filtered out (std=0)
    
    def predict_oversample_factor(self):
        """Predict oversample factor from historical effective-group ratio."""
        if len(self.pass_rate_history) == 0:
            # First step: use the configured initial factor.
            return self.initial_oversample_factor
        
        # Calculate moving average of recent pass rates
        recent_pass_rates = list(self.pass_rate_history)
        avg_pass_rate = np.mean(recent_pass_rates)
        std_pass_rate = np.std(recent_pass_rates)
        
        # Add safety margin based on std (higher variance = more safety margin)
        safety_margin = 1.0 + std_pass_rate
        
        # Compute required oversample factor
        if avg_pass_rate > 0:
            predicted_factor = (1.0 / avg_pass_rate) * safety_margin
        else:
            # If pass rate is 0 (all filtered), use max factor
            predicted_factor = self.max_factor
        
        # Clip to reasonable range
        predicted_factor = max(self.min_factor, min(predicted_factor, self.max_factor))
        
        return predicted_factor
    
    def update(self, num_sampled, num_passed, num_used):
        """Update statistics after a training step.
        
        Args:
            num_sampled: Number of prompts sampled from dataloader
            num_passed: Number of prompts that passed filtering (std > 0)
            num_used: Number of prompts actually used in training
        """
        # Update counters
        self.total_prompts_sampled += num_sampled
        self.total_prompts_passed += num_passed
        self.total_prompts_used += num_used
        self.total_prompts_wasted += (num_sampled - num_passed)
        
        # Update pass rate history
        pass_rate = num_passed / num_sampled if num_sampled > 0 else 0.0
        self.pass_rate_history.append(pass_rate)
    
    def get_metrics(self):
        """Get current statistics as metrics dict.
        
        Returns:
            dict: Statistics metrics
        """
        total_sampled = self.total_prompts_sampled
        total_passed = self.total_prompts_passed
        total_used = self.total_prompts_used
        total_wasted = self.total_prompts_wasted
        
        overall_pass_rate = total_passed / total_sampled if total_sampled > 0 else 0.0
        overall_waste_rate = total_wasted / total_sampled if total_sampled > 0 else 0.0
        utilization_rate = total_used / total_sampled if total_sampled > 0 else 0.0
        
        metrics = {
            "dynamic_sampling/total_prompts_sampled": total_sampled,
            "dynamic_sampling/total_prompts_passed": total_passed,
            "dynamic_sampling/total_prompts_used": total_used,
            "dynamic_sampling/total_prompts_wasted": total_wasted,
            "dynamic_sampling/overall_pass_rate": overall_pass_rate,
            "dynamic_sampling/overall_waste_rate": overall_waste_rate,
            "dynamic_sampling/utilization_rate": utilization_rate,
        }
        
        # Recent statistics (if available)
        if len(self.pass_rate_history) > 0:
            recent_pass_rates = list(self.pass_rate_history)
            metrics["dynamic_sampling/recent_pass_rate_mean"] = np.mean(recent_pass_rates)
            metrics["dynamic_sampling/recent_pass_rate_std"] = np.std(recent_pass_rates)
        
        return metrics


class RayDOTTrainer(RayPPOTrainer):
    """Custom PPO trainer for DOT (Dynamic sampling + length truncation)."""
    
    def __init__(self, *args, **kwargs):
        """Initialize the DOT trainer."""
        super().__init__(*args, **kwargs)
        
        # State tracking for adaptive oversample (for resume support)
        self.adaptive_sampling_state = {
            "current_epoch": 0,
            "current_idx_in_epoch": 0,
            "sampler_state": None,  # For RandomSampler's generator state
        }
        
        # Dynamic Sampling config (filter_groups) with adaptive oversample.
        self.filter_groups_config = self.config.algorithm.get("filter_groups", None)
        if self.filter_groups_config and self.filter_groups_config.get("enable", False):
            print("[DOT] Predictive Dynamic Sampling enabled:")
            print(f"  - Metric: {self.filter_groups_config.metric}")
            print("  - Adaptive oversample: enabled")
            
            # Initialize dynamic sampling statistics
            self.dynamic_sampling_stats = DynamicSamplingStats(
                history_window=self.filter_groups_config.get("history_window", 20),
                initial_oversample_factor=self.filter_groups_config.get("initial_oversample_factor", 1.5),
                min_factor=self.filter_groups_config.get("min_oversample_factor", 1.2),
                max_factor=self.filter_groups_config.get("max_oversample_factor", 3.0),
            )
            print(f"  - Adaptive oversample settings:")
            print(f"    * History window: {self.filter_groups_config.get('history_window', 20)}")
            print(f"    * Initial factor: {self.filter_groups_config.get('initial_oversample_factor', 1.5)}")
            print(f"    * Factor range: [{self.filter_groups_config.get('min_oversample_factor', 1.2)}, {self.filter_groups_config.get('max_oversample_factor', 3.0)}]")
            print(f"  - Single-round sampling: One adaptive batch per step (no continue loop)")
            print(f"  - No sample discard: All passed prompts are used in training")
            print(f"  - Filters out prompts where all responses have the same metric value (std=0)")
        else:
            self.filter_groups_config = None
            self.dynamic_sampling_stats = None
            print("[DOT] Predictive Dynamic Sampling disabled")
        
        # Length Truncation config
        self.truncation_config = self.config.algorithm.get("length_truncation", None)
        if self.truncation_config and self.truncation_config.get("enable", False):
            print("[DOT] Length Truncation enabled:")
            print(f"  - Success threshold: {self.truncation_config.get('success_threshold', 1.0)}")
            print(f"  - Std multiplier (α): {self.truncation_config.get('std_multiplier', 3.0)}")
            print(f"  - Reduction margin (m): {self.truncation_config.get('min_truncation_margin', 32)}")
            print("  - Cutoff: T(q)=floor(mean(L)+α·std(L))")
        else:
            self.truncation_config = None
            print("[DOT] Length Truncation disabled")

    def fit(self):
        """Run training with DOT modifications when enabled."""
        has_dynamic_sampling = bool(self.filter_groups_config and self.filter_groups_config.get("enable", False))
        has_truncation = bool(self.truncation_config and self.truncation_config.get("enable", False))

        if has_dynamic_sampling:
            return self._fit_with_dynamic_sampling()

        if not has_truncation:
            return super().fit()

        from projects.dot.length_truncation import apply_length_truncation
        from verl.trainer.ppo import ray_trainer as ray_trainer_module

        original_compute_advantage = ray_trainer_module.compute_advantage

        def compute_advantage_with_truncation(data, *args, **kwargs):
            data, truncation_metrics = apply_length_truncation(
                data,
                reward_fn=self.reward_fn,
                success_threshold=self.truncation_config.get("success_threshold", 1.0),
                std_multiplier=self.truncation_config.get("std_multiplier", 3.0),
                min_truncation_margin=self.truncation_config.get("min_truncation_margin", 32),
                metric_name="token_level_rewards",
            )
            if "modification_metrics" not in data.meta_info:
                data.meta_info["modification_metrics"] = {}
            data.meta_info["modification_metrics"].update(truncation_metrics)
            return original_compute_advantage(data, *args, **kwargs)

        ray_trainer_module.compute_advantage = compute_advantage_with_truncation
        try:
            return super().fit()
        finally:
            ray_trainer_module.compute_advantage = original_compute_advantage
    
    def _fit_with_dynamic_sampling(self):
        """Training loop with dynamic sampling (filter_groups).
        
        DOT flow:
        1) Generate responses
        2) Compute rewards
        3) Apply length truncation (optional; before filtering)
        4) Filter prompts with std(metric)==0 (no learning signal / zero variance)
        5) Clip/pad to train_batch_size
        6) Update models
        """
        from omegaconf import OmegaConf
        from tqdm import tqdm
        
        from verl.trainer.ppo.core_algos import agg_loss
        from verl.trainer.ppo.metric_utils import compute_data_metrics, compute_throughout_metrics, compute_timing_metrics
        from verl.trainer.ppo.ray_trainer import apply_kl_penalty, compute_advantage, compute_response_mask
        from verl.trainer.ppo.reward import compute_reward
        from verl.trainer.ppo.utils import Role
        from verl.utils.metric import reduce_metrics
        from verl.utils.debug import marked_timer
        from verl.utils.tracking import Tracking
        
        logger_tracking = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )
        
        self.global_steps = 0
        self.gen_steps = 0
        
        # Load checkpoint before doing anything
        self._load_checkpoint()
        
        # Perform validation before training
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            from pprint import pprint
            
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger_tracking.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return
        
        if self.config.actor_rollout_ref.rollout.get("skip_rollout", False):
            from verl.utils.rollout_skip import RolloutSkip
            
            rollout_skip = RolloutSkip(self.config, self.actor_rollout_wg)
            rollout_skip.wrap_generate_sequences()
        
        # Add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")
        
        # We start from step 1
        self.global_steps += 1
        last_val_metrics = None
        self.max_steps_duration = 0
        
        prev_step_profile = False
        curr_step_profile = (
            self.global_steps in self.config.global_profiler.steps
            if self.config.global_profiler.steps is not None
            else False
        )
        next_step_profile = False
        
        has_truncation = bool(self.truncation_config and self.truncation_config.get("enable", False))
        if has_truncation:
            from projects.dot.length_truncation import apply_length_truncation
        
        # Resume from saved state if available
        start_epoch = self.adaptive_sampling_state["current_epoch"]
        
        for epoch in range(start_epoch, self.config.trainer.total_epochs):
            # Predictive Dynamic Sampling: direct sampling with adaptive oversampling.
            # Regenerate sampler indices for each epoch (respects shuffle)
            # This ensures: 1) shuffle works, 2) each epoch has different order
            
            # Set deterministic seed for this epoch to ensure reproducibility across resumes
            # For RandomSampler: each epoch gets a deterministic but different seed
            # For SequentialSampler: this doesn't affect it
            sampler = self.train_dataloader.sampler
            if hasattr(sampler, 'generator'):
                # RandomSampler has a generator
                base_seed = self.config.data.get("seed", 42)
                epoch_seed = base_seed + epoch
                sampler.generator.manual_seed(epoch_seed)
                print(f"[PDS] Set sampler seed to {epoch_seed} for epoch {epoch+1}")
            
            sampler_indices = list(sampler)
            
            # Resume from saved position if this is the resumed epoch
            if epoch == start_epoch:
                current_idx = self.adaptive_sampling_state["current_idx_in_epoch"]
                print(f"[PDS] Resuming epoch {epoch+1} from index {current_idx}")
            else:
                current_idx = 0
            
            print(f"[PDS] Starting epoch {epoch+1}/{self.config.trainer.total_epochs}, "
                  f"total samples: {len(sampler_indices)}, starting from index: {current_idx}")
            
            while current_idx < len(sampler_indices):
                metrics = {}
                timing_raw = {}
                
                with marked_timer("start_profile", timing_raw):
                    self._start_profiling(
                        not prev_step_profile and curr_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )
                
                # === Predictive Dynamic Sampling: predict and fetch prompts in one round ===
                oversample_factor = self.dynamic_sampling_stats.predict_oversample_factor()
                train_batch_size = self.config.data.train_batch_size
                target_sample_size = int(train_batch_size * oversample_factor)
                
                # Calculate how many samples to fetch (bounded by remaining data)
                remaining_samples = len(sampler_indices) - current_idx
                requested_sample_size = min(target_sample_size, remaining_samples)

                world_size = self.actor_rollout_wg.world_size
                rollout_n = self.config.actor_rollout_ref.rollout.n
                prompt_divisor = (
                    world_size // math.gcd(world_size, rollout_n) if world_size > 1 else 1
                )  # Ensure (num_prompts * rollout_n) % world_size == 0.

                # Ensure prompt count produces a DP-divisible number of trajectories.
                # This prevents downstream asserts in token-balancing and DataProto chunking.
                actual_sample_size = requested_sample_size
                if world_size > 1:
                    if actual_sample_size < prompt_divisor and remaining_samples >= prompt_divisor:
                        actual_sample_size = prompt_divisor
                    actual_sample_size -= actual_sample_size % prompt_divisor

                    # Not enough remaining prompts to form a divisible batch: skip the tail of this epoch.
                    # If the dataset itself is smaller than prompt_divisor, fall back to sampling with replacement.
                    if actual_sample_size == 0 and remaining_samples > 0:
                        if current_idx > 0:
                            print(
                                f"[PDS] Skipping {remaining_samples} tail prompts to keep "
                                f"(num_prompts * rollout_n) divisible by world_size "
                                f"({prompt_divisor=}, {rollout_n=}, {world_size=})."
                            )
                            current_idx = len(sampler_indices)
                            continue
                        actual_sample_size = prompt_divisor
                
                # Fetch samples directly from dataset
                if world_size > 1 and remaining_samples < actual_sample_size:
                    # Sample with replacement when the dataset/epoch remainder is too small.
                    assert len(sampler_indices) > 0
                    sample_indices = [sampler_indices[i % len(sampler_indices)] for i in range(actual_sample_size)]
                    current_idx = len(sampler_indices)
                else:
                    sample_indices = sampler_indices[current_idx : current_idx + actual_sample_size]
                    current_idx += actual_sample_size
                
                # Get samples from dataset and collate
                samples = [self.train_dataset[idx] for idx in sample_indices]
                batch_dict = self.train_dataloader.collate_fn(samples)
                
                new_batch: DataProto = DataProto.from_single_dict(batch_dict)
                num_prompts_sampled = len(sample_indices)
                
                # Add uid to batch (BEFORE _get_gen_batch!)
                new_batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(new_batch.batch))], dtype=object
                )
                
                if world_size > 1 and actual_sample_size != requested_sample_size:
                    print(
                        f"[PDS] Adjusted prompt fetch: requested={requested_sample_size}, "
                        f"used={actual_sample_size} to satisfy (num_prompts*rollout_n)%world_size==0 "
                        f"({prompt_divisor=}, {rollout_n=}, {world_size=})."
                    )

                print(f"[PDS] Predicted factor: {oversample_factor:.2f}, "
                      f"Target: {target_sample_size}, Fetched: {num_prompts_sampled} prompts "
                      f"(remaining: {len(sampler_indices) - current_idx})")
                
                # Use _get_gen_batch to properly handle async rollout mode
                gen_batch = self._get_gen_batch(new_batch)
                
                # Pass global_steps to trace (critical!)
                gen_batch.meta_info["global_steps"] = self.global_steps
                
                gen_batch_output = gen_batch.repeat(
                    repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True
                )
                
                is_last_step = self.global_steps >= self.total_training_steps
                with marked_timer("step", timing_raw):
                    # Generate a batch
                    with marked_timer("gen", timing_raw, "red"):
                        if not self.async_rollout_mode:
                            gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch_output)
                        else:
                            gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch_output)
                        
                        timing_raw.update(gen_batch_output.meta_info["timing"])
                        gen_batch_output.meta_info.pop("timing", None)
                    
                    # Repeat to align with repeated responses in rollout
                    new_batch = new_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    new_batch = new_batch.union(gen_batch_output)

                    # Compute response mask (always recompute from current responses/attention_mask)
                    new_batch.batch["response_mask"] = compute_response_mask(new_batch)
                    
                    # Balance the number of valid tokens across DP ranks
                    # NOTE: This usually changes the order of data in the batch,
                    # which won't affect the advantage calculation (since it's based on uid),
                    # but might affect the loss calculation (due to the change of mini-batching).
                    if self.config.trainer.balance_batch:
                        self._balance_batch(new_batch, metrics=metrics)
                    
                    # Compute global_valid tokens
                    new_batch.meta_info["global_token_num"] = torch.sum(new_batch.batch["attention_mask"], dim=-1).tolist()
                    
                    with marked_timer("reward", timing_raw, "yellow"):
                        # Compute reward model score
                        if self.use_rm and "rm_scores" not in new_batch.batch.keys():
                            reward_tensor = self.rm_wg.compute_rm_score(new_batch)
                            new_batch = new_batch.union(reward_tensor)
                        
                        if self.config.reward_model.launch_reward_fn_async:
                            from verl.trainer.ppo.reward import compute_reward_async
                            
                            future_reward = compute_reward_async.remote(
                                data=new_batch, config=self.config, tokenizer=self.tokenizer
                            )
                        else:
                            reward_tensor, reward_extra_infos_dict = compute_reward(new_batch, self.reward_fn)
                    
                    # Recompute old_log_probs
                    with marked_timer("old_log_prob", timing_raw, "blue"):
                        old_log_prob = self.actor_rollout_wg.compute_log_prob(new_batch)
                        entropys = old_log_prob.batch["entropys"]

                        response_masks = new_batch.batch["response_mask"]
                        loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                        entropy_agg = agg_loss(loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode)
                        old_log_prob_metrics = {"actor/entropy": entropy_agg.detach().item()}
                        metrics.update(old_log_prob_metrics)
                        old_log_prob.batch.pop("entropys")
                        new_batch = new_batch.union(old_log_prob)
                        
                        if "rollout_log_probs" in new_batch.batch.keys():
                            from verl.utils.debug.metrics import calculate_debug_metrics
                            
                            metrics.update(calculate_debug_metrics(new_batch))
                    
                    if self.use_reference_policy:
                        # Compute reference log_prob
                        with marked_timer(str(Role.RefPolicy), timing_raw, "olive"):
                            if not self.ref_in_actor:
                                ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(new_batch)
                            else:
                                ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(new_batch)
                            new_batch = new_batch.union(ref_log_prob)
                    
                    # Compute values
                    if self.use_critic:
                        with marked_timer("values", timing_raw, "cyan"):
                            values = self.critic_wg.compute_values(new_batch)
                            new_batch = new_batch.union(values)
                    
                    with marked_timer("adv", timing_raw, "brown"):
                        # We combine with rule-based rm
                        if self.config.reward_model.launch_reward_fn_async:
                            import ray
                            
                            reward_tensor, reward_extra_infos_dict = ray.get(future_reward)
                        
                        new_batch.batch["token_level_scores"] = reward_tensor
                        
                        if reward_extra_infos_dict:
                            new_batch.non_tensor_batch.update(
                                {k: np.array(v) for k, v in reward_extra_infos_dict.items()}
                            )
                        
                        # Compute rewards. apply_kl_penalty if available
                        if self.config.algorithm.use_kl_in_reward:
                            new_batch, kl_metrics = apply_kl_penalty(
                                new_batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty
                            )
                            metrics.update(kl_metrics)
                        else:
                            new_batch.batch["token_level_rewards"] = new_batch.batch["token_level_scores"]
                        
                        # === DOT: Apply Dynamic Outlier Truncation (BEFORE filtering) ===
                        # DOT is defined on all-correct groups. These groups would otherwise be filtered out
                        # as "ineffective" (std=0). Apply DOT and recompute rewards first; if truncation breaks
                        # some answers, the group becomes an effective training signal (std>0).
                        if has_truncation:
                            new_batch, truncation_metrics = apply_length_truncation(
                                new_batch,
                                reward_fn=self.reward_fn,
                                success_threshold=self.truncation_config.get("success_threshold", 1.0),
                                std_multiplier=self.truncation_config.get("std_multiplier", 3.0),
                                min_truncation_margin=self.truncation_config.get("min_truncation_margin", 32),
                                metric_name="token_level_rewards"
                            )
                            metrics.update(truncation_metrics)

                        
                        # === Dynamic Sampling Filter (AFTER DOT, BEFORE statistics) ===
                        # Keep only "effective groups": groups with non-zero variance in the chosen
                        # sequence metric after applying DOT and recomputing rewards.
                        # In GRPO-style advantage estimation, zero variance implies vanishing group-relative
                        # advantages (no learning signal).
                        metric_name = self.filter_groups_config.metric
                        if metric_name == "seq_final_reward":
                            new_batch.non_tensor_batch["seq_final_reward"] = (
                                new_batch.batch["token_level_rewards"].sum(dim=-1).cpu().numpy()
                            )
                        elif metric_name == "seq_reward":
                            new_batch.non_tensor_batch["seq_reward"] = (
                                new_batch.batch["token_level_scores"].sum(dim=-1).cpu().numpy()
                            )
                        
                        # Collect the sequence metric for each trajectory
                        prompt_uid2metric_vals = defaultdict(list)
                        for uid, metric_val in zip(
                            new_batch.non_tensor_batch["uid"], new_batch.non_tensor_batch[metric_name], strict=True
                        ):
                            prompt_uid2metric_vals[uid].append(metric_val)
                        
                        # Calculate std for each prompt's responses
                        prompt_uid2metric_std = {}
                        for prompt_uid, metric_vals in prompt_uid2metric_vals.items():
                            prompt_uid2metric_std[prompt_uid] = np.std(metric_vals)
                        
                        # Keep prompts where std > 0 or only has 1 response (diverse responses)
                        kept_prompt_uids = [
                            uid
                            for uid, std in prompt_uid2metric_std.items()
                            if std > 0 or len(prompt_uid2metric_vals[uid]) == 1
                        ]
                        num_prompts_passed = len(kept_prompt_uids)
                        
                        # Filter trajectories
                        kept_traj_idxs = []
                        for idx, traj_from_prompt_uid in enumerate(new_batch.non_tensor_batch["uid"]):
                            if traj_from_prompt_uid in kept_prompt_uids:
                                kept_traj_idxs.append(idx)
                        
                        new_batch = new_batch[kept_traj_idxs]
                        
                        # === Clip/Pad ===
                        train_batch_size = self.config.data.train_batch_size
                        expected_group_size = self.config.actor_rollout_ref.rollout.n

                        # Group by prompt uid
                        uid_to_traj_idxs = defaultdict(list)
                        for idx, uid in enumerate(new_batch.non_tensor_batch["uid"]):
                            uid_to_traj_idxs[uid].append(idx)
                        unique_uids = list(uid_to_traj_idxs.keys())

                        group_sizes = [len(idxs) for idxs in uid_to_traj_idxs.values()]
                        all_equal_group_size = len(group_sizes) > 0 and len(set(group_sizes)) == 1

                        use_legacy_prompt_clip = all_equal_group_size and group_sizes[0] == expected_group_size

                        if use_legacy_prompt_clip:
                            # Legacy behavior: keep/pad by prompt count
                            if num_prompts_passed > train_batch_size:
                                kept_uids_for_training = unique_uids[:train_batch_size]
                                final_traj_idxs = []
                                for uid in kept_uids_for_training:
                                    final_traj_idxs.extend(uid_to_traj_idxs[uid])
                                
                                new_batch = new_batch[final_traj_idxs]
                                num_prompts_used = train_batch_size
                                num_prompts_discarded = num_prompts_passed - train_batch_size
                                num_prompts_padded = 0
                            
                            elif num_prompts_passed < train_batch_size and num_prompts_passed > 0:
                                num_prompts_to_pad = train_batch_size - num_prompts_passed
                                
                                padded_uids = []
                                pad_idx = 0
                                for _ in range(num_prompts_to_pad):
                                    padded_uids.append(unique_uids[pad_idx % len(unique_uids)])
                                    pad_idx += 1
                                
                                padding_traj_idxs = []
                                for uid in padded_uids:
                                    padding_traj_idxs.extend(uid_to_traj_idxs[uid])
                                
                                padding_batch = new_batch[padding_traj_idxs]
                                
                                padding_batch.batch["response_mask"] = torch.zeros_like(padding_batch.batch["response_mask"])
                                # Fresh UIDs keep padded copies out of the real GRPO groups.
                                padding_batch.non_tensor_batch["uid"] = np.array(
                                    [str(uuid.uuid4()) for _ in range(len(padding_batch))], dtype=object
                                )
                                
                                new_batch = DataProto.concat([new_batch, padding_batch])
                                
                                num_prompts_used = train_batch_size
                                num_prompts_discarded = 0
                                num_prompts_padded = num_prompts_to_pad
                            
                            else:
                                # num_prompts_passed == train_batch_size OR 0
                                num_prompts_used = min(num_prompts_passed, train_batch_size)
                                num_prompts_discarded = max(num_prompts_passed - num_prompts_used, 0)
                                num_prompts_padded = 0
                            
                            batch = new_batch
                        
                        else:
                            # Group-aware target: fixed trajectory budget
                            target_trajs = train_batch_size * expected_group_size
                            # Shuffle to avoid positional bias
                            perm = np.random.permutation(len(unique_uids))
                            unique_uids = [unique_uids[i] for i in perm]
                            
                            group_batches = {uid: new_batch[idxs] for uid, idxs in uid_to_traj_idxs.items()}
                            
                            selected_traj_idxs = []
                            selected_uids = []
                            traj_so_far = 0
                            for uid in unique_uids:
                                if traj_so_far >= target_trajs and len(selected_uids) >= 1:
                                    break
                                idxs = uid_to_traj_idxs[uid]
                                selected_uids.append(uid)
                                selected_traj_idxs.extend(idxs)
                                traj_so_far += len(idxs)
                            
                            new_batch = new_batch[selected_traj_idxs]
                            num_prompts_used_real = len(selected_uids)
                            num_prompts_discarded = num_prompts_passed - num_prompts_used_real
                            
                            num_prompts_padded = 0
                            pad_idx = 0
                            while traj_so_far < target_trajs and unique_uids:
                                pad_uid = unique_uids[pad_idx % len(unique_uids)]
                                pad_idx += 1
                                pad_group = group_batches[pad_uid]
                                pad_group.batch["response_mask"] = torch.zeros_like(pad_group.batch["response_mask"])
                                # Fresh UIDs keep padded copies out of the real GRPO groups.
                                pad_group.non_tensor_batch["uid"] = np.array(
                                    [str(uuid.uuid4()) for _ in range(len(pad_group))], dtype=object
                                )
                                new_batch = DataProto.concat([new_batch, pad_group])
                                traj_so_far += len(pad_group)
                                num_prompts_padded += 1
                            
                            num_prompts_used = num_prompts_used_real + num_prompts_padded
                            batch = new_batch

                        # === DP padding with full groups (keep groups intact) ===
                        world_size = self.actor_rollout_wg.world_size
                        if world_size > 1:
                            # Recompute group indices on the current batch
                            uid_to_traj_idxs = defaultdict(list)
                            for idx, uid in enumerate(batch.non_tensor_batch["uid"]):
                                uid_to_traj_idxs[uid].append(idx)
                            remainder = len(batch) % world_size
                            if remainder != 0:
                                pad_trajs = 0
                                pad_prompts = 0
                                # Sort uids by current group size (ascending) to minimize over-padding
                                uid_sizes = sorted(
                                    [(uid, len(idxs)) for uid, idxs in uid_to_traj_idxs.items()],
                                    key=lambda x: x[1],
                                )
                                pad_idx = 0
                                while len(batch) % world_size != 0:
                                    pad_uid, pad_group_size = uid_sizes[pad_idx % len(uid_sizes)]
                                    pad_idx += 1
                                    pad_prompts += 1
                                    pad_trajs += pad_group_size
                                    group_idxs = uid_to_traj_idxs[pad_uid]
                                    padding_batch = batch[group_idxs]
                                    # Zero out training signal
                                    padding_batch.batch["response_mask"] = torch.zeros_like(
                                        padding_batch.batch["response_mask"]
                                    )
                                    if "token_level_rewards" in padding_batch.batch:
                                        padding_batch.batch["token_level_rewards"] = torch.zeros_like(
                                            padding_batch.batch["token_level_rewards"]
                                        )
                                    if "token_level_scores" in padding_batch.batch:
                                        padding_batch.batch["token_level_scores"] = torch.zeros_like(
                                            padding_batch.batch["token_level_scores"]
                                        )
                                    # Fresh UIDs so they don't mix with real groups
                                    padding_batch.non_tensor_batch["uid"] = np.array(
                                        [str(uuid.uuid4()) for _ in range(pad_group_size)], dtype=object
                                    )
                                    batch = DataProto.concat([batch, padding_batch])
                                metrics["dp_padding/prompts_padded"] = float(pad_prompts)
                                metrics["dp_padding/trajs_padded"] = float(pad_trajs)
                        
                        # === Dynamic Sampling Statistics ===
                        # Update statistics with actual used samples
                        self.dynamic_sampling_stats.update(
                            num_sampled=num_prompts_sampled,
                            num_passed=num_prompts_passed,
                            num_used=num_prompts_used
                        )
                        
                        # Log step-level metrics
                        step_pass_rate = num_prompts_passed / num_prompts_sampled if num_prompts_sampled > 0 else 0.0
                        metrics["dynamic_sampling/step_prompts_sampled"] = num_prompts_sampled
                        metrics["dynamic_sampling/step_prompts_passed"] = num_prompts_passed
                        metrics["dynamic_sampling/step_prompts_used"] = num_prompts_used
                        metrics["dynamic_sampling/step_prompts_discarded"] = num_prompts_discarded
                        metrics["dynamic_sampling/step_prompts_padded"] = num_prompts_padded
                        metrics["dynamic_sampling/step_pass_rate"] = step_pass_rate
                        metrics["dynamic_sampling/step_oversample_factor"] = oversample_factor
                        metrics["dynamic_sampling/step_prompts_wasted"] = num_prompts_sampled - num_prompts_passed
                        
                        # Add cumulative metrics
                        metrics.update(self.dynamic_sampling_stats.get_metrics())
                        
                        print(f"[PDS] Step {self.global_steps}: "
                              f"Sampled {num_prompts_sampled}, Passed {num_prompts_passed} ({step_pass_rate:.1%}), "
                              f"Used {num_prompts_used}, Discarded {num_prompts_discarded}, Padded {num_prompts_padded}")
                        
                        # Compute advantages (executed on the driver process)
                        norm_adv_by_std_in_grpo = self.config.algorithm.get("norm_adv_by_std_in_grpo", True)
                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.actor_rollout_ref.rollout.n,
                            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                            config=self.config.algorithm,
                        )
                    
                    # Update critic
                    if self.use_critic:
                        with marked_timer("update_critic", timing_raw, "pink"):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)
                    
                    # Implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # Update actor
                        with marked_timer("update_actor", timing_raw, "red"):
                            batch.meta_info["multi_turn"] = self.config.actor_rollout_ref.rollout.multi_turn.enable
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)
                    
                    # Log rollout generations if enabled
                    rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    if rollout_data_dir:
                        self._log_rollout_data(batch, reward_extra_infos_dict, timing_raw, rollout_data_dir)
                
                # Validate
                if (
                    self.val_reward_fn is not None
                    and self.config.trainer.test_freq > 0
                    and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0)
                ):
                    with marked_timer("testing", timing_raw, "green"):
                        val_metrics: dict = self._validate()
                        if is_last_step:
                            last_val_metrics = val_metrics
                    metrics.update(val_metrics)
                
                # Check if the ESI (Elastic Server Instance)/training plan is close to expiration
                from verl.utils.checkpoint.checkpoint_manager import should_save_ckpt_esi
                
                esi_close_to_expiration = should_save_ckpt_esi(
                    max_steps_duration=self.max_steps_duration,
                    redundant_time=self.config.trainer.esi_redundant_time,
                )
                # Check if the conditions for saving a checkpoint are met
                if self.config.trainer.save_freq > 0 and (
                    is_last_step or self.global_steps % self.config.trainer.save_freq == 0 or esi_close_to_expiration
                ):
                    if esi_close_to_expiration:
                        print("Force saving checkpoint: ESI instance expiration approaching.")
                    with marked_timer("save_checkpoint", timing_raw, "green"):
                        self._save_checkpoint()
                
                with marked_timer("stop_profile", timing_raw):
                    next_step_profile = (
                        self.global_steps + 1 in self.config.global_profiler.steps
                        if self.config.global_profiler.steps is not None
                        else False
                    )
                    self._stop_profiling(
                        curr_step_profile and not next_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )
                    prev_step_profile = curr_step_profile
                    curr_step_profile = next_step_profile
                
                steps_duration = timing_raw["step"]
                self.max_steps_duration = max(self.max_steps_duration, steps_duration)
                
                # Training metrics
                metrics.update(
                    {
                        "training/global_step": self.global_steps,
                        "training/epoch": epoch,
                    }
                )
                
                # Collect metrics
                data_metrics = compute_data_metrics(batch=batch, use_critic=self.use_critic)
                metrics.update(data_metrics)

                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                # TODO: implement actual tflpo and theoretical tflpo
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))
                # Note: mismatch metrics (KL, PPL, etc.) are collected at advantage computation
                
                # Experimental: curriculum sampler update
                from verl.experimental.dataset.sampler import AbstractCurriculumSampler
                
                if isinstance(self.train_dataloader.sampler, AbstractCurriculumSampler):
                    self.train_dataloader.sampler.update(batch=batch)
                
                # TODO: make a canonical logger that supports various backend
                logger_tracking.log(data=metrics, step=self.global_steps)
                
                progress_bar.update(1)
                self.global_steps += 1
                
                # Memory snapshot if enabled
                if (
                    hasattr(self.config.actor_rollout_ref.actor, "profiler")
                    and self.config.actor_rollout_ref.actor.profiler.tool == "torch_memory"
                ):
                    self.actor_rollout_wg.dump_memory_snapshot(
                        tag=f"post_update_step{self.global_steps}", sub_dir=f"step{self.global_steps}"
                    )
                
                if is_last_step:
                    from pprint import pprint
                    
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return
                
                # Experimental: dataset on_batch_end callback
                if hasattr(self.train_dataset, "on_batch_end"):
                    # The dataset may be changed after each training batch
                    self.train_dataset.on_batch_end(batch=batch)
                
                # Update sampling state for resume support
                self.adaptive_sampling_state["current_epoch"] = epoch
                self.adaptive_sampling_state["current_idx_in_epoch"] = current_idx
            
            # End of epoch: reset index for next epoch
            self.adaptive_sampling_state["current_idx_in_epoch"] = 0
            print(f"[PDS] Completed epoch {epoch+1}")
        
        # Check if last step checkpoint exists
        import os
        
        checkpoint_dir = os.path.join(self.config.trainer.default_local_dir, f"global_step_{self.global_steps}")
        if not os.path.exists(checkpoint_dir):
            # Save last step checkpoint
            timing_raw = defaultdict(float)
            with marked_timer("save_checkpoint", timing_raw, "green"):
                self._save_checkpoint()
            metrics = {f"timing/{k}": v for k, v in timing_raw.items()}
            logger_tracking.log(data=metrics, step=self.global_steps)
    
    def _save_checkpoint(self):
        """Override to save adaptive sampling state for resume support."""
        # Call parent's save_checkpoint first
        super()._save_checkpoint()
        
        # Save adaptive sampling state if using dynamic sampling
        if self.dynamic_sampling_stats is not None:
            import os
            
            from verl.utils.fs import local_mkdir_safe
            
            local_global_step_folder = os.path.join(
                self.config.trainer.default_local_dir, f"global_step_{self.global_steps}"
            )
            local_mkdir_safe(local_global_step_folder)
            
            # Save adaptive sampling state
            adaptive_state_path = os.path.join(local_global_step_folder, "adaptive_sampling_state.pt")
            state_to_save = {
                "adaptive_sampling_state": self.adaptive_sampling_state,
                "dynamic_sampling_stats": {
                    "pass_rate_history": list(self.dynamic_sampling_stats.pass_rate_history),
                    "total_prompts_sampled": self.dynamic_sampling_stats.total_prompts_sampled,
                    "total_prompts_passed": self.dynamic_sampling_stats.total_prompts_passed,
                    "total_prompts_used": self.dynamic_sampling_stats.total_prompts_used,
                    "total_prompts_wasted": self.dynamic_sampling_stats.total_prompts_wasted,
                },
            }
            torch.save(state_to_save, adaptive_state_path)
            print(f"Saved adaptive sampling state to {adaptive_state_path}")
    
    def _load_checkpoint(self):
        """Override to load adaptive sampling state for resume support."""
        # Call parent's load_checkpoint first
        # Parent may return 0 (no checkpoint) or None (checkpoint loaded but no explicit return)
        parent_loaded = super()._load_checkpoint()
        # Normalize to an int: if parent returned None, infer from self.global_steps
        loaded_steps = parent_loaded if parent_loaded is not None else getattr(self, "global_steps", 0)
        
        # Load adaptive sampling state if using dynamic sampling
        if self.dynamic_sampling_stats is not None and loaded_steps > 0:
            import os
            
            checkpoint_folder = self.config.trainer.default_local_dir
            
            # Find the latest checkpoint
            checkpoints = [
                d for d in os.listdir(checkpoint_folder)
                if d.startswith("global_step_") and os.path.isdir(os.path.join(checkpoint_folder, d))
            ]
            
            if checkpoints:
                # Sort by step number
                checkpoints_with_steps = []
                for ckpt in checkpoints:
                    try:
                        step = int(ckpt.split("_")[-1])
                        checkpoints_with_steps.append((step, ckpt))
                    except ValueError:
                        continue
                
                if checkpoints_with_steps:
                    latest_step, latest_ckpt_dir = max(checkpoints_with_steps, key=lambda x: x[0])
                    adaptive_state_path = os.path.join(checkpoint_folder, latest_ckpt_dir, "adaptive_sampling_state.pt")
                    
                    if os.path.exists(adaptive_state_path):
                        state_loaded = torch.load(adaptive_state_path, weights_only=False)
                        
                        # Restore adaptive sampling state
                        self.adaptive_sampling_state = state_loaded["adaptive_sampling_state"]
                        
                        # Restore dynamic sampling stats
                        stats = state_loaded["dynamic_sampling_stats"]
                        self.dynamic_sampling_stats.pass_rate_history = deque(
                            stats["pass_rate_history"], 
                            maxlen=self.dynamic_sampling_stats.history_window
                        )
                        self.dynamic_sampling_stats.total_prompts_sampled = stats["total_prompts_sampled"]
                        self.dynamic_sampling_stats.total_prompts_passed = stats["total_prompts_passed"]
                        self.dynamic_sampling_stats.total_prompts_used = stats["total_prompts_used"]
                        self.dynamic_sampling_stats.total_prompts_wasted = stats["total_prompts_wasted"]
                        
                        print(f"Loaded adaptive sampling state from {adaptive_state_path}")
                        print(f"  - Resume from epoch {self.adaptive_sampling_state['current_epoch']}, "
                              f"index {self.adaptive_sampling_state['current_idx_in_epoch']}")
                    else:
                        print(f"Adaptive sampling state not found at {adaptive_state_path}")
        
        return loaded_steps
