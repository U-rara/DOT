import statistics
from collections import defaultdict

import numpy as np


def apply_length_truncation(
    data,
    reward_fn,
    success_threshold=1.0,
    std_multiplier=3.0,
    min_truncation_margin=32,
    metric_name="token_level_rewards",
):
    """Dynamic Outlier Truncation (DOT) over rollout groups.

    DOT is a training-time post-processing step applied after sampling a GRPO-style
    rollout group. It is only applied to rollout groups that the policy already
    solves reliably (paper default: `success_threshold=1.0`, i.e., all-correct groups).

    For a rollout group with response lengths {L_i}, compute:
      T(q) = floor(mean(L) + α * std(L))

    and truncate only the extreme tail (statistical outliers):
      if L_i - T(q) >= m, set L_i := T(q)

    After truncation, recompute the task reward on the modified rollouts.

    Args:
        success_threshold: Apply DOT when group success_rate >= this threshold.
            The paper uses 1.0 (all sampled responses correct).
        std_multiplier: α in T(q)=floor(mean(L)+α·std(L)). Paper default: 3.
        min_truncation_margin: m in (L - T(q)) >= m. Paper default: 32 tokens.
    """
    # Extract required fields
    response_mask = data.batch["response_mask"]  # (batch_size, response_length)
    attention_mask = data.batch["attention_mask"]  # (batch_size, sequence_length)
    token_level_rewards = data.batch[metric_name]  # (batch_size, response_length)
    index = data.non_tensor_batch["uid"]
    bsz = response_mask.shape[0]

    # Calculate prompt length (needed for truncating attention_mask)
    responses = data.batch["responses"]
    response_length = responses.shape[-1]
    sequence_length = attention_mask.shape[-1]
    prompt_length = sequence_length - response_length

    # Step 1: Group samples by uid and compute success rate
    uid2indices = defaultdict(list)
    uid2rewards = defaultdict(list)

    for i in range(bsz):
        uid = index[i]
        uid2indices[uid].append(i)
        total_reward = token_level_rewards[i].sum().item()
        uid2rewards[uid].append(total_reward)

    total_groups = len(uid2indices)
    all_correct_groups = sum(
        1 for uid, rewards in uid2rewards.items() if all(abs(r - 1.0) < 1e-6 for r in rewards)
    )
    all_wrong_groups = sum(1 for uid, rewards in uid2rewards.items() if all(abs(r - 0.0) < 1e-6 for r in rewards))

    # Step 2: Identify high success rate groups
    high_success_groups = {}
    for uid, rewards in uid2rewards.items():
        success_count = sum(1 for r in rewards if abs(r - 1.0) < 1e-6)
        success_rate = success_count / len(rewards)
        if success_rate >= success_threshold:
            high_success_groups[uid] = {
                "indices": uid2indices[uid],
                "success_rate": success_rate,
                "group_size": len(rewards),
            }

    if len(high_success_groups) == 0:
        metrics = {
            "group/all_correct_ratio": float(all_correct_groups / total_groups if total_groups > 0 else 0.0),
            "group/all_wrong_ratio": float(all_wrong_groups / total_groups if total_groups > 0 else 0.0),
            "truncation/correct_ratio_before": 0.0,
            "truncation/correct_ratio_after": 0.0,
            "truncation/broken_answer_ratio": 0.0,
            "truncation/total_tokens_truncated": 0.0,
        }
        return data, metrics

    # Step 3: Apply truncation to each high success group
    truncation_stats = []
    truncated_tokens_per_sample = []

    for uid, info in high_success_groups.items():
        group_indices = info["indices"]

        all_lengths = []
        correct_indices = []
        for idx in group_indices:
            length = int(response_mask[idx].sum().item())
            all_lengths.append(length)
            if abs(token_level_rewards[idx].sum().item() - 1.0) < 1e-6:
                correct_indices.append(idx)

        if len(correct_indices) == 0:
            continue

        mean_all = statistics.mean(all_lengths)
        std_all = statistics.stdev(all_lengths) if len(all_lengths) > 1 else 0
        truncate_len = int(mean_all + std_multiplier * std_all)

        original_lengths = []
        actually_truncated = 0
        skipped_by_margin = 0
        tokens_truncated = 0

        for idx in group_indices:
            original_len = int(response_mask[idx].sum().item())
            original_lengths.append(original_len)

            if truncate_len < original_len:
                margin = original_len - truncate_len
                if 0 < float(min_truncation_margin) < 1.0:
                    eff_margin = max(int(original_len * float(min_truncation_margin)), 16)
                else:
                    eff_margin = int(min_truncation_margin)

                if margin >= eff_margin:
                    response_mask[idx, truncate_len:] = 0
                    attention_mask[idx, prompt_length + truncate_len :] = 0
                    actually_truncated += 1
                    tokens_truncated += (original_len - truncate_len)
                    truncated_tokens_per_sample.append(original_len - truncate_len)
                else:
                    skipped_by_margin += 1

        truncation_stats.append(
            {
                "uid": uid,
                "group_size": len(group_indices),
                "correct_count_before": len(correct_indices),
                "mean_length": mean_all,
                "std_length": std_all,
                "truncate_len": truncate_len,
                "actually_truncated": actually_truncated,
                "skipped_by_margin": skipped_by_margin,
                "tokens_truncated": tokens_truncated,
            }
        )

    # Step 4: Recompute rewards after truncation
    if "rm_scores" in data.batch:
        del data.batch["rm_scores"]

    from verl.trainer.ppo.reward import compute_reward

    new_rewards, _ = compute_reward(data, reward_fn)
    data.batch[metric_name] = new_rewards

    # Step 5: Metrics
    high_success_count = len(high_success_groups)
    metrics = {
        "truncation/easy_ratio": float(high_success_count / total_groups if total_groups > 0 else 0.0),
        "group/all_correct_ratio": float(all_correct_groups / total_groups if total_groups > 0 else 0.0),
        "group/all_wrong_ratio": float(all_wrong_groups / total_groups if total_groups > 0 else 0.0),
    }

    if truncation_stats:
        total_before = sum(s["correct_count_before"] for s in truncation_stats)
        total_after = 0
        for stats in truncation_stats:
            group_indices = high_success_groups[stats["uid"]]["indices"]
            correct_after = sum(1 for idx in group_indices if abs(new_rewards[idx].sum().item() - 1.0) < 1e-6)
            total_after += correct_after

        total_truncated = sum(s["group_size"] for s in truncation_stats)
        total_actually_truncated = sum(s["actually_truncated"] for s in truncation_stats)
        total_skipped_by_margin = sum(s["skipped_by_margin"] for s in truncation_stats)
        total_tokens_truncated = sum(s["tokens_truncated"] for s in truncation_stats)

        metrics["truncation/correct_ratio_before"] = float(total_before / total_truncated if total_truncated > 0 else 0.0)
        metrics["truncation/correct_ratio_after"] = float(total_after / total_truncated if total_truncated > 0 else 0.0)
        metrics["truncation/broken_answer_ratio"] = float(
            (total_before - total_after) / total_before if total_before > 0 else 0.0
        )
        metrics["truncation/actually_truncated"] = float(total_actually_truncated)
        metrics["truncation/skipped_by_margin"] = float(total_skipped_by_margin)
        metrics["truncation/skip_ratio"] = float(
            total_skipped_by_margin / (total_actually_truncated + total_skipped_by_margin)
            if (total_actually_truncated + total_skipped_by_margin) > 0
            else 0.0
        )
        metrics["truncation/total_tokens_truncated"] = float(total_tokens_truncated)

        if truncated_tokens_per_sample:
            metrics["truncation/tokens_truncated/mean"] = float(statistics.mean(truncated_tokens_per_sample))
            metrics["truncation/tokens_truncated/min"] = float(min(truncated_tokens_per_sample))
            metrics["truncation/tokens_truncated/max"] = float(max(truncated_tokens_per_sample))
            percentiles = np.percentile(truncated_tokens_per_sample, [10, 25, 50, 75, 90])
            metrics["truncation/tokens_truncated/p10"] = float(percentiles[0])
            metrics["truncation/tokens_truncated/p25"] = float(percentiles[1])
            metrics["truncation/tokens_truncated/p50"] = float(percentiles[2])
            metrics["truncation/tokens_truncated/p75"] = float(percentiles[3])
            metrics["truncation/tokens_truncated/p90"] = float(percentiles[4])
        else:
            metrics["truncation/tokens_truncated/mean"] = 0.0
            metrics["truncation/tokens_truncated/min"] = 0.0
            metrics["truncation/tokens_truncated/max"] = 0.0
            metrics["truncation/tokens_truncated/p10"] = 0.0
            metrics["truncation/tokens_truncated/p25"] = 0.0
            metrics["truncation/tokens_truncated/p50"] = 0.0
            metrics["truncation/tokens_truncated/p75"] = 0.0
            metrics["truncation/tokens_truncated/p90"] = 0.0

        avg_mean_length = statistics.mean([s["mean_length"] for s in truncation_stats])
        avg_std_length = statistics.mean([s["std_length"] for s in truncation_stats])
        avg_truncate_len = statistics.mean([s["truncate_len"] for s in truncation_stats])
        metrics["truncation/avg_mean_length"] = float(avg_mean_length)
        metrics["truncation/avg_std_length"] = float(avg_std_length)
        metrics["truncation/avg_truncate_len"] = float(avg_truncate_len)

    return data, metrics
