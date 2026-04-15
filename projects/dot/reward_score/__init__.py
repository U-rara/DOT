import logging
import ray

from projects.dot.reward_score.math_reward import compute_score as local_math_compute_score

logger = logging.getLogger(__name__)


@ray.remote(num_cpus=1)
def _deepscaler_compute_score_remote(solution_str, ground_truth):
    """Ray worker for heavy deepscaler math reward."""
    return local_math_compute_score(solution_str, ground_truth)


def default_compute_score(
        data_source,
        solution_str,
        ground_truth,
        extra_info=None,
        **kwargs,
):
    """
    Compute score for different datasets.

    For deepscaler datasets, the heavy math reward is executed in a Ray
    worker with a 5-second timeout to avoid hangs. Other datasets are
    computed synchronously in-process.
    """
    data_source_lower = str(data_source).lower()

    try:
        # deepscaler datasets: use local math_reward via Ray with timeout
        if "deepscaler" in data_source_lower:
            obj_ref = _deepscaler_compute_score_remote.remote(solution_str, ground_truth)
            try:
                # 5s timeout to avoid long-hanging sympy evaluations
                res = ray.get(obj_ref, timeout=5.0)
            except Exception as e:
                # Timeout or Ray task error
                try:
                    ray.cancel(obj_ref, force=True)
                except Exception:
                    pass
                logger.error(f"Timeout/error in deepscaler reward for {data_source=}: {e}")
                return {
                    "score": 0.0,
                    "acc": 0.0,
                }
        # aime/amc datasets: use math_dapo
        elif "aime" in data_source_lower or "amc" in data_source_lower:
            from verl.utils.reward_score import math_dapo
            res = math_dapo.compute_score(solution_str, ground_truth, strict_box_verify=True)
        # math-500 dataset: use verl math_reward
        elif "math-500" in data_source_lower:
            from verl.utils.reward_score import math_reward
            res = math_reward.compute_score(solution_str, ground_truth)
        else:
            raise NotImplementedError(f"Reward function is not implemented for {data_source=}")

        if isinstance(res, dict):
            # Keep only numeric fields (int/float/bool) to avoid issues
            # in downstream validation metrics (e.g., drop string fields like 'pred').
            numeric_res = {
                key: value
                for key, value in res.items()
                if isinstance(value, (int, float, bool))
            }
            return numeric_res
        else:
            return {
                "score": res,
                "acc": 1.0 if res == 1.0 else 0.0,
            }
    except Exception as e:
        logger.error(f"Error computing score for data_source={data_source}: {e}")
        return {
            "score": 0.0,
            "acc": 0.0,
        }
