import numpy as np

from rlpd.data.dataset import Dataset


def combine(offline_dict, online_dict):
    combined = {}
    for key, value in offline_dict.items():
        if isinstance(value, dict):
            combined[key] = combine(value, online_dict[key])
            continue
        tmp = np.empty(
            (value.shape[0] + online_dict[key].shape[0], *value.shape[1:]),
            dtype=value.dtype,
        )
        tmp[0::2] = value
        tmp[1::2] = online_dict[key]
        combined[key] = tmp
    return combined


def d4rl_normalize_return(env, episode_return):
    normalized_fn = getattr(env.unwrapped, "get_normalized_score", None)
    if normalized_fn is None:
        return episode_return
    return float(normalized_fn(episode_return) * 100.0)


def prefixed(metrics, prefix):
    return {f"{prefix}/{k}": v for k, v in metrics.items()}


def masked_dataset(dataset: Dataset, mask_prob: float, seed: int) -> Dataset:
    if not 0.0 < mask_prob < 1.0:
        raise ValueError(f"mask_prob must be in (0, 1), got {mask_prob}")

    rng = np.random.default_rng(seed)
    n = dataset.dataset_len
    mask = rng.binomial(1, mask_prob, n).astype(bool)

    if mask.sum() == 0:
        mask[rng.integers(0, n)] = True
    if mask.sum() == n:
        mask[rng.integers(0, n)] = False

    masked_dict = {k: v[mask] for k, v in dataset.dataset_dict.items()}
    return Dataset(masked_dict, seed=seed)
