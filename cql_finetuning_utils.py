import numpy as np


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
