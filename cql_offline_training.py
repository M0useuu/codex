#! /usr/bin/env python
import os
import d4rl
import d4rl.gym_mujoco
import d4rl.locomotion
import gym
import tqdm
import wandb
from absl import app, flags
from flax.training import checkpoints
from ml_collections import config_flags

from cql_finetuning_utils import d4rl_normalize_return, prefixed
from rlpd.agents import CQLLearner, SACLearner
from rlpd.data.d4rl_datasets import D4RLDataset
from rlpd.evaluation import evaluate
from rlpd.wrappers import wrap_gym

from rlpd.data.binary_datasets import BinaryDataset

FLAGS = flags.FLAGS

flags.DEFINE_string("project_name", "rlpd", "wandb project name.")
flags.DEFINE_string("env_name", "halfcheetah-expert-v2", "D4rl dataset name.")
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_integer("eval_episodes", 10, "Number of episodes used for evaluation.")
flags.DEFINE_integer("log_interval", 1000, "Logging interval.")
flags.DEFINE_integer("eval_interval", 5000, "Eval interval.")
flags.DEFINE_integer("batch_size", 256, "Mini batch size.")
flags.DEFINE_integer("pretrain_steps", int(1e6), "Number of offline updates.")
flags.DEFINE_boolean("tqdm", True, "Use tqdm progress bar.")
flags.DEFINE_boolean("save_video", False, "Save videos during evaluation.")
flags.DEFINE_integer("offline_utd_ratio", 1, "Offline update-to-data ratio.")
flags.DEFINE_boolean(
    "binary_include_bc", True, "Whether to include BC data in the binary datasets."
)
flags.DEFINE_string(
    "offline_checkpoint_dir",
    None,
    "Directory to save offline weights. Will save ckpt_<pretrain_steps> into this dir.",
)

config_flags.DEFINE_config_file(
    "config",
    "configs/cql_config.py",
    "File path to the training hyperparameter configuration.",
    lock_config=False,
)


def compute_mean_std(observations, eps: float = 1e-3) -> Tuple:
    mean = observations.mean(axis=0)
    std = observations.std(axis=0) + eps
    return mean, std


def normalize_states(states, mean, std):
    return (states - mean) / std


def maybe_modify_reward(dataset_dict, env_name: str, reward_scale: float, reward_bias: float):
    if any(s in env_name for s in ("halfcheetah", "hopper", "walker2d")):
        # Match common D4RL normalization: scale rewards to roughly per-episode horizon.
        episode_returns = []
        episode_return = 0.0
        for r, done in zip(dataset_dict["rewards"], dataset_dict["dones"]):
            episode_return += float(r)
            if done:
                episode_returns.append(episode_return)
                episode_return = 0.0

        if episode_returns:
            ret_min, ret_max = min(episode_returns), max(episode_returns)
            if ret_max > ret_min:
                dataset_dict["rewards"] /= ret_max - ret_min
                dataset_dict["rewards"] *= 1000.0

    dataset_dict["rewards"] = dataset_dict["rewards"] * reward_scale + reward_bias


def main(_):
    if FLAGS.offline_checkpoint_dir is None:
        raise ValueError("--offline_checkpoint_dir must be provided for offline training.")

    wandb.init(project=FLAGS.project_name)
    wandb.config.update(FLAGS)

    env = gym.make(FLAGS.env_name)
    env = wrap_gym(env, rescale_actions=True)
    env.seed(FLAGS.seed)

    if "binary" in FLAGS.env_name:
        ds = BinaryDataset(env, include_bc_data=FLAGS.binary_include_bc)
    else:
        ds = D4RLDataset(env)

    if getattr(FLAGS.config, "normalize_reward", False):
        maybe_modify_reward(
            ds.dataset_dict,
            FLAGS.env_name,
            reward_scale=getattr(FLAGS.config, "reward_scale", 1.0),
            reward_bias=getattr(FLAGS.config, "reward_bias", 0.0),
        )

    state_mean, state_std = None, None
    if getattr(FLAGS.config, "normalize_states", False):
        state_mean, state_std = compute_mean_std(ds.dataset_dict["observations"])
        ds.dataset_dict["observations"] = normalize_states(
            ds.dataset_dict["observations"], state_mean, state_std
        )
        ds.dataset_dict["next_observations"] = normalize_states(
            ds.dataset_dict["next_observations"], state_mean, state_std
        )

        def normalize_state(obs):
            return (obs - state_mean) / state_std

        env = gym.wrappers.TransformObservation(env, normalize_state)

    eval_env = gym.make(FLAGS.env_name)
    eval_env = wrap_gym(eval_env, rescale_actions=True)
    if state_mean is not None:
        eval_env = gym.wrappers.TransformObservation(
            eval_env, lambda obs: (obs - state_mean) / state_std
        )
    eval_env.seed(FLAGS.seed + 42)

    kwargs = dict(FLAGS.config)
    model_cls = kwargs.pop("model_cls")
    agent = globals()[model_cls].create(
        FLAGS.seed, env.observation_space, env.action_space, **kwargs
    )

    for i in tqdm.tqdm(range(FLAGS.pretrain_steps), smoothing=0.1, disable=not FLAGS.tqdm):
        offline_batch = ds.sample(FLAGS.batch_size * FLAGS.offline_utd_ratio)
        if "antmaze" in FLAGS.env_name:
            offline_batch["rewards"] = offline_batch["rewards"] * 10.0 - 5.0

        agent, update_info = agent.update(offline_batch, FLAGS.offline_utd_ratio)

        if i % FLAGS.log_interval == 0:
            wandb.log(prefixed(update_info, "offline-training"), step=i)

        if i % FLAGS.eval_interval == 0:
            eval_info = evaluate(
                agent,
                eval_env,
                num_episodes=FLAGS.eval_episodes,
                save_video=FLAGS.save_video,
            )
            eval_info["return"] = d4rl_normalize_return(eval_env, eval_info["return"])
            wandb.log(prefixed(eval_info, "offline-evaluation"), step=i)

    os.makedirs(FLAGS.offline_checkpoint_dir, exist_ok=True)
    checkpoints.save_checkpoint(
        FLAGS.offline_checkpoint_dir,
        agent,
        step=FLAGS.pretrain_steps,
        overwrite=True,
        keep=20,
    )


if __name__ == "__main__":
    app.run(main)
