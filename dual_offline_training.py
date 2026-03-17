#! /usr/bin/env python
import os

import d4rl
import d4rl.gym_mujoco
import d4rl.locomotion
import gym
import numpy as np
import tqdm
import wandb
from absl import app, flags
from flax.training import checkpoints
from ml_collections import config_flags

from cql_finetuning_utils import d4rl_normalize_return, prefixed
from rlpd.agents import DualAdaptiveLearner
from rlpd.data.binary_datasets import BinaryDataset
from rlpd.data.d4rl_datasets import D4RLDataset
from rlpd.evaluation import evaluate
from rlpd.wrappers import wrap_gym

FLAGS = flags.FLAGS

flags.DEFINE_string("project_name", "rlpd", "wandb project name.")
flags.DEFINE_string("env_name", "halfcheetah-expert-v2", "D4rl dataset name.")
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_integer("eval_episodes", 10, "Number of episodes used for evaluation.")
flags.DEFINE_integer("log_interval", 1000, "Logging interval.")
flags.DEFINE_integer("eval_interval", 5000, "Eval interval.")
flags.DEFINE_integer("batch_size", 256, "Per-agent mini batch size.")
flags.DEFINE_integer("pretrain_steps", int(1e6), "Number of offline updates.")
flags.DEFINE_boolean("tqdm", True, "Use tqdm progress bar.")
flags.DEFINE_boolean("save_video", False, "Save videos during evaluation.")
flags.DEFINE_integer("offline_utd_ratio", 1, "Offline update-to-data ratio.")
flags.DEFINE_float(
    "offline_mask_ratio",
    0.9,
    "Mask ratio used by convert_d4rl_mask to split one offline batch into two datasets.",
)
flags.DEFINE_boolean(
    "binary_include_bc", True, "Whether to include BC data in the binary datasets."
)
flags.DEFINE_string(
    "offline_checkpoint_dir",
    None,
    "Absolute directory to save offline weights. Will save dual ckpt and per-agent ckpts.",
)

config_flags.DEFINE_config_file(
    "config",
    "configs/dual_adaptive_config.py",
    "File path to the training hyperparameter configuration.",
    lock_config=False,
)


def _index_batch(batch, idx):
    return {k: v[idx] for k, v in batch.items()}


def _resample_to_size(batch, size, rng):
    n = batch["observations"].shape[0]
    if n == 0:
        raise ValueError("Masked batch is empty. Adjust --offline_mask_ratio.")
    if n == size:
        return batch
    replace = n < size
    idx = rng.choice(n, size=size, replace=replace)
    return _index_batch(batch, idx)


def convert_d4rl_mask(batch, mask_ratio, target_size, rng):
    """Split one offline batch into two masked datasets with resampling to equal size."""

    n = batch["observations"].shape[0]
    mask = rng.uniform(size=n) < mask_ratio
    if mask.sum() == 0:
        mask[rng.integers(0, n)] = True
    if mask.sum() == n:
        mask[rng.integers(0, n)] = False

    batch_a = _index_batch(batch, np.where(mask)[0])
    batch_b = _index_batch(batch, np.where(~mask)[0])

    batch_a = _resample_to_size(batch_a, target_size, rng)
    batch_b = _resample_to_size(batch_b, target_size, rng)
    return batch_a, batch_b


def _save_two_agent_checkpoints(agent, checkpoint_dir, step):
    agent1_state = {
        "actor": agent.actor,
        "critic": agent.critic,
        "target_critic": agent.target_critic,
        "temp": agent.temp,
    }
    agent2_state = {
        "actor": agent.actor2,
        "critic": agent.critic2,
        "target_critic": agent.target_critic2,
        "temp": agent.temp2,
    }

    checkpoints.save_checkpoint(
        os.path.join(checkpoint_dir, "agent1"),
        agent1_state,
        step=step,
        overwrite=True,
        keep=20,
    )
    checkpoints.save_checkpoint(
        os.path.join(checkpoint_dir, "agent2"),
        agent2_state,
        step=step,
        overwrite=True,
        keep=20,
    )


def main(_):
    if FLAGS.offline_checkpoint_dir is None:
        raise ValueError("--offline_checkpoint_dir must be provided for offline training.")
    if not os.path.isabs(FLAGS.offline_checkpoint_dir):
        raise ValueError(
            "--offline_checkpoint_dir must be an absolute path, "
            f"got: {FLAGS.offline_checkpoint_dir}"
        )
    if not 0.0 < FLAGS.offline_mask_ratio < 1.0:
        raise ValueError("--offline_mask_ratio must be in (0, 1).")

    rng = np.random.default_rng(FLAGS.seed)

    wandb.init(project=FLAGS.project_name)
    wandb.config.update(FLAGS)

    env = gym.make(FLAGS.env_name)
    env = wrap_gym(env, rescale_actions=False)
    env.seed(FLAGS.seed)

    if "binary" in FLAGS.env_name:
        ds = BinaryDataset(env, include_bc_data=FLAGS.binary_include_bc)
    else:
        ds = D4RLDataset(env)

    eval_env = gym.make(FLAGS.env_name)
    eval_env = wrap_gym(eval_env, rescale_actions=False)
    eval_env.seed(FLAGS.seed + 42)

    kwargs = dict(FLAGS.config)
    model_cls = kwargs.pop("model_cls")
    agent = globals()[model_cls].create(
        FLAGS.seed, env.observation_space, env.action_space, **kwargs
    )

    per_agent_size = FLAGS.batch_size * FLAGS.offline_utd_ratio

    for i in tqdm.tqdm(range(FLAGS.pretrain_steps), smoothing=0.1, disable=not FLAGS.tqdm):
        sampled = ds.sample(2 * per_agent_size)
        if "antmaze" in FLAGS.env_name:
            sampled["rewards"] = sampled["rewards"] * 10.0 - 5.0

        offline_batch_0, offline_batch_1 = convert_d4rl_mask(
            sampled,
            FLAGS.offline_mask_ratio,
            per_agent_size,
            rng,
        )

        agent, update_info = agent.update_offline(
            offline_batch_0, offline_batch_1, FLAGS.offline_utd_ratio
        )

        if i % FLAGS.log_interval == 0:
            update_info["offline_mask_ratio"] = FLAGS.offline_mask_ratio
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
        os.path.join(FLAGS.offline_checkpoint_dir, "dual"),
        agent,
        step=FLAGS.pretrain_steps,
        overwrite=True,
        keep=20,
    )
    _save_two_agent_checkpoints(agent, FLAGS.offline_checkpoint_dir, FLAGS.pretrain_steps)


if __name__ == "__main__":
    app.run(main)
