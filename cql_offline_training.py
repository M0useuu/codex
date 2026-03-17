#! /usr/bin/env python
import os

import d4rl
import d4rl.gym_mujoco
import d4rl.locomotion
import dmcgym
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

    eval_env = gym.make(FLAGS.env_name)
    eval_env = wrap_gym(eval_env, rescale_actions=True)
    eval_env.seed(FLAGS.seed + 42)

    kwargs = dict(FLAGS.config)
    model_cls = kwargs.pop("model_cls")
    agent = globals()[model_cls].create(
        FLAGS.seed, env.observation_space, env.action_space, **kwargs
    )

    for i in tqdm.tqdm(range(FLAGS.pretrain_steps), smoothing=0.1, disable=not FLAGS.tqdm):
        offline_batch = ds.sample(FLAGS.batch_size * FLAGS.offline_utd_ratio)
        if "antmaze" in FLAGS.env_name:
            offline_batch["rewards"] = offline_batch["rewards"] - 1

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
