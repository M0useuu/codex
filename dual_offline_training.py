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

from rlpd.agents import CQLLearner
from rlpd.data.binary_datasets import BinaryDataset
from rlpd.data.d4rl_datasets import D4RLDataset
from rlpd.evaluation import evaluate
from rlpd.wrappers import wrap_gym
from utils import d4rl_normalize_return, masked_dataset, prefixed

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
    "offline_mask_prob",
    0.9,
    "Mask probability used to initialize two masked offline datasets.",
)
flags.DEFINE_boolean(
    "binary_include_bc", True, "Whether to include BC data in the binary datasets."
)
flags.DEFINE_string(
    "offline_checkpoint_dir",
    None,
    "Absolute directory to save offline weights. Saves agent1 and agent2 CQL checkpoints.",
)

config_flags.DEFINE_config_file(
    "config",
    "configs/cql_config.py",
    "File path to the offline CQL hyperparameter configuration.",
    lock_config=False,
)


def _save_single_agent_checkpoint(agent, checkpoint_dir, step):
    payload = {
        "actor": agent.actor,
        "critic": agent.critic,
        "target_critic": agent.target_critic,
        "temp": agent.temp,
    }
    checkpoints.save_checkpoint(
        checkpoint_dir,
        payload,
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

    wandb.init(project=FLAGS.project_name)
    wandb.config.update(FLAGS)

    env = gym.make(FLAGS.env_name)
    env = wrap_gym(env, rescale_actions=False)
    env.seed(FLAGS.seed)

    if "binary" in FLAGS.env_name:
        raw_ds = BinaryDataset(env, include_bc_data=FLAGS.binary_include_bc)
    else:
        raw_ds = D4RLDataset(env)

    ds_agent1 = masked_dataset(raw_ds, FLAGS.offline_mask_prob, FLAGS.seed)
    ds_agent2 = masked_dataset(raw_ds, FLAGS.offline_mask_prob, FLAGS.seed + 1)

    eval_env = gym.make(FLAGS.env_name)
    eval_env = wrap_gym(eval_env, rescale_actions=False)
    eval_env.seed(FLAGS.seed + 42)

    kwargs = dict(FLAGS.config)
    model_cls = kwargs.pop("model_cls")
    if model_cls != "CQLLearner":
        raise ValueError(f"dual_offline_training requires CQLLearner config, got: {model_cls}")

    agent1 = globals()[model_cls].create(
        FLAGS.seed, env.observation_space, env.action_space, **kwargs
    )
    agent2 = globals()[model_cls].create(
        FLAGS.seed + 1, env.observation_space, env.action_space, **kwargs
    )

    for i in tqdm.tqdm(range(FLAGS.pretrain_steps), smoothing=0.1, disable=not FLAGS.tqdm):
        offline_batch_1 = ds_agent1.sample(FLAGS.batch_size * FLAGS.offline_utd_ratio)
        offline_batch_2 = ds_agent2.sample(FLAGS.batch_size * FLAGS.offline_utd_ratio)
        if "antmaze" in FLAGS.env_name:
            offline_batch_1["rewards"] = offline_batch_1["rewards"] * 10.0 - 5.0
            offline_batch_2["rewards"] = offline_batch_2["rewards"] * 10.0 - 5.0

        agent1, update_info_1 = agent1.update(offline_batch_1, FLAGS.offline_utd_ratio)
        agent2, update_info_2 = agent2.update(offline_batch_2, FLAGS.offline_utd_ratio)

        if i % FLAGS.log_interval == 0:
            metrics = {
                **prefixed(update_info_1, "offline-training/agent1"),
                **prefixed(update_info_2, "offline-training/agent2"),
                "offline-training/offline_mask_prob": FLAGS.offline_mask_prob,
            }
            wandb.log(metrics, step=i)

        if i % FLAGS.eval_interval == 0:
            eval_info_1 = evaluate(
                agent1,
                eval_env,
                num_episodes=FLAGS.eval_episodes,
                save_video=FLAGS.save_video,
            )
            eval_info_2 = evaluate(
                agent2,
                eval_env,
                num_episodes=FLAGS.eval_episodes,
                save_video=FLAGS.save_video,
            )
            eval_info_1["return"] = d4rl_normalize_return(eval_env, eval_info_1["return"])
            eval_info_2["return"] = d4rl_normalize_return(eval_env, eval_info_2["return"])
            wandb.log(
                {
                    **prefixed(eval_info_1, "offline-evaluation/agent1"),
                    **prefixed(eval_info_2, "offline-evaluation/agent2"),
                },
                step=i,
            )

    agent1_dir = os.path.join(FLAGS.offline_checkpoint_dir, "agent1")
    agent2_dir = os.path.join(FLAGS.offline_checkpoint_dir, "agent2")
    os.makedirs(agent1_dir, exist_ok=True)
    os.makedirs(agent2_dir, exist_ok=True)

    _save_single_agent_checkpoint(agent1, agent1_dir, FLAGS.pretrain_steps)
    _save_single_agent_checkpoint(agent2, agent2_dir, FLAGS.pretrain_steps)


if __name__ == "__main__":
    app.run(main)
