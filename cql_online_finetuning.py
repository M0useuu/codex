#! /usr/bin/env python
import os
import pickle

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

from cql_finetuning_utils import combine, d4rl_normalize_return, prefixed
from rlpd.agents import CQLLearner, SACLearner
from rlpd.data import ReplayBuffer
from rlpd.data.d4rl_datasets import D4RLDataset
from rlpd.evaluation import evaluate
from rlpd.wrappers import wrap_gym

from rlpd.data.binary_datasets import BinaryDataset

FLAGS = flags.FLAGS

flags.DEFINE_string("project_name", "rlpd", "wandb project name.")
flags.DEFINE_string("env_name", "halfcheetah-expert-v2", "D4rl dataset name.")
flags.DEFINE_float("offline_ratio", 0.5, "Offline ratio.")
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_integer("eval_episodes", 10, "Number of episodes used for evaluation.")
flags.DEFINE_integer("log_interval", 1000, "Logging interval.")
flags.DEFINE_integer("eval_interval", 5000, "Eval interval.")
flags.DEFINE_integer("batch_size", 256, "Mini batch size.")
flags.DEFINE_integer("max_steps", int(1e6), "Number of training steps.")
flags.DEFINE_integer(
    "start_training", int(1e4), "Number of training steps to start training."
)
flags.DEFINE_integer("pretrain_steps", 0, "Global step offset for logging.")
flags.DEFINE_boolean("tqdm", True, "Use tqdm progress bar.")
flags.DEFINE_boolean("save_video", False, "Save videos during evaluation.")
flags.DEFINE_boolean("checkpoint_model", False, "Save agent checkpoint on evaluation.")
flags.DEFINE_boolean(
    "checkpoint_buffer", False, "Save agent replay buffer on evaluation."
)
flags.DEFINE_integer("online_utd_ratio", 5, "Online update-to-data ratio.")
flags.DEFINE_boolean(
    "binary_include_bc", True, "Whether to include BC data in the binary datasets."
)
flags.DEFINE_string(
    "offline_checkpoint_dir",
    None,
    "Directory of offline weights. Online finetuning must load from this dir.",
)

config_flags.DEFINE_config_file(
    "config",
    "configs/cql_config.py",
    "File path to the training hyperparameter configuration.",
    lock_config=False,
)


def main(_):
    assert 0.0 <= FLAGS.offline_ratio <= 1.0
    if FLAGS.offline_checkpoint_dir is None:
        raise ValueError("--offline_checkpoint_dir must be provided for online finetuning.")

    wandb.init(project=FLAGS.project_name)
    wandb.config.update(FLAGS)

    exp_prefix = f"s{FLAGS.seed}_online"
    log_dir = os.path.join(FLAGS.log_dir, exp_prefix)

    if FLAGS.checkpoint_model:
        chkpt_dir = os.path.join(log_dir, "checkpoints")
        os.makedirs(chkpt_dir, exist_ok=True)

    if FLAGS.checkpoint_buffer:
        buffer_dir = os.path.join(log_dir, "buffers")
        os.makedirs(buffer_dir, exist_ok=True)

    env = gym.make(FLAGS.env_name)
    env = wrap_gym(env, rescale_actions=False)
    env = gym.wrappers.RecordEpisodeStatistics(env, deque_size=1)
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

    latest_ckpt = checkpoints.latest_checkpoint(FLAGS.offline_checkpoint_dir)
    if latest_ckpt is None:
        raise FileNotFoundError(
            f"No checkpoint found in --offline_checkpoint_dir={FLAGS.offline_checkpoint_dir}."
        )
    agent = checkpoints.restore_checkpoint(FLAGS.offline_checkpoint_dir, target=agent)

    replay_buffer = ReplayBuffer(
        env.observation_space, env.action_space, FLAGS.max_steps
    )
    replay_buffer.seed(FLAGS.seed)

    latest_episode_metrics = None
    observation, done = env.reset(), False
    for i in tqdm.tqdm(range(FLAGS.max_steps + 1), smoothing=0.1, disable=not FLAGS.tqdm):
        if i < FLAGS.start_training:
            action = env.action_space.sample()
        else:
            action, agent = agent.sample_actions(observation)

        next_observation, reward, done, info = env.step(action)
        mask = 1.0 if (not done or "TimeLimit.truncated" in info) else 0.0

        replay_buffer.insert(
            dict(
                observations=observation,
                actions=action,
                rewards=reward,
                masks=mask,
                dones=done,
                next_observations=next_observation,
            )
        )
        observation = next_observation

        if done:
            observation, done = env.reset(), False
            latest_episode_metrics = {
                "return": d4rl_normalize_return(eval_env, info["episode"]["r"]),
                "length": info["episode"]["l"],
                "time": info["episode"]["t"],
            }

        if i >= FLAGS.start_training:
            online_batch = replay_buffer.sample(
                int(FLAGS.batch_size * FLAGS.online_utd_ratio * (1 - FLAGS.offline_ratio))
            )
            offline_batch = ds.sample(
                int(FLAGS.batch_size * FLAGS.online_utd_ratio * FLAGS.offline_ratio)
            )
            batch = combine(offline_batch, online_batch)
            if "antmaze" in FLAGS.env_name:
                batch["rewards"] = batch["rewards"] * 10.0 - 5.0

            agent, update_info = agent.update(batch, FLAGS.online_utd_ratio)

            if i % FLAGS.log_interval == 0:
                train_metrics = prefixed(update_info, "training")
                if latest_episode_metrics is not None:
                    train_metrics.update(prefixed(latest_episode_metrics, "training"))
                wandb.log(train_metrics, step=i + FLAGS.pretrain_steps)

        if i % FLAGS.eval_interval == 0:
            eval_info = evaluate(
                agent,
                eval_env,
                num_episodes=FLAGS.eval_episodes,
                save_video=FLAGS.save_video,
            )
            eval_info["return"] = d4rl_normalize_return(eval_env, eval_info["return"])
            wandb.log(prefixed(eval_info, "evaluation"), step=i + FLAGS.pretrain_steps)

            if FLAGS.checkpoint_model:
                checkpoints.save_checkpoint(
                    chkpt_dir, agent, step=i, keep=20, overwrite=True
                )

            if FLAGS.checkpoint_buffer:
                with open(os.path.join(buffer_dir, "buffer"), "wb") as f:
                    pickle.dump(replay_buffer, f, pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    app.run(main)
