#! /usr/bin/env python
import os
import pickle

import d4rl
import d4rl.gym_mujoco
import d4rl.locomotion
import dmcgym
import gym
import numpy as np
import tqdm
from absl import app, flags

_HAS_CHECKPOINTS = True
try:
    from flax.training import checkpoints
except Exception:
    _HAS_CHECKPOINTS = False
    print("Not loading checkpointing functionality.")
from ml_collections import config_flags

import wandb
from rlpd.agents import CQLLearner, SACLearner
from rlpd.data import ReplayBuffer
from rlpd.data.d4rl_datasets import D4RLDataset

try:
    from rlpd.data.binary_datasets import BinaryDataset
except Exception:
    print("not importing binary dataset")
from rlpd.evaluation import evaluate
from rlpd.wrappers import wrap_gym

FLAGS = flags.FLAGS

flags.DEFINE_string("project_name", "rlpd", "wandb project name.")
flags.DEFINE_string("env_name", "halfcheetah-expert-v2", "D4rl dataset name.")
flags.DEFINE_float("offline_ratio", 0.5, "Offline ratio for online finetuning.")
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_integer("eval_episodes", 10, "Number of episodes used for evaluation.")
flags.DEFINE_integer("log_interval", 1000, "Logging interval.")
flags.DEFINE_integer("eval_interval", 5000, "Eval interval.")
flags.DEFINE_integer("batch_size", 256, "Mini batch size.")
flags.DEFINE_integer("max_steps", int(1e6), "Number of online training steps.")
flags.DEFINE_integer(
    "start_training", int(1e4), "Number of training steps to start online updates."
)
flags.DEFINE_integer("pretrain_steps", 0, "Number of offline updates.")
flags.DEFINE_boolean("tqdm", True, "Use tqdm progress bar.")
flags.DEFINE_boolean("save_video", False, "Save videos during evaluation.")
flags.DEFINE_boolean("checkpoint_model", False, "Save agent checkpoint on evaluation.")
flags.DEFINE_boolean(
    "checkpoint_buffer", False, "Save agent replay buffer on evaluation."
)
flags.DEFINE_integer("offline_utd_ratio", 1, "Offline update-to-data ratio.")
flags.DEFINE_integer("online_utd_ratio", 5, "Online update-to-data ratio.")
flags.DEFINE_boolean(
    "binary_include_bc", True, "Whether to include BC data in the binary datasets."
)
flags.DEFINE_enum(
    "mode",
    "both",
    ["offline", "online", "both"],
    "Run mode: offline-only pretraining, online-only finetuning (load checkpoint), or both.",
)
flags.DEFINE_string(
    "offline_model_dir",
    None,
    "Directory for saving/loading offline pretrained model checkpoints.",
)
flags.DEFINE_integer(
    "offline_model_step",
    -1,
    "Checkpoint step to load for online mode. -1 means latest.",
)

config_flags.DEFINE_config_file(
    "config",
    "configs/cql_config.py",
    "File path to the training hyperparameter configuration.",
    lock_config=False,
)


def combine(one_dict, other_dict):
    combined = {}

    for k, v in one_dict.items():
        if isinstance(v, dict):
            combined[k] = combine(v, other_dict[k])
        else:
            tmp = np.empty(
                (v.shape[0] + other_dict[k].shape[0], *v.shape[1:]), dtype=v.dtype
            )
            tmp[0::2] = v
            tmp[1::2] = other_dict[k]
            combined[k] = tmp

    return combined


def make_envs_and_dataset():
    env = gym.make(FLAGS.env_name)
    env = wrap_gym(env, rescale_actions=True)
    env = gym.wrappers.RecordEpisodeStatistics(env, deque_size=1)
    env.seed(FLAGS.seed)

    if "binary" in FLAGS.env_name:
        ds = BinaryDataset(env, include_bc_data=FLAGS.binary_include_bc)
    else:
        ds = D4RLDataset(env)

    eval_env = gym.make(FLAGS.env_name)
    eval_env = wrap_gym(eval_env, rescale_actions=True)
    eval_env.seed(FLAGS.seed + 42)

    return env, eval_env, ds


def create_agent(env):
    kwargs = dict(FLAGS.config)
    model_cls = kwargs.pop("model_cls")
    return globals()[model_cls].create(
        FLAGS.seed, env.observation_space, env.action_space, **kwargs
    )


def maybe_get_model_dir(base_log_dir):
    if FLAGS.offline_model_dir is not None:
        model_dir = FLAGS.offline_model_dir
    else:
        model_dir = os.path.join(base_log_dir, "offline_checkpoints")
    os.makedirs(model_dir, exist_ok=True)
    return model_dir


def run_offline(agent, ds, eval_env, base_log_step=0):
    for i in tqdm.tqdm(
        range(0, FLAGS.pretrain_steps), smoothing=0.1, disable=not FLAGS.tqdm
    ):
        offline_batch = ds.sample(FLAGS.batch_size * FLAGS.offline_utd_ratio)
        batch = {}
        for k, v in offline_batch.items():
            batch[k] = v
            if "antmaze" in FLAGS.env_name and k == "rewards":
                batch[k] -= 1

        agent, update_info = agent.update(batch, FLAGS.offline_utd_ratio)

        if i % FLAGS.log_interval == 0:
            for k, v in update_info.items():
                wandb.log({f"offline-training/{k}": v}, step=base_log_step + i)

        if i % FLAGS.eval_interval == 0:
            eval_info = evaluate(agent, eval_env, num_episodes=FLAGS.eval_episodes)
            for k, v in eval_info.items():
                wandb.log({f"offline-evaluation/{k}": v}, step=base_log_step + i)

    return agent


def save_offline_agent(agent, model_dir):
    if not _HAS_CHECKPOINTS:
        raise RuntimeError("flax.training.checkpoints is unavailable; cannot save model.")
    checkpoints.save_checkpoint(
        ckpt_dir=model_dir,
        target=agent,
        step=FLAGS.pretrain_steps,
        keep=20,
        overwrite=True,
    )


def load_offline_agent(agent, model_dir):
    if not _HAS_CHECKPOINTS:
        raise RuntimeError("flax.training.checkpoints is unavailable; cannot load model.")
    restored = checkpoints.restore_checkpoint(
        ckpt_dir=model_dir,
        target=agent,
        step=None if FLAGS.offline_model_step < 0 else FLAGS.offline_model_step,
    )
    return restored


def run_online(agent, env, ds, eval_env, log_dir, log_step_offset=0):
    if FLAGS.checkpoint_model and _HAS_CHECKPOINTS:
        chkpt_dir = os.path.join(log_dir, "checkpoints")
        os.makedirs(chkpt_dir, exist_ok=True)
    else:
        chkpt_dir = None

    if FLAGS.checkpoint_buffer:
        buffer_dir = os.path.join(log_dir, "buffers")
        os.makedirs(buffer_dir, exist_ok=True)
    else:
        buffer_dir = None

    replay_buffer = ReplayBuffer(env.observation_space, env.action_space, FLAGS.max_steps)
    replay_buffer.seed(FLAGS.seed)

    observation, done = env.reset(), False
    for i in tqdm.tqdm(
        range(0, FLAGS.max_steps + 1), smoothing=0.1, disable=not FLAGS.tqdm
    ):
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
            for k, v in info["episode"].items():
                decode = {"r": "return", "l": "length", "t": "time"}
                wandb.log({f"training/{decode[k]}": v}, step=i + log_step_offset)

        if i >= FLAGS.start_training:
            online_batch = replay_buffer.sample(
                int(FLAGS.batch_size * FLAGS.online_utd_ratio * (1 - FLAGS.offline_ratio))
            )
            offline_batch = ds.sample(
                int(FLAGS.batch_size * FLAGS.online_utd_ratio * FLAGS.offline_ratio)
            )
            batch = combine(offline_batch, online_batch)

            if "antmaze" in FLAGS.env_name:
                batch["rewards"] -= 1

            agent, update_info = agent.update(batch, FLAGS.online_utd_ratio)

            if i % FLAGS.log_interval == 0:
                for k, v in update_info.items():
                    wandb.log({f"training/{k}": v}, step=i + log_step_offset)

        if i % FLAGS.eval_interval == 0:
            eval_info = evaluate(
                agent,
                eval_env,
                num_episodes=FLAGS.eval_episodes,
                save_video=FLAGS.save_video,
            )
            for k, v in eval_info.items():
                wandb.log({f"evaluation/{k}": v}, step=i + log_step_offset)

            if FLAGS.checkpoint_model and _HAS_CHECKPOINTS:
                checkpoints.save_checkpoint(
                    chkpt_dir, agent, step=i, keep=20, overwrite=True
                )

            if FLAGS.checkpoint_buffer:
                with open(os.path.join(buffer_dir, "buffer"), "wb") as f:
                    pickle.dump(replay_buffer, f, pickle.HIGHEST_PROTOCOL)

    return agent


def main(_):
    assert 0.0 <= FLAGS.offline_ratio <= 1.0

    wandb.init(project=FLAGS.project_name)
    wandb.config.update(FLAGS)

    exp_prefix = f"s{FLAGS.seed}_{FLAGS.pretrain_steps}pretrain"
    if hasattr(FLAGS.config, "critic_layer_norm") and FLAGS.config.critic_layer_norm:
        exp_prefix += "_LN"
    log_dir = os.path.join(FLAGS.log_dir, exp_prefix)

    env, eval_env, ds = make_envs_and_dataset()
    agent = create_agent(env)

    model_dir = maybe_get_model_dir(log_dir)

    if FLAGS.mode in ("offline", "both"):
        agent = run_offline(agent, ds, eval_env)
        save_offline_agent(agent, model_dir)
        print(f"Saved offline pretrained checkpoint to: {model_dir}")

    if FLAGS.mode == "online":
        agent = load_offline_agent(agent, model_dir)
        print(f"Loaded offline pretrained checkpoint from: {model_dir}")

    if FLAGS.mode in ("online", "both"):
        log_offset = FLAGS.pretrain_steps if FLAGS.mode == "both" else 0
        run_online(agent, env, ds, eval_env, log_dir, log_step_offset=log_offset)


if __name__ == "__main__":
    app.run(main)
