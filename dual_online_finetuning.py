#! /usr/bin/env python
import os
import pickle
import time

import d4rl
import d4rl.gym_mujoco
import d4rl.locomotion
import gym
import tqdm
import wandb
from absl import app, flags
from flax.training import checkpoints
from ml_collections import config_flags

from utils import combine, d4rl_normalize_return, prefixed
from rlpd.agents import DualAdaptiveLearner
from rlpd.data import ReplayBuffer
from rlpd.data.binary_datasets import BinaryDataset
from rlpd.data.d4rl_datasets import D4RLDataset
from rlpd.evaluation import evaluate
from rlpd.wrappers import wrap_gym

# DEBUG_NAN_GUARD_START
import numpy as np


def _check_finite(name, value, step):
    """Temporary debug guard to locate NaN/Inf sources during rollout."""
    arr = np.asarray(value)
    if not np.all(np.isfinite(arr)):
        raise FloatingPointError(
            "[DEBUG_NAN_GUARD] Non-finite value detected: "
            f"name={name}, step={step}, shape={arr.shape}, "
            f"min={np.nanmin(arr)}, max={np.nanmax(arr)}"
        )


# DEBUG_NAN_GUARD_END

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
    "eval_at_step0",
    False,
    "Whether to run evaluation at step 0. Disabling avoids long startup stalls.",
)
flags.DEFINE_boolean(
    "precompile_update",
    False,
    "Compile the jitted update function before online interaction. May take a long time on CPU.",
)
flags.DEFINE_integer(
    "progress_interval",
    1000,
    "Write lightweight progress heartbeat every N steps.",
)
flags.DEFINE_string(
    "wandb_mode",
    "online",
    "Weights & Biases mode: online/offline/disabled.",
)
flags.DEFINE_boolean(
    "binary_include_bc", True, "Whether to include BC data in the binary datasets."
)
flags.DEFINE_string(
    "offline_checkpoint_dir",
    None,
    "Absolute directory of offline CQL weights. Online finetuning loads agent1 and agent2 from this dir.",
)
flags.DEFINE_boolean(
    "debug_nan_guard",
    True,
    "Temporary debug guard: checks observation/action/reward finite values to locate Mujoco NaNs.",
)
flags.DEFINE_integer(
    "debug_nan_guard_log_interval",
    1000,
    "Temporary debug guard: print finite-range heartbeat every N steps (<=0 to disable).",
)

config_flags.DEFINE_config_file(
    "config",
    "configs/dual_adaptive_config.py",
    "File path to the training hyperparameter configuration.",
    lock_config=False,
)


def _load_pretrained_dual_agent(agent, checkpoint_dir):
    agent1_dir = os.path.join(checkpoint_dir, "agent1")
    agent2_dir = os.path.join(checkpoint_dir, "agent2")

    if (
        checkpoints.latest_checkpoint(agent1_dir) is None
        or checkpoints.latest_checkpoint(agent2_dir) is None
    ):
        raise FileNotFoundError(
            "No checkpoint found. Expected both "
            "'<offline_checkpoint_dir>/agent1' and '<offline_checkpoint_dir>/agent2'."
        )

    agent1_state = checkpoints.restore_checkpoint(
        agent1_dir,
        target={
            "actor": agent.actor,
            "critic": agent.critic,
            "target_critic": agent.target_critic,
            "temp": agent.temp,
        },
    )
    agent2_state = checkpoints.restore_checkpoint(
        agent2_dir,
        target={
            "actor": agent.actor2,
            "critic": agent.critic2,
            "target_critic": agent.target_critic2,
            "temp": agent.temp2,
        },
    )

    return agent.replace(
        actor=agent1_state["actor"],
        critic=agent1_state["critic"],
        target_critic=agent1_state["target_critic"],
        temp=agent1_state["temp"],
        actor2=agent2_state["actor"],
        critic2=agent2_state["critic"],
        target_critic2=agent2_state["target_critic"],
        temp2=agent2_state["temp"],
    )


def main(_):
    assert 0.0 <= FLAGS.offline_ratio <= 1.0
    if FLAGS.offline_checkpoint_dir is None:
        raise ValueError("--offline_checkpoint_dir must be provided for online finetuning.")
    if not os.path.isabs(FLAGS.offline_checkpoint_dir):
        raise ValueError(
            "--offline_checkpoint_dir must be an absolute path, "
            f"got: {FLAGS.offline_checkpoint_dir}"
        )

    wandb.init(project=FLAGS.project_name, mode=FLAGS.wandb_mode)
    wandb.config.update(FLAGS)

    exp_prefix = f"s{FLAGS.seed}_dual_online"
    log_dir = os.path.abspath(os.path.join(FLAGS.log_dir, exp_prefix))

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

    agent = _load_pretrained_dual_agent(agent, FLAGS.offline_checkpoint_dir)

    if FLAGS.precompile_update:
        compile_batch = ds.sample(int(FLAGS.batch_size * FLAGS.online_utd_ratio))
        agent.update.lower(compile_batch, FLAGS.online_utd_ratio).compile()

    replay_buffer = ReplayBuffer(
        env.observation_space, env.action_space, FLAGS.max_steps
    )
    replay_buffer.seed(FLAGS.seed)

    latest_episode_metrics = None
    observation, done = env.reset(), False
    if FLAGS.debug_nan_guard:
        # DEBUG_NAN_GUARD
        _check_finite("reset_observation", observation, step=0)

    loop_start_time = time.time()
    last_progress_time = loop_start_time
    last_progress_step = 0
    for i in tqdm.tqdm(range(FLAGS.max_steps + 1), smoothing=0.1, disable=not FLAGS.tqdm):
        if i < FLAGS.start_training:
            action = env.action_space.sample()
        else:
            action, agent = agent.sample_actions(observation)

        if FLAGS.debug_nan_guard:
            # DEBUG_NAN_GUARD
            _check_finite("observation_before_step", observation, step=i)
            _check_finite("action_before_step", action, step=i)
            if (
                FLAGS.debug_nan_guard_log_interval > 0
                and i > 0
                and i % FLAGS.debug_nan_guard_log_interval == 0
            ):
                obs_arr = np.asarray(observation)
                act_arr = np.asarray(action)
                tqdm.tqdm.write(
                    "[DEBUG_NAN_GUARD] "
                    f"step={i} obs[min,max]=({np.min(obs_arr):.4e},{np.max(obs_arr):.4e}) "
                    f"act[min,max]=({np.min(act_arr):.4e},{np.max(act_arr):.4e})"
                )

        next_observation, reward, done, info = env.step(action)
        if FLAGS.debug_nan_guard:
            # DEBUG_NAN_GUARD
            _check_finite("next_observation_after_step", next_observation, step=i)
            _check_finite("reward_after_step", reward, step=i)

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

        if FLAGS.progress_interval > 0 and i > 0 and i % FLAGS.progress_interval == 0:
            now = time.time()
            dt = max(now - last_progress_time, 1e-6)
            delta_steps = i - last_progress_step
            sps = delta_steps / dt
            phase = "warmup" if i < FLAGS.start_training else "training"
            tqdm.tqdm.write(
                f"[progress] step={i} phase={phase} sps={sps:.2f} elapsed={now - loop_start_time:.1f}s"
            )
            last_progress_time = now
            last_progress_step = i

        should_eval = i % FLAGS.eval_interval == 0 and (FLAGS.eval_at_step0 or i > 0)
        if should_eval:
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
