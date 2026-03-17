"""Conservative Q-Learning learner for continuous control."""

from functools import partial
from typing import Dict, Optional, Sequence, Tuple

import flax
import gym
import jax
import jax.numpy as jnp
import optax
from flax import struct
from flax.training.train_state import TrainState

from rlpd.agents.agent import Agent
from rlpd.agents.sac.sac_learner import decay_mask_fn
from rlpd.agents.sac.temperature import Temperature
from rlpd.data.dataset import DatasetDict
from rlpd.distributions import TanhNormal
from rlpd.networks import (
    MLP,
    Ensemble,
    MLPResNetV2,
    StateActionValue,
    subsample_ensemble,
)


class CQLLearner(Agent):
    critic: TrainState
    target_critic: TrainState
    temp: TrainState
    cql_alpha_lagrange: Optional[TrainState]
    tau: float
    discount: float
    target_entropy: float
    num_qs: int = struct.field(pytree_node=False)
    num_min_qs: Optional[int] = struct.field(pytree_node=False)
    backup_entropy: bool = struct.field(pytree_node=False)
    cql_n_actions: int = struct.field(pytree_node=False)
    cql_importance_sample: bool = struct.field(pytree_node=False)
    cql_temp: float = struct.field(pytree_node=False)
    cql_alpha: float = struct.field(pytree_node=False)
    cql_max_target_backup: bool = struct.field(pytree_node=False)
    cql_lagrange: bool = struct.field(pytree_node=False)
    cql_target_action_gap: float = struct.field(pytree_node=False)
    cql_clip_diff_min: float = struct.field(pytree_node=False)
    cql_clip_diff_max: float = struct.field(pytree_node=False)

    @classmethod
    def create(
        cls,
        seed: int,
        observation_space: gym.Space,
        action_space: gym.Space,
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        temp_lr: float = 3e-4,
        hidden_dims: Sequence[int] = (256, 256),
        discount: float = 0.99,
        tau: float = 0.005,
        num_qs: int = 2,
        num_min_qs: Optional[int] = None,
        critic_dropout_rate: Optional[float] = None,
        critic_weight_decay: Optional[float] = None,
        critic_layer_norm: bool = False,
        target_entropy: Optional[float] = None,
        init_temperature: float = 1.0,
        backup_entropy: bool = True,
        use_pnorm: bool = False,
        use_critic_resnet: bool = False,
        cql_n_actions: int = 10,
        cql_importance_sample: bool = True,
        cql_temp: float = 1.0,
        cql_alpha: float = 5.0,
        cql_max_target_backup: bool = False,
        cql_lagrange: bool = False,
        cql_target_action_gap: float = 10.0,
        cql_alpha_lr: float = 3e-4,
        cql_clip_diff_min: float = -jnp.inf,
        cql_clip_diff_max: float = jnp.inf,
    ):
        action_dim = action_space.shape[-1]
        observations = observation_space.sample()
        actions = action_space.sample()

        if target_entropy is None:
            target_entropy = -action_dim

        rng = jax.random.PRNGKey(seed)
        rng, actor_key, critic_key, temp_key, cql_alpha_key = jax.random.split(rng, 5)

        actor_base_cls = partial(
            MLP, hidden_dims=hidden_dims, activate_final=True, use_pnorm=use_pnorm
        )
        actor_def = TanhNormal(actor_base_cls, action_dim)
        actor_params = actor_def.init(actor_key, observations)["params"]
        actor = TrainState.create(
            apply_fn=actor_def.apply,
            params=actor_params,
            tx=optax.adam(learning_rate=actor_lr),
        )

        if use_critic_resnet:
            critic_base_cls = partial(
                MLPResNetV2,
                num_blocks=1,
            )
        else:
            critic_base_cls = partial(
                MLP,
                hidden_dims=hidden_dims,
                activate_final=True,
                dropout_rate=critic_dropout_rate,
                use_layer_norm=critic_layer_norm,
                use_pnorm=use_pnorm,
            )
        critic_cls = partial(StateActionValue, base_cls=critic_base_cls)
        critic_def = Ensemble(critic_cls, num=num_qs)
        critic_params = critic_def.init(critic_key, observations, actions)["params"]

        if critic_weight_decay is not None:
            tx = optax.adamw(
                learning_rate=critic_lr,
                weight_decay=critic_weight_decay,
                mask=decay_mask_fn,
            )
        else:
            tx = optax.adam(learning_rate=critic_lr)

        critic = TrainState.create(apply_fn=critic_def.apply, params=critic_params, tx=tx)
        target_critic_def = Ensemble(critic_cls, num=num_min_qs or num_qs)
        target_critic = TrainState.create(
            apply_fn=target_critic_def.apply,
            params=critic_params,
            tx=optax.GradientTransformation(lambda _: None, lambda _: None),
        )

        temp_def = Temperature(init_temperature)
        temp_params = temp_def.init(temp_key)["params"]
        temp = TrainState.create(
            apply_fn=temp_def.apply,
            params=temp_params,
            tx=optax.adam(learning_rate=temp_lr),
        )

        cql_alpha_lagrange = None
        if cql_lagrange:
            cql_alpha_def = Temperature(1.0)
            cql_alpha_params = cql_alpha_def.init(cql_alpha_key)["params"]
            cql_alpha_lagrange = TrainState.create(
                apply_fn=cql_alpha_def.apply,
                params=cql_alpha_params,
                tx=optax.adam(learning_rate=cql_alpha_lr),
            )

        return cls(
            rng=rng,
            actor=actor,
            critic=critic,
            target_critic=target_critic,
            temp=temp,
            cql_alpha_lagrange=cql_alpha_lagrange,
            target_entropy=target_entropy,
            tau=tau,
            discount=discount,
            num_qs=num_qs,
            num_min_qs=num_min_qs,
            backup_entropy=backup_entropy,
            cql_n_actions=cql_n_actions,
            cql_importance_sample=cql_importance_sample,
            cql_temp=cql_temp,
            cql_alpha=cql_alpha,
            cql_max_target_backup=cql_max_target_backup,
            cql_lagrange=cql_lagrange,
            cql_target_action_gap=cql_target_action_gap,
            cql_clip_diff_min=cql_clip_diff_min,
            cql_clip_diff_max=cql_clip_diff_max,
        )

    def update_actor(self, batch: DatasetDict) -> Tuple[Agent, Dict[str, float]]:
        key, rng = jax.random.split(self.rng)
        key2, rng = jax.random.split(rng)

        def actor_loss_fn(actor_params) -> Tuple[jnp.ndarray, Dict[str, float]]:
            dist = self.actor.apply_fn({"params": actor_params}, batch["observations"])
            actions = dist.sample(seed=key)
            log_probs = dist.log_prob(actions)
            qs = self.critic.apply_fn(
                {"params": self.critic.params},
                batch["observations"],
                actions,
                True,
                rngs={"dropout": key2},
            )
            q = qs.mean(axis=0)
            actor_loss = (
                log_probs * self.temp.apply_fn({"params": self.temp.params}) - q
            ).mean()
            return actor_loss, {"actor_loss": actor_loss, "entropy": -log_probs.mean()}

        grads, actor_info = jax.grad(actor_loss_fn, has_aux=True)(self.actor.params)
        actor = self.actor.apply_gradients(grads=grads)
        return self.replace(actor=actor, rng=rng), actor_info

    def update_temperature(self, entropy: float) -> Tuple[Agent, Dict[str, float]]:
        def temperature_loss_fn(temp_params):
            temperature = self.temp.apply_fn({"params": temp_params})
            temp_loss = temperature * (entropy - self.target_entropy).mean()
            return temp_loss, {
                "temperature": temperature,
                "temperature_loss": temp_loss,
            }

        grads, temp_info = jax.grad(temperature_loss_fn, has_aux=True)(self.temp.params)
        temp = self.temp.apply_gradients(grads=grads)
        return self.replace(temp=temp), temp_info

    def _sample_n_actions(
        self, observations: jnp.ndarray, rng: jax.random.PRNGKey
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jax.random.PRNGKey]:
        batch_size = observations.shape[0]
        expanded_obs = jnp.broadcast_to(
            observations[:, None, :],
            (batch_size, self.cql_n_actions, observations.shape[-1]),
        ).reshape(batch_size * self.cql_n_actions, observations.shape[-1])
        key, rng = jax.random.split(rng)
        dist = self.actor.apply_fn({"params": self.actor.params}, expanded_obs)
        actions = dist.sample(seed=key)
        log_probs = dist.log_prob(actions)
        actions = actions.reshape(batch_size, self.cql_n_actions, -1)
        log_probs = log_probs.reshape(batch_size, self.cql_n_actions)
        return actions, log_probs, rng

    def update_critic(self, batch: DatasetDict) -> Tuple[TrainState, Dict[str, float]]:
        rng = self.rng

        dist = self.actor.apply_fn(
            {"params": self.actor.params}, batch["next_observations"]
        )

        key, rng = jax.random.split(rng)

        if self.cql_max_target_backup:
            next_actions, next_log_probs, rng = self._sample_n_actions(
                batch["next_observations"], rng
            )
            batch_size = batch["next_observations"].shape[0]
            tiled_next_obs = jnp.repeat(
                batch["next_observations"], repeats=self.cql_n_actions, axis=0
            )
            next_actions_flat = next_actions.reshape(batch_size * self.cql_n_actions, -1)

            key, rng = jax.random.split(rng)
            target_params = subsample_ensemble(
                key, self.target_critic.params, self.num_min_qs, self.num_qs
            )
            key, rng = jax.random.split(rng)
            next_qs = self.target_critic.apply_fn(
                {"params": target_params},
                tiled_next_obs,
                next_actions_flat,
                True,
                rngs={"dropout": key},
            )
            next_qs = next_qs.reshape(self.num_min_qs or self.num_qs, batch_size, self.cql_n_actions)
            max_target_idx = jnp.argmax(next_qs.min(axis=0), axis=-1)
            next_actions = next_actions[jnp.arange(batch_size), max_target_idx]
            next_log_probs = next_log_probs[jnp.arange(batch_size), max_target_idx]
        else:
            next_actions = dist.sample(seed=key)
            next_log_probs = dist.log_prob(next_actions)

        key, rng = jax.random.split(rng)
        target_params = subsample_ensemble(
            key, self.target_critic.params, self.num_min_qs, self.num_qs
        )

        key, rng = jax.random.split(rng)
        next_qs = self.target_critic.apply_fn(
            {"params": target_params},
            batch["next_observations"],
            next_actions,
            True,
            rngs={"dropout": key},
        )
        next_q = next_qs.min(axis=0)

        target_q = batch["rewards"] + self.discount * batch["masks"] * next_q

        if self.backup_entropy:
            target_q -= (
                self.discount
                * batch["masks"]
                * self.temp.apply_fn({"params": self.temp.params})
                * next_log_probs
            )

        key, rng = jax.random.split(rng)
        random_actions = jax.random.uniform(
            key,
            shape=(batch["observations"].shape[0], self.cql_n_actions, batch["actions"].shape[-1]),
            minval=-1.0,
            maxval=1.0,
        )

        cur_actions, cur_log_probs, rng = self._sample_n_actions(batch["observations"], rng)
        next_actions_cql, next_log_probs_cql, rng = self._sample_n_actions(
            batch["next_observations"], rng
        )

        random_density = jnp.log(0.5 ** random_actions.shape[-1])

        batch_size = batch["observations"].shape[0]
        tiled_obs = jnp.repeat(batch["observations"], repeats=self.cql_n_actions, axis=0)
        tiled_next_obs = jnp.repeat(batch["next_observations"], repeats=self.cql_n_actions, axis=0)

        random_actions_flat = random_actions.reshape(batch_size * self.cql_n_actions, -1)
        cur_actions_flat = cur_actions.reshape(batch_size * self.cql_n_actions, -1)
        next_actions_flat = next_actions_cql.reshape(batch_size * self.cql_n_actions, -1)

        key, rng = jax.random.split(rng)

        def critic_loss_fn(critic_params) -> Tuple[jnp.ndarray, Dict[str, float]]:
            qs = self.critic.apply_fn(
                {"params": critic_params},
                batch["observations"],
                batch["actions"],
                True,
                rngs={"dropout": key},
            )

            td_loss = ((qs - target_q) ** 2).mean()

            cql_obs = jnp.concatenate([tiled_obs, tiled_obs, tiled_next_obs], axis=0)
            cql_actions = jnp.concatenate(
                [random_actions_flat, cur_actions_flat, next_actions_flat], axis=0
            )
            cql_qs = self.critic.apply_fn(
                {"params": critic_params}, cql_obs, cql_actions, True, rngs={"dropout": key}
            )
            q_rand, q_cur, q_next = jnp.split(cql_qs, 3, axis=1)
            q_rand = q_rand.reshape(self.num_qs, batch_size, self.cql_n_actions)
            q_cur = q_cur.reshape(self.num_qs, batch_size, self.cql_n_actions)
            q_next = q_next.reshape(self.num_qs, batch_size, self.cql_n_actions)

            if self.cql_importance_sample:
                cat_q = jnp.concatenate(
                    [
                        q_rand - random_density,
                        q_cur - cur_log_probs[None, ...],
                        q_next - next_log_probs_cql[None, ...],
                    ],
                    axis=-1,
                )
            else:
                cat_q = jnp.concatenate([q_rand, q_cur, q_next], axis=-1)

            cql_q_ood = jax.scipy.special.logsumexp(cat_q / self.cql_temp, axis=-1) * self.cql_temp
            cql_diff = jnp.clip(cql_q_ood - qs, self.cql_clip_diff_min, self.cql_clip_diff_max)
            cql_loss_per_q = cql_diff.mean(axis=1)

            if self.cql_lagrange:
                alpha_prime = self.cql_alpha_lagrange.apply_fn(
                    {"params": self.cql_alpha_lagrange.params}
                )
                alpha_prime = jnp.clip(alpha_prime, 0.0, 1e6)
                cql_loss = (alpha_prime * self.cql_alpha * (cql_loss_per_q - self.cql_target_action_gap)).mean()
            else:
                alpha_prime = 1.0
                cql_loss = (self.cql_alpha * cql_loss_per_q).mean()

            critic_loss = td_loss + cql_loss

            info = {
                "critic_loss": critic_loss,
                "td_loss": td_loss,
                "cql_loss": cql_loss,
                "cql_q_diff": cql_loss_per_q.mean(),
                "q": qs.mean(),
                "q_random": q_rand.mean(),
                "q_current": q_cur.mean(),
                "q_next": q_next.mean(),
                "cql_alpha_prime": alpha_prime,
            }
            return critic_loss, info

        grads, info = jax.grad(critic_loss_fn, has_aux=True)(self.critic.params)
        critic = self.critic.apply_gradients(grads=grads)

        cql_alpha_lagrange = self.cql_alpha_lagrange
        if self.cql_lagrange:
            def alpha_loss_fn(alpha_params):
                alpha_prime = cql_alpha_lagrange.apply_fn({"params": alpha_params})
                alpha_prime = jnp.clip(alpha_prime, 0.0, 1e6)
                alpha_loss = -alpha_prime * self.cql_alpha * (info["cql_q_diff"] - self.cql_target_action_gap)
                return alpha_loss.mean(), {"cql_alpha_loss": alpha_loss.mean()}

            alpha_grads, alpha_info = jax.grad(alpha_loss_fn, has_aux=True)(
                cql_alpha_lagrange.params
            )
            cql_alpha_lagrange = cql_alpha_lagrange.apply_gradients(grads=alpha_grads)
            info.update(alpha_info)

        target_critic_params = optax.incremental_update(
            critic.params, self.target_critic.params, self.tau
        )
        target_critic = self.target_critic.replace(params=target_critic_params)

        return (
            self.replace(
                critic=critic,
                target_critic=target_critic,
                cql_alpha_lagrange=cql_alpha_lagrange,
                rng=rng,
            ),
            info,
        )

    @partial(jax.jit, static_argnames="utd_ratio")
    def update(self, batch: DatasetDict, utd_ratio: int):
        batch_size = batch["observations"].shape[0] // utd_ratio

        def make_mini_batch(i):
            return jax.tree_util.tree_map(
                lambda x: jax.lax.dynamic_slice_in_dim(x, i * batch_size, batch_size, axis=0),
                batch,
            )

        def critic_scan_fn(carry, i):
            agent = carry
            mini_batch = make_mini_batch(i)
            agent, critic_info = agent.update_critic(mini_batch)
            return agent, critic_info

        new_agent, critic_infos = jax.lax.scan(critic_scan_fn, self, jnp.arange(utd_ratio))
        critic_info = jax.tree_util.tree_map(lambda x: x[-1], critic_infos)

        mini_batch = make_mini_batch(utd_ratio - 1)

        new_agent, actor_info = new_agent.update_actor(mini_batch)
        new_agent, temp_info = new_agent.update_temperature(actor_info["entropy"])

        return new_agent, {**actor_info, **critic_info, **temp_info}
