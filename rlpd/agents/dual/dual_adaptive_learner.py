"""Dual-agent adaptive learner with shared min-max target backup."""

from functools import partial
from typing import Dict, Optional, Sequence, Tuple

import gym
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import struct
from flax.training.train_state import TrainState

from rlpd.agents.agent import Agent
from rlpd.agents.sac.sac_learner import decay_mask_fn
from rlpd.agents.sac.temperature import Temperature
from rlpd.data.dataset import DatasetDict
from rlpd.distributions import TanhNormal
from rlpd.networks import Ensemble, MLP, MLPResNetV2, StateActionValue, subsample_ensemble


class DualAdaptiveLearner(Agent):
    actor2: TrainState
    critic: TrainState
    critic2: TrainState
    target_critic: TrainState
    target_critic2: TrainState
    temp: TrainState
    temp2: TrainState
    tau: float
    discount: float
    target_entropy: float
    num_qs: int = struct.field(pytree_node=False)
    num_min_qs: Optional[int] = struct.field(pytree_node=False)
    backup_entropy: bool = struct.field(pytree_node=False)
    target_minmax_weight: float = struct.field(pytree_node=False)
    action_selection_temperature: float = struct.field(pytree_node=False)

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
        target_minmax_weight: float = 0.75,
        action_selection_temperature: float = 1.0,
    ):
        action_dim = action_space.shape[-1]
        observations = observation_space.sample()
        actions = action_space.sample()

        if target_entropy is None:
            target_entropy = -action_dim / 2

        rng = jax.random.PRNGKey(seed)
        rng, actor_key1, actor_key2, critic_key1, critic_key2, temp_key1, temp_key2 = (
            jax.random.split(rng, 7)
        )

        actor_base_cls = partial(
            MLP, hidden_dims=hidden_dims, activate_final=True, use_pnorm=use_pnorm
        )
        actor_def = TanhNormal(actor_base_cls, action_dim)

        actor_params1 = actor_def.init(actor_key1, observations)["params"]
        actor_params2 = actor_def.init(actor_key2, observations)["params"]

        actor1 = TrainState.create(
            apply_fn=actor_def.apply,
            params=actor_params1,
            tx=optax.adam(learning_rate=actor_lr),
        )
        actor2 = TrainState.create(
            apply_fn=actor_def.apply,
            params=actor_params2,
            tx=optax.adam(learning_rate=actor_lr),
        )

        if use_critic_resnet:
            critic_base_cls = partial(MLPResNetV2, num_blocks=1)
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
        critic_params1 = critic_def.init(critic_key1, observations, actions)["params"]
        critic_params2 = critic_def.init(critic_key2, observations, actions)["params"]

        if critic_weight_decay is not None:
            tx = optax.adamw(
                learning_rate=critic_lr,
                weight_decay=critic_weight_decay,
                mask=decay_mask_fn,
            )
        else:
            tx = optax.adam(learning_rate=critic_lr)

        critic1 = TrainState.create(apply_fn=critic_def.apply, params=critic_params1, tx=tx)
        critic2 = TrainState.create(apply_fn=critic_def.apply, params=critic_params2, tx=tx)

        target_critic_def = Ensemble(critic_cls, num=num_min_qs or num_qs)
        target_critic1 = TrainState.create(
            apply_fn=target_critic_def.apply,
            params=critic_params1,
            tx=optax.GradientTransformation(lambda _: None, lambda _: None),
        )
        target_critic2 = TrainState.create(
            apply_fn=target_critic_def.apply,
            params=critic_params2,
            tx=optax.GradientTransformation(lambda _: None, lambda _: None),
        )

        temp_def = Temperature(init_temperature)
        temp_params1 = temp_def.init(temp_key1)["params"]
        temp_params2 = temp_def.init(temp_key2)["params"]
        temp1 = TrainState.create(
            apply_fn=temp_def.apply,
            params=temp_params1,
            tx=optax.adam(learning_rate=temp_lr),
        )
        temp2 = TrainState.create(
            apply_fn=temp_def.apply,
            params=temp_params2,
            tx=optax.adam(learning_rate=temp_lr),
        )

        return cls(
            rng=rng,
            actor=actor1,
            actor2=actor2,
            critic=critic1,
            critic2=critic2,
            target_critic=target_critic1,
            target_critic2=target_critic2,
            temp=temp1,
            temp2=temp2,
            target_entropy=target_entropy,
            tau=tau,
            discount=discount,
            num_qs=num_qs,
            num_min_qs=num_min_qs,
            backup_entropy=backup_entropy,
            target_minmax_weight=target_minmax_weight,
            action_selection_temperature=action_selection_temperature,
        )

    def _sample_action_candidates(
        self, observations: jnp.ndarray, eval_mode: bool, rng: jax.random.PRNGKey
    ):
        dist1 = self.actor.apply_fn({"params": self.actor.params}, observations)
        dist2 = self.actor2.apply_fn({"params": self.actor2.params}, observations)

        if eval_mode:
            a1 = dist1.mode()
            a2 = dist2.mode()
        else:
            key1, rng = jax.random.split(rng)
            key2, rng = jax.random.split(rng)
            a1 = dist1.sample(seed=key1)
            a2 = dist2.sample(seed=key2)

        return a1, a2, rng

    def _q_for_action(self, observations: jnp.ndarray, actions: jnp.ndarray, critic: TrainState):
        qs = critic.apply_fn({"params": critic.params}, observations, actions, False)
        return qs.min(axis=0)

    def _select_action_by_q(
        self, observations: np.ndarray, eval_mode: bool
    ) -> Tuple[np.ndarray, Agent]:
        obs = jnp.asarray(observations)
        squeeze_back = False
        if obs.ndim == 1:
            obs = obs[None, :]
            squeeze_back = True

        a1, a2, rng = self._sample_action_candidates(obs, eval_mode, self.rng)
        q1 = self._q_for_action(obs, a1, self.critic)
        q2 = self._q_for_action(obs, a2, self.critic2)
        logits = jnp.stack([q1, q2], axis=-1) * self.action_selection_temperature

        if eval_mode:
            idx = jnp.argmax(logits, axis=-1)
        else:
            key, rng = jax.random.split(rng)
            idx = jax.random.categorical(key, logits=logits, axis=-1)

        idx = idx[:, None]
        actions = jnp.where(idx == 0, a1, a2)

        if squeeze_back:
            actions = actions[0]

        return np.asarray(actions), self.replace(rng=rng)

    def sample_actions(self, observations: np.ndarray) -> Tuple[np.ndarray, Agent]:
        return self._select_action_by_q(observations, eval_mode=False)

    def eval_actions(self, observations: np.ndarray) -> np.ndarray:
        actions, _ = self._select_action_by_q(observations, eval_mode=True)
        return actions

    def _next_target_q(self, batch: DatasetDict, rng: jax.random.PRNGKey):
        next_obs = batch["next_observations"]
        dist1 = self.actor.apply_fn({"params": self.actor.params}, next_obs)
        dist2 = self.actor2.apply_fn({"params": self.actor2.params}, next_obs)

        key1, rng = jax.random.split(rng)
        key2, rng = jax.random.split(rng)
        next_actions1 = dist1.sample(seed=key1)
        next_actions2 = dist2.sample(seed=key2)

        key, rng = jax.random.split(rng)
        target_params1 = subsample_ensemble(
            key, self.target_critic.params, self.num_min_qs, self.num_qs
        )
        key, rng = jax.random.split(rng)
        target_params2 = subsample_ensemble(
            key, self.target_critic2.params, self.num_min_qs, self.num_qs
        )

        key, rng = jax.random.split(rng)
        next_qs1 = self.target_critic.apply_fn(
            {"params": target_params1},
            next_obs,
            next_actions1,
            True,
            rngs={"dropout": key},
        )
        key, rng = jax.random.split(rng)
        next_qs2 = self.target_critic2.apply_fn(
            {"params": target_params2},
            next_obs,
            next_actions2,
            True,
            rngs={"dropout": key},
        )

        next_q1 = next_qs1.min(axis=0)
        next_q2 = next_qs2.min(axis=0)

        min_q = jnp.minimum(next_q1, next_q2)
        max_q = jnp.maximum(next_q1, next_q2)
        mixed_next_q = (
            self.target_minmax_weight * min_q + (1.0 - self.target_minmax_weight) * max_q
        )

        if self.backup_entropy:
            temp1 = self.temp.apply_fn({"params": self.temp.params})
            temp2 = self.temp2.apply_fn({"params": self.temp2.params})
            entropy_penalty = 0.5 * (
                temp1 * dist1.log_prob(next_actions1) + temp2 * dist2.log_prob(next_actions2)
            )
            mixed_next_q = mixed_next_q - entropy_penalty

        target_q = batch["rewards"] + self.discount * batch["masks"] * mixed_next_q
        return target_q, next_q1, next_q2, rng

    def _update_critic_with_batch(
        self, batch1: DatasetDict, batch2: DatasetDict
    ) -> Tuple[Agent, Dict[str, float]]:
        target_q1, next_q1_a, next_q1_b, rng = self._next_target_q(batch1, self.rng)
        target_q2, next_q2_a, next_q2_b, rng = self._next_target_q(batch2, rng)

        key1, rng = jax.random.split(rng)
        key2, rng = jax.random.split(rng)

        def critic1_loss_fn(critic_params):
            qs = self.critic.apply_fn(
                {"params": critic_params},
                batch1["observations"],
                batch1["actions"],
                True,
                rngs={"dropout": key1},
            )
            loss = ((qs - target_q1) ** 2).mean()
            return loss, {"critic1_loss": loss, "critic1_q": qs.mean()}

        def critic2_loss_fn(critic_params):
            qs = self.critic2.apply_fn(
                {"params": critic_params},
                batch2["observations"],
                batch2["actions"],
                True,
                rngs={"dropout": key2},
            )
            loss = ((qs - target_q2) ** 2).mean()
            return loss, {"critic2_loss": loss, "critic2_q": qs.mean()}

        grads1, info1 = jax.grad(critic1_loss_fn, has_aux=True)(self.critic.params)
        grads2, info2 = jax.grad(critic2_loss_fn, has_aux=True)(self.critic2.params)

        critic1 = self.critic.apply_gradients(grads=grads1)
        critic2 = self.critic2.apply_gradients(grads=grads2)

        target_critic1_params = optax.incremental_update(
            critic1.params, self.target_critic.params, self.tau
        )
        target_critic2_params = optax.incremental_update(
            critic2.params, self.target_critic2.params, self.tau
        )

        target_critic1 = self.target_critic.replace(params=target_critic1_params)
        target_critic2 = self.target_critic2.replace(params=target_critic2_params)

        info = {
            **info1,
            **info2,
            "target_q1": target_q1.mean(),
            "target_q2": target_q2.mean(),
            "target_q1_agent1": next_q1_a.mean(),
            "target_q1_agent2": next_q1_b.mean(),
            "target_q2_agent1": next_q2_a.mean(),
            "target_q2_agent2": next_q2_b.mean(),
        }

        return (
            self.replace(
                critic=critic1,
                critic2=critic2,
                target_critic=target_critic1,
                target_critic2=target_critic2,
                rng=rng,
            ),
            info,
        )

    def _update_actor_with_batch(
        self, batch1: DatasetDict, batch2: DatasetDict
    ) -> Tuple[Agent, Dict[str, float]]:
        key1, rng = jax.random.split(self.rng)
        key2, rng = jax.random.split(rng)
        key3, rng = jax.random.split(rng)
        key4, rng = jax.random.split(rng)

        def actor1_loss_fn(actor_params):
            dist = self.actor.apply_fn({"params": actor_params}, batch1["observations"])
            actions = dist.sample(seed=key1)
            log_probs = dist.log_prob(actions)
            qs = self.critic.apply_fn(
                {"params": self.critic.params},
                batch1["observations"],
                actions,
                True,
                rngs={"dropout": key2},
            )
            q = qs.mean(axis=0)
            temperature = self.temp.apply_fn({"params": self.temp.params})
            actor_loss = (log_probs * temperature - q).mean()
            return actor_loss, {"actor1_loss": actor_loss, "entropy1": -log_probs.mean()}

        def actor2_loss_fn(actor_params):
            dist = self.actor2.apply_fn({"params": actor_params}, batch2["observations"])
            actions = dist.sample(seed=key3)
            log_probs = dist.log_prob(actions)
            qs = self.critic2.apply_fn(
                {"params": self.critic2.params},
                batch2["observations"],
                actions,
                True,
                rngs={"dropout": key4},
            )
            q = qs.mean(axis=0)
            temperature = self.temp2.apply_fn({"params": self.temp2.params})
            actor_loss = (log_probs * temperature - q).mean()
            return actor_loss, {"actor2_loss": actor_loss, "entropy2": -log_probs.mean()}

        grads1, info1 = jax.grad(actor1_loss_fn, has_aux=True)(self.actor.params)
        grads2, info2 = jax.grad(actor2_loss_fn, has_aux=True)(self.actor2.params)

        actor1 = self.actor.apply_gradients(grads=grads1)
        actor2 = self.actor2.apply_gradients(grads=grads2)

        return self.replace(actor=actor1, actor2=actor2, rng=rng), {**info1, **info2}

    def _update_temperature(self, entropy1: float, entropy2: float):
        def temp_loss_fn(temp_params, entropy):
            temperature = self.temp.apply_fn({"params": temp_params})
            loss = temperature * (entropy - self.target_entropy).mean()
            return loss, {"temperature": temperature, "temperature_loss": loss}

        grads1, info1 = jax.grad(temp_loss_fn, has_aux=True)(self.temp.params, entropy1)
        temp1 = self.temp.apply_gradients(grads=grads1)

        def temp2_loss_fn(temp_params, entropy):
            temperature = self.temp2.apply_fn({"params": temp_params})
            loss = temperature * (entropy - self.target_entropy).mean()
            return loss, {"temperature2": temperature, "temperature2_loss": loss}

        grads2, info2 = jax.grad(temp2_loss_fn, has_aux=True)(self.temp2.params, entropy2)
        temp2 = self.temp2.apply_gradients(grads=grads2)

        return self.replace(temp=temp1, temp2=temp2), {**info1, **info2}

    @partial(jax.jit, static_argnames="utd_ratio")
    def update(self, batch: DatasetDict, utd_ratio: int):
        new_agent = self
        for i in range(utd_ratio):

            def slice_fn(x):
                assert x.shape[0] % utd_ratio == 0
                batch_size = x.shape[0] // utd_ratio
                return x[batch_size * i : batch_size * (i + 1)]

            mini_batch = jax.tree_util.tree_map(slice_fn, batch)
            new_agent, critic_info = new_agent._update_critic_with_batch(mini_batch, mini_batch)

        new_agent, actor_info = new_agent._update_actor_with_batch(mini_batch, mini_batch)
        new_agent, temp_info = new_agent._update_temperature(
            actor_info["entropy1"], actor_info["entropy2"]
        )

        return new_agent, {**critic_info, **actor_info, **temp_info}

    @partial(jax.jit, static_argnames="utd_ratio")
    def update_offline(self, batch1: DatasetDict, batch2: DatasetDict, utd_ratio: int):
        """Offline update for two dataset views (masked split)."""

        new_agent = self
        for i in range(utd_ratio):

            def slice_fn(x):
                assert x.shape[0] % utd_ratio == 0
                mini_size = x.shape[0] // utd_ratio
                return x[mini_size * i : mini_size * (i + 1)]

            mini_batch1 = jax.tree_util.tree_map(slice_fn, batch1)
            mini_batch2 = jax.tree_util.tree_map(slice_fn, batch2)
            new_agent, critic_info = new_agent._update_critic_with_batch(
                mini_batch1, mini_batch2
            )

        new_agent, actor_info = new_agent._update_actor_with_batch(mini_batch1, mini_batch2)
        new_agent, temp_info = new_agent._update_temperature(
            actor_info["entropy1"], actor_info["entropy2"]
        )

        return new_agent, {**critic_info, **actor_info, **temp_info}
