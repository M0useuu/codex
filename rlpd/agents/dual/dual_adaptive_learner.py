"""Dual-agent adaptive learner with uncertainty-aware target mixing."""

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
    action_selection_temperature: float = struct.field(pytree_node=False)
    ensemble_ratio: float = struct.field(pytree_node=False)
    actor_bc_coef: float = struct.field(pytree_node=False)

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
        action_selection_temperature: float = 1.0,
        ensemble_ratio: float = 0.5,
        actor_bc_coef: float = 0.5,
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

        actor1 = TrainState.create(
            apply_fn=actor_def.apply,
            params=actor_def.init(actor_key1, observations)["params"],
            tx=optax.adam(learning_rate=actor_lr),
        )
        actor2 = TrainState.create(
            apply_fn=actor_def.apply,
            params=actor_def.init(actor_key2, observations)["params"],
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
        temp1 = TrainState.create(
            apply_fn=temp_def.apply,
            params=temp_def.init(temp_key1)["params"],
            tx=optax.adam(learning_rate=temp_lr),
        )
        temp2 = TrainState.create(
            apply_fn=temp_def.apply,
            params=temp_def.init(temp_key2)["params"],
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
            action_selection_temperature=action_selection_temperature,
            ensemble_ratio=ensemble_ratio,
            actor_bc_coef=actor_bc_coef,
        )

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

        dist1 = self.actor.apply_fn({"params": self.actor.params}, obs)
        dist2 = self.actor2.apply_fn({"params": self.actor2.params}, obs)

        rng = self.rng
        if eval_mode:
            a1 = dist1.mode()
            a2 = dist2.mode()
        else:
            key1, rng = jax.random.split(rng)
            key2, rng = jax.random.split(rng)
            a1 = dist1.sample(seed=key1)
            a2 = dist2.sample(seed=key2)

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


    @jax.jit
    def _eval_actions_jit(self, obs: jnp.ndarray) -> jnp.ndarray:
        dist1 = self.actor.apply_fn({"params": self.actor.params}, obs)
        dist2 = self.actor2.apply_fn({"params": self.actor2.params}, obs)

        a1 = dist1.mode()
        a2 = dist2.mode()

        q1 = self._q_for_action(obs, a1, self.critic)
        q2 = self._q_for_action(obs, a2, self.critic2)
        idx = jnp.argmax(jnp.stack([q1, q2], axis=-1) * self.action_selection_temperature, axis=-1)
        return jnp.where(idx[:, None] == 0, a1, a2)

    def sample_actions(self, observations: np.ndarray) -> Tuple[np.ndarray, Agent]:
        return self._select_action_by_q(observations, eval_mode=False)

    def eval_actions(self, observations: np.ndarray) -> np.ndarray:
        obs = jnp.asarray(observations)
        squeeze_back = False
        if obs.ndim == 1:
            obs = obs[None, :]
            squeeze_back = True

        actions = self._eval_actions_jit(obs)
        if squeeze_back:
            actions = actions[0]
        return np.asarray(actions)

    def _compute_adaptive_targets(self, batch: DatasetDict, rng: jax.random.PRNGKey):
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
        q1_ens = self.target_critic.apply_fn(
            {"params": target_params1},
            next_obs,
            next_actions1,
            True,
            rngs={"dropout": key},
        )
        key, rng = jax.random.split(rng)
        q2_ens = self.target_critic2.apply_fn(
            {"params": target_params2},
            next_obs,
            next_actions2,
            True,
            rngs={"dropout": key},
        )

        q1_min = q1_ens.min(axis=0)
        q2_min = q2_ens.min(axis=0)

        if q1_ens.shape[0] >= 2:
            unc1 = jnp.abs(q1_ens[0] - q1_ens[1])
            unc2 = jnp.abs(q2_ens[0] - q2_ens[1])
        else:
            unc1 = jnp.zeros_like(q1_min)
            unc2 = jnp.zeros_like(q2_min)
        unc = 0.5 * (unc1 + unc2)

        eps = 1e-6
        gap1 = jax.nn.relu(q2_min - q1_min)
        ratio1 = self.ensemble_ratio * gap1 / (gap1 + unc + eps)
        mixed1 = q1_min + ratio1 * gap1

        gap2 = jax.nn.relu(q1_min - q2_min)
        ratio2 = self.ensemble_ratio * gap2 / (gap2 + unc + eps)
        mixed2 = q2_min + ratio2 * gap2

        if self.backup_entropy:
            temp1 = self.temp.apply_fn({"params": self.temp.params})
            temp2 = self.temp2.apply_fn({"params": self.temp2.params})
            mixed1 = mixed1 - temp1 * dist1.log_prob(next_actions1)
            mixed2 = mixed2 - temp2 * dist2.log_prob(next_actions2)

        target1 = batch["rewards"] + self.discount * batch["masks"] * mixed1
        target2 = batch["rewards"] + self.discount * batch["masks"] * mixed2

        stats = {
            "uncertainty": unc.mean(),
            "gap1": gap1.mean(),
            "gap2": gap2.mean(),
            "ratio1": ratio1.mean(),
            "ratio2": ratio2.mean(),
            "target_q1": mixed1.mean(),
            "target_q2": mixed2.mean(),
        }
        return target1, target2, stats, rng

    def _update_critic(self, batch: DatasetDict):
        target1, target2, target_stats, rng = self._compute_adaptive_targets(batch, self.rng)

        key1, rng = jax.random.split(rng)
        key2, rng = jax.random.split(rng)

        def critic1_loss_fn(critic_params):
            qs = self.critic.apply_fn(
                {"params": critic_params},
                batch["observations"],
                batch["actions"],
                True,
                rngs={"dropout": key1},
            )
            loss = ((qs - target1) ** 2).mean()
            return loss, {"critic1_loss": loss, "critic1_q": qs.mean()}

        def critic2_loss_fn(critic_params):
            qs = self.critic2.apply_fn(
                {"params": critic_params},
                batch["observations"],
                batch["actions"],
                True,
                rngs={"dropout": key2},
            )
            loss = ((qs - target2) ** 2).mean()
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

        new_agent = self.replace(
            critic=critic1,
            critic2=critic2,
            target_critic=self.target_critic.replace(params=target_critic1_params),
            target_critic2=self.target_critic2.replace(params=target_critic2_params),
            rng=rng,
        )
        return new_agent, {**info1, **info2, **target_stats}

    def _update_actor(self, batch: DatasetDict):
        key1, rng = jax.random.split(self.rng)
        key2, rng = jax.random.split(rng)
        key3, rng = jax.random.split(rng)
        key4, rng = jax.random.split(rng)

        def actor1_loss_fn(actor_params):
            dist = self.actor.apply_fn({"params": actor_params}, batch["observations"])
            actions = dist.sample(seed=key1)
            log_probs = dist.log_prob(actions)
            q = self.critic.apply_fn(
                {"params": self.critic.params},
                batch["observations"],
                actions,
                True,
                rngs={"dropout": key2},
            ).mean(axis=0)
            temp = self.temp.apply_fn({"params": self.temp.params})
            sac_loss = (log_probs * temp - q).mean()
            bc_loss = -dist.log_prob(batch["actions"]).mean()
            actor_loss = sac_loss + self.actor_bc_coef * bc_loss
            return actor_loss, {
                "actor1_loss": actor_loss,
                "actor1_sac_loss": sac_loss,
                "actor1_bc_loss": bc_loss,
                "entropy1": -log_probs.mean(),
            }

        def actor2_loss_fn(actor_params):
            dist = self.actor2.apply_fn({"params": actor_params}, batch["observations"])
            actions = dist.sample(seed=key3)
            log_probs = dist.log_prob(actions)
            q = self.critic2.apply_fn(
                {"params": self.critic2.params},
                batch["observations"],
                actions,
                True,
                rngs={"dropout": key4},
            ).mean(axis=0)
            temp = self.temp2.apply_fn({"params": self.temp2.params})
            sac_loss = (log_probs * temp - q).mean()
            bc_loss = -dist.log_prob(batch["actions"]).mean()
            actor_loss = sac_loss + self.actor_bc_coef * bc_loss
            return actor_loss, {
                "actor2_loss": actor_loss,
                "actor2_sac_loss": sac_loss,
                "actor2_bc_loss": bc_loss,
                "entropy2": -log_probs.mean(),
            }

        grads1, info1 = jax.grad(actor1_loss_fn, has_aux=True)(self.actor.params)
        grads2, info2 = jax.grad(actor2_loss_fn, has_aux=True)(self.actor2.params)

        return (
            self.replace(
                actor=self.actor.apply_gradients(grads=grads1),
                actor2=self.actor2.apply_gradients(grads=grads2),
                rng=rng,
            ),
            {**info1, **info2},
        )

    def _update_temperature(self, entropy1: float, entropy2: float):
        def t1_loss_fn(temp_params):
            t = self.temp.apply_fn({"params": temp_params})
            loss = t * (entropy1 - self.target_entropy).mean()
            return loss, {"temperature": t, "temperature_loss": loss}

        def t2_loss_fn(temp_params):
            t = self.temp2.apply_fn({"params": temp_params})
            loss = t * (entropy2 - self.target_entropy).mean()
            return loss, {"temperature2": t, "temperature2_loss": loss}

        g1, i1 = jax.grad(t1_loss_fn, has_aux=True)(self.temp.params)
        g2, i2 = jax.grad(t2_loss_fn, has_aux=True)(self.temp2.params)

        return (
            self.replace(
                temp=self.temp.apply_gradients(grads=g1),
                temp2=self.temp2.apply_gradients(grads=g2),
            ),
            {**i1, **i2},
        )

    @partial(jax.jit, static_argnames="utd_ratio")
    def update(self, batch: DatasetDict, utd_ratio: int):
        bs = batch["observations"].shape[0] // utd_ratio

        def make_mini_batch(i):
            return jax.tree_util.tree_map(
                lambda x: jax.lax.dynamic_slice_in_dim(x, i * bs, bs, axis=0),
                batch,
            )

        def critic_scan_fn(carry, i):
            agent = carry
            mini_batch = make_mini_batch(i)
            agent, critic_info = agent._update_critic(mini_batch)
            return agent, critic_info

        new_agent, critic_infos = jax.lax.scan(critic_scan_fn, self, jnp.arange(utd_ratio))
        critic_info = jax.tree_util.tree_map(lambda x: x[-1], critic_infos)

        mini_batch = make_mini_batch(utd_ratio - 1)

        new_agent, actor_info = new_agent._update_actor(mini_batch)
        new_agent, temp_info = new_agent._update_temperature(
            actor_info["entropy1"], actor_info["entropy2"]
        )

        return new_agent, {**critic_info, **actor_info, **temp_info}

    @partial(jax.jit, static_argnames="utd_ratio")
    def update_offline(self, batch1: DatasetDict, batch2: DatasetDict, utd_ratio: int):
        # Kept for API compatibility; online path uses update().
        merged_batch = {}
        for k in batch1.keys():
            merged_batch[k] = jnp.concatenate([batch1[k], batch2[k]], axis=0)
        return self.update(merged_batch, utd_ratio)
