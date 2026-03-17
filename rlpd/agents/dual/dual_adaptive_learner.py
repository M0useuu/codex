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

    @staticmethod
    def add_action(actions1: jnp.ndarray, actions2: jnp.ndarray) -> jnp.ndarray:
        return 0.5 * (actions1 + actions2)

    @jax.jit
    def _sample_actions_jit(
        self, obs: jnp.ndarray, rng: jax.random.PRNGKey
    ) -> Tuple[jnp.ndarray, jax.random.PRNGKey]:
        dist1 = self.actor.apply_fn({"params": self.actor.params}, obs)
        dist2 = self.actor2.apply_fn({"params": self.actor2.params}, obs)

        key1, rng = jax.random.split(rng)
        key2, rng = jax.random.split(rng)
        a1 = dist1.sample(seed=key1)
        a2 = dist2.sample(seed=key2)

        q1 = self._q_for_action(obs, a1, self.critic)
        q2 = self._q_for_action(obs, a2, self.critic2)
        logits = jnp.stack([q1, q2], axis=-1) * self.action_selection_temperature

        key, rng = jax.random.split(rng)
        idx = jax.random.categorical(key, logits=logits, axis=-1)
        actions = jnp.where(idx[:, None] == 0, a1, a2)
        return actions, rng


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
        obs = jnp.asarray(observations)
        squeeze_back = False
        if obs.ndim == 1:
            obs = obs[None, :]
            squeeze_back = True

        actions, rng = self._sample_actions_jit(obs, self.rng)
        if squeeze_back:
            actions = actions[0]

        return np.asarray(actions), self.replace(rng=rng)

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

        q01_next = q1_ens[0]
        q02_next = q1_ens[1] if q1_ens.shape[0] > 1 else q1_ens[0]
        q11_next = q2_ens[0]
        q12_next = q2_ens[1] if q2_ens.shape[0] > 1 else q2_ens[0]

        q0_next = jnp.minimum(q01_next, q02_next)
        q1_next = jnp.minimum(q11_next, q12_next)
        unc = 0.5 * (jnp.abs(q01_next - q02_next) + jnp.abs(q11_next - q12_next))

        eps = 1e-6
        gap0 = jax.nn.relu(q1_next - q0_next)
        ratio0 = self.ensemble_ratio * gap0 / (gap0 + unc + eps)
        mixed0 = q0_next + ratio0 * gap0

        gap1 = jax.nn.relu(q0_next - q1_next)
        ratio1 = self.ensemble_ratio * gap1 / (gap1 + unc + eps)
        mixed1 = q1_next + ratio1 * gap1

        if self.backup_entropy:
            temp1 = self.temp.apply_fn({"params": self.temp.params})
            temp2 = self.temp2.apply_fn({"params": self.temp2.params})
            mixed0 = mixed0 - temp1 * dist1.log_prob(next_actions1)
            mixed1 = mixed1 - temp2 * dist2.log_prob(next_actions2)

        target0 = batch["rewards"] + self.discount * batch["masks"] * mixed0
        target1 = batch["rewards"] + self.discount * batch["masks"] * mixed1

        stats = {
            "uncertainty": unc.mean(),
            "gap0": gap0.mean(),
            "gap1": gap1.mean(),
            "ratio0": ratio0.mean(),
            "ratio1": ratio1.mean(),
            "target_q0": mixed0.mean(),
            "target_q1": mixed1.mean(),
        }
        return target0, target1, stats, rng

    def _update_critic(self, batch: DatasetDict):
        target0, target1, target_stats, rng = self._compute_adaptive_targets(batch, self.rng)

        key1, rng = jax.random.split(rng)
        key2, rng = jax.random.split(rng)

        def critic_loss_fn(critic_params0, critic_params1):
            qs0 = self.critic.apply_fn(
                {"params": critic_params0},
                batch["observations"],
                batch["actions"],
                True,
                rngs={"dropout": key1},
            )
            qs1 = self.critic2.apply_fn(
                {"params": critic_params1},
                batch["observations"],
                batch["actions"],
                True,
                rngs={"dropout": key2},
            )
            loss0 = ((qs0[0] - target0) ** 2).mean() + ((qs0[1] - target0) ** 2).mean()
            loss1 = ((qs1[0] - target1) ** 2).mean() + ((qs1[1] - target1) ** 2).mean()
            total = loss0 + loss1
            info = {
                "critic1_loss": loss0,
                "critic2_loss": loss1,
                "critic1_q": qs0.mean(),
                "critic2_q": qs1.mean(),
            }
            return total, info

        (grads0, grads1), info = jax.grad(critic_loss_fn, argnums=(0, 1), has_aux=True)(
            self.critic.params, self.critic2.params
        )

        critic1 = self.critic.apply_gradients(grads=grads0)
        critic2 = self.critic2.apply_gradients(grads=grads1)

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
        return new_agent, {**info, **target_stats}

    def _update_actor(self, batch: DatasetDict):
        key1, rng = jax.random.split(self.rng)
        key2, rng = jax.random.split(rng)
        key3, rng = jax.random.split(rng)
        key4, rng = jax.random.split(rng)

        def actor_loss_fn(actor_params0, actor_params1):
            dist0 = self.actor.apply_fn({"params": actor_params0}, batch["observations"])
            new_actions0 = dist0.sample(seed=key1)
            log_pi0 = dist0.log_prob(new_actions0)
            q0_all = self.critic.apply_fn(
                {"params": self.critic.params},
                batch["observations"],
                new_actions0,
                True,
                rngs={"dropout": key2},
            )
            q0 = jnp.minimum(q0_all[0], q0_all[1])
            alpha0 = self.temp.apply_fn({"params": self.temp.params})
            policy_loss0 = (alpha0 * log_pi0 - q0).mean()
            bc_loss0 = -dist0.log_prob(batch["actions"]).mean()
            actor_loss0 = policy_loss0 + self.actor_bc_coef * bc_loss0

            dist1 = self.actor2.apply_fn({"params": actor_params1}, batch["observations"])
            new_actions1 = dist1.sample(seed=key3)
            log_pi1 = dist1.log_prob(new_actions1)
            q1_all = self.critic2.apply_fn(
                {"params": self.critic2.params},
                batch["observations"],
                new_actions1,
                True,
                rngs={"dropout": key4},
            )
            q1 = jnp.minimum(q1_all[0], q1_all[1])
            alpha1 = self.temp2.apply_fn({"params": self.temp2.params})
            policy_loss1 = (alpha1 * log_pi1 - q1).mean()
            bc_loss1 = -dist1.log_prob(batch["actions"]).mean()
            actor_loss1 = policy_loss1 + self.actor_bc_coef * bc_loss1

            total_loss = actor_loss0 + actor_loss1
            info = {
                "actor1_loss": actor_loss0,
                "actor1_sac_loss": policy_loss0,
                "actor1_bc_loss": bc_loss0,
                "entropy1": -log_pi0.mean(),
                "actor2_loss": actor_loss1,
                "actor2_sac_loss": policy_loss1,
                "actor2_bc_loss": bc_loss1,
                "entropy2": -log_pi1.mean(),
            }
            return total_loss, info

        (grads0, grads1), info = jax.grad(actor_loss_fn, argnums=(0, 1), has_aux=True)(
            self.actor.params, self.actor2.params
        )

        return (
            self.replace(
                actor=self.actor.apply_gradients(grads=grads0),
                actor2=self.actor2.apply_gradients(grads=grads1),
                rng=rng,
            ),
            info,
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
