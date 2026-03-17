# CQL offline 训练效果差异排查（对照你给的 PyTorch CQL 参考实现）

以下结论基于本仓库现有代码路径（`cql_offline_training.py` + `configs/cql_config.py` + `rlpd/agents/sac/cql_learner.py`）。

## 1) 最容易被忽略：你是否跑错了入口脚本

- 纯离线 CQL 应该用 `cql_offline_training.py`，它只做离线更新并在末尾保存离线 checkpoint。
- `train_finetuning.py` 是「离线+在线微调」流程，会在 pretrain 后继续环境交互与混合采样，不等价于纯 offline CQL。

如果你拿纯 offline 基线去对比，但实际运行了 `train_finetuning.py`，曲线和最终分数会明显不同。

## 2) 关键超参默认值与参考实现不一致（影响最大）

### (a) Actor 学习率高了一个数量级

- 本仓库默认 `actor_lr = 3e-4`（从 `td_config.py` 继承）。
- 你贴的参考实现里 `policy_lr = 3e-5`。

这通常会让策略更新过快，离线 setting 下更容易产生不稳定和性能退化。

### (b) 网络深度不一致

- 本仓库默认 `hidden_dims=(256, 256)`，即两层 MLP。
- 你贴的参考实现 actor/critic 多处是三层 256（尤其 Q 通常三层）。

网络表达能力与训练动态都可能不同，直接造成 score gap。

### (c) 状态归一化没有默认做

- 本仓库 `D4RLDataset` 直接读取数据，没有对 state 做 mean/std normalize。
- 你贴的参考实现默认对 `observations` 和 `next_observations` 做归一化，并在 eval env 使用同样变换。

在 D4RL locomotion 上，是否归一化常常显著影响学习稳定性与收敛速度。

## 3) 训练目标与数据处理细节不同

### (a) 终止处理逻辑不同（mask 构造）

- 本仓库用 trajectory boundary 规则构造 `dones`/`masks`：不仅看 `terminal`，还看相邻状态是否连续。
- 你贴的实现只从 `terminals` 直接读 `dones`。

这会改变 bootstrapping 范围，从而影响 Q 值尺度和最终策略。

### (b) Action 预裁剪

- 本仓库在读 D4RL 数据时会把动作裁剪到 `[-1+eps, 1-eps]`。
- 参考实现通常直接使用数据动作（不一定裁剪）。

这会改变 critic 在边界动作附近的学习分布。

## 4) CQL 算法实现形态不同（JAX/Flax vs PyTorch）

即使公式名义相同，以下实现差异也会累积成显著性能差：

- 本仓库是 JAX/Flax/Optax 的函数式实现，随机数与 update 路径不同。
- critic/target 使用 ensemble 与可选子集最小化（`num_qs`/`num_min_qs` 机制）。
- update 入口支持 UTD 切片（同一大 batch 分多次 mini-update）。

这些都可能让同一组“看起来相同”的超参得到不同的有效优化轨迹。

## 5) 离线 CQL在本仓库的对齐建议（按优先级）

1. **先确认入口**：使用 `cql_offline_training.py`。  
2. **先把 actor lr 降到 `3e-5`**（最关键）。  
3. **把网络改成三层**：`--config.hidden_dims="(256,256,256)"`。  
4. **补 state normalization**（代码层加 mean/std 归一化，或在 wrapper 中做一致处理）。  
5. 保持 `cql_alpha=10, cql_n_actions=10, cql_importance_sample=True` 与参考一致后，再逐项 ablation。  
6. 固定随机种子做多 seed 对比（至少 3~5 个 seed），避免单次偶然波动。

## 6) 建议的最小对齐命令（示例）

```bash
python cql_offline_training.py \
  --env_name=halfcheetah-medium-expert-v2 \
  --pretrain_steps=1000000 \
  --batch_size=256 \
  --offline_checkpoint_dir=./ckpts/cql_hc_me \
  --config=configs/cql_config.py \
  --config.actor_lr=3e-5 \
  --config.hidden_dims="(256,256,256)"
```

如果你愿意，我下一步可以直接给你一个“与参考实现逐项对齐”的 patch（含 state normalize 开关），让你一键复现实验再做消融。
