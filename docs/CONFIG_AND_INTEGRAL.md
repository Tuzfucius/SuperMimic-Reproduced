# 配置、积分策略与实验复现

本文记录 V0.1 YAML 配置、积分状态和 Sweep 的实际行为。它描述工程实现与验证范围，不代表积分机制已经带来性能提升。

## 配置入口

配置从 `TrainConfig` 默认值开始，按以下顺序覆盖：

1. 默认 Dataclass。
2. YAML 的 `_base_` 父配置（相对当前 YAML 路径，可为多个文件）。
3. 当前 YAML。
4. 重复指定的 `--set key=value`。
5. 显式传入的旧 Tyro CLI 参数。

所有值会做类型和依赖检查。`resolved.yaml` 保存最终有效配置；`config.yaml` 保存最初指定的 YAML。`--dry-run` 会显示最终配置、策略观测/goal/积分输入维度和运行目录，不初始化仿真器。

```bash
uv run sm-train --cfg configs/experiments/baseline.yaml --dry-run
uv run sm-train --cfg configs/experiments/integral_actor.yaml \
  --set integral.tau_s=3.0 --set seed=1 --dry-run
uv run sm-train --cfg configs/experiments/integral_actor.yaml \
  --set policy.history=8 --dry-run
```

Policy 每帧观测由关节位置 29、关节速度 29、基座角速度 3、投影重力 3 和上一动作 29 组成，总计 93 维。历史长度为 `H` 时，Actor Decoder 的观测输入为 `93H`；环境 Observation History 与网络维度共用 `policy.history`。默认 `H=10` 时观测仍为 930 维，默认 Actor Decoder 首层仍为 `32 + 930 = 962` 维，积分关闭时可严格加载原版权重。`policy.history` 必须为正整数，并满足 Replay 容量约束：`env.replay_capacity > policy.history + 1` 且 `env.replay_capacity > wm_trainer.window + 1`，也就是容量至少为 `max(policy.history, wm_trainer.window) + 2`。不通过补零或截断改变观测。

紧凑动作库由 `data.library` 推导。留空 `env.compact_library` 时，解析阶段设置为 `data.library` 是否存在；显式配置与数据源冲突会报错。Trainer、Resume、Dry Run 和实验元数据使用同一解析结果。

## 积分状态

V0.1 使用策略可见的相对关节位置误差：

```text
e_t = goal_t[:29] - policy_obs_t.newest_joint_pos[29:]
z_(t+1) = clamp(exp(-dt / tau_s) * z_t + dt * e_t, -clip_abs, clip_abs)
```

`tau_s` 控制泄漏速度，`clip_abs` 限制每个关节积分值的绝对幅度；`dt` 固定为环境控制周期 0.02 s。误差两侧都使用相对默认关节姿态的约定。Actor 不读取仿真器关节状态；观测侧按配置保留偏置和噪声。

`integral.enabled` 默认关闭。启用后，`actor_input` 和 `critic_input` 分别控制是否将 29 维 `z_t` 拼入 Actor 和 Critic。`full` 将梯度沿积分递推保留；`detach` 在每步截断旧积分状态的递归梯度，但当前误差仍能接收梯度。两种模式使用相同的前向积分公式。

数据对齐规则：采集时动作 `a_t`、策略状态 `s_t` 和积分状态 `z_t` 写在同一条 Replay transition；Replay 起点恢复和该状态对应的 `z_t`；Imagination 使用同一递推在动作之后产生 `z_(t+1)`。训练在 `env.step()` 返回后提交该步误差，再按配置处理 Episode 边界；评测 Runner 在下一次策略调用前提交上一步误差，数值次序相同。`reset_on_episode=true` 时，Episode Reset 清零对应环境的积分和待提交误差。`false` 时，显式或自动 Reset 都保留积分状态及待提交误差；Runner 在下一次策略调用时恰好提交一次终止步误差，随后继续累积新 Episode 的误差。评测器对完成的环境调用 `runner.reset(env_ids)`；末帧 Clamp 在相邻调用间保持同一 frame 时不会误判为边界。Play 的手动重置也按此配置处理。

该积分输入只积累关节误差，不含全局根部平移或偏航误差，因此不能观测或保证修正全局漂移。它也不使用评测时额外可得的特权观测。

## B0 / I1 / I2 / I3

| ID | 配置 | 梯度 |
|---|---|---|
| B0 | `integral.enabled: false` | 原始策略，无积分输入 |
| I1 | 启用积分，Actor 输入 | `full` |
| I2 | I1 + Critic 输入 | `full` |
| I3 | 与 I1 相同 | `detach` |

这些名称定义实现配置，不代表消融结果或性能结论。基准配置在 `configs/experiments/`，Sweep 清单在 `configs/sweeps/integral_ablation.yaml`。

## Sweep、身份与汇总

Grid 的每个参数组合先规范化为按键排序的 JSON，再以 SHA-256 前 12 位生成稳定 `variant_id`。组合顺序变化不改变 ID；每次运行名包含实验 ID、variant ID 和 Seed，完整 `params` 也保存在行结果及汇总中。

训练身份包含最终解析配置、Seed、实验 ID、参数组合、代码 Commit、训练数据内容指纹和初始化来源。`init_from` 会记录 `mode=init_from`、标准化 checkpoint 路径及其内容 SHA-256；Scratch 则记录 `mode=scratch`。初始化权重内容改变会改变训练身份，即使路径、文件大小和时间戳不变。`run_name`、`log_root` 不改变训练身份。Sweep 不支持 Resume；含 `resume` 的 YAML 或 `--set` 会在 Dry Run/训练启动前报错，请使用 `sm-train` 独立 Resume。Git dirty 状态会记录；工作区不干净时禁止复用已有训练结果。

数据集指纹使用版本标记 `sha256-content-v1`。`.npz` 目录按相对路径排序，对每个文件记录相对路径、大小和流式内容 SHA-256，再对规范化清单计算 SHA-256；紧凑库 `.pt`、Manifest 和其他单文件输入也使用相同内容哈希。缺失或空数据路径会报错，不会生成占位指纹。哈希按 1 MiB 块读取，不把动作文件载入内存。一个 Sweep 进程会在内存中缓存已读过的文件摘要，因此不同 Seed/Variant 不会反复读取相同大文件；缓存不落盘，要求数据在同一 Sweep 期间保持不变。首次运行需要顺序读取每个独立数据文件一次，成本与数据总字节数成正比。

评测身份独立包含评测协议版本、评测数据集清单、评测参数、Checkpoint 路径与内容 SHA-256，以及当前代码 Commit/dirty 状态。训练身份相同而评测身份变化时，保留训练 Checkpoint、将旧 benchmark 归档到 `evaluation_history/`，然后重新评测。任一身份不匹配都不会静默跳过；旧元数据缺少训练身份或无法解析时标记为未验证，不猜测其配置。Benchmark 文件或评测元数据缺失也会触发补评测。

```bash
uv run sm-sweep --cfg configs/sweeps/integral_ablation.yaml --dry-run
uv run sm-sweep --cfg configs/sweeps/integral_ablation.yaml --gpu 0
```

每个运行保留原始 benchmark 与指标。`summary.json` / `summary.csv` 保存每次运行；`aggregate.json` / `aggregate.csv` 按 `experiment ID + variant_id` 分组，只统计成功且具有评测结果的 Seed。分组报告预期 Seed 数、有效 Seed 数和缺失 Seed。均值使用有效 Seed 的等权均值，标准差使用样本标准差；只有一个有效 Seed 时标准差为 `null`，不会伪报为 0。

## Checkpoint、Resume 与 Init From

```bash
# 从指定配置开始训练
uv run sm-train --cfg configs/experiments/baseline.yaml

# 在原实验中 Resume；语义配置必须相同
uv run sm-train --cfg configs/experiments/integral_actor.yaml \
  --set resume=logs/I1/checkpoints/ckpt_latest.pt

# 只加载权重并创建新实验
uv run sm-train --cfg configs/experiments/integral_actor.yaml \
  --set run_name=I1_warmstart --set init_from=logs/I1/checkpoints/ckpt_latest.pt

# 从 Checkpoint 配置恢复 history 并运行 Appendix F Benchmark
uv run python scripts/eval_benchmark.py --ckpt logs/I1/checkpoints/ckpt_latest.pt \
  --npz-dir data/npz/lafan1 --num-envs 256
```

Resume 对最终有效配置做严格比较，积分参数等训练语义发生变化时拒绝继续；日志频率、输出位置、设备和训练上限可按实现规则调整。旧 Checkpoint 缺少新增配置字段时以 Dataclass 默认值补齐；紧凑库模式只在旧配置确实缺少该字段且 `data.library` 明确存在时推导。`init_from` 只加载模型权重，不恢复优化器、计数器或采样状态，但其标准化路径和内容摘要属于 Sweep 训练身份。Resume 不恢复仿真状态和 Replay Buffer，会在新环境重新执行 Replay Warmup；原配置和历史评测文件保留。

## 验证范围与限制

- 当前完整自动化测试为 **79 passed、17 skipped**。通过项包括初始化权重身份、内容指纹、Reset 边界语义（CPU 与 CUDA）、Replay 容量限制，以及既有的可变历史网络尺寸、配置解析、Resume、Sweep 聚合、Recorder/Replay 和 Imagination 对齐；17 项跳过均因本环境没有下载 LAFAN1 动作数据。
- 本次修改文件 Ruff 与 `compileall` 通过。仓库整体 Ruff 仍有 **31 项既有问题**，均在未修改文件中。
- **Toy World Model**测试验证 Full/Detach 梯度流和策略更新链路；它不能替代 MuJoCo Warp 训练证据。
- RTX 4080 的 GPU smoke 对 B0/I1/I2/I3 各运行 5 个环境迭代、3 次 WM/Actor/Critic 更新，检查非零 Replay 积分、梯度、有限模型权重、Checkpoint 加载、单条合成 Benchmark 和四组 Sweep 汇总。使用的是 2 秒合成关节轨迹；每个策略都在该短片上触发失败终止，因此这只是工程链路证据，不是动作数据上的验证，也不用于研究性能比较。
- 本工作区没有 LAFAN1 或 BONES-SEED 数据。**正式数据上的训练、完整数据 Benchmark、收敛性、跨 Seed 稳定性和积分带来的性能收益仍未验证**；不得把实现完成或合成 smoke 结果解释成科研性能提升。
