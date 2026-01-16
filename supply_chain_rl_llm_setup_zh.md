# RL + LLM 供应链方案（ReplenishmentEnv）

## 仓库现状
- 当前环境：`ReplenishmentEnv/env/replenishment_env.py` 是 Gym `Env`，通过 `make_env(config_name, wrapper_names, mode, vis_path, update_config)` 构建，常用封装有 `DefaultWrapper`、`ObservationWrapper`、`FlattenWrapper` 等。配置与数据在 `ReplenishmentEnv/config`、`ReplenishmentEnv/data`，基线（OR + MARL）在 `ReplenishmentEnv/Baseline`。快速调试：`make_env("sku200.single_store.standard")`，检查 `env.observation_space`、`env.action_space`、`env.reset()`、`step()`；可视化输出到 `output/`。

## 外部参考（GitHub，2024-xx）
- 供应链 RL 环境与基线：`frenkowski/SCIMAI-Gym`（库存管理 Gym 环境、DRL 基线）、`xiaotianliu01/Multi-Agent-Deep-Reinforcement-Learning-on-Multi-Echelon-Inventory-Management`（多层级 MADRL）、`waleed-6/Multi-Agent-Supply-Chain-Simulation-using-Reinforcement-Learning`（多城市 PPO 示例）、`kishorkukreja/SupplyChainv0_gym`（简化 Gym 环境）。可参考观测/动作设计与奖励定义。
- 通用 MARL/PPO 框架：RLlib（Ray）、Stable-Baselines3 + `sb3-contrib`（PPO、MAPPO/IPPO）、CleanRL（精简版 PPO 参考）、Tianshou（向量化训练器）、MARLlib（MAPPO/QMIX/QTRAN on PettingZoo）、PyMARL2/QTRAN 论文。PPO 参考实现：`AlirezaShamsoshoara/RL-from-zero`（PyTorch PPO/MARL）。
- LLM + agent 框架（编排，不做 RL 更新）：AutoGen、LangGraph、CrewAI；工具调用示例：`suryarapeti/SmartRetail`（LLM + 零售多代理）、`ttimg/rl_llm_multiagent_system_for_finance_modeling`（LLM+RL 混合）、`RainbowArena`（LLM + MARL 工具包思路）。

## 推荐文件结构（仓库根目录下新增）
- `rl/`  
  - `configs/`（环境与算法 YAML；如 `env/replenishment.yaml`、`algo/ppo.yaml`、`algo/ippo.yaml`、`algo/rllib_ppo.yaml`）  
  - `envs/`（Gym 接口封装、PettingZoo 适配、向量化环境构建）  
  - `agents/`（算法实现或 SB3/RLlib/CleanRL 封装）  
  - `llm/`（提示模板、查询环境/状态的工具函数、策略解释/分析脚本）  
  - `scripts/`（`train_sb3.py`、`train_rllib.py`、`train_cleanrl.py`、`rollout.py`、`evaluate.py`）  
  - `experiments/`（wandb/TensorBoard 日志与 checkpoint）  
  - `docs/`（笔记、消融计划、超参表）  
- 保持 `ReplenishmentEnv` 现有结构不变；配置指向 `ReplenishmentEnv/config`，数据指向 `ReplenishmentEnv/data`。

## 需求对照与补齐方案
- 环境统一：所有实验（LLM 多智能体、传统 RL、多模态混合）都通过 `rl/envs/` 的适配层调用 `ReplenishmentEnv`，保证同一观测/动作/奖励定义。
- 三类实验覆盖：
  - LLM-based multi-agent（无 RL）：`llm/agents/` 下实现 AutoGen/LangGraph/CrewAI 流程；`llm/tools/` 提供只读/可控接口（状态快照、滚动模拟、what-if）；入口脚本 `llm/run_llm_agents.py`，配置放 `llm/configs/*.yaml`（指定角色、工具配额、步长限制）。
  - 传统网络 multi-agent（有 RL）：`rl/agents/{sb3,rllib,cleanrl}/` 支持 PPO/GRPO/MAPPO；配置放 `rl/configs/algo/{ppo,grpo,mappo}.yaml`；训练脚本 `rl/scripts/train_{sb3,rllib,cleanrl}.py`；评估脚本 `rl/scripts/rollout.py`、`evaluate.py`。如需 GRPO，可在 `rl/agents/custom/grpo.py` 实现自定义 loss/优势估计。
  - 混合（LLM + 传统网络）：LLM 负责规划/指挥，不参与梯度；RL 策略负责动作。放在 `llm/scenarios/hybrid/`（提示 + 策略调用逻辑），桥接工具 `llm/tools/policy_bridge.py` 读取 RL checkpoint、调用 `rl/agents/*` 推理接口；入口 `llm/run_hybrid.py`。
- 集成功能与复用：
  - 统一配置：`rl/configs/env/*.yaml` 指向 `ReplenishmentEnv/config` 数据；`llm/configs/` 复用同名环境，避免重复描述。
  - 统一日志与输出：`experiments/llm_only/`、`experiments/rl/`、`experiments/hybrid/` 分桶存放 wandb/TensorBoard/可视化；公用 `output/` 或子目录。
  - 依赖拆分：`requirements_rl.txt`（torch, gymnasium, stable-baselines3, sb3-contrib, ray[rllib], tianshou 等）、`requirements_llm.txt`（autogen/langgraph/crewai、openai/anthropic 客户端）按需安装；顶层 `requirements.txt` 可精简为共同依赖。
  - CLI 与脚本：提供 `Makefile` 或 `scripts/` 简化常用命令（训练、评估、可视化、LLM 推理），减少手动拼参。

## 方案 A：SB3 PPO（最快路径）
- 理由：单/多智能体易封装；PPO 来自 `stable-baselines3`；MAPPO/IPPO 可用 `sb3-contrib` 或自定义共享策略。
- 安装：`pip install stable-baselines3 sb3-contrib gymnasium==0.29 torch wandb`。
- 适配器：创建 `rl/envs/replenishment_sb3.py`，暴露 `gym.Env`，`reset()` -> `(obs, info)`，`step(action)` -> `(obs, reward, terminated, truncated, info)`；观测若为字典可用 `FlattenWrapper`；确保 `action_space` 为 `spaces.Box`/`Discrete`。示例：
```python
from ReplenishmentEnv import make_env
import gymnasium as gym

class ReplenishmentSB3(gym.Env):
    def __init__(self, config_name="sku200.single_store.standard", mode="train"):
        self._env = make_env(config_name, wrapper_names=["FlattenWrapper"], mode=mode)
        self.observation_space = self._env.observation_space
        self.action_space = self._env.action_space
    def reset(self, seed=None, options=None):
        obs = self._env.reset()
        return obs, {}
    def step(self, action):
        obs, reward, done, info = self._env.step(action)
        return obs, reward, done, False, info
```
- 训练（单智能体或集中式）：`python rl/scripts/train_sb3.py --algo ppo --env sku200.single_store.standard --total-steps 2e6 --logdir experiments/sb3/ppo`；使用 `VecNormalize` 进行观测/奖励归一化，`CallbackList` 做 checkpoint。
- MAPPO/IPPO：若动作需按仓库/SKU 切分，可（a）集中为单向量训练一策略（现状），或（b）拆分为多 agent、共享策略并用 `sb3-contrib` `MultiAgentPPO`（或自定义循环）。先集中式，验证奖励后再做多智能体。
- 评估：`python rl/scripts/rollout.py --checkpoint ckpt.zip --episodes 50 --mode test --vis output/ppo_vis`。

## 方案 B：RLlib PPO/MAPPO（可扩展、分布式）
- 理由：内置多智能体 API，向量化 rollout worker，适合每 agent 独立策略的 MARL。
- 安装：`pip install "ray[rllib]==2.10.0" torch wandb`。
- 适配器：创建 `rl/envs/replenishment_rllib.py`，实现 `ray.rllib.env.MultiAgentEnv`（按 agent 切分观测/动作），或封装为单智能体 `gym.Env`。
- 配置模板（`rl/configs/algo/rllib_ppo.yaml`）示例：
```yaml
env: rl.envs.replenishment_rllib:ReplenishmentMultiAgent
env_config:
  config_name: sku200.single_store.standard
  wrapper_names: [FlattenWrapper]
  mode: train
framework: torch
num_workers: 4
rollout_fragment_length: 200
train_batch_size: 3200
sgd_minibatch_size: 256
gamma: 0.99
lambda: 0.95
lr: 0.0003
vf_clip_param: 100.0
multiagent:
  policies: {shared: {}}  # 定义观测/动作空间
  policy_mapping_fn: !!python/name:rl.envs.replenishment_rllib.map_all_shared
```
- 训练：`python rl/scripts/train_rllib.py --config rl/configs/algo/rllib_ppo.yaml --run PPO`；可添加 `evaluation_interval`、`evaluation_num_episodes`，用 `ray.tune.logger.WandbLoggerCallback` 接 wandb。
- MARL 算法：RLlib MAPPO（PPO 配置 `use_kl_loss=False`，集中 critic 经 `postprocess_fn`），QMIX/QTRAN via RLlib contrib，或换用 MARLlib（PettingZoo 适配）。

## 方案 C：CleanRL/Tianshou（轻量研究）
- CleanRL PPO 单文件，便于自定义奖励归一化或 GAE；使用 `VecEnv` 提升吞吐。
- Tianshou 提供多智能体接口（`MultiAgentVectorEnv`）与 on/off-policy 训练器，适合混合 PPO 与 Q-learning。安装：`pip install tianshou`。
- 环境转换同方案 A，接入训练循环；日志用 TensorBoard/W&B。

## 使用仓库已有 MARL 基线
- `ReplenishmentEnv/Baseline/MARL_algorithm` 已含 IPPO/QTRAN，配置在 `Baseline/MARL_algorithm/config`。可通过调整 `replenishment.yaml` 的 `task_type`，运行 `python main.py --config=ippo --env-config=replenishment`。可将代码迁移到 `rl/agents` 统一风格，并增加 PPO 超参搜索。

## LLM 集成（非 RL 更新）
- 目标：用 LLM 做规划/分析，不做梯度更新。建议：
  - `llm/tools.py`：安全工具函数（`get_state_snapshot(env)`、`simulate_policy(policy_fn, n_steps)`、`what_if(demand_scale)`）。
  - `llm/prompts/`：角色提示（规划/策略设计、分析/误差诊断、报告/业务指标）。
  - 框架：AutoGen 或 LangGraph 进行多 agent 编排；OpenAI/Anthropic 客户端封装在统一接口后面。保持与 PPO 训练解耦，仅消费 rollout/log，给出超参建议或生成启发式（如补货点建议）。
  - 防护：限制每次仿真步数；将观测/动作压缩为摘要，避免 token 膨胀。

## 数据与配置提示
- 先看 `ReplenishmentEnv/config/demo.yml`、`sku200.single_store.standard` 理解观测/动作结构。关键字段：`env.mode`、`env.sku_list`、`warehouse[*].sku.shared_data/static_data/dynamic_data`、`reward_function`、`output_state`、`action`。
- 实验配置放在 `rl/configs/env/`，引用 `ReplenishmentEnv/data` 的 CSV，避免重复数据。
- 若动作连续，用 `squash_output`/`tanh` 做缩放；若离散，用分类分布。
- 使用 `lookback_len` 和需求历史；如需历史堆叠，可用现有 `HistoryWrapper`。

## 实验设计
- 基线：OR 策略在 `Baseline/OR_algorithm`（如 Base-stock、(s,S)）用于对比，输出 KPI。
- PPO 网格：batch {8k, 16k}，学习率 {3e-4, 1e-4}，熵系数 {0, 0.01}，clip {0.1, 0.2}。按 train/val/test 划分评估。
- 多智能体：集中训练、分布执行（共享策略、独立观测） vs 完全集中 critic；先集中式 PPO，再到 MAPPO/IPPO。
- 指标：利润、缺货、库存持有成本、服务水平；日志用 wandb/TensorBoard；结合环境可视化做 sanity check。

## 下一步实施
- 选择后端（方案 A 快速，方案 B 可扩展），创建 `rl/` 目录结构。
- 编写 SB3 适配器 + 训练脚本，跑 1e4 步冒烟测试确认奖励信号。
- 添加 LLM 工具与提示目录，支持分析/what-if。
- 配置 wandb 或 TensorBoard 日志，checkpoint 放在 `experiments/`。
