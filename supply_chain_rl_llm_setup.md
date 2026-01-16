# RL + LLM supply-chain plan (ReplenishmentEnv)

## Repo context
- Current env: `ReplenishmentEnv` is a Gym `Env` (`ReplenishmentEnv/env/replenishment_env.py`) built via `make_env(config_name, wrapper_names, mode, vis_path, update_config)` with wrappers such as `DefaultWrapper`, `ObservationWrapper`, `FlattenWrapper`, etc. Config/data live under `ReplenishmentEnv/config` and `ReplenishmentEnv/data`; baselines under `ReplenishmentEnv/Baseline` (OR + MARL).
- For quick inspection/debug, load with `make_env("sku200.single_store.standard")` (or other YAML) and check `env.observation_space`, `env.action_space`, `env.reset()`, and `.step()` outputs; visualization goes to `output/`.

## External references (GitHub, 2024-xx)
- Supply-chain RL envs: `frenkowski/SCIMAI-Gym` (Gym envs for inventory Mgmt, DRL baselines), `xiaotianliu01/Multi-Agent-Deep-Reinforcement-Learning-on-Multi-Echelon-Inventory-Management` (MADRL for multi-echelon), `waleed-6/Multi-Agent-Supply-Chain-Simulation-using-Reinforcement-Learning` (multi-city PPO demo), `kishorkukreja/SupplyChainv0_gym` (simpler gym env). Useful for ideas on observation/action shaping and reward.
- General MARL/PPO frameworks: RLlib (Ray), Stable-Baselines3 + `sb3-contrib` (PPO, MAPPO/IPPO via multi-agent wrapper), CleanRL (lean PPO reference), Tianshou (vectorized trainers), MARLlib (MAPPO/QMIX/QTRAN on PettingZoo), PyMARL2/QTRAN papers. Reference PPO impls: `AlirezaShamsoshoara/RL-from-zero` (PyTorch PPO/MARL from scratch).
- LLM + agent frameworks (for orchestration only, not RL): AutoGen, LangGraph, CrewAI; tool-use examples: `suryarapeti/SmartRetail` (LLM + retail multi-agent), `ttimg/rl_llm_multiagent_system_for_finance_modeling` (LLM+RL hybrid), `RainbowArena` (LLM + MARL toolkit ideas).

## Recommended file structure (add under repo root)
- `rl/` (new)  
  - `configs/` (env + algo YAML; separate `env/replenishment.yaml`, `algo/ppo.yaml`, `algo/ippo.yaml`, `algo/rllib_ppo.yaml`)  
  - `envs/` (gym interface wrappers, PettingZoo adaptor, vectorized env builders)  
  - `agents/` (algo implementations or wrappers around SB3/RLlib/CleanRL)  
  - `llm/` (prompt templates, tool functions to query env/state, policy explanation/analysis scripts)  
  - `scripts/` (`train_sb3.py`, `train_rllib.py`, `train_cleanrl.py`, `rollout.py`, `evaluate.py`)  
  - `experiments/` (wandb/logdir checkpoints + tensorboard)  
  - `docs/` (notes, ablation plan, hyperparam tables)  
- Keep existing `ReplenishmentEnv` untouched; point configs to `ReplenishmentEnv/config` and data to `ReplenishmentEnv/data`.

## Integration option A: SB3 PPO (fastest path)
- Why: simple single/multi-agent via wrappers; PPO from `stable-baselines3`; MAPPO/IPPO available via `sb3-contrib` or custom shared policy.
- Steps
  1) Install: `pip install stable-baselines3 sb3-contrib gymnasium==0.29 torch wandb`.
  2) Adapter: create `rl/envs/replenishment_sb3.py` exposing `gym.Env` with `reset()` -> `(obs, info)`, `step(action)` -> `(obs, reward, terminated, truncated, info)`. Use `FlattenWrapper` if obs is dict; ensure `action_space` is `spaces.Box`/`Discrete`. Example skeleton:
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
  3) Training (single-agent or centralized): `python rl/scripts/train_sb3.py --algo ppo --env sku200.single_store.standard --total-steps 2e6 --logdir experiments/sb3/ppo`. Use `VecNormalize` for reward/obs scaling; `CallbackList` for checkpointing.
  4) MAPPO/IPPO: if action needs per-warehouse/per-SKU, either (a) centralize actions in one vector (current env) and train one policy, or (b) split env per agent and use shared policy with `sb3-contrib` `MultiAgentPPO` (or custom loop). Start with centralized; move to multi-agent after verifying reward shaping.
  5) Evaluation: `python rl/scripts/rollout.py --checkpoint ckpt.zip --episodes 50 --mode test --vis output/ppo_vis`.

## Integration option B: RLlib PPO/MAPPO (scales + distributed)
- Why: built-in multi-agent API, vectorized rollout workers, good for MARL with per-agent policies.
- Steps
  1) Install: `pip install \"ray[rllib]==2.10.0\" torch wandb`.
  2) Create `rl/envs/replenishment_rllib.py` implementing `ray.rllib.env.MultiAgentEnv` if you want separate agents (per warehouse/SKU). Map each agent_id to its own observation/action slice; otherwise wrap as single-agent `gym.Env`.
  3) Config template (`rl/configs/algo/rllib_ppo.yaml`):
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
  policies: {shared: {}}  # define obs/action spaces
  policy_mapping_fn: !!python/name:rl.envs.replenishment_rllib.map_all_shared
```
  4) Train: `python rl/scripts/train_rllib.py --config rl/configs/algo/rllib_ppo.yaml --run PPO`. Add `evaluation_interval`, `evaluation_num_episodes`, and `wandb` logger via `ray.tune.logger.WandbLoggerCallback`.
  5) MARL algorithms: plug in RLlib’s MAPPO (`PPO` with `use_kl_loss=False`, centralized critic via `postprocess_fn`), QMIX/QTRAN via RLlib contrib, or swap to MARLlib (PettingZoo adaptor).

## Integration option C: CleanRL/Tianshou (lightweight research)
- CleanRL PPO gives readable single-file baselines; easiest to customize reward normalization or GAE changes. Use `VecEnv` wrappers for throughput.
- Tianshou offers multi-agent interfaces (`MultiAgentVectorEnv`) and on-policy/off-policy trainers; good for mixing PPO and Q-learning. Install: `pip install tianshou`.
- Convert env similarly to Option A; plug into trainer loops; log via TensorBoard/W&B.

## Using existing MARL baselines in repo
- `ReplenishmentEnv/Baseline/MARL_algorithm` already includes IPPO/QTRAN; configs at `Baseline/MARL_algorithm/config`. You can start from there by editing `task_type` in `replenishment.yaml` and running `python main.py --config=ippo --env-config=replenishment`. Consider porting that code into `rl/agents` for consistency and adding PPO hyperparam sweeps.

## LLM integration (non-RL)
- Goal: use LLM agents for planning/analysis, not gradient updates. Suggested flow:
  - `llm/tools.py`: safe tool functions (`get_state_snapshot(env)`, `simulate_policy(policy_fn, n_steps)`, `what_if(demand_scale)`).
  - `llm/prompts/`: role prompts for planner (policy designer), analyst (error analysis), reporter (business metrics).
  - Framework options: AutoGen or LangGraph to orchestrate tool-using agents; OpenAI/Anthropic clients behind an interface. Keep LLM separate from PPO training; only consume rollouts/logs to suggest hyperparams or generate heuristics (e.g., reorder point suggestions).
  - Guardrails: cap simulation depth per call; serialize obs/action into small summaries to avoid token bloat.

## Data/config guidance
- Start from `ReplenishmentEnv/config/demo.yml` and `sku200.single_store.standard` to understand shapes. Key fields: `env.mode`, `env.sku_list`, `warehouse[*].sku.shared_data/static_data/dynamic_data`, `reward_function`, `output_state`, `action`.
- For experiments, create new configs under `rl/configs/env/` that reference existing CSVs in `ReplenishmentEnv/data`; avoid duplicating data files.
- If action space is continuous, ensure PPO uses `squash_output`/`tanh` with appropriate scaling; if discrete, use categorical.
- Use `lookback_len` and demand history in obs; consider history stacking wrapper if needed (`HistoryWrapper` exists).

## Experiment design
- Baselines: OR policies (Base-stock, (s,S)) in `Baseline/OR_algorithm` for comparison; run to get KPIs.
- PPO sweeps: batch size {8k,16k}, learning rate {3e-4,1e-4}, entropy coeff {0.0,0.01}, clip {0.1,0.2}. Eval on val/test split (env mode).
- Multi-agent variants: centralized training with decentralized execution (share policy, separate obs) vs fully centralized critic; begin with centralized PPO then move to MAPPO/IPPO.
- Metrics: profit, stockouts, holding cost, service level; log via wandb/TensorBoard; use env visualizer outputs for sanity checks.

## Next steps to implement
- Decide on backend (Option A quickest, Option B scalable). Create `rl/` skeleton above.
- Build SB3 adapter + train script; run smoke test 1e4 steps to confirm reward signal.
- Add LLM tooling folder with prompts + tool functions for analysis/what-if simulations.
- Set up wandb or TensorBoard logging; keep checkpoints under `experiments/`.
