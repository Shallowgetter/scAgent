# Supply-Chain RL + LLM Playground

This repo hosts the **ReplenishmentEnv** simulator and scaffolding for three experiment tracks:
1. LLM-based multi-agent planning (no RL updates).
2. Multi-agent RL with traditional networks (PPO/GRPO/MAPPO, etc.).
3. Hybrid: LLM planners coordinating RL policies (LLM does not train via gradients).

## Layout
- `ReplenishmentEnv/` — upstream simulator (configs/data/wrappers/baselines).
- `baseline/` — legacy resources (kept intact).
- `rl/` — RL stack: env adapters, agents, configs, scripts, experiments, docs.
  - `configs/{env,algo}/` — YAML configs for env/algo (to be filled).
  - `envs/` — Gym/PettingZoo adapters around `ReplenishmentEnv.make_env`.
  - `agents/{sb3,rllib,cleanrl,custom}/` — algo wrappers/implementations.
  - `scripts/` — training/eval entrypoints (`train_sb3.py`, `train_rllib.py`, `train_cleanrl.py`, `rollout.py`, `evaluate.py` placeholders).
  - `experiments/` — logs/checkpoints (segmented by backend).
- `llm/` — LLM orchestration: agents, configs, tools, hybrid scenarios.
  - `run_llm_agents.py` — LLM-only multi-agent entrypoint (placeholder).
  - `run_hybrid.py` — LLM+RL hybrid entrypoint (placeholder).
  - `tools/policy_bridge.py` — loads RL policies for LLM planners (placeholder).
- `experiments/{llm_only,rl,hybrid}/` — output/log roots by track.
- `docs/` — change/setup notes.
- `requirements_rl.txt`, `requirements_llm.txt` — split dependency sets (opt-in).

## Experiment tracks
- **LLM-only multi-agent**: Orchestrate roles (planner/analyst/reporter) via AutoGen/LangGraph/CrewAI; tools expose safe state snapshots, bounded rollouts, and what-if probes. No gradient updates.
- **RL multi-agent**: Train centralized or decentralized policies via SB3, RLlib, or CleanRL/Tianshou. Algorithms: PPO, MAPPO/IPPO, GRPO (custom). Checkpoints/logs saved under `experiments/rl`.
- **Hybrid**: LLM handles planning/coordination; RL policy produces actions. Bridge loads checkpoints and enforces rollout limits; LLM remains non-trainable.

## Getting started
1) Install the simulator (from repo root; use editable to develop):
```
pip install -e ./ReplenishmentEnv
```
If network or permission issues block install, you can run with a local path:
```
export PYTHONPATH=$PWD/ReplenishmentEnv:$PYTHONPATH
```
and invoke scripts as `python -m baseline.env_test` from repo root.
2) Install dependencies (pick what you need):
```
pip install -r requirements_rl.txt
pip install -r requirements_llm.txt
```
3) Smoke-test env:
```python
from ReplenishmentEnv import make_env
env = make_env("sku200.single_store.standard", wrapper_names=["FlattenWrapper"], mode="test")
obs = env.reset()
print(env.observation_space, env.action_space)
```
4) If you run baseline/tests from this repo (e.g., `python baseline/env_test.py`), ensure you are in the same virtualenv where `ReplenishmentEnv` was installed (`pip install -e ./ReplenishmentEnv`) or set `PYTHONPATH` to the repo root.

## Version notes (RL stack)
- RLlib 2.10.0 requires `gymnasium==0.28.1`; we pin SB3/SB3-contrib to 2.2.1 to match that gymnasium version. If you prefer SB3 2.3+ (gymnasium 0.29), use a separate virtualenv or upgrade Ray/RLlib accordingly.
- LLM stack is independent; keep it in its own environment if desired.

## Next steps
- Implement env adapters in `rl/envs/` and wire training scripts in `rl/scripts/`.
- Add YAML configs under `rl/configs/` for env/algo defaults (ppo, grpo, mappo, rllib_ppo).
- Flesh out LLM tool functions and orchestration graphs under `llm/`.
- Standardize logging to WandB/TensorBoard under `experiments/` and visualization to `output/`.
