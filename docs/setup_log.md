# Setup & Change Log (LLM + RL Integration Scaffolding)

Date: 2026-01-12

## Changes added
- Created scaffolding for three experiment tracks (LLM-only, RL, hybrid) while keeping `ReplenishmentEnv` untouched.
- Added top-level directories: `rl/` (configs, env adapters, agents, scripts, experiments, docs), `llm/` (agents, configs, tools, hybrid scenarios), `experiments/{llm_only,rl,hybrid}` for outputs.
- Added split dependency files: `requirements_rl.txt`, `requirements_llm.txt`.
- Added placeholder entrypoints:
  - `rl/scripts/{train_sb3.py,train_rllib.py,train_cleanrl.py,rollout.py,evaluate.py}`
  - `llm/run_llm_agents.py`, `llm/run_hybrid.py`, `llm/tools/policy_bridge.py`
- Added root `README.md` summarizing architecture and usage.
- Adjusted RL dependency pins to satisfy RLlib: `gymnasium==0.28.1`, `stable-baselines3==2.2.1`, `sb3-contrib==2.2.1`, `ray[rllib]==2.10.0`.

## Environment setup (current)
1) Install simulator (editable for dev):
```
pip install -e ./ReplenishmentEnv
```
If installation is blocked (no network/permissions), use path-based import when running scripts:
```
export PYTHONPATH=$PWD/ReplenishmentEnv:$PYTHONPATH
```
2) Install dependencies as needed:
```
pip install -r requirements_rl.txt
pip install -r requirements_llm.txt
```

## Follow-ups to implement
- Fill `rl/envs/` adapters that wrap `ReplenishmentEnv.make_env` for SB3/RLlib/PettingZoo.
- Add algorithm/env YAML defaults under `rl/configs/` (ppo/grpo/mappo/rllib_ppo).
- Implement training/eval logic in `rl/scripts/` and KPI aggregation in `evaluate.py`.
- Implement LLM orchestration graphs and tool safety limits in `llm/`.
- Standardize logging/checkpoint conventions under `experiments/` and visualization under `output/`.
- If you need SB3 2.3+ (gymnasium 0.29), use a separate virtualenv from RLlib 2.10.0, which pins gymnasium 0.28.1.
- When running scripts/tests inside this repo (e.g., `python baseline/env_test.py`), use the same interpreter where `pip install -e ./ReplenishmentEnv` was executed, or set `PYTHONPATH` to the repo root.
