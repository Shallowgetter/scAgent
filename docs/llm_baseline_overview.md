# LLM-Only Multi-Agent Baseline (ReplenishmentEnv)

## Goals
- Provide a non-RL, LLM-based multi-agent controller where each tier (store/warehouse/DC) decides its own order quantities.
- Keep core integrations reusable: pluggable LLM providers (GPT, Gemini, DeepSeek, etc.), customizable prompts, action partitioning across tiers, and externalized metrics (global cost).

## Components
- `llm/tools/model_registry.py`: Provider-agnostic LLM client builder with dummy mode (offline). Supports OpenAI-compatible endpoints; keys from env vars (`OPENAI_API_KEY`, `GEMINI_API_KEY`, `DEEPSEEK_API_KEY` or custom).
- `llm/tools/prompt_templates.py`: System/user prompt templates; fully customizable per config.
- `llm/agents/llm_agent.py`: Per-tier agent with demand history buffer, prompt rendering, JSON action parsing (pads/trims to action slice).
- `llm/agents/multi_agent_runner.py`: Orchestrates multiple LLM agents over `ReplenishmentEnv`; partitions action vector by tier; runs episodes; handles reset quirks.
- `llm/utils/metrics.py`: Tracks cumulative reward and global cost (= -total reward).
- `llm/configs/llm_supply_chain.yaml`: Example config (tiers, LLM provider/model, prompts, rollout settings, dummy mode).
- `llm/run_llm_agents.py`: CLI entrypoint; infers action_dim from env, builds agents/partition, runs episodes, saves `experiments/llm_only/metrics.json`.

## How it works
1) Env is created via `ReplenishmentEnv.make_env` with wrappers (defaults to `FlattenWrapper`).
2) Total action dimension is inferred from `env.action_space.shape`. It is split evenly across tiers unless `agents.action_partition` is set.
3) Each agent receives its observation slice and recent demand history; generates JSON `{ "actions": [...] }`; runner composes the full action vector.
4) Metrics are logged per episode (global cost).

## Usage
```
export PYTHONPATH=$PWD/ReplenishmentEnv:$PYTHONPATH  # or pip install -e ./ReplenishmentEnv
pip install -r requirements_llm.txt

python -m llm.run_llm_agents --config llm/configs/llm_supply_chain.yaml --use-dummy
```
- Remove `--use-dummy` and set API keys to call real LLMs. Change provider/model/prompts in the YAML as needed.
- Adjust `agents.action_partition` if your env action space should be split non-uniformly across tiers.

## Extension ideas
- Add W&B/TensorBoard logging hooks to `llm/run_llm_agents.py`.
- Extend `llm/utils/metrics.py` with per-tier KPIs and service-level stats.
- Add multi-turn prompt flows by enriching `llm_agent.py` to maintain dialogue state.
- Plug additional providers via `model_registry.py` (custom base URLs or SDKs).
