import argparse
import json
import math
import os
from pathlib import Path
from typing import List

import yaml

from llm.agents.llm_agent import LLMInventoryAgent
from llm.agents.multi_agent_runner import ActionPartition, MultiAgentRunner
from llm.tools.model_registry import make_llm_client


def load_cfg(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_agents(cfg: dict, total_action_dim: int) -> ActionPartition:
    levels = cfg["agents"].get("levels", [])
    if not levels:
        raise ValueError("No agent levels provided in config")
    sizes = cfg["agents"].get("action_partition")
    if isinstance(sizes, list):
        if sum(sizes) != total_action_dim:
            raise ValueError(f"Action partition sum {sum(sizes)} != action_dim {total_action_dim}")
        partition = ActionPartition(names=[lvl["name"] for lvl in levels], sizes=sizes)
    else:
        partition = ActionPartition.even_split(total_action_dim, len(levels))
    return partition


def init_agents(cfg: dict, partition: ActionPartition) -> List[LLMInventoryAgent]:
    llm_cfg = cfg.get("llm", {})
    client = make_llm_client(llm_cfg)
    system_prompt = llm_cfg.get("system_prompt")
    user_prompt = llm_cfg.get("user_prompt")
    agents: List[LLMInventoryAgent] = []
    for name, size in zip(partition.names, partition.sizes):
        agents.append(
            LLMInventoryAgent(
                name=name,
                llm_client=client,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                history_limit=llm_cfg.get("history_limit", 20),
            )
        )
    return agents


def infer_action_dim(env_cfg: dict) -> int:
    from ReplenishmentEnv import make_env

    env = make_env(
        env_cfg.get("config_name", "sku200.single_store.standard"),
        wrapper_names=env_cfg.get("wrapper_names", ["FlattenWrapper"]),
        mode=env_cfg.get("mode", "test"),
    )
    dim = int(math.prod(env.action_space.shape))
    return dim


def main():
    parser = argparse.ArgumentParser(description="LLM-based multi-agent baseline")
    parser.add_argument("--config", default="llm/configs/llm_supply_chain.yaml")
    parser.add_argument("--episodes", type=int, default=None, help="override episodes")
    parser.add_argument("--use-dummy", action="store_true", help="force dummy LLM (no API calls)")
    args = parser.parse_args()

    cfg = load_cfg(args.config)
    if args.episodes is not None:
        cfg.setdefault("rollout", {})["episodes"] = args.episodes
    if args.use_dummy:
        cfg.setdefault("llm", {})["use_dummy"] = True

    env_cfg = cfg.get("env", {})
    action_dim = infer_action_dim(env_cfg)
    partition = build_agents(cfg, action_dim)
    agents = init_agents(cfg, partition)

    runner = MultiAgentRunner(env_cfg=env_cfg, agents=agents, action_partition=partition, max_steps=cfg.get("rollout", {}).get("max_steps", 200))

    episodes = cfg.get("rollout", {}).get("episodes", 1)
    seed = cfg.get("rollout", {}).get("seed", 0)

    summaries = []
    for ep in range(episodes):
        metrics = runner.run_episode(seed=seed + ep)
        summaries.append(metrics.summary())
        print(f"Episode {ep}: {metrics.summary()}")

    out_path = Path("experiments/llm_only/metrics.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2)
    print(f"Saved metrics to {out_path}")


if __name__ == "__main__":
    main()
