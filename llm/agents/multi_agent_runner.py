import math
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence

import numpy as np
from ReplenishmentEnv import make_env

from llm.agents.llm_agent import LLMInventoryAgent
from llm.utils.metrics import EpisodeMetrics


@dataclass
class ActionPartition:
    names: List[str]
    sizes: List[int]

    @classmethod
    def even_split(cls, total: int, count: int):
        base = total // count
        remainder = total % count
        sizes = [base + (1 if i < remainder else 0) for i in range(count)]
        return cls(names=[f"tier_{i}" for i in range(count)], sizes=sizes)


class MultiAgentRunner:
    def __init__(self, env_cfg: Dict[str, Any], agents: Sequence[LLMInventoryAgent], action_partition: ActionPartition, max_steps: int = 200):
        self.env = make_env(
            env_cfg.get("config_name", "sku200.single_store.standard"),
            wrapper_names=env_cfg.get("wrapper_names", ["FlattenWrapper"]),
            mode=env_cfg.get("mode", "test"),
        )
        self.agents = list(agents)
        self.partition = action_partition
        self.max_steps = max_steps

    def _compose_action(self, per_agent_actions: List[List[float]]) -> np.ndarray:
        flat: List[float] = []
        for acts, size in zip(per_agent_actions, self.partition.sizes):
            flat.extend((acts + [0.0] * size)[:size])
        return np.asarray(flat, dtype=np.float32)

    def run_episode(self, seed: int = 0) -> EpisodeMetrics:
        obs = self._reset_env(seed)
        metrics = EpisodeMetrics()
        step = 0
        # Assume flattened observation; for history, we track last demand snapshot from obs tail
        while step < self.max_steps:
            if isinstance(obs, tuple):
                obs_val = obs[0]
            else:
                obs_val = obs
            demand_snapshot = obs_val[-10:].tolist() if hasattr(obs_val, "tolist") else str(obs_val)[-10:]
            for agent in self.agents:
                agent.update_history(demand_snapshot)
            per_agent_actions: List[List[float]] = []
            start = 0
            for agent, size in zip(self.agents, self.partition.sizes):
                slice_obs = self._slice_obs(obs_val, start, size)
                per_agent_actions.append(agent.propose_action(step=step, action_dim=size, observation=slice_obs))
                start += size
            action = self._compose_action(per_agent_actions)
            obs, reward, done, info = self.env.step(action)
            metrics.record(reward, info)
            step += 1
            if done:
                break
        return metrics

    def _reset_env(self, seed: int):
        try:
            return self.env.reset(seed=seed)
        except TypeError:
            return self.env.reset()

    def _slice_obs(self, obs_val: Any, start: int, size: int):
        try:
            return np.asarray(obs_val)[start : start + size].tolist()
        except Exception:
            return obs_val
