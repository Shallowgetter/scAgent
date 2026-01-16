from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np


@dataclass
class EpisodeMetrics:
    rewards: List[float] = field(default_factory=list)
    infos: List[Dict] = field(default_factory=list)

    def record(self, reward: float, info: Dict):
        try:
            reward_val = float(reward)
        except Exception:
            try:
                reward_val = float(np.sum(reward))
            except Exception:
                reward_val = 0.0
        self.rewards.append(reward_val)
        self.infos.append(info)

    @property
    def total_reward(self) -> float:
        return float(sum(self.rewards))

    @property
    def global_cost(self) -> float:
        # Define global cost as negative cumulative reward (assuming reward = -cost)
        return -self.total_reward

    def summary(self) -> Dict:
        return {
            "total_reward": self.total_reward,
            "global_cost": self.global_cost,
            "steps": len(self.rewards),
        }
