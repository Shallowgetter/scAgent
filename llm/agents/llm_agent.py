import json
from dataclasses import dataclass, field
from typing import Any, Dict, List

from llm.tools.model_registry import LLMClient
from llm.tools.prompt_templates import render_prompts


@dataclass
class LLMInventoryAgent:
    name: str
    llm_client: LLMClient
    system_prompt: str
    user_prompt: str
    history_limit: int = 20
    demand_history: List[Any] = field(default_factory=list)

    def update_history(self, demand_snapshot: Any):
        self.demand_history.append(demand_snapshot)
        if len(self.demand_history) > self.history_limit:
            self.demand_history = self.demand_history[-self.history_limit :]

    def _build_messages(self, step: int, action_dim: int, observation: Any) -> List[Dict[str, str]]:
        prompts = render_prompts(
            tier_name=self.name,
            step=step,
            action_dim=action_dim,
            observation=str(observation),
            demand_history=str(self.demand_history),
            system_prompt=self.system_prompt,
            user_prompt=self.user_prompt,
        )
        return [
            {"role": "system", "content": prompts["system"]},
            {"role": "user", "content": prompts["user"]},
        ]

    def propose_action(self, step: int, action_dim: int, observation: Any) -> List[float]:
        messages = self._build_messages(step=step, action_dim=action_dim, observation=observation)
        raw = self.llm_client.chat(messages)
        return self._parse_actions(raw, action_dim)

    def _parse_actions(self, raw: str, action_dim: int) -> List[float]:
        try:
            payload = json.loads(raw)
            actions = payload.get("actions", [])
            if not isinstance(actions, list):
                raise ValueError
            # Pad/trim to action_dim
            actions = (actions + [0.0] * action_dim)[:action_dim]
            return [float(x) for x in actions]
        except Exception:
            return [0.0] * action_dim
