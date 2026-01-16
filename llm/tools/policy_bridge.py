"""Placeholder bridge for calling RL policies from LLM tools.

Expected responsibilities:
- Load SB3/RLlib/CleanRL checkpoints
- Expose a small inference API for LLM planners (given state -> action/proposal)
- Enforce rollout length and safety limits
"""

def load_policy(checkpoint_path: str):
    raise NotImplementedError("Load RL policy from checkpoint")


def act(policy, observation):
    raise NotImplementedError("Run policy inference with safety guards")
