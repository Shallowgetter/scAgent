from typing import Dict

DEFAULT_SYSTEM_PROMPT = (
    "You are a supply-chain inventory planner. "
    "You control one tier (e.g., store, warehouse, DC). "
    "Propose non-negative order quantities per SKU given demand history, inventory, and capacity. "
    "Respond with JSON: {\"actions\": [numbers...]}. Keep it concise."
)

DEFAULT_USER_PROMPT = (
    "Context:\n"
    "Tier: {tier_name}\n"
    "Step: {step}\n"
    "Action slice length: {action_dim}\n"
    "Observation summary: {observation}\n"
    "Recent demand history: {demand_history}\n"
    "Business goal: minimize total cost while avoiding stockouts.\n"
    "Return strictly JSON with key 'actions' of length {action_dim}."
)


def render_prompts(tier_name: str, step: int, action_dim: int, observation: str, demand_history: str, system_prompt: str = DEFAULT_SYSTEM_PROMPT, user_prompt: str = DEFAULT_USER_PROMPT) -> Dict[str, str]:
    return {
        "system": system_prompt,
        "user": user_prompt.format(
            tier_name=tier_name,
            step=step,
            action_dim=action_dim,
            observation=observation,
            demand_history=demand_history,
        ),
    }
