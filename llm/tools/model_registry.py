import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

try:
    from openai import OpenAI  # OpenAI-compatible SDK
except Exception:  # pragma: no cover - optional dependency
    OpenAI = None


@dataclass
class LLMClient:
    provider: str
    model: str
    temperature: float = 0.2
    max_tokens: int = 256
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    use_dummy: bool = False

    def _get_client(self) -> Optional[Any]:
        if self.use_dummy:
            return None
        if OpenAI is None:
            raise RuntimeError("openai package not available; set use_dummy=True or install openai")
        key = self.api_key or os.getenv(self._key_env())
        if not key:
            raise RuntimeError(f"Missing API key for provider {self.provider}; set {self._key_env()}")
        return OpenAI(api_key=key, base_url=self.base_url)

    def _key_env(self) -> str:
        if self.provider.lower() in {"gpt", "openai"}:
            return "OPENAI_API_KEY"
        if self.provider.lower() == "gemini":
            return "GEMINI_API_KEY"
        if self.provider.lower() == "deepseek":
            return "DEEPSEEK_API_KEY"
        return "LLM_API_KEY"

    def chat(self, messages: List[Dict[str, str]]) -> str:
        if self.use_dummy:
            # Deterministic stub to avoid network; returns zero actions.
            return "{\"actions\": []}"
        client = self._get_client()
        response = client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            messages=messages,
        )
        return response.choices[0].message.content


def make_llm_client(cfg: Dict[str, Any]) -> LLMClient:
    return LLMClient(
        provider=cfg.get("provider", "gpt"),
        model=cfg.get("model", "gpt-4o-mini"),
        temperature=cfg.get("temperature", 0.2),
        max_tokens=cfg.get("max_tokens", 256),
        base_url=cfg.get("base_url"),
        api_key=cfg.get("api_key"),
        use_dummy=cfg.get("use_dummy", False),
    )
