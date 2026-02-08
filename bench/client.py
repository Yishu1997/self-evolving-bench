from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from openai import OpenAI


@dataclass
class LLMClient:
    """
    Thin wrapper around an OpenAI API–compatible endpoint.

    Works with:
      - OpenAI: base_url=https://api.openai.com/v1
      - OpenAI-compatible servers (vLLM, LM Studio, etc.)
      - Gateways that implement the OpenAI API surface
    """
    base_url: str
    api_key: str
    model: str

    def __post_init__(self) -> None:
        self._client = OpenAI(base_url=self.base_url, api_key=self.api_key)

    def chat(self,
             messages: List[Dict[str, str]],
             *,
             model: Optional[str] = None,
             temperature: float = 0.2,
             max_tokens: int = 800,
             response_format: Optional[Dict[str, Any]] = None) -> str:
        m = model or self.model
        kwargs: Dict[str, Any] = {
            "model": m,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if response_format is not None:
            kwargs["response_format"] = response_format
        resp = self._client.chat.completions.create(**kwargs)
        return resp.choices[0].message.content or ""

    def embeddings(self, texts: List[str], *, model: str = "text-embedding-3-small") -> List[List[float]]:
        """
        Optional. Only works if the endpoint supports embeddings.
        Not used by default in this repo (we use local TF-IDF similarity),
        but provided in case you'd like to extend it.
        """
        resp = self._client.embeddings.create(model=model, input=texts)
        return [d.embedding for d in resp.data]
