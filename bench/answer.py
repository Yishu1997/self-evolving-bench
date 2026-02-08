from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from .client import LLMClient


@dataclass
class Answerer:
    client: LLMClient
    temperature: float = 0.2
    max_tokens: int = 900

    def answer(self, question: str, *, model: Optional[str] = None) -> str:
        messages = [
            {"role": "system", "content": "You are a helpful assistant. Follow instructions carefully."},
            {"role": "user", "content": question},
        ]
        return self.client.chat(
            messages=messages,
            model=model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        ).strip()
