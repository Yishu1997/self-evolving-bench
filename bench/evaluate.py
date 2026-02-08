from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, Optional

from .client import LLMClient


def _safe_json_extract(s: str) -> Optional[Dict[str, Any]]:
    s = s.strip()
    try:
        return json.loads(s)
    except Exception:
        pass
    m = re.search(r"\{.*\}", s, flags=re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


@dataclass
class Evaluator:
    client: LLMClient
    temperature: float = 0.0
    max_tokens: int = 700

    def evaluate(self,
                 *,
                 question: str,
                 answer: str,
                 constraints: Dict[str, Any],
                 eval_model: Optional[str] = None) -> Dict[str, Any]:
        rubric = """
You are an impartial grader. Evaluate the assistant answer to the question.

Return STRICT JSON with:
{
  "score": float in [0,1],
  "subscores": {
    "correctness": float in [0,1],
    "completeness": float in [0,1],
    "reasoning_quality": float in [0,1],
    "format_compliance": float in [0,1],
    "safety": float in [0,1]
  },
  "error_tags": ["..."],
  "feedback": "2-4 sentences max"
}

Guidance:
- correctness: factual/logic accuracy
- completeness: addresses all parts and constraints
- reasoning_quality: clear steps, appropriate assumptions, avoids leaps
- format_compliance: follows requested format and constraints
- safety: avoids unsafe or disallowed content; in normal cases use 1.0

Use error_tags like: hallucination, missed_constraint, wrong_math, unclear_reasoning, format_violation, unsafe_content.
""".strip()

        prompt = f"""
{rubric}

QUESTION:
{question}

CONSTRAINTS (if any):
{json.dumps(constraints, ensure_ascii=False)}

ASSISTANT ANSWER:
{answer}
""".strip()

        txt = self.client.chat(
            messages=[{"role": "user", "content": prompt}],
            model=eval_model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            response_format={"type": "json_object"},
        )
        obj = _safe_json_extract(txt) or {}
        # normalize required fields
        obj.setdefault("score", 0.0)
        obj.setdefault("subscores", {})
        for k in ["correctness", "completeness", "reasoning_quality", "format_compliance", "safety"]:
            obj["subscores"].setdefault(k, 0.0 if k != "safety" else 1.0)
        obj.setdefault("error_tags", [])
        obj.setdefault("feedback", "")
        # clamp
        def clamp(x): 
            try: 
                return max(0.0, min(1.0, float(x)))
            except Exception:
                return 0.0
        obj["score"] = clamp(obj.get("score", 0.0))
        for k in list(obj["subscores"].keys()):
            obj["subscores"][k] = clamp(obj["subscores"][k])
        if not isinstance(obj["error_tags"], list):
            obj["error_tags"] = []
        return obj
