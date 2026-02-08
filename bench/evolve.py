from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple
from collections import Counter, deque


DEFAULT_SKILLS = [
    "structured reasoning",
    "data analysis",
    "experimental design",
    "statistical thinking",
    "reading comprehension",
    "constraint following",
    "robustness / uncertainty",
    "summarization",
    "code generation",
]


@dataclass
class EvolutionPolicy:
    window: int = 30
    focus_top_k_tags: int = 3
    difficulty_min: int = 1
    difficulty_max: int = 5

    def next_focus(self, eval_history: List[Dict[str, Any]]) -> Tuple[List[str], List[str]]:
        """
        Returns (focus_skills, focus_tags)
        """
        recent = eval_history[-self.window:] if eval_history else []
        tags = []
        low_subskills = []

        for e in recent:
            tags.extend(e.get("error_tags", []) or [])
            subs = (e.get("subscores", {}) or {})
            # track low areas
            for k, v in subs.items():
                try:
                    if float(v) < 0.6 and k != "safety":
                        low_subskills.append(k)
                except Exception:
                    continue

        tag_counts = Counter(tags)
        top_tags = [t for t, _ in tag_counts.most_common(self.focus_top_k_tags)]

        sub_counts = Counter(low_subskills)
        top_subs = [s for s, _ in sub_counts.most_common(3)]

        # Map subscores to skills for generation bias
        skill_map = {
            "correctness": "structured reasoning",
            "completeness": "constraint following",
            "reasoning_quality": "structured reasoning",
            "format_compliance": "constraint following",
            "safety": "robustness / uncertainty",
        }
        focus_skills = []
        for s in top_subs:
            focus_skills.append(skill_map.get(s, s))

        if not focus_skills:
            focus_skills = ["mixed reasoning and analysis"]

        return focus_skills, top_tags

    def adjust_difficulty(self, current: int, ema_value: float) -> int:
        """
        Simple curriculum:
        - if ema >= 0.82 -> increase difficulty
        - if ema <= 0.62 -> decrease difficulty
        """
        nxt = current
        if ema_value >= 0.82:
            nxt += 1
        elif ema_value <= 0.62:
            nxt -= 1
        return max(self.difficulty_min, min(self.difficulty_max, nxt))
