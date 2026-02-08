from __future__ import annotations

import hashlib
import json
import random
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .client import LLMClient


def normalize_text(s: str) -> str:
    s = s.lower()
    s = re.sub(r"\s+", " ", s).strip()
    # keep alnum and basic punctuation spacing for stable hashes
    s = re.sub(r"[^\w\s\.\,\?\!\:\;\-\(\)\[\]\/]", "", s)
    return s


def text_hash(s: str) -> str:
    return hashlib.sha256(normalize_text(s).encode("utf-8")).hexdigest()


@dataclass
class NoveltyFilter:
    max_sim: float = 0.88
    max_history: int = 5000

    def __post_init__(self) -> None:
        self._hashes: set[str] = set()
        self._texts: list[str] = []

    def seed(self, previous_questions: List[str]) -> None:
        for q in previous_questions[-self.max_history:]:
            self.add(q)

    def add(self, question: str) -> None:
        h = text_hash(question)
        self._hashes.add(h)
        self._texts.append(question)
        if len(self._texts) > self.max_history:
            # drop oldest
            old = self._texts.pop(0)
            self._hashes.discard(text_hash(old))

    def is_novel(self, question: str) -> Tuple[bool, Dict[str, Any]]:
        """
        Returns (is_novel, info).
        info includes hash_dup, max_sim, sim_to_index.
        """
        h = text_hash(question)
        if h in self._hashes:
            return False, {"hash_dup": True, "max_sim": 1.0, "sim_to_index": None}

        if not self._texts:
            return True, {"hash_dup": False, "max_sim": 0.0, "sim_to_index": None}

        # TF-IDF cosine similarity against history (local, deterministic)
        corpus = self._texts + [question]
        vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1).fit_transform(corpus)
        sims = cosine_similarity(vec[-1], vec[:-1]).flatten()
        max_sim = float(np.max(sims)) if sims.size else 0.0
        idx = int(np.argmax(sims)) if sims.size else None

        if max_sim >= self.max_sim:
            return False, {"hash_dup": False, "max_sim": max_sim, "sim_to_index": idx}
        return True, {"hash_dup": False, "max_sim": max_sim, "sim_to_index": idx}


def _safe_json_extract(s: str) -> Optional[Dict[str, Any]]:
    """
    Attempt to extract a JSON object from a model response.
    """
    s = s.strip()
    # if it is already JSON
    try:
        return json.loads(s)
    except Exception:
        pass
    # try to find a JSON block
    m = re.search(r"\{.*\}", s, flags=re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


@dataclass
class QuestionGenerator:
    client: LLMClient
    temperature: float = 0.9
    max_tokens: int = 700
    recent_context: int = 6

    def generate(self,
                 *,
                 topic: Optional[str],
                 difficulty: int,
                 focus_skills: List[str],
                 avoid_recent: List[str]) -> Dict[str, Any]:
        """
        Returns a dict with keys: question, topic, difficulty, skills, constraints.
        """
        avoid_examples = "\n".join([f"- {q}" for q in avoid_recent[-self.recent_context:]]) or "- (none)"
        skill_str = ", ".join(focus_skills) if focus_skills else "mixed reasoning and analysis"
        topic_str = topic or "life sciences, data science, reasoning, and software engineering"

        prompt = f"""
You are a benchmark author. Create ONE novel evaluation question.

Goals:
- Be useful for evaluating a GenAI assistant in professional settings.
- Focus area: {topic_str}
- Target difficulty: {difficulty}/5
- Prefer skills: {skill_str}

Hard constraints:
- The question must be self-contained (no external links required).
- The question must be meaningfully different from the examples below.
- The question should be answerable in ~3-6 minutes by a strong assistant.

Avoid being similar to these recent questions:
{avoid_examples}

Return STRICT JSON with this schema:
{{
  "question": "...",
  "topic": "...",
  "difficulty": 1,
  "skills": ["...","..."],
  "constraints": {{
     "format": "short essay | bullet list | code | json",
     "must_include": ["..."],
     "must_avoid": ["..."]
  }}
}}
""".strip()

        txt = self.client.chat(
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            response_format={"type": "json_object"},
        )
        obj = _safe_json_extract(txt) or {}
        # minimal normalization
        obj.setdefault("topic", topic_str)
        obj.setdefault("difficulty", difficulty)
        obj.setdefault("skills", focus_skills or ["analysis"])
        if "question" not in obj:
            obj["question"] = txt.strip()
        obj.setdefault("constraints", {"format": "short essay", "must_include": [], "must_avoid": []})
        return obj


def generate_novel_question(gen: QuestionGenerator,
                            novelty: NoveltyFilter,
                            *,
                            topic: Optional[str],
                            difficulty: int,
                            focus_skills: List[str],
                            avoid_recent: List[str],
                            max_regen: int = 6) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Generate a novel question with regeneration attempts.
    Returns (question_obj, novelty_info).
    """
    last_info: Dict[str, Any] = {}
    for _ in range(max_regen):
        qobj = gen.generate(topic=topic, difficulty=difficulty, focus_skills=focus_skills, avoid_recent=avoid_recent)
        qtext = str(qobj.get("question", "")).strip()
        ok, info = novelty.is_novel(qtext)
        last_info = info
        if ok:
            novelty.add(qtext)
            return qobj, info
    # if still not novel, return the last and mark it
    qobj = gen.generate(topic=topic, difficulty=difficulty, focus_skills=focus_skills, avoid_recent=avoid_recent)
    return qobj, {"forced": True, **last_info}
