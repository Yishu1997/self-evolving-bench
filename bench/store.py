from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional


def write_jsonl(path: str, rows: Iterable[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def read_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    if not os.path.exists(path):
        return []
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


@dataclass
class RunStore:
    run_dir: str

    def __post_init__(self) -> None:
        os.makedirs(self.run_dir, exist_ok=True)
        self.questions_path = os.path.join(self.run_dir, "questions.jsonl")
        self.answers_path = os.path.join(self.run_dir, "answers.jsonl")
        self.evals_path = os.path.join(self.run_dir, "evals.jsonl")
        self.metrics_path = os.path.join(self.run_dir, "metrics.json")

    def append_question(self, row: Dict[str, Any]) -> None:
        write_jsonl(self.questions_path, [row])

    def append_answer(self, row: Dict[str, Any]) -> None:
        write_jsonl(self.answers_path, [row])

    def append_eval(self, row: Dict[str, Any]) -> None:
        write_jsonl(self.evals_path, [row])

    def load_history_questions(self) -> list[Dict[str, Any]]:
        return list(read_jsonl(self.questions_path))

    def load_history_evals(self) -> list[Dict[str, Any]]:
        return list(read_jsonl(self.evals_path))

    def save_metrics(self, metrics: Dict[str, Any]) -> None:
        with open(self.metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
