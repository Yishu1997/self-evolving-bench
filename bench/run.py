from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import random
from typing import Any, Dict, Optional

import yaml
from tqdm import trange
from rich.console import Console
from rich.table import Table

from .client import LLMClient
from .generate import NoveltyFilter, QuestionGenerator, generate_novel_question
from .answer import Answerer
from .evaluate import Evaluator
from .evolve import EvolutionPolicy
from .ema import EMAScore, alpha_from_half_life
from .store import RunStore


def now_run_dir() -> str:
    ts = dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return os.path.join("runs", ts)


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    ap = argparse.ArgumentParser(description="Self-evolving benchmark generator")
    ap.add_argument("--base-url", required=True, help="OpenAI-compatible API base URL, e.g. https://api.openai.com/v1")
    ap.add_argument("--api-key", required=True, help="API key for the endpoint")
    ap.add_argument("--model", required=True, help="Model name for question generation + answering")
    ap.add_argument("--eval-model", default=None, help="Optional different model for evaluation (LLM judge)")
    ap.add_argument("--config", default="configs/default.yaml", help="Path to YAML config")
    ap.add_argument("--n", type=int, default=20, help="Number of questions to run")
    ap.add_argument("--topic", default=None, help="Optional topic focus")
    ap.add_argument("--difficulty", type=int, default=2, help="Initial difficulty 1-5")
    ap.add_argument("--half-life", type=float, default=20.0, help="EMA half-life in number of questions")
    ap.add_argument("--seed", type=int, default=7, help="Random seed")
    ap.add_argument("--run-dir", default=None, help="Optional explicit run output directory")
    ap.add_argument("--max-sim", type=float, default=None, help="Override novelty.max_sim")
    args = ap.parse_args()

    random.seed(args.seed)

    cfg = load_config(args.config)

    # allow CLI override
    if args.max_sim is not None:
        cfg["novelty"]["max_sim"] = float(args.max_sim)

    run_dir = args.run_dir or now_run_dir()
    store = RunStore(run_dir)

    client = LLMClient(base_url=args.base_url, api_key=args.api_key, model=args.model)
    qgen = QuestionGenerator(
        client=client,
        temperature=float(cfg["generation"]["temperature"]),
        max_tokens=int(cfg["generation"]["max_tokens"]),
        recent_context=int(cfg["generation"]["recent_context"]),
    )
    answerer = Answerer(
        client=client,
        temperature=float(cfg["answering"]["temperature"]),
        max_tokens=int(cfg["answering"]["max_tokens"]),
    )
    evaluator = Evaluator(
        client=client,
        temperature=float(cfg["evaluation"]["temperature"]),
        max_tokens=int(cfg["evaluation"]["max_tokens"]),
    )
    novelty = NoveltyFilter(
        max_sim=float(cfg["novelty"]["max_sim"]),
        max_history=int(cfg["novelty"]["max_history"]),
    )
    policy = EvolutionPolicy(
        window=int(cfg["evolution"]["window"]),
        focus_top_k_tags=int(cfg["evolution"]["focus_top_k_tags"]),
        difficulty_min=int(cfg["evolution"]["difficulty_min"]),
        difficulty_max=int(cfg["evolution"]["difficulty_max"]),
    )

    # seed novelty with previous run questions if any
    prev_qs = [q.get("question", "") for q in store.load_history_questions()]
    novelty.seed([q for q in prev_qs if isinstance(q, str)])

    alpha = alpha_from_half_life(args.half_life)
    ema = EMAScore(alpha=alpha)

    console = Console()
    console.print(f"[bold]Run directory:[/bold] {run_dir}")
    console.print(f"[bold]Novelty max_sim:[/bold] {novelty.max_sim}")
    console.print(f"[bold]EMA alpha:[/bold] {alpha:.4f} (half-life={args.half_life})\n")

    difficulty = max(1, min(5, int(args.difficulty)))
    metrics = {
        "base_url": args.base_url,
        "model": args.model,
        "eval_model": args.eval_model or args.model,
        "topic": args.topic,
        "seed": args.seed,
        "ema_alpha": alpha,
        "steps": [],
    }

    for t in trange(args.n, desc="benchmark"):
        eval_hist = store.load_history_evals()
        focus_skills, focus_tags = policy.next_focus(eval_hist)
        avoid_recent = [q.get("question", "") for q in store.load_history_questions()][-12:]

        qobj, novelty_info = generate_novel_question(
            qgen,
            novelty,
            topic=args.topic,
            difficulty=difficulty,
            focus_skills=focus_skills,
            avoid_recent=avoid_recent,
            max_regen=int(cfg["generation"]["max_regen"]),
        )

        qtext = str(qobj.get("question", "")).strip()
        constraints = qobj.get("constraints", {}) or {}

        store.append_question({
            "t": t,
            "difficulty": difficulty,
            "topic": qobj.get("topic"),
            "skills": qobj.get("skills"),
            "constraints": constraints,
            "question": qtext,
            "novelty": novelty_info,
            "focus_skills": focus_skills,
            "focus_tags": focus_tags,
        })

        ans = answerer.answer(qtext, model=args.model)
        store.append_answer({
            "t": t,
            "model": args.model,
            "answer": ans,
        })

        ev = evaluator.evaluate(
            question=qtext,
            answer=ans,
            constraints=constraints,
            eval_model=args.eval_model,
        )
        store.append_eval({
            "t": t,
            "eval_model": args.eval_model or args.model,
            **ev,
        })

        ema_val = ema.update(float(ev.get("score", 0.0)))
        difficulty = policy.adjust_difficulty(difficulty, ema_val)

        step = {
            "t": t,
            "score": float(ev.get("score", 0.0)),
            "ema": float(ema_val),
            "difficulty_next": difficulty,
            "error_tags": ev.get("error_tags", []),
        }
        metrics["steps"].append(step)
        store.save_metrics(metrics)

        # pretty console summary (short)
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("t", justify="right")
        table.add_column("score", justify="right")
        table.add_column("ema", justify="right")
        table.add_column("difficulty_next", justify="right")
        table.add_column("tags", justify="left")
        table.add_row(
            str(t),
            f"{step['score']:.2f}",
            f"{step['ema']:.2f}",
            str(step["difficulty_next"]),
            ", ".join(step["error_tags"][:4]),
        )
        console.print(table)

    console.print("\n[bold green]Done.[/bold green] Logs saved to run directory.")


if __name__ == "__main__":
    main()
