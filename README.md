# Self-Evolving Benchmark Generator

A lightweight, **self-evolving benchmark generator** for GenAI systems.

It continuously:
1) **Generates novel questions** (with de-dup + semantic similarity checks)  
2) Calls a **user-supplied OpenAI API–compatible endpoint** for:
   - question generation
   - answering
   - evaluation (LLM as judge)
3) Tracks performance using an **Exponential Moving Average (EMA)** score  
4) **Evolves** future questions based on observed failure modes


## Features

- **OpenAI-compatible endpoint** support via `base_url` (e.g., OpenAI, Azure-compatible gateways, local OpenAI-compatible servers).
- **Novelty enforcement**
  - exact de-dup (normalized hash)
  - semantic similarity filter (local TF‑IDF cosine)
- **LLM-as-judge evaluation**
  - strict JSON schema
  - subscores + error tags
- **Self-evolution policy**
  - focuses on weak areas (error tags / low subscores)
  - adjusts difficulty based on EMA trend
- **Reproducible logs**
  - `questions.jsonl`, `answers.jsonl`, `evals.jsonl`, `metrics.json`

---

## Quickstart

### 1) Install

```bash
python -m venv .venv
source .venv/bin/activate  # (Windows: .venv\Scripts\activate)
pip install -r requirements.txt
```

### 2) Run a benchmark loop

```bash
python -m bench.run \
  --base-url "https://api.openai.com/v1" \
  --api-key "$OPENAI_API_KEY" \
  --model "gpt-4o-mini" \
  --eval-model "gpt-4o-mini" \
  --n 30
```

If you're using a local OpenAI-compatible server, set `--base-url` accordingly.

---

## CLI Options (most useful)

- `--n` Number of questions to generate
- `--difficulty` Initial difficulty level (1–5)
- `--half-life` EMA half-life (in questions)
- `--topic` Optional topic focus (e.g., "life sciences", "statistics", "RAG")
- `--run-dir` Optional explicit run directory
- `--seed` Random seed for reproducibility
- `--max-sim` Semantic similarity threshold (0–1); lower = stricter novelty

---

## Outputs

A run folder is created under `runs/`:

```
runs/2026-02-06_12-34-56/
  questions.jsonl
  answers.jsonl
  evals.jsonl
  metrics.json
```

---

## How Novelty Works

A question is accepted only if:
1) Its **normalized hash** is not in the history, and
2) Its **semantic similarity** to previous questions is below `--max-sim`

Semantic similarity uses a local TF‑IDF cosine similarity check. This avoids relying on embeddings support from the endpoint.

---

## Evaluation Schema

The evaluator returns strict JSON:

```json
{
  "score": 0.0,
  "subscores": {
    "correctness": 0.0,
    "completeness": 0.0,
    "reasoning_quality": 0.0,
    "format_compliance": 0.0,
    "safety": 1.0
  },
  "error_tags": ["hallucination", "missed_constraint"],
  "feedback": "Short explanation..."
}
```

---

## Self-Evolution Policy (high level)

- Tracks rolling counts of `error_tags` and low subscores.
- Generates future questions weighted toward weak skills.
- Adjusts difficulty up/down based on EMA trend.

---

## Repo Layout

```
self-evolving-bench/
  README.md
  requirements.txt
  bench/
    client.py
    generate.py
    answer.py
    evaluate.py
    evolve.py
    store.py
    ema.py
    run.py
  configs/
    default.yaml
  tests/
    test_ema.py
    test_dedup.py
```

---

## Notes for Interview Discussion

- EMA stabilizes noisy per-question scores.
- Novelty uses both lexical and semantic checks.
- “Self-evolving” is implemented as a simple, defensible curriculum:
  - focus on error tags / weak subscores
  - difficulty ramps with sustained performance

---

## License

MIT
