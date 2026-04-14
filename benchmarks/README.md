# SmartSplit Benchmark

Compares SmartSplit's multi-LLM routing against single-model baselines.

## What it measures

For each prompt in the dataset, the benchmark:
1. Sends it through **SmartSplit** (decompose + route)
2. Sends it directly to each **baseline provider** (Groq, Gemini, Cerebras, OpenRouter, Mistral)
3. Compares latency, token usage, and success rate
4. Scores each response with **LLM-as-judge** (1-10, chain-of-thought rubric)
5. Runs **pairwise comparisons** (SmartSplit vs each baseline — win/draw/loss)

## Methodology

- **Judge selection**: Uses Cerebras as primary judge (avoids self-bias since Groq/Gemini are baselines). Falls back to Gemini → Groq if unavailable.
- **Scoring rubric**: Correctness (0-3), Completeness (0-3), Clarity (0-2), Quality (0-2). Chain-of-thought reasoning before final score.
- **Pairwise comparison**: Direct A/B comparison with position-independent judging.
- **Results breakdown**: By method, by category (mono/multi-domain), and by domain (code, reasoning, translation, etc.)

## Dataset

`dataset.json` contains 33 prompts across 4 categories:

| Category | Count | Examples |
|----------|-------|---------|
| **mono-domain** | 15 | Pure code, math, translation, creative... |
| **multi-domain-simple** | 10 | Code + translation, math + writing... |
| **multi-domain-complex** | 5 | Code + math + writing, search + code + creative... |
| **multi-turn** | 3 | Conversation with context (system + history + follow-up) |

SmartSplit should win on multi-domain prompts (where routing adds value), match baselines on mono-domain, and excel on multi-turn (context preservation).

## Usage

```bash
# 1. Start SmartSplit
smartsplit

# 2. Run the benchmark (in another terminal)
python benchmarks/run_benchmark.py

# 3. With LLM-as-judge scoring (recommended)
python benchmarks/run_benchmark.py --judge

# 4. Choose baselines
python benchmarks/run_benchmark.py --judge --baselines groq,gemini,cerebras,openrouter,mistral
```

## Requirements

- SmartSplit running on `http://127.0.0.1:8420`
- API keys set for baseline providers (`GROQ_API_KEY`, `GEMINI_API_KEY`, `CEREBRAS_API_KEY`, etc.)
- For `--judge`: at least one of `CEREBRAS_API_KEY`, `GEMINI_API_KEY`, or `GROQ_API_KEY`

## Output

Results are saved to `benchmarks/results/` as JSON. Console output:

```
Method          Success    Latency    Tokens   Score
--------------------------------------------------
  smartsplit     100.0%    2340ms       850     7.8
  groq            96.7%    1200ms       620     6.5
  gemini         100.0%    1800ms       740     7.2

By Domain (judge scores):
  Domain          smartsplit         groq       gemini
  -----------------------------------------------
  code                  8.2          7.0          7.5
  reasoning             7.8          6.0          8.0
  translation           8.5          7.0          8.0
  math                  7.5          5.0          8.0

Win Rates (SmartSplit vs baselines):
  smartsplit_vs_groq: 18W / 5D / 7L (60.0%)
  smartsplit_vs_gemini: 15W / 8D / 7L (50.0%)
```

## Adding prompts

Add entries to `dataset.json`:

```json
{
  "id": "unique-id",
  "category": "mono-domain|multi-domain-simple|multi-domain-complex",
  "expected_domains": ["code", "math"],
  "prompt": "Your test prompt here"
}
```
