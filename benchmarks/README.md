# SmartSplit Benchmark

Compares SmartSplit's multi-LLM routing against single-model baselines.

## What it measures

For each prompt in the dataset, the benchmark:
1. Sends it through **SmartSplit** (decompose + route)
2. Sends it directly to each **baseline provider** (Groq, Gemini, Cerebras)
3. Compares latency, token usage, and success rate
4. Optionally scores each response with **LLM-as-judge** (1-10)

## Dataset

`dataset.json` contains 30 prompts across 3 categories:

| Category | Count | Examples |
|----------|-------|---------|
| **mono-domain** | 15 | Pure code, math, translation, creative... |
| **multi-domain-simple** | 8 | Code + translation, math + writing... |
| **multi-domain-complex** | 7 | Code + math + writing, search + code + creative... |
| **multi-turn** | 3 | Conversation with context (system + history + follow-up) |

SmartSplit should win on multi-domain prompts (where routing adds value), match baselines on mono-domain ones, and excel on multi-turn (where context preservation matters).

## Usage

```bash
# 1. Start SmartSplit
smartsplit

# 2. Run the benchmark (in another terminal)
python benchmarks/run_benchmark.py

# 3. With LLM-as-judge scoring
python benchmarks/run_benchmark.py --judge

# 4. Choose baselines
python benchmarks/run_benchmark.py --baselines groq,gemini,cerebras
```

## Requirements

- SmartSplit running on `http://127.0.0.1:8420`
- API keys set for baseline providers (`GROQ_API_KEY`, `GEMINI_API_KEY`, `CEREBRAS_API_KEY`)
- For `--judge`: `GROQ_API_KEY` (uses Groq as the judge LLM)

## Output

Results are saved to `benchmarks/results/` as JSON. Console output:

```
Method          Success    Latency    Tokens   Score
--------------------------------------------------
  smartsplit     100.0%    2340ms       850     7.8
  groq            96.7%    1200ms       620     6.5
  gemini         100.0%    1800ms       740     7.2

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
