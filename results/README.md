# Results

This directory stores evaluation results.

Run `evaluation/scripts/run_evaluation.py` to populate:

```bash
python evaluation/scripts/run_evaluation.py \
    --dataset evaluation/datasets/factual_queries.jsonl \
    --strategy all \
    --output results/factual_results.json \
    --mock
```

Then compute metrics:

```bash
python evaluation/scripts/compute_metrics.py \
    --results results/factual_results.json
```

And analyze failure modes:

```bash
python evaluation/scripts/analyze_failure_modes.py \
    --results results/factual_results.json
```

## Expected Output Files

| File | Description |
|---|---|
| `factual_results.json` | Raw per-query results for factual dataset |
| `factual_results_metrics.json` | Aggregate metrics per strategy |
| `ambiguous_results.json` | Results for ambiguous queries |
| `adversarial_results.json` | Results for adversarial prompts |
| `missing_knowledge_results.json` | Results for missing knowledge queries |
| `long_context_results.json` | Results for long context queries |
