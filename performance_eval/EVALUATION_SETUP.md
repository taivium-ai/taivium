# Performance Evaluation Setup

## Hardware Environment
- **Machine:** MacBook Pro
- **Processor:** Apple M2 Pro (10-core CPU, 16-core GPU)
- **Memory:** 32GB
- **OS:** macOS

## Dataset & Evaluation
- **Dataset:** ai4privacy/pii-masking-300k (177,677 samples)
- **Profile:** privacy (strict PII detection)
- **Validation Split:** Used for benchmarking

## Performance Metrics
- **Metric:** Span-based Precision, Recall, F1
- **Canonicalization:** Greedy highest-confidence overlapping span selection
- **Confidence Scores:** See engine.py for per-detector thresholds

## Running Evaluations

### Full Evaluation (all backends)
```bash
python3 performance_eval/main_evaluation.py
```
Runtime: ~10-15 minutes (tests gliner and openai-privacy-filter backends for long-text detection)

### Custom Backend Selection
```bash
python3 performance_eval/main_evaluation.py --context-ner-backends gliner
# or
python3 performance_eval/main_evaluation.py --context-ner-backends openai-privacy-filter
# or both:
python3 performance_eval/main_evaluation.py --context-ner-backends gliner,openai-privacy-filter
```

### Cached Evaluation (reuse previous results)
```bash
python3 performance_eval/main_evaluation.py --skip-baselines
```
Runtime: ~2-3 minutes (skips Taivium evaluation, loads cached results from previous runs)

### View Performance Report
Open: `web/index.html`

## Performance History
- **Location:** `performance_eval/.cache/*_performance_history.json`
- **Format:** JSON array of runs with timestamp, metrics, and hardware info
- **Tracking:** History persists across commits (JSON only, no pickles)
- **Trend Chart:** `performance_eval/.cache/*_performance_trend.png` (generated with matplotlib)

## Detectors Evaluated
Taivium adaptive privacy detection engine with configurable long-text backends:
1. **Short-text route** (text < 100 chars): Regex + spaCy NER (fixed)
2. **Long-text route** (text >= 100 chars): Regex + context backend (configurable)

### Available Backends for Long-Text Detection
- **gliner** (default) - GLiNER-based contextual NER
- **openai-privacy-filter** - OpenAI API-based LLM privacy detection

## Notes
- JSON report files and performance history are committed for trend tracking
- Each run appends a new entry to the history JSON with full metadata
- Cache files (`.cache/`) persist results across runs for fast iteration with `--skip-baselines`
