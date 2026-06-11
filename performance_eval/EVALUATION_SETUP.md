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

### Full Evaluation (all detectors)
```bash
python3 performance_eval/main_evaluation.py
```
Runtime: ~10-15 minutes

### Fast Iteration (Taivium only, cached baselines)
```bash
python3 performance_eval/main_evaluation.py --skip-baselines
```
Runtime: ~2-3 minutes

### View Performance Report
Open: `web/index.html`

## Performance History
- **Location:** `performance_eval/.cache/*_performance_history.json`
- **Format:** JSON array of runs with timestamp, metrics, and hardware info
- **Tracking:** History persists across commits (JSON only, no pickles)
- **Trend Chart:** `performance_eval/.cache/*_performance_trend.png` (generated with matplotlib)

## Detectors Evaluated
1. **SpaCy** (en_core_web_lg) - NER baseline
2. **Presidio** (default config) - Reference implementation
3. **Taivium** (en_core_web_lg) - Custom privacy detection engine

## Notes
- Pickle files (`.pkl`) are excluded from Git (see `.gitignore`)
- HTML reports and performance history JSON are committed for trend tracking
- Each run appends a new entry to the history JSON with full metadata
