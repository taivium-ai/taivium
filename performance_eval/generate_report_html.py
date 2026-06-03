"""Generate a latest-only performance report HTML page in the web folder.

This script reads the newest delta-matrix artifact from performance_eval/.cache,
collects matching per-model report metadata, and writes a single HTML file that
contains only the latest summary and comparison matrices.
"""

from __future__ import annotations

import argparse
import datetime as dt
import html
import json
import pickle
import re
from pathlib import Path
from typing import Any


MODEL_LEGEND_RE = re.compile(r"^(?P<label>.+?)\s*\[(?P<cache>[^\]]+)\]$")
MODEL_LABEL_RE = re.compile(r"^(?P<detector>.+?)\s*\((?P<model>.+)\)$")


def _find_latest_delta(cache_dir: Path) -> Path:
    candidates = sorted(
        cache_dir.glob("*_delta_matrices.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(
            f"No delta matrix JSON found in {cache_dir}. Run main_evaluation.py first."
        )
    return candidates[0]


def _safe_float(value: Any) -> str:
    if isinstance(value, (int, float)):
        return f"{value:.6f}"
    return "-"


def _extract_total_ms_per_sample(cache_dir: Path, cache_name: str, detector: str, model: str) -> str:
    if not cache_name or cache_name == "no-cache":
        return "-"

    safe_model_name = model.replace("/", "_")
    pkl_candidates = [
        cache_dir / f"{cache_name}__{detector}__{safe_model_name}.pkl",
        cache_dir / f"{cache_name}.pkl",
    ]
    pkl_path = next((p for p in pkl_candidates if p.exists()), None)
    if pkl_path is None:
        return "-"

    try:
        with open(pkl_path, "rb") as f:
            payload = pickle.load(f)
    except (pickle.UnpicklingError, OSError, EOFError):
        return "-"

    total_time = None
    n_samples = None
    if isinstance(payload, tuple) and len(payload) == 4:
        _, _, total_time, n_samples = payload
    if isinstance(total_time, (int, float)) and isinstance(n_samples, int) and n_samples > 0:
        return f"{(total_time / n_samples) * 1000:.3f}"
    return "-"


def _load_report_for_legend_entry(cache_dir: Path, legend_entry: str) -> dict[str, Any]:
    legend_match = MODEL_LEGEND_RE.match(legend_entry)
    if not legend_match:
        return {
            "label": legend_entry,
            "detector": "unknown",
            "model": "unknown",
            "cache_name": "no-cache",
            "metrics": {},
            "dataset": "-",
            "profile": "-",
            "timing_ms": "-",
        }

    label = legend_match.group("label")
    cache_name = legend_match.group("cache")

    label_match = MODEL_LABEL_RE.match(label)
    detector = label_match.group("detector") if label_match else "unknown"
    model = label_match.group("model") if label_match else "unknown"

    metrics: dict[str, Any] = {}
    dataset = "-"
    profile = "-"
    safe_model_name = model.replace("/", "_")
    report_candidates = [
        cache_dir / f"{cache_name}__{detector}__{safe_model_name}_{detector}_report.json",
        cache_dir / f"{cache_name}_{detector}_report.json",
    ]
    report_path = next((p for p in report_candidates if p.exists()), None)
    if report_path is None:
        # Fallback: take newest matching report from this run-cache namespace.
        globbed = sorted(
            cache_dir.glob(f"{cache_name}*_{detector}_report.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        report_path = globbed[0] if globbed else None

    if report_path is not None:
        with open(report_path, "r", encoding="utf-8") as f:
            report = json.load(f)
        metrics = report.get("metrics", {})
        dataset = report.get("dataset", "-")
        profile = report.get("label_profile", "-")

    return {
        "label": label,
        "detector": detector,
        "model": model,
        "cache_name": cache_name,
        "metrics": metrics,
        "dataset": dataset,
        "profile": profile,
        "timing_ms": _extract_total_ms_per_sample(cache_dir, cache_name, detector, model),
    }


def _cell_class(value: float, highlight_positive: bool) -> str:
    if value == 0:
        return "neutral"
    if highlight_positive:
        return "good" if value > 0 else "bad"
    return "good" if value < 0 else "bad"


def _render_matrix(metric_name: str, data: dict[str, Any]) -> str:
    model_names = data.get("model_names", [])
    matrix = data.get("matrix", [])
    highlight_positive = bool(data.get("highlight_positive", True))

    header = "".join(f"<th>{html.escape(name)}</th>" for name in model_names)

    rows = []
    for i, row_name in enumerate(model_names):
        cells = []
        for j, _ in enumerate(model_names):
            value = float(matrix[i][j])
            cls = _cell_class(value, highlight_positive)
            cells.append(f'<td class="{cls}">{value:+.4f}</td>')
        rows.append(
            "<tr>"
            f"<th>{html.escape(row_name)}</th>"
            f"{''.join(cells)}"
            "</tr>"
        )

    return (
        f"<section class=\"panel\">"
        f"<h3>{html.escape(metric_name.upper())} Delta Matrix</h3>"
        "<div class=\"table-wrap\"><table>"
        f"<thead><tr><th>Baseline \\ Compare</th>{header}</tr></thead>"
        f"<tbody>{''.join(rows)}</tbody>"
        "</table></div></section>"
    )


def generate_html(cache_dir: Path, output_file: Path) -> None:
    latest_delta = _find_latest_delta(cache_dir)
    with open(latest_delta, "r", encoding="utf-8") as f:
        delta = json.load(f)

    matrix_keys = [k for k in ("precision", "recall", "f1", "timing_ms") if k in delta]
    if not matrix_keys:
        raise ValueError(f"No matrix sections found in {latest_delta}")

    model_legend = delta[matrix_keys[0]].get("model_legend", [])
    model_rows = [_load_report_for_legend_entry(cache_dir, entry) for entry in model_legend]

    dataset = next((r["dataset"] for r in model_rows if r["dataset"] != "-"), "-")
    profile = next((r["profile"] for r in model_rows if r["profile"] != "-"), "-")
    generated = dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")

    summary_rows = []
    for row in model_rows:
        metrics = row.get("metrics", {})
        summary_rows.append(
            "<tr>"
            f"<td>{html.escape(row['detector'])}</td>"
            f"<td>{html.escape(row['model'])}</td>"
            f"<td>{_safe_float(metrics.get('precision'))}</td>"
            f"<td>{_safe_float(metrics.get('recall'))}</td>"
            f"<td>{_safe_float(metrics.get('f1'))}</td>"
            f"<td>{html.escape(str(row['timing_ms']))}</td>"
            "</tr>"
        )

    matrices_html = "".join(_render_matrix(key, delta[key]) for key in matrix_keys)

    page = f"""<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>Taivium Performance Report</title>
  <style>
    :root {{
      --bg: #f6f8fb;
      --card: #ffffff;
      --ink: #0f1d2e;
      --muted: #52627a;
      --line: #d8e1ee;
      --good: #e7f8ec;
      --bad: #ffe9ec;
      --neutral: #f2f5fa;
      --brand: #0b5fb0;
    }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; font-family: "Segoe UI", "SF Pro Text", sans-serif; background: linear-gradient(180deg, #edf4ff 0%, var(--bg) 240px); color: var(--ink); }}
    .shell {{ max-width: 1280px; margin: 0 auto; padding: 24px; }}
    .hero {{ background: radial-gradient(circle at top right, #cde6ff, #ffffff 65%); border: 1px solid var(--line); border-radius: 14px; padding: 18px 20px; }}
    h1 {{ margin: 0 0 6px; font-size: 28px; letter-spacing: 0.2px; }}
    .meta {{ color: var(--muted); margin: 0; }}
    .panel {{ margin-top: 18px; background: var(--card); border: 1px solid var(--line); border-radius: 12px; padding: 14px; }}
    h2, h3 {{ margin: 0 0 10px; }}
    .table-wrap {{ overflow: auto; }}
    table {{ width: 100%; border-collapse: collapse; min-width: 920px; }}
    th, td {{ border: 1px solid var(--line); padding: 8px 10px; text-align: right; font-variant-numeric: tabular-nums; }}
    th:first-child, td:first-child {{ text-align: left; }}
    thead th {{ background: #f0f5fc; position: sticky; top: 0; z-index: 1; }}
    td.good {{ background: var(--good); }}
    td.bad {{ background: var(--bad); }}
    td.neutral {{ background: var(--neutral); color: #5d6b80; }}
    .legend {{ display: flex; gap: 14px; color: var(--muted); font-size: 13px; }}
    .legend span::before {{ content: ""; display: inline-block; width: 10px; height: 10px; border-radius: 2px; margin-right: 6px; border: 1px solid #c7d3e4; vertical-align: -1px; }}
    .legend .g::before {{ background: var(--good); }}
    .legend .b::before {{ background: var(--bad); }}
    .legend .n::before {{ background: var(--neutral); }}
    a {{ color: var(--brand); text-decoration: none; }}
    a:hover {{ text-decoration: underline; }}
  </style>
</head>
<body>
  <div class=\"shell\">
    <section class=\"hero\">
      <h1>Taivium Latest Performance Report</h1>
      <p class=\"meta\">Generated: {html.escape(generated)} | Dataset: {html.escape(dataset)} | Profile: {html.escape(profile)}</p>
      <p class=\"meta\">Source delta file: {html.escape(latest_delta.name)}</p>
    </section>

    <section class=\"panel\">
      <h2>Latest Summary</h2>
      <div class=\"table-wrap\"><table>
        <thead>
          <tr>
            <th>Detector</th><th>Model</th><th>Precision</th><th>Recall</th><th>F1</th><th>Timing (ms/sample)</th>
          </tr>
        </thead>
        <tbody>
          {''.join(summary_rows)}
        </tbody>
      </table></div>
    </section>

    <section class=\"panel\">
      <h2>Comparison Matrices</h2>
      <p class=\"meta\">Cell value is Δ(column - row). For timing, negative values are better (faster).</p>
      <div class=\"legend\"><span class=\"g\">better</span><span class=\"b\">worse</span><span class=\"n\">same</span></div>
    </section>
    {matrices_html}
  </div>
</body>
</html>
"""

    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(page, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate latest performance report HTML")
    parser.add_argument(
        "--cache-dir",
        default=str(Path(__file__).resolve().parent / ".cache"),
        help="Directory containing evaluation cache artifacts",
    )
    parser.add_argument(
        "--output",
        default=str(Path(__file__).resolve().parents[1] / "web" / "index.html"),
        help="Output HTML path",
    )
    args = parser.parse_args()

    generate_html(Path(args.cache_dir), Path(args.output))


if __name__ == "__main__":
    main()