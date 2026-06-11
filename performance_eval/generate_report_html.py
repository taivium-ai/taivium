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
import re
import shutil
import subprocess
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
        "timing_ms": "-",
    }


def _cell_class(value: float, highlight_positive: bool) -> str:
    if value == 0:
        return "neutral"
    if highlight_positive:
        return "good" if value > 0 else "bad"
    return "good" if value < 0 else "bad"


def _delta_cell_class(metric_name: str, delta_value: float | None) -> str:
    """Determine cell class for delta columns based on metric type.
    
    F1 delta: positive is better (higher F1) → good
    Timing delta: negative is better (faster) → good
    """
    if delta_value is None or delta_value == 0:
        return "neutral"
    if metric_name == "f1":
        return "good" if delta_value > 0 else "bad"
    elif metric_name == "timing":
        return "good" if delta_value < 0 else "bad"
    return "neutral"


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


def _render_latency_delta_matrix(latest_report: dict[str, Any]) -> str:
    """Render a model×model latency delta matrix from latest_report timing data.

    Cell (row, col) = timing_ms[col] - timing_ms[row].
    Negative means the column model is faster; positive means slower.
    """
    models = latest_report.get("models", {})
    model_names = [n for n, v in models.items() if isinstance(v.get("timing_ms"), (int, float))]
    if len(model_names) < 2:
        return ""

    header = "".join(f"<th>{html.escape(name)}</th>" for name in model_names)
    rows = []
    for row_name in model_names:
        row_timing = float(models[row_name]["timing_ms"])
        cells = []
        for col_name in model_names:
            col_timing = float(models[col_name]["timing_ms"])
            delta = col_timing - row_timing
            # negative = col is faster → good; positive = col is slower → bad
            if abs(delta) < 1e-9:
                cls = "neutral"
                cell_text = f"{delta:+.4f}"
            else:
                cls = "good" if delta < 0 else "bad"
                pct = (delta / row_timing) * 100 if row_timing != 0 else 0.0
                cell_text = f"{delta:+.4f} ({pct:+.1f}%)"
            cells.append(f'<td class="{cls}">{cell_text}</td>')
        rows.append(
            "<tr>"
            f"<th>{html.escape(row_name)}</th>"
            f"{''.join(cells)}"
            "</tr>"
        )

    return (
        '<section class="panel">'
        "<h3>TIMING Latency Delta Matrix</h3>"
        '<p class="meta">Cell value is Δ ms/sample (column − row) and relative % change. '
        "Negative (green) means column model is faster than row model.</p>"
        '<div class="table-wrap"><table>'
        f"<thead><tr><th>Baseline \\ Compare</th>{header}</tr></thead>"
        f"<tbody>{''.join(rows)}</tbody>"
        "</table></div></section>"
    )


def _trend_plot_filename(dataset: str, profile: str) -> str:
    safe_dataset = dataset.replace("/", "_")
    return f"{safe_dataset}_{profile}_performance_trend.png"


def _label_distribution_filename(dataset: str, profile: str) -> str:
    safe_dataset = dataset.replace("/", "_")
    return f"{safe_dataset}_{profile}_label_distribution.png"


def _copy_trend_plot_if_exists(
    cache_dir: Path, output_file: Path, dataset: str, profile: str
) -> str | None:
    source = cache_dir / _trend_plot_filename(dataset, profile)
    if not source.exists():
        return None

    assets_dir = output_file.parent / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)
    target = assets_dir / source.name
    shutil.copy2(source, target)
    return f"assets/{source.name}"


def _copy_label_distribution_if_exists(
    cache_dir: Path, output_file: Path, dataset: str, profile: str
) -> str | None:
    source = cache_dir / _label_distribution_filename(dataset, profile)
    if not source.exists():
        return None

    assets_dir = output_file.parent / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)
    target = assets_dir / source.name
    shutil.copy2(source, target)
    return f"assets/{source.name}"


def _latest_report_path(cache_dir: Path) -> Path:
    # cache_dir is performance_eval/.cache → parent is performance_eval/ → parent is project root
    project_root = cache_dir.parent.parent
    return project_root / "performance_eval" / "results" / "latest_report.json"


def _load_latest_report(cache_dir: Path) -> dict[str, Any]:
    latest_path = _latest_report_path(cache_dir)
    if not latest_path.exists():
        raise FileNotFoundError(
            f"No latest report found at {latest_path}. Run main_evaluation.py first."
        )
    with open(latest_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Invalid latest report format at {latest_path}")
    return data


def _load_git_report_versions(cache_dir: Path, max_versions: int = 4) -> list[dict[str, Any]]:
    """Load up to `max_versions` committed versions of latest_report.json from git."""
    latest_path = _latest_report_path(cache_dir)
    project_root = cache_dir.parent.parent

    try:
        rel_path = latest_path.relative_to(project_root).as_posix()
    except ValueError:
        return []

    try:
        rev_list_cmd = [
            "git",
            "-C",
            str(project_root),
            "rev-list",
            f"--max-count={max_versions}",
            "HEAD",
            "--",
            rel_path,
        ]
        rev_list = subprocess.run(
            rev_list_cmd,
            check=False,
            capture_output=True,
            text=True,
        )
        if rev_list.returncode != 0:
            return []
        shas = [line.strip() for line in rev_list.stdout.splitlines() if line.strip()]
    except OSError:
        return []

    history: list[dict[str, Any]] = []
    for sha in shas:
        try:
            show_cmd = ["git", "-C", str(project_root), "show", f"{sha}:{rel_path}"]
            show_result = subprocess.run(
                show_cmd,
                check=False,
                capture_output=True,
                text=True,
            )
            if show_result.returncode != 0:
                continue
            item = json.loads(show_result.stdout)
            if isinstance(item, dict):
                history.append(item)
        except (OSError, json.JSONDecodeError):
            continue

    return history


def _load_history(cache_dir: Path) -> list[dict[str, Any]]:
    """Combine current latest report with last 4 committed versions from git."""
    current = _load_latest_report(cache_dir)
    committed = _load_git_report_versions(cache_dir, max_versions=4)

    combined = [current, *committed]
    deduped: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for item in combined:
        key = (
            str(item.get("timestamp", "")),
            str(item.get("dataset", "")),
            str(item.get("profile", "")),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)

    deduped.sort(key=lambda x: str(x.get("timestamp", "")))
    return deduped


def _latest_metric_deltas(history: list[dict[str, Any]], metric: str) -> dict[str, float | None]:
    """Return per-model deltas between latest and previous run for a metric."""
    if len(history) < 2:
        return {}

    previous_models = history[-2].get("models", {})
    current_models = history[-1].get("models", {})
    deltas: dict[str, float | None] = {}
    for model_name, model_values in current_models.items():
        current_value = model_values.get(metric)
        previous_value = previous_models.get(model_name, {}).get(metric)
        if isinstance(current_value, (int, float)) and isinstance(previous_value, (int, float)):
            deltas[model_name] = float(current_value) - float(previous_value)
        else:
            deltas[model_name] = None
    return deltas


def _render_last_runs_table(history: list[dict[str, Any]], model_labels: list[str]) -> str:
    if not history:
        return '<p class="meta">No history file found yet.</p>'

    recent_runs = history[-5:]
    headers = "".join(f"<th>{html.escape(label)}</th>" for label in model_labels)
    rows = []
    for run in recent_runs:
        timestamp = str(run.get("timestamp", "-")).replace("T", " ").replace("Z", " UTC")
        models = run.get("models", {})
        cells = []
        for label in model_labels:
            f1_value = models.get(label, {}).get("f1")
            if isinstance(f1_value, (int, float)):
                cells.append(f"<td>{f1_value:.4f}</td>")
            else:
                cells.append("<td>-</td>")
        rows.append(f"<tr><th>{html.escape(timestamp)}</th>{''.join(cells)}</tr>")

    return (
        '<div class="table-wrap"><table class="trend-mini">'
        f'<thead><tr><th>Run</th>{headers}</tr></thead>'
        f"<tbody>{''.join(rows)}</tbody>"
        "</table></div>"
    )


def _render_trend_panel(
    trend_plot_rel: str | None,
    history: list[dict[str, Any]],
    model_labels: list[str],
) -> str:
    runs_table_html = _render_last_runs_table(history, model_labels)
    if trend_plot_rel:
        return (
            '<section class="panel">'
            '<h2>Performance Trend</h2>'
            '<p class="meta">Historical trend for F1 and timing across runs.</p>'
            '<div class="trend-grid">'
            f'<img src="{html.escape(trend_plot_rel)}" alt="Performance trend chart" '
            'style="width:100%;border:1px solid var(--line);border-radius:10px;" />'
            '<div>'
            '<h3>Last 5 Runs (F1)</h3>'
            f'{runs_table_html}'
            '</div>'
            '</div>'
            '</section>'
        )
    return (
        '<section class="panel">'
        '<h2>Performance Trend</h2>'
        '<p class="meta">Trend chart not found yet. Run main_evaluation.py at least once with trend enabled.</p>'
        '<h3>Last 5 Runs (F1)</h3>'
        f'{runs_table_html}'
        '</section>'
    )


def generate_html(cache_dir: Path, output_file: Path) -> None:
    latest_report = _load_latest_report(cache_dir)
    history = _load_history(cache_dir)

    latest_delta = _find_latest_delta(cache_dir)
    with open(latest_delta, "r", encoding="utf-8") as f:
        delta = json.load(f)

    matrix_keys = [k for k in ("precision", "recall", "f1", "timing_ms") if k in delta]
    if not matrix_keys:
        raise ValueError(f"No matrix sections found in {latest_delta}")

    dataset = str(latest_report.get("dataset", "-"))
    profile = str(latest_report.get("profile", "-"))
    generated = dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    hw = latest_report.get("hardware", {})
    hw_device = str(hw.get("device", ""))
    hw_chip = str(hw.get("chip", ""))
    hw_cores = hw.get("cores")
    hw_parts = [p for p in [hw_device, hw_chip, (f"{hw_cores}-core" if hw_cores else "")] if p]
    hw_str = " · ".join(hw_parts) if hw_parts else ""
    trend_plot_rel = _copy_trend_plot_if_exists(cache_dir, output_file, dataset, profile)

    current_models = latest_report.get("models", {})
    trend_labels = list(current_models.keys())
    f1_deltas = _latest_metric_deltas(history, "f1")
    timing_deltas = _latest_metric_deltas(history, "timing_ms")

    distribution_img_rel = _copy_label_distribution_if_exists(
        cache_dir, output_file, dataset, profile
    )

    summary_rows = []
    for model_label, metrics in current_models.items():
        detector_match = MODEL_LABEL_RE.match(model_label)
        detector = detector_match.group("detector") if detector_match else model_label
        model = detector_match.group("model") if detector_match else "-"

        f1_delta = f1_deltas.get(model_label)
        timing_delta = timing_deltas.get(model_label)
        f1_delta_text = f"{f1_delta:+.4f}" if isinstance(f1_delta, (int, float)) else "-"
        timing_delta_text = (
            f"{timing_delta:+.2f}" if isinstance(timing_delta, (int, float)) else "-"
        )
        f1_class = _delta_cell_class("f1", f1_delta)
        timing_class = _delta_cell_class("timing", timing_delta)

        timing_value = metrics.get("timing_ms")
        timing_text = f"{timing_value:.3f}" if isinstance(timing_value, (int, float)) else "-"

        summary_rows.append(
            "<tr>"
            f"<td>{html.escape(detector)}</td>"
            f"<td>{html.escape(model)}</td>"
            f"<td>{_safe_float(metrics.get('precision'))}</td>"
            f"<td>{_safe_float(metrics.get('recall'))}</td>"
            f"<td>{_safe_float(metrics.get('f1'))}</td>"
            f'<td class="{f1_class}">{html.escape(f1_delta_text)}</td>'
            f"<td>{html.escape(timing_text)}</td>"
            f'<td class="{timing_class}">{html.escape(timing_delta_text)}</td>'
            "</tr>"
        )

    matrices_html = "".join(_render_matrix(key, delta[key]) for key in matrix_keys)
    latency_delta_matrix_html = _render_latency_delta_matrix(latest_report)

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
    .trend-grid {{ display: grid; grid-template-columns: 2fr 1fr; gap: 14px; align-items: start; }}
    .table-wrap {{ overflow: auto; }}
    table {{ width: 100%; border-collapse: collapse; min-width: 920px; }}
    table.trend-mini {{ min-width: 560px; }}
    table.trend-mini th, table.trend-mini td {{ font-size: 12px; padding: 6px 8px; }}
    @media (max-width: 980px) {{ .trend-grid {{ grid-template-columns: 1fr; }} }}
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
      <p class=\"meta\">Generated: {html.escape(generated)} | Dataset: {html.escape(dataset)} | Profile: {html.escape(profile)}</p>      {'<p class="meta">Hardware: ' + html.escape(hw_str) + '</p>' if hw_str else ''}      <p class=\"meta\">Source delta file: {html.escape(latest_delta.name)}</p>
    </section>

    <section class=\"panel\">
      <h2>Latest Summary</h2>
      <div class=\"table-wrap\"><table>
        <thead>
          <tr>
                        <th>Detector</th><th>Model</th><th>Precision</th><th>Recall</th><th>F1</th><th>F1 Δ</th><th>Timing (ms/sample)</th><th>Timing Δ (ms)</th>
          </tr>
        </thead>
        <tbody>
          {''.join(summary_rows)}
        </tbody>
      </table></div>
    </section>

        {_render_trend_panel(trend_plot_rel, history, trend_labels)}
    {f'<section class="panel"><h2>Dataset Distribution</h2><p class="meta">Label distribution for {html.escape(dataset)} ({html.escape(profile)} profile)</p><div style="text-align:center;"><img src="{html.escape(distribution_img_rel)}" alt="Label distribution chart" style="max-width:100%;height:auto;border:1px solid var(--line);border-radius:10px;" /></div></section>' if distribution_img_rel else ''}
    <section class=\"panel\">
      <h2>Comparison Matrices</h2>
      <p class=\"meta\">Cell value is Δ(column - row). For timing, negative values are better (faster).</p>
      <div class=\"legend\"><span class=\"g\">better</span><span class=\"b\">worse</span><span class=\"n\">same</span></div>
    </section>
    {matrices_html}

    <section class=\"panel\">
      <h2>Latency Comparison</h2>
      <p class=\"meta\">Pairwise latency comparison (ms/sample). Derived from this run's timing data.</p>
      <div class=\"legend\"><span class=\"g\">faster</span><span class=\"b\">slower</span><span class=\"n\">same</span></div>
    </section>
    {latency_delta_matrix_html}

    <section class="panel">
      <h2>Methodology</h2>
      <h3>Dataset Label Schema</h3>
      <p class="meta">The {html.escape(profile.title())} profile includes the following entity labels:</p>
      <ul style="font-size: 13px; line-height: 1.6; color: var(--muted);">
        <li><strong>PERSON:</strong> Individual names and personal identifiers</li>
        <li><strong>ORG:</strong> Organization and company names</li>
        <li><strong>LOCATION:</strong> Geographic locations (cities, countries, regions)</li>
        <li><strong>EMAIL:</strong> Email addresses</li>
        <li><strong>PHONE:</strong> Telephone numbers</li>
        <li><strong>API_KEY:</strong> API keys and authentication tokens</li>
        <li><strong>DATE:</strong> Dates and temporal expressions</li>
        <li><strong>IP:</strong> IP addresses (IPv4 and IPv6)</li>
        <li><strong>SOCIALNUMBER:</strong> Social security numbers, credit card numbers, cryptocurrency addresses</li>
        <li><strong>USERNAME:</strong> Usernames and login identifiers</li>
      </ul>
      <h3>Detector Output Mapping</h3>
      <p class="meta"><strong>Taivium:</strong> Uses native entity labels matching the schema directly.</p>
      <p class="meta"><strong>Presidio:</strong> Maps its entity types to the label schema as follows:</p>
      <div class="table-wrap"><table style="font-size: 12px;">
        <thead><tr style="background: #f0f5fc;"><th style="text-align: left;">Presidio Entity Type</th><th style="text-align: left;">Mapped Label</th></tr></thead>
        <tbody>
          <tr><td>PERSON</td><td>PERSON</td></tr>
          <tr><td>ORGANIZATION</td><td>ORG</td></tr>
          <tr><td>LOCATION</td><td>LOCATION</td></tr>
          <tr><td>EMAIL_ADDRESS</td><td>EMAIL</td></tr>
          <tr><td>PHONE_NUMBER</td><td>PHONE</td></tr>
          <tr><td>DATE_TIME</td><td>DATE</td></tr>
          <tr><td>IP_ADDRESS</td><td>IP</td></tr>
          <tr><td>US_SSN, CREDIT_CARD, CRYPTO</td><td>SOCIALNUMBER</td></tr>
          <tr><td>USERNAME</td><td>USERNAME</td></tr>
        </tbody>
      </table></div>
    </section>
  </div>
</body>
</html>
"""

    # Save to web folder
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(page, encoding="utf-8")

    # Also save to .cache folder
    cache_html_path = cache_dir / "latest_report.html"
    cache_html_path.write_text(page, encoding="utf-8")


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