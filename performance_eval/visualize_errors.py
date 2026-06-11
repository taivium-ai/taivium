"""Generate an interactive HTML visualization of detection errors."""

import json
import argparse
from pathlib import Path
from typing import Optional


def load_errors(error_json_path: Path) -> dict:
    """Load error data from JSON file."""
    with open(error_json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def highlight_spans(text: str, false_positives: list, false_negatives: list) -> str:
    """
    Highlight false positives and false negatives in text.
    
    Args:
        text: The original text
        false_positives: List of [start, end, label] for FP
        false_negatives: List of [start, end, label] for FN
    
    Returns:
        HTML-escaped text with span highlights
    """
    # Create a list of (start, end, label, is_fp)
    spans = []
    for start, end, label in false_positives:
        spans.append((start, end, label, True, False))  # is_fp=True, is_fn=False
    for start, end, label in false_negatives:
        spans.append((start, end, label, False, True))  # is_fp=False, is_fn=True
    
    # Sort by start position
    spans.sort(key=lambda x: x[0])
    
    # Handle overlapping spans by merging them
    if not spans:
        return escape_html(text)
    
    result = []
    last_end = 0
    i = 0
    
    while i < len(spans):
        start, end, label, is_fp, is_fn = spans[i]
        
        # Add text before this span
        if start > last_end:
            result.append(escape_html(text[last_end:start]))
        
        # Check for overlapping spans at this position
        overlapping = [spans[i]]
        j = i + 1
        while j < len(spans) and spans[j][0] < end:
            if spans[j][1] > end:
                end = spans[j][1]
            overlapping.append(spans[j])
            j += 1
        
        # Determine CSS classes for overlapping spans
        has_fp = any(span[3] for span in overlapping)
        has_fn = any(span[4] for span in overlapping)
        
        css_class = "error-span"
        if has_fp and has_fn:
            css_class = "error-span both"
            title = f"FP+FN: {text[start:end]}"
        elif has_fp:
            css_class = "error-span fp"
            title = f"False Positive: {text[start:end]}"
        elif has_fn:
            css_class = "error-span fn"
            title = f"False Negative: {text[start:end]}"
        
        result.append(f'<span class="{css_class}" title="{title}">')
        result.append(escape_html(text[start:end]))
        result.append('</span>')
        
        last_end = end
        i = j
    
    # Add remaining text
    if last_end < len(text):
        result.append(escape_html(text[last_end:]))
    
    return ''.join(result)


def escape_html(text: str) -> str:
    """Escape HTML special characters."""
    return (text
            .replace('&', '&amp;')
            .replace('<', '&lt;')
            .replace('>', '&gt;')
            .replace('"', '&quot;')
            .replace("'", '&#39;')
            .replace('\n', '<br>'))


def generate_html(error_data: dict, output_path: Path) -> None:
    """Generate interactive HTML visualization."""
    errors = error_data.get('errors', [])
    
    html_parts = [
        """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Taivium Detection Errors - Visualization</title>
    <style>
        * {
            box-sizing: border-box;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            line-height: 1.6;
            color: #333;
            background: #f5f5f5;
            margin: 0;
            padding: 0;
        }
        
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            text-align: center;
        }
        
        .header h1 {
            margin: 0 0 10px 0;
            font-size: 28px;
        }
        
        .header p {
            margin: 5px 0;
            opacity: 0.95;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }
        
        .controls {
            background: white;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            display: flex;
            gap: 10px;
            align-items: center;
            flex-wrap: wrap;
        }
        
        .controls input {
            padding: 8px 12px;
            border: 1px solid #ddd;
            border-radius: 4px;
            width: 200px;
        }
        
        .controls button {
            padding: 8px 16px;
            background: #667eea;
            color: white;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-size: 14px;
        }
        
        .controls button:hover {
            background: #5568d3;
        }
        
        .stats {
            display: flex;
            gap: 20px;
            margin-bottom: 20px;
            font-size: 14px;
        }
        
        .stat-item {
            background: white;
            padding: 10px 15px;
            border-radius: 4px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        
        .stat-label {
            color: #666;
            font-size: 12px;
            text-transform: uppercase;
        }
        
        .stat-value {
            font-size: 18px;
            font-weight: bold;
            color: #333;
        }
        
        .error-card {
            background: white;
            border-left: 4px solid #ddd;
            padding: 15px;
            margin-bottom: 15px;
            border-radius: 4px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        
        .error-card.has-errors {
            border-left-color: #ff6b6b;
        }
        
        .error-index {
            font-weight: bold;
            color: #667eea;
            margin-bottom: 8px;
            font-size: 14px;
        }
        
        .error-text {
            background: #f9f9f9;
            padding: 12px;
            border-radius: 4px;
            font-family: "Monaco", "Courier New", monospace;
            font-size: 13px;
            line-height: 1.5;
            word-wrap: break-word;
            max-height: 200px;
            overflow-y: auto;
        }
        
        .error-summary {
            display: flex;
            gap: 20px;
            margin-top: 10px;
            font-size: 13px;
        }
        
        .error-span {
            background: #fff3cd;
            padding: 2px 4px;
            border-radius: 2px;
            cursor: help;
            position: relative;
        }
        
        .error-span.fp {
            background: #f8d7da;
            color: #721c24;
            border: 1px dashed #f5c6cb;
        }
        
        .error-span.fn {
            background: #d1ecf1;
            color: #0c5460;
            border: 1px dashed #bee5eb;
        }
        
        .error-span.both {
            background: linear-gradient(90deg, #f8d7da 50%, #d1ecf1 50%);
            color: #333;
            border: 1px dashed #999;
        }
        
        .error-span .label {
            font-size: 10px;
            font-weight: bold;
            margin-left: 2px;
            vertical-align: super;
            opacity: 0.8;
        }
        
        .legend {
            background: white;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 20px;
            display: flex;
            gap: 30px;
            flex-wrap: wrap;
        }
        
        .legend-item {
            display: flex;
            align-items: center;
            gap: 8px;
        }
        
        .legend-box {
            width: 30px;
            height: 30px;
            border-radius: 4px;
            border: 1px dashed #999;
        }
        
        .legend-box.fp {
            background: #f8d7da;
            border-color: #f5c6cb;
        }
        
        .legend-box.fn {
            background: #d1ecf1;
            border-color: #bee5eb;
        }
        
        .legend-box.both {
            background: linear-gradient(90deg, #f8d7da 50%, #d1ecf1 50%);
        }
        
        .pagination {
            display: flex;
            gap: 5px;
            justify-content: center;
            margin-top: 20px;
            flex-wrap: wrap;
        }
        
        .pagination button {
            padding: 6px 10px;
            background: #f0f0f0;
            color: #333;
            border: 1px solid #ddd;
            border-radius: 4px;
            cursor: pointer;
            font-size: 12px;
        }
        
        .pagination button:hover {
            background: #e0e0e0;
        }
        
        .pagination button.active {
            background: #667eea;
            color: white;
            border-color: #667eea;
        }
        
        .pagination button:disabled {
            opacity: 0.5;
            cursor: not-allowed;
        }
        
        .footer {
            text-align: center;
            color: #666;
            font-size: 12px;
            margin-top: 30px;
            padding: 20px;
            border-top: 1px solid #eee;
        }
        
        .hidden {
            display: none;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>Taivium Detection Errors - Visualization</h1>
        <p>Interactive error analysis with highlighted false positives and false negatives</p>
    </div>
    
    <div class="container">
        <div class="controls">
            <input type="text" id="searchInput" placeholder="Search by index or text...">
            <button onclick="filterErrors()">Filter</button>
            <button onclick="resetFilter()">Reset</button>
        </div>
        
        <div class="stats">
            <div class="stat-item">
                <div class="stat-label">Total Errors</div>
                <div class="stat-value">""" + str(len(errors)) + """</div>
            </div>
            <div class="stat-item">
                <div class="stat-label">Total False Positives</div>
                <div class="stat-value">""" + str(sum(len(e.get('false_positives', [])) for e in errors)) + """</div>
            </div>
            <div class="stat-item">
                <div class="stat-label">Total False Negatives</div>
                <div class="stat-value">""" + str(sum(len(e.get('false_negatives', [])) for e in errors)) + """</div>
            </div>
        </div>
        
        <div class="legend">
            <div class="legend-item">
                <div class="legend-box fp"></div>
                <span><strong>False Positive</strong> - Detector found but shouldn't have</span>
            </div>
            <div class="legend-item">
                <div class="legend-box fn"></div>
                <span><strong>False Negative</strong> - Detector missed (ground truth)</span>
            </div>
            <div class="legend-item">
                <div class="legend-box both"></div>
                <span><strong>Boundary Overlap</strong> - Both FP and FN in same region</span>
            </div>
        </div>
        
        <div id="errorsContainer"></div>
        
        <div class="pagination" id="pagination"></div>
        
        <div class="footer">
            Generated for Taivium PII detection evaluation on ai4privacy/pii-masking-300k dataset
        </div>
    </div>
    
    <script>
        const allErrors = """ + json.dumps(errors) + """;
        let currentPage = 1;
        const errorsPerPage = 10;
        let filteredErrors = allErrors;
        
        function renderErrors() {
            const container = document.getElementById('errorsContainer');
            const start = (currentPage - 1) * errorsPerPage;
            const end = start + errorsPerPage;
            const pageErrors = filteredErrors.slice(start, end);
            
            container.innerHTML = '';
            
            pageErrors.forEach(error => {
                const hasErrors = error.false_positives.length > 0 || error.false_negatives.length > 0;
                const card = document.createElement('div');
                card.className = 'error-card' + (hasErrors ? ' has-errors' : '');
                
                const fpCount = error.false_positives.length;
                const fnCount = error.false_negatives.length;
                
                // Group by label
                const fpByLabel = {};
                const fnByLabel = {};
                
                error.false_positives.forEach(([start, end, label]) => {
                    fpByLabel[label] = (fpByLabel[label] || 0) + 1;
                });
                
                error.false_negatives.forEach(([start, end, label]) => {
                    fnByLabel[label] = (fnByLabel[label] || 0) + 1;
                });
                
                const allLabels = new Set([...Object.keys(fpByLabel), ...Object.keys(fnByLabel)]);
                const labelBreakdown = Array.from(allLabels)
                    .map(label => {
                        const fp = fpByLabel[label] || 0;
                        const fn = fnByLabel[label] || 0;
                        return `${label} (FP:${fp} FN:${fn})`;
                    })
                    .join(' | ');
                
                const highlightedText = highlightSpans(
                    error.text,
                    error.false_positives,
                    error.false_negatives
                );
                
                card.innerHTML = `
                    <div class="error-index">Error #${error.index}</div>
                    <div class="error-text">${highlightedText}</div>
                    <div class="error-summary">
                        <span style="color: #721c24;"><strong>FP:</strong> ${fpCount}</span>
                        <span style="color: #0c5460;"><strong>FN:</strong> ${fnCount}</span>
                        <span style="color: #555; margin-left: auto;"><strong>Labels:</strong> ${labelBreakdown}</span>
                    </div>
                `;
                
                container.appendChild(card);
            });
            
            renderPagination();
        }
        
        function renderPagination() {
            const pageCount = Math.ceil(filteredErrors.length / errorsPerPage);
            const pagination = document.getElementById('pagination');
            pagination.innerHTML = '';
            
            for (let i = 1; i <= pageCount; i++) {
                const btn = document.createElement('button');
                btn.textContent = i;
                btn.className = i === currentPage ? 'active' : '';
                btn.onclick = () => {
                    currentPage = i;
                    renderErrors();
                    window.scrollTo(0, 0);
                };
                pagination.appendChild(btn);
            }
        }
        
        function highlightSpans(text, fps, fns) {
            const spans = [];
            fps.forEach(([start, end, label]) => {
                spans.push({start, end, label, type: 'fp'});
            });
            fns.forEach(([start, end, label]) => {
                spans.push({start, end, label, type: 'fn'});
            });
            
            spans.sort((a, b) => a.start - b.start);
            
            let result = '';
            let lastEnd = 0;
            
            spans.forEach((span, idx) => {
                if (span.start > lastEnd) {
                    result += escapeHtml(text.substring(lastEnd, span.start));
                }
                
                const className = span.type === 'fp' ? 'fp' : 'fn';
                const typeLabel = span.type === 'fp' ? 'FP' : 'FN';
                const title = `${typeLabel}: ${span.label} - "${text.substring(span.start, span.end)}"`;
                
                result += `<span class="error-span ${className}" title="${title}">`;
                result += escapeHtml(text.substring(span.start, span.end));
                result += `<span class="label">${span.label}</span>`;
                result += '</span>';
                
                lastEnd = span.end;
            });
            
            if (lastEnd < text.length) {
                result += escapeHtml(text.substring(lastEnd));
            }
            
            return result;
        }
        
        function escapeHtml(text) {
            const div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        }
        
        function filterErrors() {
            const query = document.getElementById('searchInput').value.toLowerCase();
            
            if (!query) {
                filteredErrors = allErrors;
            } else {
                filteredErrors = allErrors.filter(error => {
                    const indexMatch = error.index.toString().includes(query);
                    const textMatch = error.text.toLowerCase().includes(query);
                    return indexMatch || textMatch;
                });
            }
            
            currentPage = 1;
            renderErrors();
        }
        
        function resetFilter() {
            document.getElementById('searchInput').value = '';
            filteredErrors = allErrors;
            currentPage = 1;
            renderErrors();
        }
        
        // Initial render
        renderErrors();
    </script>
</body>
</html>
""",
    ]
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(''.join(html_parts))
    
    print(f"✅ Generated HTML visualization: {output_path}")
    print(f"   Total errors: {len(errors)}")
    print(f"   Total FPs: {sum(len(e.get('false_positives', [])) for e in errors)}")
    print(f"   Total FNs: {sum(len(e.get('false_negatives', [])) for e in errors)}")


def main():
    parser = argparse.ArgumentParser(
        description='Generate interactive HTML visualization of detection errors'
    )
    parser.add_argument(
        '--input',
        type=Path,
        default=Path('performance_eval/.cache/latest_taivium_detection_errors.json'),
        help='Path to error JSON file'
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('performance_eval/.cache/latest_taivium_detection_errors.html'),
        help='Path to output HTML file'
    )
    
    args = parser.parse_args()
    
    if not args.input.exists():
        print(f"❌ Error file not found: {args.input}")
        return 1
    
    print(f"📖 Loading errors from: {args.input}")
    error_data = load_errors(args.input)
    
    print(f"✨ Generating HTML visualization...")
    generate_html(error_data, args.output)
    
    return 0


if __name__ == '__main__':
    exit(main())
