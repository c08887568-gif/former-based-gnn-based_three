import argparse
import csv
import html
import json
import math
import re
from collections import defaultdict
from pathlib import Path


DEFAULT_OUTPUT_DIR = "outputs/prediction_html"
DEFAULT_PREDICTION_DIR = "diagnostics/predictions"
DEFAULT_TILE_URL = "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"


def parse_args():
    parser = argparse.ArgumentParser(description="Generate interactive HTML maps from detailed test predictions.")
    parser.add_argument("--prediction_dir", default=DEFAULT_PREDICTION_DIR)
    parser.add_argument("--prediction_csv", nargs="*", default=None)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--run_name", nargs="*", default=None)
    parser.add_argument("--tile_url", default=DEFAULT_TILE_URL)
    parser.add_argument("--max_traces", type=int, default=None)
    return parser.parse_args()


def safe_slug(value):
    value = str(value)
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", value)
    value = value.strip("._-")
    return value or "trace"


def as_float(value, default=0.0):
    try:
        if value in (None, ""):
            return default
        return float(value)
    except ValueError:
        return default


def as_int(value, default=0):
    try:
        if value in (None, ""):
            return default
        return int(float(value))
    except ValueError:
        return default


def as_bool(value):
    if isinstance(value, bool):
        return value
    return str(value).lower() in {"true", "1", "yes", "y"}


def read_prediction_csv(path):
    rows = []
    with Path(path).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows.append(
                {
                    "run_name": row.get("run_name", ""),
                    "trace_id": row.get("trace_id", ""),
                    "sample_index": as_int(row.get("sample_index")),
                    "point_index": as_int(row.get("point_index")),
                    "longitude": as_float(row.get("longitude")),
                    "latitude": as_float(row.get("latitude")),
                    "true_label": as_int(row.get("true_label")),
                    "true_name": row.get("true_name", ""),
                    "pred_label": as_int(row.get("pred_label")),
                    "pred_name": row.get("pred_name", ""),
                    "is_correct": as_bool(row.get("is_correct")),
                    "error_type": row.get("error_type", ""),
                    "prob_road": as_float(row.get("prob_road")),
                    "prob_field": as_float(row.get("prob_field")),
                    "confidence": as_float(row.get("confidence")),
                    "prob_margin": as_float(row.get("prob_margin")),
                }
            )
    return rows


def discover_prediction_files(prediction_dir, prediction_csv):
    if prediction_csv:
        return [Path(path) for path in prediction_csv]
    return sorted(Path(prediction_dir).glob("*_test_predictions_detailed.csv"))


def label_name(label):
    return "road" if int(label) == 0 else "field"


def summarize_points(points):
    total = len(points)
    correct = sum(1 for point in points if point["is_correct"])
    errors = total - correct
    true_road = sum(1 for point in points if point["true_label"] == 0)
    pred_road = sum(1 for point in points if point["pred_label"] == 0)
    true_field = total - true_road
    pred_field = total - pred_road
    accuracy = correct / total if total else 0.0
    return {
        "total": total,
        "correct": correct,
        "errors": errors,
        "accuracy": accuracy,
        "true_road": true_road,
        "true_field": true_field,
        "pred_road": pred_road,
        "pred_field": pred_field,
    }


def fmt_rate(value):
    return f"{value * 100:.2f}%"


def page_shell(title, body, extra_head=""):
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(title)}</title>
  {extra_head}
  <style>
    :root {{
      --bg: #f7f8fb;
      --panel: #ffffff;
      --text: #18202f;
      --muted: #647085;
      --line: #d9dee8;
      --road: #2563eb;
      --field: #16a34a;
      --error: #dc2626;
      --ok: #768196;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}
    a {{ color: #174ea6; text-decoration: none; }}
    a:hover {{ text-decoration: underline; }}
    .page {{
      width: min(1280px, calc(100vw - 32px));
      margin: 0 auto;
      padding: 24px 0 40px;
    }}
    .topbar {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      margin-bottom: 18px;
    }}
    h1 {{
      margin: 0;
      font-size: 24px;
      line-height: 1.2;
      letter-spacing: 0;
    }}
    h2 {{
      margin: 28px 0 12px;
      font-size: 18px;
      letter-spacing: 0;
    }}
    .muted {{ color: var(--muted); }}
    .summary {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
      gap: 10px;
      margin: 14px 0 18px;
    }}
    .metric {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px;
    }}
    .metric .label {{
      color: var(--muted);
      font-size: 12px;
    }}
    .metric .value {{
      margin-top: 4px;
      font-size: 20px;
      font-weight: 650;
    }}
    .toolbar {{
      display: flex;
      gap: 10px;
      align-items: center;
      flex-wrap: wrap;
      margin: 12px 0;
    }}
    input, select {{
      min-height: 36px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #fff;
      color: var(--text);
      padding: 0 10px;
      font-size: 14px;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      overflow: hidden;
    }}
    th, td {{
      padding: 10px 12px;
      border-bottom: 1px solid var(--line);
      text-align: left;
      font-size: 13px;
      vertical-align: top;
    }}
    th {{
      background: #eef2f7;
      color: #334155;
      font-weight: 650;
    }}
    tr:last-child td {{ border-bottom: 0; }}
    .badge {{
      display: inline-flex;
      align-items: center;
      min-height: 22px;
      border-radius: 999px;
      padding: 0 8px;
      font-size: 12px;
      border: 1px solid var(--line);
      background: #fff;
    }}
    .map-page {{
      width: 100vw;
      height: 100vh;
      overflow: hidden;
    }}
    #map {{
      width: 100vw;
      height: 100vh;
    }}
    .map-header {{
      position: fixed;
      z-index: 900;
      top: 12px;
      left: 56px;
      right: 16px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      pointer-events: none;
    }}
    .map-title {{
      max-width: min(780px, calc(100vw - 240px));
      background: rgba(255,255,255,0.94);
      border: 1px solid rgba(217,222,232,0.95);
      border-radius: 8px;
      padding: 8px 10px;
      box-shadow: 0 8px 24px rgba(15, 23, 42, 0.12);
      pointer-events: auto;
    }}
    .map-title strong {{
      display: block;
      font-size: 14px;
      word-break: break-all;
    }}
    .map-title span {{
      display: block;
      color: var(--muted);
      font-size: 12px;
      margin-top: 2px;
    }}
    .back-link {{
      background: rgba(255,255,255,0.94);
      border: 1px solid rgba(217,222,232,0.95);
      border-radius: 8px;
      padding: 8px 10px;
      pointer-events: auto;
      box-shadow: 0 8px 24px rgba(15, 23, 42, 0.12);
      white-space: nowrap;
    }}
    .layer-switch {{
      background: rgba(255,255,255,0.96);
      border: 1px solid rgba(217,222,232,0.95);
      border-radius: 8px;
      padding: 8px;
      box-shadow: 0 8px 24px rgba(15, 23, 42, 0.12);
    }}
    .layer-switch button {{
      display: block;
      width: 82px;
      min-height: 32px;
      margin: 0 0 6px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #fff;
      color: var(--text);
      cursor: pointer;
      font-size: 13px;
    }}
    .layer-switch button:last-child {{ margin-bottom: 0; }}
    .layer-switch button.active {{
      background: #18202f;
      color: #fff;
      border-color: #18202f;
    }}
    .legend {{
      position: fixed;
      z-index: 900;
      right: 16px;
      bottom: 22px;
      background: rgba(255,255,255,0.96);
      border: 1px solid rgba(217,222,232,0.95);
      border-radius: 8px;
      padding: 10px 12px;
      box-shadow: 0 8px 24px rgba(15, 23, 42, 0.12);
      font-size: 12px;
    }}
    .legend-row {{
      display: flex;
      align-items: center;
      gap: 8px;
      margin: 4px 0;
    }}
    .dot {{
      width: 10px;
      height: 10px;
      border-radius: 50%;
      display: inline-block;
    }}
    .tooltip-table {{
      border-collapse: collapse;
      background: transparent;
      border: 0;
      width: auto;
    }}
    .tooltip-table td {{
      border: 0;
      padding: 2px 5px;
      font-size: 12px;
    }}
    .tooltip-table td:first-child {{
      color: #647085;
      white-space: nowrap;
    }}
    @media (max-width: 720px) {{
      .page {{ width: min(100vw - 20px, 1280px); padding-top: 16px; }}
      .topbar {{ align-items: flex-start; flex-direction: column; }}
      .map-header {{ left: 8px; right: 8px; top: 74px; }}
      .map-title {{ max-width: calc(100vw - 16px); }}
      .back-link {{ display: none; }}
    }}
  </style>
</head>
<body>
{body}
</body>
</html>
"""


def experiment_summary(run_name, traces):
    all_points = [point for points in traces.values() for point in points]
    summary = summarize_points(all_points)
    return {
        "run_name": run_name,
        "trace_count": len(traces),
        **summary,
    }


def write_top_index(output_dir, experiments):
    rows = []
    for experiment in experiments:
        rows.append(
            f"""<tr>
  <td><a href="{experiment['slug']}/index.html">{html.escape(experiment['run_name'])}</a></td>
  <td>{experiment['trace_count']}</td>
  <td>{experiment['total']}</td>
  <td>{fmt_rate(experiment['accuracy'])}</td>
  <td>{experiment['errors']}</td>
  <td>{experiment['true_road']} / {experiment['pred_road']}</td>
  <td>{experiment['true_field']} / {experiment['pred_field']}</td>
</tr>"""
        )
    body = f"""<main class="page">
  <div class="topbar">
    <div>
      <h1>测试集轨迹预测 HTML</h1>
      <div class="muted">选择实验后进入轨迹列表。</div>
    </div>
  </div>
  <table>
    <thead>
      <tr>
        <th>实验</th>
        <th>轨迹数</th>
        <th>点数</th>
        <th>准确率</th>
        <th>错点数</th>
        <th>true/pred road</th>
        <th>true/pred field</th>
      </tr>
    </thead>
    <tbody>
      {''.join(rows)}
    </tbody>
  </table>
</main>"""
    (output_dir / "index.html").write_text(page_shell("测试集轨迹预测 HTML", body), encoding="utf-8")


def write_experiment_index(output_dir, experiment, traces):
    trace_rows = []
    trace_options = []
    for trace_id, points in sorted(traces.items()):
        summary = summarize_points(points)
        slug = safe_slug(trace_id)
        rel_link = f"traces/{slug}.html"
        trace_options.append(f'<option value="{html.escape(rel_link)}">{html.escape(trace_id)}</option>')
        trace_rows.append(
            f"""<tr data-trace="{html.escape(trace_id.lower())}">
  <td><a href="{html.escape(rel_link)}">{html.escape(trace_id)}</a></td>
  <td>{summary['total']}</td>
  <td>{fmt_rate(summary['accuracy'])}</td>
  <td>{summary['errors']}</td>
  <td>{summary['true_road']} / {summary['pred_road']}</td>
  <td>{summary['true_field']} / {summary['pred_field']}</td>
</tr>"""
        )
    body = f"""<main class="page">
  <div class="topbar">
    <div>
      <h1>{html.escape(experiment['run_name'])}</h1>
      <div class="muted">选择一条测试集轨迹查看真实图、预测图和错点图。</div>
    </div>
    <a href="../index.html">返回实验列表</a>
  </div>
  <section class="summary">
    <div class="metric"><div class="label">轨迹数</div><div class="value">{experiment['trace_count']}</div></div>
    <div class="metric"><div class="label">点数</div><div class="value">{experiment['total']}</div></div>
    <div class="metric"><div class="label">准确率</div><div class="value">{fmt_rate(experiment['accuracy'])}</div></div>
    <div class="metric"><div class="label">错点数</div><div class="value">{experiment['errors']}</div></div>
  </section>
  <div class="toolbar">
    <input id="traceSearch" type="search" placeholder="搜索轨迹" aria-label="搜索轨迹">
    <select id="traceSelect" aria-label="选择轨迹">
      <option value="">选择轨迹</option>
      {''.join(trace_options)}
    </select>
  </div>
  <table id="traceTable">
    <thead>
      <tr>
        <th>轨迹</th>
        <th>点数</th>
        <th>准确率</th>
        <th>错点数</th>
        <th>true/pred road</th>
        <th>true/pred field</th>
      </tr>
    </thead>
    <tbody>{''.join(trace_rows)}</tbody>
  </table>
</main>
<script>
const search = document.getElementById('traceSearch');
const select = document.getElementById('traceSelect');
const rows = Array.from(document.querySelectorAll('#traceTable tbody tr'));
search.addEventListener('input', () => {{
  const query = search.value.trim().toLowerCase();
  rows.forEach(row => {{
    row.style.display = row.dataset.trace.includes(query) ? '' : 'none';
  }});
}});
select.addEventListener('change', () => {{
  if (select.value) window.location.href = select.value;
}});
</script>"""
    run_dir = output_dir / experiment["slug"]
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "index.html").write_text(page_shell(experiment["run_name"], body), encoding="utf-8")


def point_for_js(point):
    return {
        "lat": point["latitude"],
        "lon": point["longitude"],
        "sample_index": point["sample_index"],
        "point_index": point["point_index"],
        "true_label": point["true_label"],
        "true_name": point["true_name"] or label_name(point["true_label"]),
        "pred_label": point["pred_label"],
        "pred_name": point["pred_name"] or label_name(point["pred_label"]),
        "is_correct": point["is_correct"],
        "error_type": point["error_type"],
        "prob_road": point["prob_road"],
        "prob_field": point["prob_field"],
        "confidence": point["confidence"],
        "prob_margin": point["prob_margin"],
    }


def write_trace_page(output_dir, run_name, run_slug, trace_id, points, tile_url):
    points = sorted(points, key=lambda item: (item["sample_index"], item["point_index"]))
    summary = summarize_points(points)
    lat_values = [point["latitude"] for point in points]
    lon_values = [point["longitude"] for point in points]
    center = [
        sum(lat_values) / len(lat_values) if lat_values else 0,
        sum(lon_values) / len(lon_values) if lon_values else 0,
    ]
    data = [point_for_js(point) for point in points]
    data_json = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    extra_head = """<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>"""
    body = f"""<div class="map-page">
  <div class="map-header">
    <div class="map-title">
      <strong>{html.escape(trace_id)}</strong>
      <span>{html.escape(run_name)} · 点数 {summary['total']} · 准确率 {fmt_rate(summary['accuracy'])} · 错点 {summary['errors']}</span>
    </div>
    <a class="back-link" href="../index.html">返回轨迹列表</a>
  </div>
  <div id="map"></div>
  <div class="legend" id="legend">
    <div class="legend-row"><span class="dot" style="background:#2563eb"></span>road</div>
    <div class="legend-row"><span class="dot" style="background:#16a34a"></span>field</div>
    <div class="legend-row"><span class="dot" style="background:#f97316"></span>道路预测成农田</div>
    <div class="legend-row"><span class="dot" style="background:#a855f7"></span>田地预测成道路</div>
    <div class="legend-row"><span class="dot" style="background:#dc2626"></span>其他错点</div>
  </div>
</div>
<script>
const points = {data_json};
const tileUrl = {json.dumps(tile_url)};
const map = L.map('map', {{ zoomControl: true }}).setView([{center[0]:.8f}, {center[1]:.8f}], 16);
L.tileLayer(tileUrl, {{
  maxZoom: 21,
  attribution: '&copy; OpenStreetMap contributors'
}}).addTo(map);

const colors = {{
  road: '#2563eb',
  field: '#16a34a',
  roadAsField: '#f97316',
  fieldAsRoad: '#a855f7',
  errorOther: '#dc2626',
  ok: '#768196'
}};

function labelColor(label) {{
  return Number(label) === 0 ? colors.road : colors.field;
}}

function errorColor(point) {{
  if (point.error_type === 'road_as_field' || (Number(point.true_label) === 0 && Number(point.pred_label) === 1)) {{
    return colors.roadAsField;
  }}
  if (point.error_type === 'field_as_road' || (Number(point.true_label) === 1 && Number(point.pred_label) === 0)) {{
    return colors.fieldAsRoad;
  }}
  return colors.errorOther;
}}

function pct(value) {{
  return `${{(Number(value) * 100).toFixed(2)}}%`;
}}

function tooltipHtml(point) {{
  return `<table class="tooltip-table">
    <tr><td>sample</td><td>${{point.sample_index}}</td></tr>
    <tr><td>point</td><td>${{point.point_index}}</td></tr>
    <tr><td>lon</td><td>${{Number(point.lon).toFixed(7)}}</td></tr>
    <tr><td>lat</td><td>${{Number(point.lat).toFixed(7)}}</td></tr>
    <tr><td>true</td><td>${{point.true_label}} / ${{point.true_name}}</td></tr>
    <tr><td>pred</td><td>${{point.pred_label}} / ${{point.pred_name}}</td></tr>
    <tr><td>correct</td><td>${{point.is_correct}}</td></tr>
    <tr><td>error</td><td>${{point.error_type || 'none'}}</td></tr>
    <tr><td>prob road</td><td>${{pct(point.prob_road)}}</td></tr>
    <tr><td>prob field</td><td>${{pct(point.prob_field)}}</td></tr>
    <tr><td>confidence</td><td>${{pct(point.confidence)}}</td></tr>
    <tr><td>margin</td><td>${{pct(point.prob_margin)}}</td></tr>
  </table>`;
}}

function makeMarker(point, mode) {{
  let color = colors.ok;
  let radius = 4;
  let fillOpacity = 0.78;
  if (mode === 'true') color = labelColor(point.true_label);
  if (mode === 'pred') color = labelColor(point.pred_label);
  if (mode === 'error') {{
    color = errorColor(point);
    radius = 5;
    fillOpacity = 0.95;
  }}
  const marker = L.circleMarker([point.lat, point.lon], {{
    radius,
    color,
    weight: mode === 'error' ? 2 : 1,
    fillColor: color,
    fillOpacity,
  }});
  marker.bindTooltip(tooltipHtml(point), {{ sticky: true, direction: 'top', opacity: 0.96 }});
  marker.bindPopup(tooltipHtml(point), {{ maxWidth: 320 }});
  return marker;
}}

const path = points.map(point => [point.lat, point.lon]);
const routeStyle = {{ color: '#475569', weight: 2, opacity: 0.45 }};
const layers = {{
  true: L.layerGroup([L.polyline(path, routeStyle), ...points.map(point => makeMarker(point, 'true'))]),
  pred: L.layerGroup([L.polyline(path, routeStyle), ...points.map(point => makeMarker(point, 'pred'))]),
  error: L.layerGroup([
    L.polyline(path, {{ color: '#64748b', weight: 2, opacity: 0.28 }}),
    ...points.filter(point => !point.is_correct).map(point => makeMarker(point, 'error'))
  ]),
}};

let activeLayer = 'true';
layers.true.addTo(map);
if (path.length > 1) map.fitBounds(L.latLngBounds(path), {{ padding: [32, 32] }});

const SwitchControl = L.Control.extend({{
  options: {{ position: 'topleft' }},
  onAdd: function() {{
    const div = L.DomUtil.create('div', 'layer-switch');
    div.innerHTML = `
      <button type="button" data-layer="true" class="active">正确图</button>
      <button type="button" data-layer="pred">预测图</button>
      <button type="button" data-layer="error">错点图</button>
    `;
    L.DomEvent.disableClickPropagation(div);
    div.querySelectorAll('button').forEach(button => {{
      button.addEventListener('click', () => switchLayer(button.dataset.layer, div));
    }});
    return div;
  }}
}});
map.addControl(new SwitchControl());

function switchLayer(name, control) {{
  if (name === activeLayer) return;
  map.removeLayer(layers[activeLayer]);
  layers[name].addTo(map);
  activeLayer = name;
  control.querySelectorAll('button').forEach(button => {{
    button.classList.toggle('active', button.dataset.layer === name);
  }});
}}
</script>"""
    trace_dir = output_dir / run_slug / "traces"
    trace_dir.mkdir(parents=True, exist_ok=True)
    (trace_dir / f"{safe_slug(trace_id)}.html").write_text(
        page_shell(f"{run_name} - {trace_id}", body, extra_head=extra_head),
        encoding="utf-8",
    )


def group_predictions(files, run_filter):
    grouped = defaultdict(lambda: defaultdict(list))
    for path in files:
        for point in read_prediction_csv(path):
            run_name = point["run_name"] or Path(path).stem.replace("_test_predictions_detailed", "")
            if run_filter and run_name not in run_filter:
                continue
            grouped[run_name][point["trace_id"]].append(point)
    return grouped


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    files = discover_prediction_files(args.prediction_dir, args.prediction_csv)
    grouped = group_predictions(files, set(args.run_name or []))
    experiments = []
    for run_name, traces in sorted(grouped.items()):
        if args.max_traces is not None:
            traces = dict(list(sorted(traces.items()))[: args.max_traces])
        slug = safe_slug(run_name)
        experiment = experiment_summary(run_name, traces)
        experiment["slug"] = slug
        experiments.append(experiment)
        write_experiment_index(output_dir, experiment, traces)
        for trace_id, points in sorted(traces.items()):
            write_trace_page(output_dir, run_name, slug, trace_id, points, args.tile_url)
    write_top_index(output_dir, experiments)
    print(output_dir / "index.html")
    for experiment in experiments:
        print(output_dir / experiment["slug"] / "index.html")


if __name__ == "__main__":
    main()
