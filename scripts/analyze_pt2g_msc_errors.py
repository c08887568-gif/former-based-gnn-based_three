import argparse
import csv
import json
import math
import os
import re
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

RUN_NAME = "PT2G_MSC_v1_finetune_40ep"
SHORT_NAME = "PT2G_MSC_v1"
LABEL_NAMES = {0: "road", 1: "field"}
FEATURE_COLUMNS = [f"feature_{idx:02d}" for idx in range(43)]


def parse_args():
    parser = argparse.ArgumentParser(description="Detailed PT2G_MSC_v1 error analysis.")
    parser.add_argument("--run_name", default=RUN_NAME)
    parser.add_argument("--short_name", default=SHORT_NAME)
    parser.add_argument("--run_dir", default=f"runs/{RUN_NAME}")
    parser.add_argument("--cache_dir", default="cache/wheat_non_iid")
    parser.add_argument("--graph_cache_path", default="cache/pretrained_graphs/PT2G_topk3")
    parser.add_argument("--output_dir", default=f"diagnostics/{SHORT_NAME}_error_analysis")
    parser.add_argument("--summary_csv", default=f"results/{SHORT_NAME}_error_summary.csv")
    parser.add_argument("--report_path", default=f"analysis/{SHORT_NAME}_detailed_error_analysis_report.md")
    parser.add_argument("--pack_path", default=f"analysis_packs/{SHORT_NAME}_detailed_error_analysis_for_chatgpt.zip")
    parser.add_argument("--html_dir", default=f"outputs/prediction_html/{SHORT_NAME}")
    parser.add_argument("--baseline_error_dir", default="diagnostics/pt2g_finetune_40ep_error_analysis")
    parser.add_argument("--baseline_test_predictions", default="diagnostics/predictions/PT2G_finetune_40ep_test_predictions_detailed.csv")
    parser.add_argument("--baseline_summary", default="results/ptg_finetune_40ep_test_summary.csv")
    parser.add_argument("--max_len", type=int, default=1000)
    parser.add_argument("--near_boundary", type=int, default=20)
    parser.add_argument("--force_eval", action="store_true")
    return parser.parse_args()


def as_float(value, default=0.0):
    try:
        if value in (None, ""):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def as_int(value, default=0):
    try:
        if value in (None, ""):
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def read_csv(path):
    path = Path(path)
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, fieldnames, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def safe_slug(value):
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("._-")
    return value or "trace"


def label_name(label):
    return LABEL_NAMES.get(int(label), str(label))


def entropy_binary(prob_road, prob_field):
    total = 0.0
    for value in (prob_road, prob_field):
        value = max(float(value), 1e-12)
        total -= value * math.log(value)
    return total


def haversine_m(lon1, lat1, lon2, lat2):
    radius = 6371000.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
    return 2.0 * radius * math.atan2(math.sqrt(a), math.sqrt(max(1.0 - a, 0.0)))


def lonlat_to_xy_m(lon, lat, ref_lat):
    x = math.radians(lon) * 6371000.0 * math.cos(math.radians(ref_lat))
    y = math.radians(lat) * 6371000.0
    return x, y


def turn_angle_deg(prev_coord, coord, next_coord):
    lon0, lat0 = prev_coord
    lon1, lat1 = coord
    lon2, lat2 = next_coord
    ref_lat = (lat0 + lat1 + lat2) / 3.0
    x0, y0 = lonlat_to_xy_m(lon0, lat0, ref_lat)
    x1, y1 = lonlat_to_xy_m(lon1, lat1, ref_lat)
    x2, y2 = lonlat_to_xy_m(lon2, lat2, ref_lat)
    v1 = np.array([x1 - x0, y1 - y0], dtype=np.float64)
    v2 = np.array([x2 - x1, y2 - y1], dtype=np.float64)
    n1 = float(np.linalg.norm(v1))
    n2 = float(np.linalg.norm(v2))
    if n1 < 1e-9 or n2 < 1e-9:
        return 0.0
    cosine = float(np.dot(v1, v2) / (n1 * n2))
    cosine = max(-1.0, min(1.0, cosine))
    return math.degrees(math.acos(cosine))


def quantile(values, q):
    values = sorted(as_float(value) for value in values if value not in (None, ""))
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    pos = (len(values) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return values[lo]
    frac = pos - lo
    return values[lo] * (1.0 - frac) + values[hi] * frac


def mean(values):
    values = [as_float(value) for value in values if value not in (None, "")]
    return sum(values) / len(values) if values else 0.0


def std(values):
    values = [as_float(value) for value in values if value not in (None, "")]
    if len(values) < 2:
        return 0.0
    return float(np.std(values, ddof=0))


def cohen_d(a, b):
    a = np.asarray([as_float(value) for value in a if value not in (None, "")], dtype=np.float64)
    b = np.asarray([as_float(value) for value in b if value not in (None, "")], dtype=np.float64)
    if len(a) == 0 or len(b) == 0:
        return 0.0
    pooled = math.sqrt((float(np.var(a)) + float(np.var(b))) / 2.0)
    if pooled < 1e-12:
        return 0.0
    return float((np.mean(b) - np.mean(a)) / pooled)


def prediction_paths(output_dir):
    output_dir = Path(output_dir)
    return {
        "valid": output_dir / "valid_point_predictions_with_features.csv",
        "test": output_dir / "test_point_predictions_with_features.csv",
    }


def generate_predictions_if_needed(args):
    paths = prediction_paths(args.output_dir)
    if not args.force_eval and all(path.exists() for path in paths.values()):
        return

    import torch
    from fine_tune import build_data, build_model, load_graph_cache_split, merge_graph_cache_edges
    from utils.utils import get_default_device

    config = dict(
        effective_pretrained_path=None,
        pretrain_mode="edge_weight",
        segment_context_mode="msc",
    )
    device = get_default_device()
    model = build_model(config, device)
    state_path = Path(args.run_dir) / "best_model.pt"
    if not state_path.exists():
        missing = [str(path) for path in paths.values() if not path.exists()]
        raise FileNotFoundError(f"MSC_PREDICTIONS_MISSING_AND_CHECKPOINT_NOT_FOUND: missing={missing} checkpoint={state_path}")
    model.load_state_dict(torch.load(state_path, map_location=device))
    model.eval()

    graph_caches = {
        split: load_graph_cache_split(args.graph_cache_path, split, required=True)
        for split in ("valid", "test")
    }

    for split in ("valid", "test"):
        out_path = paths[split]
        out_path.parent.mkdir(parents=True, exist_ok=True)
        items = torch.load(Path(args.cache_dir) / f"{split}.pt", map_location="cpu")
        trace_crop_counts = defaultdict(int)
        rows = []
        with torch.no_grad():
            for sample_index, item in enumerate(items):
                points = item["points"].clone().detach().to(torch.float32).to(device)
                labels = item["labels"].clone().detach().to(torch.int64).squeeze().to(device)
                edge_index_raw = item["edge_index"]
                trace_id = str(item["trace_id"])
                coordinates = item.get("coordinates")
                if coordinates is None:
                    coordinates = torch.zeros((points.shape[0], 2), dtype=torch.float32)
                coordinates = coordinates.clone().detach().cpu().to(torch.float32)
                crop_index = trace_crop_counts[trace_id]
                trace_crop_counts[trace_id] += 1
                edge_index, edge_weight = merge_graph_cache_edges(
                    edge_index_raw,
                    graph_caches[split],
                    trace_id,
                    points,
                    device,
                    audit_path=None,
                    run_name=args.run_name,
                    split=split,
                    epoch=None,
                    batch_id=sample_index,
                )
                data = build_data(points, edge_index, labels, edge_weight=edge_weight)
                logits = model.test_step(data)
                probs = torch.softmax(logits, dim=1).detach().cpu().numpy()
                preds = np.argmax(probs, axis=1)
                labels_cpu = labels.detach().cpu().numpy()
                points_cpu = points.detach().cpu().numpy()
                coords_cpu = coordinates.numpy()
                for point_index in range(points_cpu.shape[0]):
                    true_label = int(labels_cpu[point_index])
                    pred_label = int(preds[point_index])
                    prob_road = float(probs[point_index, 0])
                    prob_field = float(probs[point_index, 1])
                    confidence = max(prob_road, prob_field)
                    if true_label == pred_label:
                        error_type = "correct"
                    elif true_label == 0 and pred_label == 1:
                        error_type = "road_as_field"
                    else:
                        error_type = "field_as_road"
                    row = dict(
                        split=split,
                        trace_id=trace_id,
                        sample_index=sample_index,
                        point_index=point_index,
                        global_index=crop_index * args.max_len + point_index,
                        crop_index=crop_index,
                        position_in_crop=point_index,
                        true_label=true_label,
                        pred_label=pred_label,
                        prob_road=prob_road,
                        prob_field=prob_field,
                        confidence=confidence,
                        entropy=entropy_binary(prob_road, prob_field),
                        margin=abs(prob_road - prob_field),
                        error_type=error_type,
                        is_error=error_type != "correct",
                        is_high_conf_error=(error_type != "correct" and confidence >= 0.8),
                        longitude=float(coords_cpu[point_index, 0]),
                        latitude=float(coords_cpu[point_index, 1]),
                    )
                    for feature_idx in range(43):
                        row[f"feature_{feature_idx:02d}"] = float(points_cpu[point_index, feature_idx])
                    rows.append(row)
        add_geometry_features(rows, args.max_len, args.near_boundary)
        fields = point_prediction_fields()
        write_csv(out_path, fields, rows)
        print(f"wrote {out_path} rows={len(rows)}")


def point_prediction_fields():
    return [
        "split",
        "trace_id",
        "sample_index",
        "point_index",
        "global_index",
        "crop_index",
        "position_in_crop",
        "true_label",
        "pred_label",
        "prob_road",
        "prob_field",
        "confidence",
        "entropy",
        "margin",
        "error_type",
        "is_error",
        "is_high_conf_error",
        "longitude",
        "latitude",
        "prev_step_distance_m",
        "next_step_distance_m",
        "local_step_mean_m",
        "local_step_std_m",
        "local_turn_angle_deg",
        "local_density_1m",
        "local_density_2m",
        "stationary_flag",
        "stationary_run_length",
        "trace_position_ratio",
        "near_crop_boundary_flag",
        "distance_to_crop_boundary",
        "near_trace_start_end_flag",
        *FEATURE_COLUMNS,
    ]


def add_geometry_features(rows, max_len=1000, near_boundary=20):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["split"], row["trace_id"])].append(row)
    for (_split, _trace_id), trace_rows in grouped.items():
        trace_rows.sort(key=lambda row: (as_int(row.get("global_index")), as_int(row.get("point_index"))))
        n = len(trace_rows)
        coords = [(as_float(row.get("longitude")), as_float(row.get("latitude"))) for row in trace_rows]
        prev_steps = [0.0] * n
        next_steps = [0.0] * n
        for idx in range(n):
            if idx > 0:
                prev_steps[idx] = haversine_m(*coords[idx - 1], *coords[idx])
            if idx < n - 1:
                next_steps[idx] = haversine_m(*coords[idx], *coords[idx + 1])
        stationary_flags = []
        for idx, row in enumerate(trace_rows):
            lo = max(1, idx - 2)
            hi = min(n - 1, idx + 2)
            local_steps = [haversine_m(*coords[step_idx - 1], *coords[step_idx]) for step_idx in range(lo, hi + 1)]
            turn = 0.0
            if 0 < idx < n - 1:
                turn = turn_angle_deg(coords[idx - 1], coords[idx], coords[idx + 1])
            density_1m = 0
            density_2m = 0
            for other_idx in range(max(0, idx - 5), min(n, idx + 6)):
                if other_idx == idx:
                    continue
                distance = haversine_m(*coords[idx], *coords[other_idx])
                if distance <= 1.0:
                    density_1m += 1
                if distance <= 2.0:
                    density_2m += 1
            local_mean = mean(local_steps)
            stationary = bool((prev_steps[idx] <= 0.5 and next_steps[idx] <= 0.5) or local_mean <= 0.5)
            stationary_flags.append(stationary)
            position_in_crop = as_int(row.get("position_in_crop"))
            crop_len = min(max_len, n - as_int(row.get("crop_index")) * max_len) if max_len else n
            distance_to_crop_boundary = min(position_in_crop, max(crop_len - 1 - position_in_crop, 0))
            ratio = idx / max(n - 1, 1)
            row.update(
                prev_step_distance_m=prev_steps[idx],
                next_step_distance_m=next_steps[idx],
                local_step_mean_m=local_mean,
                local_step_std_m=std(local_steps),
                local_turn_angle_deg=turn,
                local_density_1m=density_1m,
                local_density_2m=density_2m,
                stationary_flag=stationary,
                trace_position_ratio=ratio,
                near_crop_boundary_flag=distance_to_crop_boundary <= near_boundary,
                distance_to_crop_boundary=distance_to_crop_boundary,
                near_trace_start_end_flag=(ratio <= 0.05 or ratio >= 0.95),
            )
        start = 0
        while start < n:
            end = start
            while end + 1 < n and stationary_flags[end + 1] == stationary_flags[start]:
                end += 1
            run_length = end - start + 1 if stationary_flags[start] else 0
            for idx in range(start, end + 1):
                trace_rows[idx]["stationary_run_length"] = run_length
            start = end + 1


def load_point_rows(output_dir):
    rows = []
    for split, path in prediction_paths(output_dir).items():
        split_rows = read_csv(path)
        for row in split_rows:
            row["split"] = row.get("split") or split
        rows.extend(split_rows)
    return rows


def metric_summary(rows):
    labels = np.asarray([as_int(row["true_label"]) for row in rows])
    preds = np.asarray([as_int(row["pred_label"]) for row in rows])
    if len(labels) == 0:
        return {}
    report = classification_report(labels, preds, labels=[0, 1], target_names=["road", "field"], output_dict=True, zero_division=0)
    pred_road_rate = float((preds == 0).mean())
    pred_field_rate = float((preds == 1).mean())
    true_road_rate = float((labels == 0).mean())
    true_field_rate = float((labels == 1).mean())
    return dict(
        accuracy=float((labels == preds).mean()),
        macro_f1=float(report["macro avg"]["f1-score"]),
        road_precision=float(report["road"]["precision"]),
        road_recall=float(report["road"]["recall"]),
        road_f1=float(report["road"]["f1-score"]),
        field_precision=float(report["field"]["precision"]),
        field_recall=float(report["field"]["recall"]),
        field_f1=float(report["field"]["f1-score"]),
        true_road_rate=true_road_rate,
        pred_road_rate=pred_road_rate,
        true_field_rate=true_field_rate,
        pred_field_rate=pred_field_rate,
        road_as_field_count=sum(1 for row in rows if row["error_type"] == "road_as_field"),
        field_as_road_count=sum(1 for row in rows if row["error_type"] == "field_as_road"),
        high_conf_error_count=sum(1 for row in rows if str(row.get("is_high_conf_error")).lower() == "true"),
        collapse_flag=bool(pred_road_rate < 0.05 or pred_field_rate < 0.05),
        confusion_matrix=json.dumps(confusion_matrix(labels, preds, labels=[0, 1]).tolist(), ensure_ascii=False),
    )


def write_error_summary(args, rows):
    out_rows = []
    for split in ("valid", "test"):
        split_rows = [row for row in rows if row["split"] == split]
        summary = metric_summary(split_rows)
        if summary:
            out_rows.append(dict(split=split, **summary))
    fields = ["split", *list(out_rows[0].keys())[1:]] if out_rows else ["split"]
    write_csv(args.summary_csv, fields, out_rows)
    return out_rows


def segment_confidence_stats(points):
    n = len(points)
    confidences = [as_float(row["confidence"]) for row in points]
    third = max(1, n // 3)
    edge_points = points[:third] + points[-third:]
    mid_start = n // 3
    mid_end = max(mid_start + 1, 2 * n // 3)
    middle_points = points[mid_start:mid_end]
    edge_conf = mean([row["confidence"] for row in edge_points])
    middle_conf = mean([row["confidence"] for row in middle_points])
    return mean(confidences), max(confidences), middle_conf, edge_conf, middle_conf - edge_conf


def make_error_segments(args, rows):
    errors = [row for row in rows if row["error_type"] != "correct"]
    grouped = defaultdict(list)
    for row in errors:
        grouped[(row["split"], row["trace_id"], row["error_type"])].append(row)
    segments = []
    for (split, trace_id, error_type), group in grouped.items():
        group.sort(key=lambda row: as_int(row["global_index"]))
        current = []
        prev_index = None
        for row in group:
            global_index = as_int(row["global_index"])
            if current and global_index != prev_index + 1:
                segments.append(segment_row(args, current))
                current = []
            current.append(row)
            prev_index = global_index
        if current:
            segments.append(segment_row(args, current))
    segments.sort(key=lambda row: (row["split"], row["trace_id"], as_int(row["start_index"])))
    fields = list(segments[0].keys()) if segments else []
    write_csv(Path(args.output_dir) / "error_segments.csv", fields, segments)
    top_long = sorted([row for row in segments if as_int(row["length"]) >= 20], key=lambda row: as_int(row["length"]), reverse=True)
    write_csv(Path(args.output_dir) / "top_long_error_segments.csv", fields, top_long[:200])
    return segments


def segment_row(args, points):
    first = points[0]
    last = points[-1]
    confidence_mean, confidence_max, middle_conf, edge_conf, middle_minus_edge = segment_confidence_stats(points)
    crop_indices = {as_int(row.get("crop_index")) for row in points}
    near_crop = any(str(row.get("near_crop_boundary_flag")).lower() == "true" for row in points)
    cross_crop = len(crop_indices) > 1
    return dict(
        split=first["split"],
        trace_id=first["trace_id"],
        error_type=first["error_type"],
        start_index=first["global_index"],
        end_index=last["global_index"],
        length=len(points),
        mean_confidence=confidence_mean,
        max_confidence=confidence_max,
        middle_confidence=middle_conf,
        edge_confidence=edge_conf,
        middle_minus_edge_confidence=middle_minus_edge,
        mean_prob_road=mean([row["prob_road"] for row in points]),
        mean_prob_field=mean([row["prob_field"] for row in points]),
        mean_entropy=mean([row["entropy"] for row in points]),
        mean_margin=mean([row["margin"] for row in points]),
        mean_prev_step_distance_m=mean([row["prev_step_distance_m"] for row in points]),
        mean_next_step_distance_m=mean([row["next_step_distance_m"] for row in points]),
        mean_local_step_mean_m=mean([row["local_step_mean_m"] for row in points]),
        mean_local_turn_angle_deg=mean([row["local_turn_angle_deg"] for row in points]),
        p75_turn_angle_deg=quantile([row["local_turn_angle_deg"] for row in points], 0.75),
        p95_turn_angle_deg=quantile([row["local_turn_angle_deg"] for row in points], 0.95),
        mean_local_density_1m=mean([row["local_density_1m"] for row in points]),
        mean_local_density_2m=mean([row["local_density_2m"] for row in points]),
        stationary_rate=mean([1.0 if str(row["stationary_flag"]).lower() == "true" else 0.0 for row in points]),
        mean_stationary_run_length=mean([row["stationary_run_length"] for row in points]),
        mean_trace_position_ratio=mean([row["trace_position_ratio"] for row in points]),
        cross_crop_boundary=cross_crop,
        near_crop_boundary=near_crop,
        inside_crop=(not cross_crop and not near_crop),
        high_conf_segment=confidence_mean >= 0.8,
    )


def long_error_causes(args, segments):
    long_segments = [row for row in segments if as_int(row["length"]) >= 20]
    cause_names = [
        "cross_crop_related",
        "inside_crop_state_drift",
        "road_curve_related",
        "strong_curve_or_uturn_related",
        "stationary_dense_related",
        "confidence_drift_related",
        "high_confidence_segment",
    ]
    enriched = []
    for row in long_segments:
        out = dict(row)
        error_type = row["error_type"]
        out["cross_crop_related"] = str(row["cross_crop_boundary"]).lower() == "true" or str(row["near_crop_boundary"]).lower() == "true"
        out["inside_crop_state_drift"] = not out["cross_crop_related"]
        out["road_curve_related"] = (
            error_type == "road_as_field"
            and as_float(row["p95_turn_angle_deg"]) >= 45.0
            and as_float(row["stationary_rate"]) < 0.5
        )
        out["strong_curve_or_uturn_related"] = (
            error_type == "road_as_field"
            and as_float(row["p95_turn_angle_deg"]) >= 90.0
            and as_float(row["mean_local_step_mean_m"]) > 1.0
        )
        out["stationary_dense_related"] = (
            error_type == "field_as_road"
            and (
                as_float(row["stationary_rate"]) >= 0.6
                or as_float(row["mean_local_step_mean_m"]) <= 0.5
                or as_float(row["mean_local_density_1m"]) >= 4.0
            )
        )
        out["confidence_drift_related"] = as_float(row["middle_minus_edge_confidence"]) >= 0.05
        out["high_confidence_segment"] = as_float(row["mean_confidence"]) >= 0.8
        enriched.append(out)
    fields = list(enriched[0].keys()) if enriched else list(segments[0].keys()) + cause_names if segments else cause_names
    write_csv(Path(args.output_dir) / "long_error_segments_with_causes.csv", fields, enriched)

    total_segments = len(enriched)
    total_points = sum(as_int(row["length"]) for row in enriched)
    stats = []
    for cause in cause_names:
        selected = [row for row in enriched if str(row.get(cause)).lower() == "true"]
        points_count = sum(as_int(row["length"]) for row in selected)
        stats.append(
            dict(
                cause=cause,
                segments_count=len(selected),
                points_count=points_count,
                segments_ratio=len(selected) / total_segments if total_segments else 0.0,
                points_ratio=points_count / total_points if total_points else 0.0,
            )
        )
    write_csv(Path(args.output_dir) / "long_error_cause_stats.csv", ["cause", "segments_count", "points_count", "segments_ratio", "points_ratio"], stats)
    return enriched, stats


def feature_error_contrast(args, rows):
    correct = [row for row in rows if row["error_type"] == "correct"]
    all_error = [row for row in rows if row["error_type"] != "correct"]
    road_as_field = [row for row in rows if row["error_type"] == "road_as_field"]
    field_as_road = [row for row in rows if row["error_type"] == "field_as_road"]
    high_conf_error = [row for row in rows if str(row.get("is_high_conf_error")).lower() == "true"]
    out = []
    for feature in FEATURE_COLUMNS:
        correct_values = [row[feature] for row in correct]
        all_values = [row[feature] for row in all_error]
        road_values = [row[feature] for row in road_as_field]
        field_values = [row[feature] for row in field_as_road]
        high_values = [row[feature] for row in high_conf_error]
        row = dict(
            feature=feature,
            correct_mean=mean(correct_values),
            all_error_mean=mean(all_values),
            road_as_field_mean=mean(road_values),
            field_as_road_mean=mean(field_values),
            high_conf_error_mean=mean(high_values),
            all_error_minus_correct=mean(all_values) - mean(correct_values),
            road_as_field_minus_correct=mean(road_values) - mean(correct_values),
            field_as_road_minus_correct=mean(field_values) - mean(correct_values),
            effect_size_all_error=cohen_d(correct_values, all_values),
            effect_size_road_as_field=cohen_d(correct_values, road_values),
            effect_size_field_as_road=cohen_d(correct_values, field_values),
        )
        row["sort_abs_effect"] = max(
            abs(row["effect_size_all_error"]),
            abs(row["effect_size_road_as_field"]),
            abs(row["effect_size_field_as_road"]),
        )
        out.append(row)
    out.sort(key=lambda row: row["sort_abs_effect"], reverse=True)
    fields = [
        "feature",
        "correct_mean",
        "all_error_mean",
        "road_as_field_mean",
        "field_as_road_mean",
        "high_conf_error_mean",
        "all_error_minus_correct",
        "road_as_field_minus_correct",
        "field_as_road_minus_correct",
        "effect_size_all_error",
        "effect_size_road_as_field",
        "effect_size_field_as_field" if False else "effect_size_field_as_road",
    ]
    write_csv(Path(args.output_dir) / "feature_error_contrast.csv", fields, out)
    return out


def spatial_and_trace_stats(args, rows, segments):
    grid = defaultdict(lambda: Counter())
    grid_traces = defaultdict(set)
    for row in rows:
        lon_bin = math.floor(as_float(row["longitude"]) / 0.001) * 0.001
        lat_bin = math.floor(as_float(row["latitude"]) / 0.001) * 0.001
        key = (row["split"], f"{lon_bin:.3f}", f"{lat_bin:.3f}")
        grid[key]["total_points"] += 1
        grid_traces[key].add(row["trace_id"])
        if row["error_type"] != "correct":
            grid[key]["error_points"] += 1
            grid[key][row["error_type"]] += 1
        if str(row.get("is_high_conf_error")).lower() == "true":
            grid[key]["high_conf_error_points"] += 1
    grid_rows = []
    for (split, lon_bin, lat_bin), counter in grid.items():
        road = counter["road_as_field"]
        field = counter["field_as_road"]
        dominant = "road_as_field" if road >= field and road > 0 else "field_as_road" if field > 0 else "none"
        grid_rows.append(
            dict(
                split=split,
                lon_bin=lon_bin,
                lat_bin=lat_bin,
                total_points=counter["total_points"],
                error_points=counter["error_points"],
                error_rate=counter["error_points"] / counter["total_points"] if counter["total_points"] else 0.0,
                road_as_field_points=road,
                field_as_road_points=field,
                high_conf_error_points=counter["high_conf_error_points"],
                dominant_error_type=dominant,
                trace_count=len(grid_traces[(split, lon_bin, lat_bin)]),
            )
        )
    grid_rows.sort(key=lambda row: as_int(row["error_points"]), reverse=True)
    write_csv(Path(args.output_dir) / "spatial_error_grid_0p001deg.csv", list(grid_rows[0].keys()) if grid_rows else [], grid_rows)

    trace = defaultdict(lambda: Counter())
    trace_values = defaultdict(lambda: defaultdict(list))
    long_segments_by_trace = defaultdict(lambda: Counter())
    for row in rows:
        key = (row["split"], row["trace_id"])
        trace[key]["total_points"] += 1
        if row["error_type"] != "correct":
            trace[key]["error_points"] += 1
            trace[key][row["error_type"]] += 1
            trace_values[key]["confidence"].append(row["confidence"])
            trace_values[key]["turn"].append(row["local_turn_angle_deg"])
            trace_values[key]["stationary"].append(1.0 if str(row["stationary_flag"]).lower() == "true" else 0.0)
        if str(row.get("is_high_conf_error")).lower() == "true":
            trace[key]["high_conf_error"] += 1
    for segment in segments:
        if as_int(segment["length"]) >= 20:
            key = (segment["split"], segment["trace_id"])
            long_segments_by_trace[key]["long_error_segments"] += 1
            long_segments_by_trace[key]["long_error_points"] += as_int(segment["length"])
    trace_rows = []
    for key, counter in trace.items():
        split, trace_id = key
        road = counter["road_as_field"]
        field = counter["field_as_road"]
        dominant = "road_as_field" if road >= field and road > 0 else "field_as_road" if field > 0 else "none"
        total = counter["total_points"]
        trace_rows.append(
            dict(
                split=split,
                trace_id=trace_id,
                total_points=total,
                error_points=counter["error_points"],
                error_rate=counter["error_points"] / total if total else 0.0,
                road_as_field=road,
                field_as_road=field,
                high_conf_error=counter["high_conf_error"],
                long_error_segments=long_segments_by_trace[key]["long_error_segments"],
                long_error_points=long_segments_by_trace[key]["long_error_points"],
                dominant_error_type=dominant,
                mean_confidence_error=mean(trace_values[key]["confidence"]),
                mean_local_turn_error=mean(trace_values[key]["turn"]),
                mean_stationary_rate_error=mean(trace_values[key]["stationary"]),
            )
        )
    trace_rows.sort(key=lambda row: as_int(row["error_points"]), reverse=True)
    write_csv(Path(args.output_dir) / "trace_error_summary.csv", list(trace_rows[0].keys()) if trace_rows else [], trace_rows)
    return grid_rows, trace_rows


def msc_behavior(args, rows):
    candidates = [
        Path("diagnostics/PT2G_MSC_v1_finetune_40ep_segment_context_audit.csv"),
        Path("diagnostics/PT2G_MSC_v1_segment_context_audit.csv"),
        Path(args.output_dir) / "PT2G_MSC_v1_segment_context_audit.csv",
    ]
    audit_rows = []
    for path in candidates:
        audit_rows = read_csv(path)
        if audit_rows:
            break
    out = []
    for row in audit_rows:
        out.append(
            dict(
                epoch=row.get("epoch"),
                segment_scale=row.get("segment_scale"),
                context_to_fused_ratio=row.get("context_to_fused_ratio"),
                fused_norm_before_msc=row.get("fused_norm_mean_before_msc"),
                fused_norm_after_msc=row.get("fused_norm_mean_after_msc"),
                context_norm=row.get("context_norm_mean"),
            )
        )
    fields = ["epoch", "segment_scale", "context_to_fused_ratio", "fused_norm_before_msc", "fused_norm_after_msc", "context_norm"]
    write_csv(Path(args.output_dir) / "msc_behavior_analysis.csv", fields, out)
    return out


def compare_fixed_new_errors(args, msc_test_rows):
    baseline_rows = read_csv(args.baseline_test_predictions)
    if not baseline_rows:
        write_csv(Path(args.output_dir) / "msc_fixed_points.csv", [], [])
        write_csv(Path(args.output_dir) / "msc_new_errors.csv", [], [])
        write_csv(Path(args.output_dir) / "msc_fixed_vs_new_error_contrast.csv", [], [])
        return [], [], []
    base_lookup = {
        (row["trace_id"], as_int(row["sample_index"]), as_int(row["point_index"])): row
        for row in baseline_rows
    }
    fixed = []
    new_errors = []
    for row in msc_test_rows:
        key = (row["trace_id"], as_int(row["sample_index"]), as_int(row["point_index"]))
        base = base_lookup.get(key)
        if base is None:
            continue
        base_error = base.get("error_type") != "correct"
        msc_error = row["error_type"] != "correct"
        out = dict(row)
        out.update(
            baseline_pred_label=base.get("pred_label"),
            baseline_error_type=base.get("error_type"),
            baseline_prob_road=base.get("prob_road"),
            baseline_prob_field=base.get("prob_field"),
            baseline_confidence=base.get("confidence"),
        )
        if base_error and not msc_error:
            fixed.append(out)
        elif (not base_error) and msc_error:
            new_errors.append(out)
    fields = point_prediction_fields() + [
        "baseline_pred_label",
        "baseline_error_type",
        "baseline_prob_road",
        "baseline_prob_field",
        "baseline_confidence",
    ]
    write_csv(Path(args.output_dir) / "msc_fixed_points.csv", fields, fixed)
    write_csv(Path(args.output_dir) / "msc_new_errors.csv", fields, new_errors)
    contrast = fixed_new_contrast(fixed, new_errors)
    write_csv(Path(args.output_dir) / "msc_fixed_vs_new_error_contrast.csv", list(contrast[0].keys()) if contrast else [], contrast)
    return fixed, new_errors, contrast


def fixed_new_contrast(fixed, new_errors):
    rows = []
    for feature in FEATURE_COLUMNS + ["local_turn_angle_deg", "local_density_1m", "local_step_mean_m", "confidence", "margin"]:
        fixed_values = [row.get(feature) for row in fixed]
        new_values = [row.get(feature) for row in new_errors]
        rows.append(
            dict(
                feature=feature,
                fixed_mean=mean(fixed_values),
                new_error_mean=mean(new_values),
                new_minus_fixed=mean(new_values) - mean(fixed_values),
                effect_size_new_vs_fixed=cohen_d(fixed_values, new_values),
            )
        )
    rows.sort(key=lambda row: abs(row["effect_size_new_vs_fixed"]), reverse=True)
    return rows


def error_count_summary_from_rows(rows):
    segments = make_segments_in_memory(rows)
    return dict(
        accuracy=metric_summary(rows).get("accuracy", 0.0),
        macro_f1=metric_summary(rows).get("macro_f1", 0.0),
        road_f1=metric_summary(rows).get("road_f1", 0.0),
        field_f1=metric_summary(rows).get("field_f1", 0.0),
        pred_road_rate=metric_summary(rows).get("pred_road_rate", 0.0),
        road_as_field=sum(1 for row in rows if row.get("error_type") == "road_as_field"),
        field_as_road=sum(1 for row in rows if row.get("error_type") == "field_as_road"),
        high_conf_error=sum(1 for row in rows if str(row.get("is_high_conf_error")).lower() == "true" or (row.get("error_type") != "correct" and as_float(row.get("confidence")) >= 0.8)),
        long_road_as_field=sum(1 for segment in segments if segment["error_type"] == "road_as_field" and segment["length"] >= 20),
        long_field_as_road=sum(1 for segment in segments if segment["error_type"] == "field_as_road" and segment["length"] >= 20),
    )


def make_segments_in_memory(rows):
    errors = [row for row in rows if row.get("error_type") != "correct"]
    grouped = defaultdict(list)
    for row in errors:
        grouped[(row.get("trace_id"), row.get("error_type"))].append(row)
    segments = []
    for (_trace, error_type), group in grouped.items():
        group.sort(key=lambda row: (as_int(row.get("sample_index")), as_int(row.get("point_index"))))
        current = []
        prev_sample = prev_point = None
        for row in group:
            sample = as_int(row.get("sample_index"))
            point = as_int(row.get("point_index"))
            continuous = current and sample == prev_sample and point == prev_point + 1
            if current and not continuous:
                segments.append(dict(error_type=error_type, length=len(current)))
                current = []
            current.append(row)
            prev_sample, prev_point = sample, point
        if current:
            segments.append(dict(error_type=error_type, length=len(current)))
    return segments


def pt2g_vs_msc_delta(args, msc_test_rows):
    baseline_rows = read_csv(args.baseline_test_predictions)
    if not baseline_rows:
        write_csv(Path(args.output_dir) / "pt2g_vs_msc_error_delta.csv", [], [])
        return {}
    base = error_count_summary_from_rows(baseline_rows)
    msc = error_count_summary_from_rows(msc_test_rows)
    row = dict(
        accuracy_delta=msc["accuracy"] - base["accuracy"],
        macro_f1_delta=msc["macro_f1"] - base["macro_f1"],
        road_f1_delta=msc["road_f1"] - base["road_f1"],
        field_f1_delta=msc["field_f1"] - base["field_f1"],
        road_as_field_delta=msc["road_as_field"] - base["road_as_field"],
        field_as_road_delta=msc["field_as_road"] - base["field_as_road"],
        long_road_as_field_delta=msc["long_road_as_field"] - base["long_road_as_field"],
        long_field_as_road_delta=msc["long_field_as_road"] - base["long_field_as_road"],
        high_conf_error_delta=msc["high_conf_error"] - base["high_conf_error"],
        pred_road_rate_delta=msc["pred_road_rate"] - base["pred_road_rate"],
        pt2g_road_as_field=base["road_as_field"],
        msc_road_as_field=msc["road_as_field"],
        pt2g_field_as_road=base["field_as_road"],
        msc_field_as_road=msc["field_as_road"],
    )
    write_csv(Path(args.output_dir) / "pt2g_vs_msc_error_delta.csv", list(row.keys()), [row])
    return row


def html_escape(value):
    return str(value).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


def generate_html(args, test_rows):
    html_dir = Path(args.html_dir)
    trace_dir = html_dir / "traces"
    trace_dir.mkdir(parents=True, exist_ok=True)
    grouped = defaultdict(list)
    for row in test_rows:
        grouped[row["trace_id"]].append(row)
    index_rows = []
    for trace_id, points in sorted(grouped.items()):
        points.sort(key=lambda row: (as_int(row["global_index"]), as_int(row["point_index"])))
        slug = safe_slug(trace_id)
        summary = metric_summary(points)
        errors = sum(1 for row in points if row["error_type"] != "correct")
        index_rows.append(f"<tr><td><a href='traces/{slug}.html'>{html_escape(trace_id)}</a></td><td>{len(points)}</td><td>{summary.get('accuracy', 0.0)}</td><td>{errors}</td></tr>")
        write_trace_html(trace_dir / f"{slug}.html", args.short_name, trace_id, points)
    html_dir.mkdir(parents=True, exist_ok=True)
    html_dir.joinpath("index.html").write_text(
        f"""<!doctype html><html><head><meta charset='utf-8'><title>{args.short_name}</title>
<style>body{{font-family:-apple-system,BlinkMacSystemFont,Segoe UI,sans-serif;margin:24px;background:#f7f8fb;color:#172033}}table{{border-collapse:collapse;width:100%;background:white}}td,th{{border:1px solid #d8dee9;padding:8px;text-align:left}}a{{color:#174ea6}}</style></head>
<body><h1>{args.short_name} 测试集轨迹预测图</h1><p>点开轨迹后可切换真实图、预测图、错点图、高置信错点图。</p><table><thead><tr><th>trace</th><th>points</th><th>accuracy</th><th>errors</th></tr></thead><tbody>{''.join(index_rows)}</tbody></table></body></html>""",
        encoding="utf-8",
    )


def write_trace_html(path, run_name, trace_id, points):
    data = []
    for row in points:
        data.append(
            {
                "lat": as_float(row["latitude"]),
                "lon": as_float(row["longitude"]),
                "trace_id": row["trace_id"],
                "point_index": as_int(row["point_index"]),
                "global_index": as_int(row["global_index"]),
                "true_label": as_int(row["true_label"]),
                "pred_label": as_int(row["pred_label"]),
                "prob_road": as_float(row["prob_road"]),
                "prob_field": as_float(row["prob_field"]),
                "confidence": as_float(row["confidence"]),
                "error_type": row["error_type"],
                "local_turn_angle_deg": as_float(row["local_turn_angle_deg"]),
                "local_density_1m": as_float(row["local_density_1m"]),
                "stationary_flag": str(row["stationary_flag"]).lower() == "true",
                "is_high_conf_error": str(row["is_high_conf_error"]).lower() == "true",
            }
        )
    center_lat = mean([point["lat"] for point in data])
    center_lon = mean([point["lon"] for point in data])
    path.write_text(
        f"""<!doctype html><html><head><meta charset='utf-8'><title>{html_escape(trace_id)}</title>
<link rel='stylesheet' href='https://unpkg.com/leaflet@1.9.4/dist/leaflet.css'><script src='https://unpkg.com/leaflet@1.9.4/dist/leaflet.js'></script>
<style>html,body,#map{{height:100%;margin:0}}.panel{{position:fixed;z-index:999;left:56px;top:12px;background:white;border:1px solid #d8dee9;border-radius:8px;padding:8px;box-shadow:0 8px 24px #0002}}button{{display:block;margin:4px 0;width:110px}}.tip td{{padding:2px 5px}}</style></head>
<body><div id='map'></div><div class='panel'><b>{html_escape(run_name)}</b><br>{html_escape(trace_id)}<button onclick="showLayer('true')">真实图</button><button onclick="showLayer('pred')">预测图</button><button onclick="showLayer('error')">错点图</button><button onclick="showLayer('high')">高置信错点图</button><a href='../index.html'>返回</a></div>
<script>
const points = {json.dumps(data, ensure_ascii=False, separators=(",", ":"))};
const map = L.map('map').setView([{center_lat}, {center_lon}], 16);
L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{maxZoom: 21, attribution: '&copy; OpenStreetMap'}}).addTo(map);
function colorLabel(v){{return Number(v)===0?'#2563eb':'#16a34a'}}
function errorColor(p){{if(p.error_type==='road_as_field')return '#f97316'; if(p.error_type==='field_as_road')return '#a855f7'; return '#dc2626'}}
function tip(p){{return `<table class='tip'><tr><td>trace_id</td><td>${{p.trace_id}}</td></tr><tr><td>point_index</td><td>${{p.point_index}}</td></tr><tr><td>true_label</td><td>${{p.true_label}}</td></tr><tr><td>pred_label</td><td>${{p.pred_label}}</td></tr><tr><td>prob_road</td><td>${{p.prob_road.toFixed(6)}}</td></tr><tr><td>prob_field</td><td>${{p.prob_field.toFixed(6)}}</td></tr><tr><td>confidence</td><td>${{p.confidence.toFixed(6)}}</td></tr><tr><td>error_type</td><td>${{p.error_type}}</td></tr><tr><td>local_turn_angle_deg</td><td>${{p.local_turn_angle_deg.toFixed(3)}}</td></tr><tr><td>local_density_1m</td><td>${{p.local_density_1m}}</td></tr><tr><td>stationary_flag</td><td>${{p.stationary_flag}}</td></tr></table>`}}
function marker(p, mode){{let c=mode==='true'?colorLabel(p.true_label):mode==='pred'?colorLabel(p.pred_label):errorColor(p); let m=L.circleMarker([p.lat,p.lon],{{radius:mode==='high'?6:4,color:c,fillColor:c,fillOpacity:.85,weight:mode==='high'?2:1}}); m.bindTooltip(tip(p),{{sticky:true}}); return m}}
const route = points.map(p=>[p.lat,p.lon]);
const layers = {{
 true: L.layerGroup([L.polyline(route,{{color:'#64748b',weight:2,opacity:.35}}), ...points.map(p=>marker(p,'true'))]),
 pred: L.layerGroup([L.polyline(route,{{color:'#64748b',weight:2,opacity:.35}}), ...points.map(p=>marker(p,'pred'))]),
 error: L.layerGroup([L.polyline(route,{{color:'#64748b',weight:2,opacity:.2}}), ...points.filter(p=>p.error_type!=='correct').map(p=>marker(p,'error'))]),
 high: L.layerGroup([L.polyline(route,{{color:'#64748b',weight:2,opacity:.2}}), ...points.filter(p=>p.is_high_conf_error).map(p=>marker(p,'high'))])
}};
let active='true'; layers.true.addTo(map); if(route.length>1) map.fitBounds(L.latLngBounds(route),{{padding:[32,32]}});
function showLayer(name){{map.removeLayer(layers[active]); layers[name].addTo(map); active=name}}
</script></body></html>""",
        encoding="utf-8",
    )


def build_report(args, summaries, segments, long_segments, cause_stats, feature_rows, grid_rows, trace_rows, behavior, fixed, new_errors, delta):
    test_summary = next((row for row in summaries if row["split"] == "test"), {})
    valid_summary = next((row for row in summaries if row["split"] == "valid"), {})
    test_segments = [row for row in segments if row["split"] == "test"]
    long_test = [row for row in long_segments if row["split"] == "test"]
    error_counts = Counter()
    error_counts["road_as_field"] = as_int(test_summary.get("road_as_field_count"))
    error_counts["field_as_road"] = as_int(test_summary.get("field_as_road_count"))
    top_trace_points = sum(as_int(row["error_points"]) for row in trace_rows if row.get("split") == "test")
    top5_trace = sum(as_int(row["error_points"]) for row in [row for row in trace_rows if row.get("split") == "test"][:5])
    top10_trace = sum(as_int(row["error_points"]) for row in [row for row in trace_rows if row.get("split") == "test"][:10])
    grid_test = [row for row in grid_rows if row.get("split") == "test"]
    total_grid_errors = sum(as_int(row["error_points"]) for row in grid_test)
    top5_grid = sum(as_int(row["error_points"]) for row in grid_test[:5])
    top10_grid = sum(as_int(row["error_points"]) for row in grid_test[:10])
    top_all = sorted(feature_rows, key=lambda row: abs(as_float(row["effect_size_all_error"])), reverse=True)[:10]
    top_road = sorted(feature_rows, key=lambda row: abs(as_float(row["effect_size_road_as_field"])), reverse=True)[:10]
    top_field = sorted(feature_rows, key=lambda row: abs(as_float(row["effect_size_field_as_road"])), reverse=True)[:10]

    def feature_list(rows, key):
        return ", ".join(f"{row['feature']}({as_float(row[key]):.3f})" for row in rows)

    cause_map = {row["cause"]: row for row in cause_stats}
    cross_ratio = as_float(cause_map.get("cross_crop_related", {}).get("points_ratio"))
    inside_ratio = as_float(cause_map.get("inside_crop_state_drift", {}).get("points_ratio"))
    road_curve_ratio = as_float(cause_map.get("road_curve_related", {}).get("segments_ratio"))
    stationary_ratio = as_float(cause_map.get("stationary_dense_related", {}).get("segments_ratio"))
    high_conf = as_int(test_summary.get("high_conf_error_count"))
    final_behavior = behavior[-1] if behavior else {}
    recommendations = []
    if inside_ratio >= 0.5:
        recommendations.append("PT2G_MSC_v2")
    if road_curve_ratio >= 0.4:
        recommendations.append("PT2G_MSC_RC_v1")
    if stationary_ratio >= 0.4:
        recommendations.append("PT2G_MSC_SD_v1")
    if top_trace_points and top5_trace / top_trace_points >= 0.5:
        recommendations.append("PT2G_MSC_HM_v1")
    if cross_ratio >= 0.3:
        recommendations.append("PT2G_TSC_v1")
    if not recommendations:
        recommendations.append("PT2G_MSC_v2")

    report = [
        "# PT2G_MSC_v1 详细错误分析报告",
        "",
        "## 核心指标",
        "",
        f"- valid macro-F1: {valid_summary.get('macro_f1', '')}",
        f"- test macro-F1: {test_summary.get('macro_f1', '')}",
        f"- test road-F1: {test_summary.get('road_f1', '')}",
        f"- test field-F1: {test_summary.get('field_f1', '')}",
        f"- test road_as_field: {error_counts['road_as_field']}",
        f"- test field_as_road: {error_counts['field_as_road']}",
        f"- test high_conf_error: {high_conf}",
        f"- long error segments(length>=20): {len(long_test)}",
        f"- final segment_scale: {final_behavior.get('segment_scale', '')}",
        f"- final context_to_fused_ratio: {final_behavior.get('context_to_fused_ratio', '')}",
        "",
        "## 必答问题",
        "",
        f"1. 当前主要错误类型排序：{error_counts.most_common()}.",
        f"2. road_as_field 是否仍然是主要问题：{'是' if error_counts['road_as_field'] >= error_counts['field_as_road'] else '否'}，数量 {error_counts['road_as_field']}。",
        f"3. field_as_road 是否同等重要：{'是' if error_counts['field_as_road'] >= 0.8 * max(error_counts['road_as_field'], 1) else '否'}，数量 {error_counts['field_as_road']}。",
        f"4. 长段错误还有 {len(long_test)} 段；相对 PT2G 的长段变化见 `pt2g_vs_msc_error_delta.csv`。",
        f"5. 长段错误归因：inside crop points ratio={inside_ratio}，cross/near crop points ratio={cross_ratio}。",
        f"6. road_as_field 和弯道/折返关系：road_curve_related segment ratio={road_curve_ratio}。",
        f"7. field_as_road 和停留密集关系：stationary_dense_related segment ratio={stationary_ratio}。",
        f"8. 高置信错点共同特征：见 feature contrast，高置信错点数 {high_conf}；top high-conf related features 可从 `feature_error_contrast.csv` 的 high_conf_error_mean 继续查。",
        f"9. 错误集中性：top5 trace={top5_trace / top_trace_points if top_trace_points else 0.0}，top10 trace={top10_trace / top_trace_points if top_trace_points else 0.0}，top5 grid={top5_grid / total_grid_errors if total_grid_errors else 0.0}，top10 grid={top10_grid / total_grid_errors if total_grid_errors else 0.0}。",
        f"10. MSC 修正点数={len(fixed)}，新制造错点数={len(new_errors)}。修正/新增错点明细见 msc_fixed_points.csv 和 msc_new_errors.csv。",
        f"11. 下一步推荐顺序：{', '.join(dict.fromkeys(recommendations))}。",
        "",
        "## 和原 PT2G 对比",
        "",
        f"- accuracy_delta: {delta.get('accuracy_delta', '')}",
        f"- macro_f1_delta: {delta.get('macro_f1_delta', '')}",
        f"- road_f1_delta: {delta.get('road_f1_delta', '')}",
        f"- field_f1_delta: {delta.get('field_f1_delta', '')}",
        f"- road_as_field_delta: {delta.get('road_as_field_delta', '')}",
        f"- field_as_road_delta: {delta.get('field_as_road_delta', '')}",
        f"- high_conf_error_delta: {delta.get('high_conf_error_delta', '')}",
        f"- pred_road_rate_delta: {delta.get('pred_road_rate_delta', '')}",
        "",
        "## 特征差异 Top 10",
        "",
        f"- all_error: {feature_list(top_all, 'effect_size_all_error')}",
        f"- road_as_field: {feature_list(top_road, 'effect_size_road_as_field')}",
        f"- field_as_road: {feature_list(top_field, 'effect_size_field_as_road')}",
        "",
        "## 输出文件",
        "",
        f"- 逐点预测：`{args.output_dir}/valid_point_predictions_with_features.csv` 和 `test_point_predictions_with_features.csv`",
        f"- 连续错误段：`{args.output_dir}/error_segments.csv`",
        f"- 长段归因：`{args.output_dir}/long_error_segments_with_causes.csv`",
        f"- HTML：`{args.html_dir}/index.html`",
    ]
    Path(args.report_path).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report_path).write_text("\n".join(report) + "\n", encoding="utf-8")
    return dict(
        main_error=error_counts.most_common(1)[0][0] if error_counts else "",
        road_as_field=error_counts["road_as_field"],
        field_as_road=error_counts["field_as_road"],
        long_segments=len(long_test),
        high_conf_error=high_conf,
        top_trace_concentration=top5_trace / top_trace_points if top_trace_points else 0.0,
        recommendation=dict.fromkeys(recommendations),
    )


def make_pack(args):
    pack_path = Path(args.pack_path)
    pack_path.parent.mkdir(parents=True, exist_ok=True)
    if pack_path.exists():
        pack_path.unlink()
    include = [
        Path("scripts/analyze_pt2g_msc_errors.py"),
        Path(args.output_dir),
        Path(args.report_path),
        Path(args.html_dir) / "index.html",
    ]
    with zipfile.ZipFile(pack_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for item in include:
            if item.is_dir():
                for path in sorted(item.rglob("*")):
                    if path.is_file() and path.suffix != ".pt":
                        zf.write(path, path.as_posix())
            elif item.exists() and item.is_file():
                zf.write(item, item.as_posix())


def main():
    args = parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    generate_predictions_if_needed(args)
    rows = load_point_rows(args.output_dir)
    summaries = write_error_summary(args, rows)
    segments = make_error_segments(args, rows)
    long_segments, cause_stats = long_error_causes(args, segments)
    feature_rows = feature_error_contrast(args, rows)
    grid_rows, trace_rows = spatial_and_trace_stats(args, rows, segments)
    behavior = msc_behavior(args, rows)
    test_rows = [row for row in rows if row["split"] == "test"]
    fixed, new_errors, _contrast = compare_fixed_new_errors(args, test_rows)
    delta = pt2g_vs_msc_delta(args, test_rows)
    generate_html(args, test_rows)
    report_summary = build_report(args, summaries, segments, long_segments, cause_stats, feature_rows, grid_rows, trace_rows, behavior, fixed, new_errors, delta)
    make_pack(args)
    recommendation = next(iter(report_summary["recommendation"].keys())) if report_summary["recommendation"] else ""
    print(f"main_error_type={report_summary['main_error']}")
    print(f"road_as_field={report_summary['road_as_field']}")
    print(f"field_as_road={report_summary['field_as_road']}")
    print(f"long_error_segments={report_summary['long_segments']}")
    print(f"high_conf_error_points={report_summary['high_conf_error']}")
    print(f"top5_trace_error_concentration={report_summary['top_trace_concentration']}")
    print(f"recommended_next_module={recommendation}")
    print(f"report_path={args.report_path}")
    print(f"pack_path={args.pack_path}")
    print(f"html_path={Path(args.html_dir) / 'index.html'}")


if __name__ == "__main__":
    main()
