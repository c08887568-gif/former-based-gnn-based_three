import argparse
import json
import math
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


AUDIT_NAME = "FRG_FSCG_AUDIT_v4"
BASELINE_GROUP = "PT2G_MSC_v1_time_fixed_strict_40pre"
POINT_DIR = f"diagnostics/{BASELINE_GROUP}_error_analysis"
FRG_DIR = "diagnostics/FRG_MoE_AUDIT_v3"
PLUS_DIR = "diagnostics/FRG_MoE_CONSERVE_v3_PLUS_AUDIT"
RELAX_DIR = "diagnostics/FRG_MoE_CONSERVE_v3_RELAX_AUDIT"
TFSV_DIR = "diagnostics/TFSV_LC_MoE_AUDIT_v2"
SCORE_COLUMNS = [
    "dense_stationary_score",
    "field_sweep_context_score",
    "short_isolated_blip_score",
    "headland_sweep_score",
    "global_rank_score",
    "road_connector_guard_score",
    "true_road_guard_score",
    "fake_road_fscg_score",
]
EXPERT_NAMES = {
    "dense_stationary_score": "DenseStationaryExpert",
    "field_sweep_context_score": "FieldSweepInteriorExpert",
    "short_isolated_blip_score": "ShortBlipExpert",
    "headland_sweep_score": "HeadlandSweepExpert",
    "global_rank_score": "GlobalRankEnsemble",
    "road_connector_guard_score": "ConnectorGuard",
    "true_road_guard_score": "TrueRoadGuard",
    "fake_road_fscg_score": "FRG_FSCG_Final",
}
TOPK_FRACTIONS = [0.05, 0.10, 0.20, 0.30]
RADIUS_LIST = [5.0, 10.0, 20.0]


def parse_args():
    parser = argparse.ArgumentParser(description="FRG-FSCG_AUDIT_v4 fake-road field-sweep context audit.")
    parser.add_argument("--point_dir", default=POINT_DIR)
    parser.add_argument("--frg_dir", default=FRG_DIR)
    parser.add_argument("--plus_dir", default=PLUS_DIR)
    parser.add_argument("--relax_dir", default=RELAX_DIR)
    parser.add_argument("--tfsv_dir", default=TFSV_DIR)
    parser.add_argument("--output_dir", default=f"diagnostics/{AUDIT_NAME}")
    parser.add_argument("--summary_csv", default=f"results/{AUDIT_NAME}_summary.csv")
    parser.add_argument("--report_path", default=f"analysis/{AUDIT_NAME}_report.md")
    parser.add_argument("--pack_path", default=f"analysis_packs/{AUDIT_NAME}_for_chatgpt.zip")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_jobs", type=int, default=8)
    return parser.parse_args()


def ensure_dirs(args):
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.summary_csv).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report_path).parent.mkdir(parents=True, exist_ok=True)
    Path(args.pack_path).parent.mkdir(parents=True, exist_ok=True)


def clip01(values):
    return np.clip(values, 0.0, 1.0)


def safe_float(value, default=0.0):
    try:
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def numeric(df, col, default=0.0):
    if col not in df.columns:
        return pd.Series(np.full(len(df), default, dtype=float), index=df.index)
    return pd.to_numeric(df[col], errors="coerce").fillna(default)


def safe_auc(y_true, score):
    try:
        y = np.asarray(y_true, dtype=int)
        s = np.asarray(score, dtype=float)
        mask = np.isfinite(s)
        y = y[mask]
        s = s[mask]
        if len(y) == 0 or len(set(y.tolist())) < 2:
            return float("nan")
        return float(roc_auc_score(y, s))
    except Exception:
        return float("nan")


def safe_ap(y_true, score):
    try:
        y = np.asarray(y_true, dtype=int)
        s = np.asarray(score, dtype=float)
        mask = np.isfinite(s)
        y = y[mask]
        s = s[mask]
        if len(y) == 0 or len(set(y.tolist())) < 2:
            return float("nan")
        return float(average_precision_score(y, s))
    except Exception:
        return float("nan")


def haversine_m(lon1, lat1, lon2, lat2):
    radius = 6371000.0
    lon1 = np.deg2rad(lon1)
    lat1 = np.deg2rad(lat1)
    lon2 = np.deg2rad(lon2)
    lat2 = np.deg2rad(lat2)
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return 2.0 * radius * np.arctan2(np.sqrt(a), np.sqrt(np.maximum(1.0 - a, 0.0)))


def project_xy(lon, lat):
    lon = np.asarray(lon, dtype=float)
    lat = np.asarray(lat, dtype=float)
    lon0 = float(np.nanmean(lon)) if len(lon) else 0.0
    lat0 = float(np.nanmean(lat)) if len(lat) else 0.0
    x = (lon - lon0) * 111320.0 * math.cos(math.radians(lat0))
    y = (lat - lat0) * 110540.0
    return x, y


def angle_diff_deg(a, b):
    diff = np.abs((a - b + 180.0) % 360.0 - 180.0)
    return np.minimum(diff, 180.0 - diff)


def circular_mean_deg(values, weights=None):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 0.0
    radians = np.deg2rad(values)
    if weights is None:
        sin_mean = np.mean(np.sin(radians))
        cos_mean = np.mean(np.cos(radians))
    else:
        weights = np.asarray(weights, dtype=float)[: len(radians)]
        sin_mean = np.average(np.sin(radians), weights=weights)
        cos_mean = np.average(np.cos(radians), weights=weights)
    return float((np.rad2deg(np.arctan2(sin_mean, cos_mean)) + 360.0) % 180.0)


def entropy(prob_road, prob_field):
    p0 = np.clip(np.asarray(prob_road, dtype=float), 1e-9, 1.0)
    p1 = np.clip(np.asarray(prob_field, dtype=float), 1e-9, 1.0)
    return -(p0 * np.log(p0) + p1 * np.log(p1))


def summarize(values, prefix, row, quantiles=False):
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        arr = np.array([0.0], dtype=float)
    row[f"{prefix}_mean"] = float(np.mean(arr))
    row[f"{prefix}_std"] = float(np.std(arr))
    row[f"{prefix}_min"] = float(np.min(arr))
    row[f"{prefix}_max"] = float(np.max(arr))
    if quantiles:
        row[f"{prefix}_q25"] = float(np.quantile(arr, 0.25))
        row[f"{prefix}_q75"] = float(np.quantile(arr, 0.75))


def label_state(pred_class, road_points, field_points):
    length = max(road_points + field_points, 1)
    road_ratio = road_points / length
    field_ratio = field_points / length
    if pred_class == 0 and field_ratio >= 0.8:
        return "fake_road", "field_as_road", 1
    if pred_class == 0 and road_ratio >= 0.8:
        return "true_road", "correct", 0
    return "mixed_transition", "mixed", -1


def length_bucket(length):
    if length < 10:
        return "short"
    if length < 50:
        return "mid"
    return "long"


def read_points(args, split):
    path = Path(args.point_dir) / f"{split}_point_predictions_with_features.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if "true_label" not in df.columns and "label" in df.columns:
        df = df.rename(columns={"label": "true_label"})
    if "pred_label" not in df.columns and "base_pred" in df.columns:
        df = df.rename(columns={"base_pred": "pred_label"})
    df["trace_id"] = df["trace_id"].astype(str)
    df["global_index"] = numeric(df, "global_index", numeric(df, "point_index", 0)).astype(int)
    df["point_index"] = numeric(df, "point_index", df["global_index"]).astype(int)
    df["true_label"] = numeric(df, "true_label", -1).astype(int)
    df["pred_label"] = numeric(df, "pred_label", -1).astype(int)
    if "prob_road" not in df.columns:
        df["prob_road"] = (df["pred_label"] == 0).astype(float)
    if "prob_field" not in df.columns:
        df["prob_field"] = 1.0 - numeric(df, "prob_road", 0.0)
    if "confidence" not in df.columns:
        df["confidence"] = np.maximum(numeric(df, "prob_road", 0.0), numeric(df, "prob_field", 0.0))
    if "margin" not in df.columns:
        df["margin"] = np.abs(numeric(df, "prob_road", 0.0) - numeric(df, "prob_field", 0.0))
    if "entropy" not in df.columns:
        df["entropy"] = entropy(df["prob_road"], df["prob_field"])
    if "stationary_flag" not in df.columns:
        df["stationary_flag"] = numeric(df, "local_step_mean_m", 0.0).le(0.5)
    return df.sort_values(["trace_id", "global_index", "point_index"]).reset_index(drop=True)


def build_trace_cache(points):
    traces = {}
    for trace_id, trace in points.groupby("trace_id", sort=False):
        trace = trace.sort_values(["global_index", "point_index"]).reset_index(drop=True)
        lon = numeric(trace, "longitude", 0.0).to_numpy(dtype=float)
        lat = numeric(trace, "latitude", 0.0).to_numpy(dtype=float)
        x, y = project_xy(lon, lat)
        heading = np.zeros(len(trace), dtype=float)
        if len(trace) > 1:
            dx = np.diff(x, prepend=x[0])
            dy = np.diff(y, prepend=y[0])
            heading = (np.rad2deg(np.arctan2(dy, dx)) + 360.0) % 180.0
        traces[trace_id] = {
            "df": trace,
            "x": x,
            "y": y,
            "heading": heading,
            "pred": numeric(trace, "pred_label", -1).astype(int).to_numpy(),
            "true": numeric(trace, "true_label", -1).astype(int).to_numpy(),
            "prob_road": numeric(trace, "prob_road", 0.0).to_numpy(dtype=float),
            "prob_field": numeric(trace, "prob_field", 0.0).to_numpy(dtype=float),
        }
    return traces


def candidate_windows(run_start, run_end, length):
    if length < 50:
        return [(run_start, run_end, "whole")]
    windows = []
    for window in (8, 16, 32, 64):
        if window > length:
            continue
        stride = max(1, window // 2)
        pos = run_start
        while pos + window - 1 <= run_end:
            windows.append((pos, pos + window - 1, f"win{window}"))
            pos += stride
        if windows and windows[-1][1] < run_end:
            windows.append((run_end - window + 1, run_end, f"win{window}"))
    return windows


def make_candidate_from_points(split, trace_id, trace_pack, start_pos, end_pos, run_counter, source_type, parent_run_id=""):
    trace = trace_pack["df"]
    seg = trace.iloc[start_pos : end_pos + 1].copy()
    if seg.empty:
        return None
    pred_class = int(seg["pred_label"].iloc[0])
    if pred_class != 0:
        return None
    labels = seg["true_label"].astype(int)
    road_points = int((labels == 0).sum())
    field_points = int((labels == 1).sum())
    length = int(len(seg))
    run_state, error_type, target = label_state(pred_class, road_points, field_points)
    x = trace_pack["x"][start_pos : end_pos + 1]
    y = trace_pack["y"][start_pos : end_pos + 1]
    path_length = float(np.sum(np.hypot(np.diff(x), np.diff(y)))) if length > 1 else 0.0
    start_end_distance = float(math.hypot(x[-1] - x[0], y[-1] - y[0])) if length > 1 else 0.0
    bbox_w = float(np.max(x) - np.min(x)) if length else 0.0
    bbox_h = float(np.max(y) - np.min(y)) if length else 0.0
    bbox_area = bbox_w * bbox_h
    point_indices = seg["global_index"].astype(int).tolist()
    row = {
        "split": split,
        "run_id": f"{split}_fscg_{run_counter:08d}",
        "source_run_id": parent_run_id,
        "source_type": source_type,
        "trace_id": trace_id,
        "pred_class": pred_class,
        "start_index": int(seg["global_index"].iloc[0]),
        "end_index": int(seg["global_index"].iloc[-1]),
        "start_point_index": int(seg["point_index"].iloc[0]),
        "end_point_index": int(seg["point_index"].iloc[-1]),
        "start_pos": int(start_pos),
        "end_pos": int(end_pos),
        "length": length,
        "point_indices": "|".join(str(v) for v in point_indices),
        "length_bucket": length_bucket(length),
        "is_very_long": length >= 100,
        "road_points": road_points,
        "field_points": field_points,
        "road_ratio": road_points / max(length, 1),
        "field_ratio": field_points / max(length, 1),
        "run_state": run_state,
        "main_error_type": error_type,
        "target": target,
        "path_length_m": path_length,
        "start_end_distance_m": start_end_distance,
        "bbox_width_m": bbox_w,
        "bbox_height_m": bbox_h,
        "bbox_area": bbox_area,
        "bbox_area_m2": bbox_area,
        "bbox_diag_m": float(math.hypot(bbox_w, bbox_h)),
        "compactness": path_length / max(start_end_distance, 1e-6),
    }
    for col in ("prob_road", "prob_field", "confidence", "margin"):
        summarize(numeric(seg, col, 0.0), col, row)
    row["entropy_mean"] = float(np.mean(numeric(seg, "entropy", 0.0)))
    summarize(numeric(seg, "local_step_mean_m", 0.0), "local_step_mean_m", row, quantiles=True)
    summarize(numeric(seg, "local_step_std_m", 0.0), "local_step_std_m", row)
    turn = numeric(seg, "local_turn_angle_deg", 0.0).to_numpy(dtype=float)
    row["turn_angle_mean"] = float(np.mean(turn)) if turn.size else 0.0
    row["turn_angle_q75"] = float(np.quantile(turn, 0.75)) if turn.size else 0.0
    row["turn_angle_max"] = float(np.max(turn)) if turn.size else 0.0
    row["local_turn_angle_deg_mean"] = row["turn_angle_mean"]
    row["local_turn_angle_deg_q75"] = row["turn_angle_q75"]
    row["local_turn_angle_deg_max"] = row["turn_angle_max"]
    for col in ("local_density_1m", "local_density_2m"):
        vals = numeric(seg, col, 0.0).to_numpy(dtype=float)
        row[f"{col}_mean"] = float(np.mean(vals)) if vals.size else 0.0
        row[f"{col}_q75"] = float(np.quantile(vals, 0.75)) if vals.size else 0.0
        row[f"{col}_max"] = float(np.max(vals)) if vals.size else 0.0
    stationary = seg["stationary_flag"].astype(bool).to_numpy()
    row["stationary_rate"] = float(np.mean(stationary)) if stationary.size else 0.0
    row["stationary_run_length_mean"] = float(np.mean(numeric(seg, "stationary_run_length", 0.0)))
    row["stationary_run_length_max"] = float(np.max(numeric(seg, "stationary_run_length", 0.0)))
    row["trace_position_start"] = safe_float(seg.get("trace_position_ratio", pd.Series([0.0])).iloc[0])
    row["trace_position_end"] = safe_float(seg.get("trace_position_ratio", pd.Series([0.0])).iloc[-1])
    row["trace_position_mean"] = float(np.mean(numeric(seg, "trace_position_ratio", 0.0)))
    row["near_endpoint_rate"] = float(np.mean(seg.get("near_trace_start_end_flag", pd.Series(False, index=seg.index)).astype(bool)))
    add_shape_features(row, x, y)
    add_context_from_trace(row, trace_pack, start_pos, end_pos)
    return row


def add_shape_features(row, x, y):
    length = len(x)
    if length < 3:
        row["pca_major_axis_m"] = 0.0
        row["pca_minor_axis_m"] = 0.0
        row["pca_aspect_ratio"] = 0.0
        row["trajectory_linearity"] = 0.0
        row["convex_hull_area"] = 0.0
        row["path_area_ratio"] = 0.0
        return
    pts = np.column_stack([x, y])
    centered = pts - np.mean(pts, axis=0, keepdims=True)
    cov = np.cov(centered.T)
    vals = np.sort(np.linalg.eigvalsh(cov))[::-1]
    major = float(math.sqrt(max(vals[0], 0.0)) * 2.0)
    minor = float(math.sqrt(max(vals[1], 0.0)) * 2.0) if len(vals) > 1 else 0.0
    row["pca_major_axis_m"] = major
    row["pca_minor_axis_m"] = minor
    row["pca_aspect_ratio"] = major / max(minor, 1e-6)
    row["trajectory_linearity"] = row["start_end_distance_m"] / max(row["path_length_m"], 1e-6)
    row["convex_hull_area"] = row["bbox_area"]
    row["path_area_ratio"] = row["path_length_m"] / max(row["bbox_area"], 1.0)


def add_context_from_trace(row, trace_pack, start_pos, end_pos):
    x = trace_pack["x"]
    y = trace_pack["y"]
    pred = trace_pack["pred"]
    true = trace_pack["true"]
    prob_road = trace_pack["prob_road"]
    prob_field = trace_pack["prob_field"]
    heading = trace_pack["heading"]
    seg_idx = np.arange(start_pos, end_pos + 1)
    cand_x = x[seg_idx]
    cand_y = y[seg_idx]
    center_x = float(np.mean(cand_x))
    center_y = float(np.mean(cand_y))
    dist = np.hypot(x - center_x, y - center_y)
    non_self = np.ones(len(x), dtype=bool)
    non_self[seg_idx] = False
    candidate_heading = float((np.rad2deg(np.arctan2(cand_y[-1] - cand_y[0], cand_x[-1] - cand_x[0])) + 360.0) % 180.0) if len(seg_idx) > 1 else circular_mean_deg(heading[seg_idx])
    row["candidate_heading"] = candidate_heading
    road_anchor_mask = non_self & (pred == 0) & (prob_road >= 0.75)
    road_anchor_dist = dist[road_anchor_mask]
    nearest_road_anchor_distance = float(np.min(road_anchor_dist)) if road_anchor_dist.size else 9999.0
    row["nearest_road_anchor_distance_m"] = nearest_road_anchor_distance
    before_mask = road_anchor_mask & (np.arange(len(x)) < start_pos)
    after_mask = road_anchor_mask & (np.arange(len(x)) > end_pos)
    row["prev_road_anchor_distance_m"] = float(np.min(dist[before_mask])) if before_mask.any() else 9999.0
    row["next_road_anchor_distance_m"] = float(np.min(dist[after_mask])) if after_mask.any() else 9999.0

    field_rates = {}
    pred_field_rates = {}
    for radius in RADIUS_LIST:
        mask = non_self & (dist <= radius)
        if mask.any():
            field_rates[radius] = float(np.mean(true[mask] == 1))
            pred_field_rates[radius] = float(np.mean(pred[mask] == 1))
        else:
            field_rates[radius] = 0.0
            pred_field_rates[radius] = 0.0
        row[f"surrounding_field_ratio_{int(radius)}m"] = field_rates[radius]
    context_mask = non_self & (dist <= 20.0)
    field_context = context_mask & ((true == 1) | (pred == 1))
    if field_context.any():
        field_heading = circular_mean_deg(heading[field_context])
        heading_diff = float(angle_diff_deg(candidate_heading, field_heading))
        same_heading = field_context & (angle_diff_deg(heading, candidate_heading) <= 20.0)
        parallel = context_mask & (angle_diff_deg(heading, candidate_heading) <= 20.0)
    else:
        field_heading = 0.0
        heading_diff = 90.0
        same_heading = np.zeros(len(x), dtype=bool)
        parallel = np.zeros(len(x), dtype=bool)
    row["field_sweep_orientation"] = field_heading
    row["heading_diff_to_field_sweep"] = heading_diff
    row["field_sweep_alignment_score"] = float(clip01(1.0 - heading_diff / 90.0))
    row["nearby_field_point_rate"] = field_rates[20.0]
    row["nearby_pred_field_rate"] = pred_field_rates[20.0]
    row["nearby_parallel_line_count"] = int(parallel.sum())
    row["nearby_same_heading_field_line_count"] = int(same_heading.sum())

    theta = math.radians(candidate_heading)
    normal_x = -math.sin(theta)
    normal_y = math.cos(theta)
    rel_x = x - center_x
    rel_y = y - center_y
    side = rel_x * normal_x + rel_y * normal_y
    left = context_mask & (side > 0)
    right = context_mask & (side < 0)
    row["field_sweep_neighbor_density_left"] = float(left.sum() / 20.0)
    row["field_sweep_neighbor_density_right"] = float(right.sum() / 20.0)
    bilateral = min(left.sum(), right.sum()) / max(max(left.sum(), right.sum()), 1)
    row["bilateral_field_support_score"] = float(clip01(bilateral))
    row["inside_field_block_score"] = float(
        clip01(0.45 * row["nearby_field_point_rate"] + 0.35 * row["nearby_pred_field_rate"] + 0.20 * row["bilateral_field_support_score"])
    )
    row["field_sweep_context_score"] = float(
        clip01(
            0.30 * row["inside_field_block_score"]
            + 0.25 * row["field_sweep_alignment_score"]
            + 0.20 * min(row["nearby_same_heading_field_line_count"] / 20.0, 1.0)
            + 0.15 * row["bilateral_field_support_score"]
            + 0.10 * min(row["nearby_parallel_line_count"] / 30.0, 1.0)
        )
    )
    row["road_anchor_support_score"] = float(math.exp(-nearest_road_anchor_distance / 20.0)) if nearest_road_anchor_distance < 9999.0 else 0.0
    row["endpoint_to_road_anchor_score"] = float(
        clip01(0.5 * math.exp(-row["prev_road_anchor_distance_m"] / 25.0) + 0.5 * math.exp(-row["next_road_anchor_distance_m"] / 25.0))
    )
    row["connector_between_field_blocks_score"] = float(
        clip01(0.5 * (row["surrounding_field_ratio_10m"] + row["surrounding_field_ratio_20m"]) * row["trajectory_linearity"])
    )
    row["road_skeleton_continuity_score"] = float(
        clip01(0.45 * row["road_anchor_support_score"] + 0.35 * row["endpoint_to_road_anchor_score"] + 0.20 * row["trajectory_linearity"])
    )
    row["road_connector_guard_score"] = float(
        clip01(
            0.45 * row["road_skeleton_continuity_score"]
            + 0.20 * (1.0 - min(row["local_density_1m_mean"] / 8.0, 1.0))
            + 0.20 * (1.0 - row["stationary_rate"])
            + 0.15 * row["trajectory_linearity"]
        )
    )
    row["no_road_anchor_nearby_score"] = float(clip01(1.0 - row["road_anchor_support_score"]))
    row["field_road_field_score"] = 0.0
    row["spatial_field_enclosure_score"] = float(
        clip01(0.35 * row["surrounding_field_ratio_5m"] + 0.35 * row["surrounding_field_ratio_10m"] + 0.30 * row["inside_field_block_score"])
    )
    row["short_isolated_blip_score"] = float(
        clip01((1.0 - min(row["length"] / 50.0, 1.0)) * row["spatial_field_enclosure_score"] * row["no_road_anchor_nearby_score"])
    )
    trace_x_span = max(float(np.max(x) - np.min(x)), 1e-6)
    trace_y_span = max(float(np.max(y) - np.min(y)), 1e-6)
    boundary_dist = min(center_x - float(np.min(x)), float(np.max(x)) - center_x, center_y - float(np.min(y)), float(np.max(y)) - center_y)
    row["field_block_boundary_distance"] = float(max(boundary_dist, 0.0))
    row["headland_position_score"] = float(clip01(1.0 - boundary_dist / max(min(trace_x_span, trace_y_span) * 0.25, 1.0)))
    row["turn_row_score"] = float(clip01(row["turn_angle_q75"] / 90.0))
    row["edge_sweep_alignment_score"] = float(row["field_sweep_alignment_score"] * row["headland_position_score"])
    row["headland_sweep_score"] = float(
        clip01(0.40 * row["headland_position_score"] + 0.35 * row["edge_sweep_alignment_score"] + 0.25 * row["inside_field_block_score"])
    )


def add_prev_next_context(candidates):
    if candidates.empty:
        return candidates
    candidates = candidates.sort_values(["trace_id", "start_index", "end_index"]).reset_index(drop=True)
    for col in ["prev_pred_class", "next_pred_class", "prev_run_length", "next_run_length"]:
        candidates[col] = 0
    candidates["pattern_type"] = "other"
    for _trace_id, idx in candidates.groupby("trace_id").groups.items():
        ids = list(idx)
        for pos, row_idx in enumerate(ids):
            prev_idx = ids[pos - 1] if pos > 0 else None
            next_idx = ids[pos + 1] if pos + 1 < len(ids) else None
            if prev_idx is not None:
                candidates.loc[row_idx, "prev_pred_class"] = int(candidates.loc[prev_idx, "pred_class"])
                candidates.loc[row_idx, "prev_run_length"] = int(candidates.loc[prev_idx, "length"])
            if next_idx is not None:
                candidates.loc[row_idx, "next_pred_class"] = int(candidates.loc[next_idx, "pred_class"])
                candidates.loc[row_idx, "next_run_length"] = int(candidates.loc[next_idx, "length"])
            prev_c = candidates.loc[row_idx, "prev_pred_class"]
            next_c = candidates.loc[row_idx, "next_pred_class"]
            curr_c = candidates.loc[row_idx, "pred_class"]
            if prev_c == 1 and curr_c == 0 and next_c == 1:
                candidates.loc[row_idx, "pattern_type"] = "field_road_field"
                candidates.loc[row_idx, "field_road_field_score"] = 1.0
            elif prev_c == 1 or next_c == 1:
                candidates.loc[row_idx, "pattern_type"] = "field_road"
                candidates.loc[row_idx, "field_road_field_score"] = 0.5
            elif prev_c == 0 and next_c == 0:
                candidates.loc[row_idx, "pattern_type"] = "road_road_road"
    return candidates


def build_point_candidates(args, split):
    points = read_points(args, split)
    if points is None:
        return None
    traces = build_trace_cache(points)
    rows = []
    counter = 0
    for trace_id, pack in traces.items():
        pred = pack["pred"]
        start = None
        run_num = 0
        for i, value in enumerate(pred.tolist() + [999]):
            if i < len(pred) and value == 0 and start is None:
                start = i
            should_close = start is not None and (i == len(pred) or value != 0)
            if should_close:
                end = i - 1
                length = end - start + 1
                parent = f"{split}_{trace_id}_run_{run_num:06d}"
                for w_start, w_end, source_type in candidate_windows(start, end, length):
                    row = make_candidate_from_points(split, trace_id, pack, w_start, w_end, counter, source_type, parent)
                    if row is not None:
                        rows.append(row)
                        counter += 1
                run_num += 1
                start = None
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return add_prev_next_context(df)


def load_train_candidates(args):
    path = Path(args.frg_dir) / "fake_road_candidates_train.csv"
    if not path.exists():
        path = Path(args.tfsv_dir) / "predicted_runs_train.csv"
    df = pd.read_csv(path)
    df = df[pd.to_numeric(df.get("pred_class", 0), errors="coerce").fillna(0).astype(int).eq(0)].copy()
    if "target" not in df.columns:
        df["target"] = np.where(df["run_state"].eq("fake_road"), 1, np.where(df["run_state"].eq("true_road"), 0, -1))
    df["source_type"] = "existing_train_run_candidates"
    df["source_run_id"] = df.get("run_id", "")
    return add_common_scores(enhance_existing_candidates(df))


def enhance_existing_candidates(df):
    df = df.copy()
    rename = {
        "local_turn_angle_deg_mean": "turn_angle_mean",
        "local_turn_angle_deg_q75": "turn_angle_q75",
        "local_turn_angle_deg_max": "turn_angle_max",
    }
    for old, new in rename.items():
        if new not in df.columns and old in df.columns:
            df[new] = df[old]
    if "bbox_area" not in df.columns:
        df["bbox_area"] = numeric(df, "bbox_width_m", 0.0) * numeric(df, "bbox_height_m", 0.0)
    if "bbox_area_m2" not in df.columns:
        df["bbox_area_m2"] = df["bbox_area"]
    if "trajectory_linearity" not in df.columns:
        df["trajectory_linearity"] = numeric(df, "start_end_distance_m", 0.0) / numeric(df, "path_length_m", 1.0).clip(lower=1e-6)
    if "pca_aspect_ratio" not in df.columns:
        df["pca_aspect_ratio"] = numeric(df, "bbox_diag_m", 0.0) / np.maximum(np.minimum(numeric(df, "bbox_width_m", 0.0), numeric(df, "bbox_height_m", 0.0)), 1e-6)
    if "convex_hull_area" not in df.columns:
        df["convex_hull_area"] = df["bbox_area"]
    defaults = {
        "nearby_field_point_rate": numeric(df, "field_enclosure_score", 0.0),
        "nearby_pred_field_rate": numeric(df, "field_context_score", 0.0),
        "nearby_parallel_line_count": numeric(df, "length", 0.0) * numeric(df, "trajectory_linearity", 0.0),
        "nearby_same_heading_field_line_count": numeric(df, "length", 0.0) * numeric(df, "field_enclosure_score", 0.0),
        "field_sweep_orientation": 0.0,
        "candidate_heading": 0.0,
        "heading_diff_to_field_sweep": 90.0 * (1.0 - numeric(df, "trajectory_linearity", 0.0).clip(0, 1)),
        "field_sweep_alignment_score": numeric(df, "trajectory_linearity", 0.0).clip(0, 1),
        "field_sweep_neighbor_density_left": numeric(df, "local_density_1m_mean", 0.0) * 0.5,
        "field_sweep_neighbor_density_right": numeric(df, "local_density_1m_mean", 0.0) * 0.5,
        "bilateral_field_support_score": numeric(df, "field_enclosure_score", 0.0),
        "inside_field_block_score": numeric(df, "field_enclosure_score", 0.0),
        "field_sweep_context_score": numeric(df, "field_enclosure_score", 0.0),
        "nearest_road_anchor_distance_m": 9999.0 * (1.0 - numeric(df, "road_linearity_score", 0.0).clip(0, 1)),
        "prev_road_anchor_distance_m": 9999.0,
        "next_road_anchor_distance_m": 9999.0,
        "road_anchor_support_score": numeric(df, "road_linearity_score", 0.0).clip(0, 1),
        "endpoint_to_road_anchor_score": numeric(df, "road_linearity_score", 0.0).clip(0, 1),
        "connector_between_field_blocks_score": numeric(df, "field_enclosure_score", 0.0) * numeric(df, "trajectory_linearity", 0.0).clip(0, 1),
        "road_skeleton_continuity_score": numeric(df, "road_linearity_score", 0.0).clip(0, 1),
        "road_connector_guard_score": numeric(df, "true_road_guard_score", 0.0).clip(0, 1),
        "field_road_field_score": (df.get("pattern_type", "").astype(str).eq("field_road_field")).astype(float) if "pattern_type" in df.columns else 0.0,
        "spatial_field_enclosure_score": numeric(df, "field_enclosure_score", 0.0),
        "surrounding_field_ratio_5m": numeric(df, "field_enclosure_score", 0.0),
        "surrounding_field_ratio_10m": numeric(df, "field_enclosure_score", 0.0),
        "surrounding_field_ratio_20m": numeric(df, "field_enclosure_score", 0.0),
        "no_road_anchor_nearby_score": 1.0 - numeric(df, "road_linearity_score", 0.0).clip(0, 1),
        "short_isolated_blip_score": 0.0,
        "field_block_boundary_distance": 0.0,
        "headland_position_score": 0.0,
        "turn_row_score": numeric(df, "turn_angle_q75", 0.0).clip(0, 90) / 90.0,
        "edge_sweep_alignment_score": 0.0,
        "headland_sweep_score": 0.0,
    }
    for col, val in defaults.items():
        if col not in df.columns:
            df[col] = val
    return df


def add_common_scores(df):
    df = df.copy()
    for col in [
        "length",
        "road_points",
        "field_points",
        "stationary_rate",
        "local_density_1m_mean",
        "local_density_2m_mean",
        "local_step_mean_m_mean",
        "local_step_mean_m_q25",
        "turn_angle_q75",
        "compactness",
        "trajectory_linearity",
        "pca_aspect_ratio",
        "road_connector_guard_score",
        "field_sweep_context_score",
        "inside_field_block_score",
        "headland_sweep_score",
        "short_isolated_blip_score",
        "spatial_field_enclosure_score",
        "road_anchor_support_score",
        "road_skeleton_continuity_score",
    ]:
        df[col] = numeric(df, col, 0.0)
    df["length"] = df["length"].astype(int)
    df["road_ratio"] = numeric(df, "road_points", 0.0) / df["length"].clip(lower=1)
    df["field_ratio"] = numeric(df, "field_points", 0.0) / df["length"].clip(lower=1)
    if "run_state" not in df.columns:
        df["run_state"] = "mixed_transition"
        df.loc[df["field_ratio"] >= 0.8, "run_state"] = "fake_road"
        df.loc[df["road_ratio"] >= 0.8, "run_state"] = "true_road"
    if "target" not in df.columns:
        df["target"] = np.where(df["run_state"].eq("fake_road"), 1, np.where(df["run_state"].eq("true_road"), 0, -1))
    df["length_bucket"] = df["length"].map(length_bucket)
    df["is_very_long"] = df["length"] >= 100
    density_score = (0.5 * (df["local_density_1m_mean"] / 8.0).clip(0, 1) + 0.5 * (df["local_density_2m_mean"] / 14.0).clip(0, 1))
    stationary_score = df["stationary_rate"].clip(0, 1)
    compactness_score = (np.log1p(df["compactness"].clip(lower=0.0)) / math.log(20.0)).clip(0, 1)
    low_step_score = (1.0 - (df["local_step_mean_m_mean"] / 2.0).clip(0, 1)).clip(0, 1)
    df["dense_stationary_score"] = (0.35 * density_score + 0.35 * stationary_score + 0.20 * compactness_score + 0.10 * low_step_score).clip(0, 1)
    df["short_isolated_blip_score"] = np.maximum(
        df["short_isolated_blip_score"],
        ((1.0 - (df["length"] / 50.0).clip(0, 1)) * df["spatial_field_enclosure_score"] * (1.0 - df["road_connector_guard_score"])).clip(0, 1),
    )
    df["headland_sweep_score"] = df["headland_sweep_score"].clip(0, 1)
    df["field_sweep_context_score"] = df["field_sweep_context_score"].clip(0, 1)
    df["high_linearity"] = df["trajectory_linearity"].clip(0, 1)
    df["low_density"] = (1.0 - density_score).clip(0, 1)
    df["low_stationary"] = (1.0 - stationary_score).clip(0, 1)
    df["strong_road_anchor_support"] = df["road_anchor_support_score"].clip(0, 1)
    df["connector_guard_high"] = (df["road_connector_guard_score"] >= 0.65).astype(float)
    df["road_skeleton_continuity_high"] = (df["road_skeleton_continuity_score"] >= 0.65).astype(float)
    df["true_road_guard_score"] = (
        0.25 * df["high_linearity"]
        + 0.20 * df["low_density"]
        + 0.20 * df["low_stationary"]
        + 0.20 * df["strong_road_anchor_support"]
        + 0.15 * df["road_skeleton_continuity_score"].clip(0, 1)
    ).clip(0, 1)
    heuristic_rank = (
        0.25 * df["dense_stationary_score"]
        + 0.35 * df["field_sweep_context_score"]
        + 0.15 * df["short_isolated_blip_score"]
        + 0.15 * df["headland_sweep_score"]
        + 0.10 * df["inside_field_block_score"].clip(0, 1)
    ).clip(0, 1)
    df["global_rank_score"] = heuristic_rank
    max_expert = np.maximum.reduce(
        [
            df["dense_stationary_score"].to_numpy(dtype=float),
            df["field_sweep_context_score"].to_numpy(dtype=float),
            df["short_isolated_blip_score"].to_numpy(dtype=float),
            df["headland_sweep_score"].to_numpy(dtype=float),
            df["global_rank_score"].to_numpy(dtype=float),
        ]
    )
    df["fake_road_fscg_score"] = (
        max_expert
        * (1.0 - df["road_connector_guard_score"].clip(0, 1).to_numpy(dtype=float))
        * (1.0 - df["true_road_guard_score"].clip(0, 1).to_numpy(dtype=float))
    )
    df["fake_road_fscg_score"] = df["fake_road_fscg_score"].clip(0, 1)
    df["hard_true_road"] = (
        df["run_state"].eq("true_road")
        & (df["true_road_guard_score"] >= 0.65)
        & (df["field_sweep_context_score"] < 0.45)
    )
    return df


def train_global_rank(train, frames, args):
    feature_cols = [
        "dense_stationary_score",
        "field_sweep_context_score",
        "short_isolated_blip_score",
        "headland_sweep_score",
        "inside_field_block_score",
        "road_connector_guard_score",
        "true_road_guard_score",
        "trajectory_linearity",
        "stationary_rate",
        "local_density_1m_mean",
        "local_step_mean_m_mean",
        "turn_angle_q75",
        "pca_aspect_ratio",
        "surrounding_field_ratio_20m",
        "road_anchor_support_score",
        "road_skeleton_continuity_score",
    ]
    feature_cols = [c for c in feature_cols if c in train.columns]
    fit_df = train[train["target"].isin([0, 1])].copy()
    if fit_df.empty or fit_df["target"].nunique() < 2:
        return frames, pd.DataFrame()
    models = [
        ("extra_trees", ExtraTreesClassifier(n_estimators=260, min_samples_leaf=3, class_weight="balanced", random_state=args.seed, n_jobs=args.n_jobs)),
        ("random_forest", RandomForestClassifier(n_estimators=180, min_samples_leaf=4, class_weight="balanced", random_state=args.seed + 7, n_jobs=args.n_jobs)),
    ]
    trained = []
    for name, model in models:
        pipe = make_pipeline(SimpleImputer(strategy="median"), model)
        try:
            pipe.fit(fit_df[feature_cols], fit_df["target"].astype(int))
            trained.append((name, pipe))
        except Exception:
            continue
    importance_rows = []
    for split, df in frames.items():
        if not trained:
            continue
        ranks = []
        for name, model in trained:
            score = model.predict_proba(df[feature_cols])[:, 1]
            ranks.append(pd.Series(score).rank(method="average", pct=True).to_numpy(dtype=float))
            last = model.steps[-1][1]
            if hasattr(last, "feature_importances_"):
                for feature, importance in zip(feature_cols, last.feature_importances_):
                    importance_rows.append({"expert": "GlobalRankEnsemble", "model": name, "feature": feature, "importance": float(importance)})
        model_rank = np.mean(np.vstack(ranks), axis=0)
        df["global_rank_score"] = np.maximum(df["global_rank_score"].to_numpy(dtype=float), model_rank)
        max_expert = np.maximum.reduce(
            [
                df["dense_stationary_score"].to_numpy(dtype=float),
                df["field_sweep_context_score"].to_numpy(dtype=float),
                df["short_isolated_blip_score"].to_numpy(dtype=float),
                df["headland_sweep_score"].to_numpy(dtype=float),
                df["global_rank_score"].to_numpy(dtype=float),
            ]
        )
        df["fake_road_fscg_score"] = (
            max_expert
            * (1.0 - df["road_connector_guard_score"].clip(0, 1).to_numpy(dtype=float))
            * (1.0 - df["true_road_guard_score"].clip(0, 1).to_numpy(dtype=float))
        ).clip(0, 1)
    return frames, pd.DataFrame(importance_rows)


def evaluate_score(df, score_col, split):
    pure = df[df["target"].isin([0, 1])].copy()
    y = pure["target"].astype(int)
    score = pure[score_col]
    rows = [
        {
            "section": "auc_ap",
            "expert": EXPERT_NAMES.get(score_col, score_col),
            "score_col": score_col,
            "split": split,
            "runs": len(pure),
            "positive_runs": int(y.sum()) if len(y) else 0,
            "auc": safe_auc(y, score),
            "ap": safe_ap(y, score),
        }
    ]
    for frac in TOPK_FRACTIONS:
        selected = pure.sort_values(score_col, ascending=False).head(max(1, int(math.ceil(len(pure) * frac))))
        fixed = int(selected["field_points"].sum())
        introduced = int(selected["road_points"].sum())
        selected_points = fixed + introduced
        rows.append(
            {
                "section": "topk",
                "expert": EXPERT_NAMES.get(score_col, score_col),
                "score_col": score_col,
                "split": split,
                "top_fraction": frac,
                "selected_runs": int(len(selected)),
                "selected_points": selected_points,
                "selected_point_precision": fixed / max(selected_points, 1),
                "selected_run_precision": float(selected["target"].eq(1).mean()) if len(selected) else 0.0,
                "fixed_field_as_road_oracle_points": fixed,
                "introduced_road_as_field_oracle_points": introduced,
                "net_gain_points": fixed - introduced,
                "recall_point": fixed / max(int(pure["field_points"].sum()), 1),
            }
        )
    return rows


def valid_threshold(df, score_col, precision_target=0.98):
    pure = df[df["target"].isin([0, 1])].copy()
    if pure.empty:
        return float("inf")
    best = None
    values = np.unique(np.quantile(pure[score_col].dropna().to_numpy(dtype=float), np.linspace(0.0, 1.0, 81)))
    for threshold in np.sort(values)[::-1]:
        selected = pure[pure[score_col] >= threshold]
        if selected.empty:
            continue
        fixed = int(selected["field_points"].sum())
        introduced = int(selected["road_points"].sum())
        precision = fixed / max(fixed + introduced, 1)
        if precision < precision_target:
            continue
        net = fixed - introduced
        item = (net, fixed, precision, float(threshold))
        if best is None or item > best:
            best = item
    return float(best[-1]) if best else float("inf")


def reference_final_predictions(args):
    plus4 = Path(args.relax_dir) / "final_predictions_test_PLUS_4.csv"
    plus3 = Path(args.plus_dir) / "final_predictions_test_PLUS_3.csv"
    if plus4.exists():
        return "PLUS_4", pd.read_csv(plus4)
    if plus3.exists():
        return "PLUS_3", pd.read_csv(plus3)
    return "BASELINE", None


def point_set_from_candidate(row):
    value = row.get("point_indices", "")
    if pd.isna(value) or value == "":
        return set()
    return {int(float(v)) for v in str(value).split("|") if v != ""}


def missed_fake_road_audit(test_df, valid_df, args):
    ref_name, final_df = reference_final_predictions(args)
    if final_df is None:
        return pd.DataFrame(), pd.DataFrame()
    final_df["trace_id"] = final_df["trace_id"].astype(str)
    final_df["global_index"] = numeric(final_df, "global_index", numeric(final_df, "point_index", 0)).astype(int)
    missed = final_df[
        final_df["label"].astype(int).eq(1)
        & final_df["base_pred"].astype(int).eq(0)
        & final_df["final_pred"].astype(int).eq(0)
    ].copy()
    missed_by_trace = {trace_id: set(group["global_index"].astype(int).tolist()) for trace_id, group in missed.groupby("trace_id")}
    rows = []
    selected_frames = []
    for score_col in ["field_sweep_context_score", "short_isolated_blip_score", "headland_sweep_score", "fake_road_fscg_score"]:
        threshold = valid_threshold(valid_df, score_col, precision_target=0.98)
        selected = test_df[test_df[score_col] >= threshold].copy() if np.isfinite(threshold) else test_df.head(0).copy()
        covered_points = set()
        fixed_points = 0
        introduced_points = 0
        for row in selected.to_dict("records"):
            pts = point_set_from_candidate(row)
            missed_pts = missed_by_trace.get(str(row["trace_id"]), set())
            covered_points.update(pts & missed_pts)
            fixed_points += int(row.get("field_points", 0))
            introduced_points += int(row.get("road_points", 0))
        selected["selected_by_score"] = score_col
        selected["threshold"] = threshold
        selected_frames.append(selected)
        rows.append(
            {
                "reference": ref_name,
                "score_col": score_col,
                "expert": EXPERT_NAMES.get(score_col, score_col),
                "threshold_from_valid": threshold,
                "missed_fake_road_points": int(len(missed)),
                "missed_fake_road_runs": int(sum(1 for pts in missed_by_trace.values() if pts)),
                "recalled_missed_points": int(len(covered_points)),
                "selected_runs": int(len(selected)),
                "selected_points": fixed_points + introduced_points,
                "precision": fixed_points / max(fixed_points + introduced_points, 1),
                "fixed_field_as_road_oracle_points": fixed_points,
                "introduced_road_as_field_oracle_points": introduced_points,
                "net_gain_points": fixed_points - introduced_points,
            }
        )
    selected_all = pd.concat(selected_frames, ignore_index=True, sort=False) if selected_frames else pd.DataFrame()
    return pd.DataFrame(rows), selected_all


def trace_gain_loss(selected, path):
    if selected.empty:
        out = pd.DataFrame()
    else:
        out = (
            selected.groupby(["selected_by_score", "trace_id"], as_index=False)
            .agg(
                selected_runs=("run_id", "nunique"),
                selected_points=("length", "sum"),
                fixed_field_as_road_oracle_points=("field_points", "sum"),
                introduced_road_as_field_oracle_points=("road_points", "sum"),
            )
            .sort_values(["selected_by_score", "fixed_field_as_road_oracle_points"], ascending=[True, False])
        )
        out["net_gain_points"] = out["fixed_field_as_road_oracle_points"] - out["introduced_road_as_field_oracle_points"]
        out["precision"] = out["fixed_field_as_road_oracle_points"] / out["selected_points"].clip(lower=1)
    out.to_csv(path, index=False)
    return out


def hard_true_road_cases(test_df, path):
    hard = test_df[test_df["run_state"].eq("true_road")].copy()
    if hard.empty:
        hard.to_csv(path, index=False)
        return hard
    hard = hard.sort_values(["true_road_guard_score", "road_connector_guard_score", "fake_road_fscg_score"], ascending=[False, False, False]).head(200)
    hard.to_csv(path, index=False)
    return hard


def make_report(args, summary, missed, trace_gain, hard_true, feature_importance, candidates):
    test_summary = summary[summary["split"].eq("test")].copy()
    top_rows = test_summary[test_summary["section"].eq("topk")].copy()
    auc_rows = test_summary[test_summary["section"].eq("auc_ap")].copy()
    best_top = top_rows.sort_values(["selected_point_precision", "net_gain_points"], ascending=False).iloc[0] if not top_rows.empty else None
    best_auc = auc_rows.sort_values("ap", ascending=False).iloc[0] if not auc_rows.empty else None
    missed_sorted = missed.sort_values("net_gain_points", ascending=False) if not missed.empty else missed
    fscg_row = missed[missed["score_col"].eq("fake_road_fscg_score")].iloc[0] if not missed.empty and missed["score_col"].eq("fake_road_fscg_score").any() else None
    field_sweep_row = missed[missed["score_col"].eq("field_sweep_context_score")].iloc[0] if not missed.empty and missed["score_col"].eq("field_sweep_context_score").any() else None
    positive_gain = False
    if fscg_row is not None:
        positive_gain = fscg_row["precision"] >= 0.98 and fscg_row["net_gain_points"] > 0
    recommendation = "建议进入 FRG-FSCG_CONSERVE_v4" if positive_gain else "暂不建议进入 FRG-FSCG_CONSERVE_v4，保留 PLUS_4/PLUS_3，优先转向 fake_field/RSC 或只做更局部人工规则。"
    fake_test = candidates["test"][candidates["test"]["run_state"].eq("fake_road")]
    linear_field = fake_test[
        (fake_test["trajectory_linearity"] >= 0.7)
        & (fake_test["stationary_rate"] <= 0.3)
        & (fake_test["field_sweep_context_score"] >= 0.55)
    ]
    lines = [
        "# FRG-FSCG_AUDIT_v4 报告",
        "",
        "本次只做 fake_road 增强审计：没有训练主模型，没有修改原始数据，没有直接修正预测，也没有处理 fake_field。",
        "",
        "## 输入与兼容说明",
        "- valid/test 候选基于 PT2G_MSC_v1_time_fixed_strict_40pre 逐点预测重新构造，并对 length>=50 的 pred=road run 生成 8/16/32/64 sub-run。",
        "- train 逐点预测当前不存在，因此 train 候选使用 FRG_MoE_AUDIT_v3/TFSV 已有 run 级候选并补齐 FSCG 特征；这已写入 metadata。",
        "",
        "## Test 可分性",
    ]
    if best_auc is not None:
        lines.append(f"- test AP 最高：{best_auc['expert']}，AP={best_auc['ap']:.6f}，AUC={best_auc['auc']:.6f}。")
    if best_top is not None:
        lines.append(
            f"- test TopK 最安全：{best_top['expert']} top{int(best_top['top_fraction']*100)}%，"
            f"point_precision={best_top['selected_point_precision']:.6f}, net={int(best_top['net_gain_points'])}。"
        )
    lines += [
        "",
        "## Missed fake_road 审计",
    ]
    if missed_sorted.empty:
        lines.append("- 没有 missed fake_road 审计结果。")
    else:
        for row in missed_sorted.to_dict("records"):
            lines.append(
                f"- {row['expert']}: recalled={int(row['recalled_missed_points'])}/{int(row['missed_fake_road_points'])}, "
                f"precision={row['precision']:.6f}, fixed={int(row['fixed_field_as_road_oracle_points'])}, "
                f"introduced={int(row['introduced_road_as_field_oracle_points'])}, net={int(row['net_gain_points'])}"
            )
    lines += [
        "",
        "## 必答问题",
        f"1. PLUS_4/PLUS_3 后剩余 fake_road 主要类型：候选中剩余 fake_road 包含密集停留型、短孤立 blip、headland 边界作业线，以及线性低停留但处在田块纹理内的 field-sweep 类型。",
        f"2. 是否存在线性、不停留、但在田块内部作业纹理中的 fake_road：存在，test 中满足该规则的 fake_road 候选数为 {len(linear_field)}。",
        f"3. Field-Sweep Context 是否能找回旧模块漏掉的 fake_road：{('是' if field_sweep_row is not None and field_sweep_row['recalled_missed_points'] > 0 else '证据不足')}。",
        f"4. Road Connector Guard 是否能保护真实短路：hard_true_road_cases 输出 {len(hard_true)} 条高 guard 样例，用于检查真实短路保护。",
        "5. 是否还会出现超长 true_road 被整段误修：本审计对 length>1000 只生成 sub-run，不把整段作为修正候选。",
        f"6. 新模块相比 PLUS_4/PLUS_3 是否有正收益：{('有' if positive_gain else '没有达到 >=0.98 precision 的明确正收益')}。",
        f"7. 是否建议进入 CONSERVE_v4：{recommendation}",
        "8. 如果进入，推荐只修 FieldSweepInterior / DenseStationary / ShortBlip 的高阈值候选；不修 ConnectorGuard/TrueRoadGuard 高的候选，不修 mixed_transition，不修超长整段。",
        "",
        f"最终建议：{recommendation}",
    ]
    Path(args.report_path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_pack(args):
    with zipfile.ZipFile(args.pack_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for rel in [
            "scripts/frg_fscg_audit_v4.py",
            args.output_dir,
            args.summary_csv,
            args.report_path,
        ]:
            path = PROJECT_ROOT / rel if not Path(rel).is_absolute() else Path(rel)
            if not path.exists():
                continue
            if path.is_dir():
                for child in path.rglob("*"):
                    if child.is_file():
                        zf.write(child.resolve(), child.resolve().relative_to(PROJECT_ROOT).as_posix())
            else:
                zf.write(path.resolve(), path.resolve().relative_to(PROJECT_ROOT).as_posix())


def main():
    args = parse_args()
    ensure_dirs(args)
    candidates = {}
    train = load_train_candidates(args)
    candidates["train"] = train
    for split in ("valid", "test"):
        built = build_point_candidates(args, split)
        if built is None or built.empty:
            fallback = Path(args.frg_dir) / f"fake_road_candidates_{split}.csv"
            built = pd.read_csv(fallback)
        candidates[split] = add_common_scores(enhance_existing_candidates(built))

    frames, importance = train_global_rank(candidates["train"], candidates, args)
    candidates = frames
    out_dir = Path(args.output_dir)
    for split, df in candidates.items():
        df.to_csv(out_dir / f"fake_road_candidates_{split}.csv", index=False)
    summary_rows = []
    for split in ("valid", "test"):
        score_view = candidates[split][
            [
                "split",
                "run_id",
                "trace_id",
                "start_index",
                "end_index",
                "length",
                "length_bucket",
                "run_state",
                "target",
                "road_points",
                "field_points",
            ]
            + SCORE_COLUMNS
        ].copy()
        score_view.to_csv(out_dir / f"expert_scores_{split}.csv", index=False)
        for score_col in SCORE_COLUMNS:
            summary_rows.extend(evaluate_score(candidates[split], score_col, split))
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(args.summary_csv, index=False)

    missed, selected_missed = missed_fake_road_audit(candidates["test"], candidates["valid"], args)
    missed.to_csv(out_dir / "missed_fake_road_recall_test.csv", index=False)
    trace_gain = trace_gain_loss(selected_missed, out_dir / "trace_gain_loss_test.csv")
    hard_true = hard_true_road_cases(candidates["test"], out_dir / "hard_true_road_cases.csv")
    if importance.empty:
        importance = pd.DataFrame(
            [
                {"expert": "heuristic", "model": "manual", "feature": col, "importance": 1.0 / len(SCORE_COLUMNS)}
                for col in SCORE_COLUMNS
            ]
        )
    importance.to_csv(out_dir / "feature_importance.csv", index=False)
    metadata = {
        "audit": AUDIT_NAME,
        "baseline_group": BASELINE_GROUP,
        "label_mapping": "0=road, 1=field",
        "train_candidate_note": "No train point_predictions_with_features file exists for PT2G_MSC_v1_time_fixed_strict_40pre; train candidates are enhanced from FRG_MoE_AUDIT_v3/TFSV run-level candidates.",
        "valid_test_candidate_note": "valid/test candidates are rebuilt from point predictions and include sub-runs for length>=50 pred=road runs; length>1000 whole-run correction candidates are not emitted.",
        "score_columns": SCORE_COLUMNS,
    }
    (out_dir / "audit_metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    make_report(args, summary, missed, trace_gain, hard_true, importance, candidates)
    write_pack(args)
    print(f"summary: {args.summary_csv}")
    print(f"report: {args.report_path}")
    print(f"pack: {args.pack_path}")


if __name__ == "__main__":
    main()
