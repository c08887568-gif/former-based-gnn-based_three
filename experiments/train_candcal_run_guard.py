import csv
import json
import math
import sys
import time
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataset import CachedGraphDataset
from fieldroaddatapipeline.dataloader import FieldRoadDataLoader
from fine_tune import (
    build_data,
    build_model,
    get_cached_aux_features,
    load_graph_cache_split,
    merge_graph_cache_edges,
    normalize_trace_id,
    unpack_batch,
)
from models.CandidateRunFieldGuardCalibration import (
    CANDCAL_RUN_GUARD_DEFAULTS,
    CandidateRunFieldGuardCalibration,
    compute_pos_weight,
)
from utils.motion_state_features import AUX_FEATURE_NAMES, haversine_m
from utils.threading_config import apply_torch_thread_config, configure_default_threads
from utils.utils import get_default_device


configure_default_threads()
apply_torch_thread_config(torch)


RUN_NAME = "PT2G_MSC_RC_CANDCAL_RUN_GUARD_v1"
RUN_DIR = Path("runs") / RUN_NAME
DIAG_DIR = Path("diagnostics") / RUN_NAME
RESULT_PATH = Path("results") / f"{RUN_NAME}_summary.csv"
REPORT_PATH = Path("analysis") / f"{RUN_NAME}_report.md"
PACK_PATH = Path("analysis_packs") / f"{RUN_NAME}_for_chatgpt.zip"
LOG_PATH = Path("logs/candcal_run_guard/CANDCAL_RUN_GUARD_v1.log")

POINT_GUARD_DELTA_V2 = {
    "test_macro_f1": 0.887351,
    "field_as_road": 3431,
    "long_field_as_road_segments": 33,
}

COMMON_CONFIG = {
    "use_pretrain": True,
    "pretrained_path": "weights/PT2_edge_weight_pretrain.pt",
    "effective_pretrained_path": None,
    "pretrain_mode": "edge_weight",
    "cache_dir": "cache/wheat_non_iid",
    "graph_cache_path": "cache/pretrained_graphs/PT2G_topk3",
    "segment_context_mode": "msc",
    "msc_aux_mode": "rc",
    "skip_test": False,
    "run_name": RUN_NAME,
}

SCALAR_FEATURE_NAMES = [
    "run_length",
    "trace_position_start",
    "trace_position_end",
    "trace_position_mean",
    "near_endpoint_rate",
    "prob_road_mean",
    "prob_road_std",
    "prob_road_min",
    "prob_road_max",
    "prob_field_mean",
    "margin_mean",
    "margin_std",
    "confidence_mean",
    "stationary_rate",
    "stationary_run_length_mean",
    "stationary_run_length_max",
    "local_density_1m_mean",
    "local_density_1m_max",
    "local_density_1m_q75",
    "local_density_2m_mean",
    "local_step_mean_m_mean",
    "local_step_mean_m_min",
    "local_step_mean_m_q25",
    "local_step_std_m_mean",
    "local_turn_angle_deg_mean",
    "local_turn_angle_deg_max",
    "start_end_distance_m",
    "bbox_width_m",
    "bbox_height_m",
    "bbox_diag_m",
    "path_length_m",
    "compactness",
    "branch_disagreement_road_mean",
    "branch_disagreement_road_max",
    "branch_disagreement_field_mean",
    "branch_disagreement_field_max",
]
ENHANCED_FEATURE_NAMES = [f"enhanced_feature_mean_{idx:03d}" for idx in range(216)]
FEATURE_NAMES = SCALAR_FEATURE_NAMES + ENHANCED_FEATURE_NAMES

RUN_CSV_FIELDS = [
    "candidate_run_id",
    "split",
    "trace_id",
    "start_index",
    "end_index",
    "length",
    "point_indices",
    "label_road_points",
    "label_field_points",
    "field_ratio",
    "road_ratio",
    "run_target",
    "target_status",
    "run_type",
    "run_guard_score",
    "selected_by_best_threshold",
    "enhanced_feature_mean_norm",
] + SCALAR_FEATURE_NAMES

THRESHOLD_FIELDS = [
    "epoch",
    "threshold",
    "base_accuracy",
    "base_macro_f1",
    "base_road_f1",
    "base_field_f1",
    "base_pred_road_rate",
    "base_pred_field_rate",
    "base_road_as_field",
    "base_field_as_road",
    "cal_accuracy",
    "cal_macro_f1",
    "cal_road_f1",
    "cal_field_f1",
    "cal_pred_road_rate",
    "cal_pred_field_rate",
    "cal_road_as_field",
    "cal_field_as_road",
    "delta_macro_f1",
    "delta_road_f1",
    "delta_field_f1",
    "delta_road_as_field",
    "delta_field_as_road",
    "candidate_runs",
    "candidate_run_points",
    "selected_runs",
    "selected_run_points",
    "guard_applied_point_rate",
    "fixed_field_as_road",
    "introduced_road_as_field",
    "fixed_minus_introduced",
    "selected_false_run_rate",
]

SUMMARY_FIELDS = [
    "group",
    "best_epoch",
    "best_threshold",
    "base_test_accuracy",
    "base_test_macro_f1",
    "base_test_road_f1",
    "base_test_field_f1",
    "base_pred_road_rate",
    "base_road_as_field",
    "base_field_as_road",
    "base_long_road_as_field_segments",
    "base_long_field_as_road_segments",
    "base_long_field_as_road_points",
    "cal_test_accuracy",
    "cal_test_macro_f1",
    "cal_test_road_f1",
    "cal_test_field_f1",
    "cal_pred_road_rate",
    "cal_road_as_field",
    "cal_field_as_road",
    "cal_long_road_as_field_segments",
    "cal_long_field_as_road_segments",
    "cal_long_field_as_road_points",
    "delta_macro_f1",
    "delta_road_f1",
    "delta_field_f1",
    "delta_road_as_field",
    "delta_field_as_road",
    "delta_long_field_as_road_segments",
    "delta_long_field_as_road_points",
    "candidate_runs",
    "candidate_run_points",
    "selected_runs",
    "selected_run_points",
    "guard_applied_point_rate",
    "fixed_field_as_road",
    "introduced_road_as_field",
    "fixed_minus_introduced",
]

POINT_TEST_FIELDS = [
    "split",
    "trace_id",
    "sample_index",
    "crop_index",
    "point_index",
    "global_index",
    "label",
    "base_pred",
    "final_pred",
    "base_prob_road",
    "base_prob_field",
    "final_prob_road",
    "final_prob_field",
    "candidate_mask",
    "candidate_run_id",
    "run_guard_score",
    "selected_by_guard",
    "fixed_field_as_road",
    "introduced_road_as_field",
    "error_type_base",
    "error_type_final",
    "lon",
    "lat",
] + AUX_FEATURE_NAMES

LONG_FIELDS = [
    "split",
    "stage",
    "trace_id",
    "error_type",
    "start_index",
    "end_index",
    "length",
]

TOP_LONG_FIELDS = [
    "trace_id",
    "start_index",
    "end_index",
    "length",
    "base_error_type",
    "candidate_run_id",
    "run_guard_score",
    "selected_by_guard",
    "fixed_points_in_segment",
    "remaining_error_points",
    "coverage_rate",
]


def write_csv(path, fieldnames, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def safe_div(num, den):
    return float(num) / float(den) if den else 0.0


def mean(values):
    values = [float(v) for v in values if v is not None and not math.isnan(float(v))]
    return float(np.mean(values)) if values else 0.0


def std(values):
    values = [float(v) for v in values if v is not None and not math.isnan(float(v))]
    return float(np.std(values, ddof=0)) if len(values) > 1 else 0.0


def q(values, quantile):
    values = [float(v) for v in values if v is not None and not math.isnan(float(v))]
    return float(np.quantile(values, quantile)) if values else 0.0


def min_value(values):
    values = [float(v) for v in values if v is not None and not math.isnan(float(v))]
    return float(np.min(values)) if values else 0.0


def max_value(values):
    values = [float(v) for v in values if v is not None and not math.isnan(float(v))]
    return float(np.max(values)) if values else 0.0


def error_type(label, pred):
    if int(label) == int(pred):
        return "correct"
    if int(label) == 0 and int(pred) == 1:
        return "road_as_field"
    return "field_as_road"


def class_metrics(labels, preds):
    labels = np.asarray(labels, dtype=np.int64)
    preds = np.asarray(preds, dtype=np.int64)
    if labels.size == 0:
        return dict(
            accuracy=0.0,
            macro_f1=0.0,
            road_f1=0.0,
            field_f1=0.0,
            pred_road_rate=0.0,
            pred_field_rate=0.0,
            road_as_field=0,
            field_as_road=0,
        )
    tp_road = int(((labels == 0) & (preds == 0)).sum())
    fp_road = int(((labels == 1) & (preds == 0)).sum())
    fn_road = int(((labels == 0) & (preds == 1)).sum())
    tp_field = int(((labels == 1) & (preds == 1)).sum())

    def f1(tp, fp, fn):
        precision = safe_div(tp, tp + fp)
        recall = safe_div(tp, tp + fn)
        return safe_div(2.0 * precision * recall, precision + recall)

    road_f1 = f1(tp_road, fp_road, fn_road)
    field_f1 = f1(tp_field, fn_road, fp_road)
    return dict(
        accuracy=float((labels == preds).mean()),
        macro_f1=(road_f1 + field_f1) / 2.0,
        road_f1=road_f1,
        field_f1=field_f1,
        pred_road_rate=float((preds == 0).mean()),
        pred_field_rate=float((preds == 1).mean()),
        road_as_field=fn_road,
        field_as_road=fp_road,
    )


def find_rc_checkpoint():
    candidates = [
        Path("runs/PT2G_MSC_RC_v1_finetune_40ep/best_model.pt"),
        Path("runs/PT2G_MSC_RC_v1_finetune_40ep/model_best.pth"),
        Path("weights/PT2G_MSC_RC_v1_finetune_40ep.pt"),
    ]
    for path in candidates:
        if path.exists():
            return path
    raise SystemExit("RC_CHECKPOINT_NOT_FOUND")


def build_frozen_model(device, checkpoint_path):
    model = build_model(COMMON_CONFIG, device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    for param in model.parameters():
        param.requires_grad = False
    model.eval()
    return model


def make_loaders():
    cache_dir = Path(COMMON_CONFIG["cache_dir"])
    datasets = {
        split: CachedGraphDataset(cache_dir / f"{split}.pt", mode=split, return_coordinates=True)
        for split in ("train", "valid", "test")
    }
    return {
        split: FieldRoadDataLoader(dataset, batch_size=1, shuffle=False, drop_last=True)
        for split, dataset in datasets.items()
    }


def make_graph_caches():
    graph_cache_path = COMMON_CONFIG["graph_cache_path"]
    return {
        split: load_graph_cache_split(graph_cache_path, split, required=(split != "test"))
        for split in ("train", "valid", "test")
    }


def coordinates_to_numpy(coordinates):
    if coordinates is None:
        return None
    if isinstance(coordinates, torch.Tensor):
        arr = coordinates.detach().cpu().numpy()
    else:
        arr = np.asarray(coordinates)
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 2 or arr.shape[1] < 2:
        return None
    return arr[:, :2].astype(np.float64, copy=False)


def candidate_mask_from_row(row):
    return (
        int(row["base_pred"]) == 0
        and (
            int(row["stationary_flag"]) == 1
            or float(row["local_step_mean_m"]) <= 0.8
            or float(row["local_density_1m"]) >= 4
            or float(row["stationary_run_length"]) >= 10
        )
    )


def infer_point_rows(model, loaders, graph_caches, device):
    rows = []
    aux_cache = {}
    trace_crop_counts = defaultdict(int)
    with torch.no_grad():
        for split in ("train", "valid", "test"):
            for sample_index, batch in enumerate(loaders[split]()):
                points, labels, adjs, trace_id, coordinates = unpack_batch(batch)
                points = points.clone().detach().to(torch.float32).squeeze(0).to(device)
                labels = labels.clone().detach().to(torch.int64).squeeze().to(device)
                aux_features = get_cached_aux_features(coordinates, points, device, aux_cache, split, trace_id)
                edge_index, edge_weight = merge_graph_cache_edges(
                    adjs,
                    graph_caches[split],
                    trace_id,
                    points,
                    device,
                    audit_path=None,
                    run_name=RUN_NAME,
                    split=split,
                    epoch=None,
                    batch_id=sample_index,
                )
                data = build_data(points, edge_index, labels, edge_weight=edge_weight, aux_features=aux_features)
                pred, pred_image, pred_graph, aux_dict = model(data, return_features=True)
                probs = torch.softmax(pred, dim=1)
                image_probs = torch.softmax(pred_image, dim=1)
                graph_probs = torch.softmax(pred_graph, dim=1)
                base_pred = torch.argmax(pred, dim=1)
                trace_id_norm = normalize_trace_id(trace_id)
                crop_index = trace_crop_counts[(split, trace_id_norm)]
                trace_crop_counts[(split, trace_id_norm)] += 1
                coords = coordinates_to_numpy(coordinates)
                labels_cpu = labels.detach().cpu().numpy().astype(int)
                pred_cpu = pred.detach().cpu().numpy()
                base_pred_cpu = base_pred.detach().cpu().numpy().astype(int)
                probs_cpu = probs.detach().cpu().numpy()
                image_cpu = image_probs.detach().cpu().numpy()
                graph_cpu = graph_probs.detach().cpu().numpy()
                aux_cpu = aux_features.detach().cpu().numpy()
                enhanced_cpu = aux_dict["enhanced_feature"].detach().cpu().numpy().astype(np.float32)
                for idx, label in enumerate(labels_cpu):
                    pred_label = int(base_pred_cpu[idx])
                    prob_road = float(probs_cpu[idx, 0])
                    prob_field = float(probs_cpu[idx, 1])
                    lon = float(coords[idx, 0]) if coords is not None and idx < len(coords) else ""
                    lat = float(coords[idx, 1]) if coords is not None and idx < len(coords) else ""
                    row = dict(
                        split=split,
                        trace_id=trace_id_norm,
                        sample_index=sample_index,
                        crop_index=crop_index,
                        point_index=idx,
                        global_index=crop_index * 1000 + idx,
                        candidate_run_id="",
                        label=int(label),
                        base_pred=pred_label,
                        base_road_logit=float(pred_cpu[idx, 0]),
                        base_field_logit=float(pred_cpu[idx, 1]),
                        base_prob_road=prob_road,
                        base_prob_field=prob_field,
                        margin=abs(prob_road - prob_field),
                        confidence=max(prob_road, prob_field),
                        error_type_base=error_type(int(label), pred_label),
                        lon=lon,
                        lat=lat,
                        image_prob_road=float(image_cpu[idx, 0]),
                        image_prob_field=float(image_cpu[idx, 1]),
                        graph_prob_road=float(graph_cpu[idx, 0]),
                        graph_prob_field=float(graph_cpu[idx, 1]),
                        branch_disagreement_road=abs(float(image_cpu[idx, 0]) - float(graph_cpu[idx, 0])),
                        branch_disagreement_field=abs(float(image_cpu[idx, 1]) - float(graph_cpu[idx, 1])),
                    )
                    for aux_idx, name in enumerate(AUX_FEATURE_NAMES):
                        row[name] = float(aux_cpu[idx, aux_idx])
                    row["candidate_mask"] = candidate_mask_from_row(row)
                    row["_enhanced_feature"] = enhanced_cpu[idx] if row["candidate_mask"] else None
                    rows.append(row)
            split_rows = sum(1 for row in rows if row["split"] == split)
            print(f"infer_split_done split={split} rows={split_rows}", flush=True)
    return rows


def build_candidate_runs(point_rows, min_run_len):
    grouped = defaultdict(list)
    for row in point_rows:
        grouped[(row["split"], row["trace_id"])].append(row)

    run_rows = []
    run_counter = 0
    for (_split, _trace_id), rows in grouped.items():
        rows.sort(key=lambda row: int(row["global_index"]))
        current = []
        prev = None
        for row in rows:
            idx = int(row["global_index"])
            contiguous = current and idx == prev + 1
            if row["candidate_mask"] and (not current or contiguous):
                current.append(row)
            else:
                if len(current) >= min_run_len:
                    run_rows.append(make_run_row(f"crun_{run_counter:08d}", current))
                    run_counter += 1
                current = [row] if row["candidate_mask"] else []
            prev = idx
        if len(current) >= min_run_len:
            run_rows.append(make_run_row(f"crun_{run_counter:08d}", current))
            run_counter += 1
    return run_rows


def run_type(field_ratio):
    if field_ratio >= 0.95:
        return "pure_false_road_run"
    if field_ratio >= 0.80:
        return "mostly_false_road_run"
    if field_ratio > 0.20:
        return "mixed_run"
    if field_ratio <= 0.05:
        return "pure_true_road_run"
    return "mostly_true_road_run"


def make_run_row(run_id, rows):
    for row in rows:
        row["candidate_run_id"] = run_id
    labels = [int(row["label"]) for row in rows]
    field_points = sum(1 for value in labels if value == 1)
    road_points = len(rows) - field_points
    field_ratio = safe_div(field_points, len(rows))
    if field_ratio >= CANDCAL_RUN_GUARD_DEFAULTS["positive_field_ratio"]:
        run_target = 1
        target_status = "positive_false_road_run"
    elif field_ratio <= CANDCAL_RUN_GUARD_DEFAULTS["negative_field_ratio"]:
        run_target = 0
        target_status = "negative_true_road_run"
    else:
        run_target = -1
        target_status = "ignored_mixed_run"

    enhanced_values = [row["_enhanced_feature"] for row in rows if row.get("_enhanced_feature") is not None]
    enhanced_mean = np.mean(np.stack(enhanced_values, axis=0), axis=0).astype(np.float32) if enhanced_values else np.zeros(216, dtype=np.float32)
    scalar = make_scalar_features(rows)
    feature_vector = np.asarray([float(scalar[name]) for name in SCALAR_FEATURE_NAMES] + enhanced_mean.tolist(), dtype=np.float32)
    return dict(
        candidate_run_id=run_id,
        split=rows[0]["split"],
        trace_id=rows[0]["trace_id"],
        start_index=int(rows[0]["global_index"]),
        end_index=int(rows[-1]["global_index"]),
        length=len(rows),
        point_indices=";".join(str(int(row["global_index"])) for row in rows),
        label_road_points=road_points,
        label_field_points=field_points,
        field_ratio=field_ratio,
        road_ratio=safe_div(road_points, len(rows)),
        run_target=run_target,
        target_status=target_status,
        run_type=run_type(field_ratio),
        run_guard_score="",
        selected_by_best_threshold=False,
        enhanced_feature_mean_norm=float(np.linalg.norm(enhanced_mean)),
        _feature_vector=feature_vector,
        **scalar,
    )


def make_scalar_features(rows):
    coords = [(row["lon"], row["lat"]) for row in rows if row["lon"] != "" and row["lat"] != ""]
    spatial = spatial_stats(coords)
    return dict(
        run_length=len(rows),
        trace_position_start=float(rows[0]["trace_position_ratio"]),
        trace_position_end=float(rows[-1]["trace_position_ratio"]),
        trace_position_mean=mean([row["trace_position_ratio"] for row in rows]),
        near_endpoint_rate=mean([row["near_endpoint_flag"] for row in rows]),
        prob_road_mean=mean([row["base_prob_road"] for row in rows]),
        prob_road_std=std([row["base_prob_road"] for row in rows]),
        prob_road_min=min_value([row["base_prob_road"] for row in rows]),
        prob_road_max=max_value([row["base_prob_road"] for row in rows]),
        prob_field_mean=mean([row["base_prob_field"] for row in rows]),
        margin_mean=mean([row["margin"] for row in rows]),
        margin_std=std([row["margin"] for row in rows]),
        confidence_mean=mean([row["confidence"] for row in rows]),
        stationary_rate=mean([row["stationary_flag"] for row in rows]),
        stationary_run_length_mean=mean([row["stationary_run_length"] for row in rows]),
        stationary_run_length_max=max_value([row["stationary_run_length"] for row in rows]),
        local_density_1m_mean=mean([row["local_density_1m"] for row in rows]),
        local_density_1m_max=max_value([row["local_density_1m"] for row in rows]),
        local_density_1m_q75=q([row["local_density_1m"] for row in rows], 0.75),
        local_density_2m_mean=mean([row["local_density_2m"] for row in rows]),
        local_step_mean_m_mean=mean([row["local_step_mean_m"] for row in rows]),
        local_step_mean_m_min=min_value([row["local_step_mean_m"] for row in rows]),
        local_step_mean_m_q25=q([row["local_step_mean_m"] for row in rows], 0.25),
        local_step_std_m_mean=mean([row["local_step_std_m"] for row in rows]),
        local_turn_angle_deg_mean=mean([row["local_turn_angle_deg"] for row in rows]),
        local_turn_angle_deg_max=max_value([row["local_turn_angle_deg"] for row in rows]),
        branch_disagreement_road_mean=mean([row["branch_disagreement_road"] for row in rows]),
        branch_disagreement_road_max=max_value([row["branch_disagreement_road"] for row in rows]),
        branch_disagreement_field_mean=mean([row["branch_disagreement_field"] for row in rows]),
        branch_disagreement_field_max=max_value([row["branch_disagreement_field"] for row in rows]),
        **spatial,
    )


def spatial_stats(coords):
    empty = dict(
        start_end_distance_m=0.0,
        bbox_width_m=0.0,
        bbox_height_m=0.0,
        bbox_diag_m=0.0,
        path_length_m=0.0,
        compactness=0.0,
    )
    if not coords:
        return empty
    lons = [float(item[0]) for item in coords]
    lats = [float(item[1]) for item in coords]
    start_lon, start_lat = lons[0], lats[0]
    end_lon, end_lat = lons[-1], lats[-1]
    ref_lat = mean(lats)
    width = haversine_m(min(lons), ref_lat, max(lons), ref_lat)
    height = haversine_m(mean(lons), min(lats), mean(lons), max(lats))
    diag = math.sqrt(width * width + height * height)
    path_length = 0.0
    for idx in range(1, len(coords)):
        path_length += haversine_m(coords[idx - 1][0], coords[idx - 1][1], coords[idx][0], coords[idx][1])
    return dict(
        start_end_distance_m=haversine_m(start_lon, start_lat, end_lon, end_lat),
        bbox_width_m=width,
        bbox_height_m=height,
        bbox_diag_m=diag,
        path_length_m=path_length,
        compactness=safe_div(path_length, max(diag, 1e-6)),
    )


def split_rows(rows, split):
    return [row for row in rows if row["split"] == split]


def split_runs(runs, split):
    return [run for run in runs if run["split"] == split]


def feature_matrix(runs, feature_mean=None, feature_std=None):
    if not runs:
        return np.zeros((0, len(FEATURE_NAMES)), dtype=np.float32)
    matrix = np.stack([run["_feature_vector"] for run in runs], axis=0).astype(np.float32)
    if feature_mean is not None and feature_std is not None:
        matrix = (matrix - feature_mean) / feature_std
    return matrix.astype(np.float32)


def fit_feature_stats(train_runs):
    matrix = feature_matrix(train_runs)
    feature_mean = matrix.mean(axis=0).astype(np.float32)
    feature_std = matrix.std(axis=0).astype(np.float32)
    feature_std[feature_std < 1e-6] = 1.0
    return feature_mean, feature_std


def score_runs(head, runs, feature_mean, feature_std, device):
    if not runs:
        return {}
    head.eval()
    scores = {}
    with torch.no_grad():
        matrix = feature_matrix(runs, feature_mean, feature_std)
        tensor = torch.from_numpy(matrix).to(device)
        output = head.score(tensor).detach().cpu().numpy()
    for run, score in zip(runs, output):
        scores[run["candidate_run_id"]] = float(score)
    return scores


def train_run_guard_head(runs, device):
    labeled_train = [run for run in split_runs(runs, "train") if int(run["run_target"]) in (0, 1)]
    if not labeled_train:
        raise SystemExit("RUN_GUARD_NO_TRAIN_TARGETS")
    targets = np.asarray([int(run["run_target"]) for run in labeled_train], dtype=np.float32)
    pos_weight, positives, negatives = compute_pos_weight(
        torch.from_numpy(targets),
        CANDCAL_RUN_GUARD_DEFAULTS["pos_weight_clip"],
    )
    if pos_weight is None:
        print(f"RUN_GUARD_TARGET_CLASS_WARNING positives={positives} negatives={negatives}", flush=True)
        pos_weight = torch.tensor(1.0, dtype=torch.float32)
    feature_mean, feature_std = fit_feature_stats(labeled_train)
    head = CandidateRunFieldGuardCalibration(input_dim=len(FEATURE_NAMES)).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
    x_train = torch.from_numpy(feature_matrix(labeled_train, feature_mean, feature_std)).to(device)
    y_train = torch.from_numpy(targets).to(device)

    threshold_rows = []
    training_rows = []
    best = None
    best_state = None
    for epoch in range(1, 21):
        head.train()
        perm = torch.randperm(x_train.shape[0], device=device)
        epoch_loss = 0.0
        batch_count = 0
        for start in range(0, x_train.shape[0], 128):
            idx = perm[start : start + 128]
            optimizer.zero_grad()
            logits = head(x_train[idx])
            loss = criterion(logits, y_train[idx])
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.detach().cpu().item())
            batch_count += 1

        valid_scores = score_runs(head, split_runs(runs, "valid"), feature_mean, feature_std, device)
        epoch_sweep = evaluate_threshold_sweep(
            "valid",
            POINT_ROWS_BY_SPLIT["valid"],
            split_runs(runs, "valid"),
            valid_scores,
            epoch,
        )
        threshold_rows.extend(epoch_sweep)
        epoch_best = choose_best_threshold_row(epoch_sweep)
        training_rows.append(
            dict(
                epoch=epoch,
                train_loss=safe_div(epoch_loss, batch_count),
                valid_best_threshold=epoch_best["threshold"],
                valid_best_macro_f1=epoch_best["cal_macro_f1"],
                valid_best_fixed=epoch_best["fixed_field_as_road"],
                valid_best_introduced=epoch_best["introduced_road_as_field"],
            )
        )
        if best is None or threshold_row_is_better(epoch_best, best):
            best = dict(epoch_best)
            best_state = {key: value.detach().cpu().clone() for key, value in head.state_dict().items()}
        print(
            f"epoch={epoch} train_loss={safe_div(epoch_loss, batch_count):.6f} "
            f"valid_macro={epoch_best['cal_macro_f1']:.6f} threshold={epoch_best['threshold']} "
            f"fixed={epoch_best['fixed_field_as_road']} introduced={epoch_best['introduced_road_as_field']}",
            flush=True,
        )
    if best_state is not None:
        head.load_state_dict(best_state)
    return head, feature_mean, feature_std, best, threshold_rows, training_rows, positives, negatives


def evaluate_threshold_sweep(split, rows, runs, score_map, epoch):
    return [
        evaluate_threshold(split, rows, runs, score_map, threshold, epoch)
        for threshold in CANDCAL_RUN_GUARD_DEFAULTS["threshold_candidates"]
    ]


def evaluate_threshold(split, rows, runs, score_map, threshold, epoch=None):
    selected_run_ids = {run["candidate_run_id"] for run in runs if score_map.get(run["candidate_run_id"], 0.0) >= threshold}
    labels, base_preds, final_preds, behavior = calibrated_predictions(rows, selected_run_ids, score_map)
    base = class_metrics(labels, base_preds)
    cal = class_metrics(labels, final_preds)
    candidate_points = sum(int(run["length"]) for run in runs)
    selected_runs = [run for run in runs if run["candidate_run_id"] in selected_run_ids]
    selected_points = sum(int(run["length"]) for run in selected_runs)
    selected_false = sum(1 for run in selected_runs if float(run["field_ratio"]) >= CANDCAL_RUN_GUARD_DEFAULTS["positive_field_ratio"])
    return dict(
        epoch=epoch if epoch is not None else "",
        threshold=float(threshold),
        base_accuracy=base["accuracy"],
        base_macro_f1=base["macro_f1"],
        base_road_f1=base["road_f1"],
        base_field_f1=base["field_f1"],
        base_pred_road_rate=base["pred_road_rate"],
        base_pred_field_rate=base["pred_field_rate"],
        base_road_as_field=base["road_as_field"],
        base_field_as_road=base["field_as_road"],
        cal_accuracy=cal["accuracy"],
        cal_macro_f1=cal["macro_f1"],
        cal_road_f1=cal["road_f1"],
        cal_field_f1=cal["field_f1"],
        cal_pred_road_rate=cal["pred_road_rate"],
        cal_pred_field_rate=cal["pred_field_rate"],
        cal_road_as_field=cal["road_as_field"],
        cal_field_as_road=cal["field_as_road"],
        delta_macro_f1=cal["macro_f1"] - base["macro_f1"],
        delta_road_f1=cal["road_f1"] - base["road_f1"],
        delta_field_f1=cal["field_f1"] - base["field_f1"],
        delta_road_as_field=cal["road_as_field"] - base["road_as_field"],
        delta_field_as_road=cal["field_as_road"] - base["field_as_road"],
        candidate_runs=len(runs),
        candidate_run_points=candidate_points,
        selected_runs=len(selected_runs),
        selected_run_points=selected_points,
        guard_applied_point_rate=safe_div(selected_points, len(rows)),
        fixed_field_as_road=behavior["fixed_field_as_road"],
        introduced_road_as_field=behavior["introduced_road_as_field"],
        fixed_minus_introduced=behavior["fixed_field_as_road"] - behavior["introduced_road_as_field"],
        selected_false_run_rate=safe_div(selected_false, len(selected_runs)),
    )


def calibrated_predictions(rows, selected_run_ids, score_map):
    labels = []
    base_preds = []
    final_preds = []
    fixed = 0
    introduced = 0
    max_delta = float(CANDCAL_RUN_GUARD_DEFAULTS["max_delta"])
    for row in rows:
        label = int(row["label"])
        base_pred = int(row["base_pred"])
        road_logit = float(row["base_road_logit"])
        field_logit = float(row["base_field_logit"])
        run_id = row.get("candidate_run_id", "")
        if run_id in selected_run_ids:
            delta = max_delta * float(score_map.get(run_id, 0.0))
            road_logit -= delta
            field_logit += delta
        final_pred = 0 if road_logit >= field_logit else 1
        labels.append(label)
        base_preds.append(base_pred)
        final_preds.append(final_pred)
        if base_pred == 0 and label == 1 and final_pred == 1:
            fixed += 1
        if base_pred == 0 and label == 0 and final_pred == 1:
            introduced += 1
    return (
        np.asarray(labels, dtype=np.int64),
        np.asarray(base_preds, dtype=np.int64),
        np.asarray(final_preds, dtype=np.int64),
        dict(fixed_field_as_road=fixed, introduced_road_as_field=introduced),
    )


def threshold_row_is_better(candidate, current):
    if current is None:
        return True
    macro_delta = float(candidate["cal_macro_f1"]) - float(current["cal_macro_f1"])
    if macro_delta > 0.001:
        return True
    if macro_delta < -0.001:
        return False
    candidate_net_positive = int(candidate["fixed_minus_introduced"]) > 0
    current_net_positive = int(current["fixed_minus_introduced"]) > 0
    if candidate_net_positive != current_net_positive:
        return candidate_net_positive
    if int(candidate["introduced_road_as_field"]) != int(current["introduced_road_as_field"]):
        return int(candidate["introduced_road_as_field"]) < int(current["introduced_road_as_field"])
    cand_road_drop = float(candidate["base_road_f1"]) - float(candidate["cal_road_f1"])
    curr_road_drop = float(current["base_road_f1"]) - float(current["cal_road_f1"])
    if abs(cand_road_drop - curr_road_drop) > 1e-12:
        return cand_road_drop < curr_road_drop
    cand_rate_dist = abs(float(candidate["cal_pred_road_rate"]) - float(candidate["base_pred_road_rate"]))
    curr_rate_dist = abs(float(current["cal_pred_road_rate"]) - float(current["base_pred_road_rate"]))
    if abs(cand_rate_dist - curr_rate_dist) > 1e-12:
        return cand_rate_dist < curr_rate_dist
    if int(candidate["fixed_minus_introduced"]) != int(current["fixed_minus_introduced"]):
        return int(candidate["fixed_minus_introduced"]) > int(current["fixed_minus_introduced"])
    return float(candidate["threshold"]) > float(current["threshold"])


def choose_best_threshold_row(rows):
    best = None
    for row in rows:
        if threshold_row_is_better(row, best):
            best = row
    return best


def long_error_segments(rows, pred_key, split, stage):
    grouped = defaultdict(list)
    for row in rows:
        pred = int(row[pred_key])
        etype = error_type(row["label"], pred)
        if etype == "correct":
            continue
        grouped[(row["trace_id"], etype)].append(row)
    segments = []
    for (trace_id, etype), items in grouped.items():
        items.sort(key=lambda row: int(row["global_index"]))
        current = []
        prev = None
        for row in items:
            idx = int(row["global_index"])
            if current and idx != prev + 1:
                add_long_segment(segments, split, stage, trace_id, etype, current)
                current = []
            current.append(row)
            prev = idx
        if current:
            add_long_segment(segments, split, stage, trace_id, etype, current)
    return segments


def add_long_segment(out, split, stage, trace_id, etype, rows):
    if len(rows) < 20:
        return
    out.append(
        dict(
            split=split,
            stage=stage,
            trace_id=trace_id,
            error_type=etype,
            start_index=int(rows[0]["global_index"]),
            end_index=int(rows[-1]["global_index"]),
            length=len(rows),
        )
    )


def summarize_long_segments(segments, error_type_name):
    selected = [row for row in segments if row["error_type"] == error_type_name]
    return dict(segments=len(selected), points=sum(int(row["length"]) for row in selected))


def annotate_test_rows(rows, runs, score_map, threshold):
    run_lookup = {run["candidate_run_id"]: run for run in runs}
    selected_ids = {run_id for run_id, score in score_map.items() if float(score) >= threshold}
    max_delta = float(CANDCAL_RUN_GUARD_DEFAULTS["max_delta"])
    out = []
    for row in rows:
        run_id = row.get("candidate_run_id", "")
        selected = run_id in selected_ids
        score = float(score_map.get(run_id, 0.0)) if run_id else 0.0
        road_logit = float(row["base_road_logit"])
        field_logit = float(row["base_field_logit"])
        if selected:
            delta = max_delta * score
            road_logit -= delta
            field_logit += delta
        logits = np.asarray([road_logit, field_logit], dtype=np.float64)
        logits -= logits.max()
        probs = np.exp(logits)
        probs /= probs.sum()
        final_pred = 0 if road_logit >= field_logit else 1
        row["_final_pred"] = final_pred
        row["_run_guard_score"] = score
        row["_selected_by_guard"] = selected
        row["_final_prob_road"] = float(probs[0])
        row["_final_prob_field"] = float(probs[1])
        out_row = dict(
            split=row["split"],
            trace_id=row["trace_id"],
            sample_index=row["sample_index"],
            crop_index=row["crop_index"],
            point_index=row["point_index"],
            global_index=row["global_index"],
            label=row["label"],
            base_pred=row["base_pred"],
            final_pred=final_pred,
            base_prob_road=row["base_prob_road"],
            base_prob_field=row["base_prob_field"],
            final_prob_road=float(probs[0]),
            final_prob_field=float(probs[1]),
            candidate_mask=row["candidate_mask"],
            candidate_run_id=run_id,
            run_guard_score=score if run_id in run_lookup else "",
            selected_by_guard=selected,
            fixed_field_as_road=int(row["base_pred"] == 0 and row["label"] == 1 and final_pred == 1),
            introduced_road_as_field=int(row["base_pred"] == 0 and row["label"] == 0 and final_pred == 1),
            error_type_base=row["error_type_base"],
            error_type_final=error_type(row["label"], final_pred),
            lon=row["lon"],
            lat=row["lat"],
        )
        for name in AUX_FEATURE_NAMES:
            out_row[name] = row[name]
        out.append(out_row)
    return out


def top_long_field_as_road_changes(test_rows, test_runs):
    run_lookup = {run["candidate_run_id"]: run for run in test_runs}
    rows_by_trace_index = {(row["trace_id"], int(row["global_index"])): row for row in test_rows}
    base_segments = long_error_segments(test_rows, "base_pred", "test", "base")
    base_field = [row for row in base_segments if row["error_type"] == "field_as_road"]
    base_field.sort(key=lambda row: int(row["length"]), reverse=True)
    out = []
    for segment in base_field[:20]:
        segment_rows = [
            rows_by_trace_index[(segment["trace_id"], idx)]
            for idx in range(int(segment["start_index"]), int(segment["end_index"]) + 1)
            if (segment["trace_id"], idx) in rows_by_trace_index
        ]
        overlap_counts = defaultdict(int)
        for row in segment_rows:
            run_id = row.get("candidate_run_id", "")
            if run_id:
                overlap_counts[run_id] += 1
        best_run_id = ""
        if overlap_counts:
            best_run_id = max(overlap_counts.items(), key=lambda item: item[1])[0]
        best_run = run_lookup.get(best_run_id, {})
        fixed = sum(1 for row in segment_rows if int(row["label"]) == 1 and int(row["base_pred"]) == 0 and int(row["_final_pred"]) == 1)
        remaining = sum(1 for row in segment_rows if int(row["label"]) == 1 and int(row["_final_pred"]) == 0)
        candidate_points = sum(1 for row in segment_rows if row.get("candidate_run_id", ""))
        out.append(
            dict(
                trace_id=segment["trace_id"],
                start_index=segment["start_index"],
                end_index=segment["end_index"],
                length=segment["length"],
                base_error_type="field_as_road",
                candidate_run_id=best_run_id,
                run_guard_score=best_run.get("run_guard_score", ""),
                selected_by_guard=best_run.get("selected_by_best_threshold", False),
                fixed_points_in_segment=fixed,
                remaining_error_points=remaining,
                coverage_rate=safe_div(candidate_points, segment["length"]),
            )
        )
    return out


def make_summary(best_epoch, best_threshold, test_eval, base_long, cal_long):
    base_field_long = summarize_long_segments(base_long, "field_as_road")
    base_road_long = summarize_long_segments(base_long, "road_as_field")
    cal_field_long = summarize_long_segments(cal_long, "field_as_road")
    cal_road_long = summarize_long_segments(cal_long, "road_as_field")
    return dict(
        group=RUN_NAME,
        best_epoch=best_epoch,
        best_threshold=best_threshold,
        base_test_accuracy=test_eval["base_accuracy"],
        base_test_macro_f1=test_eval["base_macro_f1"],
        base_test_road_f1=test_eval["base_road_f1"],
        base_test_field_f1=test_eval["base_field_f1"],
        base_pred_road_rate=test_eval["base_pred_road_rate"],
        base_road_as_field=test_eval["base_road_as_field"],
        base_field_as_road=test_eval["base_field_as_road"],
        base_long_road_as_field_segments=base_road_long["segments"],
        base_long_field_as_road_segments=base_field_long["segments"],
        base_long_field_as_road_points=base_field_long["points"],
        cal_test_accuracy=test_eval["cal_accuracy"],
        cal_test_macro_f1=test_eval["cal_macro_f1"],
        cal_test_road_f1=test_eval["cal_road_f1"],
        cal_test_field_f1=test_eval["cal_field_f1"],
        cal_pred_road_rate=test_eval["cal_pred_road_rate"],
        cal_road_as_field=test_eval["cal_road_as_field"],
        cal_field_as_road=test_eval["cal_field_as_road"],
        cal_long_road_as_field_segments=cal_road_long["segments"],
        cal_long_field_as_road_segments=cal_field_long["segments"],
        cal_long_field_as_road_points=cal_field_long["points"],
        delta_macro_f1=test_eval["delta_macro_f1"],
        delta_road_f1=test_eval["delta_road_f1"],
        delta_field_f1=test_eval["delta_field_f1"],
        delta_road_as_field=test_eval["delta_road_as_field"],
        delta_field_as_road=test_eval["delta_field_as_road"],
        delta_long_field_as_road_segments=cal_field_long["segments"] - base_field_long["segments"],
        delta_long_field_as_road_points=cal_field_long["points"] - base_field_long["points"],
        candidate_runs=test_eval["candidate_runs"],
        candidate_run_points=test_eval["candidate_run_points"],
        selected_runs=test_eval["selected_runs"],
        selected_run_points=test_eval["selected_run_points"],
        guard_applied_point_rate=test_eval["guard_applied_point_rate"],
        fixed_field_as_road=test_eval["fixed_field_as_road"],
        introduced_road_as_field=test_eval["introduced_road_as_field"],
        fixed_minus_introduced=test_eval["fixed_minus_introduced"],
    )


def update_run_scores(runs, score_map, threshold):
    for run in runs:
        score = float(score_map.get(run["candidate_run_id"], 0.0))
        run["run_guard_score"] = score
        run["selected_by_best_threshold"] = score >= threshold


def strip_internal_rows(rows):
    for row in rows:
        row.pop("_enhanced_feature", None)


def make_report(summary, valid_best, train_counts, top_changes, rc_checkpoint):
    exceeds_rc = float(summary["cal_test_macro_f1"]) > float(summary["base_test_macro_f1"])
    exceeds_delta = float(summary["cal_test_macro_f1"]) > POINT_GUARD_DELTA_V2["test_macro_f1"]
    field_down = int(summary["cal_field_as_road"]) < int(summary["base_field_as_road"])
    long_down = int(summary["cal_long_field_as_road_segments"]) < int(summary["base_long_field_as_road_segments"])
    max_fixed = bool(top_changes and int(top_changes[0]["fixed_points_in_segment"]) > 0)
    road_ok = float(summary["cal_test_road_f1"]) >= 0.815
    pred_road_ok = float(summary["cal_pred_road_rate"]) >= 0.20
    fixed_ok = int(summary["fixed_field_as_road"]) > int(summary["introduced_road_as_field"])
    selected_false_rate = float(valid_best.get("selected_false_run_rate", 0.0))
    mainline = exceeds_rc and field_down and road_ok and pred_road_ok and fixed_ok
    lines = [
        "# PT2G_MSC_RC_CANDCAL_RUN_GUARD_v1 Report",
        "",
        "## Setup",
        "",
        f"- RC checkpoint: {rc_checkpoint}.",
        "- Frozen base: PT2G_MSC_RC_v1.",
        "- Trainable part: CandidateRunFieldGuardCalibration only.",
        f"- Train targets: positives={train_counts['positives']}, negatives={train_counts['negatives']}.",
        f"- Best epoch: {summary['best_epoch']}; best threshold from valid: {summary['best_threshold']}.",
        "",
        "## Required Answers",
        "",
        f"1. Run Guard 是否超过 RC baseline：{exceeds_rc}，macro-F1 {summary['base_test_macro_f1']} -> {summary['cal_test_macro_f1']}，delta={summary['delta_macro_f1']}.",
        f"2. Run Guard 是否超过点级 Guard DELTA_v2：{exceeds_delta}，DELTA_v2 reference macro-F1={POINT_GUARD_DELTA_V2['test_macro_f1']}.",
        f"3. field_as_road 是否显著下降：{field_down}，{summary['base_field_as_road']} -> {summary['cal_field_as_road']}，delta={summary['delta_field_as_road']}.",
        f"4. long_field_as_road 是否显著下降：{long_down}，segments {summary['base_long_field_as_road_segments']} -> {summary['cal_long_field_as_road_segments']}，points {summary['base_long_field_as_road_points']} -> {summary['cal_long_field_as_road_points']}.",
        f"5. 最大 field_as_road 长段是否被修动：{max_fixed}，top fixed={top_changes[0]['fixed_points_in_segment'] if top_changes else 0}.",
        f"6. road-F1 是否被明显破坏：{not road_ok}，road-F1 {summary['base_test_road_f1']} -> {summary['cal_test_road_f1']}.",
        f"7. pred_road_rate 是否过低：{not pred_road_ok}，cal_pred_road_rate={summary['cal_pred_road_rate']}.",
        f"8. fixed_field_as_road 是否明显大于 introduced_road_as_field：{fixed_ok}，fixed={summary['fixed_field_as_road']}，introduced={summary['introduced_road_as_field']}.",
        f"9. selected runs 是否主要集中在 false-road run：valid selected false-run rate={selected_false_rate}.",
        f"10. 该模块是否适合作为下一阶段主线：{mainline}.",
        f"11. 是否还需要继续做 road_as_field repair：{int(summary['cal_road_as_field']) > int(summary['base_road_as_field']) or float(summary['cal_test_road_f1']) < float(summary['base_test_road_f1'])}.",
        f"12. 如果失败，失败原因：{failure_reason(summary, exceeds_delta, long_down, road_ok, fixed_ok)}.",
        f"13. 打包路径：{PACK_PATH}.",
        "",
        "## Top 20 Base Field-As-Road Long Segment Changes",
        "",
        "| trace_id | start | end | length | run | score | selected | fixed | remaining | coverage |",
        "|---|---:|---:|---:|---|---:|---|---:|---:|---:|",
    ]
    for row in top_changes:
        lines.append(
            f"| {row['trace_id']} | {row['start_index']} | {row['end_index']} | {row['length']} | "
            f"{row['candidate_run_id']} | {row['run_guard_score']} | {row['selected_by_guard']} | "
            f"{row['fixed_points_in_segment']} | {row['remaining_error_points']} | {row['coverage_rate']} |"
        )
    return "\n".join(lines) + "\n"


def failure_reason(summary, exceeds_delta, long_down, road_ok, fixed_ok):
    reasons = []
    if float(summary["cal_test_macro_f1"]) < float(summary["base_test_macro_f1"]):
        reasons.append("macro-F1 below RC baseline")
    if not exceeds_delta:
        reasons.append("macro-F1 does not exceed point-level DELTA_v2")
    if not long_down:
        reasons.append("long field_as_road did not decrease")
    if not road_ok:
        reasons.append("road-F1 below 0.815")
    if not fixed_ok:
        reasons.append("introduced road_as_field is not lower than fixed field_as_road")
    return "; ".join(reasons) if reasons else "none"


def make_pack():
    paths = [
        "models/CandidateRunFieldGuardCalibration.py",
        "experiments/train_candcal_run_guard.py",
        "scripts/run_candcal_run_guard.py",
        "models/Encoder.py",
        str(RESULT_PATH),
        str(REPORT_PATH),
        str(RUN_DIR / "config_resolved.json"),
        str(RUN_DIR / "best_run_guard_threshold.json"),
        str(RUN_DIR / "run_guard_head_best.pt"),
        str(LOG_PATH),
    ]
    paths.extend(str(path) for path in DIAG_DIR.glob("*.csv"))
    PACK_PATH.parent.mkdir(parents=True, exist_ok=True)
    if PACK_PATH.exists():
        PACK_PATH.unlink()
    with zipfile.ZipFile(PACK_PATH, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for item in paths:
            path = Path(item)
            if path.exists():
                zf.write(path, path.as_posix())


POINT_ROWS_BY_SPLIT = {}


def main():
    start = time.time()
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    for directory in (RUN_DIR, DIAG_DIR, RESULT_PATH.parent, REPORT_PATH.parent, PACK_PATH.parent, LOG_PATH.parent):
        directory.mkdir(parents=True, exist_ok=True)
    for path in [
        RUN_DIR / "run_guard_head_best.pt",
        RUN_DIR / "best_run_guard_threshold.json",
        RUN_DIR / "config_resolved.json",
        RESULT_PATH,
        REPORT_PATH,
        PACK_PATH,
    ]:
        if path.exists():
            path.unlink()

    device = get_default_device()
    rc_checkpoint = find_rc_checkpoint()
    print(f"device={device}", flush=True)
    print(f"rc_checkpoint={rc_checkpoint}", flush=True)
    model = build_frozen_model(device, rc_checkpoint)
    loaders = make_loaders()
    graph_caches = make_graph_caches()
    point_rows = infer_point_rows(model, loaders, graph_caches, device)
    for split in ("train", "valid", "test"):
        POINT_ROWS_BY_SPLIT[split] = split_rows(point_rows, split)

    run_rows = build_candidate_runs(point_rows, int(CANDCAL_RUN_GUARD_DEFAULTS["min_run_len"]))
    print(f"candidate_runs_built total={len(run_rows)} train={len(split_runs(run_rows, 'train'))} valid={len(split_runs(run_rows, 'valid'))} test={len(split_runs(run_rows, 'test'))}", flush=True)

    head, feature_mean, feature_std, best_valid, threshold_rows, training_rows, positives, negatives = train_run_guard_head(run_rows, device)
    best_epoch = int(best_valid["epoch"])
    best_threshold = float(best_valid["threshold"])
    all_scores = {}
    for split in ("train", "valid", "test"):
        all_scores.update(score_runs(head, split_runs(run_rows, split), feature_mean, feature_std, device))
    update_run_scores(run_rows, all_scores, best_threshold)

    test_rows = POINT_ROWS_BY_SPLIT["test"]
    test_runs = split_runs(run_rows, "test")
    test_eval = evaluate_threshold("test", test_rows, test_runs, all_scores, best_threshold, best_epoch)
    point_test_rows = annotate_test_rows(test_rows, test_runs, all_scores, best_threshold)
    for row, point_row in zip(test_rows, point_test_rows):
        row["final_pred"] = point_row["final_pred"]

    base_long = long_error_segments(test_rows, "base_pred", "test", "base")
    cal_long = long_error_segments(test_rows, "final_pred", "test", "calibrated")
    top_changes = top_long_field_as_road_changes(test_rows, test_runs)
    summary = make_summary(best_epoch, best_threshold, test_eval, base_long, cal_long)

    config = dict(
        run_name=RUN_NAME,
        common_config=COMMON_CONFIG,
        run_guard_defaults=CANDCAL_RUN_GUARD_DEFAULTS,
        rc_checkpoint=str(rc_checkpoint),
        frozen_rc_mainline=True,
        feature_names=FEATURE_NAMES,
        feature_mean=feature_mean.tolist(),
        feature_std=feature_std.tolist(),
        train_positive_runs=positives,
        train_negative_runs=negatives,
    )
    write_json(RUN_DIR / "config_resolved.json", config)
    write_json(
        RUN_DIR / "best_run_guard_threshold.json",
        dict(
            best_epoch=best_epoch,
            best_threshold=best_threshold,
            best_valid_metrics=best_valid,
            valid_threshold_selection="valid_macro_f1_first_then_safety_tie_break",
        ),
    )
    torch.save(
        dict(
            model_state_dict=head.state_dict(),
            input_dim=len(FEATURE_NAMES),
            feature_names=FEATURE_NAMES,
            feature_mean=feature_mean,
            feature_std=feature_std,
            config=CANDCAL_RUN_GUARD_DEFAULTS,
            best_epoch=best_epoch,
            best_threshold=best_threshold,
            rc_checkpoint=str(rc_checkpoint),
        ),
        RUN_DIR / "run_guard_head_best.pt",
    )

    strip_internal_rows(point_rows)
    for split in ("train", "valid", "test"):
        write_csv(DIAG_DIR / f"{split}_candidate_runs.csv", RUN_CSV_FIELDS, split_runs(run_rows, split))
    write_csv(DIAG_DIR / "threshold_sweep_valid.csv", THRESHOLD_FIELDS, threshold_rows)
    write_csv(DIAG_DIR / "training_metrics.csv", ["epoch", "train_loss", "valid_best_threshold", "valid_best_macro_f1", "valid_best_fixed", "valid_best_introduced"], training_rows)
    write_csv(DIAG_DIR / "point_level_calibration_test.csv", POINT_TEST_FIELDS, point_test_rows)
    write_csv(DIAG_DIR / "long_error_before_after.csv", LONG_FIELDS, base_long + cal_long)
    write_csv(DIAG_DIR / "top_long_field_as_road_changes.csv", TOP_LONG_FIELDS, top_changes)
    write_csv(RESULT_PATH, SUMMARY_FIELDS, [summary])
    REPORT_PATH.write_text(make_report(summary, best_valid, dict(positives=positives, negatives=negatives), top_changes, rc_checkpoint), encoding="utf-8")
    make_pack()

    print(f"modified_files=models/CandidateRunFieldGuardCalibration.py,experiments/train_candcal_run_guard.py,scripts/run_candcal_run_guard.py,models/Encoder.py", flush=True)
    print(f"frozen_rc_mainline=True", flush=True)
    print(f"candidate_runs={len(run_rows)}", flush=True)
    print(f"best_epoch={best_epoch}", flush=True)
    print(f"best_threshold={best_threshold}", flush=True)
    print(f"base_macro_f1={summary['base_test_macro_f1']} cal_macro_f1={summary['cal_test_macro_f1']} delta={summary['delta_macro_f1']}", flush=True)
    print(f"base_field_as_road={summary['base_field_as_road']} cal_field_as_road={summary['cal_field_as_road']} delta={summary['delta_field_as_road']}", flush=True)
    print(f"base_long_field_as_road_segments={summary['base_long_field_as_road_segments']} cal_long_field_as_road_segments={summary['cal_long_field_as_road_segments']} delta={summary['delta_long_field_as_road_segments']}", flush=True)
    print(f"fixed_field_as_road={summary['fixed_field_as_road']} introduced_road_as_field={summary['introduced_road_as_field']}", flush=True)
    print(f"pack_path={PACK_PATH}", flush=True)
    print(f"elapsed_sec={time.time() - start:.2f}", flush=True)


if __name__ == "__main__":
    main()
