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
    tensor_content_hash,
    unpack_batch,
)
from utils.motion_state_features import AUX_FEATURE_NAMES, haversine_m
from utils.threading_config import apply_torch_thread_config, configure_default_threads
from utils.utils import get_default_device


configure_default_threads()
apply_torch_thread_config(torch)


RUN_NAME = "PT2G_MSC_RC_CANDCAL_RUN_GUARD_ORACLE_AUDIT"
DIAG_DIR = Path("diagnostics") / RUN_NAME
RESULT_PATH = Path("results") / f"{RUN_NAME}_summary.csv"
REPORT_PATH = Path("analysis") / f"{RUN_NAME}_report.md"
PACK_PATH = Path("analysis_packs") / f"{RUN_NAME}_for_chatgpt.zip"
LOG_PATH = Path("logs/candcal_run_guard_audit/CANDCAL_RUN_GUARD_ORACLE_AUDIT.log")

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

POINT_FIELDS = [
    "split",
    "trace_id",
    "sample_index",
    "crop_index",
    "point_index",
    "global_index",
    "candidate_run_id",
    "candidate_run_eligible",
    "label",
    "base_pred",
    "prob_road",
    "prob_field",
    "margin",
    "confidence",
    "is_correct",
    "error_type",
    "candidate_mask",
    "lon",
    "lat",
    "image_prob_road",
    "image_prob_field",
    "graph_prob_road",
    "graph_prob_field",
    "branch_disagreement_road",
    "branch_disagreement_field",
] + AUX_FEATURE_NAMES

RUN_FIELDS = [
    "candidate_run_id",
    "split",
    "trace_id",
    "start_index",
    "end_index",
    "length",
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
    "start_lon",
    "start_lat",
    "end_lon",
    "end_lat",
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
    "label_road_points",
    "label_field_points",
    "field_ratio",
    "road_ratio",
    "field_as_road_points",
    "correct_road_points",
    "other_error_points",
    "base_correct_points",
    "base_error_points",
    "run_type",
]

ORACLE_FIELDS = [
    "split",
    "oracle_name",
    "baseline_accuracy",
    "baseline_macro_f1",
    "baseline_road_f1",
    "baseline_field_f1",
    "baseline_pred_road_rate",
    "baseline_road_as_field",
    "baseline_field_as_road",
    "baseline_long_road_as_field_segments",
    "baseline_long_field_as_road_segments",
    "oracle_accuracy",
    "oracle_macro_f1",
    "oracle_road_f1",
    "oracle_field_f1",
    "oracle_pred_road_rate",
    "oracle_road_as_field",
    "oracle_field_as_road",
    "oracle_long_road_as_field_segments",
    "oracle_long_field_as_road_segments",
    "delta_macro_f1",
    "delta_road_f1",
    "delta_field_f1",
    "delta_road_as_field",
    "delta_field_as_road",
    "delta_long_field_as_road_segments",
    "introduced_road_as_field",
    "fixed_field_as_road",
    "fixed_minus_introduced",
]

HEURISTIC_FIELDS = [
    "split",
    "rule_name",
    "selected_runs",
    "selected_points",
    "selected_point_rate",
    "selected_field_ratio_mean",
    "selected_field_as_road_points",
    "selected_correct_road_points",
    "precision_for_false_road_points",
    "recall_for_field_as_road_points",
    "introduced_risk_proxy",
    "oracle_after_rule_accuracy",
    "oracle_after_rule_macro_f1",
    "oracle_after_rule_road_f1",
    "oracle_after_rule_field_f1",
    "oracle_after_rule_road_as_field",
    "oracle_after_rule_field_as_road",
    "oracle_delta_macro_f1",
    "oracle_delta_field_as_road",
    "oracle_delta_road_as_field",
]

PROBE_FIELDS = [
    "model_name",
    "split",
    "run_auc",
    "run_ap",
    "best_valid_threshold",
    "selected_runs",
    "selected_points",
    "precision_for_false_road_points",
    "recall_for_field_as_road_points",
    "oracle_delta_macro_f1_if_selected_runs_flipped",
    "introduced_risk_proxy",
]

LONG_FIELDS = [
    "split",
    "trace_id",
    "error_type",
    "start_index",
    "end_index",
    "length",
    "candidate_points_in_segment",
    "candidate_coverage_rate",
    "covered_by_candidate",
    "overlapped_candidate_run_ids",
    "best_overlap_run_id",
    "best_overlap_run_coverage",
    "run_field_ratio",
    "run_length",
]

SUMMARY_FIELDS = [
    "split",
    "baseline_macro_f1",
    "baseline_road_f1",
    "baseline_field_f1",
    "baseline_road_as_field",
    "baseline_field_as_road",
    "baseline_pred_road_rate",
    "candidate_runs",
    "candidate_run_points",
    "candidate_run_point_rate",
    "pure_false_road_runs",
    "mostly_false_road_runs",
    "mixed_runs",
    "mostly_true_road_runs",
    "pure_true_road_runs",
    "point_oracle_macro_f1",
    "pure_run_oracle_macro_f1",
    "mostly_run_oracle_macro_f1",
    "point_oracle_delta_macro_f1",
    "pure_run_oracle_delta_macro_f1",
    "mostly_run_oracle_delta_macro_f1",
    "point_oracle_delta_field_as_road",
    "pure_run_oracle_delta_field_as_road",
    "mostly_run_oracle_delta_field_as_road",
    "pure_run_introduced_road_as_field",
    "mostly_run_introduced_road_as_field",
    "long_field_as_road_segments_total",
    "long_field_as_road_segments_covered_by_candidate_run",
    "long_field_as_road_coverage_rate",
    "recommended_rule_name",
    "recommended_rule_valid_delta_macro_f1",
    "recommended_rule_test_delta_macro_f1",
    "recommend_train_run_guard",
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


def q(values, quantile):
    values = [float(v) for v in values if v is not None and not math.isnan(float(v))]
    return float(np.quantile(values, quantile)) if values else 0.0


def mean(values):
    values = [float(v) for v in values if v is not None and not math.isnan(float(v))]
    return float(np.mean(values)) if values else 0.0


def std(values):
    values = [float(v) for v in values if v is not None and not math.isnan(float(v))]
    return float(np.std(values, ddof=0)) if len(values) > 1 else 0.0


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
        return dict(accuracy=0.0, macro_f1=0.0, road_f1=0.0, field_f1=0.0, pred_road_rate=0.0, road_as_field=0, field_as_road=0)
    tp_road = int(((labels == 0) & (preds == 0)).sum())
    fp_road = int(((labels == 1) & (preds == 0)).sum())
    fn_road = int(((labels == 0) & (preds == 1)).sum())
    tp_field = int(((labels == 1) & (preds == 1)).sum())
    fp_field = fn_road
    fn_field = fp_road

    def f1(tp, fp, fn):
        precision = safe_div(tp, tp + fp)
        recall = safe_div(tp, tp + fn)
        return safe_div(2.0 * precision * recall, precision + recall)

    road_f1 = f1(tp_road, fp_road, fn_road)
    field_f1 = f1(tp_field, fp_field, fn_field)
    return dict(
        accuracy=float((labels == preds).mean()),
        macro_f1=(road_f1 + field_f1) / 2.0,
        road_f1=road_f1,
        field_f1=field_f1,
        pred_road_rate=float((preds == 0).mean()),
        road_as_field=fn_road,
        field_as_road=fp_road,
    )


def confusion_counts(labels, preds):
    labels = np.asarray(labels, dtype=np.int64)
    preds = np.asarray(preds, dtype=np.int64)
    return dict(
        total=int(labels.size),
        tp_road=int(((labels == 0) & (preds == 0)).sum()),
        fp_road=int(((labels == 1) & (preds == 0)).sum()),
        fn_road=int(((labels == 0) & (preds == 1)).sum()),
        tp_field=int(((labels == 1) & (preds == 1)).sum()),
    )


def counts_for_rows(rows):
    return confusion_counts([row["label"] for row in rows], [row["base_pred"] for row in rows])


def metrics_from_counts(counts):
    total = int(counts.get("total", 0))
    tp_road = int(counts.get("tp_road", 0))
    fp_road = int(counts.get("fp_road", 0))
    fn_road = int(counts.get("fn_road", 0))
    tp_field = int(counts.get("tp_field", 0))

    def f1(tp, fp, fn):
        precision = safe_div(tp, tp + fp)
        recall = safe_div(tp, tp + fn)
        return safe_div(2.0 * precision * recall, precision + recall)

    road_f1 = f1(tp_road, fp_road, fn_road)
    field_f1 = f1(tp_field, fn_road, fp_road)
    pred_road = tp_road + fp_road
    return dict(
        accuracy=safe_div(tp_road + tp_field, total),
        macro_f1=(road_f1 + field_f1) / 2.0,
        road_f1=road_f1,
        field_f1=field_f1,
        pred_road_rate=safe_div(pred_road, total),
        road_as_field=fn_road,
        field_as_road=fp_road,
    )


def metrics_after_flipping_runs(base_counts, selected_runs):
    selected_field = sum(int(run["field_as_road_points"]) for run in selected_runs)
    selected_road = sum(int(run["correct_road_points"]) for run in selected_runs)
    counts = dict(base_counts)
    counts["tp_road"] = max(0, int(counts["tp_road"]) - selected_road)
    counts["fp_road"] = max(0, int(counts["fp_road"]) - selected_field)
    counts["fn_road"] = int(counts["fn_road"]) + selected_road
    counts["tp_field"] = int(counts["tp_field"]) + selected_field
    return metrics_from_counts(counts)


def long_error_segments(rows, pred_key):
    grouped = defaultdict(list)
    for row in rows:
        etype = error_type(row["label"], row[pred_key])
        if etype == "correct":
            continue
        grouped[(row["split"], row["trace_id"], etype)].append(row)

    segments = []
    for (split, trace_id, etype), items in grouped.items():
        items.sort(key=lambda row: int(row["global_index"]))
        current = []
        prev = None
        for row in items:
            idx = int(row["global_index"])
            if current and idx != prev + 1:
                add_long_segment(segments, split, trace_id, etype, current)
                current = []
            current.append(row)
            prev = idx
        if current:
            add_long_segment(segments, split, trace_id, etype, current)
    return segments


def add_long_segment(out, split, trace_id, etype, rows):
    if len(rows) < 20:
        return
    out.append(
        dict(
            split=split,
            trace_id=trace_id,
            error_type=etype,
            start_index=int(rows[0]["global_index"]),
            end_index=int(rows[-1]["global_index"]),
            length=len(rows),
        )
    )


def long_counts(rows, pred_key):
    segments = long_error_segments(rows, pred_key)
    return dict(
        long_road_as_field_segments=sum(1 for row in segments if row["error_type"] == "road_as_field"),
        long_field_as_road_segments=sum(1 for row in segments if row["error_type"] == "field_as_road"),
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


def infer_points(model, loaders, graph_caches, device):
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
                pred, pred_image, pred_graph, _aux_dict = model(data, return_features=True)
                probs = torch.softmax(pred, dim=1)
                image_probs = torch.softmax(pred_image, dim=1)
                graph_probs = torch.softmax(pred_graph, dim=1)
                base_pred = torch.argmax(pred, dim=1)
                trace_id_norm = normalize_trace_id(trace_id)
                crop_index = trace_crop_counts[(split, trace_id_norm)]
                trace_crop_counts[(split, trace_id_norm)] += 1
                coords = coordinates_to_numpy(coordinates)
                labels_cpu = labels.detach().cpu().numpy().astype(int)
                base_pred_cpu = base_pred.detach().cpu().numpy().astype(int)
                probs_cpu = probs.detach().cpu().numpy()
                image_cpu = image_probs.detach().cpu().numpy()
                graph_cpu = graph_probs.detach().cpu().numpy()
                aux_cpu = aux_features.detach().cpu().numpy()
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
                        candidate_run_eligible=False,
                        label=int(label),
                        base_pred=pred_label,
                        prob_road=prob_road,
                        prob_field=prob_field,
                        margin=abs(prob_road - prob_field),
                        confidence=max(prob_road, prob_field),
                        is_correct=int(label) == pred_label,
                        error_type=error_type(int(label), pred_label),
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
                    rows.append(row)
            print(f"infer_split_done split={split} rows={sum(1 for row in rows if row['split'] == split)}", flush=True)
    return rows


def build_candidate_runs(point_rows):
    grouped = defaultdict(list)
    for idx, row in enumerate(point_rows):
        row["_row_index"] = idx
        grouped[(row["split"], row["trace_id"])].append(row)

    all_runs = []
    eligible_runs = []
    run_counter = 0
    for (split, trace_id), rows in grouped.items():
        rows.sort(key=lambda row: int(row["global_index"]))
        current = []
        prev = None
        for row in rows:
            idx = int(row["global_index"])
            contiguous = current and idx == prev + 1
            if row["candidate_mask"] and (not current or contiguous):
                current.append(row)
            else:
                if current:
                    run_counter = finalize_run(run_counter, current, all_runs, eligible_runs)
                    current = []
                if row["candidate_mask"]:
                    current = [row]
            prev = idx
        if current:
            run_counter = finalize_run(run_counter, current, all_runs, eligible_runs)

    for row in point_rows:
        row.pop("_row_index", None)
    return eligible_runs, all_runs


def finalize_run(run_counter, rows, all_runs, eligible_runs):
    run_id = f"crun_{run_counter:08d}"
    for row in rows:
        row["candidate_run_id"] = run_id
        row["candidate_run_eligible"] = len(rows) >= 5
    all_runs.append((run_id, rows))
    if len(rows) >= 5:
        eligible_runs.append(make_run_row(run_id, rows))
    return run_counter + 1


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
    labels = [int(row["label"]) for row in rows]
    field_points = sum(1 for value in labels if value == 1)
    road_points = len(labels) - field_points
    field_ratio = safe_div(field_points, len(rows))
    coords = [(row["lon"], row["lat"]) for row in rows if row["lon"] != "" and row["lat"] != ""]
    spatial = spatial_stats(coords)
    return dict(
        candidate_run_id=run_id,
        split=rows[0]["split"],
        trace_id=rows[0]["trace_id"],
        start_index=int(rows[0]["global_index"]),
        end_index=int(rows[-1]["global_index"]),
        length=len(rows),
        run_length=len(rows),
        trace_position_start=float(rows[0]["trace_position_ratio"]),
        trace_position_end=float(rows[-1]["trace_position_ratio"]),
        trace_position_mean=mean([row["trace_position_ratio"] for row in rows]),
        near_endpoint_rate=mean([row["near_endpoint_flag"] for row in rows]),
        prob_road_mean=mean([row["prob_road"] for row in rows]),
        prob_road_std=std([row["prob_road"] for row in rows]),
        prob_road_min=min_value([row["prob_road"] for row in rows]),
        prob_road_max=max_value([row["prob_road"] for row in rows]),
        prob_field_mean=mean([row["prob_field"] for row in rows]),
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
        **spatial,
        branch_disagreement_road_mean=mean([row["branch_disagreement_road"] for row in rows]),
        branch_disagreement_road_max=max_value([row["branch_disagreement_road"] for row in rows]),
        branch_disagreement_field_mean=mean([row["branch_disagreement_field"] for row in rows]),
        branch_disagreement_field_max=max_value([row["branch_disagreement_field"] for row in rows]),
        label_road_points=road_points,
        label_field_points=field_points,
        field_ratio=field_ratio,
        road_ratio=safe_div(road_points, len(rows)),
        field_as_road_points=field_points,
        correct_road_points=road_points,
        other_error_points=0,
        base_correct_points=road_points,
        base_error_points=field_points,
        run_type=run_type(field_ratio),
    )


def spatial_stats(coords):
    empty = dict(
        start_lon="",
        start_lat="",
        end_lon="",
        end_lat="",
        start_end_distance_m="",
        bbox_width_m="",
        bbox_height_m="",
        bbox_diag_m="",
        path_length_m="",
        compactness="",
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
        start_lon=start_lon,
        start_lat=start_lat,
        end_lon=end_lon,
        end_lat=end_lat,
        start_end_distance_m=haversine_m(start_lon, start_lat, end_lon, end_lat),
        bbox_width_m=width,
        bbox_height_m=height,
        bbox_diag_m=diag,
        path_length_m=path_length,
        compactness=safe_div(path_length, max(diag, 1e-6)),
    )


def split_points(point_rows, split):
    return [row for row in point_rows if row["split"] == split]


def split_runs(candidate_runs, split):
    return [row for row in candidate_runs if row["split"] == split]


def oracle_predictions(rows, runs, mode):
    labels = np.array([int(row["label"]) for row in rows], dtype=np.int64)
    base_preds = np.array([int(row["base_pred"]) for row in rows], dtype=np.int64)
    final_preds = base_preds.copy()
    run_lookup = {row["candidate_run_id"]: row for row in runs}
    eligible_ids = set(run_lookup)
    for idx, row in enumerate(rows):
        run_id = row.get("candidate_run_id", "")
        if run_id not in eligible_ids:
            continue
        run = run_lookup[run_id]
        if mode == "point_oracle":
            if labels[idx] == 1 and base_preds[idx] == 0:
                final_preds[idx] = 1
        elif mode == "pure_run_oracle":
            if float(run["field_ratio"]) >= 0.95:
                final_preds[idx] = 1
        elif mode == "mostly_run_oracle":
            if float(run["field_ratio"]) >= 0.80:
                final_preds[idx] = 1
    return labels, base_preds, final_preds


def oracle_metric_row(split, rows, runs, mode):
    labels, base_preds, final_preds = oracle_predictions(rows, runs, mode)
    baseline = class_metrics(labels, base_preds)
    oracle = class_metrics(labels, final_preds)
    base_long = long_counts(rows, "base_pred")
    oracle_rows = [dict(row, oracle_pred=int(final_preds[idx])) for idx, row in enumerate(rows)]
    oracle_long = long_counts(oracle_rows, "oracle_pred")
    fixed = int(((labels == 1) & (base_preds == 0) & (final_preds == 1)).sum())
    introduced = int(((labels == 0) & (base_preds == 0) & (final_preds == 1)).sum())
    return dict(
        split=split,
        oracle_name=mode,
        baseline_accuracy=baseline["accuracy"],
        baseline_macro_f1=baseline["macro_f1"],
        baseline_road_f1=baseline["road_f1"],
        baseline_field_f1=baseline["field_f1"],
        baseline_pred_road_rate=baseline["pred_road_rate"],
        baseline_road_as_field=baseline["road_as_field"],
        baseline_field_as_road=baseline["field_as_road"],
        baseline_long_road_as_field_segments=base_long["long_road_as_field_segments"],
        baseline_long_field_as_road_segments=base_long["long_field_as_road_segments"],
        oracle_accuracy=oracle["accuracy"],
        oracle_macro_f1=oracle["macro_f1"],
        oracle_road_f1=oracle["road_f1"],
        oracle_field_f1=oracle["field_f1"],
        oracle_pred_road_rate=oracle["pred_road_rate"],
        oracle_road_as_field=oracle["road_as_field"],
        oracle_field_as_road=oracle["field_as_road"],
        oracle_long_road_as_field_segments=oracle_long["long_road_as_field_segments"],
        oracle_long_field_as_road_segments=oracle_long["long_field_as_road_segments"],
        delta_macro_f1=oracle["macro_f1"] - baseline["macro_f1"],
        delta_road_f1=oracle["road_f1"] - baseline["road_f1"],
        delta_field_f1=oracle["field_f1"] - baseline["field_f1"],
        delta_road_as_field=oracle["road_as_field"] - baseline["road_as_field"],
        delta_field_as_road=oracle["field_as_road"] - baseline["field_as_road"],
        delta_long_field_as_road_segments=oracle_long["long_field_as_road_segments"] - base_long["long_field_as_road_segments"],
        introduced_road_as_field=introduced,
        fixed_field_as_road=fixed,
        fixed_minus_introduced=fixed - introduced,
    )


def selected_rule_predictions(rows, selected_run_ids):
    labels = np.array([int(row["label"]) for row in rows], dtype=np.int64)
    base_preds = np.array([int(row["base_pred"]) for row in rows], dtype=np.int64)
    final_preds = base_preds.copy()
    for idx, row in enumerate(rows):
        if row.get("candidate_run_id", "") in selected_run_ids:
            final_preds[idx] = 1
    return labels, base_preds, final_preds


def heuristic_conditions(run):
    length_thresholds = [20, 50, 100, 200]
    stationary_thresholds = [0.5, 0.7, 0.85]
    density_thresholds = [4, 6, 8]
    step_thresholds = [0.3, 0.5, 0.8]
    prob_thresholds = [0.70, 0.80, 0.85]
    margin_thresholds = [0.40, 0.60, 0.70]
    compactness_thresholds = [2, 5, 10]
    rules = []
    for length in length_thresholds:
        for stationary in stationary_thresholds:
            for density in density_thresholds:
                for step in step_thresholds:
                    rules.append(
                        (
                            f"motion_L{length}_S{stationary}_D{density}_STEP{step}",
                            int(run["run_length"]) >= length
                            and float(run["stationary_rate"]) >= stationary
                            and float(run["local_density_1m_mean"]) >= density
                            and float(run["local_step_mean_m_mean"]) <= step,
                        )
                    )
                    for prob in prob_thresholds:
                        rules.append(
                            (
                                f"motion_prob_L{length}_S{stationary}_D{density}_STEP{step}_P{prob}",
                                int(run["run_length"]) >= length
                                and float(run["stationary_rate"]) >= stationary
                                and float(run["local_density_1m_mean"]) >= density
                                and float(run["local_step_mean_m_mean"]) <= step
                                and float(run["prob_road_mean"]) >= prob,
                            )
                        )
                    for margin in margin_thresholds:
                        rules.append(
                            (
                                f"motion_margin_L{length}_S{stationary}_D{density}_STEP{step}_M{margin}",
                                int(run["run_length"]) >= length
                                and float(run["stationary_rate"]) >= stationary
                                and float(run["local_density_1m_mean"]) >= density
                                and float(run["local_step_mean_m_mean"]) <= step
                                and float(run["margin_mean"]) >= margin,
                            )
                        )
                    for compactness in compactness_thresholds:
                        compact_value = run.get("compactness", "")
                        compact_ok = compact_value != "" and float(compact_value) >= compactness
                        rules.append(
                            (
                                f"motion_compact_L{length}_S{stationary}_D{density}_STEP{step}_C{compactness}",
                                int(run["run_length"]) >= length
                                and float(run["stationary_rate"]) >= stationary
                                and float(run["local_density_1m_mean"]) >= density
                                and float(run["local_step_mean_m_mean"]) <= step
                                and compact_ok,
                            )
                        )
    return rules


def build_heuristic_rows(point_rows, candidate_runs):
    all_rows = []
    for split in ("train", "valid", "test"):
        rows = split_points(point_rows, split)
        runs = split_runs(candidate_runs, split)
        total_points = len(rows)
        total_field_as_road = sum(1 for row in rows if row["error_type"] == "field_as_road")
        base_counts = counts_for_rows(rows)
        baseline = metrics_from_counts(base_counts)
        rule_to_runs = defaultdict(list)
        for run in runs:
            for rule_name, selected in heuristic_conditions(run):
                if selected:
                    rule_to_runs[rule_name].append(run)
        for rule_name, selected_runs in rule_to_runs.items():
            selected_ids = {row["candidate_run_id"] for row in selected_runs}
            selected_points = sum(int(row["run_length"]) for row in selected_runs)
            selected_field = sum(int(row["field_as_road_points"]) for row in selected_runs)
            selected_road = sum(int(row["correct_road_points"]) for row in selected_runs)
            metrics = metrics_after_flipping_runs(base_counts, selected_runs)
            all_rows.append(
                dict(
                    split=split,
                    rule_name=rule_name,
                    selected_runs=len(selected_runs),
                    selected_points=selected_points,
                    selected_point_rate=safe_div(selected_points, total_points),
                    selected_field_ratio_mean=mean([run["field_ratio"] for run in selected_runs]),
                    selected_field_as_road_points=selected_field,
                    selected_correct_road_points=selected_road,
                    precision_for_false_road_points=safe_div(selected_field, selected_points),
                    recall_for_field_as_road_points=safe_div(selected_field, total_field_as_road),
                    introduced_risk_proxy=safe_div(selected_road, selected_points),
                    oracle_after_rule_accuracy=metrics["accuracy"],
                    oracle_after_rule_macro_f1=metrics["macro_f1"],
                    oracle_after_rule_road_f1=metrics["road_f1"],
                    oracle_after_rule_field_f1=metrics["field_f1"],
                    oracle_after_rule_road_as_field=metrics["road_as_field"],
                    oracle_after_rule_field_as_road=metrics["field_as_road"],
                    oracle_delta_macro_f1=metrics["macro_f1"] - baseline["macro_f1"],
                    oracle_delta_field_as_road=metrics["field_as_road"] - baseline["field_as_road"],
                    oracle_delta_road_as_field=metrics["road_as_field"] - baseline["road_as_field"],
                )
            )
        print(f"heuristic_split_done split={split} rules={len(rule_to_runs)}", flush=True)
    return all_rows


def choose_recommended_rule(heuristic_rows):
    valid_rows = [
        row
        for row in heuristic_rows
        if row["split"] == "valid"
        and int(row["selected_runs"]) > 0
        and float(row["precision_for_false_road_points"]) >= 0.65
        and float(row["introduced_risk_proxy"]) <= 0.35
    ]
    if not valid_rows:
        valid_rows = [row for row in heuristic_rows if row["split"] == "valid" and int(row["selected_runs"]) > 0]
    if not valid_rows:
        return None
    return max(
        valid_rows,
        key=lambda row: (
            float(row["oracle_delta_macro_f1"]),
            float(row["precision_for_false_road_points"]),
            float(row["recall_for_field_as_road_points"]),
            -float(row["introduced_risk_proxy"]),
        ),
    )


def run_probe(point_rows, candidate_runs, recommended_rule_name=None):
    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import average_precision_score, roc_auc_score
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
    except Exception as exc:
        return [dict(model_name="sklearn_unavailable", split="all", run_auc="", run_ap="", best_valid_threshold="", selected_runs=0, selected_points=0, precision_for_false_road_points=0.0, recall_for_field_as_road_points=0.0, oracle_delta_macro_f1_if_selected_runs_flipped=0.0, introduced_risk_proxy=0.0, error=str(exc))]

    feature_names = [
        "run_length",
        "trace_position_mean",
        "near_endpoint_rate",
        "prob_road_mean",
        "prob_road_std",
        "prob_field_mean",
        "margin_mean",
        "confidence_mean",
        "stationary_rate",
        "stationary_run_length_mean",
        "stationary_run_length_max",
        "local_density_1m_mean",
        "local_density_1m_max",
        "local_step_mean_m_mean",
        "local_step_mean_m_min",
        "local_step_std_m_mean",
        "local_turn_angle_deg_mean",
        "branch_disagreement_road_mean",
        "branch_disagreement_field_mean",
    ]

    def features(runs):
        data = []
        for run in runs:
            data.append([float(run.get(name, 0.0) or 0.0) for name in feature_names])
        return np.asarray(data, dtype=np.float64)

    train_runs = [run for run in candidate_runs if run["split"] == "train" and (float(run["field_ratio"]) >= 0.80 or float(run["field_ratio"]) <= 0.20)]
    if not train_runs:
        return []
    y_train = np.asarray([1 if float(run["field_ratio"]) >= 0.80 else 0 for run in train_runs], dtype=np.int64)
    if len(set(y_train.tolist())) < 2:
        return []
    x_train = features(train_runs)
    models = {
        "logistic_regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced")),
        "random_forest_depth5": RandomForestClassifier(max_depth=5, n_estimators=100, random_state=42, class_weight="balanced"),
    }
    rows = []
    for model_name, model in models.items():
        model.fit(x_train, y_train)
        valid_runs = split_runs(candidate_runs, "valid")
        valid_scores = model.predict_proba(features(valid_runs))[:, 1] if valid_runs else np.asarray([])
        best_threshold = choose_probe_threshold(model_name, valid_runs, valid_scores, point_rows)
        for split in ("valid", "test"):
            runs = split_runs(candidate_runs, split)
            if not runs:
                continue
            scores = model.predict_proba(features(runs))[:, 1]
            y = np.asarray([1 if float(run["field_ratio"]) >= 0.80 else 0 for run in runs], dtype=np.int64)
            auc = safe_metric(roc_auc_score, y, scores)
            ap = safe_metric(average_precision_score, y, scores)
            selected_ids = {run["candidate_run_id"] for run, score in zip(runs, scores) if float(score) >= best_threshold}
            eval_row = evaluate_selected_run_ids(point_rows, split, selected_ids, candidate_runs=runs)
            rows.append(
                dict(
                    model_name=model_name,
                    split=split,
                    run_auc=auc,
                    run_ap=ap,
                    best_valid_threshold=best_threshold,
                    selected_runs=len(selected_ids),
                    selected_points=eval_row["selected_points"],
                    precision_for_false_road_points=eval_row["precision_for_false_road_points"],
                    recall_for_field_as_road_points=eval_row["recall_for_field_as_road_points"],
                    oracle_delta_macro_f1_if_selected_runs_flipped=eval_row["oracle_delta_macro_f1"],
                    introduced_risk_proxy=eval_row["introduced_risk_proxy"],
                )
            )
    return rows


def safe_metric(fn, y, score):
    try:
        if len(set(np.asarray(y).tolist())) < 2:
            return ""
        return float(fn(y, score))
    except Exception:
        return ""


def choose_probe_threshold(model_name, runs, scores, point_rows):
    if len(runs) == 0:
        return 0.5
    best = None
    for threshold in [0.30, 0.40, 0.50, 0.60, 0.70]:
        selected_ids = {run["candidate_run_id"] for run, score in zip(runs, scores) if float(score) >= threshold}
        row = evaluate_selected_run_ids(point_rows, "valid", selected_ids, candidate_runs=runs)
        score = (
            float(row["oracle_delta_macro_f1"]),
            float(row["precision_for_false_road_points"]),
            -float(row["introduced_risk_proxy"]),
        )
        if best is None or score > best[0]:
            best = (score, threshold)
    return best[1] if best else 0.5


def evaluate_selected_run_ids(point_rows, split, selected_ids, candidate_runs=None):
    rows = split_points(point_rows, split)
    total_points = len(rows)
    total_field_as_road = sum(1 for row in rows if row["error_type"] == "field_as_road")
    if candidate_runs is None:
        selected_rows = [row for row in rows if row.get("candidate_run_id", "") in selected_ids]
        selected_points = len(selected_rows)
        selected_field = sum(1 for row in selected_rows if int(row["label"]) == 1 and int(row["base_pred"]) == 0)
        selected_road = sum(1 for row in selected_rows if int(row["label"]) == 0 and int(row["base_pred"]) == 0)
        labels, base_preds, final_preds = selected_rule_predictions(rows, selected_ids)
        baseline = class_metrics(labels, base_preds)
        metrics = class_metrics(labels, final_preds)
    else:
        selected_runs = [
            run
            for run in candidate_runs
            if run["split"] == split and run["candidate_run_id"] in selected_ids
        ]
        selected_points = sum(int(run["run_length"]) for run in selected_runs)
        selected_field = sum(int(run["field_as_road_points"]) for run in selected_runs)
        selected_road = sum(int(run["correct_road_points"]) for run in selected_runs)
        base_counts = counts_for_rows(rows)
        baseline = metrics_from_counts(base_counts)
        metrics = metrics_after_flipping_runs(base_counts, selected_runs)
    return dict(
        selected_points=selected_points,
        selected_point_rate=safe_div(selected_points, total_points),
        precision_for_false_road_points=safe_div(selected_field, selected_points),
        recall_for_field_as_road_points=safe_div(selected_field, total_field_as_road),
        introduced_risk_proxy=safe_div(selected_road, selected_points),
        oracle_delta_macro_f1=metrics["macro_f1"] - baseline["macro_f1"],
    )


def long_error_run_coverage(point_rows, candidate_runs):
    run_lookup = {row["candidate_run_id"]: row for row in candidate_runs}
    output = []
    for split in ("train", "valid", "test"):
        rows = split_points(point_rows, split)
        row_lookup = {(row["trace_id"], int(row["global_index"])): row for row in rows}
        segments = long_error_segments(rows, "base_pred")
        for segment in segments:
            segment_rows = [
                row_lookup[(segment["trace_id"], idx)]
                for idx in range(int(segment["start_index"]), int(segment["end_index"]) + 1)
                if (segment["trace_id"], idx) in row_lookup
            ]
            candidate_rows = [row for row in segment_rows if row["candidate_mask"]]
            overlap_counts = defaultdict(int)
            for row in candidate_rows:
                run_id = row.get("candidate_run_id", "")
                if run_id in run_lookup:
                    overlap_counts[run_id] += 1
            best_run_id = ""
            best_count = 0
            if overlap_counts:
                best_run_id, best_count = max(overlap_counts.items(), key=lambda item: item[1])
            best_run = run_lookup.get(best_run_id, {})
            output.append(
                dict(
                    split=split,
                    trace_id=segment["trace_id"],
                    error_type=segment["error_type"],
                    start_index=segment["start_index"],
                    end_index=segment["end_index"],
                    length=segment["length"],
                    candidate_points_in_segment=len(candidate_rows),
                    candidate_coverage_rate=safe_div(len(candidate_rows), segment["length"]),
                    covered_by_candidate=bool(overlap_counts),
                    overlapped_candidate_run_ids=";".join(sorted(overlap_counts)),
                    best_overlap_run_id=best_run_id,
                    best_overlap_run_coverage=safe_div(best_count, segment["length"]),
                    run_field_ratio=best_run.get("field_ratio", ""),
                    run_length=best_run.get("run_length", ""),
                )
            )
    return output


def build_summary_rows(point_rows, candidate_runs, oracle_rows, long_rows, recommended_rule, heuristic_rows):
    summary = []
    recommended_rule_name = recommended_rule["rule_name"] if recommended_rule else ""
    valid_delta = recommended_rule.get("oracle_delta_macro_f1", "") if recommended_rule else ""
    test_rule = None
    if recommended_rule:
        for row in heuristic_rows:
            if row["split"] == "test" and row["rule_name"] == recommended_rule_name:
                test_rule = row
                break
    test_delta = test_rule.get("oracle_delta_macro_f1", "") if test_rule else ""
    recommend = should_recommend_train(oracle_rows, heuristic_rows, long_rows, recommended_rule)
    for split in ("train", "valid", "test"):
        rows = split_points(point_rows, split)
        runs = split_runs(candidate_runs, split)
        baseline = class_metrics([row["label"] for row in rows], [row["base_pred"] for row in rows])
        oracle_by_name = {row["oracle_name"]: row for row in oracle_rows if row["split"] == split}
        long_field = [row for row in long_rows if row["split"] == split and row["error_type"] == "field_as_road"]
        covered = [row for row in long_field if str(row["covered_by_candidate"]) == "True" or row["covered_by_candidate"] is True]
        type_counts = defaultdict(int)
        for run in runs:
            type_counts[run["run_type"]] += 1
        candidate_points = sum(int(run["run_length"]) for run in runs)
        summary.append(
            dict(
                split=split,
                baseline_macro_f1=baseline["macro_f1"],
                baseline_road_f1=baseline["road_f1"],
                baseline_field_f1=baseline["field_f1"],
                baseline_road_as_field=baseline["road_as_field"],
                baseline_field_as_road=baseline["field_as_road"],
                baseline_pred_road_rate=baseline["pred_road_rate"],
                candidate_runs=len(runs),
                candidate_run_points=candidate_points,
                candidate_run_point_rate=safe_div(candidate_points, len(rows)),
                pure_false_road_runs=type_counts["pure_false_road_run"],
                mostly_false_road_runs=type_counts["mostly_false_road_run"],
                mixed_runs=type_counts["mixed_run"],
                mostly_true_road_runs=type_counts["mostly_true_road_run"],
                pure_true_road_runs=type_counts["pure_true_road_run"],
                point_oracle_macro_f1=oracle_by_name.get("point_oracle", {}).get("oracle_macro_f1", ""),
                pure_run_oracle_macro_f1=oracle_by_name.get("pure_run_oracle", {}).get("oracle_macro_f1", ""),
                mostly_run_oracle_macro_f1=oracle_by_name.get("mostly_run_oracle", {}).get("oracle_macro_f1", ""),
                point_oracle_delta_macro_f1=oracle_by_name.get("point_oracle", {}).get("delta_macro_f1", ""),
                pure_run_oracle_delta_macro_f1=oracle_by_name.get("pure_run_oracle", {}).get("delta_macro_f1", ""),
                mostly_run_oracle_delta_macro_f1=oracle_by_name.get("mostly_run_oracle", {}).get("delta_macro_f1", ""),
                point_oracle_delta_field_as_road=oracle_by_name.get("point_oracle", {}).get("delta_field_as_road", ""),
                pure_run_oracle_delta_field_as_road=oracle_by_name.get("pure_run_oracle", {}).get("delta_field_as_road", ""),
                mostly_run_oracle_delta_field_as_road=oracle_by_name.get("mostly_run_oracle", {}).get("delta_field_as_road", ""),
                pure_run_introduced_road_as_field=oracle_by_name.get("pure_run_oracle", {}).get("introduced_road_as_field", ""),
                mostly_run_introduced_road_as_field=oracle_by_name.get("mostly_run_oracle", {}).get("introduced_road_as_field", ""),
                long_field_as_road_segments_total=len(long_field),
                long_field_as_road_segments_covered_by_candidate_run=len(covered),
                long_field_as_road_coverage_rate=safe_div(len(covered), len(long_field)),
                recommended_rule_name=recommended_rule_name,
                recommended_rule_valid_delta_macro_f1=valid_delta,
                recommended_rule_test_delta_macro_f1=test_delta,
                recommend_train_run_guard=recommend["recommend_train_run_guard"],
            )
        )
    return summary


def should_recommend_train(oracle_rows, heuristic_rows, long_rows, recommended_rule):
    valid_mostly = next((row for row in oracle_rows if row["split"] == "valid" and row["oracle_name"] == "mostly_run_oracle"), None)
    valid_point = next((row for row in oracle_rows if row["split"] == "valid" and row["oracle_name"] == "point_oracle"), None)
    valid_long = [row for row in long_rows if row["split"] == "valid" and row["error_type"] == "field_as_road"]
    covered = [row for row in valid_long if str(row["covered_by_candidate"]) == "True" or row["covered_by_candidate"] is True]
    recommend = False
    reasons = []
    if valid_mostly and float(valid_mostly["delta_macro_f1"]) > 0.01:
        reasons.append("valid mostly-run oracle has clear macro-F1 headroom")
    if valid_point and valid_mostly and float(valid_mostly["delta_macro_f1"]) >= float(valid_point["delta_macro_f1"]) * 0.5:
        reasons.append("run-level oracle keeps meaningful fraction of point-oracle headroom")
    if safe_div(len(covered), len(valid_long)) >= 0.5:
        reasons.append("long field_as_road segments are substantially covered")
    if recommended_rule and float(recommended_rule["oracle_delta_macro_f1"]) > 0 and float(recommended_rule["precision_for_false_road_points"]) >= 0.65:
        reasons.append("valid heuristic finds usable false-road runs")
    recommend = len(reasons) >= 3
    return dict(
        recommend_train_run_guard=recommend,
        reason="; ".join(reasons) if reasons else "run-level oracle or heuristic evidence is not strong enough",
    )


def build_report(summary_rows, oracle_rows, heuristic_rows, probe_rows, long_rows, recommended, rc_checkpoint):
    valid_summary = next(row for row in summary_rows if row["split"] == "valid")
    test_summary = next(row for row in summary_rows if row["split"] == "test")
    best_valid_rule = recommended or {}
    test_rule = next((row for row in heuristic_rows if row["split"] == "test" and row["rule_name"] == best_valid_rule.get("rule_name")), {})
    recommend = should_recommend_train(oracle_rows, heuristic_rows, long_rows, recommended)
    top_long = [row for row in long_rows if row["split"] == "test" and row["error_type"] == "field_as_road"]
    top_long.sort(key=lambda row: int(row["length"]), reverse=True)
    lines = [
        "# PT2G_MSC_RC_CANDCAL_RUN_GUARD_ORACLE_AUDIT Report",
        "",
        "## Setup",
        "",
        f"- RC checkpoint: {rc_checkpoint}.",
        "- Base model: frozen PT2G_MSC_RC_v1.",
        "- This is an oracle/probe analysis only; no model is trained.",
        "",
        "## Required Answers",
        "",
        f"1. candidate run 数量和点数占比：valid={valid_summary['candidate_runs']} runs, point_rate={valid_summary['candidate_run_point_rate']}; test={test_summary['candidate_runs']} runs, point_rate={test_summary['candidate_run_point_rate']}.",
        f"2. candidate run 中 field_ratio 分布：valid pure_false={valid_summary['pure_false_road_runs']}, mostly_false={valid_summary['mostly_false_road_runs']}, mixed={valid_summary['mixed_runs']}, mostly_true={valid_summary['mostly_true_road_runs']}, pure_true={valid_summary['pure_true_road_runs']}.",
        f"3. 是否存在 pure_false_road_run：{int(valid_summary['pure_false_road_runs']) > 0 or int(test_summary['pure_false_road_runs']) > 0}.",
        f"4. Oracle 上限 valid: point={valid_summary['point_oracle_macro_f1']}, pure={valid_summary['pure_run_oracle_macro_f1']}, mostly={valid_summary['mostly_run_oracle_macro_f1']}; test: point={test_summary['point_oracle_macro_f1']}, pure={test_summary['pure_run_oracle_macro_f1']}, mostly={test_summary['mostly_run_oracle_macro_f1']}.",
        f"5. mostly-run oracle 是否明显高于点级 Guard v2 实际收益：{float(test_summary['mostly_run_oracle_delta_macro_f1']) > 0.001858}.",
        f"6. long_field_as_road 是否大多被 candidate run 覆盖：valid_rate={valid_summary['long_field_as_road_coverage_rate']}, test_rate={test_summary['long_field_as_road_coverage_rate']}.",
        f"7. 最大 field_as_road 长段是否能被覆盖：{top_long[0]['covered_by_candidate'] if top_long else 'no_test_long_field_as_road'}; top segment trace={top_long[0]['trace_id'] if top_long else ''}, length={top_long[0]['length'] if top_long else ''}.",
        f"8. run-level heuristic 规则是否有实用价值：{bool(best_valid_rule)}; recommended_rule={best_valid_rule.get('rule_name', '')}, valid_delta={best_valid_rule.get('oracle_delta_macro_f1', '')}, test_delta={test_rule.get('oracle_delta_macro_f1', '')}.",
        f"9. run-level probe 是否显示可分性：{probe_answer(probe_rows)}.",
        f"10. 是否建议训练 CANDCAL_RUN_GUARD_v1：{recommend['recommend_train_run_guard']}.",
        f"11. 推荐使用什么 run-level 特征和规则：{best_valid_rule.get('rule_name', 'none')}; features should include run_length, stationary_rate, density, local_step_mean, probability/margin, and compactness when available.",
        f"12. 如果不建议训练，原因：{recommend['reason']}.",
        f"13. 打包路径：{PACK_PATH}.",
        "",
        "## Top 20 Test Field-As-Road Long Segments",
        "",
        "| trace_id | start | end | length | candidate_coverage | best_run | run_field_ratio |",
        "|---|---:|---:|---:|---:|---|---:|",
    ]
    for row in top_long[:20]:
        lines.append(
            f"| {row['trace_id']} | {row['start_index']} | {row['end_index']} | {row['length']} | "
            f"{row['candidate_coverage_rate']} | {row['best_overlap_run_id']} | {row['run_field_ratio']} |"
        )
    return "\n".join(lines) + "\n"


def probe_answer(probe_rows):
    values = [float(row["run_auc"]) for row in probe_rows if row.get("split") == "valid" and row.get("run_auc") not in ("", None)]
    if not values:
        return "no usable sklearn probe result"
    return f"valid_auc_max={max(values)}"


def make_pack():
    paths = [
        "experiments/candcal_run_guard_oracle_audit.py",
        "scripts/run_candcal_run_guard_oracle_audit.py",
        "models/Encoder.py",
        str(RESULT_PATH),
        str(REPORT_PATH),
        str(LOG_PATH),
    ]
    paths.extend(str(path) for path in DIAG_DIR.glob("*.csv"))
    paths.extend(str(path) for path in DIAG_DIR.glob("*.json"))
    PACK_PATH.parent.mkdir(parents=True, exist_ok=True)
    if PACK_PATH.exists():
        PACK_PATH.unlink()
    with zipfile.ZipFile(PACK_PATH, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in paths:
            path = Path(path)
            if path.exists():
                zf.write(path, path.as_posix())


def main():
    start = time.time()
    for path in [
        DIAG_DIR / "point_predictions.csv",
        DIAG_DIR / "candidate_runs.csv",
        DIAG_DIR / "run_oracle_metrics.csv",
        DIAG_DIR / "run_heuristic_sweep.csv",
        DIAG_DIR / "run_probe_metrics.csv",
        DIAG_DIR / "long_error_run_coverage.csv",
        DIAG_DIR / "top_long_field_as_road_runs.csv",
        DIAG_DIR / "recommended_run_guard_rules.json",
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

    point_rows = infer_points(model, loaders, graph_caches, device)
    candidate_runs, _all_runs = build_candidate_runs(point_rows)
    print(f"candidate_runs_built eligible={len(candidate_runs)} all={len(_all_runs)}", flush=True)
    oracle_rows = []
    for split in ("train", "valid", "test"):
        rows = split_points(point_rows, split)
        runs = split_runs(candidate_runs, split)
        for mode in ("point_oracle", "pure_run_oracle", "mostly_run_oracle"):
            oracle_rows.append(oracle_metric_row(split, rows, runs, mode))
    print(f"oracle_rows_built rows={len(oracle_rows)}", flush=True)

    heuristic_rows = build_heuristic_rows(point_rows, candidate_runs)
    print(f"heuristic_rows_built rows={len(heuristic_rows)}", flush=True)
    recommended_rule = choose_recommended_rule(heuristic_rows)
    probe_rows = run_probe(point_rows, candidate_runs, recommended_rule["rule_name"] if recommended_rule else None)
    print(f"probe_rows_built rows={len(probe_rows)}", flush=True)
    long_rows = long_error_run_coverage(point_rows, candidate_runs)
    print(f"long_error_rows_built rows={len(long_rows)}", flush=True)
    top_long = [row for row in long_rows if row["error_type"] == "field_as_road"]
    top_long.sort(key=lambda row: int(row["length"]), reverse=True)
    summary_rows = build_summary_rows(point_rows, candidate_runs, oracle_rows, long_rows, recommended_rule, heuristic_rows)
    recommendation = should_recommend_train(oracle_rows, heuristic_rows, long_rows, recommended_rule)

    write_csv(DIAG_DIR / "point_predictions.csv", POINT_FIELDS, point_rows)
    write_csv(DIAG_DIR / "candidate_runs.csv", RUN_FIELDS, candidate_runs)
    write_csv(DIAG_DIR / "run_oracle_metrics.csv", ORACLE_FIELDS, oracle_rows)
    write_csv(DIAG_DIR / "run_heuristic_sweep.csv", HEURISTIC_FIELDS, heuristic_rows)
    write_csv(DIAG_DIR / "run_probe_metrics.csv", PROBE_FIELDS, probe_rows)
    write_csv(DIAG_DIR / "long_error_run_coverage.csv", LONG_FIELDS, long_rows)
    write_csv(DIAG_DIR / "top_long_field_as_road_runs.csv", LONG_FIELDS, top_long[:50])
    write_csv(RESULT_PATH, SUMMARY_FIELDS, summary_rows)
    write_json(
        DIAG_DIR / "recommended_run_guard_rules.json",
        dict(
            recommend_train_run_guard=recommendation["recommend_train_run_guard"],
            reason=recommendation["reason"],
            recommended_run_rule=recommended_rule.get("rule_name") if recommended_rule else "",
            recommended_threshold="valid_selected_rule",
            recommended_training_target="run_positive_if_field_ratio_ge_0p80_negative_if_le_0p20",
            recommended_rule_valid=recommended_rule or {},
            recommended_rule_test=next((row for row in heuristic_rows if recommended_rule and row["split"] == "test" and row["rule_name"] == recommended_rule["rule_name"]), {}),
        ),
    )
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(build_report(summary_rows, oracle_rows, heuristic_rows, probe_rows, long_rows, recommended_rule, rc_checkpoint), encoding="utf-8")
    make_pack()

    print(f"candidate_runs={len(candidate_runs)}", flush=True)
    for row in summary_rows:
        print(
            f"summary split={row['split']} candidate_runs={row['candidate_runs']} "
            f"mostly_oracle_macro={row['mostly_run_oracle_macro_f1']} "
            f"long_coverage={row['long_field_as_road_coverage_rate']} "
            f"recommend={row['recommend_train_run_guard']}",
            flush=True,
        )
    print(f"recommended_rule={recommended_rule.get('rule_name') if recommended_rule else ''}", flush=True)
    print(f"recommend_train_run_guard={recommendation['recommend_train_run_guard']}", flush=True)
    print(f"pack_path={PACK_PATH}", flush=True)
    print(f"elapsed_sec={time.time() - start:.2f}", flush=True)


if __name__ == "__main__":
    main()
