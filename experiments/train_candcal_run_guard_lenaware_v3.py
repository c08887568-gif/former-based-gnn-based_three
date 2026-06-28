import argparse
import csv
import itertools
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

from experiments import train_candcal_run_guard as base
from experiments import train_candcal_run_guard_v2_3runs as v2
from models.CandidateRunFieldGuardCalibration import (
    CANDCAL_RUN_GUARD_DEFAULTS,
    CandidateRunFieldGuardCalibration,
    compute_pos_weight,
)
from utils.threading_config import apply_torch_thread_config, configure_default_threads
from utils.utils import get_default_device


configure_default_threads()
apply_torch_thread_config(torch)


RUN_NAME = "PT2G_MSC_RC_CANDCAL_RUN_GUARD_LENAWARE_v3"
RUN_DIR = Path("runs") / RUN_NAME
DIAG_DIR = Path("diagnostics") / RUN_NAME
RESULT_PATH = Path("results") / f"{RUN_NAME}_summary.csv"
REPORT_PATH = Path("analysis") / f"{RUN_NAME}_report.md"
PACK_PATH = Path("analysis_packs") / f"{RUN_NAME}_for_chatgpt.zip"
LOG_PATH = Path("logs/candcal_run_guard_lenaware_v3/LENAWARE_v3.log")

COMMON_CONFIG = dict(base.COMMON_CONFIG)
COMMON_CONFIG.update(
    {
        "use_pretrain": True,
        "pretrained_path": "weights/PT2_edge_weight_pretrain.pt",
        "pretrain_mode": "edge_weight",
        "cache_dir": "cache/wheat_non_iid",
        "graph_cache_path": "cache/pretrained_graphs/PT2G_topk3",
        "segment_context_mode": "msc",
        "msc_aux_mode": "rc",
        "skip_test": False,
        "run_name": RUN_NAME,
    }
)

EXP_CFG = {
    "group": RUN_NAME,
    "weighted_loss": True,
    "adaptive_delta": True,
    "seed": 46,
}

LENAWARE_THRESHOLDS = {
    "long": [0.20, 0.25, 0.30, 0.35, 0.40],
    "mid": [0.40, 0.50, 0.60, 0.70],
    "short": [0.70, 0.80, 0.90],
}

WEIGHTED_V2_REFERENCE = {
    "cal_test_macro_f1": 0.8897600820442955,
    "cal_test_road_f1": 0.8267300063666194,
    "cal_test_field_f1": 0.9527901577219716,
    "cal_field_as_road": 3260,
    "cal_long_field_as_road_points": 2378,
}

FOCUS_SEGMENTS = [
    {
        "name": "max_719_field_as_road",
        "trace_id": "wheat_1_harvestor_124",
        "start_index": 8938,
        "end_index": 9656,
    },
    {
        "name": "field_as_road_225",
        "trace_id": "wheat_1_harvestor_80",
        "start_index": 8775,
        "end_index": 8999,
    },
]


def trace_matches(actual_trace_id, expected_short_name):
    actual = str(actual_trace_id)
    return (
        actual == expected_short_name
        or Path(actual).stem == expected_short_name
        or actual.endswith(f"/{expected_short_name}.xlsx")
    )

THRESHOLD_FIELDS = [
    "epoch",
    "long_threshold",
    "mid_threshold",
    "short_threshold",
    "threshold_config",
    "selection_status",
    "safe_threshold",
    "base_accuracy",
    "base_macro_f1",
    "base_road_f1",
    "base_field_f1",
    "base_pred_road_rate",
    "base_pred_field_rate",
    "base_road_as_field",
    "base_field_as_road",
    "base_long_road_as_field_segments",
    "base_long_field_as_road_segments",
    "base_long_road_as_field_points",
    "base_long_field_as_road_points",
    "cal_accuracy",
    "cal_macro_f1",
    "cal_road_f1",
    "cal_field_f1",
    "cal_pred_road_rate",
    "cal_pred_field_rate",
    "cal_road_as_field",
    "cal_field_as_road",
    "cal_long_road_as_field_segments",
    "cal_long_field_as_road_segments",
    "cal_long_road_as_field_points",
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
    "selected_long_runs",
    "selected_mid_runs",
    "selected_short_runs",
    "guard_applied_point_rate",
    "fixed_field_as_road",
    "introduced_road_as_field",
    "fixed_minus_introduced",
    "selected_false_run_rate",
]

SUMMARY_FIELDS = [
    "group",
    "best_epoch",
    "best_long_threshold",
    "best_mid_threshold",
    "best_short_threshold",
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
    "selected_long_runs",
    "selected_mid_runs",
    "selected_short_runs",
    "guard_applied_point_rate",
    "fixed_field_as_road",
    "introduced_road_as_field",
    "fixed_minus_introduced",
    "max_field_as_road_segment_selected",
    "max_field_as_road_segment_score",
    "max_field_as_road_segment_threshold",
    "max_field_as_road_segment_fixed_points",
    "max_field_as_road_segment_remaining_points",
]

RUN_CSV_FIELDS = list(base.RUN_CSV_FIELDS) + ["run_length_group", "threshold_used"]
POINT_TEST_FIELDS = list(v2.POINT_TEST_FIELDS) + ["run_length_group", "threshold_used"]
TOP_LONG_FIELDS = [
    "trace_id",
    "start_index",
    "end_index",
    "length",
    "candidate_run_id",
    "run_length",
    "run_length_group",
    "run_guard_score",
    "threshold_used",
    "selected_by_guard",
    "base_prob_road_mean",
    "base_margin_mean",
    "fixed_points_in_segment",
    "remaining_error_points",
    "coverage_rate",
]
TRAINING_FIELDS = [
    "epoch",
    "train_loss",
    "valid_best_long_threshold",
    "valid_best_mid_threshold",
    "valid_best_short_threshold",
    "valid_best_macro_f1",
    "valid_best_fixed",
    "valid_best_introduced",
    "valid_best_long_field_points",
    "selection_status",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Train length-aware CANDCAL Run Guard v3.")
    parser.add_argument("--epochs", type=int, default=20)
    return parser.parse_args()


def apply_common_config():
    base.RUN_NAME = RUN_NAME
    base.RUN_DIR = RUN_DIR
    base.DIAG_DIR = DIAG_DIR
    base.RESULT_PATH = RESULT_PATH
    base.REPORT_PATH = REPORT_PATH
    base.PACK_PATH = PACK_PATH
    base.LOG_PATH = LOG_PATH
    base.COMMON_CONFIG.update(COMMON_CONFIG)


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


def split_runs(runs, split):
    return [run for run in runs if run["split"] == split]


def split_rows(rows, split):
    return [row for row in rows if row["split"] == split]


def threshold_configs():
    for long_thr, mid_thr, short_thr in itertools.product(
        LENAWARE_THRESHOLDS["long"],
        LENAWARE_THRESHOLDS["mid"],
        LENAWARE_THRESHOLDS["short"],
    ):
        yield {
            "long_threshold": float(long_thr),
            "mid_threshold": float(mid_thr),
            "short_threshold": float(short_thr),
        }


def run_length_group(run):
    length = int(run.get("length", 0))
    if length >= 100:
        return "long"
    if length >= 20:
        return "mid"
    return "short"


def threshold_for_run(run, config):
    group = run_length_group(run)
    return float(config[f"{group}_threshold"])


def config_to_string(config):
    return (
        f"long={config['long_threshold']:.2f},"
        f"mid={config['mid_threshold']:.2f},"
        f"short={config['short_threshold']:.2f}"
    )


def selected_run_ids_for_config(runs, score_map, config):
    selected = set()
    for run in runs:
        score = float(score_map.get(run["candidate_run_id"], 0.0))
        if score >= threshold_for_run(run, config):
            selected.add(run["candidate_run_id"])
    return selected


def train_run_guard_head(runs, point_rows_by_split, device, epochs):
    set_seed(EXP_CFG["seed"])
    train_runs = [run for run in split_runs(runs, "train") if int(run["run_target"]) in (0, 1)]
    if not train_runs:
        raise SystemExit("RUN_GUARD_NO_TRAIN_TARGETS")

    targets = np.asarray([int(run["run_target"]) for run in train_runs], dtype=np.float32)
    pos_weight, positives, negatives = compute_pos_weight(
        torch.from_numpy(targets),
        CANDCAL_RUN_GUARD_DEFAULTS["pos_weight_clip"],
    )
    if pos_weight is None:
        print(f"RUN_GUARD_TARGET_CLASS_WARNING positives={positives} negatives={negatives}", flush=True)
        pos_weight = torch.tensor(1.0, dtype=torch.float32)

    feature_mean, feature_std = base.fit_feature_stats(train_runs)
    head = CandidateRunFieldGuardCalibration(input_dim=len(base.FEATURE_NAMES)).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device), reduction="none")
    x_train = torch.from_numpy(base.feature_matrix(train_runs, feature_mean, feature_std)).to(device)
    y_train = torch.from_numpy(targets).to(device)
    sample_weights = torch.from_numpy(v2.sample_weights_for_runs(train_runs, targets, True)).to(device)

    threshold_rows = []
    training_rows = []
    best = None
    best_state = None
    for epoch in range(1, int(epochs) + 1):
        head.train()
        perm = torch.randperm(x_train.shape[0], device=device)
        epoch_loss = 0.0
        batch_count = 0
        for start in range(0, x_train.shape[0], 128):
            idx = perm[start : start + 128]
            optimizer.zero_grad()
            logits = head(x_train[idx])
            raw_loss = criterion(logits, y_train[idx])
            loss = (raw_loss * sample_weights[idx]).mean()
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.detach().cpu().item())
            batch_count += 1

        valid_scores = base.score_runs(head, split_runs(runs, "valid"), feature_mean, feature_std, device)
        epoch_sweep = evaluate_lenaware_threshold_sweep(
            "valid",
            point_rows_by_split["valid"],
            split_runs(runs, "valid"),
            valid_scores,
            epoch,
        )
        selected = choose_best_threshold_row(epoch_sweep)
        threshold_rows.extend(epoch_sweep)
        training_rows.append(
            dict(
                epoch=epoch,
                train_loss=base.safe_div(epoch_loss, batch_count),
                valid_best_long_threshold=selected["long_threshold"],
                valid_best_mid_threshold=selected["mid_threshold"],
                valid_best_short_threshold=selected["short_threshold"],
                valid_best_macro_f1=selected["cal_macro_f1"],
                valid_best_fixed=selected["fixed_field_as_road"],
                valid_best_introduced=selected["introduced_road_as_field"],
                valid_best_long_field_points=selected["cal_long_field_as_road_points"],
                selection_status=selected["selection_status"],
            )
        )
        if best is None or threshold_row_is_better(selected, best):
            best = dict(selected)
            best_state = {key: value.detach().cpu().clone() for key, value in head.state_dict().items()}
        print(
            f"epoch={epoch} train_loss={base.safe_div(epoch_loss, batch_count):.6f} "
            f"valid_macro={selected['cal_macro_f1']:.6f} "
            f"thr=({selected['long_threshold']},{selected['mid_threshold']},{selected['short_threshold']}) "
            f"fixed={selected['fixed_field_as_road']} introduced={selected['introduced_road_as_field']} "
            f"long_field_points={selected['cal_long_field_as_road_points']} status={selected['selection_status']}",
            flush=True,
        )

    if best_state is not None:
        head.load_state_dict(best_state)
    return head, feature_mean, feature_std, best, threshold_rows, training_rows, positives, negatives


def evaluate_lenaware_threshold_sweep(split, rows, runs, score_map, epoch):
    rows_out = [
        evaluate_lenaware_threshold(split, rows, runs, score_map, config, epoch)
        for config in threshold_configs()
    ]
    selected = choose_best_threshold_row(rows_out)
    for row in rows_out:
        row["selection_status"] = "selected_" + selected["selection_status"] if row is selected else row["selection_status"]
    return rows_out


def evaluate_lenaware_threshold(split, rows, runs, score_map, config, epoch=None):
    selected_run_ids = selected_run_ids_for_config(runs, score_map, config)
    labels, base_preds, final_preds, behavior = calibrated_predictions(rows, selected_run_ids, score_map)
    base_metrics = base.class_metrics(labels, base_preds)
    cal_metrics = base.class_metrics(labels, final_preds)
    base_long = v2.long_error_segments_from_preds(rows, base_preds, split, "base", RUN_NAME)
    cal_long = v2.long_error_segments_from_preds(rows, final_preds, split, "calibrated", RUN_NAME)
    base_road_long = v2.summarize_long_segments(base_long, "road_as_field")
    base_field_long = v2.summarize_long_segments(base_long, "field_as_road")
    cal_road_long = v2.summarize_long_segments(cal_long, "road_as_field")
    cal_field_long = v2.summarize_long_segments(cal_long, "field_as_road")
    candidate_points = sum(int(run["length"]) for run in runs)
    selected_runs = [run for run in runs if run["candidate_run_id"] in selected_run_ids]
    selected_points = sum(int(run["length"]) for run in selected_runs)
    selected_false = sum(
        1
        for run in selected_runs
        if float(run["field_ratio"]) >= CANDCAL_RUN_GUARD_DEFAULTS["positive_field_ratio"]
    )
    selected_group_counts = defaultdict(int)
    for run in selected_runs:
        selected_group_counts[run_length_group(run)] += 1

    row = dict(
        epoch=epoch if epoch is not None else "",
        long_threshold=float(config["long_threshold"]),
        mid_threshold=float(config["mid_threshold"]),
        short_threshold=float(config["short_threshold"]),
        threshold_config=config_to_string(config),
        selection_status="unselected",
        base_accuracy=base_metrics["accuracy"],
        base_macro_f1=base_metrics["macro_f1"],
        base_road_f1=base_metrics["road_f1"],
        base_field_f1=base_metrics["field_f1"],
        base_pred_road_rate=base_metrics["pred_road_rate"],
        base_pred_field_rate=base_metrics["pred_field_rate"],
        base_road_as_field=base_metrics["road_as_field"],
        base_field_as_road=base_metrics["field_as_road"],
        base_long_road_as_field_segments=base_road_long["segments"],
        base_long_field_as_road_segments=base_field_long["segments"],
        base_long_road_as_field_points=base_road_long["points"],
        base_long_field_as_road_points=base_field_long["points"],
        cal_accuracy=cal_metrics["accuracy"],
        cal_macro_f1=cal_metrics["macro_f1"],
        cal_road_f1=cal_metrics["road_f1"],
        cal_field_f1=cal_metrics["field_f1"],
        cal_pred_road_rate=cal_metrics["pred_road_rate"],
        cal_pred_field_rate=cal_metrics["pred_field_rate"],
        cal_road_as_field=cal_metrics["road_as_field"],
        cal_field_as_road=cal_metrics["field_as_road"],
        cal_long_road_as_field_segments=cal_road_long["segments"],
        cal_long_field_as_road_segments=cal_field_long["segments"],
        cal_long_road_as_field_points=cal_road_long["points"],
        cal_long_field_as_road_points=cal_field_long["points"],
        delta_macro_f1=cal_metrics["macro_f1"] - base_metrics["macro_f1"],
        delta_road_f1=cal_metrics["road_f1"] - base_metrics["road_f1"],
        delta_field_f1=cal_metrics["field_f1"] - base_metrics["field_f1"],
        delta_road_as_field=cal_metrics["road_as_field"] - base_metrics["road_as_field"],
        delta_field_as_road=cal_metrics["field_as_road"] - base_metrics["field_as_road"],
        delta_long_field_as_road_segments=cal_field_long["segments"] - base_field_long["segments"],
        delta_long_field_as_road_points=cal_field_long["points"] - base_field_long["points"],
        candidate_runs=len(runs),
        candidate_run_points=candidate_points,
        selected_runs=len(selected_runs),
        selected_run_points=selected_points,
        selected_long_runs=selected_group_counts["long"],
        selected_mid_runs=selected_group_counts["mid"],
        selected_short_runs=selected_group_counts["short"],
        guard_applied_point_rate=base.safe_div(selected_points, len(rows)),
        fixed_field_as_road=behavior["fixed_field_as_road"],
        introduced_road_as_field=behavior["introduced_road_as_field"],
        fixed_minus_introduced=behavior["fixed_field_as_road"] - behavior["introduced_road_as_field"],
        selected_false_run_rate=base.safe_div(selected_false, len(selected_runs)),
    )
    row["safe_threshold"] = is_safe_threshold(row)
    return row


def calibrated_predictions(rows, selected_run_ids, score_map):
    labels = []
    base_preds = []
    final_preds = []
    fixed = 0
    introduced = 0
    for row in rows:
        label = int(row["label"])
        base_pred = int(row["base_pred"])
        road_logit = float(row["base_road_logit"])
        field_logit = float(row["base_field_logit"])
        run_id = row.get("candidate_run_id", "")
        if run_id in selected_run_ids:
            delta = v2.calibration_delta(row, float(score_map.get(run_id, 0.0)), True)
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


def is_safe_threshold(row):
    return (
        float(row["cal_pred_road_rate"]) >= 0.20
        and float(row["cal_road_f1"]) >= float(row["base_road_f1"]) - 0.006
        and int(row["fixed_field_as_road"]) > int(row["introduced_road_as_field"])
        and int(row["introduced_road_as_field"]) <= int(row["fixed_field_as_road"])
    )


def choose_best_threshold_row(rows):
    safe = [row for row in rows if is_safe_threshold(row)]
    baseline = float(rows[0]["base_macro_f1"]) if rows else 0.0
    safe_improved = [row for row in safe if float(row["cal_macro_f1"]) > baseline]
    if safe_improved:
        selected = best_by_metric_then_long(safe_improved)
        selected["selection_status"] = "safe_improved"
        return selected
    if safe:
        selected = safest_row(safe)
        selected["selection_status"] = "safe_no_macro_gain"
        return selected
    selected = safest_row(rows)
    selected["selection_status"] = "unsafe_fallback"
    return selected


def best_by_metric_then_long(rows):
    best = None
    for row in rows:
        if best is None or row_is_better(row, best):
            best = row
    return best


def row_is_better(candidate, current):
    macro_delta = float(candidate["cal_macro_f1"]) - float(current["cal_macro_f1"])
    if macro_delta > 0.001:
        return True
    if macro_delta < -0.001:
        return False
    if int(candidate["cal_long_field_as_road_points"]) != int(current["cal_long_field_as_road_points"]):
        return int(candidate["cal_long_field_as_road_points"]) < int(current["cal_long_field_as_road_points"])
    if int(candidate["cal_field_as_road"]) != int(current["cal_field_as_road"]):
        return int(candidate["cal_field_as_road"]) < int(current["cal_field_as_road"])
    if int(candidate["introduced_road_as_field"]) != int(current["introduced_road_as_field"]):
        return int(candidate["introduced_road_as_field"]) < int(current["introduced_road_as_field"])
    cand_rate = abs(float(candidate["cal_pred_road_rate"]) - float(candidate["base_pred_road_rate"]))
    curr_rate = abs(float(current["cal_pred_road_rate"]) - float(current["base_pred_road_rate"]))
    if abs(cand_rate - curr_rate) > 1e-12:
        return cand_rate < curr_rate
    if int(candidate["selected_long_runs"]) != int(current["selected_long_runs"]):
        return int(candidate["selected_long_runs"]) > int(current["selected_long_runs"])
    if float(candidate["long_threshold"]) != float(current["long_threshold"]):
        return float(candidate["long_threshold"]) < float(current["long_threshold"])
    if float(candidate["mid_threshold"]) != float(current["mid_threshold"]):
        return float(candidate["mid_threshold"]) > float(current["mid_threshold"])
    return float(candidate["short_threshold"]) > float(current["short_threshold"])


def safest_row(rows):
    return max(
        rows,
        key=lambda row: (
            int(is_safe_threshold(row)),
            int(row["fixed_minus_introduced"]),
            -int(row["introduced_road_as_field"]),
            float(row["cal_road_f1"]),
            float(row["cal_macro_f1"]),
            -abs(float(row["cal_pred_road_rate"]) - float(row["base_pred_road_rate"])),
            int(row["selected_long_runs"]),
        ),
    )


def threshold_row_is_better(candidate, current):
    if current is None:
        return True
    candidate_rank = selection_rank(candidate)
    current_rank = selection_rank(current)
    if candidate_rank != current_rank:
        return candidate_rank > current_rank
    return row_is_better(candidate, current)


def selection_rank(row):
    status = row.get("selection_status", "")
    if status.endswith("safe_improved") or status == "safe_improved":
        return 2
    if status.endswith("safe_no_macro_gain") or status == "safe_no_macro_gain":
        return 1
    return 0


def update_run_scores(runs, score_map, config):
    for run in runs:
        score = float(score_map.get(run["candidate_run_id"], 0.0))
        length_group = run_length_group(run)
        threshold = threshold_for_run(run, config)
        run["run_guard_score"] = score
        run["run_length_group"] = length_group
        run["threshold_used"] = threshold
        run["selected_by_best_threshold"] = score >= threshold


def annotate_test_rows(rows, runs, score_map, config):
    run_lookup = {run["candidate_run_id"]: run for run in runs}
    out = []
    for row in rows:
        run_id = row.get("candidate_run_id", "")
        run = run_lookup.get(run_id)
        score = float(score_map.get(run_id, 0.0)) if run else 0.0
        length_group = run_length_group(run) if run else ""
        threshold = threshold_for_run(run, config) if run else ""
        selected = bool(run and score >= float(threshold))
        road_logit = float(row["base_road_logit"])
        field_logit = float(row["base_field_logit"])
        delta = 0.0
        if selected:
            delta = v2.calibration_delta(row, score, True)
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
        row["_delta_applied"] = delta
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
            run_guard_score=score if run else "",
            selected_by_guard=selected,
            fixed_field_as_road=int(row["base_pred"] == 0 and row["label"] == 1 and final_pred == 1),
            introduced_road_as_field=int(row["base_pred"] == 0 and row["label"] == 0 and final_pred == 1),
            error_type_base=row["error_type_base"],
            error_type_final=base.error_type(row["label"], final_pred),
            lon=row["lon"],
            lat=row["lat"],
            delta_applied=delta,
            run_length_group=length_group,
            threshold_used=threshold,
        )
        for name in base.AUX_FEATURE_NAMES:
            out_row[name] = row[name]
        out.append(out_row)
    return out


def segment_overlap(a_start, a_end, b_start, b_end):
    return max(0, min(int(a_end), int(b_end)) - max(int(a_start), int(b_start)) + 1)


def selected_base_field_segments(test_rows):
    base_preds = np.asarray([int(row["base_pred"]) for row in test_rows], dtype=np.int64)
    base_segments = v2.long_error_segments_from_preds(test_rows, base_preds, "test", "base", RUN_NAME)
    base_field = [row for row in base_segments if row["error_type"] == "field_as_road"]
    base_field.sort(key=lambda row: int(row["length"]), reverse=True)
    selected = list(base_field[:20])
    selected_keys = {(row["trace_id"], int(row["start_index"]), int(row["end_index"])) for row in selected}
    for target in FOCUS_SEGMENTS:
        matches = [
            row
            for row in base_field
            if trace_matches(row["trace_id"], target["trace_id"])
            and segment_overlap(row["start_index"], row["end_index"], target["start_index"], target["end_index"]) > 0
        ]
        if not matches:
            continue
        best = max(
            matches,
            key=lambda row: segment_overlap(row["start_index"], row["end_index"], target["start_index"], target["end_index"]),
        )
        key = (best["trace_id"], int(best["start_index"]), int(best["end_index"]))
        if key not in selected_keys:
            selected.append(best)
            selected_keys.add(key)
    return selected


def top_long_field_as_road_changes(test_rows, test_runs):
    run_lookup = {run["candidate_run_id"]: run for run in test_runs}
    rows_by_trace_index = {(row["trace_id"], int(row["global_index"])): row for row in test_rows}
    out = []
    for segment in selected_base_field_segments(test_rows):
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
        best_run_id = max(overlap_counts.items(), key=lambda item: item[1])[0] if overlap_counts else ""
        best_run = run_lookup.get(best_run_id, {})
        fixed = sum(
            1
            for row in segment_rows
            if int(row["label"]) == 1 and int(row["base_pred"]) == 0 and int(row.get("_final_pred", row["base_pred"])) == 1
        )
        remaining = sum(
            1
            for row in segment_rows
            if int(row["label"]) == 1 and int(row.get("_final_pred", row["base_pred"])) == 0
        )
        candidate_points = sum(1 for row in segment_rows if row.get("candidate_run_id", ""))
        out.append(
            dict(
                trace_id=segment["trace_id"],
                start_index=segment["start_index"],
                end_index=segment["end_index"],
                length=segment["length"],
                candidate_run_id=best_run_id,
                run_length=best_run.get("length", ""),
                run_length_group=best_run.get("run_length_group", ""),
                run_guard_score=best_run.get("run_guard_score", ""),
                threshold_used=best_run.get("threshold_used", ""),
                selected_by_guard=best_run.get("selected_by_best_threshold", False),
                base_prob_road_mean=base.mean([row["base_prob_road"] for row in segment_rows]),
                base_margin_mean=base.mean([row["margin"] for row in segment_rows]),
                fixed_points_in_segment=fixed,
                remaining_error_points=remaining,
                coverage_rate=base.safe_div(candidate_points, segment["length"]),
            )
        )
    return out


def make_summary(best_epoch, best_config, test_eval, top_changes):
    max_segment = top_changes[0] if top_changes else {}
    return dict(
        group=RUN_NAME,
        best_epoch=best_epoch,
        best_long_threshold=best_config["long_threshold"],
        best_mid_threshold=best_config["mid_threshold"],
        best_short_threshold=best_config["short_threshold"],
        base_test_accuracy=test_eval["base_accuracy"],
        base_test_macro_f1=test_eval["base_macro_f1"],
        base_test_road_f1=test_eval["base_road_f1"],
        base_test_field_f1=test_eval["base_field_f1"],
        base_pred_road_rate=test_eval["base_pred_road_rate"],
        base_road_as_field=test_eval["base_road_as_field"],
        base_field_as_road=test_eval["base_field_as_road"],
        base_long_road_as_field_segments=test_eval["base_long_road_as_field_segments"],
        base_long_field_as_road_segments=test_eval["base_long_field_as_road_segments"],
        base_long_field_as_road_points=test_eval["base_long_field_as_road_points"],
        cal_test_accuracy=test_eval["cal_accuracy"],
        cal_test_macro_f1=test_eval["cal_macro_f1"],
        cal_test_road_f1=test_eval["cal_road_f1"],
        cal_test_field_f1=test_eval["cal_field_f1"],
        cal_pred_road_rate=test_eval["cal_pred_road_rate"],
        cal_road_as_field=test_eval["cal_road_as_field"],
        cal_field_as_road=test_eval["cal_field_as_road"],
        cal_long_road_as_field_segments=test_eval["cal_long_road_as_field_segments"],
        cal_long_field_as_road_segments=test_eval["cal_long_field_as_road_segments"],
        cal_long_field_as_road_points=test_eval["cal_long_field_as_road_points"],
        delta_macro_f1=test_eval["delta_macro_f1"],
        delta_road_f1=test_eval["delta_road_f1"],
        delta_field_f1=test_eval["delta_field_f1"],
        delta_road_as_field=test_eval["delta_road_as_field"],
        delta_field_as_road=test_eval["delta_field_as_road"],
        delta_long_field_as_road_segments=test_eval["delta_long_field_as_road_segments"],
        delta_long_field_as_road_points=test_eval["delta_long_field_as_road_points"],
        candidate_runs=test_eval["candidate_runs"],
        candidate_run_points=test_eval["candidate_run_points"],
        selected_runs=test_eval["selected_runs"],
        selected_run_points=test_eval["selected_run_points"],
        selected_long_runs=test_eval["selected_long_runs"],
        selected_mid_runs=test_eval["selected_mid_runs"],
        selected_short_runs=test_eval["selected_short_runs"],
        guard_applied_point_rate=test_eval["guard_applied_point_rate"],
        fixed_field_as_road=test_eval["fixed_field_as_road"],
        introduced_road_as_field=test_eval["introduced_road_as_field"],
        fixed_minus_introduced=test_eval["fixed_minus_introduced"],
        max_field_as_road_segment_selected=max_segment.get("selected_by_guard", ""),
        max_field_as_road_segment_score=max_segment.get("run_guard_score", ""),
        max_field_as_road_segment_threshold=max_segment.get("threshold_used", ""),
        max_field_as_road_segment_fixed_points=max_segment.get("fixed_points_in_segment", 0),
        max_field_as_road_segment_remaining_points=max_segment.get("remaining_error_points", 0),
    )


def focus_segment_row(top_changes, trace_id, start_index, end_index):
    matches = [
        row
        for row in top_changes
        if trace_matches(row["trace_id"], trace_id)
        and segment_overlap(row["start_index"], row["end_index"], start_index, end_index) > 0
    ]
    if not matches:
        return None
    return max(
        matches,
        key=lambda row: segment_overlap(row["start_index"], row["end_index"], start_index, end_index),
    )


def load_weighted_v2_reference():
    path = Path("results/PT2G_MSC_RC_CANDCAL_RUN_GUARD_v2_3runs_summary.csv")
    if not path.exists():
        return dict(WEIGHTED_V2_REFERENCE)
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("group") == "PT2G_MSC_RC_CANDCAL_RUN_GUARD_WEIGHTED_v2":
                return row
    return dict(WEIGHTED_V2_REFERENCE)


def yes_no(value):
    return "是" if value else "否"


def build_report(summary, best_valid, top_changes):
    weighted_ref = load_weighted_v2_reference()
    exceeds_rc = float(summary["cal_test_macro_f1"]) > float(summary["base_test_macro_f1"])
    exceeds_weighted = float(summary["cal_test_macro_f1"]) > float(weighted_ref["cal_test_macro_f1"])
    field_continue_down = int(summary["cal_field_as_road"]) < int(weighted_ref["cal_field_as_road"])
    long_continue_down = int(summary["cal_long_field_as_road_points"]) < int(weighted_ref["cal_long_field_as_road_points"])
    road_ok = float(summary["cal_test_road_f1"]) >= 0.815
    pred_rate_ok = float(summary["cal_pred_road_rate"]) >= 0.20
    fixed_ok = int(summary["fixed_field_as_road"]) > int(summary["introduced_road_as_field"])
    max_row = top_changes[0] if top_changes else {}
    row_719 = focus_segment_row(top_changes, "wheat_1_harvestor_124", 8938, 9656)
    row_225 = focus_segment_row(top_changes, "wheat_1_harvestor_80", 8775, 8999)
    max_fixed = int(max_row.get("fixed_points_in_segment", 0) or 0)
    max_remaining = int(max_row.get("remaining_error_points", 0) or 0)
    new_long_road = int(summary["cal_long_road_as_field_segments"]) > int(summary["base_long_road_as_field_segments"])
    recommend_mainline = (
        exceeds_weighted
        and field_continue_down
        and long_continue_down
        and max_fixed > 0
        and road_ok
        and pred_rate_ok
        and fixed_ok
    )
    valuable = (
        float(summary["cal_test_macro_f1"]) >= float(weighted_ref["cal_test_macro_f1"]) - 0.001
        and max_fixed > 0
        and road_ok
        and fixed_ok
    )
    repair_oracle = max_remaining > 0 or not max_fixed

    def segment_answer(row):
        if not row:
            return "未在 base long field_as_road 段中找到重叠记录。"
        selected = yes_no(str(row.get("selected_by_guard", "False")) == "True" or row.get("selected_by_guard") is True)
        return (
            f"trace={row['trace_id']} {row['start_index']}~{row['end_index']} length={row['length']}，"
            f"run={row.get('candidate_run_id', '')}，run_length={row.get('run_length', '')}，"
            f"group={row.get('run_length_group', '')}，score={row.get('run_guard_score', '')}，"
            f"threshold={row.get('threshold_used', '')}，selected={selected}，"
            f"fixed={row.get('fixed_points_in_segment', '')}，remaining={row.get('remaining_error_points', '')}。"
        )

    lines = [
        "# PT2G_MSC_RC_CANDCAL_RUN_GUARD_LENAWARE_v3 Report",
        "",
        "## Summary",
        "",
        "| group | best_epoch | long_thr | mid_thr | short_thr | macro_f1 | road_f1 | field_f1 | field_as_road | long_field_points | fixed | introduced |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        (
            f"| {summary['group']} | {summary['best_epoch']} | {summary['best_long_threshold']} | "
            f"{summary['best_mid_threshold']} | {summary['best_short_threshold']} | "
            f"{summary['cal_test_macro_f1']} | {summary['cal_test_road_f1']} | {summary['cal_test_field_f1']} | "
            f"{summary['cal_field_as_road']} | {summary['cal_long_field_as_road_points']} | "
            f"{summary['fixed_field_as_road']} | {summary['introduced_road_as_field']} |"
        ),
        "",
        "## Required Answers",
        "",
        f"1. LENAWARE_v3 是否超过 RC baseline：{yes_no(exceeds_rc)}，base_macro={summary['base_test_macro_f1']}，cal_macro={summary['cal_test_macro_f1']}。",
        f"2. 是否超过 CANDCAL_RUN_GUARD_WEIGHTED_v2：{yes_no(exceeds_weighted)}，weighted_v2_macro={weighted_ref['cal_test_macro_f1']}。",
        f"3. best threshold config：long={summary['best_long_threshold']}，mid={summary['best_mid_threshold']}，short={summary['best_short_threshold']}，valid_status={best_valid['selection_status']}。",
        f"4. 长段是否确实用了更低阈值：{yes_no(float(summary['best_long_threshold']) < float(summary['best_mid_threshold']) <= float(summary['best_short_threshold']))}。",
        f"5. 最大 719 点 field_as_road 长段是否被修动：{yes_no(bool(row_719) and int(row_719.get('fixed_points_in_segment', 0) or 0) > 0)}；{segment_answer(row_719)}",
        f"6. 225 点 field_as_road 长段是否被修动：{yes_no(bool(row_225) and int(row_225.get('fixed_points_in_segment', 0) or 0) > 0)}；{segment_answer(row_225)}",
        f"7. long_field_as_road_points 是否明显下降：{yes_no(int(summary['delta_long_field_as_road_points']) < 0)}，{summary['base_long_field_as_road_points']} -> {summary['cal_long_field_as_road_points']}。",
        f"8. field_as_road 是否继续下降：{yes_no(field_continue_down)}，weighted_v2={weighted_ref['cal_field_as_road']}，lenaware={summary['cal_field_as_road']}。",
        f"9. road-F1 是否被破坏：{yes_no(not road_ok)}，road_f1={summary['cal_test_road_f1']}。",
        f"10. pred_road_rate 是否过低：{yes_no(not pred_rate_ok)}，pred_road_rate={summary['cal_pred_road_rate']}。",
        f"11. fixed_field_as_road 是否大于 introduced_road_as_field：{yes_no(fixed_ok)}，fixed={summary['fixed_field_as_road']}，introduced={summary['introduced_road_as_field']}。",
        f"12. 是否建议把 LENAWARE_v3 作为新主线：{yes_no(recommend_mainline or valuable)}；严格主线={yes_no(recommend_mainline)}。",
        f"13. 是否建议下一步做 CANDCAL_RUN_REPAIR_ORACLE_AUDIT：{yes_no(repair_oracle)}。",
        f"14. 如果失败，失败原因：{failure_reason(summary, weighted_ref, max_fixed, road_ok, pred_rate_ok, fixed_ok, new_long_road)}。",
        f"15. 打包路径：{PACK_PATH}。",
        "",
        "## Top Long Field-As-Road Changes",
        "",
        "| trace_id | start | end | length | run_length | group | score | threshold | selected | fixed | remaining |",
        "|---|---:|---:|---:|---:|---|---:|---:|---|---:|---:|",
    ]
    for row in top_changes[:20]:
        lines.append(
            f"| {row['trace_id']} | {row['start_index']} | {row['end_index']} | {row['length']} | "
            f"{row.get('run_length', '')} | {row.get('run_length_group', '')} | {row.get('run_guard_score', '')} | "
            f"{row.get('threshold_used', '')} | {row.get('selected_by_guard', '')} | "
            f"{row.get('fixed_points_in_segment', '')} | {row.get('remaining_error_points', '')} |"
        )
    return "\n".join(lines) + "\n"


def failure_reason(summary, weighted_ref, max_fixed, road_ok, pred_rate_ok, fixed_ok, new_long_road):
    reasons = []
    if float(summary["cal_test_macro_f1"]) < float(summary["base_test_macro_f1"]):
        reasons.append("macro-F1 低于 RC baseline")
    if float(summary["cal_test_macro_f1"]) <= float(weighted_ref["cal_test_macro_f1"]):
        reasons.append("macro-F1 未超过 WEIGHTED_v2")
    if int(summary["cal_field_as_road"]) >= int(weighted_ref["cal_field_as_road"]):
        reasons.append("field_as_road 未继续低于 WEIGHTED_v2")
    if int(summary["cal_long_field_as_road_points"]) >= int(weighted_ref["cal_long_field_as_road_points"]):
        reasons.append("long_field_as_road_points 未继续低于 WEIGHTED_v2")
    if max_fixed <= 0:
        reasons.append("最大长段完全未修动")
    if not road_ok:
        reasons.append("road-F1 低于 0.815")
    if not pred_rate_ok:
        reasons.append("pred_road_rate 低于 0.20")
    if not fixed_ok:
        reasons.append("introduced_road_as_field 不小于 fixed_field_as_road")
    if new_long_road:
        reasons.append("引入新的 long_road_as_field")
    return "；".join(reasons) if reasons else "未触发失败标准"


def make_pack():
    paths = [
        "models/CandidateRunFieldGuardCalibration.py",
        "experiments/train_candcal_run_guard_lenaware_v3.py",
        "scripts/run_candcal_run_guard_lenaware_v3.py",
        "models/Encoder.py",
        str(RESULT_PATH),
        str(REPORT_PATH),
        str(RUN_DIR / "config_resolved.json"),
        str(RUN_DIR / "best_lenaware_threshold_config.json"),
        str(RUN_DIR / "run_guard_head_best.pt"),
        str(LOG_PATH),
    ]
    paths.extend(str(path) for path in sorted(DIAG_DIR.glob("*.csv")))
    PACK_PATH.parent.mkdir(parents=True, exist_ok=True)
    if PACK_PATH.exists():
        PACK_PATH.unlink()
    with zipfile.ZipFile(PACK_PATH, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for item in paths:
            path = Path(item)
            if path.exists():
                zf.write(path, path.as_posix())


def clean_old_outputs():
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    DIAG_DIR.mkdir(parents=True, exist_ok=True)
    for path in DIAG_DIR.glob("*.csv"):
        path.unlink()
    for path in (RESULT_PATH, REPORT_PATH, PACK_PATH):
        if path.exists():
            path.unlink()


def main():
    args = parse_args()
    start = time.time()
    apply_common_config()
    clean_old_outputs()
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    PACK_PATH.parent.mkdir(parents=True, exist_ok=True)

    device = get_default_device()
    rc_checkpoint = base.find_rc_checkpoint()
    print(f"device={device}", flush=True)
    print(f"rc_checkpoint={rc_checkpoint}", flush=True)
    print("frozen_rc_mainline=True", flush=True)
    model = base.build_frozen_model(device, rc_checkpoint)
    loaders = base.make_loaders()
    graph_caches = base.make_graph_caches()
    point_rows = base.infer_point_rows(model, loaders, graph_caches, device)
    point_rows_by_split = {split: split_rows(point_rows, split) for split in ("train", "valid", "test")}
    run_rows = base.build_candidate_runs(point_rows, int(CANDCAL_RUN_GUARD_DEFAULTS["min_run_len"]))
    print(
        f"candidate_runs_built total={len(run_rows)} train={len(split_runs(run_rows, 'train'))} "
        f"valid={len(split_runs(run_rows, 'valid'))} test={len(split_runs(run_rows, 'test'))}",
        flush=True,
    )

    head, feature_mean, feature_std, best_valid, threshold_rows, training_rows, positives, negatives = train_run_guard_head(
        run_rows,
        point_rows_by_split,
        device,
        args.epochs,
    )
    best_epoch = int(best_valid["epoch"])
    best_config = {
        "long_threshold": float(best_valid["long_threshold"]),
        "mid_threshold": float(best_valid["mid_threshold"]),
        "short_threshold": float(best_valid["short_threshold"]),
    }

    all_scores = {}
    for split in ("train", "valid", "test"):
        all_scores.update(base.score_runs(head, split_runs(run_rows, split), feature_mean, feature_std, device))
    update_run_scores(run_rows, all_scores, best_config)

    test_rows = point_rows_by_split["test"]
    test_runs = split_runs(run_rows, "test")
    test_eval = evaluate_lenaware_threshold("test", test_rows, test_runs, all_scores, best_config, best_epoch)
    point_test_rows = annotate_test_rows(test_rows, test_runs, all_scores, best_config)
    final_preds = np.asarray([int(row["final_pred"]) for row in point_test_rows], dtype=np.int64)
    for row, final_pred in zip(test_rows, final_preds):
        row["final_pred"] = int(final_pred)
    base_preds = np.asarray([int(row["base_pred"]) for row in test_rows], dtype=np.int64)
    base_long = v2.long_error_segments_from_preds(test_rows, base_preds, "test", "base", RUN_NAME)
    cal_long = v2.long_error_segments_from_preds(test_rows, final_preds, "test", "calibrated", RUN_NAME)
    top_changes = top_long_field_as_road_changes(test_rows, test_runs)
    summary = make_summary(best_epoch, best_config, test_eval, top_changes)

    config = dict(
        group=RUN_NAME,
        common_config=COMMON_CONFIG,
        run_guard_defaults=CANDCAL_RUN_GUARD_DEFAULTS,
        lenaware_threshold_candidates=LENAWARE_THRESHOLDS,
        adaptive_delta_defaults=v2.ADAPTIVE_DELTA_DEFAULTS,
        weighted_loss=True,
        adaptive_delta=True,
        rc_checkpoint=str(rc_checkpoint),
        frozen_rc_mainline=True,
        feature_names=base.FEATURE_NAMES,
        feature_mean=feature_mean.tolist(),
        feature_std=feature_std.tolist(),
        train_positive_runs=positives,
        train_negative_runs=negatives,
        epochs=int(args.epochs),
    )
    write_json(RUN_DIR / "config_resolved.json", config)
    write_json(
        RUN_DIR / "best_lenaware_threshold_config.json",
        dict(
            best_epoch=best_epoch,
            best_threshold_config=best_config,
            best_valid_metrics=best_valid,
            valid_threshold_selection="safe_filter_then_macro_f1_then_long_field_points",
        ),
    )
    torch.save(
        dict(
            model_state_dict=head.state_dict(),
            input_dim=len(base.FEATURE_NAMES),
            feature_names=base.FEATURE_NAMES,
            feature_mean=feature_mean,
            feature_std=feature_std,
            config=CANDCAL_RUN_GUARD_DEFAULTS,
            adaptive_delta_defaults=v2.ADAPTIVE_DELTA_DEFAULTS,
            lenaware_threshold_config=best_config,
            weighted_loss=True,
            adaptive_delta=True,
            best_epoch=best_epoch,
            rc_checkpoint=str(rc_checkpoint),
        ),
        RUN_DIR / "run_guard_head_best.pt",
    )

    for split in ("train", "valid", "test"):
        write_csv(DIAG_DIR / f"{split}_candidate_runs.csv", RUN_CSV_FIELDS, split_runs(run_rows, split))
    write_csv(DIAG_DIR / "lenaware_threshold_sweep_valid.csv", THRESHOLD_FIELDS, threshold_rows)
    write_csv(DIAG_DIR / "training_metrics.csv", TRAINING_FIELDS, training_rows)
    write_csv(DIAG_DIR / "point_level_calibration_test.csv", POINT_TEST_FIELDS, point_test_rows)
    write_csv(DIAG_DIR / "long_error_before_after.csv", v2.LONG_FIELDS, base_long + cal_long)
    write_csv(DIAG_DIR / "top_long_field_as_road_changes.csv", TOP_LONG_FIELDS, top_changes)
    write_csv(RESULT_PATH, SUMMARY_FIELDS, [summary])
    REPORT_PATH.write_text(build_report(summary, best_valid, top_changes), encoding="utf-8")
    make_pack()

    print("all_groups_completed=True", flush=True)
    print(
        f"summary group={RUN_NAME} best_epoch={summary['best_epoch']} "
        f"thresholds=({summary['best_long_threshold']},{summary['best_mid_threshold']},{summary['best_short_threshold']}) "
        f"macro={summary['cal_test_macro_f1']} road_f1={summary['cal_test_road_f1']} "
        f"field_as_road={summary['cal_field_as_road']} long_points={summary['cal_long_field_as_road_points']} "
        f"max_selected={summary['max_field_as_road_segment_selected']} "
        f"max_score={summary['max_field_as_road_segment_score']} "
        f"max_threshold={summary['max_field_as_road_segment_threshold']} "
        f"max_fixed={summary['max_field_as_road_segment_fixed_points']}",
        flush=True,
    )
    print(f"pack_path={PACK_PATH}", flush=True)
    print(f"elapsed_sec={time.time() - start:.2f}", flush=True)


if __name__ == "__main__":
    main()
