import argparse
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

from experiments import train_candcal_run_guard as base
from models.CandidateRunFieldGuardCalibration import (
    CANDCAL_RUN_GUARD_DEFAULTS,
    CandidateRunFieldGuardCalibration,
    compute_pos_weight,
)
from utils.threading_config import apply_torch_thread_config, configure_default_threads
from utils.utils import get_default_device


configure_default_threads()
apply_torch_thread_config(torch)


RUN_FAMILY = "PT2G_MSC_RC_CANDCAL_RUN_GUARD_v2_3runs"
DIAG_DIR = Path("diagnostics") / RUN_FAMILY
RESULT_PATH = Path("results") / f"{RUN_FAMILY}_summary.csv"
REPORT_PATH = Path("analysis") / f"{RUN_FAMILY}_report.md"
PACK_PATH = Path("analysis_packs") / f"{RUN_FAMILY}_for_chatgpt.zip"
LOG_PATH = Path("logs/candcal_run_guard_v2/CANDCAL_RUN_GUARD_v2_3runs.log")

EXPERIMENTS = {
    "WEIGHTED": {
        "group": "PT2G_MSC_RC_CANDCAL_RUN_GUARD_WEIGHTED_v2",
        "weighted_loss": True,
        "adaptive_delta": False,
        "seed": 43,
    },
    "ADAPTDELTA": {
        "group": "PT2G_MSC_RC_CANDCAL_RUN_GUARD_ADAPTDELTA_v2",
        "weighted_loss": False,
        "adaptive_delta": True,
        "seed": 44,
    },
    "WEIGHTED_ADAPTDELTA": {
        "group": "PT2G_MSC_RC_CANDCAL_RUN_GUARD_WEIGHTED_ADAPTDELTA_v2",
        "weighted_loss": True,
        "adaptive_delta": True,
        "seed": 45,
    },
}

ADAPTIVE_DELTA_DEFAULTS = {
    "margin_buffer": 0.05,
    "min_delta": 0.20,
    "max_delta": 1.20,
}

POINT_GUARD_DELTA_V2 = {
    "test_macro_f1": 0.887351,
    "field_as_road": 3431,
    "long_field_as_road_segments": 33,
}

THRESHOLD_FIELDS = [
    "group",
    "epoch",
    "threshold",
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
    "max_field_as_road_segment_fixed_points",
    "max_field_as_road_segment_remaining_points",
]

POINT_TEST_FIELDS = list(base.POINT_TEST_FIELDS) + ["delta_applied"]
TOP_LONG_FIELDS = [
    "group",
    "trace_id",
    "start_index",
    "end_index",
    "length",
    "candidate_run_id",
    "run_length",
    "run_guard_score",
    "selected_by_guard",
    "base_prob_road_mean",
    "base_margin_mean",
    "fixed_points_in_segment",
    "remaining_error_points",
    "coverage_rate",
]
LONG_FIELDS = ["group"] + list(base.LONG_FIELDS)
TRAINING_FIELDS = [
    "group",
    "epoch",
    "train_loss",
    "valid_best_threshold",
    "valid_best_macro_f1",
    "valid_best_fixed",
    "valid_best_introduced",
    "valid_best_long_field_points",
    "selection_status",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Run three CANDCAL Run Guard v2 experiments.")
    parser.add_argument("--exp", choices=["ALL", "WEIGHTED", "ADAPTDELTA", "WEIGHTED_ADAPTDELTA"], default="ALL")
    return parser.parse_args()


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


def sample_weights_for_runs(runs, targets, weighted_loss):
    weights = []
    for run, target in zip(runs, targets):
        if weighted_loss:
            length_weight = math.sqrt(float(run["length"])) / math.sqrt(20.0)
            length_weight = max(1.0, min(4.0, length_weight))
            if int(target) == 1:
                length_weight *= 1.5
            weights.append(length_weight)
        else:
            weights.append(1.0)
    return np.asarray(weights, dtype=np.float32)


def train_run_guard_head(exp_cfg, runs, point_rows_by_split, device):
    set_seed(int(exp_cfg["seed"]))
    train_runs = [run for run in split_runs(runs, "train") if int(run["run_target"]) in (0, 1)]
    if not train_runs:
        raise SystemExit("RUN_GUARD_NO_TRAIN_TARGETS")
    targets = np.asarray([int(run["run_target"]) for run in train_runs], dtype=np.float32)
    pos_weight, positives, negatives = compute_pos_weight(
        torch.from_numpy(targets),
        CANDCAL_RUN_GUARD_DEFAULTS["pos_weight_clip"],
    )
    if pos_weight is None:
        print(f"RUN_GUARD_TARGET_CLASS_WARNING group={exp_cfg['group']} positives={positives} negatives={negatives}", flush=True)
        pos_weight = torch.tensor(1.0, dtype=torch.float32)

    feature_mean, feature_std = base.fit_feature_stats(train_runs)
    head = CandidateRunFieldGuardCalibration(input_dim=len(base.FEATURE_NAMES)).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device), reduction="none")
    x_train = torch.from_numpy(base.feature_matrix(train_runs, feature_mean, feature_std)).to(device)
    y_train = torch.from_numpy(targets).to(device)
    sample_weights = torch.from_numpy(sample_weights_for_runs(train_runs, targets, exp_cfg["weighted_loss"])).to(device)

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
            raw_loss = criterion(logits, y_train[idx])
            loss = (raw_loss * sample_weights[idx]).mean()
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.detach().cpu().item())
            batch_count += 1

        valid_scores = base.score_runs(head, split_runs(runs, "valid"), feature_mean, feature_std, device)
        epoch_sweep = evaluate_threshold_sweep(
            exp_cfg,
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
                group=exp_cfg["group"],
                epoch=epoch,
                train_loss=base.safe_div(epoch_loss, batch_count),
                valid_best_threshold=selected["threshold"],
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
            f"group={exp_cfg['group']} epoch={epoch} train_loss={base.safe_div(epoch_loss, batch_count):.6f} "
            f"valid_macro={selected['cal_macro_f1']:.6f} threshold={selected['threshold']} "
            f"fixed={selected['fixed_field_as_road']} introduced={selected['introduced_road_as_field']} "
            f"long_field_points={selected['cal_long_field_as_road_points']} status={selected['selection_status']}",
            flush=True,
        )
    if best_state is not None:
        head.load_state_dict(best_state)
    return head, feature_mean, feature_std, best, threshold_rows, training_rows, positives, negatives


def evaluate_threshold_sweep(exp_cfg, split, rows, runs, score_map, epoch):
    rows_out = [
        evaluate_threshold(exp_cfg, split, rows, runs, score_map, threshold, epoch)
        for threshold in CANDCAL_RUN_GUARD_DEFAULTS["threshold_candidates"]
    ]
    selected = choose_best_threshold_row(rows_out)
    for row in rows_out:
        row["selection_status"] = "selected_" + selected["selection_status"] if row is selected else row["selection_status"]
    return rows_out


def evaluate_threshold(exp_cfg, split, rows, runs, score_map, threshold, epoch=None):
    selected_run_ids = {run["candidate_run_id"] for run in runs if float(score_map.get(run["candidate_run_id"], 0.0)) >= float(threshold)}
    labels, base_preds, final_preds, behavior = calibrated_predictions(rows, selected_run_ids, score_map, exp_cfg["adaptive_delta"])
    base_metrics = base.class_metrics(labels, base_preds)
    cal_metrics = base.class_metrics(labels, final_preds)
    base_long = long_error_segments_from_preds(rows, base_preds, split, "base", exp_cfg["group"])
    cal_long = long_error_segments_from_preds(rows, final_preds, split, "calibrated", exp_cfg["group"])
    base_road_long = summarize_long_segments(base_long, "road_as_field")
    base_field_long = summarize_long_segments(base_long, "field_as_road")
    cal_road_long = summarize_long_segments(cal_long, "road_as_field")
    cal_field_long = summarize_long_segments(cal_long, "field_as_road")
    candidate_points = sum(int(run["length"]) for run in runs)
    selected_runs = [run for run in runs if run["candidate_run_id"] in selected_run_ids]
    selected_points = sum(int(run["length"]) for run in selected_runs)
    selected_false = sum(1 for run in selected_runs if float(run["field_ratio"]) >= CANDCAL_RUN_GUARD_DEFAULTS["positive_field_ratio"])
    row = dict(
        group=exp_cfg["group"],
        epoch=epoch if epoch is not None else "",
        threshold=float(threshold),
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
        guard_applied_point_rate=base.safe_div(selected_points, len(rows)),
        fixed_field_as_road=behavior["fixed_field_as_road"],
        introduced_road_as_field=behavior["introduced_road_as_field"],
        fixed_minus_introduced=behavior["fixed_field_as_road"] - behavior["introduced_road_as_field"],
        selected_false_run_rate=base.safe_div(selected_false, len(selected_runs)),
    )
    row["safe_threshold"] = is_safe_threshold(row)
    return row


def calibrated_predictions(rows, selected_run_ids, score_map, adaptive_delta):
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
            delta = calibration_delta(row, float(score_map.get(run_id, 0.0)), adaptive_delta)
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


def calibration_delta(row, run_guard_score, adaptive_delta):
    if not adaptive_delta:
        return float(CANDCAL_RUN_GUARD_DEFAULTS["max_delta"]) * run_guard_score
    gap = float(row["base_road_logit"]) - float(row["base_field_logit"])
    required_delta = 0.5 * gap + ADAPTIVE_DELTA_DEFAULTS["margin_buffer"]
    required_delta = max(ADAPTIVE_DELTA_DEFAULTS["min_delta"], min(ADAPTIVE_DELTA_DEFAULTS["max_delta"], required_delta))
    score_factor = 0.80 + 0.20 * run_guard_score
    return required_delta * score_factor


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
    return float(candidate["threshold"]) > float(current["threshold"])


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


def long_error_segments_from_preds(rows, preds, split, stage, group):
    grouped = defaultdict(list)
    for row, pred in zip(rows, preds):
        etype = base.error_type(row["label"], int(pred))
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
                add_long_segment(segments, group, split, stage, trace_id, etype, current)
                current = []
            current.append(row)
            prev = idx
        if current:
            add_long_segment(segments, group, split, stage, trace_id, etype, current)
    return segments


def add_long_segment(out, group, split, stage, trace_id, etype, rows):
    if len(rows) < 20:
        return
    out.append(
        dict(
            group=group,
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


def update_run_scores(runs, score_map, threshold):
    for run in runs:
        score = float(score_map.get(run["candidate_run_id"], 0.0))
        run["run_guard_score"] = score
        run["selected_by_best_threshold"] = score >= threshold


def annotate_test_rows(exp_cfg, rows, runs, score_map, threshold):
    run_lookup = {run["candidate_run_id"]: run for run in runs}
    selected_ids = {run_id for run_id, score in score_map.items() if float(score) >= threshold}
    out = []
    for row in rows:
        run_id = row.get("candidate_run_id", "")
        selected = run_id in selected_ids
        score = float(score_map.get(run_id, 0.0)) if run_id else 0.0
        road_logit = float(row["base_road_logit"])
        field_logit = float(row["base_field_logit"])
        delta = 0.0
        if selected:
            delta = calibration_delta(row, score, exp_cfg["adaptive_delta"])
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
            run_guard_score=score if run_id in run_lookup else "",
            selected_by_guard=selected,
            fixed_field_as_road=int(row["base_pred"] == 0 and row["label"] == 1 and final_pred == 1),
            introduced_road_as_field=int(row["base_pred"] == 0 and row["label"] == 0 and final_pred == 1),
            error_type_base=row["error_type_base"],
            error_type_final=base.error_type(row["label"], final_pred),
            lon=row["lon"],
            lat=row["lat"],
            delta_applied=delta,
        )
        for name in base.AUX_FEATURE_NAMES:
            out_row[name] = row[name]
        out.append(out_row)
    return out


def top_long_field_as_road_changes(exp_cfg, test_rows, test_runs, score_map):
    run_lookup = {run["candidate_run_id"]: run for run in test_runs}
    rows_by_trace_index = {(row["trace_id"], int(row["global_index"])): row for row in test_rows}
    base_preds = np.asarray([int(row["base_pred"]) for row in test_rows], dtype=np.int64)
    base_segments = long_error_segments_from_preds(test_rows, base_preds, "test", "base", exp_cfg["group"])
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
        fixed = sum(1 for row in segment_rows if int(row["label"]) == 1 and int(row["base_pred"]) == 0 and int(row.get("_final_pred", row["base_pred"])) == 1)
        remaining = sum(1 for row in segment_rows if int(row["label"]) == 1 and int(row.get("_final_pred", row["base_pred"])) == 0)
        candidate_points = sum(1 for row in segment_rows if row.get("candidate_run_id", ""))
        out.append(
            dict(
                group=exp_cfg["group"],
                trace_id=segment["trace_id"],
                start_index=segment["start_index"],
                end_index=segment["end_index"],
                length=segment["length"],
                candidate_run_id=best_run_id,
                run_length=best_run.get("length", ""),
                run_guard_score=best_run.get("run_guard_score", ""),
                selected_by_guard=best_run.get("selected_by_best_threshold", False),
                base_prob_road_mean=base.mean([row["base_prob_road"] for row in segment_rows]),
                base_margin_mean=base.mean([row["margin"] for row in segment_rows]),
                fixed_points_in_segment=fixed,
                remaining_error_points=remaining,
                coverage_rate=base.safe_div(candidate_points, segment["length"]),
            )
        )
    return out


def make_summary(exp_cfg, best_epoch, best_threshold, test_eval, top_changes):
    max_fixed = int(top_changes[0]["fixed_points_in_segment"]) if top_changes else 0
    max_remaining = int(top_changes[0]["remaining_error_points"]) if top_changes else 0
    return dict(
        group=exp_cfg["group"],
        best_epoch=best_epoch,
        best_threshold=best_threshold,
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
        guard_applied_point_rate=test_eval["guard_applied_point_rate"],
        fixed_field_as_road=test_eval["fixed_field_as_road"],
        introduced_road_as_field=test_eval["introduced_road_as_field"],
        fixed_minus_introduced=test_eval["fixed_minus_introduced"],
        max_field_as_road_segment_fixed_points=max_fixed,
        max_field_as_road_segment_remaining_points=max_remaining,
    )


def run_one_experiment(exp_key, exp_cfg, point_rows_by_split, run_rows, device, rc_checkpoint):
    run_dir = Path("runs") / exp_cfg["group"]
    run_dir.mkdir(parents=True, exist_ok=True)
    head, feature_mean, feature_std, best_valid, threshold_rows, training_rows, positives, negatives = train_run_guard_head(
        exp_cfg,
        run_rows,
        point_rows_by_split,
        device,
    )
    best_epoch = int(best_valid["epoch"])
    best_threshold = float(best_valid["threshold"])
    all_scores = {}
    for split in ("train", "valid", "test"):
        all_scores.update(base.score_runs(head, split_runs(run_rows, split), feature_mean, feature_std, device))
    update_run_scores(run_rows, all_scores, best_threshold)

    test_rows = point_rows_by_split["test"]
    test_runs = split_runs(run_rows, "test")
    test_eval = evaluate_threshold(exp_cfg, "test", test_rows, test_runs, all_scores, best_threshold, best_epoch)
    point_test_rows = annotate_test_rows(exp_cfg, test_rows, test_runs, all_scores, best_threshold)
    final_preds = np.asarray([int(row["final_pred"]) for row in point_test_rows], dtype=np.int64)
    for row, final_pred in zip(test_rows, final_preds):
        row["final_pred"] = int(final_pred)

    base_preds = np.asarray([int(row["base_pred"]) for row in test_rows], dtype=np.int64)
    base_long = long_error_segments_from_preds(test_rows, base_preds, "test", "base", exp_cfg["group"])
    cal_long = long_error_segments_from_preds(test_rows, final_preds, "test", "calibrated", exp_cfg["group"])
    top_changes = top_long_field_as_road_changes(exp_cfg, test_rows, test_runs, all_scores)
    summary = make_summary(exp_cfg, best_epoch, best_threshold, test_eval, top_changes)

    config = dict(
        group=exp_cfg["group"],
        experiment_key=exp_key,
        common_config=base.COMMON_CONFIG,
        run_guard_defaults=CANDCAL_RUN_GUARD_DEFAULTS,
        adaptive_delta_defaults=ADAPTIVE_DELTA_DEFAULTS,
        weighted_loss=exp_cfg["weighted_loss"],
        adaptive_delta=exp_cfg["adaptive_delta"],
        rc_checkpoint=str(rc_checkpoint),
        frozen_rc_mainline=True,
        feature_names=base.FEATURE_NAMES,
        feature_mean=feature_mean.tolist(),
        feature_std=feature_std.tolist(),
        train_positive_runs=positives,
        train_negative_runs=negatives,
    )
    write_json(run_dir / "config_resolved.json", config)
    write_json(
        run_dir / "best_run_guard_threshold.json",
        dict(
            best_epoch=best_epoch,
            best_threshold=best_threshold,
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
            adaptive_delta_defaults=ADAPTIVE_DELTA_DEFAULTS,
            weighted_loss=exp_cfg["weighted_loss"],
            adaptive_delta=exp_cfg["adaptive_delta"],
            best_epoch=best_epoch,
            best_threshold=best_threshold,
            rc_checkpoint=str(rc_checkpoint),
        ),
        run_dir / "run_guard_head_best.pt",
    )

    prefix = exp_cfg["group"]
    for split in ("train", "valid", "test"):
        write_csv(DIAG_DIR / f"{prefix}_{split}_candidate_runs.csv", base.RUN_CSV_FIELDS, split_runs(run_rows, split))
    write_csv(DIAG_DIR / f"{prefix}_threshold_sweep_valid.csv", THRESHOLD_FIELDS, threshold_rows)
    write_csv(DIAG_DIR / f"{prefix}_training_metrics.csv", TRAINING_FIELDS, training_rows)
    write_csv(DIAG_DIR / f"{prefix}_point_level_calibration_test.csv", POINT_TEST_FIELDS, point_test_rows)
    write_csv(DIAG_DIR / f"{prefix}_long_error_before_after.csv", LONG_FIELDS, base_long + cal_long)
    write_csv(DIAG_DIR / f"{prefix}_top_long_field_as_road_changes.csv", TOP_LONG_FIELDS, top_changes)

    print(
        f"group_done={exp_cfg['group']} best_epoch={best_epoch} best_threshold={best_threshold} "
        f"base_macro={summary['base_test_macro_f1']} cal_macro={summary['cal_test_macro_f1']} "
        f"delta_macro={summary['delta_macro_f1']} field_as_road={summary['base_field_as_road']}->{summary['cal_field_as_road']} "
        f"long_points={summary['base_long_field_as_road_points']}->{summary['cal_long_field_as_road_points']} "
        f"max_fixed={summary['max_field_as_road_segment_fixed_points']}",
        flush=True,
    )
    return dict(
        exp_key=exp_key,
        exp_cfg=exp_cfg,
        summary=summary,
        best_valid=best_valid,
        top_changes=top_changes,
        threshold_rows=threshold_rows,
    )


def build_report(results):
    summaries = [item["summary"] for item in results]
    if not summaries:
        return "# PT2G_MSC_RC_CANDCAL_RUN_GUARD_v2_3runs Report\n\nNo experiment completed.\n"
    best_macro = max(summaries, key=lambda row: float(row["cal_test_macro_f1"]))
    best_field = min(summaries, key=lambda row: int(row["cal_field_as_road"]))
    best_long_points = min(summaries, key=lambda row: int(row["cal_long_field_as_road_points"]))
    max_fixed_best = max(summaries, key=lambda row: int(row["max_field_as_road_segment_fixed_points"]))
    weighted = next((row for row in summaries if row["group"].endswith("WEIGHTED_v2")), None)
    adapt = next((row for row in summaries if row["group"].endswith("ADAPTDELTA_v2")), None)
    both = next((row for row in summaries if row["group"].endswith("WEIGHTED_ADAPTDELTA_v2")), None)
    recommended = recommend_mainline(summaries)
    lines = [
        "# PT2G_MSC_RC_CANDCAL_RUN_GUARD_v2_3runs Report",
        "",
        "## Summary",
        "",
        "| group | best_epoch | threshold | macro_f1 | road_f1 | field_as_road | long_field_points | max_fixed | fixed | introduced |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        lines.append(
            f"| {row['group']} | {row['best_epoch']} | {row['best_threshold']} | {row['cal_test_macro_f1']} | "
            f"{row['cal_test_road_f1']} | {row['cal_field_as_road']} | {row['cal_long_field_as_road_points']} | "
            f"{row['max_field_as_road_segment_fixed_points']} | {row['fixed_field_as_road']} | {row['introduced_road_as_field']} |"
        )
    lines.extend(
        [
            "",
            "## Required Answers",
            "",
            f"1. 三组中 test macro-F1 最好：{best_macro['group']}，cal_test_macro_f1={best_macro['cal_test_macro_f1']}.",
            f"2. field_as_road 降得最多：{best_field['group']}，{best_field['base_field_as_road']} -> {best_field['cal_field_as_road']}，delta={best_field['delta_field_as_road']}.",
            f"3. long_field_as_road_points 降得最多：{best_long_points['group']}，{best_long_points['base_long_field_as_road_points']} -> {best_long_points['cal_long_field_as_road_points']}，delta={best_long_points['delta_long_field_as_road_points']}.",
            f"4. 修动最大 field_as_road 长段最多：{max_fixed_best['group']}，fixed_points={max_fixed_best['max_field_as_road_segment_fixed_points']}.",
            f"5. weighted loss 是否让模型更关注大长段：{weighted_effect_answer(weighted, adapt, both)}.",
            f"6. adaptive delta 是否能翻高置信 road logits：{adaptive_effect_answer(adapt, both)}.",
            f"7. weighted + adaptive delta 是否优于单独使用：{both_effect_answer(weighted, adapt, both)}.",
            f"8. road-F1 是否被破坏：{road_f1_answer(summaries)}.",
            f"9. pred_road_rate 是否低于安全线：{pred_road_answer(summaries)}.",
            f"10. fixed_field_as_road 是否明显大于 introduced_road_as_field：{fixed_intro_answer(summaries)}.",
            f"11. 是否建议把某组作为新主线：{recommended['recommend']}，推荐={recommended['group']}，原因={recommended['reason']}.",
            f"12. 是否还需要继续做 road_as_field repair：{road_repair_answer(summaries)}.",
            f"13. 如果失败，失败原因：{failure_answer(summaries)}.",
            f"14. 打包路径：{PACK_PATH}.",
            "",
            "## Max Segment Details",
            "",
            "| group | top_trace | length | selected | score | fixed | remaining |",
            "|---|---|---:|---|---:|---:|---:|",
        ]
    )
    for item in results:
        top = item["top_changes"][0] if item["top_changes"] else {}
        lines.append(
            f"| {item['summary']['group']} | {top.get('trace_id', '')} | {top.get('length', '')} | "
            f"{top.get('selected_by_guard', '')} | {top.get('run_guard_score', '')} | "
            f"{top.get('fixed_points_in_segment', '')} | {top.get('remaining_error_points', '')} |"
        )
    return "\n".join(lines) + "\n"


def recommend_mainline(summaries):
    eligible = [
        row for row in summaries
        if float(row["cal_test_macro_f1"]) > 0.887351
        and int(row["cal_field_as_road"]) < 3431
        and int(row["cal_long_field_as_road_segments"]) < 33
        and float(row["cal_test_road_f1"]) >= 0.815
        and float(row["cal_pred_road_rate"]) >= 0.20
        and int(row["fixed_field_as_road"]) > int(row["introduced_road_as_field"])
    ]
    if eligible:
        best = max(eligible, key=lambda row: (float(row["cal_test_macro_f1"]), -int(row["cal_long_field_as_road_points"])))
        return dict(recommend=True, group=best["group"], reason="meets v2 success criteria")
    valuable = [
        row for row in summaries
        if float(row["cal_test_macro_f1"]) >= float(row["base_test_macro_f1"])
        and int(row["delta_long_field_as_road_points"]) < 0
        and float(row["cal_test_road_f1"]) >= 0.815
    ]
    if valuable:
        best = min(valuable, key=lambda row: int(row["cal_long_field_as_road_points"]))
        return dict(recommend=False, group=best["group"], reason="not enough for new mainline, but may be useful for long-error analysis")
    best = max(summaries, key=lambda row: float(row["cal_test_macro_f1"]))
    return dict(recommend=False, group=best["group"], reason="does not meet success criteria")


def weighted_effect_answer(weighted, adapt, both):
    if weighted is None:
        return "not available"
    return str(int(weighted["delta_long_field_as_road_points"]) < 0 or int(weighted["max_field_as_road_segment_fixed_points"]) > 0)


def adaptive_effect_answer(adapt, both):
    rows = [row for row in (adapt, both) if row is not None]
    return str(any(int(row["max_field_as_road_segment_fixed_points"]) > 0 or int(row["delta_field_as_road"]) < -100 for row in rows))


def both_effect_answer(weighted, adapt, both):
    if both is None:
        return "not available"
    others = [row for row in (weighted, adapt) if row is not None]
    if not others:
        return "not available"
    return str(float(both["cal_test_macro_f1"]) >= max(float(row["cal_test_macro_f1"]) for row in others))


def road_f1_answer(summaries):
    damaged = [row["group"] for row in summaries if float(row["cal_test_road_f1"]) < 0.815]
    return "yes: " + ", ".join(damaged) if damaged else "no"


def pred_road_answer(summaries):
    low = [row["group"] for row in summaries if float(row["cal_pred_road_rate"]) < 0.20]
    return "yes: " + ", ".join(low) if low else "no"


def fixed_intro_answer(summaries):
    bad = [row["group"] for row in summaries if int(row["fixed_field_as_road"]) <= int(row["introduced_road_as_field"])]
    return "no for " + ", ".join(bad) if bad else "yes for all completed runs"


def road_repair_answer(summaries):
    return str(any(int(row["cal_road_as_field"]) > int(row["base_road_as_field"]) for row in summaries))


def failure_answer(summaries):
    failures = []
    for row in summaries:
        reasons = []
        if float(row["cal_test_macro_f1"]) < float(row["base_test_macro_f1"]):
            reasons.append("macro below RC")
        if int(row["introduced_road_as_field"]) >= int(row["fixed_field_as_road"]):
            reasons.append("introduced >= fixed")
        if int(row["delta_long_field_as_road_points"]) >= 0:
            reasons.append("long field points not reduced")
        if int(row["max_field_as_road_segment_fixed_points"]) == 0:
            reasons.append("max segment unchanged")
        if reasons:
            failures.append(f"{row['group']}: {'; '.join(reasons)}")
    return " | ".join(failures) if failures else "none"


def make_pack(completed_groups):
    paths = [
        "models/CandidateRunFieldGuardCalibration.py",
        "experiments/train_candcal_run_guard_v2_3runs.py",
        "scripts/run_candcal_run_guard_v2_3runs.py",
        "models/Encoder.py",
        str(RESULT_PATH),
        str(REPORT_PATH),
        str(LOG_PATH),
    ]
    paths.extend(str(path) for path in DIAG_DIR.glob("*.csv"))
    for group in completed_groups:
        paths.append(str(Path("runs") / group / "config_resolved.json"))
    PACK_PATH.parent.mkdir(parents=True, exist_ok=True)
    if PACK_PATH.exists():
        PACK_PATH.unlink()
    with zipfile.ZipFile(PACK_PATH, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for item in paths:
            path = Path(item)
            if path.exists():
                zf.write(path, path.as_posix())


def clean_old_outputs():
    DIAG_DIR.mkdir(parents=True, exist_ok=True)
    for path in DIAG_DIR.glob("*.csv"):
        path.unlink()
    for path in (RESULT_PATH, REPORT_PATH, PACK_PATH):
        if path.exists():
            path.unlink()


def main():
    args = parse_args()
    start = time.time()
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    PACK_PATH.parent.mkdir(parents=True, exist_ok=True)
    clean_old_outputs()

    selected_keys = list(EXPERIMENTS) if args.exp == "ALL" else [args.exp]
    device = get_default_device()
    rc_checkpoint = base.find_rc_checkpoint()
    print(f"device={device}", flush=True)
    print(f"rc_checkpoint={rc_checkpoint}", flush=True)
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

    results = []
    for exp_key in selected_keys:
        exp_cfg = dict(EXPERIMENTS[exp_key])
        print(
            f"start_group={exp_cfg['group']} weighted_loss={exp_cfg['weighted_loss']} adaptive_delta={exp_cfg['adaptive_delta']}",
            flush=True,
        )
        result = run_one_experiment(exp_key, exp_cfg, point_rows_by_split, run_rows, device, rc_checkpoint)
        results.append(result)

    summary_rows = [item["summary"] for item in results]
    write_csv(RESULT_PATH, SUMMARY_FIELDS, summary_rows)
    REPORT_PATH.write_text(build_report(results), encoding="utf-8")
    make_pack([item["summary"]["group"] for item in results])

    best = max(summary_rows, key=lambda row: float(row["cal_test_macro_f1"])) if summary_rows else None
    print(f"all_groups_completed={len(results) == len(selected_keys)}", flush=True)
    for row in summary_rows:
        print(
            f"summary group={row['group']} best_epoch={row['best_epoch']} threshold={row['best_threshold']} "
            f"macro={row['cal_test_macro_f1']} road_f1={row['cal_test_road_f1']} "
            f"field_as_road={row['cal_field_as_road']} long_points={row['cal_long_field_as_road_points']} "
            f"max_fixed={row['max_field_as_road_segment_fixed_points']}",
            flush=True,
        )
    print(f"best_group={best['group'] if best else ''}", flush=True)
    print(f"pack_path={PACK_PATH}", flush=True)
    print(f"elapsed_sec={time.time() - start:.2f}", flush=True)


if __name__ == "__main__":
    main()
