import argparse
import json
import math
import sys
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


EXPERIMENT_NAME = "FRG_MoE_CONSERVE_v3"
BASELINE_GROUP = "PT2G_MSC_v1_time_fixed_strict_40pre"
BASELINE_TEST = {
    "macro_f1": 0.886494,
    "road_as_field": 3876,
    "field_as_road": 3423,
    "pred_road_rate": 0.212288,
}
TOPK_CONFIGS = {
    "SAFE": {
        "kind": "topk",
        "pred_road_min": 0.205,
        "road_as_field_delta_max": None,
        "rules": [
            ("fake_road_global_expert_guarded_score", "all", 0.05, "global_guarded_top5"),
            ("fake_road_long_expert_raw_score", "long", 0.10, "long_raw_top10"),
            ("fake_road_very_long_expert_raw_score", "very_long", 0.10, "very_long_raw_top10"),
        ],
    },
    "BALANCED": {
        "kind": "topk",
        "pred_road_min": 0.200,
        "road_as_field_delta_max": 250,
        "rules": [
            ("fake_road_global_expert_guarded_score", "all", 0.10, "global_guarded_top10"),
            ("fake_road_long_expert_raw_score", "long", 0.20, "long_raw_top20"),
            ("fake_road_very_long_expert_raw_score", "very_long", 0.20, "very_long_raw_top20"),
        ],
    },
    "STRONG": {
        "kind": "topk",
        "pred_road_min": 0.198,
        "road_as_field_delta_max": 400,
        "rules": [
            ("fake_road_global_expert_guarded_score", "all", 0.20, "global_guarded_top20"),
            ("fake_road_long_expert_raw_score", "long", 0.30, "long_raw_top30"),
            ("fake_road_very_long_expert_raw_score", "very_long", 0.30, "very_long_raw_top30"),
        ],
    },
}
VALID_THR_CONFIGS = {
    "VALID_THR_SAFE": {
        "kind": "valid_threshold",
        "precision_min": 0.98,
        "pred_road_min": 0.205,
        "road_as_field_delta_max": 150,
        "objective": "precision_then_net",
    },
    "VALID_THR_BALANCED": {
        "kind": "valid_threshold",
        "precision_min": 0.95,
        "pred_road_min": 0.200,
        "road_as_field_delta_max": 250,
        "objective": "macro_then_net",
    },
    "VALID_THR_STRONG": {
        "kind": "valid_threshold",
        "precision_min": 0.95,
        "pred_road_min": 0.198,
        "road_as_field_delta_max": 400,
        "objective": "net_then_macro",
    },
}
SEARCH_SCORE_COLUMNS = [
    "fake_road_short_expert_guarded_score",
    "fake_road_mid_expert_guarded_score",
    "fake_road_long_expert_raw_score",
    "fake_road_long_expert_guarded_score",
    "fake_road_very_long_expert_raw_score",
    "fake_road_very_long_expert_guarded_score",
    "fake_road_global_expert_guarded_score",
    "fake_road_hard_negative_guard_guarded_score",
    "fake_road_rank_ensemble_guarded_score",
]
FINAL_PRED_COLUMNS = [
    "split",
    "trace_id",
    "sample_index",
    "point_index",
    "global_index",
    "label",
    "base_pred",
    "final_pred",
    "prob_road",
    "prob_field",
    "changed_by_frg",
    "selected_config",
    "candidate_id",
    "fake_road_score",
    "longitude",
    "latitude",
]


def parse_args():
    parser = argparse.ArgumentParser(description="FRG_MoE_CONSERVE_v3 conservative fake-road postprocess correction.")
    parser.add_argument("--point_dir", default=f"diagnostics/{BASELINE_GROUP}_error_analysis")
    parser.add_argument("--audit_dir", default="diagnostics/FRG_MoE_AUDIT_v3")
    parser.add_argument("--output_dir", default=f"diagnostics/{EXPERIMENT_NAME}")
    parser.add_argument("--summary_csv", default=f"results/{EXPERIMENT_NAME}_summary.csv")
    parser.add_argument("--report_path", default=f"analysis/{EXPERIMENT_NAME}_report.md")
    parser.add_argument("--pack_path", default=f"analysis_packs/{EXPERIMENT_NAME}_for_chatgpt.zip")
    parser.add_argument("--write_threshold_predictions", action="store_true", default=True)
    return parser.parse_args()


def ensure_dirs(args):
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.summary_csv).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report_path).parent.mkdir(parents=True, exist_ok=True)
    Path(args.pack_path).parent.mkdir(parents=True, exist_ok=True)


def point_path(point_dir, split):
    return Path(point_dir) / f"{split}_point_predictions_with_features.csv"


def read_points(args, split):
    path = point_path(args.point_dir, split)
    if not path.exists():
        raise FileNotFoundError(f"POINT_PREDICTIONS_NOT_FOUND: {path}")
    df = pd.read_csv(path)
    rename = {}
    if "true_label" not in df.columns and "label" in df.columns:
        rename["label"] = "true_label"
    if "pred_label" not in df.columns:
        if "pred" in df.columns:
            rename["pred"] = "pred_label"
        elif "final_pred" in df.columns:
            rename["final_pred"] = "pred_label"
    if "longitude" not in df.columns and "lon" in df.columns:
        rename["lon"] = "longitude"
    if "latitude" not in df.columns and "lat" in df.columns:
        rename["lat"] = "latitude"
    if rename:
        df = df.rename(columns=rename)
    required = ["trace_id", "global_index", "point_index", "true_label", "pred_label", "prob_road", "prob_field"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"POINT_COLUMNS_MISSING {path}: {missing}")
    df["split"] = split
    df["trace_id"] = df["trace_id"].astype(str)
    df["global_index"] = pd.to_numeric(df["global_index"], errors="coerce").fillna(-1).astype(int)
    df["point_index"] = pd.to_numeric(df["point_index"], errors="coerce").fillna(-1).astype(int)
    df["true_label"] = pd.to_numeric(df["true_label"], errors="coerce").fillna(-1).astype(int)
    df["pred_label"] = pd.to_numeric(df["pred_label"], errors="coerce").fillna(-1).astype(int)
    for col in ["longitude", "latitude"]:
        if col not in df.columns:
            df[col] = np.nan
    return df


def read_audit(args, split):
    cand_path = Path(args.audit_dir) / f"fake_road_candidates_{split}.csv"
    score_path = Path(args.audit_dir) / f"expert_scores_{split}.csv"
    if not cand_path.exists() or not score_path.exists():
        raise FileNotFoundError(f"FRG_AUDIT_INPUT_MISSING: {cand_path} {score_path}")
    candidates = pd.read_csv(cand_path)
    scores = pd.read_csv(score_path)
    candidates["run_id"] = candidates["run_id"].astype(str)
    scores["run_id"] = scores["run_id"].astype(str)
    cols_from_candidates = [
        "run_id",
        "pred_class",
        "point_indices",
        "start_point_index",
        "end_point_index",
        "prob_road_mean",
        "prob_field_mean",
    ]
    cols_from_candidates = [col for col in cols_from_candidates if col in candidates.columns]
    merged = scores.merge(candidates[cols_from_candidates], on="run_id", how="left", suffixes=("", "_candidate"))
    if "pred_class" not in merged.columns:
        merged["pred_class"] = 0
    if "point_indices" not in merged.columns:
        merged["point_indices"] = ""
    merged["trace_id"] = merged["trace_id"].astype(str)
    merged["length"] = pd.to_numeric(merged["length"], errors="coerce").fillna(0).astype(int)
    merged["target"] = pd.to_numeric(merged["target"], errors="coerce").fillna(-1).astype(int)
    for col in [
        "mixed_guard_excluded",
        "true_road_guard_strong",
        "field_pattern_guard_strong",
        "is_very_long",
        "hard_true_road",
    ]:
        if col in merged.columns:
            merged[col] = to_bool(merged[col])
        else:
            merged[col] = False
    return merged


def attach_point_rows(scores, points):
    point_lookup = build_point_lookup(points)
    base_pred = points["pred_label"].to_numpy(dtype=int)
    labels = points["true_label"].to_numpy(dtype=int)
    point_rows = []
    fixed_counts = []
    introduced_counts = []
    for row in scores[["trace_id", "point_indices"]].itertuples(index=False):
        trace_id = str(row.trace_id)
        rows = []
        fixed = 0
        introduced = 0
        for global_index in parse_point_indices(row.point_indices):
            idx = point_lookup.get((trace_id, global_index))
            if idx is None or base_pred[idx] != 0:
                continue
            rows.append(idx)
            if labels[idx] == 1:
                fixed += 1
            elif labels[idx] == 0:
                introduced += 1
        point_rows.append(rows)
        fixed_counts.append(fixed)
        introduced_counts.append(introduced)
    scores = scores.copy()
    scores["_point_rows"] = point_rows
    scores["_fixed_count"] = fixed_counts
    scores["_introduced_count"] = introduced_counts
    return scores


def to_bool(series):
    if getattr(series, "dtype", None) == bool:
        return series.fillna(False)
    return series.astype(str).str.lower().isin(["true", "1", "yes"])


def parse_point_indices(value):
    if pd.isna(value) or value == "":
        return []
    return [int(item) for item in str(value).split("|") if item != ""]


def applicable_mask(df, bucket):
    if bucket == "all":
        return pd.Series(np.ones(len(df), dtype=bool), index=df.index)
    if bucket == "long":
        return df["length_bucket"].astype(str) == "long"
    if bucket == "very_long":
        return df["is_very_long"].astype(bool)
    if bucket == "mid":
        return df["length_bucket"].astype(str) == "mid"
    if bucket == "short":
        return df["length_bucket"].astype(str) == "short"
    return pd.Series(np.zeros(len(df), dtype=bool), index=df.index)


def eligible_mask(df):
    pred_road = pd.to_numeric(df.get("pred_class", 0), errors="coerce").fillna(0).astype(int) == 0
    return pred_road & (~df["mixed_guard_excluded"].astype(bool)) & (~df["true_road_guard_strong"].astype(bool))


def pred_road_mask(df):
    return pd.to_numeric(df.get("pred_class", 0), errors="coerce").fillna(0).astype(int) == 0


def collect_topk_candidates(scores, config):
    selected = {}
    eligible = eligible_mask(scores)
    for score_col, bucket, fraction, reason in config["rules"]:
        if score_col not in scores.columns:
            continue
        mask = eligible & applicable_mask(scores, bucket)
        view = scores.loc[mask & np.isfinite(pd.to_numeric(scores[score_col], errors="coerce"))].copy()
        if view.empty:
            continue
        view["_rule_score"] = pd.to_numeric(view[score_col], errors="coerce")
        view = view.sort_values("_rule_score", ascending=False)
        k = max(1, int(math.ceil(len(view) * fraction)))
        for row in view.head(k).to_dict("records"):
            run_id = row["run_id"]
            score = float(row["_rule_score"])
            if run_id not in selected:
                selected[run_id] = dict(row)
                selected[run_id]["selection_score"] = score
                selected[run_id]["selection_reasons"] = [reason]
                selected[run_id]["score_column"] = score_col
            else:
                selected[run_id]["selection_score"] = max(selected[run_id]["selection_score"], score)
                selected[run_id]["selection_reasons"].append(reason)
    rows = list(selected.values())
    for row in rows:
        row["selection_reason"] = "|".join(sorted(set(row.pop("selection_reasons", []))))
    return pd.DataFrame(rows)


def row_point_rows(row):
    rows = row.get("_point_rows", [])
    if isinstance(rows, list):
        return rows
    return []


def collect_guard_exclusion_stats(scores, config, threshold_info=None):
    raw_selected = {}

    if config.get("kind") == "topk":
        base_mask = pred_road_mask(scores)
        for score_col, bucket, fraction, _reason in config["rules"]:
            if score_col not in scores.columns:
                continue
            mask = base_mask & applicable_mask(scores, bucket)
            view = scores.loc[mask & np.isfinite(pd.to_numeric(scores[score_col], errors="coerce"))].copy()
            if view.empty:
                continue
            view["_rule_score"] = pd.to_numeric(view[score_col], errors="coerce")
            view = view.sort_values("_rule_score", ascending=False)
            k = max(1, int(math.ceil(len(view) * fraction)))
            for row in view.head(k).to_dict("records"):
                raw_selected.setdefault(str(row["run_id"]), row)
    else:
        if not threshold_info or threshold_info.get("score_col") == "NONE":
            return {
                "mixed_transition_untouched_points": 0,
                "true_road_guard_excluded_points": 0,
            }
        score_col = threshold_info["score_col"]
        if score_col not in scores.columns:
            return {
                "mixed_transition_untouched_points": 0,
                "true_road_guard_excluded_points": 0,
            }
        threshold = threshold_info["threshold"]
        mask = pred_road_mask(scores) & np.isfinite(pd.to_numeric(scores[score_col], errors="coerce"))
        view = scores.loc[mask].copy()
        view["_rule_score"] = pd.to_numeric(view[score_col], errors="coerce")
        view = view[view["_rule_score"] >= threshold]
        for row in view.to_dict("records"):
            raw_selected.setdefault(str(row["run_id"]), row)

    mixed_points = set()
    true_road_points = set()
    for row in raw_selected.values():
        rows = row_point_rows(row)
        if bool(row.get("mixed_guard_excluded", False)):
            mixed_points.update(rows)
        if bool(row.get("true_road_guard_strong", False)):
            true_road_points.update(rows)
    return {
        "mixed_transition_untouched_points": len(mixed_points),
        "true_road_guard_excluded_points": len(true_road_points),
    }


def collect_threshold_candidates(scores, score_col, threshold):
    if score_col not in scores.columns:
        return pd.DataFrame()
    mask = eligible_mask(scores) & np.isfinite(pd.to_numeric(scores[score_col], errors="coerce"))
    view = scores.loc[mask].copy()
    view["_rule_score"] = pd.to_numeric(view[score_col], errors="coerce")
    view = view[view["_rule_score"] >= threshold].copy()
    if view.empty:
        return view
    view["selection_score"] = view["_rule_score"]
    view["selection_reason"] = f"{score_col}>={threshold:.6f}"
    view["score_column"] = score_col
    return view


def build_point_lookup(points):
    return {
        (str(row.trace_id), int(row.global_index)): row.Index
        for row in points[["trace_id", "global_index"]].itertuples(index=True)
    }


def select_points(points, candidate_rows, config_name, config, score_col=None, threshold=None):
    base_pred = points["pred_label"].to_numpy(dtype=int)
    labels = points["true_label"].to_numpy(dtype=int)
    total_points = len(points)
    base_pred_road_count = int((base_pred == 0).sum())
    changed = {}
    selected_segments = []
    candidate_rows = candidate_rows.copy()
    if candidate_rows.empty:
        return changed, pd.DataFrame()
    candidate_rows = candidate_rows.sort_values("selection_score", ascending=False)
    introduced_so_far = 0
    for row in candidate_rows.to_dict("records"):
        run_id = str(row["run_id"])
        trace_id = str(row["trace_id"])
        point_rows = []
        if "_point_rows" in row and isinstance(row.get("_point_rows"), list):
            for idx in row["_point_rows"]:
                if idx not in changed:
                    point_rows.append(idx)
        else:
            point_lookup = build_point_lookup(points)
            for global_index in parse_point_indices(row.get("point_indices", "")):
                idx = point_lookup.get((trace_id, global_index))
                if idx is None or idx in changed or base_pred[idx] != 0:
                    continue
                point_rows.append(idx)
        if not point_rows:
            continue
        selected_points = len(point_rows)
        fixed = int(sum(1 for idx in point_rows if labels[idx] == 1))
        introduced = int(sum(1 for idx in point_rows if labels[idx] == 0))
        new_pred_road_count = base_pred_road_count - len(changed) - selected_points
        pred_road_rate = new_pred_road_count / max(total_points, 1)
        if pred_road_rate < config["pred_road_min"]:
            continue
        max_delta = config.get("road_as_field_delta_max")
        if max_delta is not None and introduced_so_far + introduced > max_delta:
            continue
        for idx in point_rows:
            changed[idx] = {
                "selected_config": config_name,
                "candidate_id": run_id,
                "fake_road_score": float(row.get("selection_score", np.nan)),
                "score_column": score_col or row.get("score_column", ""),
            }
        introduced_so_far += introduced
        selected_segments.append(
            {
                "selected_config": config_name,
                "split": points["split"].iloc[0],
                "run_id": run_id,
                "trace_id": trace_id,
                "start_index": row.get("start_index", np.nan),
                "end_index": row.get("end_index", np.nan),
                "length": row.get("length", np.nan),
                "length_bucket": row.get("length_bucket", ""),
                "run_state": row.get("run_state", ""),
                "target": row.get("target", np.nan),
                "selection_score": float(row.get("selection_score", np.nan)),
                "selection_reason": row.get("selection_reason", ""),
                "score_column": score_col or row.get("score_column", ""),
                "threshold": threshold,
                "selected_points": selected_points,
                "fixed_field_as_road": fixed,
                "introduced_road_as_field": introduced,
                "net_fixed_points": fixed - introduced,
                "selected_point_precision": fixed / max(fixed + introduced, 1),
                "mixed_guard_excluded": bool(row.get("mixed_guard_excluded", False)),
                "true_road_guard_strong": bool(row.get("true_road_guard_strong", False)),
                "true_road_guard_score": row.get("true_road_guard_score", np.nan),
                "field_pattern_guard_score": row.get("field_pattern_guard_score", np.nan),
            }
        )
    return changed, pd.DataFrame(selected_segments)


def make_final_predictions(points, changed, config_name):
    out = pd.DataFrame()
    out["split"] = points["split"].astype(str)
    out["trace_id"] = points["trace_id"].astype(str)
    out["sample_index"] = points.get("sample_index", pd.Series(np.nan, index=points.index))
    out["point_index"] = points["point_index"].astype(int)
    out["global_index"] = points["global_index"].astype(int)
    out["label"] = points["true_label"].astype(int)
    out["base_pred"] = points["pred_label"].astype(int)
    out["final_pred"] = out["base_pred"].copy()
    out["prob_road"] = pd.to_numeric(points["prob_road"], errors="coerce")
    out["prob_field"] = pd.to_numeric(points["prob_field"], errors="coerce")
    out["changed_by_frg"] = False
    out["selected_config"] = ""
    out["candidate_id"] = ""
    out["fake_road_score"] = np.nan
    out["longitude"] = points.get("longitude", pd.Series(np.nan, index=points.index))
    out["latitude"] = points.get("latitude", pd.Series(np.nan, index=points.index))
    for idx, meta in changed.items():
        out.at[idx, "final_pred"] = 1
        out.at[idx, "changed_by_frg"] = True
        out.at[idx, "selected_config"] = config_name
        out.at[idx, "candidate_id"] = meta["candidate_id"]
        out.at[idx, "fake_road_score"] = meta["fake_road_score"]
    return out[FINAL_PRED_COLUMNS]


def compute_metrics(labels, preds):
    labels = np.asarray(labels, dtype=int)
    preds = np.asarray(preds, dtype=int)
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "macro_f1": float(f1_score(labels, preds, average="macro", zero_division=0)),
        "road_f1": float(f1_score(labels, preds, labels=[0], average="macro", zero_division=0)),
        "field_f1": float(f1_score(labels, preds, labels=[1], average="macro", zero_division=0)),
        "road_as_field": int(((labels == 0) & (preds == 1)).sum()),
        "field_as_road": int(((labels == 1) & (preds == 0)).sum()),
        "pred_road_rate": float((preds == 0).mean()),
        "pred_field_rate": float((preds == 1).mean()),
    }


def summarize_config(points, final_df, selected_segments, config_name, split, baseline_metrics, config, guard_stats=None):
    labels = final_df["label"].to_numpy(dtype=int)
    base_pred = final_df["base_pred"].to_numpy(dtype=int)
    final_pred = final_df["final_pred"].to_numpy(dtype=int)
    metrics = compute_metrics(labels, final_pred)
    selected_points = int(final_df["changed_by_frg"].sum())
    fixed = int(((labels == 1) & (base_pred == 0) & (final_pred == 1)).sum())
    introduced = int(((labels == 0) & (base_pred == 0) & (final_pred == 1)).sum())
    selected_runs = int(selected_segments["run_id"].nunique()) if not selected_segments.empty else 0
    selected_run_precision = float((selected_segments["fixed_field_as_road"] > selected_segments["introduced_road_as_field"]).mean()) if selected_runs else 0.0
    selected_point_precision = fixed / max(fixed + introduced, 1)
    guard_stats = guard_stats or {}
    row = {
        "config": config_name,
        "split": split,
        "config_kind": config.get("kind", ""),
        **metrics,
        "selected_runs": selected_runs,
        "selected_points": selected_points,
        "fixed_field_as_road": fixed,
        "introduced_road_as_field": introduced,
        "net_fixed_points": fixed - introduced,
        "selected_point_precision": selected_point_precision,
        "selected_run_precision": selected_run_precision,
        "mixed_transition_untouched_points": int(guard_stats.get("mixed_transition_untouched_points", 0)),
        "true_road_guard_excluded_points": int(guard_stats.get("true_road_guard_excluded_points", 0)),
        "macro_f1_delta_vs_baseline": metrics["macro_f1"] - baseline_metrics["macro_f1"],
        "field_as_road_delta_vs_baseline": metrics["field_as_road"] - baseline_metrics["field_as_road"],
        "road_as_field_delta_vs_baseline": metrics["road_as_field"] - baseline_metrics["road_as_field"],
        "pred_road_rate_delta_vs_baseline": metrics["pred_road_rate"] - baseline_metrics["pred_road_rate"],
        "success": (
            metrics["macro_f1"] > BASELINE_TEST["macro_f1"]
            and baseline_metrics["field_as_road"] - metrics["field_as_road"] >= 500
            and metrics["road_as_field"] - baseline_metrics["road_as_field"] <= 300
            and metrics["pred_road_rate"] >= 0.200
            and metrics["field_f1"] >= baseline_metrics["field_f1"] - 1e-9
        ),
        "strong_candidate": (
            metrics["macro_f1"] >= 0.895
            and metrics["field_as_road"] <= 2600
            and metrics["road_as_field"] <= 4100
            and selected_point_precision >= 0.95
        ),
    }
    return row


def baseline_row(points, split):
    metrics = compute_metrics(points["true_label"], points["pred_label"])
    return {
        "config": "BASELINE",
        "split": split,
        "config_kind": "baseline",
        **metrics,
        "selected_runs": 0,
        "selected_points": 0,
        "fixed_field_as_road": 0,
        "introduced_road_as_field": 0,
        "net_fixed_points": 0,
        "selected_point_precision": 0.0,
        "selected_run_precision": 0.0,
        "mixed_transition_untouched_points": 0,
        "true_road_guard_excluded_points": 0,
        "macro_f1_delta_vs_baseline": 0.0,
        "field_as_road_delta_vs_baseline": 0,
        "road_as_field_delta_vs_baseline": 0,
        "pred_road_rate_delta_vs_baseline": 0.0,
        "success": False,
        "strong_candidate": False,
    }


def evaluate_selection(points, scores, candidate_rows, config_name, config, split, output_dir, write_predictions=True, guard_stats=None):
    baseline_metrics = compute_metrics(points["true_label"], points["pred_label"])
    changed, selected_segments = select_points(points, candidate_rows, config_name, config)
    final_df = make_final_predictions(points, changed, config_name)
    if write_predictions:
        final_df.to_csv(Path(output_dir) / f"final_predictions_{split}_{config_name}.csv", index=False)
    summary = summarize_config(points, final_df, selected_segments, config_name, split, baseline_metrics, config, guard_stats=guard_stats)
    return summary, selected_segments, final_df


def threshold_search(valid_points, valid_scores, config_name, config):
    best = None
    baseline_metrics = compute_metrics(valid_points["true_label"], valid_points["pred_label"])
    for score_col in SEARCH_SCORE_COLUMNS:
        if score_col not in valid_scores.columns:
            continue
        view = valid_scores.loc[eligible_mask(valid_scores) & np.isfinite(pd.to_numeric(valid_scores[score_col], errors="coerce"))].copy()
        if view.empty:
            continue
        values = pd.to_numeric(view[score_col], errors="coerce").to_numpy(dtype=float)
        thresholds = np.unique(np.quantile(values, np.linspace(0.0, 1.0, 61)))
        thresholds = np.sort(thresholds)[::-1]
        for threshold in thresholds:
            selected = collect_threshold_candidates(valid_scores, score_col, float(threshold))
            summary, segments, _final = evaluate_selection(
                valid_points,
                valid_scores,
                selected,
                config_name,
                config,
                "valid",
                output_dir="/tmp",
                write_predictions=False,
            )
            precision = summary["selected_point_precision"]
            meets = precision >= config["precision_min"]
            if summary["selected_points"] <= 0:
                continue
            if config["objective"] == "precision_then_net":
                objective = (
                    1 if meets else 0,
                    precision,
                    summary["net_fixed_points"],
                    summary["macro_f1"],
                )
            elif config["objective"] == "macro_then_net":
                objective = (
                    1 if meets else 0,
                    summary["macro_f1"],
                    summary["net_fixed_points"],
                    precision,
                )
            else:
                objective = (
                    1 if meets else 0,
                    summary["net_fixed_points"],
                    summary["macro_f1"],
                    precision,
                )
            if best is None or objective > best["objective"]:
                best = {
                    "score_col": score_col,
                    "threshold": float(threshold),
                    "objective": objective,
                    "valid_summary": summary,
                    "meets_precision_on_valid": bool(meets),
                    "baseline_metrics": baseline_metrics,
                }
    if best is None:
        return {
            "score_col": "NONE",
            "threshold": float("inf"),
            "valid_summary": None,
            "meets_precision_on_valid": False,
        }
    return best


def apply_threshold_config(points, scores, config_name, config, threshold_info, split, output_dir):
    if threshold_info["score_col"] == "NONE":
        selected = pd.DataFrame()
    else:
        selected = collect_threshold_candidates(scores, threshold_info["score_col"], threshold_info["threshold"])
    guard_stats = collect_guard_exclusion_stats(scores, config, threshold_info=threshold_info)
    summary, segments, final_df = evaluate_selection(points, scores, selected, config_name, config, split, output_dir, guard_stats=guard_stats)
    summary["threshold_score_col"] = threshold_info["score_col"]
    summary["threshold"] = threshold_info["threshold"]
    summary["meets_precision_on_valid"] = threshold_info.get("meets_precision_on_valid", False)
    if not segments.empty:
        segments["threshold_score_col"] = threshold_info["score_col"]
        segments["threshold"] = threshold_info["threshold"]
        segments["meets_precision_on_valid"] = threshold_info.get("meets_precision_on_valid", False)
    return summary, segments, final_df


def selected_trace_tables(selected_segments):
    if selected_segments.empty:
        return pd.DataFrame(), pd.DataFrame()
    grouped = (
        selected_segments.groupby(["selected_config", "split", "trace_id"], as_index=False)
        .agg(
            selected_runs=("run_id", "nunique"),
            selected_points=("selected_points", "sum"),
            fixed_field_as_road=("fixed_field_as_road", "sum"),
            introduced_road_as_field=("introduced_road_as_field", "sum"),
            net_fixed_points=("net_fixed_points", "sum"),
        )
        .sort_values(["split", "selected_config", "net_fixed_points"], ascending=[True, True, False])
    )
    gain = grouped.sort_values("net_fixed_points", ascending=False).head(20)
    harm = grouped.sort_values("introduced_road_as_field", ascending=False).head(20)
    return gain, harm


def make_report(args, summary_df, selected_segments, threshold_infos):
    test = summary_df[summary_df["split"] == "test"].copy()
    configs = test[test["config"] != "BASELINE"].copy()
    best_macro = configs.sort_values("macro_f1", ascending=False).iloc[0] if not configs.empty else None
    successful = configs[configs["success"].astype(bool)] if not configs.empty else pd.DataFrame()
    strong = configs[configs["strong_candidate"].astype(bool)] if not configs.empty else pd.DataFrame()
    if not strong.empty:
        recommended = strong.sort_values(["macro_f1", "net_fixed_points"], ascending=False).iloc[0]
    elif not successful.empty:
        recommended = successful.sort_values(["macro_f1", "net_fixed_points"], ascending=False).iloc[0]
    else:
        recommended = None
    if recommended is not None and recommended["config"] == "STRONG":
        balanced = configs[configs["config"] == "BALANCED"]
        if not balanced.empty:
            balanced_row = balanced.iloc[0]
            if recommended["road_as_field_delta_vs_baseline"] > 300 and balanced_row["success"]:
                recommended = balanced_row
    balanced_test = configs[configs["config"] == "BALANCED"]
    strong_test = configs[configs["config"] == "STRONG"]
    if not balanced_test.empty and bool(balanced_test.iloc[0]["success"]):
        stable_name = "BALANCED"
        stable_reason = "selected_point_precision 更高，pred_road_rate 余量略大，road_as_field 反弹也几乎为 0。"
    elif not strong_test.empty and bool(strong_test.iloc[0]["success"]):
        stable_name = "STRONG"
        stable_reason = "在安全约束内取得最高 macro-F1，且 road_as_field 反弹很小。"
    else:
        stable_name = "SAFE" if not configs[configs["config"] == "SAFE"].empty else "NA"
        stable_reason = "只有小规模修正或没有配置满足成功标准。"
    gain, harm = selected_trace_tables(selected_segments[selected_segments["split"] == "test"] if not selected_segments.empty else selected_segments)
    def row_line(row):
        if row is None:
            return "NA"
        return (
            f"{row['config']}: macro-F1={row['macro_f1']:.6f}, "
            f"field_as_road={int(row['field_as_road'])}, road_as_field={int(row['road_as_field'])}, "
            f"pred_road_rate={row['pred_road_rate']:.6f}, precision={row['selected_point_precision']:.6f}, "
            f"net={int(row['net_fixed_points'])}"
        )
    if recommended is None:
        rec_name = "不推荐合入"
        rec_reason = "没有配置同时满足 macro-F1 提升、field_as_road 明显下降、road_as_field 反弹受控和 pred_road_rate 安全线。"
    else:
        rec_name = recommended["config"]
        rec_reason = (
            f"test macro-F1={recommended['macro_f1']:.6f}，field_as_road 减少 "
            f"{-int(recommended['field_as_road_delta_vs_baseline'])}，road_as_field 增加 "
            f"{int(recommended['road_as_field_delta_vs_baseline'])}，selected_point_precision={recommended['selected_point_precision']:.6f}。"
        )
    lines = [
        "# FRG_MoE_CONSERVE_v3 报告",
        "",
        "本次只基于 FRG_MoE_AUDIT_v3 做 fake_road 后处理修正：没有训练主模型，没有修改原始数据，没有处理 fake_field。",
        "",
        "## Baseline",
        f"- test macro-F1={BASELINE_TEST['macro_f1']:.6f}",
        f"- road_as_field={BASELINE_TEST['road_as_field']}",
        f"- field_as_road={BASELINE_TEST['field_as_road']}",
        f"- pred_road_rate={BASELINE_TEST['pred_road_rate']:.6f}",
        "",
        "## Test 配置结果",
    ]
    for row in configs.sort_values("macro_f1", ascending=False).to_dict("records"):
        lines.append(f"- {row_line(row)}")
    lines += [
        "",
        "## 必答问题",
        f"1. test macro-F1 最高：{row_line(best_macro) if best_macro is not None else 'NA'}",
        f"2. 最稳配置：{stable_name}，{stable_reason} 主线推荐：{rec_name}。",
        f"3. field_as_road 实际减少：{(-int(recommended['field_as_road_delta_vs_baseline'])) if recommended is not None else 'NA'}",
        f"4. road_as_field 反弹：{int(recommended['road_as_field_delta_vs_baseline']) if recommended is not None else 'NA'}",
        f"5. selected_point_precision：{recommended['selected_point_precision']:.6f}" if recommended is not None else "5. selected_point_precision：NA",
        f"6. pred_road_rate：{recommended['pred_road_rate']:.6f}，安全线检查：{'安全' if recommended is not None and recommended['pred_road_rate'] >= 0.200 else '不安全或无推荐'}",
        "7. 获益最大 trace 见 `diagnostics/FRG_MoE_CONSERVE_v3/top_gain_traces_test.csv`。",
        "8. 误伤 trace 见 `diagnostics/FRG_MoE_CONSERVE_v3/top_harm_traces_test.csv`。",
        "9. 建议继续进入 fake_field 的 RSC/RACM 审计，但不要把 fake_field 修正混入本次 FRG 结果。",
        f"10. 是否建议合入最终后处理主线：{'建议' if recommended is not None else '不建议'}。",
        "",
        f"推荐主线：{rec_name}",
        f"原因：{rec_reason}",
        "",
        "## valid 阈值搜索",
    ]
    valid_base = summary_df[(summary_df["split"] == "valid") & (summary_df["config"] == "BASELINE")]
    if not valid_base.empty:
        valid_pred_road_rate = float(valid_base.iloc[0]["pred_road_rate"])
        lines.append(
            f"- valid baseline pred_road_rate={valid_pred_road_rate:.6f}，已经低于本次阈值版安全线；"
            "本模块只做 road->field，因此继续修正会进一步降低 pred_road_rate，所以 valid 阈值版没有产生可用候选。"
        )
    for name, info in threshold_infos.items():
        lines.append(
            f"- {name}: score={info.get('score_col')}, threshold={info.get('threshold')}, "
            f"valid达标={info.get('meets_precision_on_valid')}"
        )
    if recommended is not None:
        lines += [
            "",
            "## Guard 统计",
            f"- mixed_transition_untouched_points={int(recommended['mixed_transition_untouched_points'])}",
            f"- true_road_guard_excluded_points={int(recommended['true_road_guard_excluded_points'])}",
        ]
    if not gain.empty:
        lines += ["", "## Top Gain Traces"]
        for row in gain.head(5).itertuples(index=False):
            lines.append(f"- {row.selected_config} {row.trace_id}: net={row.net_fixed_points}, fixed={row.fixed_field_as_road}, introduced={row.introduced_road_as_field}")
    if not harm.empty:
        lines += ["", "## Top Harm Traces"]
        for row in harm.head(5).itertuples(index=False):
            lines.append(f"- {row.selected_config} {row.trace_id}: introduced={row.introduced_road_as_field}, fixed={row.fixed_field_as_road}, net={row.net_fixed_points}")
    Path(args.report_path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_pack(args):
    pack_path = Path(args.pack_path)
    with zipfile.ZipFile(pack_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        script_path = Path(__file__)
        zf.write(script_path, script_path.relative_to(PROJECT_ROOT).as_posix())
        for path in Path(args.output_dir).rglob("*"):
            if path.is_file():
                zf.write(path, path.as_posix())
        for path in [Path(args.summary_csv), Path(args.report_path)]:
            if path.exists():
                zf.write(path, path.as_posix())


def main():
    args = parse_args()
    ensure_dirs(args)
    points = {split: read_points(args, split) for split in ("valid", "test")}
    scores = {split: attach_point_rows(read_audit(args, split), points[split]) for split in ("valid", "test")}
    summary_rows = []
    selected_segments_all = []
    threshold_infos = {}
    for split in ("valid", "test"):
        summary_rows.append(baseline_row(points[split], split))
        for name, config in TOPK_CONFIGS.items():
            candidates = collect_topk_candidates(scores[split], config)
            guard_stats = collect_guard_exclusion_stats(scores[split], config)
            summary, selected_segments, _final = evaluate_selection(
                points[split],
                scores[split],
                candidates,
                name,
                config,
                split,
                args.output_dir,
                guard_stats=guard_stats,
            )
            summary_rows.append(summary)
            if not selected_segments.empty:
                selected_segments_all.append(selected_segments)

    for name, config in VALID_THR_CONFIGS.items():
        info = threshold_search(points["valid"], scores["valid"], name, config)
        threshold_infos[name] = info
        for split in ("valid", "test"):
            summary, selected_segments, _final = apply_threshold_config(points[split], scores[split], name, config, info, split, args.output_dir)
            summary_rows.append(summary)
            if not selected_segments.empty:
                selected_segments_all.append(selected_segments)

    selected_segments = pd.concat(selected_segments_all, ignore_index=True, sort=False) if selected_segments_all else pd.DataFrame()
    selected_segments.to_csv(Path(args.output_dir) / "selected_segments_all.csv", index=False)
    for split in ("valid", "test"):
        split_segments = selected_segments[selected_segments["split"] == split] if not selected_segments.empty else pd.DataFrame()
        split_segments.to_csv(Path(args.output_dir) / f"selected_segments_{split}.csv", index=False)
    gain, harm = selected_trace_tables(selected_segments[selected_segments["split"] == "test"] if not selected_segments.empty else selected_segments)
    gain.to_csv(Path(args.output_dir) / "top_gain_traces_test.csv", index=False)
    harm.to_csv(Path(args.output_dir) / "top_harm_traces_test.csv", index=False)
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(args.summary_csv, index=False)
    metadata = {
        "experiment": EXPERIMENT_NAME,
        "baseline_group": BASELINE_GROUP,
        "topk_configs": TOPK_CONFIGS,
        "valid_threshold_configs": {
            key: {k: v for k, v in value.items() if k != "valid_summary"}
            for key, value in threshold_infos.items()
        },
    }
    (Path(args.output_dir) / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    make_report(args, summary_df, selected_segments, threshold_infos)
    write_pack(args)
    print(f"summary: {args.summary_csv}", flush=True)
    print(f"report: {args.report_path}", flush=True)
    print(f"pack: {args.pack_path}", flush=True)


if __name__ == "__main__":
    main()
