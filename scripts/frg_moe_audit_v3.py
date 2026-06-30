import argparse
import json
import math
import sys
import warnings
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.errors import PerformanceWarning
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
for path in (PROJECT_ROOT, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


AUDIT_NAME = "FRG_MoE_AUDIT_v3"
BASELINE_GROUP = "PT2G_MSC_v1_time_fixed_strict_40pre"
DEFAULT_RUN_DIR = "diagnostics/TFSV_LC_MoE_AUDIT_v2"
DEFAULT_POINT_DIR = f"diagnostics/{BASELINE_GROUP}_error_analysis"
EXPERT_SPECS = [
    ("fake_road_short_expert", "short"),
    ("fake_road_mid_expert", "mid"),
    ("fake_road_long_expert", "long"),
    ("fake_road_very_long_expert", "very_long"),
    ("fake_road_global_expert", "all"),
    ("fake_road_hard_negative_guard", "all"),
]
TOPK_BY_BUCKET = {
    "short": [0.05, 0.10, 0.20],
    "mid": [0.05, 0.10, 0.20],
    "long": [0.05, 0.10, 0.20, 0.30],
    "very_long": [0.05, 0.10, 0.20, 0.30],
    "all": [0.05, 0.10, 0.20, 0.30],
}
PRECISION_TARGETS = [0.90, 0.95, 0.98]
warnings.filterwarnings("ignore", category=PerformanceWarning)
ID_COLUMNS = [
    "split",
    "run_id",
    "trace_id",
    "pred_class",
    "start_index",
    "end_index",
    "length",
    "length_bucket",
    "is_very_long",
    "road_points",
    "field_points",
    "road_ratio",
    "field_ratio",
    "run_state",
    "target",
    "hard_true_road",
    "pattern_type",
]


def parse_args():
    parser = argparse.ArgumentParser(description="FRG-MoE_AUDIT_v3 fake-road guard verifier audit.")
    parser.add_argument("--run_dir", default=DEFAULT_RUN_DIR)
    parser.add_argument("--point_dir", default=DEFAULT_POINT_DIR)
    parser.add_argument("--output_dir", default=f"diagnostics/{AUDIT_NAME}")
    parser.add_argument("--summary_csv", default=f"results/{AUDIT_NAME}_summary.csv")
    parser.add_argument("--report_path", default=f"analysis/{AUDIT_NAME}_report.md")
    parser.add_argument("--pack_path", default=f"analysis_packs/{AUDIT_NAME}_for_chatgpt.zip")
    parser.add_argument("--v2_summary", default="results/TFSV_LC_MoE_AUDIT_v2_summary.csv")
    parser.add_argument("--v2_topk", default="diagnostics/TFSV_LC_MoE_AUDIT_v2/expert_topk_precision_recall.csv")
    parser.add_argument("--splits", default="train,valid,test")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_jobs", type=int, default=8)
    return parser.parse_args()


def ensure_dirs(args):
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.summary_csv).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report_path).parent.mkdir(parents=True, exist_ok=True)
    Path(args.pack_path).parent.mkdir(parents=True, exist_ok=True)


def clip01(value):
    try:
        if value is None or math.isnan(float(value)):
            return 0.0
        return float(max(0.0, min(1.0, float(value))))
    except Exception:
        return 0.0


def sigmoid(value):
    value = max(-50.0, min(50.0, float(value)))
    return 1.0 / (1.0 + math.exp(-value))


def fmt(value, digits=6):
    try:
        if value is None or math.isnan(float(value)):
            return "NA"
        return f"{float(value):.{digits}f}"
    except Exception:
        return str(value)


def numeric_col(df, column, default=0.0):
    if column not in df.columns:
        return pd.Series(np.full(len(df), default, dtype=float), index=df.index)
    return pd.to_numeric(df[column], errors="coerce").fillna(default)


def safe_auc(y_true, scores):
    try:
        y_true = np.asarray(y_true, dtype=int)
        scores = np.asarray(scores, dtype=float)
        mask = np.isfinite(scores)
        y_true = y_true[mask]
        scores = scores[mask]
        if len(y_true) == 0 or len(set(y_true.tolist())) < 2:
            return float("nan")
        return float(roc_auc_score(y_true, scores))
    except Exception:
        return float("nan")


def safe_ap(y_true, scores):
    try:
        y_true = np.asarray(y_true, dtype=int)
        scores = np.asarray(scores, dtype=float)
        mask = np.isfinite(scores)
        y_true = y_true[mask]
        scores = scores[mask]
        if len(y_true) == 0 or len(set(y_true.tolist())) < 2:
            return float("nan")
        return float(average_precision_score(y_true, scores))
    except Exception:
        return float("nan")


def length_bucket(length):
    if length < 10:
        return "short"
    if length < 50:
        return "mid"
    return "long"


def load_runs(split, args):
    path = Path(args.run_dir) / f"predicted_runs_{split}.csv"
    if not path.exists():
        raise FileNotFoundError(f"PREDICTED_RUNS_NOT_FOUND: {path}")
    df = pd.read_csv(path)
    if "pred_class" not in df.columns and "pred_label" in df.columns:
        df["pred_class"] = df["pred_label"]
    df["pred_class"] = numeric_col(df, "pred_class", -1).astype(int)
    df = df[df["pred_class"] == 0].copy()
    df["split"] = split
    return enrich_candidates(df)


def enrich_candidates(df):
    df = df.copy()
    for col in [
        "length",
        "road_points",
        "field_points",
        "road_ratio",
        "field_ratio",
        "prob_road_mean",
        "prob_road_std",
        "prob_road_min",
        "prob_road_max",
        "prob_field_mean",
        "prob_field_std",
        "prob_field_min",
        "prob_field_max",
        "margin_mean",
        "margin_std",
        "margin_min",
        "margin_max",
        "confidence_mean",
        "confidence_std",
        "confidence_min",
        "confidence_max",
        "entropy_mean",
        "local_step_mean_m_mean",
        "local_step_mean_m_min",
        "local_step_mean_m_q25",
        "local_step_mean_m_q75",
        "local_step_std_m_mean",
        "local_turn_angle_deg_mean",
        "local_turn_angle_deg_q75",
        "local_turn_angle_deg_max",
        "local_density_1m_mean",
        "local_density_1m_q75",
        "local_density_1m_max",
        "local_density_2m_mean",
        "local_density_2m_q75",
        "local_density_2m_max",
        "stationary_rate",
        "stationary_run_length_mean",
        "stationary_run_length_max",
        "trace_position_mean",
        "near_endpoint_rate",
        "path_length_m",
        "start_end_distance_m",
        "compactness",
        "bbox_width_m",
        "bbox_height_m",
        "bbox_diag_m",
        "pca_major_axis_m",
        "pca_minor_axis_m",
        "pca_aspect_ratio",
        "trajectory_linearity",
        "path_area_ratio",
        "prev_run_length",
        "next_run_length",
        "prev_prob_field_mean",
        "next_prob_field_mean",
        "prev_prob_road_mean",
        "next_prob_road_mean",
    ]:
        df[col] = numeric_col(df, col, 0.0)
    if "convex_hull_area_m2" not in df.columns:
        df["convex_hull_area_m2"] = numeric_col(df, "convex_hull_area", 0.0)
    else:
        df["convex_hull_area_m2"] = numeric_col(df, "convex_hull_area_m2", 0.0)
    df["bbox_area_m2"] = numeric_col(df, "bbox_area_m2", df["bbox_width_m"] * df["bbox_height_m"])
    df["length"] = df["length"].astype(int)
    df["length_bucket"] = df["length"].map(length_bucket)
    df["is_very_long"] = df["length"] >= 100
    df["road_ratio"] = df["road_points"] / df["length"].clip(lower=1)
    df["field_ratio"] = df["field_points"] / df["length"].clip(lower=1)
    df["run_state"] = "mixed_transition"
    df.loc[df["road_ratio"] >= 0.8, "run_state"] = "true_road"
    df.loc[df["field_ratio"] >= 0.8, "run_state"] = "fake_road"
    df["target"] = np.where(df["run_state"] == "fake_road", 1, np.where(df["run_state"] == "true_road", 0, -1))
    if "pattern_type" not in df.columns:
        df["pattern_type"] = "other"
    df["pattern_type"] = df.apply(normalize_pattern_type, axis=1)
    df["entropy_max"] = np.maximum(numeric_col(df, "entropy_max", 0.0), df["entropy_mean"] + 0.5 * df["margin_std"])
    df["low_margin_rate"] = (0.2 - df["margin_min"]).clip(lower=0.0, upper=0.2) / 0.2
    df["high_conf_road_rate"] = ((df["prob_road_mean"] - 0.70) / 0.30).clip(lower=0.0, upper=1.0)
    df["prob_road_drop_from_prev"] = (df["prev_prob_road_mean"] - df["prob_road_mean"]).clip(lower=0.0)
    df["prob_road_drop_to_next"] = (df["prob_road_mean"] - df["next_prob_road_mean"]).clip(lower=0.0)
    df["prob_road_spike_from_context"] = (
        df["prob_road_mean"] - (df["prev_prob_road_mean"] + df["next_prob_road_mean"]) * 0.5
    ).clip(lower=0.0)

    density1_score = (df["local_density_1m_mean"] / 8.0).clip(lower=0.0, upper=1.0)
    density2_score = (df["local_density_2m_mean"] / 14.0).clip(lower=0.0, upper=1.0)
    df["density_score"] = (density1_score + density2_score) * 0.5
    df["stationary_score"] = df["stationary_rate"].clip(lower=0.0, upper=1.0)
    df["compactness_score"] = np.log1p(df["compactness"].clip(lower=0.0)) / math.log(20.0)
    df["compactness_score"] = df["compactness_score"].clip(lower=0.0, upper=1.0)
    df["turn_score"] = (df["local_turn_angle_deg_q75"] / 90.0).clip(lower=0.0, upper=1.0)
    df["low_displacement_score"] = 1.0 / (1.0 + df["start_end_distance_m"].clip(lower=0.0) / 25.0)
    df["aspect_score"] = (np.log1p(df["pca_aspect_ratio"].clip(lower=0.0)) / math.log(40.0)).clip(lower=0.0, upper=1.0)
    df["distance_score"] = (df["start_end_distance_m"] / 120.0).clip(lower=0.0, upper=1.0)
    df["road_linearity_score"] = (
        0.45 * df["trajectory_linearity"].clip(lower=0.0, upper=1.0)
        + 0.30 * df["aspect_score"]
        + 0.25 * df["distance_score"]
    ).clip(lower=0.0, upper=1.0)
    df["field_dense_stationary_score"] = (
        df["stationary_score"] * df["density_score"] * df["compactness_score"]
    ).clip(lower=0.0, upper=1.0)
    df["field_work_pattern_score"] = (
        df["density_score"] * df["turn_score"] * df["low_displacement_score"]
    ).clip(lower=0.0, upper=1.0)
    df["false_road_risk_score"] = (
        0.5 + 0.5 * (df["field_dense_stationary_score"] + df["field_work_pattern_score"] - df["road_linearity_score"])
    ).clip(lower=0.0, upper=1.0)

    df["field_context_score"] = df["pattern_type"].map(
        {
            "field_road_field": 1.0,
            "field_road": 0.65,
            "road_field": 0.25,
            "road_road_road": 0.0,
            "other": 0.2,
        }
    ).fillna(0.2)
    df["context_field_strength"] = ((df["prev_prob_field_mean"] + df["next_prob_field_mean"]) * 0.5).clip(lower=0.0, upper=1.0)
    df["field_enclosure_score"] = (df["field_context_score"] * df["context_field_strength"]).clip(lower=0.0, upper=1.0)

    attach_trace_normalized_features(df)
    df["hard_true_road"] = (
        (df["run_state"] == "true_road")
        & (
            (df["stationary_rate"] >= 0.5)
            | (df["local_density_1m_mean"] >= 5.0)
            | (df["compactness"] >= 3.0)
            | ((df["field_enclosure_score"] >= 0.45) & (df["false_road_risk_score"] >= 0.45))
        )
    )
    df["true_road_guard_score"] = (
        0.25 * df["aspect_score"]
        + 0.25 * df["trajectory_linearity"].clip(lower=0.0, upper=1.0)
        + 0.20 * df["distance_score"]
        + 0.15 * (1.0 - df["stationary_score"])
        + 0.10 * (1.0 - df["density_score"])
        + 0.05 * (df["pattern_type"] == "road_road_road").astype(float)
    ).clip(lower=0.0, upper=1.0)
    df["true_road_guard_strong"] = df["true_road_guard_score"] >= 0.62
    df["field_pattern_guard_score"] = (
        0.20 * df["stationary_score"]
        + 0.20 * df["density_score"]
        + 0.20 * df["low_displacement_score"]
        + 0.15 * df["compactness_score"]
        + 0.15 * (df["pattern_type"] == "field_road_field").astype(float)
        + 0.10 * (df["trace_position_mean"] >= 0.45).astype(float)
    ).clip(lower=0.0, upper=1.0)
    df["field_pattern_guard_strong"] = df["field_pattern_guard_score"] >= 0.58
    df["mixed_guard_excluded"] = df["run_state"] == "mixed_transition"
    return df


def normalize_pattern_type(row):
    pattern = str(row.get("pattern_type", "other"))
    prev_class = row.get("prev_pred_class", "")
    next_class = row.get("next_pred_class", "")
    if pattern and pattern != "nan":
        if pattern in {"field_road_field", "field_road", "road_field", "road_road_road", "other"}:
            return pattern
    try:
        prev_i = int(float(prev_class))
        next_i = int(float(next_class))
    except Exception:
        return "other"
    if prev_i == 1 and next_i == 1:
        return "field_road_field"
    if prev_i == 1 and next_i != 1:
        return "field_road"
    if prev_i == 0 and next_i == 0:
        return "road_road_road"
    if next_i == 1:
        return "road_field"
    return "other"


def attach_trace_normalized_features(df):
    mapping = {
        "density1_zscore_in_trace": "local_density_1m_mean",
        "density2_zscore_in_trace": "local_density_2m_mean",
        "step_mean_zscore_in_trace": "local_step_mean_m_mean",
        "turn_zscore_in_trace": "local_turn_angle_deg_mean",
        "compactness_zscore_in_trace": "compactness",
        "trace_position_zscore": "trace_position_mean",
    }
    for out_col, source_col in mapping.items():
        if out_col in df.columns:
            df[out_col] = numeric_col(df, out_col, 0.0)
            continue
        df[out_col] = 0.0
        for _trace_id, idx in df.groupby("trace_id").groups.items():
            values = numeric_col(df.loc[idx], source_col, 0.0)
            std = float(values.std(ddof=0))
            if std <= 1e-9:
                df.loc[idx, out_col] = 0.0
            else:
                df.loc[idx, out_col] = (values - float(values.mean())) / std
    if "stationary_rate_relative_to_trace" not in df.columns:
        df["stationary_rate_relative_to_trace"] = 0.0
        for _trace_id, idx in df.groupby("trace_id").groups.items():
            values = numeric_col(df.loc[idx], "stationary_rate", 0.0)
            baseline = float(values.mean())
            df.loc[idx, "stationary_rate_relative_to_trace"] = values - baseline
    if "density_zscore_in_trace" in df.columns:
        df["density1_zscore_in_trace"] = numeric_col(df, "density_zscore_in_trace", 0.0)
    if "step_zscore_in_trace" in df.columns:
        df["step_mean_zscore_in_trace"] = numeric_col(df, "step_zscore_in_trace", 0.0)
    if "stationary_relative_to_trace" in df.columns:
        df["stationary_rate_relative_to_trace"] = numeric_col(df, "stationary_relative_to_trace", 0.0)


def applicable_mask(df, bucket):
    if bucket == "all":
        return pd.Series(np.ones(len(df), dtype=bool), index=df.index)
    if bucket == "very_long":
        return df["is_very_long"].astype(bool)
    return df["length_bucket"] == bucket


def pure_mask(df):
    return df["target"].isin([0, 1])


def model_feature_columns(df):
    excluded = set(ID_COLUMNS)
    excluded.update(
        {
            "point_indices",
            "main_error_type",
            "prev_run_id",
            "next_run_id",
            "source_run_id",
            "mixed_guard_excluded",
            "true_road_guard_strong",
            "field_pattern_guard_strong",
        }
    )
    cols = []
    for col in df.columns:
        if col in excluded:
            continue
        if col.endswith("_score_raw") or col.endswith("_score") and col.startswith("fake_road_"):
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            cols.append(col)
    return cols


def build_model_candidates(seed, n_jobs):
    return [
        (
            "extra_trees",
            make_pipeline(
                SimpleImputer(strategy="median"),
                ExtraTreesClassifier(
                    n_estimators=260,
                    min_samples_leaf=3,
                    class_weight="balanced",
                    random_state=seed,
                    n_jobs=n_jobs,
                ),
            ),
        ),
        (
            "random_forest",
            make_pipeline(
                SimpleImputer(strategy="median"),
                RandomForestClassifier(
                    n_estimators=220,
                    min_samples_leaf=4,
                    class_weight="balanced",
                    random_state=seed,
                    n_jobs=n_jobs,
                ),
            ),
        ),
        (
            "gradient_boosting",
            make_pipeline(
                SimpleImputer(strategy="median"),
                GradientBoostingClassifier(n_estimators=140, max_depth=3, subsample=0.85, random_state=seed),
            ),
        ),
        (
            "hist_gradient_boosting",
            make_pipeline(
                SimpleImputer(strategy="median"),
                HistGradientBoostingClassifier(max_iter=140, random_state=seed),
            ),
        ),
        (
            "logistic_regression",
            make_pipeline(
                SimpleImputer(strategy="median"),
                StandardScaler(),
                LogisticRegression(max_iter=3000, class_weight="balanced", random_state=seed),
            ),
        ),
    ]


def sample_weights(train_df, expert_name):
    y = train_df["target"].astype(int).to_numpy()
    weights = np.ones(len(train_df), dtype=float)
    pos = max(int((y == 1).sum()), 1)
    neg = max(int((y == 0).sum()), 1)
    weights[y == 1] *= min(5.0, neg / pos)
    hard = train_df["hard_true_road"].astype(bool).to_numpy()
    weights[(y == 0) & hard] *= 4.0 if expert_name == "fake_road_hard_negative_guard" else 2.0
    return weights


def fit_with_weights(model, x, y, weights):
    try:
        if hasattr(model, "steps"):
            last_name = model.steps[-1][0]
            model.fit(x, y, **{f"{last_name}__sample_weight": weights})
        else:
            model.fit(x, y, sample_weight=weights)
    except TypeError:
        model.fit(x, y)


def predict_scores(model, x):
    if len(x) == 0:
        return np.array([], dtype=float)
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(x)
        if probs.shape[1] == 1:
            return np.zeros(len(x), dtype=float)
        return probs[:, 1]
    values = model.decision_function(x)
    return 1.0 / (1.0 + np.exp(-values))


def guard_score(raw_scores, df):
    raw_scores = np.asarray(raw_scores, dtype=float)
    true_guard = df["true_road_guard_score"].to_numpy(dtype=float)
    field_guard = df["field_pattern_guard_score"].to_numpy(dtype=float)
    guarded = raw_scores * (1.0 - 0.55 * true_guard) + 0.20 * field_guard
    guarded[df["mixed_guard_excluded"].astype(bool).to_numpy()] = -np.inf
    return np.clip(guarded, -np.inf, 1.0)


def fit_experts(candidates, args):
    train = candidates["train"]
    valid = candidates["valid"]
    feature_cols = model_feature_columns(train)
    score_frames = {split: candidate_score_base(df) for split, df in candidates.items()}
    eval_rows = []
    feature_importances = []
    expert_models = {}

    for expert_name, bucket in EXPERT_SPECS:
        train_mask = applicable_mask(train, bucket) & pure_mask(train)
        train_df = train.loc[train_mask].copy()
        if train_df.empty or len(set(train_df["target"].astype(int).tolist())) < 2:
            eval_rows.append({"section": "expert_eval", "expert": expert_name, "bucket": bucket, "status": "skipped", "reason": "insufficient_train"})
            continue
        valid_mask = applicable_mask(valid, bucket) & pure_mask(valid)
        valid_df = valid.loc[valid_mask].copy()
        if valid_df.empty:
            eval_rows.append({"section": "expert_eval", "expert": expert_name, "bucket": bucket, "status": "skipped", "reason": "empty_valid"})
            continue
        x_train = train_df[feature_cols]
        y_train = train_df["target"].astype(int).to_numpy()
        weights = sample_weights(train_df, expert_name)
        best = None
        for model_name, model in build_model_candidates(args.seed, args.n_jobs):
            try:
                fit_with_weights(model, x_train, y_train, weights)
                valid_scores = predict_scores(model, valid_df[feature_cols])
                ap = safe_ap(valid_df["target"].astype(int), valid_scores)
                auc = safe_auc(valid_df["target"].astype(int), valid_scores)
                score = ap if not math.isnan(ap) else -1.0
                if best is None or score > best["score"]:
                    best = {"model_name": model_name, "model": model, "score": score, "valid_ap": ap, "valid_auc": auc}
            except Exception as exc:
                eval_rows.append(
                    {
                        "section": "model_fit_error",
                        "expert": expert_name,
                        "bucket": bucket,
                        "model": model_name,
                        "error": str(exc),
                    }
                )
        if best is None:
            eval_rows.append({"section": "expert_eval", "expert": expert_name, "bucket": bucket, "status": "skipped", "reason": "no_model"})
            continue
        expert_models[expert_name] = best
        for split, df in candidates.items():
            app = applicable_mask(df, bucket)
            raw_col = f"{expert_name}_raw_score"
            guarded_col = f"{expert_name}_guarded_score"
            score_frames[split][raw_col] = np.nan
            score_frames[split][guarded_col] = np.nan
            if app.any():
                scores = predict_scores(best["model"], df.loc[app, feature_cols])
                score_frames[split].loc[app, raw_col] = scores
                score_frames[split].loc[app, guarded_col] = guard_score(scores, df.loc[app])
            eval_mask = app & pure_mask(df)
            y = df.loc[eval_mask, "target"].astype(int).to_numpy()
            raw = score_frames[split].loc[eval_mask, raw_col].to_numpy(dtype=float)
            guarded = score_frames[split].loc[eval_mask, guarded_col].to_numpy(dtype=float)
            eval_rows.append(
                {
                    "section": "expert_eval",
                    "expert": expert_name,
                    "bucket": bucket,
                    "model": best["model_name"],
                    "split": split,
                    "runs": int(eval_mask.sum()),
                    "positives": int(y.sum()) if len(y) else 0,
                    "raw_auc": safe_auc(y, raw),
                    "raw_ap": safe_ap(y, raw),
                    "guarded_auc": safe_auc(y, guarded),
                    "guarded_ap": safe_ap(y, guarded),
                    "status": "ok",
                }
            )
        feature_importances.extend(extract_feature_importance(best["model"], feature_cols, expert_name, best["model_name"]))

    rank_rows, rank_importance = fit_rank_ensemble(candidates, feature_cols, args, score_frames)
    eval_rows.extend(rank_rows)
    feature_importances.extend(rank_importance)
    return score_frames, pd.DataFrame(eval_rows), pd.DataFrame(feature_importances)


def candidate_score_base(df):
    keep = [
        "split",
        "run_id",
        "trace_id",
        "start_index",
        "end_index",
        "length",
        "length_bucket",
        "is_very_long",
        "road_points",
        "field_points",
        "road_ratio",
        "field_ratio",
        "run_state",
        "target",
        "hard_true_road",
        "pattern_type",
        "mixed_guard_excluded",
        "true_road_guard_score",
        "true_road_guard_strong",
        "field_pattern_guard_score",
        "field_pattern_guard_strong",
        "false_road_risk_score",
        "field_dense_stationary_score",
        "field_work_pattern_score",
        "road_linearity_score",
    ]
    keep = [col for col in keep if col in df.columns]
    return df[keep].copy()


def extract_feature_importance(model, feature_cols, expert_name, model_name):
    last = model.steps[-1][1] if hasattr(model, "steps") else model
    rows = []
    if hasattr(last, "feature_importances_"):
        values = last.feature_importances_
    elif hasattr(last, "coef_"):
        values = np.abs(last.coef_).reshape(-1)
    else:
        return rows
    for feature, importance in zip(feature_cols, values):
        rows.append(
            {
                "expert": expert_name,
                "model": model_name,
                "feature": feature,
                "importance": float(importance),
            }
        )
    return rows


def fit_rank_ensemble(candidates, feature_cols, args, score_frames):
    train = candidates["train"]
    valid = candidates["valid"]
    train_df = train.loc[pure_mask(train)].copy()
    valid_df = valid.loc[pure_mask(valid)].copy()
    rows = []
    importances = []
    if train_df.empty or len(set(train_df["target"].astype(int).tolist())) < 2 or valid_df.empty:
        rows.append({"section": "expert_eval", "expert": "fake_road_rank_ensemble", "bucket": "all", "status": "skipped", "reason": "insufficient_data"})
        return rows, importances
    fitted = []
    x_train = train_df[feature_cols]
    y_train = train_df["target"].astype(int).to_numpy()
    weights = sample_weights(train_df, "fake_road_rank_ensemble")
    for model_name, model in build_model_candidates(args.seed + 17, args.n_jobs):
        try:
            fit_with_weights(model, x_train, y_train, weights)
            fitted.append((model_name, model))
            importances.extend(extract_feature_importance(model, feature_cols, "fake_road_rank_ensemble", model_name))
        except Exception as exc:
            rows.append({"section": "model_fit_error", "expert": "fake_road_rank_ensemble", "model": model_name, "error": str(exc)})
    if not fitted:
        rows.append({"section": "expert_eval", "expert": "fake_road_rank_ensemble", "bucket": "all", "status": "skipped", "reason": "no_model"})
        return rows, importances
    for split, df in candidates.items():
        raw_scores = rank_average_scores(fitted, df[feature_cols])
        raw_col = "fake_road_rank_ensemble_raw_score"
        guarded_col = "fake_road_rank_ensemble_guarded_score"
        score_frames[split][raw_col] = raw_scores
        score_frames[split][guarded_col] = guard_score(raw_scores, df)
        eval_mask = pure_mask(df)
        y = df.loc[eval_mask, "target"].astype(int).to_numpy()
        rows.append(
            {
                "section": "expert_eval",
                "expert": "fake_road_rank_ensemble",
                "bucket": "all",
                "model": "rank_average",
                "split": split,
                "runs": int(eval_mask.sum()),
                "positives": int(y.sum()) if len(y) else 0,
                "raw_auc": safe_auc(y, score_frames[split].loc[eval_mask, raw_col]),
                "raw_ap": safe_ap(y, score_frames[split].loc[eval_mask, raw_col]),
                "guarded_auc": safe_auc(y, score_frames[split].loc[eval_mask, guarded_col]),
                "guarded_ap": safe_ap(y, score_frames[split].loc[eval_mask, guarded_col]),
                "status": "ok",
            }
        )
    return rows, importances


def rank_average_scores(fitted_models, x):
    if len(x) == 0:
        return np.array([], dtype=float)
    ranks = []
    for _model_name, model in fitted_models:
        scores = predict_scores(model, x)
        ranks.append(pd.Series(scores).rank(method="average", pct=True).to_numpy(dtype=float))
    return np.mean(np.vstack(ranks), axis=0)


def threshold_metrics(df, score_col, raw_col, threshold, bucket="all", mode="threshold", target_precision=None):
    app = applicable_mask(df, bucket)
    score = pd.to_numeric(df[score_col], errors="coerce")
    raw = pd.to_numeric(df[raw_col], errors="coerce") if raw_col in df.columns else score
    raw_above = app & (raw >= threshold)
    selected = app & (score >= threshold) & (~df["mixed_guard_excluded"].astype(bool))
    mixed_excluded = raw_above & df["mixed_guard_excluded"].astype(bool)
    true_guard_excluded = raw_above & (~selected) & df["true_road_guard_strong"].astype(bool)
    selected_df = df.loc[selected]
    possible_error_points = int(df.loc[app & (~df["mixed_guard_excluded"].astype(bool)), "field_points"].sum())
    fake_total = df.loc[app & (df["run_state"] == "fake_road")]
    selected_fake = selected_df[selected_df["run_state"] == "fake_road"]
    selected_runs = int(len(selected_df))
    fixed_points = int(selected_df["field_points"].sum()) if selected_runs else 0
    introduced_points = int(selected_df["road_points"].sum()) if selected_runs else 0
    fake_total_runs = int(len(fake_total))
    fake_total_points = int(fake_total["field_points"].sum()) if fake_total_runs else 0
    field_guard_selected = selected_df["field_pattern_guard_strong"].astype(bool) if selected_runs else pd.Series([], dtype=bool)
    return {
        "mode": mode,
        "target_precision": target_precision,
        "bucket": bucket,
        "threshold": float(threshold),
        "selected_runs": selected_runs,
        "selected_points": fixed_points + introduced_points,
        "precision_by_run": int((selected_df["run_state"] == "fake_road").sum()) / selected_runs if selected_runs else 0.0,
        "precision_by_point": fixed_points / max(fixed_points + introduced_points, 1),
        "recall_by_run": int(len(selected_fake)) / max(fake_total_runs, 1),
        "recall_by_point": fixed_points / max(possible_error_points, 1),
        "fixed_field_as_road_points": fixed_points,
        "introduced_road_as_field_points": introduced_points,
        "net_gain_points": fixed_points - introduced_points,
        "mixed_guard_excluded_runs": int(mixed_excluded.sum()),
        "mixed_guard_excluded_points": int((df.loc[mixed_excluded, "field_points"] + df.loc[mixed_excluded, "road_points"]).sum()) if mixed_excluded.any() else 0,
        "true_road_guard_excluded_runs": int(true_guard_excluded.sum()),
        "true_road_guard_excluded_points": int((df.loc[true_guard_excluded, "field_points"] + df.loc[true_guard_excluded, "road_points"]).sum()) if true_guard_excluded.any() else 0,
        "field_pattern_guard_selected_runs": int(field_guard_selected.sum()) if selected_runs else 0,
        "field_pattern_guard_selected_points": int((selected_df.loc[field_guard_selected, "field_points"] + selected_df.loc[field_guard_selected, "road_points"]).sum()) if selected_runs else 0,
    }


def select_valid_thresholds(valid_scores, expert, bucket):
    score_col = f"{expert}_guarded_score"
    raw_col = f"{expert}_raw_score"
    if score_col not in valid_scores.columns:
        return []
    app = applicable_mask(valid_scores, bucket)
    view = valid_scores.loc[app & pure_mask(valid_scores) & np.isfinite(pd.to_numeric(valid_scores[score_col], errors="coerce"))]
    if view.empty:
        return []
    scores = np.unique(np.quantile(view[score_col].to_numpy(dtype=float), np.linspace(0.0, 1.0, 151)))
    rows = []
    for target in PRECISION_TARGETS:
        best_meeting = None
        best_fallback = None
        for threshold in scores:
            metrics = threshold_metrics(valid_scores, score_col, raw_col, threshold, bucket=bucket, mode="threshold", target_precision=target)
            if metrics["selected_runs"] <= 0:
                continue
            meets = metrics["precision_by_point"] >= target
            meeting_objective = (
                metrics["fixed_field_as_road_points"],
                metrics["net_gain_points"],
                metrics["precision_by_point"],
                metrics["precision_by_run"],
            )
            fallback_objective = (
                metrics["precision_by_point"],
                metrics["net_gain_points"],
                metrics["fixed_field_as_road_points"],
                metrics["precision_by_run"],
            )
            if meets and (best_meeting is None or meeting_objective > best_meeting["_objective"]):
                best_meeting = {**metrics, "_objective": meeting_objective}
            if best_fallback is None or fallback_objective > best_fallback["_objective"]:
                best_fallback = {**metrics, "_objective": fallback_objective}
        best = best_meeting or best_fallback
        if best is None:
            best = threshold_metrics(valid_scores, score_col, raw_col, 1.1, bucket=bucket, mode="threshold", target_precision=target)
            best["_objective"] = (0, 0, 0, 0, 0)
        best["meets_target_on_valid"] = bool(best_meeting is not None)
        best.pop("_objective", None)
        best["expert"] = expert
        rows.append(best)
    return rows


def apply_threshold_rows(scores_df, valid_rows):
    out = []
    for row in valid_rows:
        expert = row["expert"]
        score_col = f"{expert}_guarded_score"
        raw_col = f"{expert}_raw_score"
        if score_col not in scores_df.columns:
            continue
        metrics = threshold_metrics(
            scores_df,
            score_col,
            raw_col,
            row["threshold"],
            bucket=row["bucket"],
            mode="threshold_from_valid",
            target_precision=row["target_precision"],
        )
        metrics["expert"] = expert
        metrics["meets_target_on_valid"] = row.get("meets_target_on_valid", np.nan)
        out.append(metrics)
    return out


def topk_rows(scores_df, experts):
    rows = []
    for expert, bucket in experts:
        score_col = f"{expert}_guarded_score"
        raw_col = f"{expert}_raw_score"
        if score_col not in scores_df.columns:
            continue
        buckets = ["all", "short", "mid", "long", "very_long"] if bucket == "all" else [bucket]
        for eval_bucket in buckets:
            app = applicable_mask(scores_df, eval_bucket) & (~scores_df["mixed_guard_excluded"].astype(bool))
            view = scores_df.loc[app & np.isfinite(pd.to_numeric(scores_df[score_col], errors="coerce"))].sort_values(score_col, ascending=False)
            if view.empty:
                continue
            for frac in TOPK_BY_BUCKET.get(eval_bucket, [0.05, 0.10, 0.20]):
                k = max(1, int(math.ceil(len(view) * frac)))
                threshold = float(view[score_col].iloc[k - 1])
                row = threshold_metrics(scores_df, score_col, raw_col, threshold, bucket=eval_bucket, mode=f"top_{int(frac * 100)}pct")
                row["expert"] = expert
                row["target_precision"] = np.nan
                rows.append(row)
    return rows


def build_threshold_outputs(score_frames):
    expert_buckets = EXPERT_SPECS + [("fake_road_rank_ensemble", "all")]
    valid_rows = []
    for expert, bucket in expert_buckets:
        valid_rows.extend(select_valid_thresholds(score_frames["valid"], expert, bucket))
    valid_topk = topk_rows(score_frames["valid"], expert_buckets)
    test_rows = apply_threshold_rows(score_frames["test"], valid_rows)
    test_topk = topk_rows(score_frames["test"], expert_buckets)
    return pd.DataFrame(valid_rows + valid_topk), pd.DataFrame(test_rows + test_topk)


def write_top_selected(scores_df, path, max_rows=400):
    score_cols = [col for col in scores_df.columns if col.endswith("_guarded_score")]
    out = scores_df.copy()
    if score_cols:
        out["best_guarded_score"] = out[score_cols].max(axis=1, skipna=True)
        out["best_expert"] = out[score_cols].idxmax(axis=1).str.replace("_guarded_score", "", regex=False)
    else:
        out["best_guarded_score"] = 0.0
        out["best_expert"] = ""
    keep_cols = [
        "split",
        "run_id",
        "trace_id",
        "start_index",
        "end_index",
        "length",
        "length_bucket",
        "is_very_long",
        "road_points",
        "field_points",
        "road_ratio",
        "field_ratio",
        "run_state",
        "pattern_type",
        "hard_true_road",
        "mixed_guard_excluded",
        "true_road_guard_score",
        "field_pattern_guard_score",
        "false_road_risk_score",
        "field_dense_stationary_score",
        "field_work_pattern_score",
        "road_linearity_score",
        "best_expert",
        "best_guarded_score",
    ]
    keep_cols += score_cols
    keep_cols = [col for col in keep_cols if col in out.columns]
    out.sort_values("best_guarded_score", ascending=False).head(max_rows)[keep_cols].to_csv(path, index=False)


def hard_true_road_cases(candidates, score_frames, path):
    frames = []
    for split, df in candidates.items():
        scores = score_frames[split]
        base = df.loc[df["hard_true_road"].astype(bool)].copy()
        if base.empty:
            continue
        score_cols = [col for col in scores.columns if col.endswith("_guarded_score")]
        cols = ["run_id", "best_guarded_score"]
        scores_view = scores[["run_id"] + score_cols].copy()
        scores_view["best_guarded_score"] = scores_view[score_cols].max(axis=1, skipna=True) if score_cols else 0.0
        merged = base.merge(scores_view[cols], on="run_id", how="left")
        frames.append(merged)
    if frames:
        out = pd.concat(frames, ignore_index=True).sort_values("best_guarded_score", ascending=False)
        keep = [
            "split",
            "run_id",
            "trace_id",
            "length",
            "road_points",
            "field_points",
            "road_ratio",
            "field_ratio",
            "stationary_rate",
            "local_density_1m_mean",
            "compactness",
            "pca_aspect_ratio",
            "trajectory_linearity",
            "pattern_type",
            "false_road_risk_score",
            "true_road_guard_score",
            "best_guarded_score",
        ]
        out[[col for col in keep if col in out.columns]].head(400).to_csv(path, index=False)
    else:
        pd.DataFrame().to_csv(path, index=False)


def read_v2_baseline(args):
    baseline = {
        "global_fake_road_test_ap": float("nan"),
        "long_fake_road_test_ap": float("nan"),
        "global_fake_road_valid_ap": float("nan"),
        "long_fake_road_valid_ap": float("nan"),
        "long_top20_point_precision": float("nan"),
    }
    path = Path(args.v2_summary)
    if path.exists():
        df = pd.read_csv(path)
        for split in ("valid", "test"):
            for expert, key in [
                ("global_fake_road_ensemble", f"global_fake_road_{split}_ap"),
                ("fake_road_long_expert", f"long_fake_road_{split}_ap"),
            ]:
                row = df[(df.get("expert") == expert) & (df.get("split") == split)]
                if not row.empty and "ap" in row.columns:
                    baseline[key] = float(row.iloc[0]["ap"])
    topk_path = Path(args.v2_topk)
    if topk_path.exists():
        df = pd.read_csv(topk_path)
        row = df[
            (df.get("expert") == "fake_road_long_expert")
            & (df.get("split") == "test")
            & (df.get("bucket") == "long")
            & (df.get("topk") == "top_20pct")
        ]
        if not row.empty:
            baseline["long_top20_point_precision"] = float(row.iloc[0].get("selected_point_precision", float("nan")))
    return baseline


def trace_overlap_summary(candidates):
    traces = {split: set(df["trace_id"].astype(str).unique().tolist()) for split, df in candidates.items()}
    return {
        "train_valid_overlap": len(traces.get("train", set()) & traces.get("valid", set())),
        "train_test_overlap": len(traces.get("train", set()) & traces.get("test", set())),
        "valid_test_overlap": len(traces.get("valid", set()) & traces.get("test", set())),
    }


def make_report(args, candidates, eval_df, valid_sweep, test_sweep, feature_importance, baseline):
    def best_ap(split, expert):
        rows = eval_df[(eval_df["section"] == "expert_eval") & (eval_df["split"] == split) & (eval_df["expert"] == expert)]
        if rows.empty:
            return float("nan"), float("nan")
        row = rows.iloc[0]
        return float(row.get("raw_ap", float("nan"))), float(row.get("guarded_ap", float("nan")))

    global_raw_ap, global_guarded_ap = best_ap("test", "fake_road_global_expert")
    rank_raw_ap, rank_guarded_ap = best_ap("test", "fake_road_rank_ensemble")
    long_raw_ap, long_guarded_ap = best_ap("test", "fake_road_long_expert")
    best_test_ap_row = eval_df[(eval_df["section"] == "expert_eval") & (eval_df["split"] == "test")].copy()
    if not best_test_ap_row.empty:
        best_test_ap_row["best_ap"] = best_test_ap_row[["raw_ap", "guarded_ap"]].max(axis=1)
        best_expert_row = best_test_ap_row.sort_values("best_ap", ascending=False).iloc[0]
    else:
        best_expert_row = {}

    threshold_test = test_sweep[test_sweep["mode"] == "threshold_from_valid"].copy() if not test_sweep.empty else pd.DataFrame()
    high_precision = threshold_test[threshold_test["precision_by_point"] >= 0.95] if not threshold_test.empty else pd.DataFrame()
    stable_high_precision = high_precision[high_precision.get("meets_target_on_valid", False).astype(bool)] if not high_precision.empty else pd.DataFrame()
    best_threshold = None
    if not stable_high_precision.empty:
        best_threshold = stable_high_precision.sort_values(["net_gain_points", "fixed_field_as_road_points"], ascending=False).iloc[0]
    elif not high_precision.empty:
        best_threshold = high_precision.sort_values(["net_gain_points", "fixed_field_as_road_points"], ascending=False).iloc[0]
    elif not threshold_test.empty:
        best_threshold = threshold_test.sort_values(["precision_by_point", "net_gain_points"], ascending=False).iloc[0]

    feature_lines = []
    if feature_importance is not None and not feature_importance.empty:
        top_features = (
            feature_importance.groupby("feature", as_index=False)["importance"].mean().sort_values("importance", ascending=False).head(12)
        )
        feature_lines = [f"- {row.feature}: {fmt(row.importance)}" for row in top_features.itertuples(index=False)]
    else:
        feature_lines = ["- NA"]

    hard_cases = candidates["test"][candidates["test"]["hard_true_road"].astype(bool)]
    overlap = trace_overlap_summary(candidates)
    enter = (
        (max(global_raw_ap, rank_raw_ap, global_guarded_ap, rank_guarded_ap) >= 0.90 or max(long_raw_ap, long_guarded_ap) >= 0.92)
        and best_threshold is not None
        and bool(best_threshold.get("meets_target_on_valid", False))
        and float(best_threshold["precision_by_point"]) >= 0.95
        and int(best_threshold["net_gain_points"]) > 0
    )
    if enter:
        recommendation = "建议进入 FRG_MoE_CONSERVE_v3，优先使用 valid 阈值下 test 仍保持高精度的配置。"
    else:
        recommendation = "暂不建议直接进入 FRG_MoE_CONSERVE_v3；fake_road 已接近上限，若需要保守修正可优先沿用 TFSV-LC_MoE_AUDIT_v2 的 fake_road 结果或只选极高置信 TopK。"

    best_threshold_line = "NA"
    if best_threshold is not None:
        best_threshold_line = (
            f"{best_threshold['expert']} / {best_threshold['bucket']} / target={best_threshold['target_precision']}，"
            f"valid达标={best_threshold.get('meets_target_on_valid', 'NA')}，"
            f"test point precision={fmt(best_threshold['precision_by_point'])}，"
            f"fixed={int(best_threshold['fixed_field_as_road_points'])}，"
            f"introduced={int(best_threshold['introduced_road_as_field_points'])}，"
            f"net_gain={int(best_threshold['net_gain_points'])}"
        )

    lines = [
        "# FRG-MoE_AUDIT_v3 报告",
        "",
        "本次只做 fake_road 审计：没有训练主模型，没有修改最终预测，也没有处理 fake_field。",
        "",
        "## 输入",
        f"- 基线：`{BASELINE_GROUP}`",
        f"- predicted_runs：`{args.run_dir}`",
        f"- 输出目录：`{args.output_dir}`",
        f"- trace 泄漏检查：{overlap}",
        "",
        "## 与 TFSV-LC_MoE_AUDIT_v2 对比",
        f"- v2 global fake_road test AP：{fmt(baseline['global_fake_road_test_ap'])}",
        f"- FRG global raw/guarded test AP：{fmt(global_raw_ap)} / {fmt(global_guarded_ap)}",
        f"- FRG rank ensemble raw/guarded test AP：{fmt(rank_raw_ap)} / {fmt(rank_guarded_ap)}",
        f"- v2 long fake_road test AP：{fmt(baseline['long_fake_road_test_ap'])}",
        f"- FRG long raw/guarded test AP：{fmt(long_raw_ap)} / {fmt(long_guarded_ap)}",
        f"- v2 long top20 point precision：{fmt(baseline['long_top20_point_precision'])}",
        "",
        "## 最有价值专家",
        f"- test AP 最高专家：{best_expert_row.get('expert', 'NA')}，raw_ap={fmt(best_expert_row.get('raw_ap', float('nan')))}，guarded_ap={fmt(best_expert_row.get('guarded_ap', float('nan')))}",
        f"- high precision 阈值最佳配置：{best_threshold_line}",
        "",
        "## 关键特征",
        *feature_lines,
        "",
        "## Guard 审计",
        f"- test hard_true_road 候选数：{len(hard_cases)}",
    ]
    if best_threshold is not None:
        lines.extend(
            [
                f"- MixedGuard 排除点数：{int(best_threshold['mixed_guard_excluded_points'])}",
                f"- TrueRoadGuard 排除点数：{int(best_threshold['true_road_guard_excluded_points'])}",
                f"- FieldPatternGuard 选中点数：{int(best_threshold['field_pattern_guard_selected_points'])}",
            ]
        )
    lines.extend(
        [
            "",
            "## 必答结论",
            f"1. fake_road 是否比 v2 更可分：global AP 对比见上；本次最佳 test AP 为 {fmt(best_expert_row.get('best_ap', float('nan')))}。",
            f"2. global fake_road AP 是否提升：v2={fmt(baseline['global_fake_road_test_ap'])}，FRG global/rank 见上。",
            f"3. long fake_road AP 是否提升：v2={fmt(baseline['long_fake_road_test_ap'])}，FRG long={fmt(max(long_raw_ap, long_guarded_ap))}。",
            "4. short/mid/long/very_long 的价值见 `results/FRG_MoE_AUDIT_v3_summary.csv` 中各 expert_eval 行。",
            "5. 最重要特征见 `feature_importance_fake_road.csv` 和本报告关键特征。",
            "6. 最容易误伤的 true_road 见 `hard_true_road_false_positive_cases.csv`。",
            "7. MixedGuard 已在阈值评估中强制排除 mixed_transition。",
            "8. TrueRoadGuard 通过 raw>=threshold 但 guarded<threshold 的点数统计其误伤压制作用。",
            "9. valid 阈值在 test 的稳定性见 `threshold_sweep_test.csv`，没有用 test 选阈值。",
            f"10. 是否建议进入 FRG_MoE_CONSERVE_v3：{recommendation}",
            f"11. 若进入修正，推荐配置：{best_threshold_line}",
            "",
            "## 输出文件",
            f"- `diagnostics/{AUDIT_NAME}/`",
            f"- `results/{AUDIT_NAME}_summary.csv`",
            f"- `analysis_packs/{AUDIT_NAME}_for_chatgpt.zip`",
        ]
    )
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
    splits = [item.strip() for item in args.splits.split(",") if item.strip()]
    candidates = {}
    for split in splits:
        print(f"FRG-MoE loading candidates split={split}", flush=True)
        candidates[split] = load_runs(split, args)
        candidates[split].to_csv(Path(args.output_dir) / f"fake_road_candidates_{split}.csv", index=False)
        print(f"split={split} pred_road_candidates={len(candidates[split])}", flush=True)

    score_frames, eval_df, feature_importance = fit_experts(candidates, args)
    for split in splits:
        if split in score_frames:
            score_frames[split].to_csv(Path(args.output_dir) / f"expert_scores_{split}.csv", index=False)
    valid_sweep, test_sweep = build_threshold_outputs(score_frames)
    valid_sweep.insert(0, "split", "valid")
    test_sweep.insert(0, "split", "test")
    valid_sweep.to_csv(Path(args.output_dir) / "threshold_sweep_valid.csv", index=False)
    test_sweep.to_csv(Path(args.output_dir) / "threshold_sweep_test.csv", index=False)
    write_top_selected(score_frames["valid"], Path(args.output_dir) / "top_selected_fake_road_valid.csv")
    write_top_selected(score_frames["test"], Path(args.output_dir) / "top_selected_fake_road_test.csv")
    hard_true_road_cases(candidates, score_frames, Path(args.output_dir) / "hard_true_road_false_positive_cases.csv")
    if feature_importance.empty:
        feature_importance = pd.DataFrame(columns=["expert", "model", "feature", "importance"])
    feature_importance.sort_values(["expert", "importance"], ascending=[True, False]).to_csv(
        Path(args.output_dir) / "feature_importance_fake_road.csv", index=False
    )

    summary_parts = [eval_df]
    threshold_summary = pd.concat([valid_sweep, test_sweep], ignore_index=True)
    threshold_summary.insert(0, "section", "threshold_sweep")
    summary_parts.append(threshold_summary)
    summary = pd.concat(summary_parts, ignore_index=True, sort=False)
    summary.to_csv(args.summary_csv, index=False)
    baseline = read_v2_baseline(args)
    make_report(args, candidates, eval_df, valid_sweep, test_sweep, feature_importance, baseline)
    metadata = {
        "audit_name": AUDIT_NAME,
        "baseline_group": BASELINE_GROUP,
        "run_dir": args.run_dir,
        "splits": splits,
        "experts": [name for name, _bucket in EXPERT_SPECS] + ["fake_road_rank_ensemble"],
        "trace_overlap": trace_overlap_summary(candidates),
    }
    (Path(args.output_dir) / "audit_metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    write_pack(args)
    print(f"summary: {args.summary_csv}", flush=True)
    print(f"report: {args.report_path}", flush=True)
    print(f"pack: {args.pack_path}", flush=True)


if __name__ == "__main__":
    main()
