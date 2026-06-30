import argparse
import importlib.util
import json
import math
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def load_conserve_module():
    module_path = PROJECT_ROOT / "scripts" / "frg_moe_conserve_v3.py"
    spec = importlib.util.spec_from_file_location("frg_moe_conserve_v3_base", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BASE = load_conserve_module()

EXPERIMENT_NAME = "FRG_MoE_CONSERVE_v3_PLUS_AUDIT"
BASELINE_GROUP = "PT2G_MSC_v1_time_fixed_strict_40pre"
BASELINE_TEST_REF = {
    "macro_f1": 0.886494,
    "road_as_field": 3876,
    "field_as_road": 3423,
    "pred_road_rate": 0.212288,
}
STRONG_TEST_REF = {
    "macro_f1": 0.902254,
    "road_f1": 0.845296,
    "field_f1": 0.959213,
    "road_as_field": 3879,
    "field_as_road": 2277,
    "selected_point_precision": 0.997389,
}
FOCUS_TRACE_NAMES = [
    "wheat_1_harvestor_45",
    "wheat_1_harvestor_58",
    "wheat_1_harvestor_80",
    "wheat_1_harvestor_50",
    "wheat_1_harvestor_124",
]
RULE_COLUMNS = {
    "global": ("fake_road_global_expert_guarded_score", "all"),
    "long": ("fake_road_long_expert_raw_score", "long"),
    "very_long": ("fake_road_very_long_expert_raw_score", "very_long"),
}
TOPK_CONFIGS = {
    "REPRO_STRONG": {"fractions": {"global": 0.20, "long": 0.30, "very_long": 0.30}},
    "PLUS_1": {"fractions": {"global": 0.25, "long": 0.35, "very_long": 0.35}},
    "PLUS_2": {"fractions": {"global": 0.30, "long": 0.40, "very_long": 0.40}},
    "PLUS_3": {"fractions": {"global": 0.35, "long": 0.45, "very_long": 0.45}},
}
THRESHOLD_CONFIGS = {
    "VALID_PLUS_SAFE": "safe",
    "VALID_PLUS_BALANCED": "balanced",
    "VALID_PLUS_MAXGAIN": "maxgain",
}
SAFETY_LIMITS = {
    "selected_point_precision_min": 0.98,
    "road_as_field_delta_vs_baseline_max": 50,
    "road_as_field_delta_vs_strong_max": 50,
    "pred_road_rate_drop_max": 0.014,
    "macro_f1_gain_vs_strong_min": 0.0005,
}


def parse_args():
    parser = argparse.ArgumentParser(description="FRG_MoE_CONSERVE_v3_PLUS_AUDIT fake-road expansion audit.")
    parser.add_argument("--point_dir", default=f"diagnostics/{BASELINE_GROUP}_error_analysis")
    parser.add_argument("--audit_dir", default="diagnostics/FRG_MoE_AUDIT_v3")
    parser.add_argument("--conserve_dir", default="diagnostics/FRG_MoE_CONSERVE_v3")
    parser.add_argument("--conserve_summary", default="results/FRG_MoE_CONSERVE_v3_summary.csv")
    parser.add_argument("--output_dir", default=f"diagnostics/{EXPERIMENT_NAME}")
    parser.add_argument("--summary_csv", default=f"results/{EXPERIMENT_NAME}_summary.csv")
    parser.add_argument("--report_path", default=f"analysis/{EXPERIMENT_NAME}_report.md")
    parser.add_argument("--pack_path", default=f"analysis_packs/{EXPERIMENT_NAME}_for_chatgpt.zip")
    parser.add_argument("--splits", default="valid,test", help="Comma-separated splits to score. Use train,valid,test for train backfill.")
    return parser.parse_args()


def ensure_dirs(args):
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.summary_csv).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report_path).parent.mkdir(parents=True, exist_ok=True)
    Path(args.pack_path).parent.mkdir(parents=True, exist_ok=True)


def finite_score(series):
    return np.isfinite(pd.to_numeric(series, errors="coerce"))


def point_key(df):
    return list(zip(df["trace_id"].astype(str), df["global_index"].astype(int)))


def eligible_mask(scores):
    pred_road = pd.to_numeric(scores.get("pred_class", 0), errors="coerce").fillna(0).astype(int).eq(0)
    return pred_road & (~scores["mixed_guard_excluded"].astype(bool)) & (~scores["true_road_guard_strong"].astype(bool))


def collect_topk_candidates(scores, config_name, config):
    selected = {}
    eligible = eligible_mask(scores)
    for rule_name, fraction in config["fractions"].items():
        score_col, bucket = RULE_COLUMNS[rule_name]
        if score_col not in scores.columns:
            continue
        mask = eligible & BASE.applicable_mask(scores, bucket) & finite_score(scores[score_col])
        view = scores.loc[mask].copy()
        if view.empty:
            continue
        view["_rule_score"] = pd.to_numeric(view[score_col], errors="coerce")
        view = view.sort_values("_rule_score", ascending=False)
        k = max(1, int(math.ceil(len(view) * fraction)))
        reason = f"{rule_name}_{score_col}_top{int(fraction * 100)}"
        for row in view.head(k).to_dict("records"):
            run_id = str(row["run_id"])
            score = float(row["_rule_score"])
            if run_id not in selected:
                selected[run_id] = dict(row)
                selected[run_id]["selection_score"] = score
                selected[run_id]["selection_reasons"] = [reason]
                selected[run_id]["score_column"] = score_col
            else:
                selected[run_id]["selection_score"] = max(float(selected[run_id]["selection_score"]), score)
                selected[run_id]["selection_reasons"].append(reason)
    rows = list(selected.values())
    for row in rows:
        row["selection_reason"] = "|".join(sorted(set(row.pop("selection_reasons", []))))
        row["selected_config"] = config_name
    return pd.DataFrame(rows)


def collect_threshold_candidates(scores, config_name, thresholds):
    selected = {}
    eligible = eligible_mask(scores)
    for rule_name, threshold in thresholds.items():
        score_col, bucket = RULE_COLUMNS[rule_name]
        if score_col not in scores.columns or not np.isfinite(threshold):
            continue
        mask = eligible & BASE.applicable_mask(scores, bucket) & finite_score(scores[score_col])
        view = scores.loc[mask].copy()
        if view.empty:
            continue
        view["_rule_score"] = pd.to_numeric(view[score_col], errors="coerce")
        view = view[view["_rule_score"] >= float(threshold)].sort_values("_rule_score", ascending=False)
        reason = f"{rule_name}_{score_col}>={threshold:.6f}"
        for row in view.to_dict("records"):
            run_id = str(row["run_id"])
            score = float(row["_rule_score"])
            if run_id not in selected:
                selected[run_id] = dict(row)
                selected[run_id]["selection_score"] = score
                selected[run_id]["selection_reasons"] = [reason]
                selected[run_id]["score_column"] = score_col
            else:
                selected[run_id]["selection_score"] = max(float(selected[run_id]["selection_score"]), score)
                selected[run_id]["selection_reasons"].append(reason)
    rows = list(selected.values())
    for row in rows:
        row["selection_reason"] = "|".join(sorted(set(row.pop("selection_reasons", []))))
        row["selected_config"] = config_name
    return pd.DataFrame(rows)


def select_points_with_plus_safety(points, candidate_rows, config_name, baseline_metrics, enforce_safety=True):
    base_pred = points["pred_label"].to_numpy(dtype=int)
    labels = points["true_label"].to_numpy(dtype=int)
    total_points = len(points)
    changed = {}
    selected_segments = []
    if candidate_rows.empty:
        return changed, pd.DataFrame()
    candidate_rows = candidate_rows.sort_values("selection_score", ascending=False)
    introduced_so_far = 0
    for row in candidate_rows.to_dict("records"):
        point_rows = []
        if isinstance(row.get("_point_rows"), list):
            for idx in row["_point_rows"]:
                if idx not in changed and base_pred[idx] == 0:
                    point_rows.append(idx)
        if not point_rows:
            continue
        selected_points = len(point_rows)
        fixed = int(sum(1 for idx in point_rows if labels[idx] == 1))
        introduced = int(sum(1 for idx in point_rows if labels[idx] == 0))
        new_changed_count = len(changed) + selected_points
        pred_drop = new_changed_count / max(total_points, 1)
        if enforce_safety:
            if introduced_so_far + introduced > SAFETY_LIMITS["road_as_field_delta_vs_baseline_max"]:
                continue
            if pred_drop > SAFETY_LIMITS["pred_road_rate_drop_max"]:
                continue
        for idx in point_rows:
            changed[idx] = {
                "selected_config": config_name,
                "candidate_id": str(row["run_id"]),
                "fake_road_score": float(row.get("selection_score", np.nan)),
                "score_column": row.get("score_column", ""),
            }
        introduced_so_far += introduced
        selected_segments.append(
            {
                "selected_config": config_name,
                "split": points["split"].iloc[0],
                "run_id": str(row["run_id"]),
                "trace_id": str(row["trace_id"]),
                "start_index": row.get("start_index", np.nan),
                "end_index": row.get("end_index", np.nan),
                "length": row.get("length", np.nan),
                "length_bucket": row.get("length_bucket", ""),
                "run_state": row.get("run_state", ""),
                "target": row.get("target", np.nan),
                "selection_score": float(row.get("selection_score", np.nan)),
                "selection_reason": row.get("selection_reason", ""),
                "score_column": row.get("score_column", ""),
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


def summarize_config(points, final_df, selected_segments, config_name, config_kind, baseline_metrics, strong_metrics):
    labels = final_df["label"].to_numpy(dtype=int)
    base_pred = final_df["base_pred"].to_numpy(dtype=int)
    final_pred = final_df["final_pred"].to_numpy(dtype=int)
    metrics = BASE.compute_metrics(labels, final_pred)
    selected_points = int(final_df["changed_by_frg"].sum())
    fixed = int(((labels == 1) & (base_pred == 0) & (final_pred == 1)).sum())
    introduced = int(((labels == 0) & (base_pred == 0) & (final_pred == 1)).sum())
    selected_runs = int(selected_segments["run_id"].nunique()) if not selected_segments.empty else 0
    selected_run_precision = float((selected_segments["fixed_field_as_road"] > selected_segments["introduced_road_as_field"]).mean()) if selected_runs else 0.0
    selected_point_precision = fixed / max(fixed + introduced, 1)
    pred_road_rate_drop = baseline_metrics["pred_road_rate"] - metrics["pred_road_rate"]
    macro_gain_vs_strong = metrics["macro_f1"] - strong_metrics["macro_f1"]
    road_delta_vs_baseline = metrics["road_as_field"] - baseline_metrics["road_as_field"]
    road_delta_vs_strong = metrics["road_as_field"] - strong_metrics["road_as_field"]
    safety_pass = (
        selected_point_precision >= SAFETY_LIMITS["selected_point_precision_min"]
        and road_delta_vs_baseline <= SAFETY_LIMITS["road_as_field_delta_vs_baseline_max"]
        and road_delta_vs_strong <= SAFETY_LIMITS["road_as_field_delta_vs_strong_max"]
        and pred_road_rate_drop <= SAFETY_LIMITS["pred_road_rate_drop_max"] + 1e-12
    )
    upgrade_value = safety_pass and macro_gain_vs_strong >= SAFETY_LIMITS["macro_f1_gain_vs_strong_min"]
    return {
        "config": config_name,
        "split": str(final_df["split"].iloc[0]) if len(final_df) else "",
        "config_kind": config_kind,
        **metrics,
        "selected_runs": selected_runs,
        "selected_points": selected_points,
        "fixed_field_as_road": fixed,
        "introduced_road_as_field": introduced,
        "net_fixed_points": fixed - introduced,
        "selected_point_precision": selected_point_precision,
        "selected_run_precision": selected_run_precision,
        "pred_road_rate_drop": pred_road_rate_drop,
        "macro_f1_delta_vs_baseline": metrics["macro_f1"] - baseline_metrics["macro_f1"],
        "macro_f1_delta_vs_strong": macro_gain_vs_strong,
        "field_as_road_delta_vs_baseline": metrics["field_as_road"] - baseline_metrics["field_as_road"],
        "field_as_road_delta_vs_strong": metrics["field_as_road"] - strong_metrics["field_as_road"],
        "road_as_field_delta_vs_baseline": road_delta_vs_baseline,
        "road_as_field_delta_vs_strong": road_delta_vs_strong,
        "safety_pass": safety_pass,
        "upgrade_value": upgrade_value,
    }


def evaluate_config(points, candidate_rows, config_name, config_kind, baseline_metrics, strong_metrics, output_dir, write_predictions=True):
    changed, selected_segments = select_points_with_plus_safety(points, candidate_rows, config_name, baseline_metrics)
    final_df = BASE.make_final_predictions(points, changed, config_name)
    if write_predictions:
        final_df.to_csv(Path(output_dir) / f"final_predictions_{points['split'].iloc[0]}_{config_name}.csv", index=False)
    summary = summarize_config(points, final_df, selected_segments, config_name, config_kind, baseline_metrics, strong_metrics)
    return summary, selected_segments, final_df, changed


def threshold_candidates_for_rule(scores, rule_name):
    score_col, bucket = RULE_COLUMNS[rule_name]
    if score_col not in scores.columns:
        return np.array([float("inf")])
    mask = eligible_mask(scores) & BASE.applicable_mask(scores, bucket) & finite_score(scores[score_col])
    values = pd.to_numeric(scores.loc[mask, score_col], errors="coerce").dropna().to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.array([float("inf")])
    quantiles = np.array([0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.925, 0.95, 0.975, 0.99, 1.0])
    thresholds = np.unique(np.quantile(values, quantiles))
    thresholds = np.sort(thresholds)[::-1]
    return np.concatenate([thresholds, np.array([float("inf")])])


def search_valid_thresholds(valid_points, valid_scores, baseline_metrics, strong_metrics):
    candidates = []
    rule_thresholds = {rule: threshold_candidates_for_rule(valid_scores, rule) for rule in RULE_COLUMNS}
    for global_thr in rule_thresholds["global"]:
        for long_thr in rule_thresholds["long"]:
            for very_thr in rule_thresholds["very_long"]:
                thresholds = {"global": global_thr, "long": long_thr, "very_long": very_thr}
                selected = collect_threshold_candidates(valid_scores, "VALID_SEARCH", thresholds)
                summary, _segments, _final, _changed = evaluate_config(
                    valid_points,
                    selected,
                    "VALID_SEARCH",
                    "valid_threshold_search",
                    baseline_metrics,
                    strong_metrics,
                    output_dir="/tmp",
                    write_predictions=False,
                )
                if summary["selected_points"] <= 0:
                    continue
                if not summary["safety_pass"]:
                    continue
                candidates.append({**summary, "thresholds": thresholds})
    if not candidates:
        return {name: {"thresholds": {"global": float("inf"), "long": float("inf"), "very_long": float("inf")}, "valid_summary": None} for name in THRESHOLD_CONFIGS}
    def pick(mode):
        if mode == "safe":
            return max(candidates, key=lambda x: (x["selected_point_precision"], x["macro_f1"], x["net_fixed_points"]))
        if mode == "balanced":
            return max(candidates, key=lambda x: (x["macro_f1"], x["selected_point_precision"], x["net_fixed_points"]))
        return max(candidates, key=lambda x: (x["net_fixed_points"], x["macro_f1"], x["selected_point_precision"]))
    return {
        name: {"thresholds": pick(mode)["thresholds"], "valid_summary": pick(mode)}
        for name, mode in THRESHOLD_CONFIGS.items()
    }


def load_conserve_strong_metrics(args, points):
    path = Path(args.conserve_summary)
    if path.exists():
        df = pd.read_csv(path)
        strong = df[(df["config"].eq("STRONG")) & (df["split"].eq("test"))]
        if not strong.empty:
            row = strong.iloc[0].to_dict()
            return {key: row[key] for key in ["accuracy", "macro_f1", "road_f1", "field_f1", "road_as_field", "field_as_road", "pred_road_rate", "pred_field_rate"] if key in row}
    strong_path = Path(args.conserve_dir) / "final_predictions_test_STRONG.csv"
    if strong_path.exists():
        df = pd.read_csv(strong_path)
        return BASE.compute_metrics(df["label"], df["final_pred"])
    return {
        **BASE.compute_metrics(points["test"]["true_label"], points["test"]["pred_label"]),
        **{key: value for key, value in STRONG_TEST_REF.items() if key in ["macro_f1", "road_as_field", "field_as_road", "pred_road_rate"]},
    }


def make_extra_over_strong(final_dfs, output_dir):
    if "test_REPRO_STRONG" not in final_dfs:
        return pd.DataFrame()
    strong_df = final_dfs["test_REPRO_STRONG"]
    strong_changed = set(point_key(strong_df[strong_df["changed_by_frg"].astype(bool)]))
    rows = []
    for key, df in final_dfs.items():
        if not key.startswith("test_"):
            continue
        config = key.replace("test_", "")
        if config == "REPRO_STRONG":
            continue
        changed = df[df["changed_by_frg"].astype(bool)].copy()
        if changed.empty:
            continue
        changed["_point_key"] = point_key(changed)
        extra = changed[~changed["_point_key"].isin(strong_changed)].copy()
        if extra.empty:
            continue
        extra["config"] = config
        extra["is_fixed_field_as_road"] = extra["label"].astype(int).eq(1)
        extra["is_introduced_road_as_field"] = extra["label"].astype(int).eq(0)
        rows.append(extra.drop(columns=["_point_key"]))
    out = pd.concat(rows, ignore_index=True, sort=False) if rows else pd.DataFrame()
    out.to_csv(Path(output_dir) / "extra_selected_over_STRONG_test.csv", index=False)
    return out


def make_trace_gain_loss(final_dfs, selected_segments, output_dir):
    rows = []
    strong_changed = set()
    if "test_REPRO_STRONG" in final_dfs:
        strong_changed = set(point_key(final_dfs["test_REPRO_STRONG"][final_dfs["test_REPRO_STRONG"]["changed_by_frg"].astype(bool)]))
    for key, df in final_dfs.items():
        if not key.startswith("test_"):
            continue
        config = key.replace("test_", "")
        tmp = df.copy()
        tmp["_changed"] = tmp["changed_by_frg"].astype(bool)
        tmp["_fixed"] = tmp["_changed"] & tmp["label"].astype(int).eq(1)
        tmp["_introduced"] = tmp["_changed"] & tmp["label"].astype(int).eq(0)
        tmp["_field_as_road"] = tmp["label"].astype(int).eq(1) & tmp["final_pred"].astype(int).eq(0)
        tmp["_road_as_field"] = tmp["label"].astype(int).eq(0) & tmp["final_pred"].astype(int).eq(1)
        tmp["_point_key"] = point_key(tmp)
        tmp["_extra_over_strong"] = tmp["_changed"] & (~tmp["_point_key"].isin(strong_changed))
        grouped = tmp.groupby("trace_id", as_index=False).agg(
            selected_points=("_changed", "sum"),
            fixed_field_as_road=("_fixed", "sum"),
            introduced_road_as_field=("_introduced", "sum"),
            field_as_road_final=("_field_as_road", "sum"),
            road_as_field_final=("_road_as_field", "sum"),
            extra_points_over_strong=("_extra_over_strong", "sum"),
        )
        grouped["selected_config"] = config
        grouped["net_fixed_points"] = grouped["fixed_field_as_road"] - grouped["introduced_road_as_field"]
        grouped["focus_trace"] = grouped["trace_id"].astype(str).apply(lambda v: any(name in v for name in FOCUS_TRACE_NAMES))
        rows.append(grouped)
    out = pd.concat(rows, ignore_index=True, sort=False) if rows else pd.DataFrame()
    if not out.empty:
        out = out.sort_values(["selected_config", "net_fixed_points"], ascending=[True, False])
    out.to_csv(Path(output_dir) / "trace_gain_loss_test.csv", index=False)
    return out


def make_trace_focus(trace_gain_loss, output_dir):
    if trace_gain_loss.empty:
        out = pd.DataFrame()
    else:
        focus = trace_gain_loss[trace_gain_loss["focus_trace"].astype(bool)].copy()
        residual_top = trace_gain_loss[trace_gain_loss["selected_config"].eq("REPRO_STRONG")].sort_values("field_as_road_final", ascending=False).head(10)
        out = pd.concat([focus, residual_top], ignore_index=True, sort=False).drop_duplicates(["selected_config", "trace_id"])
    out.to_csv(Path(output_dir) / "trace_focus_test.csv", index=False)
    return out


def residual_after_strong(points, scores, strong_changed):
    labels = points["true_label"].to_numpy(dtype=int)
    base_pred = points["pred_label"].to_numpy(dtype=int)
    residual = {idx for idx in range(len(points)) if labels[idx] == 1 and base_pred[idx] == 0 and idx not in strong_changed}
    eligible_rows = set()
    guard_rows = set()
    for row in scores.to_dict("records"):
        rows = row.get("_point_rows", [])
        if not isinstance(rows, list):
            continue
        if eligible_mask(pd.DataFrame([row])).iloc[0]:
            eligible_rows.update(rows)
        elif bool(row.get("mixed_guard_excluded", False)) or bool(row.get("true_road_guard_strong", False)):
            guard_rows.update(rows)
    in_eligible = residual & eligible_rows
    in_guard = residual & guard_rows
    not_candidate = residual - eligible_rows - guard_rows
    return {
        "strong_remaining_field_as_road_points": len(residual),
        "remaining_in_eligible_candidate_points": len(in_eligible),
        "remaining_guard_excluded_points": len(in_guard),
        "remaining_not_candidate_points": len(not_candidate),
        "remaining_not_candidate_rate": len(not_candidate) / max(len(residual), 1),
    }


def write_report(args, summary_df, extra_df, trace_gain_loss, trace_focus, residual_stats, threshold_infos, repro_ok):
    test = summary_df[summary_df["split"].eq("test")].copy()
    baseline = test[test["config"].eq("BASELINE")].iloc[0]
    strong = test[test["config"].eq("REPRO_STRONG")].iloc[0]
    plus = test[test["config"].isin(["PLUS_1", "PLUS_2", "PLUS_3", "VALID_PLUS_SAFE", "VALID_PLUS_BALANCED", "VALID_PLUS_MAXGAIN"])].copy()
    upgradeable = plus[plus["upgrade_value"].astype(bool)] if not plus.empty else pd.DataFrame()
    if not upgradeable.empty:
        recommended = upgradeable.sort_values(["macro_f1", "net_fixed_points"], ascending=False).iloc[0]
        rec = f"升级到 {recommended['config']}"
        reason = (
            f"macro-F1={recommended['macro_f1']:.6f}，相比 STRONG 提升 {recommended['macro_f1_delta_vs_strong']:.6f}，"
            f"selected_precision={recommended['selected_point_precision']:.6f}，road_as_field 相比 STRONG 增加 {int(recommended['road_as_field_delta_vs_strong'])}。"
        )
    else:
        best_plus = plus.sort_values("macro_f1", ascending=False).iloc[0] if not plus.empty else None
        if best_plus is not None and best_plus["macro_f1_delta_vs_strong"] < SAFETY_LIMITS["macro_f1_gain_vs_strong_min"]:
            rec = "继续使用 STRONG"
            reason = f"最佳 PLUS 相比 STRONG 的 macro-F1 提升只有 {best_plus['macro_f1_delta_vs_strong']:.6f}，低于 0.0005。"
        else:
            rec = "停止 fake_road 优化，转向 fake_field / RSC"
            reason = "没有 PLUS 配置同时满足精度、安全和有效提升标准。"
    best_test = test[test["config"].ne("BASELINE")].sort_values("macro_f1", ascending=False).iloc[0]
    extra_summary = pd.DataFrame()
    if not extra_df.empty:
        extra_summary = extra_df.groupby("config", as_index=False).agg(
            extra_points=("changed_by_frg", "size"),
            extra_fixed_field_as_road=("is_fixed_field_as_road", "sum"),
            extra_introduced_road_as_field=("is_introduced_road_as_field", "sum"),
        )
    lines = [
        "# FRG_MoE_CONSERVE_v3_PLUS_AUDIT 报告",
        "",
        "本次只做 fake_road 进一步扩展审计：没有训练主模型，没有修改原始数据，没有处理 fake_field。",
        "",
        "## STRONG 复现",
        f"- 复现是否接近既有 STRONG：{repro_ok}",
        f"- REPRO_STRONG test macro-F1={strong['macro_f1']:.6f}, road_as_field={int(strong['road_as_field'])}, field_as_road={int(strong['field_as_road'])}, precision={strong['selected_point_precision']:.6f}",
        f"- 既有参考 macro-F1={STRONG_TEST_REF['macro_f1']:.6f}, road_as_field={STRONG_TEST_REF['road_as_field']}, field_as_road={STRONG_TEST_REF['field_as_road']}, precision={STRONG_TEST_REF['selected_point_precision']:.6f}",
        "",
        "## Test 结果",
    ]
    for row in test.sort_values("macro_f1", ascending=False).to_dict("records"):
        lines.append(
            f"- {row['config']}: macro-F1={row['macro_f1']:.6f}, road-F1={row['road_f1']:.6f}, field-F1={row['field_f1']:.6f}, "
            f"road_as_field={int(row['road_as_field'])}, field_as_road={int(row['field_as_road'])}, "
            f"pred_road_rate={row['pred_road_rate']:.6f}, precision={row['selected_point_precision']:.6f}, "
            f"gain_vs_STRONG={row['macro_f1_delta_vs_strong']:.6f}, safety={row['safety_pass']}, upgrade={row['upgrade_value']}"
        )
    lines += [
        "",
        "## PLUS 相比 STRONG 的额外修正",
    ]
    if extra_summary.empty:
        lines.append("- 没有配置在 STRONG 之外额外修正点。")
    else:
        for row in extra_summary.sort_values("extra_points", ascending=False).to_dict("records"):
            lines.append(
                f"- {row['config']}: extra_points={int(row['extra_points'])}, "
                f"extra_fixed={int(row['extra_fixed_field_as_road'])}, extra_introduced={int(row['extra_introduced_road_as_field'])}"
            )
    lines += [
        "",
        "## STRONG 后剩余 fake_road 可分性",
        f"- remaining field_as_road={residual_stats['strong_remaining_field_as_road_points']}",
        f"- 仍在可修候选段中的点={residual_stats['remaining_in_eligible_candidate_points']}",
        f"- 被 guard 排除的点={residual_stats['remaining_guard_excluded_points']}",
        f"- 不在当前候选段中的点={residual_stats['remaining_not_candidate_points']}，占比={residual_stats['remaining_not_candidate_rate']:.4f}",
        "",
        "## valid 阈值搜索",
    ]
    for name, info in threshold_infos.items():
        th = info["thresholds"]
        valid = info.get("valid_summary")
        if valid is None:
            lines.append(f"- {name}: 未找到满足 valid 安全约束的阈值。")
        else:
            lines.append(
                f"- {name}: global={th['global']:.6f}, long={th['long']:.6f}, very_long={th['very_long']:.6f}, "
                f"valid macro-F1={valid['macro_f1']:.6f}, precision={valid['selected_point_precision']:.6f}, net={int(valid['net_fixed_points'])}"
            )
    if not trace_gain_loss.empty:
        lines += ["", "## 获益最大 Trace"]
        gain = trace_gain_loss[trace_gain_loss["selected_config"].ne("REPRO_STRONG")].sort_values("net_fixed_points", ascending=False).head(8)
        for row in gain.to_dict("records"):
            lines.append(
                f"- {row['selected_config']} {row['trace_id']}: net={int(row['net_fixed_points'])}, "
                f"fixed={int(row['fixed_field_as_road'])}, introduced={int(row['introduced_road_as_field'])}"
            )
        lines += ["", "## 误伤 Trace"]
        harm = trace_gain_loss[trace_gain_loss["introduced_road_as_field"] > 0].sort_values("introduced_road_as_field", ascending=False).head(8)
        for row in harm.to_dict("records"):
            lines.append(
                f"- {row['selected_config']} {row['trace_id']}: introduced={int(row['introduced_road_as_field'])}, "
                f"fixed={int(row['fixed_field_as_road'])}, net={int(row['net_fixed_points'])}"
            )
    if not trace_focus.empty:
        lines += ["", "## PLUS_TRACE_FOCUS"]
        focus_top = trace_focus[
            trace_focus["selected_config"].astype(str).isin(["REPRO_STRONG", "PLUS_1", "PLUS_2", "PLUS_3"])
            & trace_focus["focus_trace"].astype(bool)
        ].copy()
        focus_top = focus_top.sort_values(["trace_id", "selected_config"])
        for row in focus_top.to_dict("records"):
            lines.append(
                f"- {row['selected_config']} {row['trace_id']}: selected={int(row['selected_points'])}, "
                f"fixed={int(row['fixed_field_as_road'])}, introduced={int(row['introduced_road_as_field'])}, "
                f"remaining_field_as_road={int(row['field_as_road_final'])}, extra_over_STRONG={int(row['extra_points_over_strong'])}"
            )
        lines.append("- 重点 trace 明细见 `diagnostics/FRG_MoE_CONSERVE_v3_PLUS_AUDIT/trace_focus_test.csv`。")
    lines += [
        "",
        "## 结论",
        f"- test macro-F1 最高配置：{best_test['config']} ({best_test['macro_f1']:.6f})",
        f"- 推荐：{rec}",
        f"- 原因：{reason}",
    ]
    if plus["macro_f1_delta_vs_strong"].max() < 0.001:
        lines.append("- PLUS 收益小于 0.001 macro-F1，建议停止继续压榨 fake_road，转向 fake_field / RSC。")
    lines += [
        "",
        f"最终推荐：{rec}",
        f"原因：{reason}",
    ]
    Path(args.report_path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_pack(args):
    pack_path = Path(args.pack_path)
    with zipfile.ZipFile(pack_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for rel in [
            "scripts/frg_moe_conserve_v3_plus_audit.py",
            "scripts/frg_moe_conserve_v3.py",
            args.summary_csv,
            args.report_path,
        ]:
            path = PROJECT_ROOT / rel if not Path(rel).is_absolute() else Path(rel)
            if path.exists():
                zf.write(path, path.relative_to(PROJECT_ROOT).as_posix())
        for path in Path(args.output_dir).rglob("*"):
            if path.is_file():
                resolved = path.resolve()
                zf.write(resolved, resolved.relative_to(PROJECT_ROOT).as_posix())


def main():
    args = parse_args()
    ensure_dirs(args)
    splits = [item.strip() for item in str(args.splits).split(",") if item.strip()]
    if "valid" not in splits or "test" not in splits:
        raise ValueError("FRG_PLUS_REQUIRES_VALID_AND_TEST_FOR_THRESHOLD_AND_REPORT")
    points = {split: BASE.read_points(args, split) for split in splits}
    scores = {split: BASE.attach_point_rows(BASE.read_audit(args, split), points[split]) for split in splits}
    baseline_metrics = {split: BASE.compute_metrics(points[split]["true_label"], points[split]["pred_label"]) for split in splits}
    strong_test_metrics = load_conserve_strong_metrics(args, points)
    strong_metrics = {split: baseline_metrics[split] for split in splits}
    strong_metrics["test"] = strong_test_metrics
    final_dfs = {}
    changed_maps = {}
    summary_rows = []
    selected_segments_all = []

    for split in splits:
        base_final = BASE.make_final_predictions(points[split], {}, "BASELINE")
        summary_rows.append(summarize_config(points[split], base_final, pd.DataFrame(), "BASELINE", "baseline", baseline_metrics[split], strong_metrics[split]))
        for name, config in TOPK_CONFIGS.items():
            candidates = collect_topk_candidates(scores[split], name, config)
            summary, segments, final_df, changed = evaluate_config(
                points[split], candidates, name, "fixed_topk", baseline_metrics[split], strong_metrics[split], args.output_dir
            )
            summary_rows.append(summary)
            final_dfs[f"{split}_{name}"] = final_df
            changed_maps[f"{split}_{name}"] = changed
            if not segments.empty:
                selected_segments_all.append(segments)

    threshold_infos = search_valid_thresholds(points["valid"], scores["valid"], baseline_metrics["valid"], strong_metrics["valid"])
    for name, info in threshold_infos.items():
        for split in splits:
            candidates = collect_threshold_candidates(scores[split], name, info["thresholds"])
            summary, segments, final_df, changed = evaluate_config(
                points[split], candidates, name, "valid_threshold", baseline_metrics[split], strong_metrics[split], args.output_dir
            )
            summary["threshold_global"] = info["thresholds"]["global"]
            summary["threshold_long"] = info["thresholds"]["long"]
            summary["threshold_very_long"] = info["thresholds"]["very_long"]
            summary_rows.append(summary)
            final_dfs[f"{split}_{name}"] = final_df
            changed_maps[f"{split}_{name}"] = changed
            if not segments.empty:
                segments["threshold_global"] = info["thresholds"]["global"]
                segments["threshold_long"] = info["thresholds"]["long"]
                segments["threshold_very_long"] = info["thresholds"]["very_long"]
                selected_segments_all.append(segments)

    selected_segments = pd.concat(selected_segments_all, ignore_index=True, sort=False) if selected_segments_all else pd.DataFrame()
    selected_segments.to_csv(Path(args.output_dir) / "selected_segments_all.csv", index=False)
    for split in splits:
        split_segments = selected_segments[selected_segments["split"].eq(split)] if not selected_segments.empty else pd.DataFrame()
        split_segments.to_csv(Path(args.output_dir) / f"selected_segments_{split}.csv", index=False)

    extra_df = make_extra_over_strong(final_dfs, args.output_dir)
    trace_gain_loss = make_trace_gain_loss(final_dfs, selected_segments, args.output_dir)
    trace_focus = make_trace_focus(trace_gain_loss, args.output_dir)
    strong_changed = set(changed_maps.get("test_REPRO_STRONG", {}).keys())
    residual_stats = residual_after_strong(points["test"], scores["test"], strong_changed)
    pd.DataFrame([residual_stats]).to_csv(Path(args.output_dir) / "strong_residual_fake_road_candidate_coverage.csv", index=False)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(args.summary_csv, index=False)
    repro = summary_df[(summary_df["split"].eq("test")) & (summary_df["config"].eq("REPRO_STRONG"))].iloc[0]
    repro_ok = (
        abs(float(repro["macro_f1"]) - STRONG_TEST_REF["macro_f1"]) <= 0.0005
        and abs(int(repro["road_as_field"]) - STRONG_TEST_REF["road_as_field"]) <= 5
        and abs(int(repro["field_as_road"]) - STRONG_TEST_REF["field_as_road"]) <= 5
    )
    metadata = {
        "experiment": EXPERIMENT_NAME,
        "baseline_group": BASELINE_GROUP,
        "baseline_test_ref": BASELINE_TEST_REF,
        "strong_test_ref": STRONG_TEST_REF,
        "safety_limits": SAFETY_LIMITS,
        "topk_configs": TOPK_CONFIGS,
        "threshold_infos": {
            name: {
                "thresholds": info["thresholds"],
                "valid_summary": None if info["valid_summary"] is None else {
                    key: value for key, value in info["valid_summary"].items() if isinstance(value, (int, float, str, bool))
                },
            }
            for name, info in threshold_infos.items()
        },
        "repro_strong_ok": bool(repro_ok),
        "input_column_notes": "Used FRG_MoE_AUDIT_v3 expert score columns directly; no label or raw data modification.",
    }
    (Path(args.output_dir) / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    write_report(args, summary_df, extra_df, trace_gain_loss, trace_focus, residual_stats, threshold_infos, bool(repro_ok))
    write_pack(args)
    print(f"summary: {args.summary_csv}")
    print(f"report: {args.report_path}")
    print(f"pack: {args.pack_path}")


if __name__ == "__main__":
    main()
