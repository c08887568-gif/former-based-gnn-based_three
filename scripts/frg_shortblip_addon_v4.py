import json
import math
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


AUDIT_NAME = "FRG_SHORTBLIP_ADDON_v4"
OUT_DIR = PROJECT_ROOT / "diagnostics" / AUDIT_NAME
RESULT_PATH = PROJECT_ROOT / "results" / f"{AUDIT_NAME}_summary.csv"
REPORT_PATH = PROJECT_ROOT / "analysis" / f"{AUDIT_NAME}_report.md"
PACK_PATH = PROJECT_ROOT / "analysis_packs" / f"{AUDIT_NAME}_for_chatgpt.zip"

PLUS_AUDIT_DIR = PROJECT_ROOT / "diagnostics" / "FRG_MoE_CONSERVE_v3_PLUS_AUDIT"
RELAX_AUDIT_DIR = PROJECT_ROOT / "diagnostics" / "FRG_MoE_CONSERVE_v3_RELAX_AUDIT"
FSCG_DIR = PROJECT_ROOT / "diagnostics" / "FRG_FSCG_AUDIT_v4"

BASELINE = {
    "accuracy": np.nan,
    "macro_f1": 0.886494,
    "road_f1": np.nan,
    "field_f1": np.nan,
    "road_as_field": 3876,
    "field_as_road": 3423,
    "pred_road_rate": 0.212288,
}

BASE_CONFIGS = {
    "PLUS_3": {
        "train": PLUS_AUDIT_DIR / "final_predictions_train_PLUS_3.csv",
        "valid": PLUS_AUDIT_DIR / "final_predictions_valid_PLUS_3.csv",
        "test": PLUS_AUDIT_DIR / "final_predictions_test_PLUS_3.csv",
    },
    "PLUS_4": {
        "valid": RELAX_AUDIT_DIR / "final_predictions_valid_PLUS_4.csv",
        "test": RELAX_AUDIT_DIR / "final_predictions_test_PLUS_4.csv",
    },
}

ADDON_CONFIGS = {
    "PLUS_3_SB_SAFE": {
        "base": "PLUS_3",
        "min_precision": 0.98,
        "max_length": 10,
        "audit_only": False,
    },
    "PLUS_3_SB_BALANCED": {
        "base": "PLUS_3",
        "min_precision": 0.97,
        "max_length": 16,
        "audit_only": False,
    },
    "PLUS_4_SB_SAFE": {
        "base": "PLUS_4",
        "min_precision": 0.98,
        "max_length": 10,
        "audit_only": False,
    },
    "PLUS_4_SB_BALANCED": {
        "base": "PLUS_4",
        "min_precision": 0.97,
        "max_length": 16,
        "audit_only": False,
    },
    "PLUS_4_SB_STRONG_AUDIT": {
        "base": "PLUS_4",
        "min_precision": 0.95,
        "max_length": 20,
        "audit_only": True,
    },
}

RISK_TRACES = [
    "wheat_1_harvestor_124",
    "wheat_1_harvestor_95",
    "wheat_1_harvestor_128",
]


def ensure_dirs():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    PACK_PATH.parent.mkdir(parents=True, exist_ok=True)


def normalize_prediction_columns(df):
    df = df.copy()
    if "label" not in df.columns and "true_label" in df.columns:
        df = df.rename(columns={"true_label": "label"})
    if "base_pred" not in df.columns and "pred_label" in df.columns:
        df = df.rename(columns={"pred_label": "base_pred"})
    if "final_pred" not in df.columns:
        df["final_pred"] = df["base_pred"]
    for col in ("label", "base_pred", "final_pred", "global_index", "point_index"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(-1).astype(int)
    df["trace_id"] = df["trace_id"].astype(str)
    if "changed_by_frg" not in df.columns:
        df["changed_by_frg"] = df["final_pred"].ne(df["base_pred"])
    return df


def load_predictions(base_name, split):
    path = BASE_CONFIGS[base_name][split]
    if not path.exists():
        raise FileNotFoundError(f"Missing base prediction file: {path}")
    return normalize_prediction_columns(pd.read_csv(path))


def split_available(base_name, split):
    return split in BASE_CONFIGS.get(base_name, {}) and BASE_CONFIGS[base_name][split].exists()


def load_shortblip_candidates(split):
    path = FSCG_DIR / f"fake_road_candidates_{split}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing FSCG candidates: {path}")
    df = pd.read_csv(path)
    df["trace_id"] = df["trace_id"].astype(str)
    for col in ("pred_class", "length", "start_index", "end_index", "road_points", "field_points"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
    if "short_isolated_blip_score" not in df.columns:
        raise ValueError("FRG_FSCG_AUDIT_v4 candidates do not contain short_isolated_blip_score")
    df["short_isolated_blip_score"] = pd.to_numeric(df["short_isolated_blip_score"], errors="coerce").fillna(0.0)
    if "source_type" not in df.columns:
        df["source_type"] = "unknown"
    if "run_state" not in df.columns:
        df["run_state"] = "unknown"
    if "target" not in df.columns:
        field_ratio = pd.to_numeric(df.get("field_ratio", 0), errors="coerce").fillna(0)
        road_ratio = pd.to_numeric(df.get("road_ratio", 0), errors="coerce").fillna(0)
        df["target"] = np.where(field_ratio >= 0.8, 1, np.where(road_ratio >= 0.8, 0, -1))
    return df


def parse_point_indices(value):
    if pd.isna(value):
        return []
    out = []
    for part in str(value).split("|"):
        part = part.strip()
        if not part:
            continue
        try:
            out.append(int(float(part)))
        except ValueError:
            continue
    return out


def make_point_key(trace_id, global_index):
    return f"{trace_id}@@{int(global_index)}"


def prediction_lookup(df):
    cols = ["trace_id", "global_index", "label", "base_pred", "final_pred"]
    tmp = df[cols].copy()
    tmp["point_key"] = [make_point_key(t, g) for t, g in zip(tmp["trace_id"], tmp["global_index"])]
    return tmp.set_index("point_key")


def changed_point_keys(df):
    changed = df[df["final_pred"].ne(df["base_pred"])]
    return set(make_point_key(t, g) for t, g in zip(changed["trace_id"], changed["global_index"]))


def base_stats_from_predictions(df):
    changed = df[df["final_pred"].ne(df["base_pred"])].copy()
    fixed = int(((changed["base_pred"] == 0) & (changed["label"] == 1) & (changed["final_pred"] == 1)).sum())
    introduced = int(((changed["base_pred"] == 0) & (changed["label"] == 0) & (changed["final_pred"] == 1)).sum())
    return {
        "selected_points": int(len(changed)),
        "fixed_field_as_road": fixed,
        "introduced_road_as_field": introduced,
        "net_fixed_points": fixed - introduced,
        "selected_point_precision": fixed / max(fixed + introduced, 1),
    }


def metrics_from_predictions(df):
    y_true = df["label"].astype(int).to_numpy()
    y_pred = df["final_pred"].astype(int).to_numpy()
    road_as_field = int(((y_true == 0) & (y_pred == 1)).sum())
    field_as_road = int(((y_true == 1) & (y_pred == 0)).sum())
    pred_road_rate = float(np.mean(y_pred == 0))
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "road_f1": float(f1_score(y_true, y_pred, labels=[0], average="macro", zero_division=0)),
        "field_f1": float(f1_score(y_true, y_pred, labels=[1], average="macro", zero_division=0)),
        "road_as_field": road_as_field,
        "field_as_road": field_as_road,
        "pred_road_rate": pred_road_rate,
        "pred_field_rate": 1.0 - pred_road_rate,
    }


def candidate_point_rows(candidate, pred_lut, base_changed):
    trace_id = str(candidate["trace_id"])
    keys = []
    for idx in parse_point_indices(candidate.get("point_indices", "")):
        key = make_point_key(trace_id, idx)
        if key in base_changed:
            continue
        if key not in pred_lut.index:
            continue
        point = pred_lut.loc[key]
        if int(point["base_pred"]) != 0 or int(point["final_pred"]) != 0:
            continue
        keys.append(key)
    return keys


def prepare_addon_candidates(base_name, split, max_length):
    pred_df = load_predictions(base_name, split)
    candidates = load_shortblip_candidates(split)
    candidates = candidates[
        candidates["pred_class"].eq(0)
        & candidates["length"].le(max_length)
        & candidates["run_state"].ne("mixed_transition")
        & candidates["source_type"].eq("whole")
    ].copy()
    pred_lut = prediction_lookup(pred_df)
    base_changed = changed_point_keys(pred_df)
    rows = []
    for _, cand in candidates.iterrows():
        keys = candidate_point_rows(cand, pred_lut, base_changed)
        if not keys:
            continue
        point_df = pred_lut.loc[keys]
        if isinstance(point_df, pd.Series):
            point_df = point_df.to_frame().T
        fixed = int((point_df["label"].astype(int) == 1).sum())
        introduced = int((point_df["label"].astype(int) == 0).sum())
        row = cand.to_dict()
        row["eligible_point_keys"] = "|".join(keys)
        row["eligible_points"] = int(len(keys))
        row["addon_fixed_field_as_road"] = fixed
        row["addon_introduced_road_as_field"] = introduced
        row["addon_net_gain"] = fixed - introduced
        row["addon_point_precision_oracle"] = fixed / max(fixed + introduced, 1)
        row["addon_run_positive"] = int(fixed > introduced)
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    out["base_config"] = base_name
    out["split"] = split
    return out


def selected_stats(cands, threshold):
    if cands.empty:
        return empty_selection_stats(threshold)
    selected = cands[cands["short_isolated_blip_score"] >= threshold].copy()
    return selection_stats_from_selected(selected, threshold)


def empty_selection_stats(threshold):
    return {
        "threshold": float(threshold),
        "selected_runs": 0,
        "selected_points": 0,
        "fixed_field_as_road": 0,
        "introduced_road_as_field": 0,
        "net_gain": 0,
        "selected_point_precision": 0.0,
        "selected_run_precision": 0.0,
    }


def selection_stats_from_selected(selected, threshold):
    if selected.empty:
        return empty_selection_stats(threshold)
    fixed = int(selected["addon_fixed_field_as_road"].sum())
    introduced = int(selected["addon_introduced_road_as_field"].sum())
    return {
        "threshold": float(threshold),
        "selected_runs": int(len(selected)),
        "selected_points": int(selected["eligible_points"].sum()),
        "fixed_field_as_road": fixed,
        "introduced_road_as_field": introduced,
        "net_gain": fixed - introduced,
        "selected_point_precision": fixed / max(fixed + introduced, 1),
        "selected_run_precision": float(selected["addon_run_positive"].mean()) if len(selected) else 0.0,
    }


def threshold_grid(cands):
    if cands.empty:
        return [math.inf]
    scores = np.sort(cands["short_isolated_blip_score"].dropna().unique())[::-1]
    if len(scores) <= 300:
        return scores.tolist()
    quantiles = np.linspace(0.0, 1.0, 301)
    grid = np.quantile(scores, quantiles)
    return sorted(set(float(x) for x in grid), reverse=True)


def sweep_thresholds(base_name, split, max_length):
    cands = prepare_addon_candidates(base_name, split, max_length)
    rows = []
    for threshold in threshold_grid(cands):
        row = selected_stats(cands, threshold)
        row.update({"base_config": base_name, "split": split, "max_length": max_length})
        rows.append(row)
    return pd.DataFrame(rows), cands


def pick_threshold(valid_sweep, min_precision):
    if valid_sweep.empty:
        return math.inf, empty_selection_stats(math.inf)
    ok = valid_sweep[
        (valid_sweep["selected_points"] > 0)
        & (valid_sweep["selected_point_precision"] >= min_precision)
        & (valid_sweep["introduced_road_as_field"] <= 80)
    ].copy()
    if ok.empty:
        return math.inf, empty_selection_stats(math.inf)
    ok = ok.sort_values(
        ["net_gain", "selected_point_precision", "selected_points", "threshold"],
        ascending=[False, False, False, False],
    )
    row = ok.iloc[0].to_dict()
    return float(row["threshold"]), row


def trace_guard_from_selected(selected, min_precision):
    if selected.empty:
        return []
    grouped = selected.groupby("trace_id").agg(
        selected_points=("eligible_points", "sum"),
        fixed=("addon_fixed_field_as_road", "sum"),
        introduced=("addon_introduced_road_as_field", "sum"),
    )
    grouped["precision"] = grouped["fixed"] / (grouped["fixed"] + grouped["introduced"]).replace(0, np.nan)
    grouped["precision"] = grouped["precision"].fillna(0.0)
    disabled = grouped[(grouped["selected_points"] >= 5) & (grouped["precision"] < min_precision)]
    return disabled.index.astype(str).tolist()


def apply_addon(base_name, split, config_name, threshold, max_length, disabled_traces=None):
    disabled_traces = set(disabled_traces or [])
    base_df = load_predictions(base_name, split)
    cands = prepare_addon_candidates(base_name, split, max_length)
    selected = cands[cands["short_isolated_blip_score"] >= threshold].copy() if not cands.empty else cands
    if disabled_traces and not selected.empty:
        selected = selected[~selected["trace_id"].astype(str).isin(disabled_traces)].copy()
    if selected.empty:
        final_df = base_df.copy()
        final_df["changed_by_shortblip_addon"] = False
        final_df["shortblip_candidate_id"] = np.nan
        final_df["shortblip_score"] = np.nan
        return final_df, selected

    point_to_candidate = {}
    for _, row in selected.sort_values("short_isolated_blip_score", ascending=False).iterrows():
        for key in str(row["eligible_point_keys"]).split("|"):
            if key and key not in point_to_candidate:
                point_to_candidate[key] = (row["run_id"], float(row["short_isolated_blip_score"]))

    final_df = base_df.copy()
    final_df["point_key"] = [make_point_key(t, g) for t, g in zip(final_df["trace_id"], final_df["global_index"])]
    final_df["changed_by_shortblip_addon"] = final_df["point_key"].isin(point_to_candidate)
    final_df["shortblip_candidate_id"] = final_df["point_key"].map(lambda k: point_to_candidate.get(k, (np.nan, np.nan))[0])
    final_df["shortblip_score"] = final_df["point_key"].map(lambda k: point_to_candidate.get(k, (np.nan, np.nan))[1])
    addon_mask = final_df["changed_by_shortblip_addon"]
    final_df.loc[addon_mask, "final_pred"] = 1
    final_df.loc[addon_mask, "selected_config"] = config_name
    final_df = final_df.drop(columns=["point_key"])
    return final_df, selected


def summarize_config(config_name, split, final_df, selected, base_name):
    metrics = metrics_from_predictions(final_df)
    changed = final_df[final_df["final_pred"].ne(final_df["base_pred"])].copy()
    addon = final_df[final_df.get("changed_by_shortblip_addon", False).astype(bool)].copy()
    fixed_total = int(((changed["base_pred"] == 0) & (changed["label"] == 1) & (changed["final_pred"] == 1)).sum())
    introduced_total = int(((changed["base_pred"] == 0) & (changed["label"] == 0) & (changed["final_pred"] == 1)).sum())
    fixed_addon = int(((addon["base_pred"] == 0) & (addon["label"] == 1) & (addon["final_pred"] == 1)).sum())
    introduced_addon = int(((addon["base_pred"] == 0) & (addon["label"] == 0) & (addon["final_pred"] == 1)).sum())
    total_precision = fixed_total / max(fixed_total + introduced_total, 1)
    addon_precision = fixed_addon / max(fixed_addon + introduced_addon, 1)
    run_precision = float(selected["addon_run_positive"].mean()) if selected is not None and len(selected) else 0.0
    base_df = load_predictions(base_name, split)
    base_selected_points = int(base_df["final_pred"].ne(base_df["base_pred"]).sum())
    row = {
        "config": config_name,
        "split": split,
        "base_config": base_name,
        **metrics,
        "selected_points_total": int(len(changed)),
        "selected_points_base": base_selected_points,
        "selected_points_shortblip_addon": int(len(addon)),
        "fixed_field_as_road_total": fixed_total,
        "introduced_road_as_field_total": introduced_total,
        "net_fixed_points_total": fixed_total - introduced_total,
        "fixed_field_as_road_addon": fixed_addon,
        "introduced_road_as_field_addon": introduced_addon,
        "net_gain_addon": fixed_addon - introduced_addon,
        "selected_point_precision_total": total_precision,
        "selected_point_precision_addon": addon_precision,
        "selected_run_precision_addon": run_precision,
        "pred_road_rate_drop": BASELINE["pred_road_rate"] - metrics["pred_road_rate"],
    }
    return row


def summarize_base(base_name, split):
    df = load_predictions(base_name, split)
    metrics = metrics_from_predictions(df)
    base_stats = base_stats_from_predictions(df)
    return {
        "config": f"BASE_{base_name}",
        "split": split,
        "base_config": base_name,
        **metrics,
        "selected_points_total": base_stats["selected_points"],
        "selected_points_base": base_stats["selected_points"],
        "selected_points_shortblip_addon": 0,
        "fixed_field_as_road_total": base_stats["fixed_field_as_road"],
        "introduced_road_as_field_total": base_stats["introduced_road_as_field"],
        "net_fixed_points_total": base_stats["net_fixed_points"],
        "fixed_field_as_road_addon": 0,
        "introduced_road_as_field_addon": 0,
        "net_gain_addon": 0,
        "selected_point_precision_total": base_stats["selected_point_precision"],
        "selected_point_precision_addon": 0.0,
        "selected_run_precision_addon": 0.0,
        "pred_road_rate_drop": BASELINE["pred_road_rate"] - metrics["pred_road_rate"],
    }


def summarize_baseline(split):
    base_df = load_predictions("PLUS_3", split)
    df = base_df.copy()
    df["final_pred"] = df["base_pred"]
    metrics = metrics_from_predictions(df)
    return {
        "config": "BASELINE_PT2G_MSC_v1_time_fixed_strict_40pre",
        "split": split,
        "base_config": "none",
        **metrics,
        "selected_points_total": 0,
        "selected_points_base": 0,
        "selected_points_shortblip_addon": 0,
        "fixed_field_as_road_total": 0,
        "introduced_road_as_field_total": 0,
        "net_fixed_points_total": 0,
        "fixed_field_as_road_addon": 0,
        "introduced_road_as_field_addon": 0,
        "net_gain_addon": 0,
        "selected_point_precision_total": 0.0,
        "selected_point_precision_addon": 0.0,
        "selected_run_precision_addon": 0.0,
        "pred_road_rate_drop": 0.0,
    }


def trace_gain_loss(base_name, split, final_df):
    base_df = load_predictions(base_name, split)
    base_metrics = (
        base_df.assign(
            base_field_as_road=((base_df["label"] == 1) & (base_df["final_pred"] == 0)).astype(int),
            base_road_as_field=((base_df["label"] == 0) & (base_df["final_pred"] == 1)).astype(int),
        )
        .groupby("trace_id")[["base_field_as_road", "base_road_as_field"]]
        .sum()
    )
    addon = final_df[final_df.get("changed_by_shortblip_addon", False).astype(bool)].copy()
    if addon.empty:
        out = base_metrics.copy()
        out["addon_selected_points"] = 0
        out["addon_fixed_field_as_road"] = 0
        out["addon_introduced_road_as_field"] = 0
    else:
        addon["addon_selected_points"] = 1
        addon["addon_fixed_field_as_road"] = ((addon["label"] == 1) & (addon["base_pred"] == 0)).astype(int)
        addon["addon_introduced_road_as_field"] = ((addon["label"] == 0) & (addon["base_pred"] == 0)).astype(int)
        extra = addon.groupby("trace_id")[[
            "addon_selected_points",
            "addon_fixed_field_as_road",
            "addon_introduced_road_as_field",
        ]].sum()
        out = base_metrics.join(extra, how="outer").fillna(0)
    final_group = (
        final_df.assign(
            field_as_road_after=((final_df["label"] == 1) & (final_df["final_pred"] == 0)).astype(int),
            road_as_field_after=((final_df["label"] == 0) & (final_df["final_pred"] == 1)).astype(int),
        )
        .groupby("trace_id")[["field_as_road_after", "road_as_field_after"]]
        .sum()
    )
    out = out.join(final_group, how="outer").fillna(0).reset_index()
    out = out.rename(columns={"base_field_as_road": "field_as_road_before", "base_road_as_field": "road_as_field_before"})
    out["addon_net_gain"] = out["addon_fixed_field_as_road"] - out["addon_introduced_road_as_field"]
    out["addon_precision"] = out["addon_fixed_field_as_road"] / (out["addon_fixed_field_as_road"] + out["addon_introduced_road_as_field"]).replace(0, np.nan)
    out["addon_precision"] = out["addon_precision"].fillna(0.0)
    return out[[
        "trace_id",
        "addon_selected_points",
        "addon_fixed_field_as_road",
        "addon_introduced_road_as_field",
        "addon_net_gain",
        "addon_precision",
        "field_as_road_before",
        "field_as_road_after",
        "road_as_field_before",
        "road_as_field_after",
    ]].sort_values(["addon_net_gain", "addon_selected_points"], ascending=False)


def build_report(summary, threshold_records, trace_gain):
    test_rows = summary[summary["split"] == "test"].copy()
    plus3 = test_rows[test_rows["config"] == "BASE_PLUS_3"].iloc[0]
    plus4 = test_rows[test_rows["config"] == "BASE_PLUS_4"].iloc[0]
    base_better = "PLUS_4" if plus4["macro_f1"] >= plus3["macro_f1"] else "PLUS_3"

    candidates = test_rows[test_rows["config"].isin([c for c, spec in ADDON_CONFIGS.items() if not spec["audit_only"]])].copy()
    candidates["base_macro"] = candidates["base_config"].map(
        {
            "PLUS_3": float(plus3["macro_f1"]),
            "PLUS_4": float(plus4["macro_f1"]),
        }
    )
    candidates["macro_delta_vs_base"] = candidates["macro_f1"] - candidates["base_macro"]
    candidates["addon_safe"] = (
        (candidates["macro_delta_vs_base"] >= 0.0005)
        & (candidates["selected_point_precision_addon"] >= 0.97)
        & (candidates["introduced_road_as_field_addon"] <= 80)
    )
    safe = candidates[candidates["addon_safe"]].sort_values("macro_f1", ascending=False)
    if safe.empty:
        recommended = f"{base_better}"
        reason = "ShortBlip add-on 没有同时满足 macro-F1 提升、precision 和新增 road_as_field 安全条件。"
    else:
        recommended = str(safe.iloc[0]["config"])
        reason = "ShortBlip add-on 满足安全条件并带来可见 macro-F1 提升。"

    best_addon = candidates.sort_values("macro_f1", ascending=False).iloc[0] if not candidates.empty else None
    strong = test_rows[test_rows["config"] == "PLUS_4_SB_STRONG_AUDIT"]
    strong_text = ""
    if not strong.empty:
        row = strong.iloc[0]
        strong_text = (
            f"- STRONG_AUDIT: macro-F1={row['macro_f1']:.6f}, "
            f"addon_precision={row['selected_point_precision_addon']:.6f}, "
            f"addon introduced={int(row['introduced_road_as_field_addon'])}。\n"
        )

    risk_lines = []
    risk_reference_config = str(best_addon["config"]) if best_addon is not None else recommended
    risk_trace_gain = trace_gain[trace_gain.get("config", "").eq(risk_reference_config)] if not trace_gain.empty else trace_gain
    for risk in RISK_TRACES:
        hit = risk_trace_gain[risk_trace_gain["trace_id"].astype(str).str.contains(risk, regex=False)] if not risk_trace_gain.empty else pd.DataFrame()
        if hit.empty:
            risk_lines.append(f"- {risk}: {risk_reference_config} test add-on 未命中。")
        else:
            h = hit.iloc[0]
            risk_lines.append(
                f"- {risk}: {risk_reference_config} add-on selected={int(h['addon_selected_points'])}, "
                f"fixed={int(h['addon_fixed_field_as_road'])}, introduced={int(h['addon_introduced_road_as_field'])}, "
                f"net={int(h['addon_net_gain'])}, precision={h['addon_precision']:.6f}。"
            )

    lines = [
        "# FRG_SHORTBLIP_ADDON_v4 报告",
        "",
        "本次只做 fake_road 的 PLUS + ShortBlipExpert 组合修正实验：没有训练主模型，没有修改原始数据，没有处理 fake_field，也没有使用完整 FRG-FSCG final score。",
        "",
        "## Base 复现",
        f"- BASE_PLUS_3 test: macro-F1={plus3['macro_f1']:.6f}, road-F1={plus3['road_f1']:.6f}, field-F1={plus3['field_f1']:.6f}, road_as_field={int(plus3['road_as_field'])}, field_as_road={int(plus3['field_as_road'])}。",
        f"- BASE_PLUS_4 test: macro-F1={plus4['macro_f1']:.6f}, road-F1={plus4['road_f1']:.6f}, field-F1={plus4['field_f1']:.6f}, road_as_field={int(plus4['road_as_field'])}, field_as_road={int(plus4['field_as_road'])}。",
        f"- Base 更好：{base_better}。",
        "",
        "## ShortBlip Add-on 结果",
    ]
    for _, row in candidates.sort_values("config").iterrows():
        base_macro = plus3["macro_f1"] if row["base_config"] == "PLUS_3" else plus4["macro_f1"]
        lines.append(
            f"- {row['config']}: macro-F1={row['macro_f1']:.6f}, delta_vs_base={row['macro_f1'] - base_macro:.6f}, "
            f"addon_points={int(row['selected_points_shortblip_addon'])}, fixed={int(row['fixed_field_as_road_addon'])}, "
            f"introduced={int(row['introduced_road_as_field_addon'])}, addon_precision={row['selected_point_precision_addon']:.6f}。"
        )
    if strong_text:
        lines.extend(["", "## Strong Audit", strong_text.rstrip()])

    threshold_df = pd.DataFrame(threshold_records)
    lines.extend([
        "",
        "## Valid 阈值",
        threshold_df.to_markdown(index=False) if not threshold_df.empty else "无可用阈值。",
        "",
        "## Trace 风险检查",
        f"以下按 test macro-F1 最高的非 STRONG add-on 配置 {risk_reference_config} 检查风险 trace。",
        *risk_lines,
        "",
        "## 必答问题",
        f"1. PLUS_3 和 PLUS_4 哪个作为 base 更好：{base_better}。",
    ])
    if best_addon is not None:
        lines.append(
            f"2. ShortBlip add-on 是否真的带来额外收益：最佳 add-on 为 {best_addon['config']}，"
            f"相对 base macro-F1 变化 {best_addon['macro_delta_vs_base']:.6f}。"
        )
        lines.append(f"3. add-on 多修 field_as_road：{int(best_addon['fixed_field_as_road_addon'])}。")
        lines.append(f"4. add-on 多引入 road_as_field：{int(best_addon['introduced_road_as_field_addon'])}。")
        lines.append(
            f"5. add-on precision：{best_addon['selected_point_precision_addon']:.6f}，"
            f"{'达到' if best_addon['selected_point_precision_addon'] >= 0.97 else '未达到'} 0.97。"
        )
    else:
        lines.extend([
            "2. ShortBlip add-on 是否真的带来额外收益：没有有效 add-on 候选。",
            "3. add-on 多修 field_as_road：0。",
            "4. add-on 多引入 road_as_field：0。",
            "5. add-on precision：0。",
        ])
    max_trace_introduced = int(trace_gain["addon_introduced_road_as_field"].max()) if not trace_gain.empty else 0
    lines.extend([
        f"6. 是否存在 trace 级集中误伤：所有 add-on 配置中单 trace 最大新增 road_as_field={max_trace_introduced}。",
        f"7. SAFE / BALANCED / STRONG_AUDIT 哪个最推荐：{recommended if recommended in ADDON_CONFIGS else '不推荐 add-on'}。",
        f"8. 是否建议合入 ShortBlip add-on：{'建议' if recommended in ADDON_CONFIGS else '不建议'}。",
        f"9. 如果收益很小，是否建议停止 fake_road 优化，转向 fake_field / RSC：{'是' if recommended not in ADDON_CONFIGS else '暂不'}。",
        f"10. 最终推荐主线：{recommended}。",
        "",
        f"推荐主线：{recommended}",
        f"原因：{reason}",
    ])
    return "\n".join(lines) + "\n"


def make_pack():
    with zipfile.ZipFile(PACK_PATH, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in [
            PROJECT_ROOT / "scripts" / "frg_shortblip_addon_v4.py",
            RESULT_PATH,
            REPORT_PATH,
            OUT_DIR / "shortblip_selected_segments_valid.csv",
            OUT_DIR / "shortblip_selected_segments_test.csv",
            OUT_DIR / "addon_extra_selected_over_base_test.csv",
            OUT_DIR / "trace_gain_loss_test.csv",
            OUT_DIR / "shortblip_threshold_sweep_valid.csv",
            OUT_DIR / "shortblip_threshold_sweep_test.csv",
        ]:
            if path.exists():
                zf.write(path, path.relative_to(PROJECT_ROOT))
        for path in sorted(OUT_DIR.glob("final_predictions_*.csv")):
            zf.write(path, path.relative_to(PROJECT_ROOT))


def main():
    ensure_dirs()
    splits = ["valid", "test"]
    if split_available("PLUS_3", "train"):
        splits = ["train", "valid", "test"]
    summary_rows = []
    threshold_records = []
    valid_sweeps = []
    test_sweeps = []
    selected_valid = []
    selected_test = []
    trace_outputs = []
    extra_selected_test = []

    for split in splits:
        if split_available("PLUS_3", split):
            summary_rows.append(summarize_baseline(split))
        for base_name in BASE_CONFIGS:
            if split_available(base_name, split):
                summary_rows.append(summarize_base(base_name, split))

    for config_name, spec in ADDON_CONFIGS.items():
        base_name = spec["base"]
        max_length = int(spec["max_length"])
        valid_sweep, _ = sweep_thresholds(base_name, "valid", max_length)
        test_sweep, _ = sweep_thresholds(base_name, "test", max_length)
        threshold, picked = pick_threshold(valid_sweep, float(spec["min_precision"]))
        valid_sweep["config"] = config_name
        test_sweep["config"] = config_name
        valid_sweeps.append(valid_sweep)
        test_sweeps.append(test_sweep)
        valid_candidates_for_guard = prepare_addon_candidates(base_name, "valid", max_length)
        valid_selected_for_guard = (
            valid_candidates_for_guard[valid_candidates_for_guard["short_isolated_blip_score"] >= threshold].copy()
            if not valid_candidates_for_guard.empty
            else valid_candidates_for_guard
        )
        disabled_traces = trace_guard_from_selected(valid_selected_for_guard, float(spec["min_precision"]))
        threshold_records.append(
            {
                "config": config_name,
                "base_config": base_name,
                "min_precision": spec["min_precision"],
                "max_length": max_length,
                "threshold": threshold,
                "valid_selected_points": int(picked.get("selected_points", 0)),
                "valid_precision": float(picked.get("selected_point_precision", 0.0)),
                "valid_net_gain": int(picked.get("net_gain", 0)),
                "valid_disabled_trace_count": len(disabled_traces),
                "valid_disabled_traces": "|".join(disabled_traces),
            }
        )

        for split in splits:
            if not split_available(base_name, split):
                continue
            final_df, selected = apply_addon(base_name, split, config_name, threshold, max_length, disabled_traces)
            final_path = OUT_DIR / f"final_predictions_{split}_{config_name}.csv"
            final_df.to_csv(final_path, index=False)
            summary_rows.append(summarize_config(config_name, split, final_df, selected, base_name))
            selected = selected.copy()
            selected["config"] = config_name
            selected["threshold"] = threshold
            if split == "valid":
                selected_valid.append(selected)
            elif split == "test":
                selected_test.append(selected)
                selected["config"] = config_name
                extra_selected_test.append(selected)
                tg = trace_gain_loss(base_name, split, final_df)
                tg["config"] = config_name
                trace_outputs.append(tg)
            elif split == "train":
                selected.to_csv(OUT_DIR / f"shortblip_selected_segments_train_{config_name}.csv", index=False)

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(RESULT_PATH, index=False)

    if valid_sweeps:
        pd.concat(valid_sweeps, ignore_index=True).to_csv(OUT_DIR / "shortblip_threshold_sweep_valid.csv", index=False)
    if test_sweeps:
        pd.concat(test_sweeps, ignore_index=True).to_csv(OUT_DIR / "shortblip_threshold_sweep_test.csv", index=False)
    if selected_valid:
        pd.concat(selected_valid, ignore_index=True).to_csv(OUT_DIR / "shortblip_selected_segments_valid.csv", index=False)
    else:
        pd.DataFrame().to_csv(OUT_DIR / "shortblip_selected_segments_valid.csv", index=False)
    if selected_test:
        pd.concat(selected_test, ignore_index=True).to_csv(OUT_DIR / "shortblip_selected_segments_test.csv", index=False)
    else:
        pd.DataFrame().to_csv(OUT_DIR / "shortblip_selected_segments_test.csv", index=False)
    if extra_selected_test:
        pd.concat(extra_selected_test, ignore_index=True).to_csv(OUT_DIR / "addon_extra_selected_over_base_test.csv", index=False)
    else:
        pd.DataFrame().to_csv(OUT_DIR / "addon_extra_selected_over_base_test.csv", index=False)
    if trace_outputs:
        trace_gain = pd.concat(trace_outputs, ignore_index=True)
    else:
        trace_gain = pd.DataFrame()
    trace_gain.to_csv(OUT_DIR / "trace_gain_loss_test.csv", index=False)

    metadata = {
        "audit_name": AUDIT_NAME,
        "base_configs": {k: {s: str(p.relative_to(PROJECT_ROOT)) for s, p in v.items()} for k, v in BASE_CONFIGS.items()},
        "fscg_dir": str(FSCG_DIR.relative_to(PROJECT_ROOT)),
        "note": "Only ShortBlipExpert short_isolated_blip_score is used; fake_road_fscg_score is intentionally not used.",
        "threshold_records": threshold_records,
    }
    (OUT_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    REPORT_PATH.write_text(build_report(summary, threshold_records, trace_gain), encoding="utf-8")
    make_pack()
    print(f"summary: {RESULT_PATH.relative_to(PROJECT_ROOT)}")
    print(f"report: {REPORT_PATH.relative_to(PROJECT_ROOT)}")
    print(f"pack: {PACK_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
