import argparse
import csv
import json
import zipfile
from pathlib import Path


GROUPS = [
    {
        "group": "PT2G_finetune_40ep",
        "run_dir": "runs/PT2G_finetune_40ep",
        "error_dir": "diagnostics/pt2g_finetune_40ep_error_analysis",
        "note": "PT2G baseline",
    },
    {
        "group": "PT2G_MSC_v1_finetune_40ep",
        "run_dir": "runs/PT2G_MSC_v1_finetune_40ep",
        "error_dir": "diagnostics/PT2G_MSC_v1_error_analysis",
        "note": "PT2G + MSC",
    },
    {
        "group": "PT2G_MSC_SD_v1_finetune_40ep",
        "run_dir": "runs/PT2G_MSC_SD_v1_finetune_40ep",
        "error_dir": "diagnostics/PT2G_MSC_SD_v1_error_analysis",
        "note": "PT2G + MSC + SD",
    },
    {
        "group": "PT2G_MSC_RC_v1_finetune_40ep",
        "run_dir": "runs/PT2G_MSC_RC_v1_finetune_40ep",
        "error_dir": "diagnostics/PT2G_MSC_RC_v1_error_analysis",
        "note": "PT2G + MSC + RC",
    },
    {
        "group": "PT2G_MSC_RCSD_v1_finetune_40ep",
        "run_dir": "runs/PT2G_MSC_RCSD_v1_finetune_40ep",
        "error_dir": "diagnostics/PT2G_MSC_RCSD_v1_error_analysis",
        "note": "PT2G + MSC + RC + SD",
    },
]


FIELDS = [
    "group",
    "valid_accuracy",
    "valid_macro_f1",
    "valid_road_f1",
    "valid_field_f1",
    "test_accuracy",
    "test_macro_f1",
    "test_road_f1",
    "test_field_f1",
    "pred_road_rate",
    "pred_field_rate",
    "collapse_flag",
    "road_as_field",
    "field_as_road",
    "long_road_as_field_segments",
    "long_field_as_road_segments",
    "high_conf_error",
    "notes",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize PT2G_MSC SD/RC/RCSD experiments.")
    parser.add_argument("--summary_csv", default="results/PT2G_MSC_aux_v1_summary.csv")
    parser.add_argument("--report_path", default="analysis/PT2G_MSC_aux_v1_report.md")
    parser.add_argument("--pack_path", default="analysis_packs/PT2G_MSC_aux_v1_for_chatgpt.zip")
    return parser.parse_args()


def read_json(path):
    path = Path(path)
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def read_rows(path):
    path = Path(path)
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def as_float(value, default=0.0):
    try:
        if value in (None, ""):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def as_bool(value):
    if isinstance(value, bool):
        return value
    return str(value).lower() in ("1", "true", "yes")


def metric_bundle(run_dir):
    run_dir = Path(run_dir)
    valid = read_json(run_dir / "metrics_summary.json")
    test = read_json(run_dir / "test_metrics_summary.json")
    pred_road_rate = test.get("pred_road_rate", valid.get("pred_road_rate", ""))
    pred_field_rate = test.get("pred_field_rate", 1.0 - as_float(pred_road_rate) if pred_road_rate != "" else "")
    return dict(
        valid_accuracy=valid.get("valid_accuracy", ""),
        valid_macro_f1=valid.get("valid_macro_f1", ""),
        valid_road_f1=valid.get("valid_road_f1", ""),
        valid_field_f1=valid.get("valid_field_f1", ""),
        test_accuracy=test.get("test_accuracy", ""),
        test_macro_f1=test.get("test_macro_f1", ""),
        test_road_f1=test.get("test_road_f1", ""),
        test_field_f1=test.get("test_field_f1", ""),
        pred_road_rate=pred_road_rate,
        pred_field_rate=pred_field_rate,
        collapse_flag=test.get(
            "collapse_flag",
            bool(pred_road_rate != "" and (as_float(pred_road_rate) < 0.05 or as_float(pred_road_rate) > 0.95)),
        ),
    )


def segment_length(row):
    return int(as_float(row.get("length", row.get("length_points", 0))))


def error_stats(error_dir):
    error_dir = Path(error_dir)
    segments = read_rows(error_dir / "error_segments.csv")
    point_candidates = [
        error_dir / "test_point_predictions_with_features.csv",
        error_dir / "point_predictions_with_features.csv",
    ]
    points = []
    for path in point_candidates:
        points = read_rows(path)
        if points:
            break

    test_segments = [row for row in segments if row.get("split", "test") == "test"]
    road_as_field = sum(segment_length(row) for row in test_segments if row.get("error_type") == "road_as_field")
    field_as_road = sum(segment_length(row) for row in test_segments if row.get("error_type") == "field_as_road")
    long_road = sum(
        1
        for row in test_segments
        if row.get("error_type") == "road_as_field" and segment_length(row) >= 20
    )
    long_field = sum(
        1
        for row in test_segments
        if row.get("error_type") == "field_as_road" and segment_length(row) >= 20
    )
    if points:
        test_points = [row for row in points if row.get("split", "test") == "test"]
        high_conf = sum(
            1
            for row in test_points
            if row.get("error_type") not in ("", "correct", None)
            and (as_bool(row.get("is_high_conf_error")) or as_float(row.get("confidence")) >= 0.8)
        )
        if road_as_field == 0 and field_as_road == 0:
            road_as_field = sum(1 for row in test_points if row.get("error_type") == "road_as_field")
            field_as_road = sum(1 for row in test_points if row.get("error_type") == "field_as_road")
    else:
        high_conf = ""
    return dict(
        road_as_field=road_as_field if segments or points else "",
        field_as_road=field_as_road if segments or points else "",
        long_road_as_field_segments=long_road if segments else "",
        long_field_as_road_segments=long_field if segments else "",
        high_conf_error=high_conf,
    )


def build_rows():
    rows = []
    for group in GROUPS:
        run_dir = Path(group["run_dir"])
        metrics = metric_bundle(run_dir)
        errors = error_stats(group["error_dir"])
        notes = [group["note"]]
        if not (run_dir / "metrics_summary.json").exists():
            notes.append("missing_valid_metrics")
        if not (run_dir / "test_metrics_summary.json").exists():
            notes.append("missing_test_metrics")
        if errors["road_as_field"] == "":
            notes.append("missing_error_analysis")
        rows.append(dict(group=group["group"], **metrics, **errors, notes="; ".join(notes)))
    return rows


def best_group(rows, field):
    candidates = [row for row in rows if row.get(field) not in ("", None)]
    if not candidates:
        return "NA"
    return max(candidates, key=lambda row: as_float(row[field]))["group"]


def row_by_group(rows, group):
    for row in rows:
        if row["group"] == group:
            return row
    return {}


def delta(new_row, old_row, field):
    if not new_row or not old_row or new_row.get(field) in ("", None) or old_row.get(field) in ("", None):
        return "NA"
    return f"{as_float(new_row[field]) - as_float(old_row[field]):.6f}"


def write_summary(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_report(path, rows):
    pt2g = row_by_group(rows, "PT2G_finetune_40ep")
    msc = row_by_group(rows, "PT2G_MSC_v1_finetune_40ep")
    sd = row_by_group(rows, "PT2G_MSC_SD_v1_finetune_40ep")
    rc = row_by_group(rows, "PT2G_MSC_RC_v1_finetune_40ep")
    rcsd = row_by_group(rows, "PT2G_MSC_RCSD_v1_finetune_40ep")
    lines = [
        "# PT2G_MSC_aux_v1 Report",
        "",
        "## 对比范围",
        "",
        "- PT2G_finetune_40ep",
        "- PT2G_MSC_v1_finetune_40ep",
        "- PT2G_MSC_SD_v1_finetune_40ep",
        "- PT2G_MSC_RC_v1_finetune_40ep",
        "- PT2G_MSC_RCSD_v1_finetune_40ep",
        "",
        "## 核心结论",
        "",
        f"- valid macro-F1 最好：{best_group(rows, 'valid_macro_f1')}。",
        f"- test macro-F1 最好：{best_group(rows, 'test_macro_f1')}。",
        f"- road-F1 最好：{best_group(rows, 'test_road_f1')}。",
        f"- field-F1 最好：{best_group(rows, 'test_field_f1')}。",
        f"- SD field_as_road 相对 MSC_v1 变化：{delta(sd, msc, 'field_as_road')}。",
        f"- SD long_field_as_road 相对 MSC_v1 变化：{delta(sd, msc, 'long_field_as_road_segments')}。",
        f"- RC road_as_field 相对 MSC_v1 变化：{delta(rc, msc, 'road_as_field')}。",
        f"- RC long_road_as_field 相对 MSC_v1 变化：{delta(rc, msc, 'long_road_as_field_segments')}。",
        f"- RC field_as_road 相对 MSC_v1 变化：{delta(rc, msc, 'field_as_road')}。",
        f"- RCSD road_as_field 相对 MSC_v1 变化：{delta(rcsd, msc, 'road_as_field')}。",
        f"- RCSD field_as_road 相对 MSC_v1 变化：{delta(rcsd, msc, 'field_as_road')}。",
        "",
        "## 分类坍塌检查",
        "",
    ]
    for row in rows:
        lines.append(
            f"- {row['group']}: pred_road_rate={row.get('pred_road_rate', '')}, "
            f"pred_field_rate={row.get('pred_field_rate', '')}, collapse_flag={row.get('collapse_flag', '')}。"
        )
    lines.extend(
        [
            "",
            "## 保留建议",
            "",
            "- 若 SD 同时降低 field_as_road 和 long_field_as_road，且 valid macro-F1 不低于 MSC_v1，优先保留 SD。",
            "- 若 RC 降低 road_as_field 和 long_road_as_field，同时 field_as_road 不明显增加，优先保留 RC。",
            "- 若 RCSD 的 test macro-F1 不低于 MSC_v1，且两类错误至少一类下降、另一类不恶化，优先进入下一阶段。",
            "- 若三组指标缺失，说明训练或错误分析尚未完成，本报告只作为汇总模板。",
        ]
    )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_pack(pack_path, summary_csv, report_path):
    include = [
        Path("models/MSCResidualControlModules.py"),
        Path("models/SegmentContextModule.py"),
        Path("utils/motion_state_features.py"),
        Path("models/Encoder.py"),
        Path("dataset.py"),
        Path("fine_tune.py"),
        Path(summary_csv),
        Path(report_path),
    ]
    include.extend(sorted(Path("diagnostics").glob("*msc_aux_audit.csv")))
    pack_path = Path(pack_path)
    pack_path.parent.mkdir(parents=True, exist_ok=True)
    if pack_path.exists():
        pack_path.unlink()
    with zipfile.ZipFile(pack_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for item in include:
            if item.exists() and item.is_file() and item.suffix != ".pt":
                zf.write(item, item.as_posix())


def main():
    args = parse_args()
    rows = build_rows()
    write_summary(args.summary_csv, rows)
    write_report(args.report_path, rows)
    make_pack(args.pack_path, args.summary_csv, args.report_path)
    print(f"summary_path={args.summary_csv}")
    print(f"report_path={args.report_path}")
    print(f"pack_path={args.pack_path}")
    print(f"best_valid_macro_f1={best_group(rows, 'valid_macro_f1')}")
    print(f"best_test_macro_f1={best_group(rows, 'test_macro_f1')}")


if __name__ == "__main__":
    main()
