import argparse
import csv
import json
import zipfile
from pathlib import Path


DEFAULT_BASELINE_RUN = Path("runs/PT2G_finetune_40ep")
DEFAULT_MSC_RUN = Path("runs/PT2G_MSC_v1_finetune_40ep")
DEFAULT_AUDIT = Path("diagnostics/PT2G_MSC_v1_finetune_40ep_segment_context_audit.csv")
DEFAULT_SUMMARY = Path("results/PT2G_MSC_v1_summary.csv")
DEFAULT_REPORT = Path("analysis/PT2G_MSC_v1_report.md")
DEFAULT_PACK = Path("analysis_packs/PT2G_MSC_v1_for_chatgpt.zip")


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize PT2G_MSC_v1 after fine-tuning.")
    parser.add_argument("--baseline_run", default=str(DEFAULT_BASELINE_RUN))
    parser.add_argument("--msc_run", default=str(DEFAULT_MSC_RUN))
    parser.add_argument("--audit_csv", default=str(DEFAULT_AUDIT))
    parser.add_argument("--summary_csv", default=str(DEFAULT_SUMMARY))
    parser.add_argument("--report_path", default=str(DEFAULT_REPORT))
    parser.add_argument("--pack_path", default=str(DEFAULT_PACK))
    return parser.parse_args()


def read_json(path):
    path = Path(path)
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv_rows(path):
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


def metric_bundle(run_dir):
    run_dir = Path(run_dir)
    valid = read_json(run_dir / "metrics_summary.json")
    test = read_json(run_dir / "test_metrics_summary.json")
    pred_road_rate = test.get("pred_road_rate", valid.get("pred_road_rate", 0.0))
    pred_field_rate = test.get("pred_field_rate", 1.0 - as_float(pred_road_rate))
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
        collapse_flag=test.get("collapse_flag", bool(as_float(pred_road_rate) < 0.05 or as_float(pred_field_rate) < 0.05)),
    )


def last_audit_row(path):
    rows = read_csv_rows(path)
    return rows[-1] if rows else {}


def write_summary(path, row):
    fields = [
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
        "segment_scale",
        "notes",
    ]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow(row)


def delta_text(new_value, old_value):
    if new_value == "" or old_value == "":
        return "NA"
    return f"{as_float(new_value) - as_float(old_value):.6f}"


def error_segment_stats(error_analysis_dir):
    error_analysis_dir = Path(error_analysis_dir)
    segments = read_csv_rows(error_analysis_dir / "error_segments.csv")
    points = read_csv_rows(error_analysis_dir / "point_predictions_with_features.csv")
    return dict(
        road_as_field_points=sum(int(as_float(row.get("length_points"))) for row in segments if row.get("error_type") == "road_as_field"),
        field_as_road_points=sum(int(as_float(row.get("length_points"))) for row in segments if row.get("error_type") == "field_as_road"),
        long_road_as_field_segments=sum(1 for row in segments if row.get("error_type") == "road_as_field" and as_float(row.get("length_points")) >= 20),
        long_field_as_road_segments=sum(1 for row in segments if row.get("error_type") == "field_as_road" and as_float(row.get("length_points")) >= 20),
        high_conf_error_points=sum(
            1
            for row in points
            if row.get("error_type") not in ("", "correct", None) and as_float(row.get("confidence")) >= 0.8
        ),
    )


def build_report(baseline, msc, audit, error_compare):
    segment_scale = audit.get("segment_scale", "")
    context_ratio = audit.get("context_to_fused_ratio", "")
    lines = [
        "# PT2G_MSC_v1 Report",
        "",
        "## Status",
        "",
        "- Experiment: PT2G_MSC_v1_finetune_40ep.",
        "- Module: Multi-scale Segment Context Module after fused image+graph features and before the final head.",
        "- Cache: default cache/wheat_non_iid.",
        "- Graph cache: cache/pretrained_graphs/PT2G_topk3.",
        "- Pretrain mode: edge_weight.",
        "",
        "## Metric Comparison",
        "",
        f"- valid macro-F1 delta: {delta_text(msc['valid_macro_f1'], baseline['valid_macro_f1'])}.",
        f"- valid road-F1 delta: {delta_text(msc['valid_road_f1'], baseline['valid_road_f1'])}.",
        f"- test macro-F1 delta: {delta_text(msc['test_macro_f1'], baseline['test_macro_f1'])}.",
        f"- test road-F1 delta: {delta_text(msc['test_road_f1'], baseline['test_road_f1'])}.",
        f"- pred_road_rate: {msc['pred_road_rate']}.",
        f"- collapse_flag: {msc['collapse_flag']}.",
        f"- final segment_scale: {segment_scale}.",
        f"- final context_to_fused_ratio: {context_ratio}.",
        "",
        "## Error Analysis",
        "",
    ]
    if error_compare:
        lines.extend(
            [
                f"- road_as_field points delta: {error_compare['road_as_field_points_delta']}.",
                f"- field_as_road points delta: {error_compare['field_as_road_points_delta']}.",
                f"- long_road_as_field_segments delta: {error_compare['long_road_as_field_segments_delta']}.",
                f"- long_field_as_road_segments delta: {error_compare['long_field_as_road_segments_delta']}.",
                f"- confidence>=0.8 error points delta: {error_compare['high_conf_error_points_delta']}.",
            ]
        )
    else:
        lines.append("- Error-analysis CSVs were not both available, so detailed error deltas were skipped.")

    lines.extend(
        [
            "",
            "## Decision Notes",
            "",
            "- Success mainly depends on valid macro-F1 improvement, valid road-F1 not dropping, stable pred_road_rate, and segment_scale moving away from zero.",
            "- If segment_scale remains near zero, MSC was not effectively used.",
            "- If context_to_fused_ratio is very large and metrics degrade, MSC may be too strong.",
        ]
    )
    return "\n".join(lines) + "\n"


def make_pack(pack_path, paths):
    pack_path = Path(pack_path)
    pack_path.parent.mkdir(parents=True, exist_ok=True)
    if pack_path.exists():
        pack_path.unlink()
    with zipfile.ZipFile(pack_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in paths:
            path = Path(path)
            if path.exists() and path.is_file():
                zf.write(path, path.as_posix())


def main():
    args = parse_args()
    baseline_run = Path(args.baseline_run)
    msc_run = Path(args.msc_run)
    baseline = metric_bundle(baseline_run)
    msc = metric_bundle(msc_run)
    audit = last_audit_row(args.audit_csv)
    segment_scale = audit.get("segment_scale", "")

    summary_row = dict(
        group="PT2G_MSC_v1",
        **msc,
        segment_scale=segment_scale,
        notes="completed" if (msc_run / "metrics_summary.json").exists() else "missing_run_metrics",
    )
    write_summary(args.summary_csv, summary_row)

    baseline_errors = error_segment_stats("diagnostics/pt2g_finetune_40ep_error_analysis")
    msc_errors = error_segment_stats("diagnostics/PT2G_MSC_v1_error_analysis")
    error_compare = None
    if any(baseline_errors.values()) and any(msc_errors.values()):
        error_compare = {
            f"{key}_delta": msc_errors[key] - baseline_errors[key]
            for key in baseline_errors.keys()
        }

    report = build_report(baseline, msc, audit, error_compare)
    report_path = Path(args.report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")

    make_pack(
        args.pack_path,
        [
            "models/SegmentContextModule.py",
            "models/Encoder.py",
            "fine_tune.py",
            "runs/PT2G_MSC_v1_finetune_40ep/config_resolved.json",
            "runs/PT2G_MSC_v1_finetune_40ep/training_metrics.csv",
            args.summary_csv,
            args.report_path,
            args.audit_csv,
        ],
    )

    print(f"summary_path={args.summary_csv}")
    print(f"report_path={args.report_path}")
    print(f"pack_path={args.pack_path}")
    print(f"segment_scale={segment_scale}")


if __name__ == "__main__":
    main()
