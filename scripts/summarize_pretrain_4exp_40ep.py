import csv
import json
import zipfile
from pathlib import Path


GROUPS = [
    ("PT0_no_pretrain", "none", "PT0_no_pretrain_40ep"),
    ("PT1_current_pretrain", "current_masked", "PT1_current_finetune_40ep"),
    ("PT2_edge_weight_pretrain", "self_supervised_edge_weight", "PT2_edge_weight_finetune_40ep"),
    ("PT3_edge_type_weight_pretrain", "self_supervised_edge_type_weight", "PT3_edge_type_weight_finetune_40ep"),
]

PRETRAIN_RUNS = {
    "PT1_current_pretrain": "PT1_current_pretrain_40ep",
    "PT2_edge_weight_pretrain": "PT2_edge_weight_pretrain_40ep",
    "PT3_edge_type_weight_pretrain": "PT3_edge_type_weight_pretrain_40ep",
}


def read_json(path):
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def format_metric(value):
    if value is None or value == "":
        return ""
    return f"{float(value):.6f}"


def build_rows():
    rows = []
    for group, pretrain_type, finetune_run in GROUPS:
        metrics_path = Path("runs") / finetune_run / "metrics_summary.json"
        metrics = read_json(metrics_path)
        if metrics is None:
            rows.append(
                dict(
                    group=group,
                    pretrain_type=pretrain_type,
                    finetune_run=finetune_run,
                    valid_accuracy="",
                    valid_macro_f1="",
                    valid_road_f1="",
                    valid_field_f1="",
                    best_epoch="",
                    notes=f"metrics missing: {metrics_path}",
                )
            )
            continue
        rows.append(
            dict(
                group=group,
                pretrain_type=pretrain_type,
                finetune_run=finetune_run,
                valid_accuracy=format_metric(metrics.get("valid_accuracy")),
                valid_macro_f1=format_metric(metrics.get("valid_macro_f1")),
                valid_road_f1=format_metric(metrics.get("valid_road_f1")),
                valid_field_f1=format_metric(metrics.get("valid_field_f1")),
                best_epoch=metrics.get("best_epoch", ""),
                notes="completed",
            )
        )
    return rows


def best_row(rows, key):
    completed = [row for row in rows if row.get(key) not in ("", None)]
    if not completed:
        return None
    return max(completed, key=lambda row: float(row[key]))


def write_summary_csv(rows):
    path = Path("results/pretrain_4exp_40ep_summary.csv")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "group",
        "pretrain_type",
        "finetune_run",
        "valid_accuracy",
        "valid_macro_f1",
        "valid_road_f1",
        "valid_field_f1",
        "best_epoch",
        "notes",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def write_report(rows):
    path = Path("analysis/pretrain_4exp_40ep_report.md")
    path.parent.mkdir(parents=True, exist_ok=True)
    completed = [row for row in rows if row["notes"] == "completed"]
    best_macro = best_row(rows, "valid_macro_f1")
    best_road = best_row(rows, "valid_road_f1")
    best_field = best_row(rows, "valid_field_f1")
    recommended = best_macro["group"] if best_macro else "暂无"

    lines = [
        "# 四组预训练实验 40 轮结果报告",
        "",
        f"四组是否都跑完：{'是' if len(completed) == 4 else '否'}。",
        "",
        "## Valid 指标",
        "",
        "| group | accuracy | macro-F1 | road-F1 | field-F1 | best_epoch |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['group']} | {row['valid_accuracy']} | {row['valid_macro_f1']} | "
            f"{row['valid_road_f1']} | {row['valid_field_f1']} | {row['best_epoch']} |"
        )
    lines.extend(
        [
            "",
            f"macro-F1 最好：{best_macro['group'] if best_macro else '暂无'}。",
            f"road-F1 最好：{best_road['group'] if best_road else '暂无'}。",
            f"field-F1 最好：{best_field['group'] if best_field else '暂无'}。",
            "",
            f"推荐下一阶段 baseline：{recommended}。",
            "",
            "如果 PT2/PT3 没有提升，可能原因包括：自监督边权/边类型信号只由 masked reconstruction 间接约束，未必与 road/field 判别边界完全一致；K=4 的软类型没有额外分散约束，可能学到弱区分或局部塌缩；当前 fine-tune 数据量和类别分布也可能让预训练收益被监督阶段覆盖。",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def write_zip(summary_csv, report_path):
    zip_path = Path("analysis_packs/pretrain_4exp_40ep_results_for_chatgpt.zip")
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    wanted_names = {
        "config_resolved.json",
        "command.txt",
        "pretrain_load_audit.json",
        "training_metrics.csv",
        "metrics_summary.json",
        "pretrain_summary.json",
        "train_log.csv",
        "valid_loss_curve.csv",
        "edge_weight_statistics.csv",
        "edge_type_statistics.csv",
        "dry_run_summary.json",
    }
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(summary_csv, summary_csv.as_posix())
        zf.write(report_path, report_path.as_posix())
        for run_dir in Path("runs").glob("PT*40ep"):
            if not run_dir.is_dir():
                continue
            for path in run_dir.iterdir():
                if path.name in wanted_names and path.is_file():
                    zf.write(path, path.as_posix())
    return zip_path


def main():
    rows = build_rows()
    summary_csv = write_summary_csv(rows)
    report_path = write_report(rows)
    zip_path = write_zip(summary_csv, report_path)
    print(f"summary_csv={summary_csv}")
    print(f"report={report_path}")
    print(f"zip={zip_path}")


if __name__ == "__main__":
    main()
