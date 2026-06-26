import argparse
import csv
import json
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from build_wheat_merge_duplicate_cache import (
    align_raw_to_wheat43,
    build_merge_groups,
    build_merged_features,
    load_split_files,
    read_raw_wheat,
    read_wheat43,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export merged wheat 43-dimensional Excel files without overwriting original data."
    )
    parser.add_argument("--raw_dir", default="wheat/sampled_wheat")
    parser.add_argument("--wheat43_dir", default="wheat/sampled_wheat_43")
    parser.add_argument("--split_dir", default="wheat/Non-Identically_Distributed_Coco")
    parser.add_argument("--json_prefix", default="sampled_wheat_43")
    parser.add_argument("--output_dir", default="wheat/sampled_wheat_43_merge_d05_dt10_s1")
    parser.add_argument("--audit_output", default="diagnostics/wheat_merge_43_excel_audit.csv")
    parser.add_argument("--distance_threshold", type=float, default=0.5)
    parser.add_argument("--time_threshold", type=float, default=10.0)
    parser.add_argument("--speed_threshold", type=float, default=1.0)
    parser.add_argument("--diameter_threshold", type=float, default=1.0)
    parser.add_argument("--max_group_size", type=int, default=6)
    parser.add_argument("--max_group_seconds", type=float, default=30.0)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--dry_run_traces", type=int, default=3)
    return parser.parse_args()


def iter_split_files(args):
    seen = set()
    for split in ("train", "valid", "test"):
        for file_name in load_split_files(args.split_dir, args.json_prefix, split):
            if file_name in seen:
                continue
            seen.add(file_name)
            yield split, file_name


def export_one(split, file_name, args, output_dir):
    raw_path = Path(args.raw_dir) / file_name
    wheat43_path = Path(args.wheat43_dir) / file_name
    raw_df = read_raw_wheat(raw_path)
    wheat43_df = read_wheat43(wheat43_path)
    aligned_raw, _aligned_raw_indices, fallback_count = align_raw_to_wheat43(raw_df, wheat43_df)
    groups = build_merge_groups(aligned_raw, args)
    features, labels, mixed_label_groups, merge_counts = build_merged_features(wheat43_df, groups)

    output = pd.DataFrame(features)
    output[43] = labels.astype(int)
    output.columns = list(range(44))
    output_path = output_dir / file_name
    if not args.dry_run:
        output.to_excel(output_path, index=False)

    original_points = int(len(wheat43_df))
    merged_points = int(len(output))
    merge_groups = int(sum(1 for count in merge_counts if count > 1))
    return dict(
        file=file_name,
        split=split,
        output_path=str(output_path),
        original_points=original_points,
        merged_points=merged_points,
        reduction_rate=(original_points - merged_points) / original_points if original_points else 0.0,
        merge_groups=merge_groups,
        avg_merge_count=sum(merge_counts) / len(merge_counts) if merge_counts else 0.0,
        max_merge_count=max(merge_counts) if merge_counts else 0,
        mixed_label_groups=int(mixed_label_groups),
        mixed_label_rate=float(mixed_label_groups / max(merge_groups, 1)),
        fallback_alignments=int(fallback_count),
        feature6_min=float(features[:, 5].min()) if len(features) else 0.0,
        feature6_max=float(features[:, 5].max()) if len(features) else 0.0,
        feature6_mean=float(features[:, 5].mean()) if len(features) else 0.0,
    )


def write_audit(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "file",
        "split",
        "output_path",
        "original_points",
        "merged_points",
        "reduction_rate",
        "merge_groups",
        "avg_merge_count",
        "max_merge_count",
        "mixed_label_groups",
        "mixed_label_rate",
        "fallback_alignments",
        "feature6_min",
        "feature6_max",
        "feature6_mean",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    if not args.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for index, (split, file_name) in enumerate(iter_split_files(args), start=1):
        row = export_one(split, file_name, args, output_dir)
        rows.append(row)
        print(
            f"{index:03d} split={split} file={file_name} "
            f"original={row['original_points']} merged={row['merged_points']} "
            f"reduction={row['reduction_rate']:.6f}",
            flush=True,
        )
        if args.dry_run and index >= args.dry_run_traces:
            break

    if not args.dry_run:
        write_audit(args.audit_output, rows)
        manifest = dict(
            output_dir=str(output_dir),
            audit_output=args.audit_output,
            source_wheat43_dir=args.wheat43_dir,
            source_raw_dir=args.raw_dir,
            file_count=len(rows),
            note="Merged 43-dimensional features are raw-scale Excel data with column 43 as label; original directories are not overwritten.",
        )
        (output_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"WROTE {output_dir} files={len(rows)}")
        print(f"WROTE {args.audit_output}")


if __name__ == "__main__":
    main()
