import argparse
import csv
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.wheat_feature_rebuild import (  # noqa: E402
    align_raw_to_wheat43,
    read_raw_wheat,
    read_wheat_43,
    rebuild_wheat_43_features,
)


FIELDS = [
    "file",
    "rows_raw",
    "rows_43",
    "aligned_rows",
    "alignment_fallback_count",
    "max_abs_diff",
    "mean_abs_diff",
    "max_abs_diff_excluding_col5",
    "mean_abs_diff_excluding_col5",
    "label_match_rate",
    "audit_pass",
    "notes",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Audit whether wheat/sampled_wheat raw files can rebuild sampled_wheat_43 features.")
    parser.add_argument("--raw_dir", default="wheat/sampled_wheat")
    parser.add_argument("--wheat43_dir", default="wheat/sampled_wheat_43")
    parser.add_argument("--output", default="diagnostics/wheat_43_rebuild_audit.csv")
    parser.add_argument("--limit_files", type=int, default=None)
    parser.add_argument("--mean_threshold", type=float, default=10.0)
    parser.add_argument("--mean_threshold_excluding_col5", type=float, default=0.2)
    return parser.parse_args()


def audit_file(raw_path, wheat43_path, args):
    raw_df = read_raw_wheat(raw_path)
    wheat43_df = read_wheat_43(wheat43_path)
    aligned_raw, _aligned_indices, fallback_count = align_raw_to_wheat43(raw_df, wheat43_df)
    rebuilt = rebuild_wheat_43_features(aligned_raw)

    shared_rows = min(len(rebuilt), len(wheat43_df))
    rebuilt_values = rebuilt.iloc[:shared_rows, :43].to_numpy(dtype=float)
    target_values = wheat43_df.iloc[:shared_rows, :43].to_numpy(dtype=float)
    diff = np.abs(rebuilt_values - target_values)
    if diff.size:
        max_abs_diff = float(np.nanmax(diff))
        mean_abs_diff = float(np.nanmean(diff))
        keep_cols = [idx for idx in range(43) if idx != 5]
        diff_excluding_col5 = diff[:, keep_cols]
        max_abs_diff_excluding_col5 = float(np.nanmax(diff_excluding_col5))
        mean_abs_diff_excluding_col5 = float(np.nanmean(diff_excluding_col5))
    else:
        max_abs_diff = 0.0
        mean_abs_diff = 0.0
        max_abs_diff_excluding_col5 = 0.0
        mean_abs_diff_excluding_col5 = 0.0

    rebuilt_labels = rebuilt.iloc[:shared_rows, 43].to_numpy(dtype=int)
    target_labels = wheat43_df.iloc[:shared_rows, 43].to_numpy(dtype=int)
    label_match_rate = float((rebuilt_labels == target_labels).mean()) if shared_rows else 0.0
    audit_pass = label_match_rate == 1.0 and mean_abs_diff_excluding_col5 <= args.mean_threshold_excluding_col5
    notes = ""
    if not audit_pass:
        notes = "rebuild differs from sampled_wheat_43; inspect before formal merge-cache training"
    elif mean_abs_diff > args.mean_threshold:
        notes = "core 42 feature columns pass; historical geometry column 5 remains approximate"
    if fallback_count:
        notes = (notes + "; " if notes else "") + "coordinate alignment used fallback"

    return dict(
        file=raw_path.name,
        rows_raw=len(raw_df),
        rows_43=len(wheat43_df),
        aligned_rows=shared_rows,
        alignment_fallback_count=fallback_count,
        max_abs_diff=max_abs_diff,
        mean_abs_diff=mean_abs_diff,
        max_abs_diff_excluding_col5=max_abs_diff_excluding_col5,
        mean_abs_diff_excluding_col5=mean_abs_diff_excluding_col5,
        label_match_rate=label_match_rate,
        audit_pass=audit_pass,
        notes=notes,
    )


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    raw_dir = Path(args.raw_dir)
    wheat43_dir = Path(args.wheat43_dir)
    rows = []
    for index, raw_path in enumerate(sorted(raw_dir.glob("*.xlsx"))):
        if args.limit_files is not None and index >= args.limit_files:
            break
        wheat43_path = wheat43_dir / raw_path.name
        if not wheat43_path.exists():
            rows.append(
                dict(
                    file=raw_path.name,
                    rows_raw="",
                    rows_43="",
                    aligned_rows="",
                    alignment_fallback_count="",
                    max_abs_diff="",
                    mean_abs_diff="",
                    max_abs_diff_excluding_col5="",
                    mean_abs_diff_excluding_col5="",
                    label_match_rate="",
                    audit_pass=False,
                    notes="sampled_wheat_43 file missing",
                )
            )
            continue
        row = audit_file(raw_path, wheat43_path, args)
        rows.append(row)
        print(
            f"{row['file']} mean_abs_diff={row['mean_abs_diff']:.6f} "
            f"mean_ex_col5={row['mean_abs_diff_excluding_col5']:.6f} "
            f"label_match={row['label_match_rate']:.6f} pass={row['audit_pass']}",
            flush=True,
        )

    output = Path(args.output)
    write_csv(output, rows)
    pass_count = sum(1 for row in rows if row["audit_pass"])
    print(f"WROTE {output} files={len(rows)} pass={pass_count}/{len(rows)}")


if __name__ == "__main__":
    main()
