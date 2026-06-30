import argparse
import csv
import json
import math
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.neighbors import BallTree
from sklearn.preprocessing import MinMaxScaler

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.wheat_feature_rebuild import (
    RAW_COLUMNS,
    align_raw_to_wheat43,
    read_raw_wheat,
    read_wheat_43,
    rebuild_wheat_43_features,
)


CONFIRMED_TIME_DISORDER = {
    "train": {
        "wheat_1_harvestor_15.xlsx",
        "wheat_1_harvestor_17.xlsx",
        "wheat_1_harvestor_3.xlsx",
        "wheat_1_harvestor_94.xlsx",
    },
    "valid": {
        "wheat_1_harvestor_77.xlsx",
    },
    "test": {
        "wheat_1_harvestor_114.xlsx",
        "wheat_1_harvestor_51.xlsx",
    },
}

SPLITS = ("train", "valid", "test")

AUDIT_FIELDS = [
    "split",
    "file",
    "fixed_by_time_sort",
    "raw_rows",
    "original_43_rows",
    "strict_43_rows",
    "row_count_match",
    "original_road",
    "original_field",
    "strict_road",
    "strict_field",
    "label_count_match",
    "align_fallback_count",
    "max_coord_diff",
    "time_inversions_before",
    "time_inversions_after",
    "original_adj_edges",
    "strict_adj_edges",
    "strict_temporal_gt50_edges",
    "notes",
]

CACHE_FIELDS = [
    "split",
    "original_points",
    "strict_points",
    "points_match",
    "original_samples",
    "strict_samples",
    "original_edges",
    "strict_edges",
    "notes",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Build strict time-fixed wheat cache without changing sample subset.")
    parser.add_argument("--raw_dir", default="wheat/sampled_wheat")
    parser.add_argument("--wheat43_dir", default="wheat/sampled_wheat_43")
    parser.add_argument("--adj_dir", default="wheat/sampled_wheat_adj")
    parser.add_argument("--split_dir", default="wheat/Non-Identically_Distributed_Coco")
    parser.add_argument("--wheat43_out_dir", default="wheat/sampled_wheat_43_time_fixed_strict")
    parser.add_argument("--adj_out_dir", default="wheat/sampled_wheat_adj_time_fixed_strict")
    parser.add_argument("--split_out_dir", default="wheat/Non-Identically_Distributed_Coco_time_fixed_strict")
    parser.add_argument("--cache_out_dir", default="cache/wheat_non_iid_time_fixed_strict")
    parser.add_argument("--diagnostics_dir", default="diagnostics/wheat_time_order_fix_strict")
    parser.add_argument("--analysis_path", default="analysis/wheat_time_order_fix_strict_report.md")
    parser.add_argument("--spatial_topk", type=int, default=3)
    parser.add_argument("--max_len", type=int, default=1000)
    parser.add_argument("--drop_rate", type=float, default=0.0)
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def ensure_dirs(args):
    for path in [
        args.wheat43_out_dir,
        args.adj_out_dir,
        args.split_out_dir,
        args.cache_out_dir,
        args.diagnostics_dir,
        Path(args.analysis_path).parent,
    ]:
        Path(path).mkdir(parents=True, exist_ok=True)


def read_split(split_dir, split):
    path = Path(split_dir) / f"sampled_wheat_43_{split}.json"
    return json.loads(path.read_text(encoding="utf-8"))


def split_files(split_dir, split):
    data = read_split(split_dir, split)
    return [item["file_name"] for item in data["trajectories"]]


def write_csv(path, rows, fields):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def is_confirmed(split, file_name):
    return file_name in CONFIRMED_TIME_DISORDER.get(split, set())


def parse_time(series):
    return pd.to_datetime(series, errors="coerce", format="mixed")


def time_inversions(raw_df):
    times = parse_time(raw_df["时间"])
    return int((times.diff().dt.total_seconds().dropna() < 0).sum())


def label_counts_43(df):
    labels = df[43].astype(int)
    return int((labels == 0).sum()), int((labels == 1).sum())


def haversine_m(lon1, lat1, lon2, lat2):
    radius = 6371000.0
    phi1 = math.radians(float(lat1))
    phi2 = math.radians(float(lat2))
    dphi = math.radians(float(lat2) - float(lat1))
    dlambda = math.radians(float(lon2) - float(lon1))
    a = math.sin(dphi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
    return 2.0 * radius * math.atan2(math.sqrt(a), math.sqrt(max(1.0 - a, 0.0)))


def edge_distance_stats(edges, coords_lonlat):
    if len(edges) == 0:
        return dict(gt50=0)
    gt50 = 0
    for src, dst in edges:
        lon1, lat1 = coords_lonlat[int(src)]
        lon2, lat2 = coords_lonlat[int(dst)]
        if haversine_m(lon1, lat1, lon2, lat2) > 50.0:
            gt50 += 1
    return dict(gt50=gt50)


def build_edges(coords_lonlat, spatial_topk):
    n = int(len(coords_lonlat))
    edges = set()
    for idx in range(n):
        for gap in (1, 2):
            dst = idx + gap
            if dst < n:
                edges.add((idx, dst))
                edges.add((dst, idx))
    if n > 1 and spatial_topk > 0:
        latlon_rad = np.deg2rad(np.column_stack([coords_lonlat[:, 1], coords_lonlat[:, 0]]))
        tree = BallTree(latlon_rad, metric="haversine")
        k = min(spatial_topk + 1, n)
        _dist, indices = tree.query(latlon_rad, k=k)
        for src in range(n):
            for dst in indices[src, 1:]:
                if int(dst) != src:
                    edges.add((src, int(dst)))
    if not edges:
        return np.empty((0, 2), dtype=np.int64)
    return np.asarray(sorted(edges), dtype=np.int64)


def strict_fix_one_file(args, split, file_name):
    raw_path = Path(args.raw_dir) / file_name
    old_43_path = Path(args.wheat43_dir) / file_name
    old_adj_path = Path(args.adj_dir) / file_name.replace(".xlsx", ".npy")
    out_43_path = Path(args.wheat43_out_dir) / file_name
    out_adj_path = Path(args.adj_out_dir) / file_name.replace(".xlsx", ".npy")

    raw_df = read_raw_wheat(raw_path)
    old_43 = read_wheat_43(old_43_path)
    old_adj = np.load(old_adj_path)
    old_road, old_field = label_counts_43(old_43)

    fixed = is_confirmed(split, file_name)
    align_fallback = 0
    max_coord_diff = 0.0
    before_inv = ""
    after_inv = ""
    notes = "copied_original_43_and_adj"
    if fixed:
        aligned_raw, _indices, align_fallback = align_raw_to_wheat43(raw_df, old_43)
        coord_diff = np.abs(aligned_raw["经度"].to_numpy(dtype=float) - old_43[41].to_numpy(dtype=float))
        coord_diff += np.abs(aligned_raw["纬度"].to_numpy(dtype=float) - old_43[42].to_numpy(dtype=float))
        max_coord_diff = float(coord_diff.max()) if len(coord_diff) else 0.0
        before_inv = time_inversions(aligned_raw)
        sortable = aligned_raw.copy()
        sortable["_time_sort_key"] = parse_time(sortable["时间"])
        sortable["_original_subset_order"] = np.arange(len(sortable), dtype=np.int64)
        sorted_raw = sortable.sort_values(["_time_sort_key", "_original_subset_order"], kind="mergesort")
        sorted_raw = sorted_raw.drop(columns=["_time_sort_key", "_original_subset_order"]).reset_index(drop=True)
        after_inv = time_inversions(sorted_raw)
        strict_43 = rebuild_wheat_43_features(sorted_raw[RAW_COLUMNS].copy())
        coords = strict_43[[41, 42]].to_numpy(dtype=float)
        strict_adj = build_edges(coords, args.spatial_topk)
        notes = "matched_original_43_subset_to_raw_rows_then_stable_sorted_by_time"
        if not args.dry_run:
            strict_43.to_excel(out_43_path, index=False)
            np.save(out_adj_path, strict_adj)
    else:
        strict_43 = old_43
        strict_adj = old_adj
        if not args.dry_run:
            shutil.copy2(old_43_path, out_43_path)
            shutil.copy2(old_adj_path, out_adj_path)

    strict_road, strict_field = label_counts_43(strict_43)
    temporal_edges = strict_adj[np.abs(strict_adj[:, 0] - strict_adj[:, 1]) <= 2] if len(strict_adj) else strict_adj
    strict_temporal_gt50 = edge_distance_stats(temporal_edges, strict_43[[41, 42]].to_numpy(dtype=float))["gt50"]

    return {
        "split": split,
        "file": file_name,
        "fixed_by_time_sort": fixed,
        "raw_rows": len(raw_df),
        "original_43_rows": len(old_43),
        "strict_43_rows": len(strict_43),
        "row_count_match": len(old_43) == len(strict_43),
        "original_road": old_road,
        "original_field": old_field,
        "strict_road": strict_road,
        "strict_field": strict_field,
        "label_count_match": old_road == strict_road and old_field == strict_field,
        "align_fallback_count": align_fallback,
        "max_coord_diff": max_coord_diff,
        "time_inversions_before": before_inv,
        "time_inversions_after": after_inv,
        "original_adj_edges": len(old_adj),
        "strict_adj_edges": len(strict_adj),
        "strict_temporal_gt50_edges": strict_temporal_gt50,
        "notes": notes,
    }


def write_split_jsons(args):
    for split in SPLITS:
        data = read_split(args.split_dir, split)
        if args.dry_run:
            continue
        for name in [
            f"sampled_wheat_43_{split}.json",
            f"sampled_wheat_43_time_fixed_strict_{split}.json",
        ]:
            (Path(args.split_out_dir) / name).write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def minmax_normalize(features):
    if len(features) == 0:
        return features.astype(np.float32)
    return MinMaxScaler().fit_transform(features).astype(np.float32)


def crop_cache_items(features, labels, edge_index, trace_id, coordinates, max_len, drop_rate):
    items = []
    total = int(features.shape[0])
    for start in range(0, total, max_len):
        end = min(start + max_len, total)
        if end - start < drop_rate * max_len:
            continue
        mask = (
            (edge_index[0] >= start)
            & (edge_index[0] < end)
            & (edge_index[1] >= start)
            & (edge_index[1] < end)
        )
        cropped_edge_index = edge_index[:, mask] - start
        items.append(
            dict(
                points=torch.from_numpy(features[start:end]).to(torch.float32),
                labels=torch.from_numpy(labels[start:end].reshape(-1, 1).astype(np.uint8)),
                edge_index=cropped_edge_index.to(torch.long),
                trace_id=trace_id,
                coordinates=torch.from_numpy(coordinates[start:end].astype(np.float32)),
            )
        )
    return items


def cache_stats(cache_path):
    if not Path(cache_path).exists():
        return dict(samples=0, points=0, edges=0)
    data = torch.load(cache_path, map_location="cpu")
    return dict(
        samples=len(data),
        points=sum(int(item["points"].shape[0]) for item in data),
        edges=sum(int(item["edge_index"].shape[1]) for item in data),
    )


def rebuild_cache(args):
    rows = []
    for split in SPLITS:
        items = []
        total_points = 0
        total_edges = 0
        for file_name in split_files(args.split_dir, split):
            wheat43_path = Path(args.wheat43_out_dir) / file_name
            adj_path = Path(args.adj_out_dir) / file_name.replace(".xlsx", ".npy")
            wheat43 = read_wheat_43(wheat43_path)
            raw_features = wheat43.iloc[:, :43].to_numpy(dtype=np.float32)
            features = minmax_normalize(raw_features)
            labels = wheat43[43].to_numpy(dtype=np.uint8)
            coordinates = wheat43[[41, 42]].to_numpy(dtype=np.float32)
            edges_np = np.load(adj_path)
            edge_index = torch.from_numpy(edges_np.astype(np.int64)).t().contiguous()
            trace_id = str(wheat43_path)
            items.extend(crop_cache_items(features, labels, edge_index, trace_id, coordinates, args.max_len, args.drop_rate))
            total_points += int(len(features))
            total_edges += int(edge_index.shape[1])
        if not args.dry_run:
            torch.save(items, Path(args.cache_out_dir) / f"{split}.pt")
        old_stats = cache_stats(Path("cache/wheat_non_iid") / f"{split}.pt")
        rows.append(
            {
                "split": split,
                "original_points": old_stats["points"],
                "strict_points": total_points,
                "points_match": old_stats["points"] == total_points,
                "original_samples": old_stats["samples"],
                "strict_samples": len(items),
                "original_edges": old_stats["edges"],
                "strict_edges": total_edges,
                "notes": f"max_len={args.max_len};drop_rate={args.drop_rate};per_trace_minmax",
            }
        )
    if not args.dry_run:
        write_csv(Path(args.diagnostics_dir) / "cache_strict_audit.csv", rows, CACHE_FIELDS)
    return rows


def strict_validate(audit_rows, cache_rows):
    bad_rows = [row for row in audit_rows if not row["row_count_match"] or not row["label_count_match"]]
    bad_match = [row for row in audit_rows if row["fixed_by_time_sort"] and (int(row["align_fallback_count"]) != 0 or float(row["max_coord_diff"]) != 0.0)]
    bad_cache = [row for row in cache_rows if not row["points_match"]]
    test_row = next(row for row in cache_rows if row["split"] == "test")
    if int(test_row["strict_points"]) != 95361:
        bad_cache.append({**test_row, "notes": "test_points_not_95361"})
    if bad_rows or bad_match or bad_cache:
        raise SystemExit(
            "STRICT_TIME_FIXED_VALIDATION_FAILED: "
            f"row_or_label_bad={len(bad_rows)} align_bad={len(bad_match)} cache_bad={len(bad_cache)}"
        )


def make_report(args, audit_rows, cache_rows):
    fixed_rows = [row for row in audit_rows if row["fixed_by_time_sort"]]
    cache_line = "; ".join(
        f"{row['split']}: original={row['original_points']} strict={row['strict_points']} samples={row['strict_samples']}"
        for row in cache_rows
    )
    fixed_lines = [
        "| split | file | raw rows | original 43 rows | inversions before->after | align fallback | adj edges old->strict | temporal gt50 |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in fixed_rows:
        fixed_lines.append(
            f"| {row['split']} | {row['file']} | {row['raw_rows']} | {row['original_43_rows']} | "
            f"{row['time_inversions_before']}->{row['time_inversions_after']} | {row['align_fallback_count']} | "
            f"{row['original_adj_edges']}->{row['strict_adj_edges']} | {row['strict_temporal_gt50_edges']} |"
        )
    text = f"""# wheat time_fixed_strict report

本次严格版只修复原 `sampled_wheat_43` 已经包含的点，不从 `sampled_wheat` 全量 raw 行扩充样本。

## 核心规则

- 非明显乱序轨迹: 直接复制 `wheat/sampled_wheat_43` 和 `wheat/sampled_wheat_adj`。
- 7 条明显乱序轨迹: 将原 43 维点按经纬度+标签精确匹配回 raw 行，只保留这些旧 43 维已有点，再按时间稳定排序，重新计算 43 维并重建 adj。
- 严格校验行数、road/field 标签数量、split 总点数完全等于原 cache。

## 修复轨迹

{chr(10).join(fixed_lines)}

## cache 校验

`{cache_line}`。

test 总点数必须为 `95361`，当前为 `{next(row for row in cache_rows if row['split'] == 'test')['strict_points']}`。

## 输出

- 43 维: `{args.wheat43_out_dir}`
- adj: `{args.adj_out_dir}`
- cache: `{args.cache_out_dir}`
- diagnostics: `{args.diagnostics_dir}`

结论: strict 数据构建通过校验，可用于公平重训。
"""
    path = Path(args.analysis_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not args.dry_run:
        path.write_text(text, encoding="utf-8")


def main():
    args = parse_args()
    ensure_dirs(args)
    audit_rows = []
    for split in SPLITS:
        for file_name in split_files(args.split_dir, split):
            audit_rows.append(strict_fix_one_file(args, split, file_name))
    if not args.dry_run:
        write_csv(Path(args.diagnostics_dir) / "time_fixed_strict_audit.csv", audit_rows, AUDIT_FIELDS)
    write_split_jsons(args)
    cache_rows = rebuild_cache(args)
    strict_validate(audit_rows, cache_rows)
    make_report(args, audit_rows, cache_rows)

    print("STRICT_TIME_FIXED_BUILD_OK")
    for row in cache_rows:
        print(
            f"{row['split']}: original_points={row['original_points']} "
            f"strict_points={row['strict_points']} strict_samples={row['strict_samples']}"
        )
    print(f"wheat43_out_dir={args.wheat43_out_dir}")
    print(f"adj_out_dir={args.adj_out_dir}")
    print(f"cache_out_dir={args.cache_out_dir}")
    print(f"analysis_path={args.analysis_path}")


if __name__ == "__main__":
    main()
