import argparse
import csv
import json
import math
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.neighbors import BallTree
from sklearn.preprocessing import MinMaxScaler

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.wheat_feature_rebuild import RAW_COLUMNS, rebuild_wheat_43_features


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

TIME_ORDER_AUDIT_FIELDS = [
    "split",
    "file",
    "rows_before",
    "rows_after",
    "label_road_before",
    "label_field_before",
    "label_road_after",
    "label_field_after",
    "raw_time_inversions_before",
    "raw_time_inversions_after",
    "raw_gt20_step_count_before",
    "raw_gt50_step_count_before",
    "raw_gt100_step_count_before",
    "raw_gt20_step_count_after",
    "raw_gt50_step_count_after",
    "raw_gt100_step_count_after",
    "mean_step_m_before",
    "mean_step_m_after",
    "p95_step_m_before",
    "p95_step_m_after",
    "max_step_m_before",
    "max_step_m_after",
    "fixed_by_time_sort",
    "notes",
]

ADJ_AUDIT_FIELDS = [
    "split",
    "file",
    "rows",
    "edge_count",
    "temporal_edge_count_abs_gap_le2",
    "spatial_or_extra_edge_count_abs_gap_gt2",
    "temporal_edge_mean_distance_m",
    "temporal_edge_p95_distance_m",
    "temporal_edge_max_distance_m",
    "temporal_edge_gt20_count",
    "temporal_edge_gt50_count",
    "temporal_edge_gt100_count",
    "extra_edge_mean_distance_m",
    "extra_edge_p95_distance_m",
    "extra_edge_max_distance_m",
    "extra_edge_gt50_count",
    "notes",
]

CACHE_AUDIT_FIELDS = [
    "split",
    "num_samples",
    "total_points",
    "total_edges",
    "total_temporal_edges_abs_gap_le2",
    "total_extra_edges_abs_gap_gt2",
    "notes",
]

READUIT_FIELDS = [
    "split",
    "file",
    "fixed_confirmed",
    "before_time_inversions",
    "after_time_inversions",
    "before_gt50_step_count",
    "after_gt50_step_count",
    "feat43_gt50_step_count",
    "temporal_edge_gt50_count",
    "before_max_step_m",
    "after_max_step_m",
    "feat43_max_step_m",
    "temporal_edge_max_distance_m",
    "notes",
]

GRAPH_CACHE_AUDIT_FIELDS = [
    "status",
    "pretrained_path",
    "output_cache",
    "cache_dir",
    "splits",
    "topk",
    "notes",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Safely fix confirmed wheat time-order disorder copies.")
    parser.add_argument("--raw_dir", default="wheat/sampled_wheat")
    parser.add_argument("--wheat43_dir", default="wheat/sampled_wheat_43")
    parser.add_argument("--adj_dir", default="wheat/sampled_wheat_adj")
    parser.add_argument("--split_dir", default="wheat/Non-Identically_Distributed_Coco")
    parser.add_argument("--raw_fixed_dir", default="wheat/sampled_wheat_time_fixed")
    parser.add_argument("--wheat43_fixed_dir", default="wheat/sampled_wheat_43_time_fixed")
    parser.add_argument("--adj_fixed_dir", default="wheat/sampled_wheat_adj_time_fixed")
    parser.add_argument("--split_fixed_dir", default="wheat/Non-Identically_Distributed_Coco_time_fixed")
    parser.add_argument("--cache_fixed_dir", default="cache/wheat_non_iid_time_fixed")
    parser.add_argument("--graph_cache_fixed_dir", default="cache/pretrained_graphs/PT2G_topk3_time_fixed")
    parser.add_argument("--diagnostics_dir", default="diagnostics/wheat_time_order_fix")
    parser.add_argument("--analysis_dir", default="analysis/wheat_time_order_fix")
    parser.add_argument("--pack_path", default="analysis_packs/wheat_time_order_fix_for_chatgpt.zip")
    parser.add_argument("--spatial_topk", type=int, default=3)
    parser.add_argument("--max_len", type=int, default=1000)
    parser.add_argument("--drop_rate", type=float, default=0.0)
    parser.add_argument("--pretrained_path", default="weights/PT2_edge_weight_pretrain.pt")
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def ensure_dirs(args):
    for path in [
        args.raw_fixed_dir,
        args.wheat43_fixed_dir,
        args.adj_fixed_dir,
        args.split_fixed_dir,
        args.cache_fixed_dir,
        args.graph_cache_fixed_dir,
        args.diagnostics_dir,
        args.analysis_dir,
        Path(args.pack_path).parent,
    ]:
        Path(path).mkdir(parents=True, exist_ok=True)


def load_split_json(split_dir, split, prefix="sampled_wheat_43"):
    path = Path(split_dir) / f"{prefix}_{split}.json"
    return json.loads(path.read_text(encoding="utf-8"))


def split_files(split_dir, split):
    data = load_split_json(split_dir, split)
    return [item["file_name"] for item in data["trajectories"]]


def write_csv(path, rows, fields):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def haversine_m(lon1, lat1, lon2, lat2):
    radius = 6371000.0
    phi1 = math.radians(float(lat1))
    phi2 = math.radians(float(lat2))
    dphi = math.radians(float(lat2) - float(lat1))
    dlambda = math.radians(float(lon2) - float(lon1))
    a = math.sin(dphi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
    return 2.0 * radius * math.atan2(math.sqrt(a), math.sqrt(max(1.0 - a, 0.0)))


def step_distances(lon, lat):
    lon = np.asarray(lon, dtype=float)
    lat = np.asarray(lat, dtype=float)
    if len(lon) < 2:
        return np.asarray([], dtype=float)
    return np.asarray([haversine_m(lon[idx - 1], lat[idx - 1], lon[idx], lat[idx]) for idx in range(1, len(lon))])


def distance_stats_from_lonlat(lon, lat):
    distances = step_distances(lon, lat)
    if len(distances) == 0:
        return dict(
            mean_step_m=0.0,
            p95_step_m=0.0,
            max_step_m=0.0,
            gt20_step_count=0,
            gt50_step_count=0,
            gt100_step_count=0,
        )
    return dict(
        mean_step_m=float(distances.mean()),
        p95_step_m=float(np.quantile(distances, 0.95)),
        max_step_m=float(distances.max()),
        gt20_step_count=int((distances > 20).sum()),
        gt50_step_count=int((distances > 50).sum()),
        gt100_step_count=int((distances > 100).sum()),
    )


def read_raw_full(path):
    df = pd.read_excel(path)
    missing = [column for column in RAW_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"RAW_WHEAT_COLUMNS_MISSING: {path} missing={missing}")
    return df


def raw_stats(df):
    times = pd.to_datetime(df["时间"], errors="coerce", format="mixed")
    labels = df["标签"].astype(int)
    stats = distance_stats_from_lonlat(df["经度"].astype(float), df["纬度"].astype(float))
    return dict(
        rows=len(df),
        label_road=int((labels == 0).sum()),
        label_field=int((labels == 1).sum()),
        time_inversions=int((times.diff().dt.total_seconds().dropna() < 0).sum()),
        duplicate_times=int(times.duplicated().sum()),
        **stats,
    )


def is_confirmed(split, file_name):
    return file_name in CONFIRMED_TIME_DISORDER.get(split, set())


def fix_raw_files(args):
    rows = []
    for split in SPLITS:
        for file_name in split_files(args.split_dir, split):
            src = Path(args.raw_dir) / file_name
            dst = Path(args.raw_fixed_dir) / file_name
            before_df = read_raw_full(src)
            before = raw_stats(before_df)
            fixed = is_confirmed(split, file_name)
            if fixed:
                times = pd.to_datetime(before_df["时间"], errors="coerce", format="mixed")
                after_df = before_df.assign(_time_sort_key=times).sort_values("_time_sort_key", kind="mergesort")
                after_df = after_df.drop(columns=["_time_sort_key"]).reset_index(drop=True)
                if not args.dry_run:
                    after_df.to_excel(dst, index=False)
                notes = "sorted_by_time_mergesort"
            else:
                after_df = before_df
                if not args.dry_run:
                    shutil.copy2(src, dst)
                notes = "copied_without_reorder"
            after = raw_stats(after_df)
            row = dict(
                split=split,
                file=file_name,
                rows_before=before["rows"],
                rows_after=after["rows"],
                label_road_before=before["label_road"],
                label_field_before=before["label_field"],
                label_road_after=after["label_road"],
                label_field_after=after["label_field"],
                raw_time_inversions_before=before["time_inversions"],
                raw_time_inversions_after=after["time_inversions"],
                raw_gt20_step_count_before=before["gt20_step_count"],
                raw_gt50_step_count_before=before["gt50_step_count"],
                raw_gt100_step_count_before=before["gt100_step_count"],
                raw_gt20_step_count_after=after["gt20_step_count"],
                raw_gt50_step_count_after=after["gt50_step_count"],
                raw_gt100_step_count_after=after["gt100_step_count"],
                mean_step_m_before=before["mean_step_m"],
                mean_step_m_after=after["mean_step_m"],
                p95_step_m_before=before["p95_step_m"],
                p95_step_m_after=after["p95_step_m"],
                max_step_m_before=before["max_step_m"],
                max_step_m_after=after["max_step_m"],
                fixed_by_time_sort=fixed,
                notes=notes,
            )
            rows.append(row)
    if not args.dry_run:
        write_csv(Path(args.diagnostics_dir) / "time_order_fix_audit.csv", rows, TIME_ORDER_AUDIT_FIELDS)
    return rows


def rebuild_wheat43(args):
    rows = []
    for split in SPLITS:
        for file_name in split_files(args.split_dir, split):
            raw_df = read_raw_full(Path(args.raw_fixed_dir) / file_name)
            fixed_raw = raw_df[RAW_COLUMNS].copy()
            wheat43 = rebuild_wheat_43_features(fixed_raw)
            if not args.dry_run:
                wheat43.to_excel(Path(args.wheat43_fixed_dir) / file_name, index=False)
            rows.append(
                dict(
                    split=split,
                    file=file_name,
                    rows=len(wheat43),
                    label_road=int((wheat43[43].astype(int) == 0).sum()),
                    label_field=int((wheat43[43].astype(int) == 1).sum()),
                    notes="rebuilt_from_time_fixed_raw",
                )
            )
    return rows


def build_edges(coords_lonlat, spatial_topk):
    n = int(len(coords_lonlat))
    edges = set()
    for idx in range(n):
        for gap in (1, 2):
            j = idx + gap
            if j < n:
                edges.add((idx, j))
                edges.add((j, idx))
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


def edge_distance_stats(edges, coords_lonlat):
    if len(edges) == 0:
        return dict(mean=0.0, p95=0.0, max=0.0, gt20=0, gt50=0, gt100=0)
    distances = []
    for src, dst in edges:
        lon1, lat1 = coords_lonlat[int(src)]
        lon2, lat2 = coords_lonlat[int(dst)]
        distances.append(haversine_m(lon1, lat1, lon2, lat2))
    distances = np.asarray(distances, dtype=float)
    return dict(
        mean=float(distances.mean()),
        p95=float(np.quantile(distances, 0.95)),
        max=float(distances.max()),
        gt20=int((distances > 20).sum()),
        gt50=int((distances > 50).sum()),
        gt100=int((distances > 100).sum()),
    )


def audit_edges(split, file_name, edges, coords_lonlat):
    gap = np.abs(edges[:, 0] - edges[:, 1]) if len(edges) else np.asarray([], dtype=int)
    temporal = edges[gap <= 2] if len(edges) else np.empty((0, 2), dtype=np.int64)
    extra = edges[gap > 2] if len(edges) else np.empty((0, 2), dtype=np.int64)
    temporal_stats = edge_distance_stats(temporal, coords_lonlat)
    extra_stats = edge_distance_stats(extra, coords_lonlat)
    return dict(
        split=split,
        file=file_name,
        rows=len(coords_lonlat),
        edge_count=len(edges),
        temporal_edge_count_abs_gap_le2=len(temporal),
        spatial_or_extra_edge_count_abs_gap_gt2=len(extra),
        temporal_edge_mean_distance_m=temporal_stats["mean"],
        temporal_edge_p95_distance_m=temporal_stats["p95"],
        temporal_edge_max_distance_m=temporal_stats["max"],
        temporal_edge_gt20_count=temporal_stats["gt20"],
        temporal_edge_gt50_count=temporal_stats["gt50"],
        temporal_edge_gt100_count=temporal_stats["gt100"],
        extra_edge_mean_distance_m=extra_stats["mean"],
        extra_edge_p95_distance_m=extra_stats["p95"],
        extra_edge_max_distance_m=extra_stats["max"],
        extra_edge_gt50_count=extra_stats["gt50"],
        notes=f"temporal_gap_1_2_plus_spatial_topk_{len(extra) and '3' or '0'}",
    )


def rebuild_adj(args):
    rows = []
    for split in SPLITS:
        for file_name in split_files(args.split_dir, split):
            wheat43 = pd.read_excel(Path(args.wheat43_fixed_dir) / file_name)
            wheat43 = wheat43.iloc[:, :44].copy()
            wheat43.columns = list(range(44))
            coords = wheat43[[41, 42]].to_numpy(dtype=float)
            edges = build_edges(coords, args.spatial_topk)
            if not args.dry_run:
                np.save(Path(args.adj_fixed_dir) / file_name.replace(".xlsx", ".npy"), edges)
            rows.append(audit_edges(split, file_name, edges, coords))
    if not args.dry_run:
        write_csv(Path(args.diagnostics_dir) / "adj_time_edge_audit.csv", rows, ADJ_AUDIT_FIELDS)
    return rows


def write_split_jsons(args):
    for split in SPLITS:
        data = load_split_json(args.split_dir, split)
        target_time_fixed = Path(args.split_fixed_dir) / f"sampled_wheat_43_time_fixed_{split}.json"
        target_compatible = Path(args.split_fixed_dir) / f"sampled_wheat_43_{split}.json"
        if not args.dry_run:
            target_time_fixed.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
            target_compatible.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


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


def rebuild_cache(args):
    audit_rows = []
    for split in SPLITS:
        split_items = []
        total_points = 0
        total_edges = 0
        total_temporal = 0
        total_extra = 0
        for file_name in split_files(args.split_dir, split):
            wheat43_path = Path(args.wheat43_fixed_dir) / file_name
            adj_path = Path(args.adj_fixed_dir) / file_name.replace(".xlsx", ".npy")
            wheat43 = pd.read_excel(wheat43_path)
            wheat43 = wheat43.iloc[:, :44].copy()
            wheat43.columns = list(range(44))
            raw_features = wheat43.iloc[:, :43].to_numpy(dtype=np.float32)
            features = minmax_normalize(raw_features)
            labels = wheat43[43].to_numpy(dtype=np.uint8)
            coordinates = wheat43[[41, 42]].to_numpy(dtype=np.float32)
            edges_np = np.load(adj_path)
            edge_index = torch.from_numpy(edges_np.astype(np.int64)).t().contiguous()
            trace_id = str(wheat43_path)
            items = crop_cache_items(features, labels, edge_index, trace_id, coordinates, args.max_len, args.drop_rate)
            split_items.extend(items)
            total_points += int(len(features))
            total_edges += int(edge_index.shape[1])
            if edge_index.numel():
                gap = torch.abs(edge_index[0] - edge_index[1])
                total_temporal += int((gap <= 2).sum().item())
                total_extra += int((gap > 2).sum().item())
        if not args.dry_run:
            torch.save(split_items, Path(args.cache_fixed_dir) / f"{split}.pt")
        audit_rows.append(
            dict(
                split=split,
                num_samples=len(split_items),
                total_points=total_points,
                total_edges=total_edges,
                total_temporal_edges_abs_gap_le2=total_temporal,
                total_extra_edges_abs_gap_gt2=total_extra,
                notes=f"max_len={args.max_len};drop_rate={args.drop_rate};points_minmax_per_trace",
            )
        )
    if not args.dry_run:
        write_csv(Path(args.diagnostics_dir) / "cache_time_fixed_audit.csv", audit_rows, CACHE_AUDIT_FIELDS)
    return audit_rows


def rebuild_graph_cache(args):
    audit_path = Path(args.diagnostics_dir) / "pt2g_graph_cache_audit.csv"
    pretrained_path = Path(args.pretrained_path)
    if not pretrained_path.exists():
        rows = [
            dict(
                status="skipped_missing_pretrained_checkpoint",
                pretrained_path=str(pretrained_path),
                output_cache=args.graph_cache_fixed_dir,
                cache_dir=args.cache_fixed_dir,
                splits="train,valid,test",
                topk=3,
                notes="PT2G graph cache was not rebuilt because weights/PT2_edge_weight_pretrain.pt is absent locally.",
            )
        ]
        write_csv(audit_path, rows, GRAPH_CACHE_AUDIT_FIELDS)
        (Path(args.graph_cache_fixed_dir) / "README_NOT_REBUILT.txt").write_text(rows[0]["notes"] + "\n", encoding="utf-8")
        return rows

    command = [
        sys.executable,
        "scripts/export_pretrained_graph_edges.py",
        "--pretrained_path",
        str(pretrained_path),
        "--pretrain_mode",
        "edge_weight",
        "--output_cache",
        args.graph_cache_fixed_dir,
        "--topk",
        "3",
        "--split",
        "train,valid,test",
        "--cache_dir",
        args.cache_fixed_dir,
        "--device",
        "cpu",
    ]
    try:
        result = subprocess.run(command, cwd=REPO_ROOT, text=True, capture_output=True, check=True)
        rows = [
            dict(
                status="rebuilt",
                pretrained_path=str(pretrained_path),
                output_cache=args.graph_cache_fixed_dir,
                cache_dir=args.cache_fixed_dir,
                splits="train,valid,test",
                topk=3,
                notes=result.stdout[-500:],
            )
        ]
    except subprocess.CalledProcessError as exc:
        rows = [
            dict(
                status="failed",
                pretrained_path=str(pretrained_path),
                output_cache=args.graph_cache_fixed_dir,
                cache_dir=args.cache_fixed_dir,
                splits="train,valid,test",
                topk=3,
                notes=(exc.stderr or exc.stdout or str(exc))[-1000:],
            )
        ]
    write_csv(audit_path, rows, GRAPH_CACHE_AUDIT_FIELDS)
    return rows


def reaudit(args, time_rows, adj_rows):
    time_by_key = {(row["split"], row["file"]): row for row in time_rows}
    adj_by_key = {(row["split"], row["file"]): row for row in adj_rows}
    rows = []
    for split in SPLITS:
        for file_name in split_files(args.split_dir, split):
            wheat43 = pd.read_excel(Path(args.wheat43_fixed_dir) / file_name)
            wheat43 = wheat43.iloc[:, :44].copy()
            wheat43.columns = list(range(44))
            feat_stats = distance_stats_from_lonlat(wheat43[41].astype(float), wheat43[42].astype(float))
            time_row = time_by_key[(split, file_name)]
            adj_row = adj_by_key[(split, file_name)]
            fixed = is_confirmed(split, file_name)
            notes = "confirmed_fixed" if fixed else "not_confirmed_copied_or_rebuilt_without_time_sort"
            rows.append(
                dict(
                    split=split,
                    file=file_name,
                    fixed_confirmed=fixed,
                    before_time_inversions=time_row["raw_time_inversions_before"],
                    after_time_inversions=time_row["raw_time_inversions_after"],
                    before_gt50_step_count=time_row["raw_gt50_step_count_before"],
                    after_gt50_step_count=time_row["raw_gt50_step_count_after"],
                    feat43_gt50_step_count=feat_stats["gt50_step_count"],
                    temporal_edge_gt50_count=adj_row["temporal_edge_gt50_count"],
                    before_max_step_m=time_row["max_step_m_before"],
                    after_max_step_m=time_row["max_step_m_after"],
                    feat43_max_step_m=feat_stats["max_step_m"],
                    temporal_edge_max_distance_m=adj_row["temporal_edge_max_distance_m"],
                    notes=notes,
                )
            )
    if not args.dry_run:
        write_csv(Path(args.diagnostics_dir) / "time_fixed_reaudit.csv", rows, READUIT_FIELDS)
    return rows


def fmt_float(value):
    return f"{float(value):.3f}"


def make_report(args, time_rows, adj_rows, cache_rows, graph_rows, reaudit_rows):
    fixed_rows = [row for row in time_rows if str(row["fixed_by_time_sort"]).lower() == "true" or row["fixed_by_time_sort"] is True]
    adj_by_key = {(row["split"], row["file"]): row for row in adj_rows}
    reaudit_by_key = {(row["split"], row["file"]): row for row in reaudit_rows}
    split_counts = {
        split: sum(1 for row in fixed_rows if row["split"] == split)
        for split in SPLITS
    }
    table_lines = [
        "| split | file | inv before->after | gt50 before->after | max step before->after | feat43 rebuilt | temporal gt50 |",
        "|---|---|---:|---:|---:|---|---:|",
    ]
    for row in fixed_rows:
        key = (row["split"], row["file"])
        adj = adj_by_key[key]
        table_lines.append(
            f"| {row['split']} | {row['file']} | {row['raw_time_inversions_before']}->{row['raw_time_inversions_after']} | "
            f"{row['raw_gt50_step_count_before']}->{row['raw_gt50_step_count_after']} | "
            f"{fmt_float(row['max_step_m_before'])}->{fmt_float(row['max_step_m_after'])} | yes | "
            f"{adj['temporal_edge_gt50_count']} |"
        )

    graph_status = graph_rows[0]["status"] if graph_rows else "unknown"
    graph_note = graph_rows[0]["notes"] if graph_rows else ""
    bad_after = [row for row in fixed_rows if int(row["raw_time_inversions_after"]) != 0]
    temporal_bad = [row for row in fixed_rows if int(adj_by_key[(row["split"], row["file"])]["temporal_edge_gt50_count"]) > 0]
    fixed_names = [f"{row['split']}/{row['file']}" for row in fixed_rows]
    cache_summary = "; ".join(
        f"{row['split']}: samples={row['num_samples']}, points={row['total_points']}, edges={row['total_edges']}"
        for row in cache_rows
    )

    text = f"""# wheat time order fix report

本次只修复 7 条已确认的明显时间乱序轨迹，不训练模型，不覆盖 `wheat/sampled_wheat`、`wheat/sampled_wheat_43`、`wheat/sampled_wheat_adj`。

## 修复范围

- train: `{split_counts['train']}` 条
- valid: `{split_counts['valid']}` 条
- test: `{split_counts['test']}` 条
- 修复轨迹: `{fixed_names}`

## 修复前后核心审计

{chr(10).join(table_lines)}

7 条明显时间乱序轨迹修复后 `raw_time_inversions_after == 0`: `{len(bad_after) == 0}`。
时间边 `gt50` 是否仍存在: `{len(temporal_bad) > 0}`；如果存在，通常表示排序后仍有真实或潜在大跳点，本次不直接修。

## 输出目录

- raw 副本: `{args.raw_fixed_dir}`
- 43 维副本: `{args.wheat43_fixed_dir}`
- adj 副本: `{args.adj_fixed_dir}`
- split json: `{args.split_fixed_dir}`
- torch cache: `{args.cache_fixed_dir}`
- PT2G graph cache: `{args.graph_cache_fixed_dir}`，状态: `{graph_status}`

## 为什么不能只排序 sampled_wheat

旧 `sampled_wheat_43` 已经按错误行序计算了速度差、方向差、滚动统计、几何项和坐标顺序。只排序 raw 文件不能修复已经固化在 43 维表里的历史特征，因此必须从排序后的 raw 重新生成 `sampled_wheat_43_time_fixed`。

## 为什么不能继续使用旧 sampled_wheat_adj

旧 adj 中的固定时间边 `i<->i+1`、`i<->i+2` 是按旧行号生成的。对时间乱序轨迹，这些边会连接远距离跳点，直接污染 GNN 的 `edge_index` 和 message passing。因此本次基于 time-fixed 行号重建时间边，并用每点 top-3 空间近邻补充空间边。

## 污染路径

- 固定时间边: 错误行序导致远距离点被当成相邻点连接。
- GNN edge_index: 旧 cache 继承错误 adj，GNN 聚合到错误上下文。
- 43 维节点特征: 前后点差分、滚动统计和几何项都依赖行顺序。
- PT2G graph cache: 补充边导出依赖旧节点特征、旧边和旧 encoder embedding。
- MSC/RC 局部运动特征: 训练时从 cache coordinates 按序计算 prev/next/local turn/density/stationary，乱序会污染这些局部运动特征；time_fixed cache 的 coordinates 已改为修复顺序。

## 为什么只修 7 条

本次只处理已确认“按时间排序后明显恢复正常”的时间乱序轨迹。潜在大跳点可能是真实转场、采样间隔、GNSS 漂移或拼接问题，需要单独定义规则和人工核验，当前不直接修改。

## cache 重建

`cache/wheat_non_iid_time_fixed` 已从 `sampled_wheat_43_time_fixed` 与 `sampled_wheat_adj_time_fixed` 重建。摘要: `{cache_summary}`。

## PT2G graph cache

状态: `{graph_status}`。
说明: `{graph_note}`。

## 是否可用于后续重新训练

time_fixed 数据可以用于后续重新训练 `PT2G_MSC_RC_v1`，但如果 `PT2G_topk3_time_fixed` 未成功重建，需要先补齐 `weights/PT2_edge_weight_pretrain.pt` 或在云端用 time_fixed cache 运行 `scripts/export_pretrained_graph_edges.py`。

推荐对比实验:

1. 原始数据 `PT2G_MSC_RC_v1`
2. time_fixed 数据 `PT2G_MSC_RC_v1`

比较 `macro-F1`、`road-F1`、`field-F1`、`road_as_field`、`field_as_road`、`long error segments`。

## 审计文件

- `diagnostics/wheat_time_order_fix/time_order_fix_audit.csv`
- `diagnostics/wheat_time_order_fix/adj_time_edge_audit.csv`
- `diagnostics/wheat_time_order_fix/cache_time_fixed_audit.csv`
- `diagnostics/wheat_time_order_fix/pt2g_graph_cache_audit.csv`
- `diagnostics/wheat_time_order_fix/time_fixed_reaudit.csv`

明确声明: 本脚本没有覆盖原始数据。
"""
    report_path = Path(args.analysis_dir) / "wheat_time_order_fix_report.md"
    if not args.dry_run:
        report_path.write_text(text, encoding="utf-8")
    return report_path


def make_pack(args):
    pack_path = Path(args.pack_path)
    if pack_path.exists():
        pack_path.unlink()
    paths = [
        "scripts/fix_wheat_time_order_confirmed.py",
        str(Path(args.analysis_dir) / "wheat_time_order_fix_report.md"),
    ]
    paths.extend(str(path) for path in sorted(Path(args.diagnostics_dir).glob("*")))
    paths.extend(str(path) for path in sorted(Path(args.split_fixed_dir).glob("*.json")))
    with zipfile.ZipFile(pack_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for item in paths:
            path = Path(item)
            if path.exists():
                zf.write(path, path.as_posix())
    return pack_path


def main():
    args = parse_args()
    ensure_dirs(args)
    print("STEP 1/8 fixing raw order copies", flush=True)
    time_rows = fix_raw_files(args)
    if args.dry_run:
        fixed_count = sum(1 for row in time_rows if row["fixed_by_time_sort"])
        print(f"DRY_RUN_COMPLETE fixed_trajectories={fixed_count}")
        for row in time_rows:
            if row["fixed_by_time_sort"]:
                print(
                    f"{row['split']} {row['file']} "
                    f"inversions {row['raw_time_inversions_before']}->{row['raw_time_inversions_after']} "
                    f"gt50 {row['raw_gt50_step_count_before']}->{row['raw_gt50_step_count_after']}",
                    flush=True,
                )
        return
    print("STEP 2/8 rebuilding 43-dim files", flush=True)
    rebuild_wheat43(args)
    print("STEP 3/8 rebuilding adj files", flush=True)
    adj_rows = rebuild_adj(args)
    print("STEP 4/8 writing split json copies", flush=True)
    write_split_jsons(args)
    print("STEP 5/8 rebuilding torch cache", flush=True)
    cache_rows = rebuild_cache(args)
    print("STEP 6/8 rebuilding or auditing PT2G graph cache", flush=True)
    graph_rows = rebuild_graph_cache(args)
    print("STEP 7/8 reaudit fixed data", flush=True)
    reaudit_rows = reaudit(args, time_rows, adj_rows)
    print("STEP 8/8 writing report and pack", flush=True)
    report_path = make_report(args, time_rows, adj_rows, cache_rows, graph_rows, reaudit_rows)
    pack_path = make_pack(args)

    fixed_count = sum(1 for row in time_rows if row["fixed_by_time_sort"])
    print(f"fixed_trajectories={fixed_count}")
    print(f"raw_fixed_dir={args.raw_fixed_dir}")
    print(f"wheat43_fixed_dir={args.wheat43_fixed_dir}")
    print(f"adj_fixed_dir={args.adj_fixed_dir}")
    print(f"cache_fixed_dir={args.cache_fixed_dir}")
    print(f"graph_cache_fixed_dir={args.graph_cache_fixed_dir}")
    print(f"report_path={report_path}")
    print(f"pack_path={pack_path}")


if __name__ == "__main__":
    main()
