import argparse
import csv
import json
import sys
from collections import deque
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.threading_config import apply_torch_thread_config, configure_default_threads

configure_default_threads()

import numpy as np
import pandas as pd
import torch
apply_torch_thread_config(torch)


AUDIT_FIELDS = [
    "file",
    "split",
    "trace_id",
    "original_points",
    "merged_points",
    "reduction_rate",
    "merge_groups",
    "avg_merge_count",
    "max_merge_count",
    "mixed_label_groups",
    "mixed_label_rate",
    "original_edges",
    "mapped_edges_before_dedup",
    "self_loops_removed",
    "duplicate_edges_removed",
    "temporal_edges_added",
    "final_edges",
    "feature6_min",
    "feature6_max",
    "feature6_mean",
    "feature6_recomputed",
    "points_normalized",
    "points_min_after_norm",
    "points_max_after_norm",
    "empty_or_abnormal",
]


RAW_COLUMNS = ["时间", "经度", "纬度", "速度", "方向", "高度", "标签"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build a torch cache after merging consecutive near-duplicate wheat GNSS points."
    )
    parser.add_argument("--raw_dir", default="wheat/sampled_wheat")
    parser.add_argument("--wheat43_dir", default="wheat/sampled_wheat_43")
    parser.add_argument("--adj_dir", default="wheat/sampled_wheat_adj")
    parser.add_argument("--split_dir", default="wheat/Non-Identically_Distributed_Coco")
    parser.add_argument("--json_prefix", default="sampled_wheat_43")
    parser.add_argument("--output_dir", default="cache/wheat_merge_d05_dt10_s1")
    parser.add_argument("--audit_output", default="diagnostics/wheat_merge_d05_audit.csv")
    parser.add_argument("--report_output", default="analysis/wheat_merge_d05_report.md")
    parser.add_argument("--max_len", type=int, default=1000)
    parser.add_argument("--drop_rate", type=float, default=0.0)
    parser.add_argument("--distance_threshold", type=float, default=0.5)
    parser.add_argument("--time_threshold", type=float, default=10.0)
    parser.add_argument("--speed_threshold", type=float, default=1.0)
    parser.add_argument("--diameter_threshold", type=float, default=1.0)
    parser.add_argument("--max_group_size", type=int, default=6)
    parser.add_argument("--max_group_seconds", type=float, default=30.0)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--dry_run_traces", type=int, default=3)
    return parser.parse_args()


def read_raw_wheat(path):
    df = pd.read_excel(path)
    missing = [column for column in RAW_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"RAW_COLUMNS_MISSING: {path} missing={missing}")
    return df[RAW_COLUMNS].copy()


def read_wheat43(path):
    df = pd.read_excel(path)
    if df.shape[1] < 44:
        raise ValueError(f"WHEAT43_COLUMNS_MISSING: {path}")
    df = df.iloc[:, :44].copy()
    df.columns = list(range(44))
    return df


def coord_key(lon, lat, label, digits=8):
    return (round(float(lon), digits), round(float(lat), digits), int(label))


def align_raw_to_wheat43(raw_df, wheat43_df):
    queues = {}
    for idx, row in raw_df.iterrows():
        key = coord_key(row["经度"], row["纬度"], row["标签"])
        queues.setdefault(key, deque()).append(int(idx))

    aligned_raw_indices = []
    last_idx = -1
    fallback_count = 0
    for _, row in wheat43_df.iterrows():
        key = coord_key(row[41], row[42], row[43])
        queue = queues.get(key)
        match_idx = None
        if queue is not None:
            while queue and queue[0] <= last_idx:
                queue.popleft()
            if queue:
                match_idx = queue.popleft()
        if match_idx is None:
            fallback_count += 1
            start = max(last_idx + 1, 0)
            stop = min(len(raw_df), start + 512)
            window = raw_df.iloc[start:stop]
            if len(window) == 0:
                match_idx = min(start, len(raw_df) - 1)
            else:
                lon = window["经度"].to_numpy(dtype=float)
                lat = window["纬度"].to_numpy(dtype=float)
                labels = window["标签"].to_numpy(dtype=int)
                score = (
                    np.abs(lon - float(row[41]))
                    + np.abs(lat - float(row[42]))
                    + (labels != int(row[43])) * 1000.0
                )
                match_idx = int(window.index[int(np.argmin(score))])
        aligned_raw_indices.append(match_idx)
        last_idx = match_idx

    aligned_raw = raw_df.iloc[aligned_raw_indices].reset_index(drop=True)
    return aligned_raw, aligned_raw_indices, fallback_count


def haversine_m(lon1, lat1, lon2, lat2):
    radius = 6371000.0
    lon1 = np.deg2rad(lon1)
    lat1 = np.deg2rad(lat1)
    lon2 = np.deg2rad(lon2)
    lat2 = np.deg2rad(lat2)
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return float(2 * radius * np.arcsin(np.sqrt(a)))


def parse_times(raw_df):
    return pd.to_datetime(raw_df["时间"], errors="coerce", format="mixed")


def group_diameter_m(raw_df):
    if len(raw_df) <= 1:
        return 0.0
    lon = raw_df["经度"].to_numpy(dtype=float)
    lat = raw_df["纬度"].to_numpy(dtype=float)
    max_dist = 0.0
    for i in range(len(raw_df)):
        for j in range(i + 1, len(raw_df)):
            max_dist = max(max_dist, haversine_m(lon[i], lat[i], lon[j], lat[j]))
    return max_dist


def pair_can_merge(prev_row, next_row, prev_time, next_time, args):
    distance = haversine_m(prev_row["经度"], prev_row["纬度"], next_row["经度"], next_row["纬度"])
    if distance > args.distance_threshold:
        return False
    if pd.isna(prev_time) or pd.isna(next_time):
        dt = 0.0
    else:
        dt = abs((next_time - prev_time).total_seconds())
    if dt > args.time_threshold:
        return False
    derived_speed = distance / max(dt, 1e-6) if dt > 0 else 0.0
    both_stop = float(prev_row["速度"]) <= args.speed_threshold and float(next_row["速度"]) <= args.speed_threshold
    return derived_speed <= args.speed_threshold or both_stop


def build_merge_groups(aligned_raw, args):
    if len(aligned_raw) == 0:
        return []
    times = parse_times(aligned_raw)
    groups = []
    current = [0]
    for idx in range(1, len(aligned_raw)):
        candidate = current + [idx]
        candidate_raw = aligned_raw.iloc[candidate]
        candidate_times = times.iloc[candidate]
        if candidate_times.notna().any():
            span_seconds = float((candidate_times.max() - candidate_times.min()).total_seconds())
        else:
            span_seconds = 0.0
        can_merge = pair_can_merge(
            aligned_raw.iloc[current[-1]],
            aligned_raw.iloc[idx],
            times.iloc[current[-1]],
            times.iloc[idx],
            args,
        )
        can_merge = can_merge and len(candidate) <= args.max_group_size
        can_merge = can_merge and span_seconds <= args.max_group_seconds
        can_merge = can_merge and group_diameter_m(candidate_raw) <= args.diameter_threshold
        if can_merge:
            current = candidate
        else:
            groups.append(current)
            current = [idx]
    groups.append(current)
    return groups


def majority_label(labels):
    labels = np.asarray(labels, dtype=int)
    if labels.size == 0:
        return 0
    counts = np.bincount(labels, minlength=2)
    return int(np.argmax(counts))


def recompute_feature6(features):
    lon = features[:, 41].astype(float)
    lat = features[:, 42].astype(float)
    values = np.zeros(len(features), dtype=np.float32)
    for idx in range(len(features)):
        future = list(range(idx + 1, min(len(features), idx + 11)))
        if len(future) == 10:
            neighbors = future
        else:
            neighbors = list(range(max(0, idx - 10), idx))
        total = 0.0
        for other in neighbors:
            total += haversine_m(lon[idx], lat[idx], lon[other], lat[other])
        values[idx] = total
    return values


def minmax_normalize_features(features):
    if len(features) == 0:
        return features.astype(np.float32), dict(
            points_min_after_norm=0.0,
            points_max_after_norm=0.0,
            points_normalized=True,
        )
    features = features.astype(np.float32, copy=True)
    feature_min = features.min(axis=0)
    feature_max = features.max(axis=0)
    denom = feature_max - feature_min
    constant_mask = denom == 0
    denom[constant_mask] = 1.0
    normalized = (features - feature_min) / denom
    normalized[:, constant_mask] = 0.0
    normalized = np.clip(normalized, 0.0, 1.0).astype(np.float32)
    return normalized, dict(
        points_min_after_norm=float(normalized.min()),
        points_max_after_norm=float(normalized.max()),
        points_normalized=True,
    )


def build_merged_features(wheat43_df, groups):
    old_features = wheat43_df.iloc[:, :43].to_numpy(dtype=np.float32)
    old_labels = wheat43_df.iloc[:, 43].to_numpy(dtype=np.uint8)
    merged_features = []
    merged_labels = []
    mixed_label_groups = 0
    merge_counts = []
    for group in groups:
        group_features = old_features[group]
        group_labels = old_labels[group]
        merged = group_features.mean(axis=0)
        merged_features.append(merged)
        label = majority_label(group_labels)
        merged_labels.append(label)
        merge_counts.append(len(group))
        if len(set(group_labels.astype(int).tolist())) > 1:
            mixed_label_groups += 1

    if merged_features:
        merged_features = np.vstack(merged_features).astype(np.float32)
        merged_features[:, 5] = recompute_feature6(merged_features)
        merged_labels = np.asarray(merged_labels, dtype=np.uint8)
    else:
        merged_features = np.empty((0, 43), dtype=np.float32)
        merged_labels = np.empty((0,), dtype=np.uint8)
    return merged_features, merged_labels, mixed_label_groups, merge_counts


def dedup_edges(edges):
    seen = set()
    deduped = []
    duplicate_count = 0
    for src, dst in edges:
        edge = (int(src), int(dst))
        if edge in seen:
            duplicate_count += 1
            continue
        seen.add(edge)
        deduped.append(edge)
    return deduped, duplicate_count


def map_edges(original_edges, old_to_new, merged_count):
    mapped = []
    self_loops_removed = 0
    for src, dst in original_edges:
        src = int(src)
        dst = int(dst)
        if src not in old_to_new or dst not in old_to_new:
            continue
        new_src = old_to_new[src]
        new_dst = old_to_new[dst]
        if new_src == new_dst:
            self_loops_removed += 1
            continue
        mapped.append((new_src, new_dst))

    mapped_edges_before_dedup = len(mapped)
    deduped, duplicate_edges_removed = dedup_edges(mapped)
    seen = set(deduped)
    temporal_edges_added = 0
    for idx in range(max(0, merged_count - 1)):
        for edge in ((idx, idx + 1), (idx + 1, idx)):
            if edge not in seen:
                seen.add(edge)
                deduped.append(edge)
                temporal_edges_added += 1

    if deduped:
        edge_index = torch.tensor(deduped, dtype=torch.long).t().contiguous()
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)

    return edge_index, dict(
        mapped_edges_before_dedup=mapped_edges_before_dedup,
        self_loops_removed=self_loops_removed,
        duplicate_edges_removed=duplicate_edges_removed,
        temporal_edges_added=temporal_edges_added,
        final_edges=int(edge_index.shape[1]),
    )


def crop_items(features, labels, edge_index, trace_id, coordinates, merge_mapping, original_indices, max_len, drop_rate):
    items = []
    total = len(features)
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
                merge_mapping=[list(group) for group in merge_mapping[start:end]],
                original_indices=[list(group) for group in original_indices[start:end]],
            )
        )
    return items


def load_split_files(split_dir, json_prefix, split):
    path = Path(split_dir) / f"{json_prefix}_{split}.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    return [item["file_name"] for item in data["trajectories"]]


def process_file(file_name, split, args):
    raw_path = Path(args.raw_dir) / file_name
    wheat43_path = Path(args.wheat43_dir) / file_name
    adj_path = Path(args.adj_dir) / file_name.replace(".xlsx", ".npy")
    trace_id = str(wheat43_path)

    raw_df = read_raw_wheat(raw_path)
    wheat43_df = read_wheat43(wheat43_path)
    aligned_raw, aligned_raw_indices, fallback_count = align_raw_to_wheat43(raw_df, wheat43_df)
    groups = build_merge_groups(aligned_raw, args)
    raw_features, labels, mixed_label_groups, merge_counts = build_merged_features(wheat43_df, groups)
    coordinates = raw_features[:, [41, 42]] if len(raw_features) else np.empty((0, 2), dtype=np.float32)
    features, norm_stats = minmax_normalize_features(raw_features)

    old_to_new = {}
    for new_idx, group in enumerate(groups):
        for old_wheat43_idx in group:
            old_to_new[int(old_wheat43_idx)] = int(new_idx)

    original_edges = np.load(adj_path)
    original_edges = original_edges[
        (original_edges[:, 0] < len(wheat43_df)) & (original_edges[:, 1] < len(wheat43_df))
    ]
    edge_index, edge_stats = map_edges(original_edges, old_to_new, len(features))
    original_indices = [[int(aligned_raw_indices[idx]) for idx in group] for group in groups]
    merge_mapping = [[int(idx) for idx in group] for group in groups]

    items = crop_items(
        features,
        labels,
        edge_index,
        trace_id,
        coordinates,
        merge_mapping,
        original_indices,
        args.max_len,
        args.drop_rate,
    )

    original_points = int(len(wheat43_df))
    merged_points = int(len(features))
    reduction_rate = float((original_points - merged_points) / original_points) if original_points else 0.0
    avg_merge_count = float(np.mean(merge_counts)) if merge_counts else 0.0
    max_merge_count = int(max(merge_counts)) if merge_counts else 0
    merge_groups = int(sum(1 for count in merge_counts if count > 1))
    mixed_label_rate = float(mixed_label_groups / max(merge_groups, 1))
    feature6 = raw_features[:, 5] if len(raw_features) else np.asarray([], dtype=np.float32)
    empty_or_abnormal = bool(original_points == 0 or merged_points == 0 or not items or fallback_count > 0)
    audit_row = dict(
        file=file_name,
        split=split,
        trace_id=trace_id,
        original_points=original_points,
        merged_points=merged_points,
        reduction_rate=reduction_rate,
        merge_groups=merge_groups,
        avg_merge_count=avg_merge_count,
        max_merge_count=max_merge_count,
        mixed_label_groups=int(mixed_label_groups),
        mixed_label_rate=mixed_label_rate,
        original_edges=int(len(original_edges)),
        feature6_min=float(feature6.min()) if len(feature6) else 0.0,
        feature6_max=float(feature6.max()) if len(feature6) else 0.0,
        feature6_mean=float(feature6.mean()) if len(feature6) else 0.0,
        feature6_recomputed=True,
        **norm_stats,
        empty_or_abnormal=empty_or_abnormal,
        **edge_stats,
    )
    return items, audit_row


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=AUDIT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def write_report(path, rows, split_counts):
    path.parent.mkdir(parents=True, exist_ok=True)
    total_original = sum(int(row["original_points"]) for row in rows)
    total_merged = sum(int(row["merged_points"]) for row in rows)
    total_reduction = (total_original - total_merged) / total_original if total_original else 0.0
    split_lines = []
    for split in ("train", "valid", "test"):
        split_rows = [row for row in rows if row["split"] == split]
        split_original = sum(int(row["original_points"]) for row in split_rows)
        split_merged = sum(int(row["merged_points"]) for row in split_rows)
        split_reduction = (split_original - split_merged) / split_original if split_original else 0.0
        split_lines.append(
            f"- {split}: original={split_original}, merged={split_merged}, "
            f"reduction_rate={split_reduction:.6f}, cache_items={split_counts.get(split, 0)}"
        )
    mixed_groups = sum(int(row["mixed_label_groups"]) for row in rows)
    merge_groups = sum(int(row["merge_groups"]) for row in rows)
    mixed_rate = mixed_groups / max(merge_groups, 1)
    original_edges = sum(int(row["original_edges"]) for row in rows)
    final_edges = sum(int(row["final_edges"]) for row in rows)
    temporal_edges = sum(int(row["temporal_edges_added"]) for row in rows)
    self_loops = sum(int(row["self_loops_removed"]) for row in rows)
    duplicate_edges = sum(int(row["duplicate_edges_removed"]) for row in rows)
    abnormal = [row["trace_id"] for row in rows if row["empty_or_abnormal"]]
    feature6_ok = all(str(row["feature6_recomputed"]).lower() == "true" for row in rows)

    text = f"""# wheat merge duplicate cache report

本次只构建连续近邻重复点合并后的 torch cache，不训练模型，不覆盖 `wheat/sampled_wheat`，不覆盖 `wheat/sampled_wheat_43`，不生成新的 Excel 中转目录。

## 输出

- cache: `cache/wheat_merge_d05_dt10_s1/`
- train/valid/test cache item 数: `{split_counts}`
- audit: `diagnostics/wheat_merge_d05_audit.csv`

## 总体点数变化

- 总原始点数: `{total_original}`
- 总合并后点数: `{total_merged}`
- 总减少比例: `{total_reduction:.6f}`

## split 点数变化

{chr(10).join(split_lines)}

## mixed-label merge groups

- mixed-label merge groups: `{mixed_groups}`
- merge groups: `{merge_groups}`
- mixed-label rate: `{mixed_rate:.6f}`
- 判断: `{'偏高，需要重点检查' if mixed_rate > 0.05 else '不高，仅作为诊断记录'}`

## edge 变化

- original_edges: `{original_edges}`
- final_edges: `{final_edges}`
- self_loops_removed: `{self_loops}`
- duplicate_edges_removed: `{duplicate_edges}`
- temporal_edges_added: `{temporal_edges}`
- 判断: `{'edge 数存在明显下降，主要来自合并后自环删除和去重；已补充时间连续双向边' if final_edges < original_edges else 'edge 数未异常下降'}`

## 第 6 维

- 第 6 维，即代码列索引 `5`，已按合并后的新经纬度序列重新计算: `{feature6_ok}`。
- 公式: `feature_6(i) = sum(distance(point_i, point_(i+k)))`, `k=1..10`；如果未来不足 10 个点，则改用前面最多 10 个点。
- 距离使用 Haversine 球面距离，单位米。
- 其它 42 维来自对应 `sampled_wheat_43` 原始行的组内均值。
- 生成模型输入 `points` 前，已按原始读取管道的方式对每条合并后轨迹的 43 个输入列执行 MinMax 归一化到 `[0, 1]`。
- `coordinates` 仍保存未归一化的原始经纬度，供 HTML 和诊断使用。

## 空轨迹或异常样本

- 异常轨迹数量: `{len(abnormal)}`
- 异常轨迹: `{abnormal[:20]}`
"""
    path.write_text(text, encoding="utf-8")


def dry_run(args):
    rows = []
    processed = 0
    for split in ("train", "valid", "test"):
        for file_name in load_split_files(args.split_dir, args.json_prefix, split):
            _items, row = process_file(file_name, split, args)
            rows.append(row)
            processed += 1
            print(
                f"DRY_RUN split={split} file={file_name} "
                f"original_points={row['original_points']} "
                f"merged_points={row['merged_points']} "
                f"reduction_rate={row['reduction_rate']:.6f} "
                f"original_edges={row['original_edges']} final_edges={row['final_edges']} "
                f"mixed_label_groups={row['mixed_label_groups']}",
                flush=True,
            )
            raw_path = Path(args.raw_dir) / file_name
            wheat43_path = Path(args.wheat43_dir) / file_name
            raw_df = read_raw_wheat(raw_path)
            wheat43_df = read_wheat43(wheat43_path)
            aligned_raw, _aligned_raw_indices, _fallback_count = align_raw_to_wheat43(raw_df, wheat43_df)
            groups = build_merge_groups(aligned_raw, args)
            raw_features, _labels, _mixed, _counts = build_merged_features(wheat43_df, groups)
            normalized_features, _norm_stats = minmax_normalize_features(raw_features)
            print("feature6_raw_first10=", [round(float(v), 6) for v in raw_features[:10, 5]], flush=True)
            print("feature6_norm_first10=", [round(float(v), 6) for v in normalized_features[:10, 5]], flush=True)
            if processed >= args.dry_run_traces:
                return rows
    return rows


def main():
    args = parse_args()
    if args.dry_run:
        dry_run(args)
        return

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    all_rows = []
    split_counts = {}
    for split in ("train", "valid", "test"):
        split_items = []
        file_names = load_split_files(args.split_dir, args.json_prefix, split)
        for idx, file_name in enumerate(file_names, start=1):
            items, row = process_file(file_name, split, args)
            split_items.extend(items)
            all_rows.append(row)
            print(
                f"{split} {idx}/{len(file_names)} {file_name} "
                f"original={row['original_points']} merged={row['merged_points']} "
                f"final_edges={row['final_edges']}",
                flush=True,
            )
        split_counts[split] = len(split_items)
        torch.save(split_items, output_dir / f"{split}.pt")
        print(f"WROTE {output_dir / f'{split}.pt'} samples={len(split_items)}", flush=True)

    write_csv(Path(args.audit_output), all_rows)
    write_report(Path(args.report_output), all_rows, split_counts)
    print(f"WROTE {args.audit_output}")
    print(f"WROTE {args.report_output}")


if __name__ == "__main__":
    main()
