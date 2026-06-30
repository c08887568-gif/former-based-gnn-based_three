from __future__ import annotations

from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd


RAW_COLUMNS = ["时间", "经度", "纬度", "速度", "方向", "高度", "标签"]
FEATURE_COLUMNS = list(range(44))


def read_raw_wheat(path: str | Path) -> pd.DataFrame:
    df = pd.read_excel(path)
    missing = [column for column in RAW_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"RAW_WHEAT_COLUMNS_MISSING: {path} missing={missing}")
    return df[RAW_COLUMNS].copy()


def read_wheat_43(path: str | Path) -> pd.DataFrame:
    df = pd.read_excel(path)
    if df.shape[1] < 44:
        raise ValueError(f"WHEAT_43_COLUMNS_MISSING: {path}")
    df = df.iloc[:, :44].copy()
    df.columns = FEATURE_COLUMNS
    return df


def _coord_key(lon, lat, label, digits=8):
    return (round(float(lon), digits), round(float(lat), digits), int(label))


def align_raw_to_wheat43(raw_df: pd.DataFrame, wheat43_df: pd.DataFrame) -> tuple[pd.DataFrame, list[int], int]:
    queues = {}
    for idx, row in raw_df.iterrows():
        key = _coord_key(row["经度"], row["纬度"], row["标签"])
        queues.setdefault(key, deque()).append(int(idx))

    aligned_indices = []
    fallback_count = 0
    last_idx = -1
    for pos, row in wheat43_df.iterrows():
        key = _coord_key(row[41], row[42], row[43])
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
            stop = min(len(raw_df), start + 256)
            window = raw_df.iloc[start:stop]
            if len(window) > 0:
                lon = window["经度"].to_numpy(dtype=float)
                lat = window["纬度"].to_numpy(dtype=float)
                label = window["标签"].to_numpy(dtype=int)
                score = np.abs(lon - float(row[41])) + np.abs(lat - float(row[42])) + (label != int(row[43])) * 1e3
                match_idx = int(window.index[int(np.argmin(score))])
            else:
                match_idx = min(pos, len(raw_df) - 1)
        aligned_indices.append(match_idx)
        last_idx = match_idx

    aligned = raw_df.iloc[aligned_indices].reset_index(drop=True)
    return aligned, aligned_indices, fallback_count


def circular_mean_degrees(values) -> float:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return 0.0
    radians = np.deg2rad(arr)
    angle = np.rad2deg(np.arctan2(np.sin(radians).mean(), np.cos(radians).mean()))
    return float(angle % 360)


ROLLING_WINDOW = 10


def _rolling_skew(series: pd.Series) -> pd.Series:
    return series.rolling(window=ROLLING_WINDOW, min_periods=1).skew().fillna(0.0)


def _window_kurt(values) -> float:
    values = pd.Series(values)
    count = len(values)
    if count <= 1:
        return 0.0
    if count == 2:
        return -2.0
    if count == 3:
        return -1.5
    return float(values.kurt())


def _rolling_kurt(series: pd.Series) -> pd.Series:
    return series.rolling(window=ROLLING_WINDOW, min_periods=1).apply(_window_kurt, raw=False).fillna(0.0)


def haversine_m(lon1, lat1, lon2, lat2):
    radius = 6371000.0
    lon1 = np.deg2rad(lon1)
    lat1 = np.deg2rad(lat1)
    lon2 = np.deg2rad(lon2)
    lat2 = np.deg2rad(lat2)
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return 2.0 * radius * np.arctan2(np.sqrt(a), np.sqrt(np.maximum(1.0 - a, 0.0)))


def feature_col5_geometry(lon, lat, horizon=10) -> np.ndarray:
    lon = np.asarray(lon, dtype=float)
    lat = np.asarray(lat, dtype=float)
    n = len(lon)
    out = np.zeros(n, dtype=float)
    for idx in range(n):
        if idx + horizon < n:
            neighbors = range(idx + 1, idx + horizon + 1)
        else:
            neighbors = range(max(0, idx - horizon), idx)
        total = 0.0
        for other in neighbors:
            total += float(haversine_m(lon[idx], lat[idx], lon[other], lat[other]))
        out[idx] = total
    return out


def rebuild_wheat_43_features(raw_df: pd.DataFrame) -> pd.DataFrame:
    df = raw_df.reset_index(drop=True).copy()
    n = len(df)
    speed = df["速度"].astype(float)
    direction = df["方向"].astype(float)
    label = df["标签"].astype(int)

    speed_diff = speed.diff().fillna(0.0)
    direction_diff = direction.diff().fillna(0.0)
    direction_current = direction.where(speed > 0, 0.0)
    if n:
        direction_current.iloc[0] = 0.0
    direction_alt = direction_current.diff().fillna(0.0)

    base = pd.DataFrame(
        dict(
            speed=speed,
            speed_diff=speed_diff,
            direction_diff=direction_diff,
            direction_current=direction_current,
            direction_alt=direction_alt,
        )
    )
    out = pd.DataFrame(index=df.index, columns=FEATURE_COLUMNS, dtype=float)
    out[0] = base["speed"]
    out[1] = base["speed_diff"]
    out[2] = base["direction_diff"]
    out[3] = base["direction_current"]
    out[4] = base["direction_alt"]
    out[5] = feature_col5_geometry(df["经度"].to_numpy(dtype=float), df["纬度"].to_numpy(dtype=float))

    std_order = ["speed_diff", "direction_alt", "direction_diff", "direction_current", "speed"]
    median_order = ["speed_diff", "direction_diff", "direction_alt", "direction_current", "speed"]
    max_order = ["direction_alt", "speed", "direction_diff", "direction_current", "speed_diff"]
    min_order = ["direction_alt", "speed", "direction_diff", "direction_current", "speed_diff"]
    skew_order = ["speed", "speed_diff", "direction_diff", "direction_current", "direction_alt"]
    kurt_order = ["direction_alt", "speed", "direction_diff", "direction_current", "speed_diff"]
    mean_order = ["speed_diff", "direction_alt", "direction_diff", "direction_current", "speed"]

    for offset, name in enumerate(std_order, start=6):
        out[offset] = base[name].rolling(window=ROLLING_WINDOW, min_periods=1).std(ddof=0).fillna(0.0)
    for offset, name in enumerate(median_order, start=11):
        out[offset] = base[name].rolling(window=ROLLING_WINDOW, min_periods=1).median()
    for offset, name in enumerate(max_order, start=16):
        out[offset] = base[name].rolling(window=ROLLING_WINDOW, min_periods=1).max()
    for offset, name in enumerate(min_order, start=21):
        out[offset] = base[name].rolling(window=ROLLING_WINDOW, min_periods=1).min()
    for offset, name in enumerate(skew_order, start=26):
        out[offset] = _rolling_skew(base[name])
    for offset, name in enumerate(kurt_order, start=31):
        out[offset] = _rolling_kurt(base[name])
    for offset, name in enumerate(mean_order, start=36):
        out[offset] = base[name].rolling(window=ROLLING_WINDOW, min_periods=1).mean()

    out[41] = df["经度"].astype(float)
    out[42] = df["纬度"].astype(float)
    out[43] = label
    return out


def majority_label(labels) -> int:
    values = np.asarray(labels, dtype=int)
    if values.size == 0:
        return 0
    counts = np.bincount(values, minlength=2)
    return int(np.argmax(counts))


def aggregate_raw_group(group: pd.DataFrame) -> dict:
    timestamps = pd.to_datetime(group["时间"], errors="coerce", format="mixed")
    if timestamps.notna().any():
        time_value = timestamps.sort_values().iloc[len(timestamps) // 2]
        time_text = time_value.strftime("%Y/%m/%d %H:%M:%S")
    else:
        time_text = str(group["时间"].iloc[len(group) // 2])
    return dict(
        时间=time_text,
        经度=float(group["经度"].median()),
        纬度=float(group["纬度"].median()),
        速度=float(group["速度"].median()),
        方向=circular_mean_degrees(group["方向"]),
        高度=float(group["高度"].median()),
        标签=majority_label(group["标签"]),
    )
