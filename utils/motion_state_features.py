import math

import numpy as np
import torch


AUX_FEATURE_NAMES = [
    "prev_step_distance_m",
    "next_step_distance_m",
    "local_step_mean_m",
    "local_step_std_m",
    "local_turn_angle_deg",
    "local_density_1m",
    "local_density_2m",
    "stationary_flag",
    "stationary_run_length",
    "trace_position_ratio",
    "near_endpoint_flag",
]


def haversine_m(lon1, lat1, lon2, lat2):
    radius = 6371000.0
    phi1 = math.radians(float(lat1))
    phi2 = math.radians(float(lat2))
    dphi = math.radians(float(lat2) - float(lat1))
    dlambda = math.radians(float(lon2) - float(lon1))
    a = math.sin(dphi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
    return 2.0 * radius * math.atan2(math.sqrt(a), math.sqrt(max(1.0 - a, 0.0)))


def _lonlat_to_xy_m(lon, lat, ref_lat):
    x = math.radians(float(lon)) * 6371000.0 * math.cos(math.radians(float(ref_lat)))
    y = math.radians(float(lat)) * 6371000.0
    return x, y


def turn_angle_deg(prev_coord, coord, next_coord, min_step_m=0.5):
    lon0, lat0 = prev_coord
    lon1, lat1 = coord
    lon2, lat2 = next_coord
    ref_lat = (float(lat0) + float(lat1) + float(lat2)) / 3.0
    x0, y0 = _lonlat_to_xy_m(lon0, lat0, ref_lat)
    x1, y1 = _lonlat_to_xy_m(lon1, lat1, ref_lat)
    x2, y2 = _lonlat_to_xy_m(lon2, lat2, ref_lat)
    v1 = np.array([x1 - x0, y1 - y0], dtype=np.float64)
    v2 = np.array([x2 - x1, y2 - y1], dtype=np.float64)
    n1 = float(np.linalg.norm(v1))
    n2 = float(np.linalg.norm(v2))
    if n1 < min_step_m or n2 < min_step_m:
        return 0.0
    cosine = float(np.dot(v1, v2) / (n1 * n2))
    cosine = max(-1.0, min(1.0, cosine))
    return math.degrees(math.acos(cosine))


def _to_numpy_coordinates(coordinates):
    if coordinates is None:
        raise ValueError("AUX_COORDINATES_NOT_FOUND")
    if isinstance(coordinates, torch.Tensor):
        array = coordinates.detach().cpu().numpy()
    else:
        array = np.asarray(coordinates)
    if array.ndim == 3 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 2 or array.shape[1] < 2:
        raise ValueError("AUX_COORDINATES_NOT_FOUND")
    return array[:, :2].astype(np.float64, copy=False)


def _mean(values):
    return float(np.mean(values)) if values else 0.0


def _std(values):
    return float(np.std(values, ddof=0)) if len(values) > 1 else 0.0


def build_motion_state_features(coordinates, points=None):
    coords = _to_numpy_coordinates(coordinates)
    n = int(coords.shape[0])
    if points is not None:
        point_count = int(points.shape[0])
        if point_count != n:
            raise ValueError(f"AUX_COORDINATES_LENGTH_MISMATCH: coordinates={n} points={point_count}")
    if n == 0:
        return torch.zeros((0, len(AUX_FEATURE_NAMES)), dtype=torch.float32)

    prev_steps = [0.0] * n
    next_steps = [0.0] * n
    for idx in range(n):
        if idx > 0:
            prev_steps[idx] = haversine_m(*coords[idx - 1], *coords[idx])
        if idx < n - 1:
            next_steps[idx] = haversine_m(*coords[idx], *coords[idx + 1])

    rows = []
    stationary_flags = []
    for idx in range(n):
        lo = max(1, idx - 2)
        hi = min(n - 1, idx + 2)
        local_steps = [haversine_m(*coords[step_idx - 1], *coords[step_idx]) for step_idx in range(lo, hi + 1)]
        local_step_mean = _mean(local_steps)
        local_step_std = _std(local_steps)

        turn = 0.0
        if 0 < idx < n - 1:
            turn = turn_angle_deg(coords[idx - 1], coords[idx], coords[idx + 1])

        density_1m = 0
        density_2m = 0
        for other_idx in range(max(0, idx - 5), min(n, idx + 6)):
            if other_idx == idx:
                continue
            distance = haversine_m(*coords[idx], *coords[other_idx])
            if distance <= 1.0:
                density_1m += 1
            if distance <= 2.0:
                density_2m += 1

        stationary = (prev_steps[idx] <= 0.5 and next_steps[idx] <= 0.5) or local_step_mean <= 0.5
        stationary_flags.append(bool(stationary))
        trace_position_ratio = idx / max(n - 1, 1)
        near_endpoint = trace_position_ratio <= 0.05 or trace_position_ratio >= 0.95
        rows.append(
            [
                prev_steps[idx],
                next_steps[idx],
                local_step_mean,
                local_step_std,
                turn,
                float(density_1m),
                float(density_2m),
                1.0 if stationary else 0.0,
                0.0,
                trace_position_ratio,
                1.0 if near_endpoint else 0.0,
            ]
        )

    start = 0
    while start < n:
        end = start
        while end + 1 < n and stationary_flags[end + 1] == stationary_flags[start]:
            end += 1
        run_length = float(end - start + 1) if stationary_flags[start] else 0.0
        for idx in range(start, end + 1):
            rows[idx][8] = run_length
        start = end + 1

    return torch.tensor(rows, dtype=torch.float32)
