from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from .config import DetectionConfig

M_TO_FT = 3.28084
ALGO_VERSION = "2026-08-07-shaft-crossarm-v18"
MAX_DBSCAN_POINTS_BULK = 150000
MAX_DBSCAN_POINTS_CLIPPED = 45000


def _estimate_ground(points_xyz: np.ndarray, grid_size: float) -> np.ndarray:
    xy = points_xyz[:, :2]
    z = points_xyz[:, 2]

    grid_xy = np.floor(xy / grid_size).astype(np.int64)
    keys = grid_xy[:, 0].astype(np.int64) * 73856093 ^ grid_xy[:, 1].astype(np.int64) * 19349663

    order = np.argsort(keys)
    keys_sorted = keys[order]
    z_sorted = z[order]

    unique_keys, starts = np.unique(keys_sorted, return_index=True)
    starts = np.append(starts, len(keys_sorted))

    ground_lookup: dict[int, float] = {}
    for i in range(len(unique_keys)):
        s = starts[i]
        e = starts[i + 1]
        ground_lookup[int(unique_keys[i])] = float(np.quantile(z_sorted[s:e], 0.05))

    ground = np.array([ground_lookup[int(k)] for k in keys], dtype=np.float64)
    return ground


def _principal_axis_span(points_xy: np.ndarray) -> float:
    if len(points_xy) < 3:
        return 0.0
    centered = points_xy - np.mean(points_xy, axis=0, keepdims=True)
    cov = centered.T @ centered / max(len(centered) - 1, 1)
    eigvals, eigvecs = np.linalg.eigh(cov)
    major = eigvecs[:, np.argmax(eigvals)]
    proj = centered @ major
    # Use a wider percentile span to better match manual endpoint picking.
    return float(np.percentile(proj, 98) - np.percentile(proj, 2))


def _principal_axis_linearity(points_xy: np.ndarray) -> float:
    if len(points_xy) < 4:
        return 0.0
    centered = points_xy - np.mean(points_xy, axis=0, keepdims=True)
    cov = centered.T @ centered / max(len(centered) - 1, 1)
    eigvals, _ = np.linalg.eigh(cov)
    e0 = float(max(eigvals[0], 1e-9))
    e1 = float(max(eigvals[1], 1e-9))
    return e1 / e0


def _top_from_continuous_bins(shaft_z: np.ndarray, bin_m: float = 0.2) -> float:
    zmin = float(np.min(shaft_z))
    zmax = float(np.max(shaft_z))
    if zmax - zmin < 0.8:
        return zmax
    bins = np.arange(zmin, zmax + bin_m, bin_m)
    hist, edges = np.histogram(shaft_z, bins=bins)
    if len(hist) == 0:
        return zmax
    min_occ = max(2, int(np.percentile(hist, 25)))
    occ = hist >= min_occ
    idx = np.where(occ)[0]
    if len(idx) == 0:
        return zmax
    top_i = int(idx[-1])
    return float(edges[top_i + 1])


def _find_best_shaft_center(points_xyz: np.ndarray, norm_z: np.ndarray, cfg: DetectionConfig) -> np.ndarray | None:
    tall = points_xyz[norm_z >= max(cfg.min_height_m, 2.5)]
    if len(tall) < 40:
        return None

    cell_m = 0.25
    grid = np.floor(tall[:, :2] / cell_m).astype(np.int32)
    keys = grid[:, 0].astype(np.int64) * 104729 + grid[:, 1].astype(np.int64) * 130363
    uniq = np.unique(keys)
    best_score = -1.0
    best_center = None
    shaft_r = max(0.28, cfg.shaft_radius_m * 0.85)
    scene_center = np.median(points_xyz[:, :2], axis=0)
    for key in uniq:
        m = keys == key
        seed = np.median(tall[m, :2], axis=0)
        dxy = points_xyz[:, :2] - seed
        radial = np.sqrt(np.sum(dxy * dxy, axis=1))
        shaft = points_xyz[radial <= shaft_r]
        if len(shaft) < 35:
            continue
        zspan = float(np.percentile(shaft[:, 2], 99) - np.percentile(shaft[:, 2], 1))
        if zspan < 5.0:
            continue
        zbin = 0.25
        bins = np.arange(float(np.min(shaft[:, 2])), float(np.max(shaft[:, 2])) + zbin, zbin)
        hist, _ = np.histogram(shaft[:, 2], bins=bins)
        occupied = float(np.count_nonzero(hist >= 2))
        # Prefer pole-like vertical columns near scene center and penalize tree-like extreme heights.
        dist_center = float(np.linalg.norm(seed - scene_center))
        center_weight = 1.0 / (1.0 + dist_center * 0.9)
        if zspan > cfg.max_reasonable_pole_height_m * 2.5:
            continue
        height_weight = 1.0
        if zspan > cfg.max_reasonable_pole_height_m:
            height_weight = 0.35
        score = (zspan * np.sqrt(float(len(shaft))) + occupied * 0.6) * center_weight * height_weight
        if score > best_score:
            best_score = score
            best_center = np.median(shaft[:, :2], axis=0)
    return best_center


def _refine_center_to_vertical_shaft(
    points_xyz: np.ndarray,
    seed_center_xy: np.ndarray,
    search_radius_m: float = 2.5,
    cell_m: float = 0.25,
) -> np.ndarray:
    dxy = points_xyz[:, :2] - seed_center_xy
    radial = np.sqrt(np.sum(dxy * dxy, axis=1))
    local = points_xyz[radial <= search_radius_m]
    if len(local) < 30:
        return seed_center_xy

    rel = local[:, :2] - seed_center_xy
    gx = np.floor(rel[:, 0] / cell_m).astype(np.int32)
    gy = np.floor(rel[:, 1] / cell_m).astype(np.int32)
    keys = gx.astype(np.int64) * 104729 + gy.astype(np.int64) * 130363

    best_score = -1.0
    best_center = seed_center_xy
    for key in np.unique(keys):
        m = keys == key
        pts = local[m]
        if len(pts) < 15:
            continue
        z_span = float(np.percentile(pts[:, 2], 99) - np.percentile(pts[:, 2], 1))
        if z_span < 2.5:
            continue
        # Score prefers tall vertical columns with enough support points.
        score = z_span * np.sqrt(float(len(pts)))
        if score > best_score:
            best_score = score
            best_center = np.median(pts[:, :2], axis=0)

    return best_center


def _choose_crossarm_band(
    points_xy: np.ndarray,
    points_z: np.ndarray,
    z_top: float,
    search_drop_m: float,
    band_thickness_m: float,
    min_drop_from_top_m: float,
    max_drop_from_top_m: float,
) -> tuple[float, float] | None:
    z_min = z_top - min(search_drop_m, max_drop_from_top_m)
    z_max = z_top - max(0.05, min_drop_from_top_m)
    valid = (points_z >= z_min) & (points_z <= z_max)
    if not np.any(valid):
        return None

    z_vals = points_z[valid]
    lo = float(np.min(z_vals))
    hi = float(np.max(z_vals))
    if hi - lo < band_thickness_m:
        return (lo, hi)

    best_band: tuple[float, float] | None = None
    best_score = -1.0
    step = max(0.15, band_thickness_m * 0.33)
    start = lo
    while start + band_thickness_m <= hi + 1e-9:
        end = start + band_thickness_m
        m = (points_z >= start) & (points_z <= end)
        if np.count_nonzero(m) >= 6:
            pts = points_xy[m]
            span = _principal_axis_span(pts)
            linearity = _principal_axis_linearity(pts)
            # Prefer bands slightly below top and with strong linear arm-like structure.
            band_mid = (start + end) * 0.5
            drop = max(0.0, z_top - band_mid)
            top_weight = 1.0 / (1.0 + drop)
            score = span * np.sqrt(float(np.count_nonzero(m))) * max(0.2, linearity) * top_weight
            if score > best_score:
                best_score = score
                best_band = (start, end)
        start += step
    return best_band


def _suppress_extreme_top_outliers(shaft_z: np.ndarray, raw_top: float) -> float:
    if len(shaft_z) < 30:
        return raw_top
    # Keep raw top when there is enough upper-shaft support near the tip.
    upper_support = int(np.count_nonzero(shaft_z >= (raw_top - 1.2)))
    if upper_support >= 4:
        return raw_top
    q998 = float(np.percentile(shaft_z, 99.8))
    q99 = float(np.percentile(shaft_z, 99.0))
    # Only clip when there is a suspicious jump likely caused by wire/noise points.
    if raw_top - q99 > 5.0:
        return max(q998, q99 + 0.5)
    return raw_top


def _pole_quality_score(row: dict[str, Any], cfg: DetectionConfig) -> float:
    h = float(row.get("height_m", 0.0))
    c = float(row.get("crossarm_m", 0.0))
    n = float(row.get("point_count", 0.0))

    # Prefer realistic utility-pole heights around configurable target.
    height_score = 1.5 - min(3.0, abs(h - cfg.prefer_pole_height_m) / 4.0)
    if h > cfg.max_reasonable_pole_height_m:
        height_score -= (h - cfg.max_reasonable_pole_height_m) * 0.9
    if h < 8.0:
        height_score -= (8.0 - h) * 1.2

    # Cross-arm length outside typical range is likely wrong context.
    crossarm_score = 0.0
    if h > 0.0 and c > h * 0.65:
        crossarm_score = -2.5
    elif 0.2 <= c <= min(4.5, h * 0.5):
        crossarm_score = 0.7
    elif c > 0.0:
        crossarm_score = -0.9

    density_score = min(1.0, np.sqrt(max(0.0, n)) / 18.0)
    return height_score + crossarm_score + 0.35 * density_score


def _bulk_post_filter(detections: list[dict[str, Any]], cfg: DetectionConfig) -> list[dict[str, Any]]:
    if not detections:
        return []

    # Step-1: quality gating for obvious non-pole fragments.
    gated: list[dict[str, Any]] = []
    for d in detections:
        h = float(d.get("height_m", 0.0))
        n = int(d.get("point_count", 0))
        if h < cfg.bulk_min_height_m or h > cfg.bulk_max_height_m:
            continue
        if n < cfg.bulk_min_point_count:
            continue
        if str(d.get("pole_conf", "Low")).lower() == "low":
            continue
        gated.append(d)
    if not gated:
        return []

    # Step-2: non-max suppression in XY to collapse duplicates/nearby fragments.
    order = sorted(gated, key=lambda d: _pole_quality_score(d, cfg), reverse=True)
    kept: list[dict[str, Any]] = []
    r2 = cfg.bulk_nms_radius_m * cfg.bulk_nms_radius_m
    for d in order:
        x = float(d["x"])
        y = float(d["y"])
        too_close = False
        for k in kept:
            dx = x - float(k["x"])
            dy = y - float(k["y"])
            if dx * dx + dy * dy <= r2:
                too_close = True
                break
        if not too_close:
            kept.append(d)
    kept.sort(key=lambda d: str(d.get("pole_id", "")))
    return kept


def _sanitize_detection(row: dict[str, Any]) -> dict[str, Any] | None:
    h = float(row.get("height_m", 0.0))
    c = float(row.get("crossarm_m", 0.0))
    z_base = float(row.get("z_base", 0.0))
    z_top = float(row.get("z_top", 0.0))
    values = [h, c, z_base, z_top, float(row.get("x", 0.0)), float(row.get("y", 0.0))]
    if not all(np.isfinite(v) for v in values):
        return None
    if h <= 0.0:
        return None
    row["height_m"] = round(max(0.0, h), 3)
    row["crossarm_m"] = round(max(0.0, min(c, h * 0.6)), 3)
    row["height_ft"] = round(row["height_m"] * M_TO_FT, 3)
    row["crossarm_ft"] = round(row["crossarm_m"] * M_TO_FT, 3)
    row["z_base"] = round(z_base, 3)
    row["z_top"] = round(max(z_top, z_base), 3)
    row["point_count"] = int(max(0, int(row.get("point_count", 0))))
    row["height_conf"] = _height_confidence(row["height_m"], row["point_count"])
    row["crossarm_conf"] = _crossarm_confidence(row["crossarm_m"], row["height_m"], row["point_count"])
    row["pole_conf"] = str(row.get("pole_conf", "Low"))
    return row


def _height_confidence(height_m: float, point_count: int) -> str:
    if 8.0 <= height_m <= 24.0 and point_count >= 120:
        return "High"
    if 5.0 <= height_m <= 30.0 and point_count >= 60:
        return "Medium"
    return "Low"


def _pole_confidence(height_m: float, point_count: int, continuity: float, radial_p90: float) -> str:
    if 8.0 <= height_m <= 30.0 and point_count >= 120 and continuity >= 0.55 and radial_p90 <= 0.32:
        return "High"
    if 5.0 <= height_m <= 35.0 and point_count >= 50 and continuity >= 0.35 and radial_p90 <= 0.50:
        return "Medium"
    return "Low"


def _crossarm_confidence(crossarm_m: float, height_m: float, point_count: int) -> str:
    if height_m <= 0.0:
        return "Low"
    if crossarm_m <= 0.0:
        # Zero can be true for no-arm poles, but still needs review often.
        if height_m >= 22.0:
            return "Medium"
        return "Low"
    ratio = crossarm_m / max(height_m, 1e-9)
    if 0.6 <= crossarm_m <= 4.5 and 0.04 <= ratio <= 0.35 and point_count >= 120:
        return "High"
    if 0.35 <= crossarm_m <= 6.0 and 0.02 <= ratio <= 0.45 and point_count >= 60:
        return "Medium"
    return "Low"


def _vertical_continuity(z_vals: np.ndarray, bin_m: float = 0.35) -> float:
    if len(z_vals) < 20:
        return 0.0
    z0 = float(np.min(z_vals))
    z1 = float(np.max(z_vals))
    if z1 - z0 < 2.0:
        return 0.0
    bins = np.arange(z0, z1 + bin_m, bin_m)
    hist, _ = np.histogram(z_vals, bins=bins)
    if len(hist) == 0:
        return 0.0
    occ = np.count_nonzero(hist >= 2)
    return float(occ) / float(len(hist))


def _crossarm_length_from_band(
    support_pts: np.ndarray,
    cross_pts: np.ndarray,
    band: tuple[float, float] | None,
    center_xy: np.ndarray,
    crossarm_min_radius_m: float,
    min_length_m: float,
) -> float:
    if band is None:
        return 0.0
    b0, b1 = band
    bm = (cross_pts[:, 2] >= b0) & (cross_pts[:, 2] <= b1)
    pts_band = cross_pts[bm, :2]
    if len(pts_band) < 6:
        return 0.0

    z_band = cross_pts[bm, 2]
    if len(z_band) >= 8 and float(np.std(z_band)) > 0.32:
        return 0.0

    # Attachment check: an actual arm should intersect near the shaft at same Z band.
    support_band = support_pts[(support_pts[:, 2] >= b0) & (support_pts[:, 2] <= b1)]
    if len(support_band) < 8:
        return 0.0
    radial_support = np.sqrt(np.sum((support_band[:, :2] - center_xy) ** 2, axis=1))
    attach_count = int(np.count_nonzero(radial_support <= max(0.16, crossarm_min_radius_m * 0.9)))

    radial_band = np.sqrt(np.sum((pts_band - center_xy) ** 2, axis=1))
    if len(radial_band) >= 8:
        r_med = float(np.median(radial_band))
        r95 = float(np.percentile(radial_band, 95))
        if r_med < max(0.35, crossarm_min_radius_m * 0.9):
            return 0.0
        if r95 > 3.4:
            return 0.0

    centered = pts_band - np.mean(pts_band, axis=0, keepdims=True)
    cov = centered.T @ centered / max(len(centered) - 1, 1)
    eigvals, eigvecs = np.linalg.eigh(cov)
    major = eigvecs[:, np.argmax(eigvals)]
    minor_std = float(np.sqrt(max(eigvals[0], 1e-9)))
    linearity = _principal_axis_linearity(pts_band)
    if minor_std > 0.45:
        return 0.0
    if linearity < 1.2:
        return 0.0

    rel = pts_band - center_xy
    proj = rel @ major
    left = proj[proj < 0.0]
    right = proj[proj > 0.0]
    left_len = abs(float(np.percentile(left, 15))) if len(left) >= 3 else 0.0
    right_len = abs(float(np.percentile(right, 85))) if len(right) >= 3 else 0.0

    # Primary mode: two-sided arm around shaft.
    if left_len > 0.0 and right_len > 0.0:
        if attach_count < 1:
            return 0.0
        if linearity < 1.8:
            return 0.0
        balance = min(left_len, right_len) / max(left_len, right_len)
        if balance < 0.20:
            return 0.0
        length = left_len + right_len
    else:
        # One-sided fallback for partial scan visibility.
        one_side = max(left_len, right_len)
        if one_side <= 0.0:
            return 0.0
        if linearity < 2.0:
            return 0.0
        if one_side > 3.0:
            return 0.0
        length = one_side

    if length < min_length_m:
        return 0.0
    return length


def _estimate_crossarm_length(
    support_pts: np.ndarray,
    cross_pts: np.ndarray,
    center_xy: np.ndarray,
    z_top: float,
    cfg: DetectionConfig,
) -> float:
    # Try primary adaptive band first.
    band = _choose_crossarm_band(
        points_xy=cross_pts[:, :2],
        points_z=cross_pts[:, 2],
        z_top=z_top,
        search_drop_m=cfg.crossarm_search_drop_m,
        band_thickness_m=cfg.crossarm_band_thickness_m,
        min_drop_from_top_m=cfg.crossarm_min_drop_from_top_m,
        max_drop_from_top_m=cfg.crossarm_max_drop_from_top_m,
    )
    length = _crossarm_length_from_band(
        support_pts=support_pts,
        cross_pts=cross_pts,
        band=band,
        center_xy=center_xy,
        crossarm_min_radius_m=cfg.crossarm_min_radius_m,
        min_length_m=cfg.crossarm_min_length_m,
    )
    if length > 0.0:
        return length

    # Fallback: test fixed near-top windows to recover missed arms.
    fallback_drops = [(0.35, 1.0), (0.8, 1.6), (1.3, 2.2), (1.8, 2.8)]
    best = 0.0
    for d0, d1 in fallback_drops:
        if d1 > cfg.crossarm_max_drop_from_top_m + 1e-9:
            continue
        b1 = z_top - d0
        b0 = z_top - d1
        if b1 <= b0:
            continue
        cand = _crossarm_length_from_band(
            support_pts=support_pts,
            cross_pts=cross_pts,
            band=(b0, b1),
            center_xy=center_xy,
            crossarm_min_radius_m=cfg.crossarm_min_radius_m,
            min_length_m=cfg.crossarm_min_length_m,
        )
        if cand > best:
            best = cand
    return best


def _robust_base_and_top(
    all_points_xyz: np.ndarray,
    support_pts: np.ndarray,
    center_xy: np.ndarray,
    support_radius_m: float,
    base_percentile: float,
    top_percentile: float,
) -> tuple[float, float]:
    # Use local neighborhood around pole center for a stable CloudCompare-like DZ.
    all_radial = np.sqrt(np.sum((all_points_xyz[:, :2] - center_xy) ** 2, axis=1))
    local_ground_mask = all_radial <= max(0.8, support_radius_m)
    if np.any(local_ground_mask):
        local_z = all_points_xyz[local_ground_mask, 2]
        z_base = float(np.percentile(local_z, base_percentile))
    else:
        z_base = float(np.percentile(support_pts[:, 2], base_percentile))

    # Robust top to suppress sparse outlier points/wires.
    z_top = float(np.percentile(support_pts[:, 2], top_percentile))
    return z_base, z_top


def _measure_at_center(
    points_xyz: np.ndarray,
    source_file: Path,
    cfg: DetectionConfig,
    center_xy: np.ndarray,
    pole_id: str,
) -> dict[str, Any] | None:
    support_radius_m = max(cfg.max_pole_radius_m, cfg.max_pole_radius_m * cfg.support_radius_factor)
    support_mask = np.sqrt(np.sum((points_xyz[:, :2] - center_xy) ** 2, axis=1)) <= support_radius_m
    support_pts = points_xyz[support_mask]
    if len(support_pts) < max(30, int(cfg.min_points_per_cluster * 0.3)):
        return None

    support_radial = np.sqrt(np.sum((support_pts[:, :2] - center_xy) ** 2, axis=1))
    shaft_pts_narrow = support_pts[support_radial <= cfg.shaft_radius_m]
    if len(shaft_pts_narrow) < max(20, int(cfg.min_points_per_cluster * 0.2)):
        shaft_pts_narrow = support_pts
    shaft_radial = np.sqrt(np.sum((shaft_pts_narrow[:, :2] - center_xy) ** 2, axis=1))
    shaft_radial_p90 = float(np.percentile(shaft_radial, 90)) if len(shaft_radial) else 1.0
    shaft_continuity = _vertical_continuity(shaft_pts_narrow[:, 2])

    shaft_radius_wide = min(cfg.max_pole_radius_m, max(cfg.shaft_radius_m * 2.2, cfg.shaft_radius_m + 0.35))
    shaft_pts_wide = support_pts[support_radial <= shaft_radius_wide]
    if len(shaft_pts_wide) < len(shaft_pts_narrow):
        shaft_pts_wide = shaft_pts_narrow

    z_base, z_top_narrow = _robust_base_and_top(
        all_points_xyz=points_xyz,
        support_pts=shaft_pts_narrow,
        center_xy=center_xy,
        support_radius_m=support_radius_m,
        base_percentile=cfg.base_percentile,
        top_percentile=cfg.top_percentile,
    )
    z_top_wide = _top_from_continuous_bins(shaft_pts_wide[:, 2])
    z_top = max(z_top_narrow, z_top_wide)
    z_top = _suppress_extreme_top_outliers(shaft_pts_wide[:, 2], z_top)
    z_top = z_top - max(0.0, cfg.top_cap_exclude_m)
    pole_height_m = max(0.0, z_top - z_base)
    if pole_height_m < cfg.min_height_m:
        return None

    cross_mask = (support_radial >= cfg.crossarm_min_radius_m) & (support_radial <= cfg.crossarm_max_radius_m)
    cross_pts = support_pts[cross_mask]
    crossarm_length_m = 0.0
    if len(cross_pts) >= 8:
        crossarm_length_m = _estimate_crossarm_length(
            support_pts=support_pts,
            cross_pts=cross_pts,
            center_xy=center_xy,
            z_top=z_top,
            cfg=cfg,
        )
    crossarm_length_m = min(crossarm_length_m, max(0.0, pole_height_m * 0.6))
    if pole_height_m > 40.0 or pole_height_m < 6.0:
        crossarm_length_m = 0.0

    row = {
        "pole_id": pole_id,
        "source_file": source_file.name,
        "algo_ver": ALGO_VERSION,
        "height_ft": round(pole_height_m * M_TO_FT, 3),
        "crossarm_ft": round(crossarm_length_m * M_TO_FT, 3),
        "height_m": round(pole_height_m, 3),
        "crossarm_m": round(crossarm_length_m, 3),
        "point_count": int(len(shaft_pts_narrow)),
        "pole_conf": _pole_confidence(pole_height_m, int(len(shaft_pts_narrow)), shaft_continuity, shaft_radial_p90),
        "x": float(center_xy[0]),
        "y": float(center_xy[1]),
        "z_base": round(z_base, 3),
        "z_top": round(z_top, 3),
        "config": asdict(cfg),
    }
    return _sanitize_detection(row)


def detect_poles(points_xyz: np.ndarray, source_file: Path, cfg: DetectionConfig) -> list[dict[str, Any]]:
    if len(points_xyz) == 0:
        return []

    ground_z = _estimate_ground(points_xyz, cfg.ground_grid_m)
    norm_z = points_xyz[:, 2] - ground_z

    # Fast path for clipped single-pole files: skip DBSCAN entirely.
    if cfg.keep_only_tallest_per_file:
        tall = points_xyz[norm_z >= max(cfg.min_height_m, 2.5)]
        if len(tall) < cfg.min_samples:
            return []

        candidate_centers: list[np.ndarray] = []
        # Candidate-1: strongest shaft from global scan.
        c0 = _find_best_shaft_center(points_xyz, norm_z, cfg)
        if c0 is not None:
            candidate_centers.append(c0)

        # Candidate-2..N: densest tall XY cells then refined to vertical shaft.
        cell_m = 0.35
        grid = np.floor(tall[:, :2] / cell_m).astype(np.int32)
        keys = grid[:, 0].astype(np.int64) * 104729 + grid[:, 1].astype(np.int64) * 130363
        uniq, counts = np.unique(keys, return_counts=True)
        order = np.argsort(counts)[::-1]
        for key in uniq[order[:10]]:
            m = keys == key
            seed = np.median(tall[m, :2], axis=0)
            center = _refine_center_to_vertical_shaft(points_xyz, seed, search_radius_m=2.8, cell_m=0.22)
            candidate_centers.append(center)

        # Fallback candidate: robust median of tall points.
        candidate_centers.append(np.median(tall[:, :2], axis=0))

        # Deduplicate candidate centers.
        deduped: list[np.ndarray] = []
        for c in candidate_centers:
            if c is None:
                continue
            if not deduped:
                deduped.append(c)
                continue
            keep = True
            for k in deduped:
                if float(np.linalg.norm(c - k)) <= 0.55:
                    keep = False
                    break
            if keep:
                deduped.append(c)

        best_row: dict[str, Any] | None = None
        best_score = -1e18
        for c in deduped:
            measured = _measure_at_center(points_xyz, source_file, cfg, c, f"{source_file.stem}_0001")
            if measured is None:
                continue
            s = _pole_quality_score(measured, cfg)
            if s > best_score:
                best_score = s
                best_row = measured
        return [best_row] if best_row is not None else []

    try:
        from sklearn.cluster import DBSCAN
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Missing dependency 'scikit-learn'. Rebuild the EXE after installing requirements."
        ) from exc

    candidate_mask = norm_z >= cfg.min_height_m
    candidate_indices_all = np.where(candidate_mask)[0]
    candidate_xy_all = points_xyz[candidate_indices_all, :2]

    if len(candidate_xy_all) < cfg.min_samples:
        return []

    # Guardrail for very dense scenes: DBSCAN can become extremely slow.
    # Use a stricter cap in clipped mode for speed; bulk mode keeps higher cap for recall.
    max_dbscan_points = MAX_DBSCAN_POINTS_CLIPPED if cfg.keep_only_tallest_per_file else MAX_DBSCAN_POINTS_BULK
    if len(candidate_xy_all) > max_dbscan_points:
        rng = np.random.default_rng(42)
        sampled_pos = rng.choice(len(candidate_xy_all), size=max_dbscan_points, replace=False)
        sampled_pos.sort()
        candidate_indices = candidate_indices_all[sampled_pos]
        candidate_xy = candidate_xy_all[sampled_pos]
    else:
        candidate_indices = candidate_indices_all
        candidate_xy = candidate_xy_all

    clustering = DBSCAN(eps=cfg.eps, min_samples=cfg.min_samples)
    labels = clustering.fit_predict(candidate_xy)

    detections: list[dict[str, Any]] = []
    global_center = _find_best_shaft_center(points_xyz, norm_z, cfg) if cfg.keep_only_tallest_per_file else None
    unique_labels = [v for v in np.unique(labels) if v >= 0]

    for idx, label in enumerate(unique_labels, start=1):
        in_label_candidates = candidate_indices[labels == label]
        cluster_pts = points_xyz[in_label_candidates]

        if len(cluster_pts) < cfg.min_points_per_cluster:
            continue

        center_xy_seed = np.median(cluster_pts[:, :2], axis=0)
        center_xy = _refine_center_to_vertical_shaft(points_xyz, center_xy_seed)
        use_global_center = global_center is not None
        if use_global_center:
            center_xy = global_center
        radial = np.sqrt(np.sum((cluster_pts[:, :2] - center_xy) ** 2, axis=1))
        if (not use_global_center) and float(np.percentile(radial, 95)) > cfg.max_pole_radius_m:
            continue

        support_radius_m = max(cfg.max_pole_radius_m, cfg.max_pole_radius_m * cfg.support_radius_factor)
        support_mask = np.sqrt(np.sum((points_xyz[:, :2] - center_xy) ** 2, axis=1)) <= support_radius_m
        support_pts = points_xyz[support_mask]
        if len(support_pts) < cfg.min_points_per_cluster:
            support_pts = cluster_pts

        support_radial = np.sqrt(np.sum((support_pts[:, :2] - center_xy) ** 2, axis=1))
        shaft_pts_narrow = support_pts[support_radial <= cfg.shaft_radius_m]
        if len(shaft_pts_narrow) < max(20, int(cfg.min_points_per_cluster * 0.2)):
            shaft_pts_narrow = support_pts
        shaft_radial = np.sqrt(np.sum((shaft_pts_narrow[:, :2] - center_xy) ** 2, axis=1))
        shaft_radial_p90 = float(np.percentile(shaft_radial, 90)) if len(shaft_radial) else 1.0
        shaft_continuity = _vertical_continuity(shaft_pts_narrow[:, 2])

        shaft_radius_wide = min(cfg.max_pole_radius_m, max(cfg.shaft_radius_m * 2.2, cfg.shaft_radius_m + 0.35))
        shaft_pts_wide = support_pts[support_radial <= shaft_radius_wide]
        if len(shaft_pts_wide) < len(shaft_pts_narrow):
            shaft_pts_wide = shaft_pts_narrow

        z_base, z_top_narrow = _robust_base_and_top(
            all_points_xyz=points_xyz,
            support_pts=shaft_pts_narrow,
            center_xy=center_xy,
            support_radius_m=support_radius_m,
            base_percentile=cfg.base_percentile,
            top_percentile=cfg.top_percentile,
        )
        z_top_wide = _top_from_continuous_bins(shaft_pts_wide[:, 2])
        z_top = max(z_top_narrow, z_top_wide)
        z_top = _suppress_extreme_top_outliers(shaft_pts_wide[:, 2], z_top)
        z_top = z_top - max(0.0, cfg.top_cap_exclude_m)
        pole_height_m = max(0.0, z_top - z_base)
        pole_height_ft = pole_height_m * M_TO_FT
        if pole_height_m < cfg.min_height_m:
            continue

        cross_mask = (support_radial >= cfg.crossarm_min_radius_m) & (support_radial <= cfg.crossarm_max_radius_m)
        cross_pts = support_pts[cross_mask]
        crossarm_length_m = 0.0
        if len(cross_pts) >= 8:
            crossarm_length_m = _estimate_crossarm_length(
                support_pts=support_pts,
                cross_pts=cross_pts,
                center_xy=center_xy,
                z_top=z_top,
                cfg=cfg,
            )
        # Physical sanity: cross-arm cannot be longer than a large fraction of pole height.
        crossarm_length_m = min(crossarm_length_m, max(0.0, pole_height_m * 0.6))
        # Very tall structures in this workflow are usually not utility poles with crossarms.
        if pole_height_m > 24.0:
            crossarm_length_m = 0.0
        if pole_height_m < 6.0:
            crossarm_length_m = 0.0
        crossarm_length_ft = crossarm_length_m * M_TO_FT

        row = {
            "pole_id": f"{source_file.stem}_{idx:04d}",
            "source_file": source_file.name,
            "algo_ver": ALGO_VERSION,
            "height_ft": round(pole_height_ft, 3),
            "crossarm_ft": round(crossarm_length_ft, 3),
            "height_m": round(pole_height_m, 3),
            "crossarm_m": round(crossarm_length_m, 3),
            "point_count": int(len(shaft_pts_narrow)),
            "pole_conf": _pole_confidence(pole_height_m, int(len(shaft_pts_narrow)), shaft_continuity, shaft_radial_p90),
            "x": float(center_xy[0]),
            "y": float(center_xy[1]),
            "z_base": round(z_base, 3),
            "z_top": round(z_top, 3),
            "config": asdict(cfg),
        }
        clean = _sanitize_detection(row)
        if clean is not None:
            detections.append(clean)

    if not detections and global_center is not None:
        measured = _measure_at_center(points_xyz, source_file, cfg, global_center, f"{source_file.stem}_0001")
        if measured is not None:
            detections.append(measured)

    if cfg.keep_only_tallest_per_file and detections:
        best = max(detections, key=lambda d: _pole_quality_score(d, cfg))
        return [best]
    if not cfg.keep_only_tallest_per_file:
        return _bulk_post_filter(detections, cfg)

    return detections
