import argparse
import sys
import time
from pathlib import Path
from typing import Callable

import numpy as np
from .config import DetectionConfig
from .detection import ALGO_VERSION, detect_poles
from .io_utils import (
    detect_linear_unit_from_crs,
    list_lidar_files,
    read_las_points_and_crs,
    transform_xy,
    unit_to_meter_scale,
    write_csv,
    write_geojson,
    write_vector_outputs,
)
from .ml_utils import filter_rows_with_model


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lidar_pole_measure",
        description="Detect poles from LAS/LAZ and export QGIS-ready GIS points.",
    )
    parser.add_argument("--input", required=True, type=Path, help="Input folder containing .las/.laz")
    parser.add_argument("--output", required=True, type=Path, help="Output folder")
    parser.add_argument("--recursive", action="store_true", help="Search input folder recursively")
    parser.add_argument("--eps", type=float, default=0.8, help="DBSCAN eps in meters (XY)")
    parser.add_argument("--min-samples", type=int, default=25, help="DBSCAN min samples")
    parser.add_argument("--min-height", type=float, default=3.0, help="Minimum pole height in meters")
    parser.add_argument(
        "--crossarm-top-band",
        type=float,
        default=1.0,
        help="Top vertical band (m) used to estimate cross-arm length",
    )
    parser.add_argument("--max-pole-radius", type=float, default=1.6, help="Maximum pole radial spread (m)")
    parser.add_argument("--min-cluster-points", type=int, default=80, help="Minimum points per detected pole")
    parser.add_argument("--ground-grid", type=float, default=1.0, help="Ground estimation grid size (m)")
    parser.add_argument(
        "--unit-override",
        choices=["auto", "meters", "us_survey_feet", "international_feet"],
        default="auto",
        help="Coordinate linear unit override for input LAS/LAZ.",
    )
    parser.add_argument(
        "--force-output-4326",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Force output GIS to EPSG:4326 (default on).",
    )
    return parser


def process_folder(
    input_dir: Path,
    output_dir: Path,
    cfg: DetectionConfig,
    recursive: bool = False,
    unit_override: str = "auto",
    force_output_4326: bool = True,
    write_geojson_output: bool = True,
    write_gpkg_output: bool = True,
    write_shp_output: bool = True,
    write_csv_output: bool = True,
    min_pole_conf: str = "low",
    ml_model_path: Path | None = None,
    ml_threshold: float = 0.5,
    progress_callback: Callable[[int, int, str], None] | None = None,
    log_callback: Callable[[str], None] | None = None,
) -> dict:
    if not input_dir.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_dir}")

    lidar_files = list_lidar_files(input_dir, recursive=recursive)
    if not lidar_files:
        return {"count": 0, "files": 0, "geojson": None, "csv": None, "gpkg": None, "shp": None}

    all_rows: list[dict] = []
    output_crs = "EPSG:4326" if force_output_4326 else "None"
    had_missing_crs = False
    failed_files: list[tuple[str, str]] = []
    if log_callback:
        log_callback(f"Algorithm version: {ALGO_VERSION}")
    for i, f in enumerate(lidar_files, start=1):
        t0 = time.perf_counter()
        if progress_callback:
            progress_callback(i - 1, len(lidar_files), f.name)
        msg = f"Processing: {f}"
        print(msg)
        if log_callback:
            log_callback(msg)
        try:
            pts_raw, source_crs = read_las_points_and_crs(f)
        except Exception as exc:
            failed_files.append((f.name, str(exc)))
            if log_callback:
                log_callback(f"  failed to read file: {exc}")
            if progress_callback:
                progress_callback(i, len(lidar_files), f.name)
            continue
        if log_callback:
            log_callback(f"  points loaded: {len(pts_raw)}")

        detected_unit = detect_linear_unit_from_crs(source_crs)
        use_unit = unit_override if unit_override != "auto" else (detected_unit or "meters")
        scale_to_m = unit_to_meter_scale(use_unit)
        if log_callback:
            crs_txt = source_crs.to_string() if source_crs is not None else "None"
            log_callback(f"  CRS: {crs_txt} | unit mode: {use_unit}")
        pts_m = pts_raw.copy()
        pts_m[:, :3] = pts_m[:, :3] * scale_to_m

        try:
            poles = detect_poles(pts_m, f, cfg)
        except Exception as exc:
            failed_files.append((f.name, str(exc)))
            if log_callback:
                log_callback(f"  failed to detect poles: {exc}")
            if progress_callback:
                progress_callback(i, len(lidar_files), f.name)
            continue
        conf_rank = {"low": 1, "medium": 2, "high": 3}
        req_rank = conf_rank.get(min_pole_conf.lower(), 1)
        poles = [p for p in poles if conf_rank.get(str(p.get("pole_conf", "Low")).lower(), 1) >= req_rank]
        if ml_model_path is not None and poles:
            try:
                before = len(poles)
                poles = filter_rows_with_model(poles, ml_model_path, threshold=ml_threshold)
                if log_callback:
                    log_callback(f"  ML filter kept {len(poles)}/{before} poles (threshold={ml_threshold:.2f})")
            except Exception as exc:
                if log_callback:
                    log_callback(f"  ML filter skipped due to error: {exc}")
        if poles:
            x_vals = np.array([row["x"] for row in poles], dtype=float) / scale_to_m
            y_vals = np.array([row["y"] for row in poles], dtype=float) / scale_to_m
            out_x, out_y, this_output_crs, missing_crs_assigned = transform_xy(
                x_vals, y_vals, source_crs, force_output_4326
            )
            output_crs = this_output_crs
            had_missing_crs = had_missing_crs or missing_crs_assigned
            for idx, row in enumerate(poles):
                row["x"] = float(out_x[idx])
                row["y"] = float(out_y[idx])
        all_rows.extend(poles)
        msg2 = f"  poles detected: {len(poles)}"
        print(msg2)
        if log_callback:
            log_callback(msg2)
            log_callback(f"  elapsed: {time.perf_counter() - t0:.1f}s")
        if progress_callback:
            progress_callback(i, len(lidar_files), f.name)

    features = []
    for row in all_rows:
        props = dict(row)
        x = props.pop("x")
        y = props.pop("y")
        props.pop("config", None)
        features.append(
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [x, y]},
                "properties": props,
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    geojson_path = output_dir / "poles.geojson"
    csv_path = output_dir / "poles.csv"
    gpkg_path = output_dir / "poles.gpkg"
    shp_path = output_dir / "poles.shp"

    if write_geojson_output:
        write_geojson(features, geojson_path)
    if write_gpkg_output or write_shp_output:
        write_vector_outputs(
            all_rows,
            gpkg_path if write_gpkg_output else None,
            shp_path if write_shp_output else None,
            output_crs=output_crs,
        )

    csv_rows = []
    all_rows_sorted = sorted(all_rows, key=lambda r: (str(r.get("source_file", "")), str(r.get("pole_id", ""))))
    for row in all_rows_sorted:
        copy = dict(row)
        copy.pop("config", None)
        csv_rows.append(copy)
    if write_csv_output:
        write_csv(csv_rows, csv_path)

    return {
        "count": len(all_rows_sorted),
        "files": len(lidar_files),
        "geojson": geojson_path if write_geojson_output else None,
        "csv": csv_path if write_csv_output else None,
        "gpkg": gpkg_path if write_gpkg_output else None,
        "shp": shp_path if write_shp_output else None,
        "output_crs": output_crs,
        "warning_missing_crs_assumed_4326": had_missing_crs,
        "failed_files": failed_files,
    }


def run() -> int:
    args = _build_parser().parse_args()
    input_dir: Path = args.input
    output_dir: Path = args.output
    cfg = DetectionConfig(
        eps=args.eps,
        min_samples=args.min_samples,
        min_height_m=args.min_height,
        max_pole_radius_m=args.max_pole_radius,
        crossarm_top_band_m=args.crossarm_top_band,
        min_points_per_cluster=args.min_cluster_points,
        ground_grid_m=args.ground_grid,
    )
    result = process_folder(
        input_dir=input_dir,
        output_dir=output_dir,
        cfg=cfg,
        recursive=args.recursive,
        unit_override=args.unit_override,
        force_output_4326=args.force_output_4326,
    )
    if result["files"] == 0:
        print(f"No LAS/LAZ files found in: {input_dir}")
        return 1
    print(f"Done. Processed files: {result['files']}")
    print(f"Done. Total poles detected: {result['count']}")
    if result["geojson"] is not None:
        print(f"GeoJSON: {result['geojson']}")
    if result["gpkg"] is not None:
        print(f"GPKG: {result['gpkg']}")
    if result["shp"] is not None:
        print(f"SHP: {result['shp']}")
    if result["csv"] is not None:
        print(f"CSV: {result['csv']}")
    print(f"Output CRS: {result['output_crs']}")
    if result["warning_missing_crs_assumed_4326"]:
        print("Warning: Some files had no CRS metadata; output was tagged EPSG:4326 without coordinate transform.")
    return 0


if __name__ == "__main__":
    if len(sys.argv) > 1:
        raise SystemExit(run())
    from .ui import run_ui

    run_ui()
