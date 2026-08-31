import json
import sqlite3
import struct
from pathlib import Path
from typing import Iterable

import laspy
import numpy as np
from pyproj import CRS, Transformer
import shapefile
from shapely.geometry import Point
from shapely.wkb import dumps as wkb_dumps

US_SURVEY_FT_TO_M = 1200.0 / 3937.0
INTL_FT_TO_M = 0.3048


def list_lidar_files(input_dir: Path, recursive: bool) -> list[Path]:
    patterns = ["*.las", "*.laz"]
    files: list[Path] = []
    for pattern in patterns:
        if recursive:
            files.extend(input_dir.rglob(pattern))
        else:
            files.extend(input_dir.glob(pattern))
    dedup = sorted(set(files))
    # If both "name.laz" and "name.copc.laz" exist, keep the original .laz only.
    names = {p.name.lower() for p in dedup}
    filtered: list[Path] = []
    for p in dedup:
        low = p.name.lower()
        if low.endswith(".copc.laz"):
            plain = low.replace(".copc.laz", ".laz")
            if plain in names:
                continue
        filtered.append(p)
    return filtered


def read_las_points(path: Path) -> np.ndarray:
    las = laspy.read(path)
    xyz = np.column_stack((las.x, las.y, las.z)).astype(np.float64, copy=False)
    return xyz


def read_las_points_and_crs(path: Path) -> tuple[np.ndarray, CRS | None]:
    las = laspy.read(path)
    xyz = np.column_stack((las.x, las.y, las.z)).astype(np.float64, copy=False)
    crs = las.header.parse_crs()
    return xyz, crs


def detect_linear_unit_from_crs(crs: CRS | None) -> str | None:
    if crs is None:
        return None
    axis_info = crs.axis_info or []
    for axis in axis_info:
        unit_name = (axis.unit_name or "").lower()
        if "metre" in unit_name or "meter" in unit_name:
            return "meters"
        if "us survey foot" in unit_name:
            return "us_survey_feet"
        if "foot" in unit_name:
            return "international_feet"
    return None


def unit_to_meter_scale(unit_name: str) -> float:
    if unit_name == "meters":
        return 1.0
    if unit_name == "us_survey_feet":
        return US_SURVEY_FT_TO_M
    if unit_name == "international_feet":
        return INTL_FT_TO_M
    raise ValueError(f"Unsupported unit override: {unit_name}")


def transform_xy(
    x: np.ndarray,
    y: np.ndarray,
    source_crs: CRS | None,
    force_output_epsg_4326: bool,
) -> tuple[np.ndarray, np.ndarray, str, bool]:
    if not force_output_epsg_4326:
        if source_crs is None:
            return x, y, "None", False
        return x, y, source_crs.to_string(), False

    if source_crs is None:
        return x, y, "EPSG:4326", True

    dst = CRS.from_epsg(4326)
    if source_crs == dst:
        return x, y, "EPSG:4326", False

    transformer = Transformer.from_crs(source_crs, dst, always_xy=True)
    lon, lat = transformer.transform(x, y)
    return np.asarray(lon), np.asarray(lat), "EPSG:4326", False


def write_geojson(features: Iterable[dict], out_path: Path) -> None:
    payload = {
        "type": "FeatureCollection",
        "features": list(features),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_csv(rows: list[dict], out_path: Path) -> None:
    import csv

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        out_path.write_text("", encoding="utf-8")
        return

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_vector_outputs(
    rows: list[dict],
    out_gpkg: Path | None,
    out_shp: Path | None,
    output_crs: str | None,
) -> None:
    if out_shp is not None:
        _write_shp(rows, out_shp, output_crs)
    if out_gpkg is not None:
        _write_gpkg(rows, out_gpkg, output_crs)


def _write_shp(rows: list[dict], out_shp: Path, output_crs: str | None) -> None:
    out_shp.parent.mkdir(parents=True, exist_ok=True)
    with shapefile.Writer(str(out_shp), shapeType=shapefile.POINT) as w:
        w.field("pole_id", "C", size=64)
        w.field("source_fil", "C", size=80)
        w.field("height_ft", "F", decimal=3)
        w.field("crossarm_f", "F", decimal=3)
        w.field("height_m", "F", decimal=3)
        w.field("crossarm_m", "F", decimal=3)
        w.field("pole_conf", "C", size=10)
        w.field("h_conf", "C", size=10)
        w.field("c_conf", "C", size=10)
        w.field("pt_count", "N")
        w.field("z_base", "F", decimal=3)
        w.field("z_top", "F", decimal=3)

        for row in rows:
            x = float(row["x"])
            y = float(row["y"])
            w.point(x, y)
            w.record(
                row.get("pole_id", ""),
                row.get("source_file", ""),
                float(row.get("height_ft", 0.0)),
                float(row.get("crossarm_ft", 0.0)),
                float(row.get("height_m", 0.0)),
                float(row.get("crossarm_m", 0.0)),
                row.get("pole_conf", ""),
                row.get("height_conf", ""),
                row.get("crossarm_conf", ""),
                int(row.get("point_count", 0)),
                float(row.get("z_base", 0.0)),
                float(row.get("z_top", 0.0)),
            )

    if output_crs and output_crs != "None":
        try:
            crs = CRS.from_user_input(output_crs)
            out_shp.with_suffix(".prj").write_text(crs.to_wkt(), encoding="utf-8")
        except Exception:
            pass


def _epsg_from_crs_text(output_crs: str | None) -> int:
    if not output_crs or output_crs == "None":
        return 0
    try:
        c = CRS.from_user_input(output_crs)
        epsg = c.to_epsg()
        return int(epsg) if epsg else 0
    except Exception:
        return 0


def _gpkg_geom_point(x: float, y: float, srs_id: int) -> bytes:
    wkb = wkb_dumps(Point(x, y), hex=False)
    # GPKG binary header: magic(2), version(1), flags(1), srs_id(int32 LE), then WKB
    # flags=1 -> little endian, no envelope, standard geometry
    header = b"GP" + bytes([0, 1]) + struct.pack("<i", srs_id)
    return header + wkb


def _write_gpkg(rows: list[dict], out_gpkg: Path, output_crs: str | None) -> None:
    out_gpkg.parent.mkdir(parents=True, exist_ok=True)
    if out_gpkg.exists():
        out_gpkg.unlink()

    srs_id = _epsg_from_crs_text(output_crs)
    srs_name = output_crs if output_crs and output_crs != "None" else "Undefined Cartesian"
    org = "EPSG" if srs_id != 0 else "NONE"
    org_coordsys_id = srs_id if srs_id != 0 else -1
    definition = CRS.from_user_input(output_crs).to_wkt() if srs_id != 0 else "undefined"

    conn = sqlite3.connect(str(out_gpkg))
    try:
        cur = conn.cursor()
        cur.execute("PRAGMA application_id = 1196437808;")
        cur.execute("PRAGMA user_version = 10300;")
        cur.execute(
            """
            CREATE TABLE gpkg_spatial_ref_sys (
                srs_name TEXT NOT NULL,
                srs_id INTEGER NOT NULL PRIMARY KEY,
                organization TEXT NOT NULL,
                organization_coordsys_id INTEGER NOT NULL,
                definition TEXT NOT NULL,
                description TEXT
            );
            """
        )
        cur.execute(
            """
            CREATE TABLE gpkg_contents (
                table_name TEXT NOT NULL PRIMARY KEY,
                data_type TEXT NOT NULL,
                identifier TEXT UNIQUE,
                description TEXT DEFAULT '',
                last_change DATETIME NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
                min_x DOUBLE, min_y DOUBLE, max_x DOUBLE, max_y DOUBLE,
                srs_id INTEGER,
                CONSTRAINT fk_gc_r_srs_id FOREIGN KEY (srs_id) REFERENCES gpkg_spatial_ref_sys(srs_id)
            );
            """
        )
        cur.execute(
            """
            CREATE TABLE gpkg_geometry_columns (
                table_name TEXT NOT NULL,
                column_name TEXT NOT NULL,
                geometry_type_name TEXT NOT NULL,
                srs_id INTEGER NOT NULL,
                z TINYINT NOT NULL,
                m TINYINT NOT NULL,
                PRIMARY KEY (table_name, column_name),
                CONSTRAINT fk_gc_tn FOREIGN KEY (table_name) REFERENCES gpkg_contents(table_name),
                CONSTRAINT fk_gc_srs FOREIGN KEY (srs_id) REFERENCES gpkg_spatial_ref_sys(srs_id)
            );
            """
        )
        cur.execute(
            """
            INSERT INTO gpkg_spatial_ref_sys
            (srs_name, srs_id, organization, organization_coordsys_id, definition, description)
            VALUES (?, ?, ?, ?, ?, ?);
            """,
            (srs_name, srs_id, org, org_coordsys_id, definition, ""),
        )
        cur.execute(
            """
            CREATE TABLE poles (
                fid INTEGER PRIMARY KEY AUTOINCREMENT,
                geom BLOB NOT NULL,
                pole_id TEXT,
                source_file TEXT,
                height_ft REAL,
                crossarm_ft REAL,
                height_m REAL,
                crossarm_m REAL,
                pole_conf TEXT,
                height_conf TEXT,
                crossarm_conf TEXT,
                point_count INTEGER,
                z_base REAL,
                z_top REAL
            );
            """
        )
        cur.execute(
            """
            INSERT INTO gpkg_contents (table_name, data_type, identifier, description, srs_id)
            VALUES ('poles', 'features', 'poles', '', ?);
            """,
            (srs_id,),
        )
        cur.execute(
            """
            INSERT INTO gpkg_geometry_columns
            (table_name, column_name, geometry_type_name, srs_id, z, m)
            VALUES ('poles', 'geom', 'POINT', ?, 0, 0);
            """,
            (srs_id,),
        )

        if rows:
            xs = [float(r["x"]) for r in rows]
            ys = [float(r["y"]) for r in rows]
            cur.execute(
                """
                UPDATE gpkg_contents
                SET min_x=?, min_y=?, max_x=?, max_y=?
                WHERE table_name='poles';
                """,
                (min(xs), min(ys), max(xs), max(ys)),
            )

        for row in rows:
            geom = _gpkg_geom_point(float(row["x"]), float(row["y"]), srs_id)
            cur.execute(
                """
                INSERT INTO poles
                (geom, pole_id, source_file, height_ft, crossarm_ft, height_m, crossarm_m, pole_conf, height_conf, crossarm_conf, point_count, z_base, z_top)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    sqlite3.Binary(geom),
                    row.get("pole_id", ""),
                    row.get("source_file", ""),
                    float(row.get("height_ft", 0.0)),
                    float(row.get("crossarm_ft", 0.0)),
                    float(row.get("height_m", 0.0)),
                    float(row.get("crossarm_m", 0.0)),
                    row.get("pole_conf", ""),
                    row.get("height_conf", ""),
                    row.get("crossarm_conf", ""),
                    int(row.get("point_count", 0)),
                    float(row.get("z_base", 0.0)),
                    float(row.get("z_top", 0.0)),
                ),
            )
        conn.commit()
    finally:
        conn.close()
