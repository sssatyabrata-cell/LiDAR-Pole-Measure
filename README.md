# Lidar Pole Measure

Standalone Python application to:
- read `.las` and `.laz` point cloud files,
- detect pole-like structures,
- estimate pole height and cross-arm length,
- export pole positions with attributes to QGIS-friendly GIS files.

## Output

The app writes:
- `poles.geojson` (load directly in QGIS),
- `poles.gpkg` (GeoPackage for QGIS),
- `poles.shp` (Shapefile),
- `poles.csv` (tabular export).

Each detected pole includes:
- `pole_id`
- `source_file`
- `height_ft` (pole height in feet)
- `crossarm_ft` (cross-arm length in feet)
- `height_m`
- `crossarm_m`
- `pole_conf` (`High` / `Medium` / `Low`)
- `height_conf` (`High` / `Medium` / `Low`)
- `crossarm_conf` (`High` / `Medium` / `Low`)
- `point_count`
- `x`
- `y`
- `z_base`
- `z_top`

## 1) Install

```powershell
cd "D:\Cursor Project\Lidar Pole Measure"
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## 2) Run (CLI)

```powershell
python -m lidar_pole_measure.main ^
  --input "D:\Unknown\Lidar_Test\LAZ" ^
  --output "D:\Cursor Project\Lidar Pole Measure\output"
```

Use `--recursive` if files are in nested folders.

If your files are directly in `D:\Unknown\Lidar_Test\LAZ`, this should detect them without `--recursive`.

## 3) Run (Desktop Window)

```powershell
python run_app.py
```

This opens a small app where you can browse input/output folders and run detection.

UI includes:
- progress bar,
- automatic system dark/light theme adaptation,
- live log box for processing messages/errors,
- clear button,
- mode tabs: `Clipped Pole Mode`, `Large LAZ Auto-Detect`, `Machine Learning`,
- coordinate unit override (`auto`, `meters`, `us_survey_feet`, `international_feet`),
- minimum pole confidence filter (`Low` / `Medium` / `High`) for large LAZ runs,
- optional ML model (`.joblib`) + threshold filter during detection,
- output format checkboxes (`GeoJSON`, `GPKG`, `SHP`, `CSV`),
- force output CRS to `EPSG:4326` (enabled by default).

Machine Learning tab:
- train a pole classifier from a labeled CSV (default label column: `pole_label`),
- save model to `.joblib`,
- use that model in Large LAZ mode to reduce false positives.

## 4) Tune Detection (Optional, CLI)

```powershell
python -m lidar_pole_measure.main ^
  --input "D:\Unknown\Lidar_Test\LAZ" ^
  --output "D:\Cursor Project\Lidar Pole Measure\output" ^
  --eps 0.9 ^
  --min-samples 30 ^
  --min-height 4.0 ^
  --crossarm-top-band 1.2
```

## 5) Build Standalone EXE

```powershell
.\build_exe.bat
```

EXE will be under `dist\lidar_pole_measure.exe`.

Double-click behavior:
- With no arguments, EXE opens desktop window mode.
- With arguments, it runs CLI mode.

To embed app icon into EXE:
- place PNG at `assets/pole_icon.png`
- run `.\build_exe.bat`
- script auto-generates `assets/pole_icon.ico` and embeds it into EXE

If you build manually, ensure dependencies are installed in the same environment and include scikit-learn bundles:

```powershell
python -m pip install -r requirements.txt
python -m pip install pyinstaller
python -m PyInstaller --noconfirm --clean --onefile --windowed --name lidar_pole_measure --paths src --collect-all sklearn --collect-all scipy --collect-all joblib --collect-all threadpoolctl run_app.py
```

## Notes

- LAZ support is provided by `laspy[lazrs]`.
- This is an automated geometric estimate. For best results, use moderate-density, well-classified utility corridor data.
- If your LAS/LAZ has CRS metadata, QGIS can still load pole coordinates correctly from GeoJSON coordinates.
