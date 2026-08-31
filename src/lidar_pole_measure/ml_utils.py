from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier

DEFAULT_FEATURE_COLUMNS = ["height_m", "crossarm_m", "point_count", "z_base", "z_top"]


def _to_label(value: str) -> int:
    t = (value or "").strip().lower()
    if t in {"1", "true", "yes", "y", "pole", "high", "keep"}:
        return 1
    if t in {"0", "false", "no", "n", "not_pole", "low", "drop"}:
        return 0
    try:
        return 1 if float(t) >= 0.5 else 0
    except Exception:
        return 0


def train_pole_model_from_csv(
    csv_path: Path,
    output_model_path: Path,
    label_column: str = "pole_label",
    feature_columns: list[str] | None = None,
) -> dict[str, Any]:
    cols = feature_columns or DEFAULT_FEATURE_COLUMNS
    X: list[list[float]] = []
    y: list[int] = []

    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if label_column not in (reader.fieldnames or []):
            raise ValueError(f"Label column not found: {label_column}")
        for row in reader:
            try:
                feat = [float(row.get(c, "nan")) for c in cols]
            except Exception:
                continue
            if not np.all(np.isfinite(feat)):
                continue
            X.append(feat)
            y.append(_to_label(str(row.get(label_column, "0"))))

    if len(X) < 30:
        raise ValueError("Need at least 30 valid labeled rows for training.")
    if len(set(y)) < 2:
        raise ValueError("Training labels must include both classes (pole and non-pole).")

    model = RandomForestClassifier(
        n_estimators=250,
        max_depth=10,
        min_samples_leaf=4,
        random_state=42,
        class_weight="balanced_subsample",
    )
    model.fit(np.asarray(X, dtype=np.float64), np.asarray(y, dtype=np.int32))

    payload = {"model": model, "feature_columns": cols, "label_column": label_column}
    output_model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, output_model_path)
    return {"rows_used": len(X), "label_column": label_column, "model_path": output_model_path}


def filter_rows_with_model(rows: list[dict[str, Any]], model_path: Path, threshold: float = 0.5) -> list[dict[str, Any]]:
    payload = joblib.load(model_path)
    model = payload["model"]
    cols: list[str] = payload["feature_columns"]
    keep: list[dict[str, Any]] = []
    for row in rows:
        feat = []
        ok = True
        for c in cols:
            try:
                feat.append(float(row.get(c, "nan")))
            except Exception:
                ok = False
                break
        if (not ok) or (not np.all(np.isfinite(feat))):
            continue
        arr = np.asarray([feat], dtype=np.float64)
        if hasattr(model, "predict_proba"):
            p = float(model.predict_proba(arr)[0, 1])
        else:
            p = float(model.predict(arr)[0])
        row["pole_ml_prob"] = round(p, 3)
        if p >= threshold:
            keep.append(row)
    return keep
