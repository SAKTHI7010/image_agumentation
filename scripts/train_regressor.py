"""Phase 3B: Inspect numeric metadata targets and train continuous regressor only where genuine labels exist."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from modeling_utils import build_leakage_safe_splits, select_numeric_feature_columns  # noqa: E402


def eval_regression(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(math.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))
    return {"mae": round(mae, 4), "rmse": round(rmse, 4), "r2": round(r2, 4)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train regressor for genuine numeric targets (Air flow [L/min]).")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()

    root: Path = args.project_root
    features_csv = root / "processed" / "features" / "features.csv"
    splits_csv = root / "manifests" / "ml_splits.csv"
    models_dir = root / "models"
    outputs_dir = root / "outputs"
    models_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(features_csv)
    splits_df = build_leakage_safe_splits(df, splits_csv)
    df = df.merge(splits_df[["image_id", "split"]], on="image_id", how="inner")

    feature_cols = select_numeric_feature_columns(df)

    # Metadata target audit
    has_froth_depth = "froth_depth" in df.columns
    valid_air_df = df[df["air_flow_l_min"].notna()].copy()
    excluded_air_df = df[df["air_flow_l_min"].isna()].copy()

    train_air = valid_air_df[valid_air_df["split"] == "train"]
    test_air = valid_air_df[valid_air_df["split"] == "test"]

    rf_reg = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("rf", RandomForestRegressor(n_estimators=300, max_features="sqrt", random_state=42, n_jobs=2)),
    ])
    rf_reg.fit(train_air[feature_cols], train_air["air_flow_l_min"].astype(float))
    test_pred = rf_reg.predict(test_air[feature_cols])
    within_metrics = eval_regression(test_air["air_flow_l_min"].to_numpy(dtype=float), test_pred)

    # Per-setpoint summary on test set
    setpoint_df = pd.DataFrame({
        "actual_air_flow": test_air["air_flow_l_min"].to_numpy(dtype=float),
        "predicted_air_flow": test_pred,
    })
    setpoint_df["abs_err"] = np.abs(setpoint_df["predicted_air_flow"] - setpoint_df["actual_air_flow"])
    per_setpoint = (
        setpoint_df.groupby("actual_air_flow")
        .agg(
            count=("predicted_air_flow", "count"),
            pred_mean=("predicted_air_flow", "mean"),
            pred_std=("predicted_air_flow", "std"),
            mae=("abs_err", "mean"),
        )
        .reset_index()
        .round(4)
        .to_dict(orient="records")
    )

    # Category-held-out GroupKFold evaluation (5 folds grouped by category_id)
    gkf = GroupKFold(n_splits=5)
    oof_preds = np.zeros(len(valid_air_df), dtype=float)
    y_all = valid_air_df["air_flow_l_min"].to_numpy(dtype=float)
    X_all = valid_air_df[feature_cols]
    groups = valid_air_df["category_id"].to_numpy()
    for tr_idx, te_idx in gkf.split(X_all, y_all, groups):
        fold_reg = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("rf", RandomForestRegressor(n_estimators=200, max_features="sqrt", random_state=42, n_jobs=2)),
        ])
        fold_reg.fit(X_all.iloc[tr_idx], y_all[tr_idx])
        oof_preds[te_idx] = fold_reg.predict(X_all.iloc[te_idx])
    category_heldout_metrics = eval_regression(y_all, oof_preds)

    # Fit final deployed regressor on train_air (keeping test_air strictly held out for reporting)
    joblib.dump(rf_reg, models_dir / "rf_airflow_regressor.joblib")

    report: dict[str, Any] = {
        "metadata_audit": {
            "froth_depth_available": has_froth_depth,
            "froth_depth_note": (
                "ImagesCatalog.xlsx does not contain measured froth depth or concentrate grade. "
                "No froth depth labels were fabricated; visual_stability_proxy (0-1) is provided as an unvalidated visual proxy, "
                "and plant measurements are required to train/validate froth depth or grade regression."
            ),
            "air_flow_target_column": "air_flow_l_min",
            "air_flow_valid_images": int(len(valid_air_df)),
            "air_flow_valid_categories": int(valid_air_df["category_id"].nunique()),
            "air_flow_excluded_unknown_images": int(len(excluded_air_df)),
            "air_flow_excluded_categories": sorted(excluded_air_df["category"].unique().tolist()),
            "air_flow_setpoints_l_min": sorted([float(x) for x in valid_air_df["air_flow_l_min"].unique()]),
        },
        "within_condition_test_metrics": {
            "num_train": int(len(train_air)),
            "num_test": int(len(test_air)),
            **within_metrics,
            "per_setpoint": per_setpoint,
        },
        "category_heldout_5fold_cv_metrics": category_heldout_metrics,
        "visual_stability_proxy_summary": {
            "stable_mean": round(float(df.loc[df["stability"] == "Stable", "visual_stability_proxy"].mean()), 4),
            "unstable_mean": round(float(df.loc[df["stability"] == "Unstable", "visual_stability_proxy"].mean()), 4),
            "note": "Unvalidated visual feature proxy computed from bubble uniformity, GLCM homogeneity, entropy, and focus.",
        },
    }

    (outputs_dir / "regression_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
