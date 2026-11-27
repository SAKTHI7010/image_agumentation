"""Phase 3C: Evaluate category generalization and held-out unsteady-category drift/novelty detection."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from modeling_utils import build_leakage_safe_splits, select_numeric_feature_columns  # noqa: E402


def compute_ood_metrics(is_ood: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    auroc = float(roc_auc_score(is_ood, scores))
    auprc = float(average_precision_score(is_ood, scores))
    in_scores = scores[is_ood == 0]
    ood_scores = scores[is_ood == 1]
    thresh_95 = float(np.quantile(in_scores, 0.95))
    tpr_at_5fpr = float(np.mean(ood_scores >= thresh_95))
    return {
        "auroc": round(auroc, 4),
        "auprc": round(auprc, 4),
        "tpr_at_5pct_fpr": round(tpr_at_5fpr, 4),
        "threshold_95pct_in_dist": round(thresh_95, 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate closed-set stable classification & held-out unsteady drift.")
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

    # Identify unsteady categories strictly from dataset metadata (Visual Characteristics -> Stability == 'Unstable')
    stable_df = df[df["stability"] == "Stable"].copy()
    unstable_df = df[df["stability"] == "Unstable"].copy()

    stable_categories = sorted(stable_df["category_id"].unique().tolist())
    unstable_categories = sorted(unstable_df["category_id"].unique().tolist())

    stable_train = stable_df[stable_df["split"].isin(["train", "val"])].reset_index(drop=True)
    stable_test = stable_df[stable_df["split"] == "test"].reset_index(drop=True)

    # Map stable categories to 0..N_stable-1
    stable_cat_to_idx = {cid: i for i, cid in enumerate(stable_categories)}
    y_st_train = stable_train["category_id"].map(stable_cat_to_idx).to_numpy()
    y_st_test = stable_test["category_id"].map(stable_cat_to_idx).to_numpy()

    # Experiment 1: Closed-set classifier trained ONLY on Stable categories
    stable_clf = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("rf", RandomForestClassifier(n_estimators=300, max_features="sqrt", random_state=42, n_jobs=2)),
    ])
    stable_clf.fit(stable_train[feature_cols], y_st_train)
    st_test_pred = stable_clf.predict(stable_test[feature_cols])
    closed_set_acc = float(accuracy_score(y_st_test, st_test_pred))
    closed_set_f1 = float(f1_score(y_st_test, st_test_pred, average="macro", zero_division=0))

    # Experiment 2: Held-out Unstable categories novelty & drift detection
    in_probs = stable_clf.predict_proba(stable_test[feature_cols])
    ood_probs = stable_clf.predict_proba(unstable_df[feature_cols])

    in_uncertainty = 1.0 - np.max(in_probs, axis=1)
    ood_uncertainty = 1.0 - np.max(ood_probs, axis=1)

    eps = 1e-12
    in_entropy = -np.sum(in_probs * np.log(np.clip(in_probs, eps, 1.0)), axis=1) / np.log(len(stable_categories))
    ood_entropy = -np.sum(ood_probs * np.log(np.clip(ood_probs, eps, 1.0)), axis=1) / np.log(len(stable_categories))

    # Feature-space Mahalanobis distance + IsolationForest trained ONLY on Stable training data
    prep = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    X_st_train_z = prep.fit_transform(stable_train[feature_cols])
    X_st_test_z = prep.transform(stable_test[feature_cols])
    X_ood_z = prep.transform(unstable_df[feature_cols])

    lw = LedoitWolf().fit(X_st_train_z)
    in_mahal = lw.mahalanobis(X_st_test_z)
    ood_mahal = lw.mahalanobis(X_ood_z)

    iso = IsolationForest(n_estimators=200, contamination=0.05, random_state=42, n_jobs=2)
    iso.fit(X_st_train_z)
    in_iso = -iso.decision_function(X_st_test_z)
    ood_iso = -iso.decision_function(X_ood_z)

    # Combined normalized novelty score (used also by full-dataset OOD detector in app)
    mahal_scale = float(np.quantile(lw.mahalanobis(X_st_train_z), 0.95))
    in_combined = 0.5 * in_uncertainty + 0.5 * np.clip(in_mahal / max(mahal_scale, 1e-6), 0.0, 2.0) / 2.0
    ood_combined = 0.5 * ood_uncertainty + 0.5 * np.clip(ood_mahal / max(mahal_scale, 1e-6), 0.0, 2.0) / 2.0

    is_ood_labels = np.concatenate([np.zeros(len(stable_test), dtype=int), np.ones(len(unstable_df), dtype=int)])
    uncertainty_metrics = compute_ood_metrics(is_ood_labels, np.concatenate([in_uncertainty, ood_uncertainty]))
    entropy_metrics = compute_ood_metrics(is_ood_labels, np.concatenate([in_entropy, ood_entropy]))
    mahal_metrics = compute_ood_metrics(is_ood_labels, np.concatenate([in_mahal, ood_mahal]))
    iso_metrics = compute_ood_metrics(is_ood_labels, np.concatenate([in_iso, ood_iso]))
    combined_metrics = compute_ood_metrics(is_ood_labels, np.concatenate([in_combined, ood_combined]))

    # Per held-out unsteady category breakdown
    unstable_df["pred_uncertainty"] = ood_uncertainty
    unstable_df["pred_entropy_norm"] = ood_entropy
    unstable_df["mahalanobis_dist"] = ood_mahal
    unstable_df["combined_novelty_score"] = ood_combined
    thresh = combined_metrics["threshold_95pct_in_dist"]
    unstable_df["flagged_drift"] = unstable_df["combined_novelty_score"] >= thresh

    per_unsteady = (
        unstable_df.groupby(["category_id", "category", "mineral_used", "air_flow_raw", "quality"])
        .agg(
            sample_count=("image_id", "count"),
            mean_uncertainty=("pred_uncertainty", "mean"),
            mean_mahalanobis=("mahalanobis_dist", "mean"),
            mean_novelty_score=("combined_novelty_score", "mean"),
            drift_flag_rate=("flagged_drift", "mean"),
            mean_visual_stability_proxy=("visual_stability_proxy", "mean"),
        )
        .reset_index()
        .round(4)
        .sort_values("category_id")
    )
    per_unsteady.to_csv(outputs_dir / "drift_per_unsteady_category.csv", index=False)

    # Experiment 3: Cross-category Stability generalization (GroupKFold by category_id)
    gkf = GroupKFold(n_splits=5)
    y_stab = (df["stability"] == "Unstable").astype(int).to_numpy()
    X_all = df[feature_cols]
    groups = df["category_id"].to_numpy()
    oof_stab_pred = np.zeros(len(df), dtype=int)
    for tr_idx, te_idx in gkf.split(X_all, y_stab, groups):
        clf = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("rf", RandomForestClassifier(n_estimators=250, class_weight="balanced", random_state=42, n_jobs=2)),
        ])
        clf.fit(X_all.iloc[tr_idx], y_stab[tr_idx])
        oof_stab_pred[te_idx] = clf.predict(X_all.iloc[te_idx])

    cross_cat_stab = {
        "accuracy": round(float(accuracy_score(y_stab, oof_stab_pred)), 4),
        "macro_f1": round(float(f1_score(y_stab, oof_stab_pred, average="macro", zero_division=0)), 4),
        "unstable_f1": round(float(f1_score(y_stab, oof_stab_pred, pos_label=1, zero_division=0)), 4),
    }

    # Fit full-training-set OOD detector (on all 45 categories' train+val split) for live API inference
    full_train_df = df[df["split"].isin(["train", "val"])].reset_index(drop=True)
    full_prep = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    X_full_z = full_prep.fit_transform(full_train_df[feature_cols])
    full_lw = LedoitWolf().fit(X_full_z)
    full_train_mahal = full_lw.mahalanobis(X_full_z)
    ood_bundle = {
        "prep": full_prep,
        "ledoit_wolf": full_lw,
        "mahal_p95_in_dist": float(np.quantile(full_train_mahal, 0.95)),
        "mahal_p99_in_dist": float(np.quantile(full_train_mahal, 0.99)),
        "feature_cols": feature_cols,
    }
    joblib.dump(ood_bundle, models_dir / "ood_detector.joblib")

    report: dict[str, Any] = {
        "metadata_identification": {
            "stable_category_count": len(stable_categories),
            "stable_category_ids": stable_categories,
            "heldout_unstable_category_count": len(unstable_categories),
            "heldout_unstable_category_ids": unstable_categories,
            "stable_train_val_images": int(len(stable_train)),
            "stable_test_images": int(len(stable_test)),
            "heldout_unstable_images": int(len(unstable_df)),
        },
        "experiment_1_closed_set_stable": {
            "accuracy": round(closed_set_acc, 4),
            "macro_f1": round(closed_set_f1, 4),
            "note": "Closed-set classifier trained only on Stable categories cannot name unseen Unstable categories.",
        },
        "experiment_2_heldout_unsteady_novelty_detection": {
            "predictive_uncertainty_1_minus_max_prob": uncertainty_metrics,
            "normalized_predictive_entropy": entropy_metrics,
            "feature_mahalanobis_distance": mahal_metrics,
            "isolation_forest_score": iso_metrics,
            "combined_novelty_score": combined_metrics,
            "in_dist_mean_novelty": round(float(np.mean(in_combined)), 4),
            "heldout_unsteady_mean_novelty": round(float(np.mean(ood_combined)), 4),
        },
        "experiment_3_cross_category_stability_5fold_groupcv": cross_cat_stab,
    }

    (outputs_dir / "drift_evaluation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
