"""Phase 3A: Train and compare feature-based Random Forest vs fine-tuned MobileNetV2 CNN."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models import MobileNet_V2_Weights, mobilenet_v2
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from cnn_utils import FrothImageDataset, get_cnn_transforms  # noqa: E402
from modeling_utils import build_leakage_safe_splits, select_numeric_feature_columns  # noqa: E402


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    category_names: list[str],
    out_metrics_csv: Path,
    out_cm_csv: Path,
) -> dict[str, Any]:
    acc = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

    labels_idx = list(range(len(category_names)))
    report_dict = classification_report(
        y_true,
        y_pred,
        labels=labels_idx,
        target_names=category_names,
        output_dict=True,
        zero_division=0,
    )
    per_cat_rows = []
    for i, cname in enumerate(category_names):
        stats = report_dict[cname]
        per_cat_rows.append({
            "category_id": i + 1,
            "category": cname,
            "precision": round(float(stats["precision"]), 4),
            "recall": round(float(stats["recall"]), 4),
            "f1_score": round(float(stats["f1-score"]), 4),
            "test_support": int(stats["support"]),
        })
    pd.DataFrame(per_cat_rows).to_csv(out_metrics_csv, index=False)

    cm = confusion_matrix(y_true, y_pred, labels=labels_idx)
    cm_df = pd.DataFrame(cm, index=category_names, columns=category_names)
    cm_df.to_csv(out_cm_csv)

    return {
        "accuracy": round(acc, 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "num_test_samples": int(len(y_true)),
    }


def build_mobilenet_v2(num_classes: int, freeze_until: int = 14) -> tuple[nn.Module, str]:
    cached_v1 = Path(torch.hub.get_dir()) / "checkpoints" / "mobilenet_v2-b0353104.pth"
    if cached_v1.exists():
        model = mobilenet_v2(weights=None)
        model.load_state_dict(torch.load(cached_v1, map_location="cpu", weights_only=True))
        weights_source = f"local_cached_imagenet1k_v1 ({cached_v1.name})"
    else:
        try:
            model = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
            weights_source = "torchvision_imagenet1k_v1"
        except Exception:
            model = mobilenet_v2(weights=None)
            weights_source = "random_init_offline"

    for idx, layer in enumerate(model.features):
        if idx < freeze_until:
            for p in layer.parameters():
                p.requires_grad = False
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.25),
        nn.Linear(in_features, num_classes),
    )
    return model, weights_source


@torch.no_grad()
def eval_cnn(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    preds_all: list[np.ndarray] = []
    probs_all: list[np.ndarray] = []
    labels_all: list[np.ndarray] = []
    for imgs, labels, _ in loader:
        imgs = imgs.to(device)
        logits = model(imgs)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        preds = np.argmax(probs, axis=1)
        preds_all.append(preds)
        probs_all.append(probs)
        labels_all.append(labels.numpy())
    return np.concatenate(labels_all), np.concatenate(preds_all), np.concatenate(probs_all)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train RF and MobileNetV2 classifiers on froth dataset.")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=3e-4)
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
    df["category_idx"] = df["category_id"].astype(int) - 1

    cat_map_df = df[["category_id", "category"]].drop_duplicates().sort_values("category_id")
    category_names = cat_map_df["category"].tolist()
    num_classes = len(category_names)

    feature_cols = select_numeric_feature_columns(df)
    train_df = df[df["split"] == "train"].reset_index(drop=True)
    val_df = df[df["split"] == "val"].reset_index(drop=True)
    test_df = df[df["split"] == "test"].reset_index(drop=True)

    # 1. Feature-based Random Forest (45-class operating condition)
    t0 = time.time()
    rf_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("rf", RandomForestClassifier(n_estimators=300, max_features="sqrt", random_state=42, n_jobs=2)),
    ])
    rf_pipe.fit(train_df[feature_cols], train_df["category_idx"])
    rf_test_pred = rf_pipe.predict(test_df[feature_cols])
    rf_metrics = evaluate_predictions(
        test_df["category_idx"].to_numpy(),
        rf_test_pred,
        category_names,
        outputs_dir / "rf_per_category_metrics.csv",
        outputs_dir / "rf_confusion_matrix.csv",
    )
    rf_metrics["train_time_sec"] = round(time.time() - t0, 2)

    joblib.dump(rf_pipe, models_dir / "rf_operating_classifier.joblib")

    # Feature importance
    rf_imp = pd.DataFrame({
        "feature": feature_cols,
        "importance": rf_pipe.named_steps["rf"].feature_importances_,
    }).sort_values("importance", ascending=False)
    rf_imp.to_csv(outputs_dir / "rf_feature_importance.csv", index=False)

    # 1b. Binary Stability Classifier (Stable vs Unstable) on same train/test split
    stab_train_y = (train_df["stability"] == "Unstable").astype(int)
    stab_test_y = (test_df["stability"] == "Unstable").astype(int)
    rf_stab = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("rf", RandomForestClassifier(n_estimators=250, class_weight="balanced", random_state=42, n_jobs=2)),
    ])
    rf_stab.fit(train_df[feature_cols], stab_train_y)
    stab_pred = rf_stab.predict(test_df[feature_cols])
    stab_metrics = {
        "accuracy": round(float(accuracy_score(stab_test_y, stab_pred)), 4),
        "macro_f1": round(float(f1_score(stab_test_y, stab_pred, average="macro", zero_division=0)), 4),
        "unstable_f1": round(float(f1_score(stab_test_y, stab_pred, pos_label=1, zero_division=0)), 4),
    }
    joblib.dump(rf_stab, models_dir / "rf_stability_classifier.joblib")

    # 2. Fine-tuned MobileNetV2 CNN (streaming from disk in batches of 16)
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_ds = FrothImageDataset(train_df, root, transform=get_cnn_transforms(train=True))
    val_ds = FrothImageDataset(val_df, root, transform=get_cnn_transforms(train=False))
    test_ds = FrothImageDataset(test_df, root, transform=get_cnn_transforms(train=False))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    cnn, weights_source = build_mobilenet_v2(num_classes=num_classes, freeze_until=14)
    cnn = cnn.to(device)
    print(f"Initialized MobileNetV2 with weights: {weights_source}")
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW([p for p in cnn.parameters() if p.requires_grad], lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))

    last_ckpt_path = models_dir / "cnn_checkpoint_last.pt"
    best_ckpt_path = models_dir / "cnn_best.pt"
    history: list[dict[str, Any]] = []
    start_epoch = 1
    best_val_f1 = -1.0

    if last_ckpt_path.exists():
        ckpt = torch.load(last_ckpt_path, map_location=device, weights_only=False)
        if ckpt.get("epochs_target") == args.epochs and ckpt.get("epoch", 0) < args.epochs:
            cnn.load_state_dict(ckpt["model_state"])
            optimizer.load_state_dict(ckpt["optimizer_state"])
            scheduler.load_state_dict(ckpt["scheduler_state"])
            start_epoch = int(ckpt["epoch"]) + 1
            best_val_f1 = float(ckpt.get("best_val_f1", -1.0))
            history = list(ckpt.get("history", []))
            print(f"Resuming CNN training from epoch {start_epoch}")

    t_cnn0 = time.time()
    for epoch in range(start_epoch, args.epochs + 1):
        cnn.train()
        running_loss = 0.0
        seen = 0
        for imgs, labels, _ in train_loader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            logits = cnn(imgs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item()) * imgs.size(0)
            seen += imgs.size(0)

        scheduler.step()
        train_loss = running_loss / max(1, seen)
        val_y, val_pred, _ = eval_cnn(cnn, val_loader, device)
        val_acc = float(accuracy_score(val_y, val_pred))
        val_f1 = float(f1_score(val_y, val_pred, average="macro", zero_division=0))

        epoch_rec = {
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "val_accuracy": round(val_acc, 4),
            "val_macro_f1": round(val_f1, 4),
        }
        history.append(epoch_rec)
        print(f"Epoch {epoch}/{args.epochs}: loss={train_loss:.4f}, val_acc={val_acc:.4f}, val_macro_f1={val_f1:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save({
                "model_state": cnn.state_dict(),
                "category_names": category_names,
                "epoch": epoch,
                "val_accuracy": val_acc,
                "val_macro_f1": val_f1,
            }, best_ckpt_path)

        torch.save({
            "epoch": epoch,
            "epochs_target": args.epochs,
            "model_state": cnn.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "best_val_f1": best_val_f1,
            "history": history,
        }, last_ckpt_path)

    cnn_train_time = round(time.time() - t_cnn0, 2)

    # Evaluate best CNN checkpoint on held-out test split
    best_ckpt = torch.load(best_ckpt_path, map_location=device, weights_only=False)
    cnn.load_state_dict(best_ckpt["model_state"])
    test_y, cnn_test_pred, _ = eval_cnn(cnn, test_loader, device)
    cnn_metrics = evaluate_predictions(
        test_y,
        cnn_test_pred,
        category_names,
        outputs_dir / "cnn_per_category_metrics.csv",
        outputs_dir / "cnn_confusion_matrix.csv",
    )
    cnn_metrics["train_time_sec"] = cnn_train_time
    cnn_metrics["best_epoch"] = int(best_ckpt["epoch"])
    cnn_metrics["history"] = history

    # Save metadata for backend inference
    meta_out = {
        "feature_columns": feature_cols,
        "category_names": category_names,
        "num_classes": num_classes,
        "split_counts": df["split"].value_counts().to_dict(),
        "batch_size": args.batch_size,
        "cnn_input_size": [224, 224],
        "device_trained": str(device),
    }
    (models_dir / "model_metadata.json").write_text(json.dumps(meta_out, indent=2), encoding="utf-8")

    comparison = {
        "split_counts": meta_out["split_counts"],
        "leakage_prevention": "StratifiedGroupKFold grouped by near-duplicate dHash clusters (leakage_group_id)",
        "random_forest_45class": rf_metrics,
        "mobilenet_v2_45class": cnn_metrics,
        "random_forest_stability_within_split": stab_metrics,
        "top_10_rf_features": rf_imp.head(10).to_dict(orient="records"),
    }
    (outputs_dir / "classifier_comparison.json").write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    print(json.dumps(comparison, indent=2))


if __name__ == "__main__":
    main()
