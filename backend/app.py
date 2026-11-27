"""Phase 5: FastAPI backend for Froth Image Analysis & Flotation Performance Monitoring."""

from __future__ import annotations

import csv
import json
import sys
import time
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
BACKEND_DIR = PROJECT_ROOT / "backend"
for p in (str(SCRIPTS_DIR), str(BACKEND_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from cnn_utils import get_cnn_transforms  # noqa: E402
from feature_utils import decode_image_bgr, extract_image_features  # noqa: E402
from project3_adapter import (  # noqa: E402
    Project3ImageFrothFeatures,
    Project3KineticModelAdapter,
    Project3OperatingInputs,
)


class LocalPredictRequest(BaseModel):
    image_id: int | None = None
    relative_path: str | None = None
    air_flow_l_min: float | None = None
    collector_dosage_gt: float | None = None
    frother_condition_gt: float | None = None
    kind_of_water: str | None = None
    mineral_used: str | None = None


class BatchAnalyzeRequest(BaseModel):
    batch_id: str = Field(default="batch_001", pattern=r"^batch_\d{3}$")
    limit: int = Field(default=50, ge=1, le=500)


def _build_cnn_architecture(num_classes: int) -> nn.Module:
    model = mobilenet_v2(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.25),
        nn.Linear(in_features, num_classes),
    )
    return model


class ModelState:
    def __init__(self) -> None:
        self.loaded: bool = False
        self.load_errors: list[str] = []
        self.rf_op: Any = None
        self.rf_stab: Any = None
        self.rf_air: Any = None
        self.ood_bundle: dict[str, Any] | None = None
        self.cnn: nn.Module | None = None
        self.cnn_transform = get_cnn_transforms(train=False)
        self.metadata: dict[str, Any] = {}
        self.master_df: pd.DataFrame | None = None
        self.category_summary_df: pd.DataFrame | None = None
        self.p3_adapter = Project3KineticModelAdapter(model=None)

    def load_all(self) -> None:
        models_dir = PROJECT_ROOT / "models"
        manifests_dir = PROJECT_ROOT / "manifests"

        try:
            self.metadata = json.loads((models_dir / "model_metadata.json").read_text(encoding="utf-8"))
            self.rf_op = joblib.load(models_dir / "rf_operating_classifier.joblib")
            self.rf_stab = joblib.load(models_dir / "rf_stability_classifier.joblib")
            self.rf_air = joblib.load(models_dir / "rf_airflow_regressor.joblib")
            for pipe in (self.rf_op, self.rf_stab, self.rf_air):
                if hasattr(pipe, "named_steps") and "rf" in pipe.named_steps:
                    pipe.named_steps["rf"].n_jobs = 1
            self.ood_bundle = joblib.load(models_dir / "ood_detector.joblib")

            ckpt = torch.load(models_dir / "cnn_best.pt", map_location="cpu", weights_only=False)
            num_classes = int(self.metadata["num_classes"])
            cnn = _build_cnn_architecture(num_classes)
            cnn.load_state_dict(ckpt["model_state"])
            cnn.eval()
            self.cnn = cnn
        except Exception as exc:
            self.load_errors.append(f"Model loading error: {exc}")

        try:
            self.master_df = pd.read_csv(manifests_dir / "master_manifest.csv")
            self.category_summary_df = pd.read_csv(manifests_dir / "category_summary.csv")
        except Exception as exc:
            self.load_errors.append(f"Manifest loading error: {exc}")

        self.loaded = len(self.load_errors) == 0


STATE = ModelState()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    STATE.load_all()
    yield


app = FastAPI(
    title="Project 7: Froth Image Analysis API",
    version="1.0.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _run_inference_on_source(
    source: Path | bytes,
    catalog_row: dict[str, Any] | None = None,
    operating_override: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if not STATE.loaded or STATE.rf_op is None or STATE.cnn is None or STATE.ood_bundle is None:
        raise HTTPException(status_code=503, detail=f"Models not ready: {STATE.load_errors}")

    # 1. Extract features + validate image
    try:
        feats = extract_image_features(source, max_work_side=512)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Corrupt or unsupported image input: {exc}") from exc

    feature_cols: list[str] = STATE.metadata["feature_columns"]
    category_names: list[str] = STATE.metadata["category_names"]
    row_df = pd.DataFrame([{c: feats.get(c, np.nan) for c in feature_cols}])

    # 2. Random Forest operating condition prediction
    rf_probs = STATE.rf_op.predict_proba(row_df)[0]
    rf_top_idx = int(np.argmax(rf_probs))
    rf_top3_idx = np.argsort(rf_probs)[::-1][:3]
    rf_top3 = [
        {"category": category_names[int(i)], "probability": round(float(rf_probs[int(i)]), 4)}
        for i in rf_top3_idx
    ]

    # 3. MobileNetV2 CNN operating condition prediction (224x224 streaming)
    bgr, _, _ = decode_image_bgr(source)
    rgb = bgr[:, :, ::-1]
    pil_img = Image.fromarray(rgb)
    tensor = STATE.cnn_transform(pil_img).unsqueeze(0)
    with torch.no_grad():
        logits = STATE.cnn(tensor)
        cnn_probs = torch.softmax(logits, dim=1).squeeze(0).numpy()
    cnn_top_idx = int(np.argmax(cnn_probs))
    cnn_top3_idx = np.argsort(cnn_probs)[::-1][:3]
    cnn_top3 = [
        {"category": category_names[int(i)], "probability": round(float(cnn_probs[int(i)]), 4)}
        for i in cnn_top3_idx
    ]

    # 4. Stability classification & score
    stab_probs = STATE.rf_stab.predict_proba(row_df)[0]
    p_unstable = float(stab_probs[1]) if len(stab_probs) > 1 else float(stab_probs[0])
    stability_score = round(1.0 - p_unstable, 4)
    predicted_stability = "Unstable" if p_unstable >= 0.5 else "Stable"

    # 5. Air flow continuous regression (only trained on valid numeric air flow)
    pred_air_flow = round(float(STATE.rf_air.predict(row_df)[0]), 2)

    # 6. Uncertainty & OOD / Drift detection
    prep = STATE.ood_bundle["prep"]
    lw = STATE.ood_bundle["ledoit_wolf"]
    p95 = float(STATE.ood_bundle["mahal_p95_in_dist"])
    p99 = float(STATE.ood_bundle["mahal_p99_in_dist"])

    X_z = prep.transform(row_df)
    mahal_dist = float(lw.mahalanobis(X_z)[0])
    rf_uncertainty = round(1.0 - float(np.max(rf_probs)), 4)
    cnn_uncertainty = round(1.0 - float(np.max(cnn_probs)), 4)
    norm_mahal = min(2.0, mahal_dist / max(p95, 1e-6)) / 2.0
    novelty_score = round(0.5 * rf_uncertainty + 0.5 * norm_mahal, 4)

    outside_training_dist = mahal_dist > p99
    seg_failed = feats["segmentation_status"] != "ok"
    drift_reasons: list[str] = []
    if outside_training_dist:
        drift_reasons.append(f"Feature Mahalanobis distance ({mahal_dist:.1f}) exceeds 99th percentile ({p99:.1f}) of training distribution.")
    elif mahal_dist > p95:
        drift_reasons.append(f"Elevated feature distance ({mahal_dist:.1f} > p95 {p95:.1f}); borderline regime shift.")
    if p_unstable >= 0.5:
        drift_reasons.append(f"Predicted froth instability ({p_unstable * 100:.1f}% unstable probability) may precede grade drift.")
    if feats["visual_stability_proxy"] < 0.45:
        drift_reasons.append(f"Low visual stability proxy ({feats['visual_stability_proxy']:.2f}).")
    if cnn_top_idx != rf_top_idx and max(rf_probs) < 0.45 and max(cnn_probs) < 0.45:
        drift_reasons.append("High classifier uncertainty and RF/CNN disagreement.")
    if seg_failed:
        drift_reasons.append(f"Bubble segmentation status is '{feats['segmentation_status']}' (confidence={feats['segmentation_confidence']}).")

    drift_warning = outside_training_dist or (p_unstable >= 0.5) or (novelty_score >= 0.55)

    # 7. Check operating metadata completeness & build Project 3 adapter payload
    op_meta = dict(operating_override or {})
    if catalog_row is not None:
        op_meta.setdefault("air_flow_l_min", None if pd.isna(catalog_row.get("air_flow_l_min")) else float(catalog_row["air_flow_l_min"]))
        op_meta.setdefault("collector_dosage_gt", None if pd.isna(catalog_row.get("collector_dosage_gt")) else float(catalog_row["collector_dosage_gt"]))
        op_meta.setdefault("frother_condition_gt", None if pd.isna(catalog_row.get("frother_condition_gt")) else float(catalog_row["frother_condition_gt"]))
        op_meta.setdefault("kind_of_water", catalog_row.get("kind_of_water"))
        op_meta.setdefault("mineral_used", catalog_row.get("mineral_used"))

    metadata_warnings: list[str] = []
    if op_meta.get("air_flow_l_min") is None:
        metadata_warnings.append("Measured air flow missing/unknown; using model-estimated air flow in Project 3 payload preview.")
    if op_meta.get("collector_dosage_gt") is None or op_meta.get("frother_condition_gt") is None:
        metadata_warnings.append("Collector or frother dosage metadata missing for this input.")

    p3_payload = STATE.p3_adapter.build_comparison_payload(
        operating=Project3OperatingInputs(
            air_flow_l_min=op_meta.get("air_flow_l_min") if op_meta.get("air_flow_l_min") is not None else pred_air_flow,
            collector_dosage_gt=op_meta.get("collector_dosage_gt"),
            frother_condition_gt=op_meta.get("frother_condition_gt"),
            kind_of_water=op_meta.get("kind_of_water"),
            mineral_used=op_meta.get("mineral_used"),
        ),
        image_features=Project3ImageFrothFeatures(
            predicted_stability_label=predicted_stability,
            predicted_unstable_prob=round(p_unstable, 4),
            visual_stability_proxy=feats["visual_stability_proxy"],
            bubble_d50_px=feats.get("bubble_d50"),
            bubble_size_span=feats.get("bubble_size_span"),
            texture_entropy=feats["texture_entropy"],
            dark_fraction=feats["dark_fraction"],
            novelty_score=novelty_score,
            drift_warning=drift_warning,
        ),
    )

    return {
        "predictions": {
            "random_forest_category": category_names[rf_top_idx],
            "random_forest_confidence": round(float(rf_probs[rf_top_idx]), 4),
            "random_forest_top3": rf_top3,
            "cnn_category": category_names[cnn_top_idx],
            "cnn_confidence": round(float(cnn_probs[cnn_top_idx]), 4),
            "cnn_top3": cnn_top3,
            "predicted_stability": predicted_stability,
            "stability_score": stability_score,
            "unstable_probability": round(p_unstable, 4),
            "visual_stability_proxy": feats["visual_stability_proxy"],
            "estimated_air_flow_l_min": pred_air_flow,
            "froth_depth_note": "Froth depth not measured in source dataset; visual_stability_proxy reported instead.",
        },
        "uncertainty_and_drift": {
            "rf_uncertainty": rf_uncertainty,
            "cnn_uncertainty": cnn_uncertainty,
            "mahalanobis_distance": round(mahal_dist, 3),
            "mahalanobis_p95_threshold": round(p95, 3),
            "mahalanobis_p99_threshold": round(p99, 3),
            "novelty_score": novelty_score,
            "outside_training_distribution": outside_training_dist,
            "drift_warning": drift_warning,
            "reasons": drift_reasons,
            "metadata_warnings": metadata_warnings,
        },
        "visual_features": feats,
        "catalog_ground_truth": catalog_row,
        "project3_integration": p3_payload,
    }


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok" if STATE.loaded else "degraded",
        "loaded": STATE.loaded,
        "errors": STATE.load_errors,
        "device": "cpu",
        "num_categories": STATE.metadata.get("num_classes", 0),
    }


@app.get("/api/summary")
def summary() -> dict[str, Any]:
    def read_json(rel: str) -> Any:
        p = PROJECT_ROOT / rel
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None

    unsteady_csv = PROJECT_ROOT / "outputs" / "drift_per_unsteady_category.csv"
    unsteady_rows = pd.read_csv(unsteady_csv).to_dict(orient="records") if unsteady_csv.exists() else []

    return {
        "inspection": read_json("manifests/inspection_report.json"),
        "split_validation": read_json("manifests/split_validation.json"),
        "feature_summary": read_json("processed/feature_summary.json"),
        "classifier_comparison": read_json("outputs/classifier_comparison.json"),
        "regression_report": read_json("outputs/regression_report.json"),
        "drift_evaluation": read_json("outputs/drift_evaluation.json"),
        "unsteady_categories_detail": unsteady_rows,
        "project3_status": {
            "connected": STATE.p3_adapter.is_connected,
            "integration_status": "pending_project3_model",
            "missing_requirements": STATE.p3_adapter.MISSING_REQUIREMENTS,
        },
    }


@app.get("/api/samples")
def list_samples(
    batch_id: str = Query(default="all"),
    category_id: int | None = Query(default=None),
    stability: str = Query(default="all"),
    limit: int = Query(default=60, ge=1, le=200),
) -> dict[str, Any]:
    if STATE.master_df is None:
        raise HTTPException(status_code=503, detail="Master manifest not loaded.")

    df = STATE.master_df
    if batch_id != "all":
        df = df[df["batch_id"] == batch_id]
    if category_id is not None:
        df = df[df["category_id"] == category_id]
    if stability != "all":
        df = df[df["stability"] == stability]

    subset = df.head(limit).copy()
    records = []
    for row in subset.to_dict(orient="records"):
        clean_row = {k: (None if pd.isna(v) else v) for k, v in row.items()}
        clean_row["thumbnail_url"] = f"/thumbnails/{int(row['image_id'])}.jpg"
        clean_row["image_url"] = f"/batch-files/{row['relative_path'].replace('data_batches/', '')}"
        records.append(clean_row)
    return {"total_matching": int(len(df)), "returned": len(records), "samples": records}


@app.post("/api/predict")
async def predict_upload(file: UploadFile = File(...)) -> dict[str, Any]:
    allowed_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    ext = Path(file.filename or "").suffix.lower()
    if ext not in allowed_exts:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file extension '{ext or 'none'}'. Supported formats: {sorted(allowed_exts)}",
        )
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    if len(content) > 25 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="Uploaded file exceeds 25 MB limit.")

    res = _run_inference_on_source(content, catalog_row=None)
    res["input_info"] = {"source": "upload", "filename": file.filename}
    return res


@app.post("/api/predict-local")
def predict_local(req: LocalPredictRequest) -> dict[str, Any]:
    if STATE.master_df is None:
        raise HTTPException(status_code=503, detail="Manifest not loaded.")

    row_dict: dict[str, Any] | None = None
    target_path: Path | None = None

    if req.image_id is not None:
        matches = STATE.master_df[STATE.master_df["image_id"] == req.image_id]
        if matches.empty:
            raise HTTPException(status_code=404, detail=f"Image ID {req.image_id} not found in manifest.")
        row_dict = {k: (None if pd.isna(v) else v) for k, v in matches.iloc[0].to_dict().items()}
        target_path = PROJECT_ROOT / str(row_dict["relative_path"])
    elif req.relative_path:
        candidate = (PROJECT_ROOT / req.relative_path).resolve()
        if not candidate.is_relative_to(PROJECT_ROOT.resolve()) or not candidate.exists():
            raise HTTPException(status_code=404, detail=f"Relative path not found: {req.relative_path}")
        target_path = candidate
        matches = STATE.master_df[STATE.master_df["relative_path"] == req.relative_path]
        if not matches.empty:
            row_dict = {k: (None if pd.isna(v) else v) for k, v in matches.iloc[0].to_dict().items()}
    else:
        raise HTTPException(status_code=400, detail="Provide either image_id or relative_path.")

    override = {
        k: v
        for k, v in {
            "air_flow_l_min": req.air_flow_l_min,
            "collector_dosage_gt": req.collector_dosage_gt,
            "frother_condition_gt": req.frother_condition_gt,
            "kind_of_water": req.kind_of_water,
            "mineral_used": req.mineral_used,
        }.items()
        if v is not None
    }
    res = _run_inference_on_source(target_path, catalog_row=row_dict, operating_override=override)
    res["input_info"] = {
        "source": "local_dataset",
        "image_id": row_dict["image_id"] if row_dict else None,
        "relative_path": str(target_path.relative_to(PROJECT_ROOT).as_posix()),
        "image_url": f"/batch-files/{target_path.relative_to(PROJECT_ROOT / 'data_batches').as_posix()}"
        if target_path.is_relative_to(PROJECT_ROOT / "data_batches")
        else None,
    }
    return res


@app.post("/api/batch-analyze")
def batch_analyze(req: BatchAnalyzeRequest) -> dict[str, Any]:
    """Process images from a physical batch incrementally one by one and write results to disk."""
    if not STATE.loaded:
        raise HTTPException(status_code=503, detail="Models not loaded.")

    batch_manifest = PROJECT_ROOT / "manifests" / f"{req.batch_id}.csv"
    if not batch_manifest.exists():
        raise HTTPException(status_code=404, detail=f"Batch manifest not found: {req.batch_id}")

    bdf = pd.read_csv(batch_manifest).head(req.limit)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_csv = PROJECT_ROOT / "outputs" / f"batch_analysis_{req.batch_id}_{ts}.csv"

    t0 = time.time()
    fieldnames = [
        "image_id",
        "relative_path",
        "true_category",
        "true_stability",
        "rf_category",
        "rf_confidence",
        "cnn_category",
        "cnn_confidence",
        "predicted_stability",
        "stability_score",
        "visual_stability_proxy",
        "estimated_air_flow_l_min",
        "bubble_count",
        "bubble_d50",
        "texture_entropy",
        "novelty_score",
        "drift_warning",
        "segmentation_status",
    ]
    preview_rows: list[dict[str, Any]] = []
    drift_count = 0
    unstable_count = 0
    rf_correct = 0
    cnn_correct = 0

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in bdf.to_dict(orient="records"):
            img_path = PROJECT_ROOT / row["relative_path"]
            clean_gt = {k: (None if pd.isna(v) else v) for k, v in row.items()}
            res = _run_inference_on_source(img_path, catalog_row=clean_gt)
            pred = res["predictions"]
            unc = res["uncertainty_and_drift"]
            vf = res["visual_features"]

            out_row = {
                "image_id": int(row["image_id"]),
                "relative_path": row["relative_path"],
                "true_category": row["category"],
                "true_stability": row["stability"],
                "rf_category": pred["random_forest_category"],
                "rf_confidence": pred["random_forest_confidence"],
                "cnn_category": pred["cnn_category"],
                "cnn_confidence": pred["cnn_confidence"],
                "predicted_stability": pred["predicted_stability"],
                "stability_score": pred["stability_score"],
                "visual_stability_proxy": pred["visual_stability_proxy"],
                "estimated_air_flow_l_min": pred["estimated_air_flow_l_min"],
                "bubble_count": vf["bubble_count"],
                "bubble_d50": vf["bubble_d50"],
                "texture_entropy": vf["texture_entropy"],
                "novelty_score": unc["novelty_score"],
                "drift_warning": unc["drift_warning"],
                "segmentation_status": vf["segmentation_status"],
            }
            writer.writerow(out_row)
            if len(preview_rows) < 30:
                preview_rows.append(out_row)
            if unc["drift_warning"]:
                drift_count += 1
            if pred["predicted_stability"] == "Unstable":
                unstable_count += 1
            if pred["random_forest_category"] == row["category"]:
                rf_correct += 1
            if pred["cnn_category"] == row["category"]:
                cnn_correct += 1

    total = len(bdf)
    return {
        "batch_id": req.batch_id,
        "processed_count": total,
        "elapsed_sec": round(time.time() - t0, 2),
        "output_csv_relative_path": out_csv.relative_to(PROJECT_ROOT).as_posix(),
        "rf_accuracy_on_run": round(rf_correct / max(1, total), 4),
        "cnn_accuracy_on_run": round(cnn_correct / max(1, total), 4),
        "predicted_unstable_count": unstable_count,
        "drift_warning_count": drift_count,
        "preview_rows": preview_rows,
    }


@app.get("/")
def serve_index() -> FileResponse:
    return FileResponse(PROJECT_ROOT / "frontend" / "index.html")


app.mount("/static", StaticFiles(directory=str(PROJECT_ROOT / "frontend")), name="static")
if (PROJECT_ROOT / "processed" / "thumbnails").exists():
    app.mount("/thumbnails", StaticFiles(directory=str(PROJECT_ROOT / "processed" / "thumbnails")), name="thumbnails")
if (PROJECT_ROOT / "data_batches").exists():
    app.mount("/batch-files", StaticFiles(directory=str(PROJECT_ROOT / "data_batches")), name="batch-files")
