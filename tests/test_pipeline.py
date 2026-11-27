"""Automated verification tests for Project 7 Froth Image Analysis pipeline & FastAPI app."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch  # Import torch before numpy/pandas/sklearn to avoid Windows c10.dll load conflict
import pandas as pd
import pytest
from fastapi import HTTPException

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

from feature_utils import extract_image_features  # noqa: E402
from project3_adapter import Project3ImageFrothFeatures, Project3KineticModelAdapter, Project3OperatingInputs  # noqa: E402


def test_phase1_manifests_and_one_to_one_split() -> None:
    split_val = json.loads((PROJECT_ROOT / "manifests" / "split_validation.json").read_text(encoding="utf-8"))
    assert split_val["exact_one_to_one_match"] is True
    assert split_val["valid_images_expected"] == 2250
    assert split_val["num_batches"] == 5

    master_df = pd.read_csv(PROJECT_ROOT / "manifests" / "master_manifest.csv")
    assert len(master_df) == 2250
    assert master_df["image_id"].nunique() == 2250


def test_leakage_safe_ml_splits() -> None:
    splits_df = pd.read_csv(PROJECT_ROOT / "manifests" / "ml_splits.csv")
    assert len(splits_df) == 2250
    tr = set(splits_df.loc[splits_df["split"] == "train", "leakage_group_id"])
    va = set(splits_df.loc[splits_df["split"] == "val", "leakage_group_id"])
    te = set(splits_df.loc[splits_df["split"] == "test", "leakage_group_id"])
    assert not (tr & va)
    assert not (tr & te)
    assert not (va & te)


def test_feature_extractor_valid_and_corrupt() -> None:
    sample_img = PROJECT_ROOT / "data_original" / "images" / "1.jpg"
    feats = extract_image_features(sample_img)
    assert feats["orig_width"] == 1180
    assert feats["orig_height"] == 784
    assert feats["velocity_available"] is False
    assert 0.0 <= feats["visual_stability_proxy"] <= 1.0

    with pytest.raises(ValueError):
        extract_image_features(b"not-a-valid-image-bytes")


def test_project3_adapter_pending_mode() -> None:
    adapter = Project3KineticModelAdapter(model=None)
    payload = adapter.build_comparison_payload(
        operating=Project3OperatingInputs(air_flow_l_min=10.0, collector_dosage_gt=25.0),
        image_features=Project3ImageFrothFeatures(
            predicted_stability_label="Stable",
            predicted_unstable_prob=0.12,
            visual_stability_proxy=0.71,
            bubble_d50_px=38.5,
            bubble_size_span=0.92,
            texture_entropy=6.4,
            dark_fraction=0.18,
            novelty_score=0.22,
            drift_warning=False,
        ),
    )
    assert payload["integration_status"] == "pending_project3_model"
    assert payload["model_connected"] is False
    assert len(payload["missing_requirements"]) >= 2


def test_fastapi_backend_routes_and_inference() -> None:
    from app import (  # noqa: E402
        STATE,
        BatchAnalyzeRequest,
        LocalPredictRequest,
        _run_inference_on_source,
        app,
        batch_analyze,
        health,
        list_samples,
        predict_local,
        serve_index,
        summary,
    )

    STATE.load_all()
    assert STATE.loaded is True

    # Verify OpenAPI schema & frontend entry
    schema = app.openapi()
    for route in ["/api/health", "/api/summary", "/api/samples", "/api/predict", "/api/predict-local", "/api/batch-analyze"]:
        assert route in schema["paths"]
    assert Path(serve_index().path).exists()

    h = health()
    assert h["status"] == "ok" and h["num_categories"] == 45

    s = summary()
    assert s["project3_status"]["integration_status"] == "pending_project3_model"
    assert s["classifier_comparison"]["mobilenet_v2_45class"]["accuracy"] > 0.9

    samples = list_samples(batch_id="batch_001", category_id=None, stability="all", limit=5)
    assert len(samples["samples"]) == 5

    # Local dataset prediction (Stable Category 4)
    res_local = predict_local(LocalPredictRequest(image_id=151))
    assert res_local["predictions"]["random_forest_category"]
    assert res_local["predictions"]["cnn_category"]
    assert res_local["visual_features"]["velocity_available"] is False

    # Byte-buffer prediction (simulates upload) + corrupt image rejection
    img_bytes = (PROJECT_ROOT / "data_original" / "images" / "151.jpg").read_bytes()
    res_bytes = _run_inference_on_source(img_bytes)
    assert res_bytes["predictions"]["cnn_category"] == res_local["predictions"]["cnn_category"]

    with pytest.raises(HTTPException) as exc_info:
        _run_inference_on_source(b"corrupt-jpeg-payload")
    assert exc_info.value.status_code == 400

    # Incremental batch analysis writes CSV to disk
    batch_res = batch_analyze(BatchAnalyzeRequest(batch_id="batch_001", limit=5))
    assert batch_res["processed_count"] == 5
    out_file = PROJECT_ROOT / batch_res["output_csv_relative_path"]
    assert out_file.exists()
    out_file.unlink(missing_ok=True)
