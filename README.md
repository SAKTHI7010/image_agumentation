---
title: Project 7 Froth Image Analysis
emoji: 🫧
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
---

# Project 7 — Froth Image Analysis for Flotation Performance Monitoring

Memory-efficient prototype for flotation froth image monitoring, operating-condition classification, froth stability estimation, held-out unsteady-category drift/novelty detection, and Project 3 flotation kinetic model integration.

## 1. Folder Structure & Dataset Paths

- **Supplied Source Archive (Untouched)**: `../21437292/froth_images.zip` and `../21437292/ImagesCatalog.xlsx`
- **Read-Only Extracted Originals**: `data_original/images/` (`2,250` `.jpg` files at `1180x784`) and `data_original/ImagesCatalog.xlsx`
- **Physical Processing Batches**: `data_batches/batch_001/` ... `batch_005/` (5 stratified batches of `450` images each)
- **Manifests**:
  - `manifests/dataset_inventory.csv` & `manifests/inspection_report.json`
  - `manifests/master_manifest.csv` & `manifests/batch_001.csv` ... `manifests/batch_005.csv`
  - `manifests/split_validation.json` (verifies 1-to-1 coverage of all 2,250 valid images)
  - `manifests/ml_splits.csv` (leakage-safe train/val/test split grouped by near-duplicate `dHash` clusters)
- **Extracted Features & Thumbnails**: `processed/features/features.csv`, `processed/features/features.parquet`, `processed/preprocessing_report.csv`, `processed/thumbnails/`
- **Trained Models**: `models/` (`rf_operating_classifier.joblib`, `rf_stability_classifier.joblib`, `rf_airflow_regressor.joblib`, `ood_detector.joblib`, `cnn_best.pt`, `model_metadata.json`)
- **Evaluation Outputs**: `outputs/` (`classifier_comparison.json`, `regression_report.json`, `drift_evaluation.json`, `drift_per_unsteady_category.csv`, per-category metrics & confusion matrices)
- **Application**: `backend/app.py`, `backend/project3_adapter.py`, `frontend/` (`index.html`, `styles.css`, `app.js`)

## 2. Dependencies & Environment

Python 3.10+ with packages listed in `requirements.txt`:

```powershell
pip install -r Project_7_Froth_Analysis/requirements.txt
```

> **Windows Note**: In environments where both NumPy/scikit-learn and PyTorch CPU DLLs are loaded in the same process, scripts import `torch` prior to `numpy`/`scikit-learn` to prevent Windows `c10.dll` OpenMP initialization conflicts.

## 3. Pipeline Execution Commands (Exact Order)

```powershell
# Phase 1 — Inspect originals & split into physical batches
python Project_7_Froth_Analysis/scripts/inspect_dataset.py
python Project_7_Froth_Analysis/scripts/split_dataset.py

# Phase 2 — Incremental, resumable feature extraction across batches
python Project_7_Froth_Analysis/scripts/extract_features.py

# Phase 3 — Modeling, regression audit & held-out unsteady drift evaluation
python Project_7_Froth_Analysis/scripts/train_regressor.py
python Project_7_Froth_Analysis/scripts/evaluate_drift.py
python Project_7_Froth_Analysis/scripts/train_classifier.py --epochs 6 --batch-size 16 --num-workers 0

# Tests
pytest Project_7_Froth_Analysis/tests/test_pipeline.py -v

# Phase 5 — Run FastAPI backend + HTML/CSS/JS dashboard locally
uvicorn backend.app:app --app-dir Project_7_Froth_Analysis --host 127.0.0.1 --port 8000
```

Then open `http://127.0.0.1:8000` in a browser.

## 4. Verified Results

### Dataset Preparation & Integrity
- **Images scanned**: 2,250 `.jpg` (`1180x784`), 45 categories (50 images/category; 28 `Stable`, 17 `Unstable`)
- **Corrupt / Unsupported / Exact duplicates / Missing labels**: `0`
- **Near-duplicate clusters (`dHash <= 3`)**: `10` clusters (`2,186` leakage-safe groups)
- **Physical batch split**: 5 batches x 450 images (`exact_one_to_one_match: true`)
- **Feature extraction**: 1,505 `ok`, 744 `low_confidence`, 1 `failed` (flagged rather than fabricated)

### Operating-Condition Classification (45 Categories, Leakage-Safe Test Set `n = 357`)
| Model | Accuracy | Macro-F1 | Weighted-F1 |
| --- | --- | --- | --- |
| Random Forest (42 visual features) | 84.87% | 0.8150 | 0.8453 |
| MobileNetV2 CNN (`224x224`, `batch_size=16`) | 96.36% | 0.9622 | 0.9626 |

- Within-split binary Stability Random Forest (`Stable` vs `Unstable`): Accuracy = `94.40%`, Macro-F1 = `0.9349`.

### Continuous Regression & Metadata Audit
- **Froth Depth**: Not present in `ImagesCatalog.xlsx` (`froth_depth_available: false`). No labels were fabricated; `visual_stability_proxy` (0–1) is provided as an unvalidated visual proxy.
- **Air Flow `[L/min]`**: Genuine numeric target (`0, 5, 10, 18` L/min) present for 42 categories (`2,100` images); Categories 1–3 (`Unknown`, `150` images) excluded.
  - Within-condition test (`n = 353`): `MAE = 2.1424 L/min`, `RMSE = 2.8038 L/min`, `R² = 0.7707`
  - Category-held-out 5-fold GroupCV: `MAE = 4.3226 L/min`, `RMSE = 5.1826 L/min`, `R² = 0.1561`

### Held-Out Unsteady-Category Drift & Novelty Detection
- **Closed-set classifier trained only on 28 Stable categories** (`test n = 242`): Accuracy = `86.78%`, Macro-F1 = `0.8333`.
- **17 Held-out Unstable categories (`n = 850` unseen images)** novelty detection vs. in-distribution test set:
  - Combined Novelty Score (`1 - max_prob` + normalized Mahalanobis): `AUROC = 0.7701`, `AUPRC = 0.9122`, `TPR@5%FPR = 34.82%`
- **Cross-category Stability generalization (5-fold GroupKFold by `category_id`)**: Accuracy = `75.56%`, Macro-F1 = `0.7437`.

## 5. Limitations & Project 3 Integration Status

1. **No Sequential Video / Frame Timestamps**: `exif_timestamp_count = 0` and no FPS metadata exist in `ImagesCatalog.xlsx`; `velocity_available` is explicitly `false` and froth velocity is not fabricated.
2. **No Measured Froth Depth or Concentrate Grade**: Plant/lab measurements are required to train and validate froth depth or grade-drift regression.
3. **Project 3 Kinetic Model Integration (`Pending`)**: No Project 3 model or schema was supplied. `backend/project3_adapter.py` implements `Project3KineticModelAdapter` and `FlotationKineticModelProtocol`, preparing both `operating_only_payload` and `operating_plus_image_payload` while marking `integration_status = "pending_project3_model"`.
