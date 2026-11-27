"""Phase 2: Incrementally extract visual features from each batch manifest with checkpoint/resume support."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from feature_utils import extract_image_features  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Incremental, resumable froth feature extraction across batches.")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--flush-every", type=int, default=25, help="Flush rows to CSV every N images.")
    parser.add_argument("--no-thumbnails", action="store_true", help="Skip generating UI preview thumbnails.")
    args = parser.parse_args()

    root: Path = args.project_root
    manifests_dir = root / "manifests"
    features_dir = root / "processed" / "features"
    thumbs_dir = root / "processed" / "thumbnails"
    features_dir.mkdir(parents=True, exist_ok=True)
    thumbs_dir.mkdir(parents=True, exist_ok=True)

    features_csv = features_dir / "features.csv"
    report_csv = root / "processed" / "preprocessing_report.csv"

    completed_ids: set[int] = set()
    existing_header: list[str] | None = None
    if features_csv.exists() and features_csv.stat().st_size > 0:
        prev_df = pd.read_csv(features_csv, usecols=["image_id"])
        completed_ids = set(prev_df["image_id"].astype(int).tolist())
        existing_header = list(pd.read_csv(features_csv, nrows=0).columns)
        print(f"Resuming feature extraction: {len(completed_ids)} images already completed.")

    batch_files = sorted(manifests_dir.glob("batch_*.csv"))
    if not batch_files:
        raise FileNotFoundError(f"No batch_*.csv manifests found in {manifests_dir}")

    processed_now = 0
    failed_count = 0
    low_conf_count = 0
    batch_report_rows: list[dict[str, Any]] = []

    fieldnames: list[str] | None = existing_header
    buffer: list[dict[str, Any]] = []

    def flush_buffer() -> None:
        nonlocal fieldnames, buffer
        if not buffer:
            return
        if fieldnames is None:
            fieldnames = list(buffer[0].keys())
        file_exists = features_csv.exists() and features_csv.stat().st_size > 0
        with features_csv.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerows(buffer)
        buffer.clear()

    for bfile in batch_files:
        bdf = pd.read_csv(bfile)
        batch_id = bfile.stem
        b_ok = 0
        b_low = 0
        b_fail = 0
        b_skipped = 0

        for row in bdf.to_dict(orient="records"):
            img_id = int(row["image_id"])
            if img_id in completed_ids:
                b_skipped += 1
                continue

            img_path = root / row["relative_path"]
            thumb_path = None if args.no_thumbnails else (thumbs_dir / f"{img_id}.jpg")

            try:
                feats = extract_image_features(img_path, max_work_side=512, thumbnail_path=thumb_path)
                status = feats["segmentation_status"]
                if status == "ok":
                    b_ok += 1
                elif status == "low_confidence":
                    b_low += 1
                    low_conf_count += 1
                else:
                    b_fail += 1
                    failed_count += 1
                extraction_error = ""
            except Exception as exc:
                b_fail += 1
                failed_count += 1
                feats = {"segmentation_status": "error", "segmentation_confidence": 0.0, "velocity_available": False}
                extraction_error = str(exc)

            combined = {
                "image_id": img_id,
                "batch_id": batch_id,
                "relative_path": row["relative_path"],
                "source_relative_path": row["source_relative_path"],
                "category_id": int(row["category_id"]),
                "category": row["category"],
                "category_slug": row["category_slug"],
                "leakage_group_id": row["leakage_group_id"],
                "mineral_used": row["mineral_used"],
                "collector_dosage_raw": row["collector_dosage_raw"],
                "collector_dosage_gt": row["collector_dosage_gt"],
                "frother_condition_raw": row["frother_condition_raw"],
                "frother_condition_gt": row["frother_condition_gt"],
                "air_flow_raw": row["air_flow_raw"],
                "air_flow_l_min": row["air_flow_l_min"],
                "kind_of_water": row["kind_of_water"],
                "special_condition": row["special_condition"],
                "stability": row["stability"],
                "froth_load": row["froth_load"],
                "visible_pulp": row["visible_pulp"],
                "quality": row["quality"],
                "extraction_error": extraction_error,
                **feats,
            }
            buffer.append(combined)
            completed_ids.add(img_id)
            processed_now += 1

            if len(buffer) >= args.flush_every:
                flush_buffer()

        flush_buffer()
        batch_report_rows.append({
            "batch_id": batch_id,
            "total_in_batch": len(bdf),
            "skipped_already_done": b_skipped,
            "segmentation_ok": b_ok,
            "segmentation_low_confidence": b_low,
            "segmentation_failed_or_error": b_fail,
        })
        print(f"Finished {batch_id}: total={len(bdf)}, skipped={b_skipped}, ok={b_ok}, low_conf={b_low}, fail={b_fail}")

    pd.DataFrame(batch_report_rows).to_csv(report_csv, index=False)

    # Also write parquet summary & json stats from disk CSV
    full_df = pd.read_csv(features_csv)
    parquet_path = features_dir / "features.parquet"
    full_df.to_parquet(parquet_path, index=False)

    summary = {
        "total_rows": len(full_df),
        "unique_images": int(full_df["image_id"].nunique()),
        "segmentation_status_counts": full_df["segmentation_status"].value_counts().to_dict(),
        "velocity_available_count": int(full_df["velocity_available"].astype(bool).sum()),
        "processed_in_this_run": processed_now,
    }
    (root / "processed" / "feature_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
