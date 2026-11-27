"""Phase 1B: Physically copy validated images into batch folders and build batch + master CSV manifests."""

from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Split validated froth images into physical batch folders.")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--target-batch-size", type=int, default=500, help="Target max images per batch.")
    args = parser.parse_args()

    root: Path = args.project_root
    manifests_dir = root / "manifests"
    batches_dir = root / "data_batches"
    inv_path = manifests_dir / "dataset_inventory.csv"
    inv_df = pd.read_csv(inv_path)

    # Separate anomalies (corrupt, unsupported, exact duplicates, missing labels) without silently discarding
    anomalies_mask = (
        (inv_df["validation_status"] != "valid")
        | (inv_df["is_exact_duplicate"].astype(bool))
        | (~inv_df["has_label"].astype(bool))
    )
    anomalies_df = inv_df[anomalies_mask].copy()
    anomalies_df.to_csv(manifests_dir / "anomalies_report.csv", index=False)

    valid_df = inv_df[~anomalies_mask].copy()
    valid_df = valid_df.sort_values(["category_id", "image_id"]).reset_index(drop=True)

    total_valid = len(valid_df)
    if total_valid == 0:
        raise RuntimeError("No valid images found in dataset_inventory.csv")

    num_batches = max(1, math.ceil(total_valid / args.target_batch_size))

    # Stratified round-robin assignment across batches within each category so every batch is balanced
    batch_assignments: list[int] = [0] * total_valid
    for _, group_idx in valid_df.groupby("category_id", sort=True).groups.items():
        for pos, row_idx in enumerate(sorted(group_idx)):
            batch_assignments[row_idx] = (pos % num_batches) + 1

    valid_df["batch_id"] = [f"batch_{b:03d}" for b in batch_assignments]

    master_rows: list[dict] = []
    batch_dfs: dict[str, list[dict]] = {f"batch_{b:03d}": [] for b in range(1, num_batches + 1)}

    for row in valid_df.to_dict(orient="records"):
        batch_id = row["batch_id"]
        cat_slug = row["category_slug"]
        src_path = root / row["source_relative_path"]
        dst_dir = batches_dir / batch_id / cat_slug
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst_path = dst_dir / row["filename"]

        if not dst_path.exists() or dst_path.stat().st_size != src_path.stat().st_size:
            shutil.copy2(src_path, dst_path)

        entry = dict(row)
        entry["relative_path"] = dst_path.relative_to(root).as_posix()
        master_rows.append(entry)
        batch_dfs[batch_id].append(entry)

    master_df = pd.DataFrame(master_rows)
    master_df.to_csv(manifests_dir / "master_manifest.csv", index=False)

    batch_counts: dict[str, int] = {}
    batch_manifest_paths: list[str] = []
    seen_in_batches: list[str] = []

    for batch_id, rows in sorted(batch_dfs.items()):
        bdf = pd.DataFrame(rows)
        bpath = manifests_dir / f"{batch_id}.csv"
        bdf.to_csv(bpath, index=False)
        batch_counts[batch_id] = len(bdf)
        batch_manifest_paths.append(bpath.relative_to(root).as_posix())
        seen_in_batches.extend(bdf["source_relative_path"].tolist())

    # Strict validation: every valid source image appears exactly once across batch manifests
    valid_sources = set(valid_df["source_relative_path"].tolist())
    batch_sources = set(seen_in_batches)
    validation = {
        "total_inventory_rows": len(inv_df),
        "valid_images_expected": len(valid_sources),
        "anomalies_reported": len(anomalies_df),
        "num_batches": num_batches,
        "batch_counts": batch_counts,
        "total_rows_across_batch_manifests": len(seen_in_batches),
        "unique_sources_in_batch_manifests": len(batch_sources),
        "exact_one_to_one_match": (len(seen_in_batches) == len(valid_sources)) and (batch_sources == valid_sources),
        "batch_manifests": batch_manifest_paths,
    }

    (manifests_dir / "split_validation.json").write_text(json.dumps(validation, indent=2), encoding="utf-8")
    if not validation["exact_one_to_one_match"]:
        raise AssertionError(f"Batch split validation failed: {validation}")

    print(json.dumps(validation, indent=2))


if __name__ == "__main__":
    main()
