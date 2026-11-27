"""Phase 1A: Inspect the froth image dataset and catalog metadata without loading all images into RAM."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import ExifTags, Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = False

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def clean_str(val: Any) -> str:
    if pd.isna(val):
        return ""
    return re.sub(r"\s+", " ", str(val).strip())


def parse_image_range(range_text: str) -> tuple[int, int]:
    match = re.match(r"^\s*(\d+)\s*-\s*(\d+)\s*$", str(range_text))
    if not match:
        raise ValueError(f"Invalid image range format: {range_text!r}")
    return int(match.group(1)), int(match.group(2))


def parse_category_num(cat_text: str) -> int:
    match = re.search(r"(\d+)", str(cat_text))
    return int(match.group(1)) if match else -1


def load_catalog_lookup(catalog_path: Path) -> tuple[dict[int, dict[str, Any]], pd.DataFrame]:
    """Load both sheets of ImagesCatalog.xlsx and return per-image-id lookup and merged category dataframe."""
    wc_df = pd.read_excel(catalog_path, sheet_name="Work Conditions")
    vc_df = pd.read_excel(catalog_path, sheet_name="Visual Characteristics")
    wc_df.columns = [clean_str(c) for c in wc_df.columns]
    vc_df.columns = [clean_str(c) for c in vc_df.columns]

    wc_cols = [
        "Category",
        "Images",
        "Mineral used",
        "Collector dosage [g/t]",
        "Frother condition [g/t]",
        "Air flow [L/min]",
        "Kind of water",
        "Special condition",
    ]
    wc_df = wc_df[wc_cols].copy()
    for col in wc_df.columns:
        wc_df[col] = wc_df[col].map(clean_str)

    for col in vc_df.columns:
        vc_df[col] = vc_df[col].map(clean_str)

    merged = pd.merge(wc_df, vc_df, on=["Category", "Images"], how="outer", suffixes=("_wc", "_vc"))
    lookup: dict[int, dict[str, Any]] = {}

    for _, row in merged.iterrows():
        cat_name = clean_str(row["Category"])
        cat_num = parse_category_num(cat_name)
        cat_slug = f"Category_{cat_num:02d}" if cat_num > 0 else cat_name.replace(" ", "_")
        start_id, end_id = parse_image_range(row["Images"])

        air_raw = clean_str(row["Air flow [L/min]"])
        collector_raw = clean_str(row["Collector dosage [g/t]"])
        frother_raw = clean_str(row["Frother condition [g/t]"])

        meta = {
            "category_id": cat_num,
            "category": cat_name,
            "category_slug": cat_slug,
            "image_range": clean_str(row["Images"]),
            "mineral_used": clean_str(row["Mineral used"]),
            "collector_dosage_raw": collector_raw,
            "collector_dosage_gt": float(collector_raw) if collector_raw.replace(".", "", 1).isdigit() else (0.0 if collector_raw.lower() == "without adding" else None),
            "frother_condition_raw": frother_raw,
            "frother_condition_gt": float(frother_raw) if frother_raw.replace(".", "", 1).isdigit() else (0.0 if frother_raw.lower() == "without adding" else None),
            "air_flow_raw": air_raw,
            "air_flow_l_min": float(air_raw) if air_raw.replace(".", "", 1).isdigit() else None,
            "kind_of_water": clean_str(row["Kind of water"]),
            "special_condition": clean_str(row["Special condition"]),
            "visual_color": clean_str(row["Color"]),
            "size_distribution": clean_str(row["Size distribution"]),
            "displayed_sizes": clean_str(row["Displayed sizes"]),
            "stability": clean_str(row["Stability"]),
            "froth_load": clean_str(row["Froth load"]),
            "visible_pulp": clean_str(row["Visible pulp"]),
            "quality": clean_str(row["Quality"]),
        }
        for img_id in range(start_id, end_id + 1):
            lookup[img_id] = meta

    return lookup, merged


def compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def compute_dhash(img: Image.Image, hash_size: int = 8) -> int:
    gray = img.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.BILINEAR)
    arr = np.asarray(gray, dtype=np.int16)
    diff = arr[:, 1:] > arr[:, :-1]
    bits = 0
    for val in diff.flatten():
        bits = (bits << 1) | int(val)
    return bits


def hamming_distance(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def extract_exif_timestamp(img: Image.Image) -> str:
    try:
        exif = img.getexif()
        if not exif:
            return ""
        for tag_id, value in exif.items():
            tag = ExifTags.TAGS.get(tag_id, str(tag_id))
            if tag in ("DateTimeOriginal", "DateTimeDigitized", "DateTime"):
                return str(value)
    except Exception:
        return ""
    return ""


def natural_sort_key(path: Path) -> tuple[int, str]:
    m = re.search(r"(\d+)", path.stem)
    return (int(m.group(1)) if m else 10**9, path.name.lower())


def assign_near_duplicate_clusters(records: list[dict[str, Any]], max_hamming: int = 3) -> int:
    """Cluster near-duplicate images per category (and globally by hash) to prevent split leakage."""
    valid_indices = [i for i, r in enumerate(records) if r["validation_status"] == "valid" and r["dhash_int"] is not None]
    parent = {i: i for i in valid_indices}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    # Exact SHA-256 duplicates across entire dataset
    by_sha: dict[str, list[int]] = defaultdict(list)
    for idx in valid_indices:
        by_sha[records[idx]["sha256"]].append(idx)
    for idxs in by_sha.values():
        for j in idxs[1:]:
            union(idxs[0], j)

    # Near-duplicates within each category (O(50^2) per category -> instant)
    by_cat: dict[int, list[int]] = defaultdict(list)
    for idx in valid_indices:
        by_cat[records[idx]["category_id"]].append(idx)

    for idxs in by_cat.values():
        n = len(idxs)
        for i in range(n):
            hi = records[idxs[i]]["dhash_int"]
            for j in range(i + 1, n):
                hj = records[idxs[j]]["dhash_int"]
                if hamming_distance(hi, hj) <= max_hamming:
                    union(idxs[i], idxs[j])

    cluster_map: dict[int, str] = {}
    cluster_sizes: Counter[int] = Counter()
    for idx in valid_indices:
        cluster_sizes[find(idx)] += 1

    near_dup_groups = 0
    for root, size in cluster_sizes.items():
        if size > 1:
            near_dup_groups += 1
        cluster_map[root] = f"grp_{root:04d}"

    for idx in valid_indices:
        records[idx]["leakage_group_id"] = cluster_map[find(idx)]
        records[idx]["leakage_group_size"] = cluster_sizes[find(idx)]

    return near_dup_groups


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect original froth image dataset and catalog.")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--max-hamming", type=int, default=3, help="dHash Hamming threshold for near-duplicate grouping.")
    args = parser.parse_args()

    root: Path = args.project_root
    orig_images_dir = root / "data_original" / "images"
    catalog_path = root / "data_original" / "ImagesCatalog.xlsx"
    manifests_dir = root / "manifests"
    manifests_dir.mkdir(parents=True, exist_ok=True)

    lookup, catalog_df = load_catalog_lookup(catalog_path)
    all_files = sorted([p for p in orig_images_dir.rglob("*") if p.is_file()], key=natural_sort_key)

    ext_counts: Counter[str] = Counter()
    seen_sha: dict[str, str] = {}
    records: list[dict[str, Any]] = []
    corrupt_count = 0
    exact_dup_count = 0
    missing_label_count = 0
    exif_ts_count = 0

    for file_path in all_files:
        ext = file_path.suffix.lower()
        ext_counts[ext] += 1
        rel_path = file_path.relative_to(root).as_posix()
        file_size = file_path.stat().st_size

        stem_match = re.match(r"^(\d+)$", file_path.stem)
        image_id = int(stem_match.group(1)) if stem_match else -1
        meta = lookup.get(image_id)
        if meta is None:
            missing_label_count += 1

        sha256 = compute_sha256(file_path)
        is_exact_dup = sha256 in seen_sha
        dup_of = seen_sha.get(sha256, "")
        if not is_exact_dup:
            seen_sha[sha256] = rel_path
        else:
            exact_dup_count += 1

        width = None
        height = None
        mode = ""
        img_format = ""
        exif_ts = ""
        dhash_hex = ""
        dhash_int = None
        validation_status = "valid"
        error_message = ""

        if ext not in IMAGE_EXTENSIONS:
            validation_status = "unsupported_extension"
            error_message = f"Unsupported extension {ext}"
            corrupt_count += 1
        else:
            try:
                with Image.open(file_path) as img:
                    img.verify()
                with Image.open(file_path) as img:
                    width, height = img.size
                    mode = img.mode
                    img_format = img.format or ""
                    exif_ts = extract_exif_timestamp(img)
                    if exif_ts:
                        exif_ts_count += 1
                    dhash_int = compute_dhash(img)
                    dhash_hex = f"{dhash_int:016x}"
            except Exception as exc:
                validation_status = "corrupt"
                error_message = str(exc)
                corrupt_count += 1

        rec: dict[str, Any] = {
            "image_id": image_id,
            "filename": file_path.name,
            "source_relative_path": rel_path,
            "extension": ext,
            "file_size_bytes": file_size,
            "width": width,
            "height": height,
            "color_mode": mode,
            "image_format": img_format,
            "sha256": sha256,
            "dhash_hex": dhash_hex,
            "dhash_int": dhash_int,
            "is_exact_duplicate": is_exact_dup,
            "duplicate_of": dup_of,
            "exif_timestamp": exif_ts,
            "validation_status": validation_status,
            "error_message": error_message,
            "has_label": meta is not None,
            "leakage_group_id": "",
            "leakage_group_size": 1,
        }
        if meta is not None:
            rec.update(meta)
        else:
            rec.update({
                "category_id": -1,
                "category": "MISSING",
                "category_slug": "MISSING",
                "image_range": "",
                "mineral_used": "",
                "collector_dosage_raw": "",
                "collector_dosage_gt": None,
                "frother_condition_raw": "",
                "frother_condition_gt": None,
                "air_flow_raw": "",
                "air_flow_l_min": None,
                "kind_of_water": "",
                "special_condition": "",
                "visual_color": "",
                "size_distribution": "",
                "displayed_sizes": "",
                "stability": "",
                "froth_load": "",
                "visible_pulp": "",
                "quality": "",
            })
        records.append(rec)

    near_dup_groups = assign_near_duplicate_clusters(records, max_hamming=args.max_hamming)

    inv_df = pd.DataFrame(records).drop(columns=["dhash_int"])
    inv_path = manifests_dir / "dataset_inventory.csv"
    inv_df.to_csv(inv_path, index=False)

    # Check catalog coverage (missing expected IDs)
    found_ids = {r["image_id"] for r in records if r["image_id"] > 0}
    expected_ids = set(lookup.keys())
    missing_from_disk = sorted(expected_ids - found_ids)

    category_summary = (
        inv_df.groupby(["category_id", "category", "stability", "quality", "air_flow_raw"], dropna=False)
        .agg(
            image_count=("image_id", "count"),
            valid_count=("validation_status", lambda s: int((s == "valid").sum())),
            exact_dup_count=("is_exact_duplicate", "sum"),
            unique_leakage_groups=("leakage_group_id", "nunique"),
        )
        .reset_index()
        .sort_values("category_id")
    )
    category_summary.to_csv(manifests_dir / "category_summary.csv", index=False)

    dims_series = inv_df.loc[inv_df["validation_status"] == "valid", ["width", "height"]].apply(
        lambda r: f"{int(r['width'])}x{int(r['height'])}", axis=1
    )
    dim_counts = dims_series.value_counts().to_dict()

    report = {
        "total_files_scanned": len(all_files),
        "valid_images": int((inv_df["validation_status"] == "valid").sum()),
        "corrupt_or_unsupported": corrupt_count,
        "exact_duplicates": exact_dup_count,
        "near_duplicate_clusters_dhash_le_3": near_dup_groups,
        "unique_leakage_groups": int(inv_df.loc[inv_df["validation_status"] == "valid", "leakage_group_id"].nunique()),
        "missing_labels": missing_label_count,
        "missing_expected_catalog_ids": len(missing_from_disk),
        "extensions": dict(ext_counts),
        "dimensions": dim_counts,
        "exif_timestamp_count": exif_ts_count,
        "categories_total": int(catalog_df["Category"].nunique()),
        "stability_distribution_categories": catalog_df["Stability"].value_counts().to_dict(),
        "air_flow_distribution_categories": catalog_df["Air flow [L/min]"].value_counts().to_dict(),
        "froth_depth_available": False,
        "video_velocity_metadata_available": exif_ts_count > 0,
    }

    (manifests_dir / "inspection_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    md_lines = [
        "# Dataset Inspection Report",
        "",
        f"- **Source images scanned**: {report['total_files_scanned']}",
        f"- **Valid readable images**: {report['valid_images']}",
        f"- **Corrupt / unsupported files**: {report['corrupt_or_unsupported']}",
        f"- **Exact SHA-256 duplicates**: {report['exact_duplicates']}",
        f"- **Near-duplicate clusters (dHash <= {args.max_hamming})**: {report['near_duplicate_clusters_dhash_le_3']}",
        f"- **Unique leakage-safe groups**: {report['unique_leakage_groups']}",
        f"- **Missing labels**: {report['missing_labels']}",
        f"- **Missing catalog IDs on disk**: {report['missing_expected_catalog_ids']}",
        f"- **Extensions**: {report['extensions']}",
        f"- **Image dimensions**: {report['dimensions']}",
        f"- **EXIF timestamps found**: {report['exif_timestamp_count']} (froth velocity unavailable without frame timestamps)",
        f"- **Froth depth column in catalog**: False (Air flow [L/min] is present for 42/45 categories)",
        "",
    ]
    (manifests_dir / "inspection_report.md").write_text("\n".join(md_lines), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
