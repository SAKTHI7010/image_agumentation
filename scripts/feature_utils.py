"""Shared memory-efficient froth image feature extraction utilities.

Extracts:
- Bubble-size distribution via CLAHE + distance-transform watershed segmentation
- Froth colour statistics (RGB, HSV, CIE Lab, dark/bright fractions)
- Texture entropy (global & local Shannon entropy, Laplacian focus, Sobel gradients, 16-level GLCM)
- Explicit froth-velocity availability flag (False without sequential timestamps)
- Transparent visual stability proxy score (0-1)
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image
from scipy import ndimage

FEATURE_COLUMNS = [
    "bubble_count",
    "bubble_coverage_ratio",
    "bubble_area_mean",
    "bubble_area_std",
    "bubble_area_median",
    "bubble_eq_diameter_mean",
    "bubble_eq_diameter_std",
    "bubble_d10",
    "bubble_d50",
    "bubble_d90",
    "bubble_size_span",
    "bubble_circularity_mean",
    "r_mean",
    "g_mean",
    "b_mean",
    "r_std",
    "g_std",
    "b_std",
    "h_mean",
    "s_mean",
    "v_mean",
    "h_std",
    "s_std",
    "v_std",
    "lab_l_mean",
    "lab_a_mean",
    "lab_b_mean",
    "lab_l_std",
    "lab_a_std",
    "lab_b_std",
    "dark_fraction",
    "bright_fraction",
    "texture_entropy",
    "local_entropy_mean",
    "local_entropy_std",
    "laplacian_var",
    "sobel_mag_mean",
    "sobel_mag_std",
    "glcm_contrast",
    "glcm_homogeneity",
    "glcm_energy",
    "glcm_correlation",
]


def _shannon_entropy_from_gray(gray_u8: np.ndarray) -> float:
    hist = np.bincount(gray_u8.ravel(), minlength=256).astype(np.float64)
    total = hist.sum()
    if total <= 0:
        return 0.0
    probs = hist[hist > 0] / total
    return float(-np.sum(probs * np.log2(probs)))


def _compute_glcm_stats(gray_u8: np.ndarray, levels: int = 16) -> dict[str, float]:
    """Compute fast 16-level symmetric GLCM statistics averaged over 0-deg and 90-deg."""
    q = (gray_u8.astype(np.int32) * levels) // 256
    q = np.clip(q, 0, levels - 1)

    glcm = np.zeros((levels, levels), dtype=np.float64)
    # Horizontal (dx=1, dy=0)
    a_h, b_h = q[:, :-1].ravel(), q[:, 1:].ravel()
    np.add.at(glcm, (a_h, b_h), 1.0)
    np.add.at(glcm, (b_h, a_h), 1.0)
    # Vertical (dx=0, dy=1)
    a_v, b_v = q[:-1, :].ravel(), q[1:, :].ravel()
    np.add.at(glcm, (a_v, b_v), 1.0)
    np.add.at(glcm, (b_v, a_v), 1.0)

    total = glcm.sum()
    if total <= 0:
        return {"glcm_contrast": 0.0, "glcm_homogeneity": 0.0, "glcm_energy": 0.0, "glcm_correlation": 0.0}

    p = glcm / total
    idx = np.arange(levels, dtype=np.float64)
    i_mat, j_mat = np.meshgrid(idx, idx, indexing="ij")
    diff = i_mat - j_mat

    contrast = float(np.sum(p * (diff ** 2)))
    homogeneity = float(np.sum(p / (1.0 + np.abs(diff))))
    energy = float(np.sum(p ** 2))

    mu_i = float(np.sum(i_mat * p))
    mu_j = float(np.sum(j_mat * p))
    sig_i = math.sqrt(max(float(np.sum(((i_mat - mu_i) ** 2) * p)), 1e-12))
    sig_j = math.sqrt(max(float(np.sum(((j_mat - mu_j) ** 2) * p)), 1e-12))
    correlation = float(np.sum(((i_mat - mu_i) * (j_mat - mu_j) * p) / (sig_i * sig_j)))

    return {
        "glcm_contrast": contrast,
        "glcm_homogeneity": homogeneity,
        "glcm_energy": energy,
        "glcm_correlation": correlation,
    }


def _segment_bubbles_watershed(work_bgr: np.ndarray, work_gray: np.ndarray, scale: float) -> dict[str, Any]:
    """Segment froth bubbles using CLAHE + Otsu + distance-transform peaks + OpenCV watershed."""
    h, w = work_gray.shape
    total_px = float(h * w)
    inv_scale = 1.0 / max(scale, 1e-6)
    inv_scale_sq = inv_scale * inv_scale

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(work_gray)
    blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)

    # Otsu separates bright bubble domes from dark lamellae valleys
    _, fg = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((3, 3), np.uint8)
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, kernel, iterations=1)

    dist = cv2.distanceTransform(fg, cv2.DIST_L2, 5)
    max_dist = float(dist.max()) if dist.size else 0.0
    if max_dist < 1.5:
        return {
            "bubble_count": 0,
            "bubble_coverage_ratio": 0.0,
            "bubble_area_mean": None,
            "bubble_area_std": None,
            "bubble_area_median": None,
            "bubble_eq_diameter_mean": None,
            "bubble_eq_diameter_std": None,
            "bubble_d10": None,
            "bubble_d50": None,
            "bubble_d90": None,
            "bubble_size_span": None,
            "bubble_circularity_mean": None,
            "segmentation_confidence": 0.0,
            "segmentation_status": "failed",
        }

    # Local maxima peaks as sure foreground seeds
    local_max = (dist == ndimage.maximum_filter(dist, size=7)) & (dist >= max(2.0, 0.22 * max_dist))
    sure_fg = np.uint8(local_max) * 255
    sure_bg = cv2.dilate(fg, kernel, iterations=2)
    unknown = cv2.subtract(sure_bg, sure_fg)

    num_labels, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    markers = cv2.watershed(work_bgr, markers)

    # Measure areas of watershed basins (labels > 1)
    labels = markers.ravel()
    valid_labels = labels[labels > 1]
    if valid_labels.size == 0:
        counts = np.array([], dtype=np.int64)
    else:
        counts = np.bincount(valid_labels)
        counts = counts[counts >= 9]
        counts = counts[counts < int(0.45 * total_px)]

    bubble_count = int(counts.size)
    coverage = float(counts.sum() / total_px) if bubble_count > 0 else 0.0

    if bubble_count < 3:
        return {
            "bubble_count": bubble_count,
            "bubble_coverage_ratio": round(coverage, 6),
            "bubble_area_mean": None,
            "bubble_area_std": None,
            "bubble_area_median": None,
            "bubble_eq_diameter_mean": None,
            "bubble_eq_diameter_std": None,
            "bubble_d10": None,
            "bubble_d50": None,
            "bubble_d90": None,
            "bubble_size_span": None,
            "bubble_circularity_mean": None,
            "segmentation_confidence": round(min(0.25, bubble_count * 0.08), 4),
            "segmentation_status": "failed",
        }

    # Approximate circularity via boundary pixels inside watershed border (-1)
    areas_orig = counts.astype(np.float64) * inv_scale_sq
    eq_diams = 2.0 * np.sqrt(areas_orig / math.pi)
    d10, d50, d90 = np.percentile(eq_diams, [10, 50, 90])
    size_span = float((d90 - d10) / max(d50, 1e-6))

    # Fast circularity estimate on top up to 150 largest regions via contour area/perimeter
    contours, _ = cv2.findContours(np.uint8(markers > 1), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    circs: list[float] = []
    for cnt in contours[:200]:
        a = cv2.contourArea(cnt)
        p = cv2.arcLength(cnt, True)
        if a >= 9.0 and p > 1e-6:
            circs.append(min(1.0, float(4.0 * math.pi * a / (p * p))))
    circ_mean = float(np.mean(circs)) if circs else 0.5

    # Confidence heuristic
    count_score = min(1.0, bubble_count / 45.0)
    cov_score = 1.0 if 0.08 <= coverage <= 0.88 else 0.4
    confidence = round(float(0.7 * count_score + 0.3 * cov_score), 4)
    status = "ok" if (bubble_count >= 10 and 0.05 <= coverage <= 0.92) else "low_confidence"

    return {
        "bubble_count": bubble_count,
        "bubble_coverage_ratio": round(coverage, 6),
        "bubble_area_mean": round(float(np.mean(areas_orig)), 4),
        "bubble_area_std": round(float(np.std(areas_orig)), 4),
        "bubble_area_median": round(float(np.median(areas_orig)), 4),
        "bubble_eq_diameter_mean": round(float(np.mean(eq_diams)), 4),
        "bubble_eq_diameter_std": round(float(np.std(eq_diams)), 4),
        "bubble_d10": round(float(d10), 4),
        "bubble_d50": round(float(d50), 4),
        "bubble_d90": round(float(d90), 4),
        "bubble_size_span": round(size_span, 4),
        "bubble_circularity_mean": round(circ_mean, 4),
        "segmentation_confidence": confidence,
        "segmentation_status": status,
    }


def compute_visual_stability_proxy(feat: dict[str, Any]) -> float:
    """Transparent 0-1 visual stability proxy derived solely from image physics/texture.

    Not a measured plant variable; reflects size uniformity, texture homogeneity, and focus.
    """
    span = feat.get("bubble_size_span")
    homog = feat.get("glcm_homogeneity", 0.5)
    entropy = feat.get("texture_entropy", 6.0)
    lap = feat.get("laplacian_var", 100.0)

    uniformity_term = 1.0 / (1.0 + max(0.0, float(span))) if span is not None else 0.25
    homog_term = min(1.0, max(0.0, float(homog)))
    # Moderate-to-high entropy (~5.5-7.5) with crisp focus tends to reflect structured froth
    entropy_term = max(0.0, 1.0 - abs(float(entropy) - 6.6) / 3.5)
    focus_term = min(1.0, math.log1p(max(0.0, float(lap))) / 7.0)

    proxy = 0.40 * uniformity_term + 0.30 * homog_term + 0.15 * entropy_term + 0.15 * focus_term
    return round(float(min(1.0, max(0.0, proxy))), 4)


def decode_image_bgr(source: Path | bytes) -> tuple[np.ndarray, int, int]:
    """Decode image from disk path or raw bytes into BGR ndarray, validating readability."""
    if isinstance(source, (str, Path)):
        path = Path(source)
        with Image.open(path) as pil_img:
            pil_img.verify()
        data = np.fromfile(str(path), dtype=np.uint8)
        bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
    else:
        arr = np.frombuffer(source, dtype=np.uint8)
        bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)

    if bgr is None or bgr.size == 0 or bgr.ndim != 3 or bgr.shape[2] != 3:
        raise ValueError("Unable to decode image as valid 3-channel color image.")
    h, w = bgr.shape[:2]
    if h < 16 or w < 16:
        raise ValueError(f"Image dimensions too small ({w}x{h}); minimum 16x16 required.")
    return bgr, w, h


def extract_image_features(
    source: Path | bytes,
    max_work_side: int = 512,
    thumbnail_path: Path | None = None,
) -> dict[str, Any]:
    """Extract all froth visual features for a single image without retaining large arrays."""
    bgr, orig_w, orig_h = decode_image_bgr(source)

    if thumbnail_path is not None and not thumbnail_path.exists():
        thumbnail_path.parent.mkdir(parents=True, exist_ok=True)
        thumb = cv2.resize(bgr, (160, 106), interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(thumbnail_path), thumb, [cv2.IMWRITE_JPEG_QUALITY, 82])

    scale = min(1.0, float(max_work_side) / float(max(orig_h, orig_w)))
    if scale < 1.0:
        work_w = max(16, int(round(orig_w * scale)))
        work_h = max(16, int(round(orig_h * scale)))
        work_bgr = cv2.resize(bgr, (work_w, work_h), interpolation=cv2.INTER_AREA)
    else:
        work_bgr = bgr

    work_gray = cv2.cvtColor(work_bgr, cv2.COLOR_BGR2GRAY)
    work_rgb = cv2.cvtColor(work_bgr, cv2.COLOR_BGR2RGB)
    work_hsv = cv2.cvtColor(work_bgr, cv2.COLOR_BGR2HSV)
    work_lab = cv2.cvtColor(work_bgr, cv2.COLOR_BGR2LAB)

    # 1. Bubble-size distribution
    bubble_stats = _segment_bubbles_watershed(work_bgr, work_gray, scale)

    # 2. Colour statistics
    rgb_mean = work_rgb.mean(axis=(0, 1))
    rgb_std = work_rgb.std(axis=(0, 1))
    hsv_mean = work_hsv.mean(axis=(0, 1))
    hsv_std = work_hsv.std(axis=(0, 1))
    lab_mean = work_lab.mean(axis=(0, 1))
    lab_std = work_lab.std(axis=(0, 1))

    v_chan = work_hsv[:, :, 2]
    dark_fraction = float(np.mean(v_chan < 50))
    bright_fraction = float(np.mean(v_chan > 200))

    # 3. Texture entropy & sharpness/gradient
    global_entropy = _shannon_entropy_from_gray(work_gray)
    # 4x4 grid local entropy
    gh, gw = work_gray.shape
    bh, bw = max(8, gh // 4), max(8, gw // 4)
    local_ents = []
    for r in range(0, gh - bh + 1, bh):
        for c in range(0, gw - bw + 1, bw):
            local_ents.append(_shannon_entropy_from_gray(work_gray[r : r + bh, c : c + bw]))
    local_ent_mean = float(np.mean(local_ents)) if local_ents else global_entropy
    local_ent_std = float(np.std(local_ents)) if local_ents else 0.0

    lap_var = float(cv2.Laplacian(work_gray, cv2.CV_64F).var())
    gx = cv2.Sobel(work_gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(work_gray, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = np.hypot(gx, gy)
    sobel_mean = float(np.mean(grad_mag))
    sobel_std = float(np.std(grad_mag))

    glcm_stats = _compute_glcm_stats(work_gray, levels=16)

    result: dict[str, Any] = {
        "orig_width": orig_w,
        "orig_height": orig_h,
        **bubble_stats,
        "r_mean": round(float(rgb_mean[0]), 4),
        "g_mean": round(float(rgb_mean[1]), 4),
        "b_mean": round(float(rgb_mean[2]), 4),
        "r_std": round(float(rgb_std[0]), 4),
        "g_std": round(float(rgb_std[1]), 4),
        "b_std": round(float(rgb_std[2]), 4),
        "h_mean": round(float(hsv_mean[0]), 4),
        "s_mean": round(float(hsv_mean[1]), 4),
        "v_mean": round(float(hsv_mean[2]), 4),
        "h_std": round(float(hsv_std[0]), 4),
        "s_std": round(float(hsv_std[1]), 4),
        "v_std": round(float(hsv_std[2]), 4),
        "lab_l_mean": round(float(lab_mean[0]), 4),
        "lab_a_mean": round(float(lab_mean[1]), 4),
        "lab_b_mean": round(float(lab_mean[2]), 4),
        "lab_l_std": round(float(lab_std[0]), 4),
        "lab_a_std": round(float(lab_std[1]), 4),
        "lab_b_std": round(float(lab_std[2]), 4),
        "dark_fraction": round(dark_fraction, 6),
        "bright_fraction": round(bright_fraction, 6),
        "texture_entropy": round(global_entropy, 6),
        "local_entropy_mean": round(local_ent_mean, 6),
        "local_entropy_std": round(local_ent_std, 6),
        "laplacian_var": round(lap_var, 4),
        "sobel_mag_mean": round(sobel_mean, 4),
        "sobel_mag_std": round(sobel_std, 4),
        **{k: round(v, 6) for k, v in glcm_stats.items()},
        "velocity_available": False,
        "froth_velocity_px_s": None,
        "velocity_note": "No genuine sequential video frames or timestamps available",
    }
    result["visual_stability_proxy"] = compute_visual_stability_proxy(result)
    return result
