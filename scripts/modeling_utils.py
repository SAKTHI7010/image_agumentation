"""Shared dataset splitting and leakage-prevention utilities for Phase 3."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


def build_leakage_safe_splits(
    features_df: pd.DataFrame,
    splits_path: Path,
    seed: int = 42,
) -> pd.DataFrame:
    """Create deterministic ~70/15/15 train/val/test splits grouped by leakage_group_id and stratified by category_id."""
    if splits_path.exists():
        existing = pd.read_csv(splits_path)
        if len(existing) == len(features_df) and set(existing["image_id"]) == set(features_df["image_id"]):
            return existing

    df = features_df[["image_id", "category_id", "category", "leakage_group_id", "batch_id"]].copy()
    df["split"] = "train"

    sgkf = StratifiedGroupKFold(n_splits=7, shuffle=True, random_state=seed)
    folds = np.zeros(len(df), dtype=int)
    for fold_idx, (_, test_idx) in enumerate(sgkf.split(df, df["category_id"], df["leakage_group_id"])):
        folds[test_idx] = fold_idx

    df.loc[folds == 0, "split"] = "test"
    df.loc[folds == 1, "split"] = "val"
    df.loc[folds >= 2, "split"] = "train"

    train_groups = set(df.loc[df["split"] == "train", "leakage_group_id"])
    val_groups = set(df.loc[df["split"] == "val", "leakage_group_id"])
    test_groups = set(df.loc[df["split"] == "test", "leakage_group_id"])
    assert not (train_groups & val_groups), "Leakage group overlap between train and val!"
    assert not (train_groups & test_groups), "Leakage group overlap between train and test!"
    assert not (val_groups & test_groups), "Leakage group overlap between val and test!"

    splits_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(splits_path, index=False)
    return df


def select_numeric_feature_columns(df: pd.DataFrame, extra_exclude: Sequence[str] = ()) -> list[str]:
    from feature_utils import FEATURE_COLUMNS

    return [c for c in FEATURE_COLUMNS if c in df.columns and c not in set(extra_exclude)]
