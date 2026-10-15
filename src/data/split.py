"""Module for grouped, stratified train/val/test splitting without data leakage.

Ensures all images belonging to the same physical banknote (note_id)
are placed exclusively into either train, val, or test set.
"""

from typing import Dict, List, Set, Tuple
import numpy as np
import pandas as pd


def verify_no_leakage(df_split: pd.DataFrame, group_col: str = "note_id") -> Tuple[bool, Dict[str, Set]]:
    """Verifies that no groups (notes) or hashes leak between splits.

    Returns:
        (is_leak_free, overlap_dict)
    """
    train_groups = set(df_split[df_split["split"] == "train"][group_col])
    val_groups = set(df_split[df_split["split"] == "val"][group_col])
    test_groups = set(df_split[df_split["split"] == "test"][group_col])

    overlap_train_val = train_groups & val_groups
    overlap_train_test = train_groups & test_groups
    overlap_val_test = val_groups & test_groups

    overlaps = {
        "train_val": overlap_train_val,
        "train_test": overlap_train_test,
        "val_test": overlap_val_test,
    }

    is_leak_free = len(overlap_train_val) == 0 and len(overlap_train_test) == 0 and len(overlap_val_test) == 0
    return is_leak_free, overlaps


def stratified_group_split(
    df: pd.DataFrame,
    class_col: str = "denomination",
    group_col: str = "note_id",
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42
) -> pd.DataFrame:
    """Performs stratified grouped splitting by class and note_id.

    For each denomination class:
    - Extracts unique note_ids
    - Shuffles note_ids deterministically with the specified seed
    - Allocates note_ids to train, val, test matching the target ratios
    - Maps the split label back to all image samples belonging to each note_id.
    """
    rng = np.random.default_rng(seed)
    df_out = df.copy()
    split_map: Dict[str, str] = {}

    for denom, group_df in df.groupby(class_col):
        unique_notes = sorted(group_df[group_col].unique())
        rng.shuffle(unique_notes)
        n_notes = len(unique_notes)

        # Compute note counts for val and test (at least 1 note each if possible)
        n_val = max(1, int(round(n_notes * val_ratio)))
        n_test = max(1, int(round(n_notes * test_ratio)))
        n_train = n_notes - n_val - n_test

        if n_train <= 0:
            # Fallback if too few notes
            n_train = max(1, n_notes - 2)
            n_val = 1
            n_test = max(1, n_notes - n_train - n_val)

        train_notes = unique_notes[:n_train]
        val_notes = unique_notes[n_train:n_train + n_val]
        test_notes = unique_notes[n_train + n_val:]

        for note in train_notes:
            split_map[note] = "train"
        for note in val_notes:
            split_map[note] = "val"
        for note in test_notes:
            split_map[note] = "test"

    df_out["split"] = df_out[group_col].map(split_map)

    # Double check no null splits
    if df_out["split"].isnull().any():
        raise ValueError("Lỗi: Tồn tại bản ghi chưa được gán split!")

    # Verify anti-leakage
    is_valid, overlaps = verify_no_leakage(df_out, group_col=group_col)
    if not is_valid:
        raise ValueError(f"Rò rỉ dữ liệu (Data Leakage) phát hiện trong split: {overlaps}")

    return df_out
