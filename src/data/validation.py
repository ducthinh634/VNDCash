"""Data validation module for VNDCash.

Responsibilities (Member A):
- Validate DataFrame metadata schema and data integrity.
- Check file existence and image readability.
- Validate split completeness and strictly check for zero data leakage.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from PIL import Image

from .split import verify_no_leakage

REQUIRED_METADATA_COLS = [
    "image_id", "filename", "rel_path", "denomination", "label_idx",
    "note_id", "width", "height", "channels", "color_mode"
]

REQUIRED_SPLIT_COLS = REQUIRED_METADATA_COLS + ["split"]


def validate_metadata_schema(df: pd.DataFrame, is_split: bool = False) -> Tuple[bool, List[str]]:
    """Checks whether the DataFrame contains all required columns with non-null values."""
    required = REQUIRED_SPLIT_COLS if is_split else REQUIRED_METADATA_COLS
    missing = [c for c in required if c not in df.columns]
    if missing:
        return False, [f"Missing required columns: {missing}"]

    errors = []
    # Check nulls
    null_counts = df[required].isnull().sum()
    for col, cnt in null_counts.items():
        if cnt > 0:
            errors.append(f"Column '{col}' has {cnt} null values")

    # Check label_idx values
    if "label_idx" in df.columns:
        invalid_labels = df[~df["label_idx"].isin(range(9))]
        if len(invalid_labels) > 0:
            errors.append(f"Found {len(invalid_labels)} rows with label_idx outside range [0, 8]")

    # Check split column
    if is_split:
        valid_splits = {"train", "val", "test"}
        actual_splits = set(df["split"].unique())
        if not actual_splits.issubset(valid_splits):
            errors.append(f"Unexpected split names: {actual_splits - valid_splits}")

    return len(errors) == 0, errors


def validate_image_files(
    df: pd.DataFrame,
    root_dir: Optional[Union[str, Path]] = None,
    sample_size: Optional[int] = None
) -> Tuple[bool, List[str]]:
    """Validates existence and readability of image files referenced in DataFrame.

    Args:
        df: Metadata or splits DataFrame.
        root_dir: Base directory for relative paths (defaults to cwd or project root).
        sample_size: If specified, validates a random subset of rows for speed.
    """
    root_path = Path(root_dir) if root_dir else Path.cwd()
    check_df = df.sample(n=sample_size, random_state=42) if (sample_size and sample_size < len(df)) else df

    errors = []
    for _, row in check_df.iterrows():
        p = root_path / row["rel_path"]
        if not p.exists():
            errors.append(f"File not found: {p}")
            continue
        try:
            with Image.open(p) as img:
                img.verify()
        except Exception as e:
            errors.append(f"Corrupt image file {p}: {e}")

    return len(errors) == 0, errors


def validate_dataset_pipeline(
    df_splits: pd.DataFrame,
    root_dir: Optional[Union[str, Path]] = None,
    group_col: str = "note_id"
) -> Dict[str, Union[bool, List[str], Dict]]:
    """Comprehensive pipeline validation checking schema, files, and anti-leakage."""
    schema_ok, schema_errors = validate_metadata_schema(df_splits, is_split=True)
    files_ok, file_errors = validate_image_files(df_splits, root_dir=root_dir, sample_size=100)
    leak_free, overlaps = verify_no_leakage(df_splits, group_col=group_col)

    all_passed = schema_ok and files_ok and leak_free

    return {
        "passed": all_passed,
        "schema_ok": schema_ok,
        "schema_errors": schema_errors,
        "files_ok": files_ok,
        "file_errors": file_errors,
        "leak_free": leak_free,
        "leak_overlaps": {k: list(v) for k, v in overlaps.items()},
        "total_samples": len(df_splits),
        "splits_distribution": df_splits["split"].value_counts().to_dict(),
        "denominations": sorted(df_splits["denomination"].unique().tolist())
    }
