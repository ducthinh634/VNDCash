import pytest
import pandas as pd
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.split import stratified_group_split, verify_no_leakage
from src.utils.paths import METADATA_CSV, SPLITS_CSV


def test_stratified_group_split_no_leakage():
    if not METADATA_CSV.exists():
        pytest.skip("metadata.csv not generated yet")

    df = pd.read_csv(METADATA_CSV)
    df_splits = stratified_group_split(df, class_col="denomination", group_col="note_id", seed=42)

    is_leak_free, overlaps = verify_no_leakage(df_splits, group_col="note_id")
    assert is_leak_free, f"Found group leakage: {overlaps}"

    # Verify every note is exclusively in one split
    note_splits = df_splits.groupby("note_id")["split"].nunique()
    assert (note_splits == 1).all(), "Some notes are assigned to multiple splits!"


def test_splits_csv_integrity():
    if not SPLITS_CSV.exists():
        pytest.skip("splits.csv not generated yet")

    df = pd.read_csv(SPLITS_CSV)
    assert len(df) == 1962, f"Expected 1962 samples, got {len(df)}"
    assert set(df["split"].unique()) == {"train", "val", "test"}
    assert df["label_idx"].min() == 0 and df["label_idx"].max() == 8
