"""Unit tests for YOLOv11 dataset integrity and Zero Data Leakage.

Author: Nguyễn Đức Thịnh (22696331) - Member A
Stack: Stack01 (Stack114)
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import yaml
import pandas as pd
import pytest

from src.vndcash.data.split import verify_zero_leakage, CLASS_NAMES_DEFAULT
from src.vndcash.data.validation import validate_yolo_label_file

METADATA_CSV = PROJECT_ROOT / "data" / "metadata" / "metadata.csv"
SPLITS_CSV = PROJECT_ROOT / "data" / "metadata" / "splits.csv"
YOLO_DIR = PROJECT_ROOT / "data" / "processed" / "yolo_dataset"


def test_metadata_csv_integrity():
    """Verify metadata.csv exists, is non-empty, and has expected schema."""
    assert METADATA_CSV.exists(), f"Missing {METADATA_CSV}"
    df = pd.read_csv(METADATA_CSV)
    assert len(df) > 0, "metadata.csv is empty"

    required_cols = [
        "image_id", "filename", "image_path", "label_path", "note_id",
        "primary_class", "denominations", "num_notes", "is_hard_case",
        "width", "height", "aspect_ratio", "file_size_bytes", "md5_hash"
    ]
    for col in required_cols:
        assert col in df.columns, f"Missing required column: {col}"

    assert df["image_id"].is_unique, "image_id values must be unique"
    assert (df["num_notes"] >= 1).all(), "All multi-note images must have at least 1 note"


def test_splits_csv_integrity():
    """Verify splits.csv exists and has train, val, and test partitions."""
    assert SPLITS_CSV.exists(), f"Missing {SPLITS_CSV}"
    df_splits = pd.read_csv(SPLITS_CSV)
    assert len(df_splits) > 0, "splits.csv is empty"
    assert "split" in df_splits.columns, "Missing 'split' column"

    splits_present = set(df_splits["split"].unique())
    assert {"train", "val", "test"}.issubset(splits_present), f"Invalid splits: {splits_present}"


def test_zero_leakage_note_id():
    """Verify that physical note_ids are strictly disjoint across train/val/test splits."""
    df_splits = pd.read_csv(SPLITS_CSV)
    is_leak_free, overlaps = verify_zero_leakage(df_splits, group_col="note_id")

    assert is_leak_free, f"Zero leakage violated! Overlaps: {overlaps}"
    assert len(overlaps["train_val"]) == 0, f"Overlap train-val: {overlaps['train_val']}"
    assert len(overlaps["train_test"]) == 0, f"Overlap train-test: {overlaps['train_test']}"
    assert len(overlaps["val_test"]) == 0, f"Overlap val-test: {overlaps['val_test']}"


def test_zero_leakage_md5_hash():
    """Verify that identical images (matching MD5 hashes) do not leak across splits."""
    df_splits = pd.read_csv(SPLITS_CSV)

    train_hashes = set(df_splits[df_splits["split"] == "train"]["md5_hash"])
    val_hashes = set(df_splits[df_splits["split"] == "val"]["md5_hash"])
    test_hashes = set(df_splits[df_splits["split"] == "test"]["md5_hash"])

    assert len(train_hashes & val_hashes) == 0, "MD5 hash collision between train and val"
    assert len(train_hashes & test_hashes) == 0, "MD5 hash collision between train and test"
    assert len(val_hashes & test_hashes) == 0, "MD5 hash collision between val and test"


def test_yolo_dataset_structure_and_alignment():
    """Verify Ultralytics directory tree and 1-to-1 match between images and labels."""
    assert YOLO_DIR.exists(), f"Missing {YOLO_DIR}"

    for split in ["train", "val", "test"]:
        img_dir = YOLO_DIR / "images" / split
        lbl_dir = YOLO_DIR / "labels" / split

        assert img_dir.exists(), f"Missing {img_dir}"
        assert lbl_dir.exists(), f"Missing {lbl_dir}"

        img_stems = {f.stem for f in img_dir.glob("*.*")}
        lbl_stems = {f.stem for f in lbl_dir.glob("*.txt")}

        assert len(img_stems) > 0, f"No images in {split}"
        assert img_stems == lbl_stems, (
            f"Mismatch between images and labels in split '{split}': "
            f"{len(img_stems - lbl_stems)} missing labels, {len(lbl_stems - img_stems)} missing images"
        )


def test_yolo_bounding_box_coordinates_validity():
    """Verify that all bounding boxes have valid class IDs and coordinates in [0.0, 1.0]."""
    for split in ["train", "val", "test"]:
        lbl_dir = YOLO_DIR / "labels" / split
        for lbl_file in lbl_dir.glob("*.txt"):
            is_valid, errors, boxes = validate_yolo_label_file(
                lbl_file, num_classes=len(CLASS_NAMES_DEFAULT)
            )
            assert is_valid, f"Invalid label file {lbl_file.name}: {errors}"

            for b in boxes:
                # Class id check
                assert 0 <= b["class_id"] < 9, f"class_id {b['class_id']} out of bounds in {lbl_file.name}"
                # Normalized coordinates check [0.0, 1.0]
                assert 0.0 <= b["x_center"] <= 1.0, f"x_center out of bounds in {lbl_file.name}"
                assert 0.0 <= b["y_center"] <= 1.0, f"y_center out of bounds in {lbl_file.name}"
                assert 0.0 < b["width"] <= 1.0, f"width invalid in {lbl_file.name}: {b['width']}"
                assert 0.0 < b["height"] <= 1.0, f"height invalid in {lbl_file.name}: {b['height']}"


def test_yolo_yaml_config():
    """Verify data.yaml exists and correctly defines 9 classes."""
    yaml_path = YOLO_DIR / "data.yaml"
    assert yaml_path.exists(), f"Missing {yaml_path}"

    with open(yaml_path, "r", encoding="utf-8") as f:
        data_cfg = yaml.safe_load(f)

    assert data_cfg["nc"] == 9, f"Expected 9 classes, got {data_cfg['nc']}"
    assert len(data_cfg["names"]) == 9, f"Expected 9 class names, got {len(data_cfg['names'])}"
    assert data_cfg["names"] == CLASS_NAMES_DEFAULT, "Class names mismatch"
