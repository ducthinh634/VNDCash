"""YOLOv11 Dataset Splitting and Ultralytics Formatting Module.

Stack: Stack01 (Stack106)
Author: Nguyễn Đức Thịnh (22696331)

Responsibilities:
- Perform Grouped Stratified Split by note_id to guarantee Zero Data Leakage.
- Copy images and labels into standard Ultralytics YOLO directory tree:
    data/processed/yolo_dataset/
        ├── data.yaml
        ├── images/
        │   ├── train/
        │   ├── val/
        │   └── test/
        └── labels/
            ├── train/
            ├── val/
            └── test/
- Verify disjoint note_ids across train, validation, and test sets.
"""

from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
import shutil
import yaml
import numpy as np
import pandas as pd


CLASS_NAMES_DEFAULT = [
    "1000", "10000", "100000", "2000", "20000", "200000", "5000", "50000", "500000"
]


def verify_zero_leakage(
    df_splits: pd.DataFrame,
    group_col: str = "note_id"
) -> Tuple[bool, Dict[str, Set]]:
    """Verifies that no groups (physical banknote IDs) overlap between splits."""
    train_groups = set(df_splits[df_splits["split"] == "train"][group_col])
    val_groups = set(df_splits[df_splits["split"] == "val"][group_col])
    test_groups = set(df_splits[df_splits["split"] == "test"][group_col])

    overlap_train_val = train_groups & val_groups
    overlap_train_test = train_groups & test_groups
    overlap_val_test = val_groups & test_groups

    overlaps = {
        "train_val": overlap_train_val,
        "train_test": overlap_train_test,
        "val_test": overlap_val_test
    }

    is_leak_free = (len(overlap_train_val) == 0 and
                    len(overlap_train_test) == 0 and
                    len(overlap_val_test) == 0)

    return is_leak_free, overlaps


def stratified_group_split_yolo(
    df: pd.DataFrame,
    group_col: str = "note_id",
    class_col: str = "primary_class",
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42
) -> pd.DataFrame:
    """Allocates samples into train/val/test splits based on note_id grouping.

    Ensures that all photos with the same note_id are strictly in ONE split.
    """
    rng = np.random.default_rng(seed)
    df_out = df.copy()
    split_map: Dict[str, str] = {}

    # Group by primary class (or material/category) to ensure balanced representation
    group_key = class_col if class_col in df.columns else None

    if group_key:
        for _, group_df in df.groupby(group_key):
            unique_notes = sorted(group_df[group_col].unique())
            rng.shuffle(unique_notes)
            n = len(unique_notes)

            n_val = max(1, int(round(n * val_ratio)))
            n_test = max(1, int(round(n * test_ratio)))
            n_train = max(1, n - n_val - n_test)

            train_notes = unique_notes[:n_train]
            val_notes = unique_notes[n_train:n_train + n_val]
            test_notes = unique_notes[n_train + n_val:]

            for note in train_notes:
                split_map[note] = "train"
            for note in val_notes:
                split_map[note] = "val"
            for note in test_notes:
                split_map[note] = "test"
    else:
        unique_notes = sorted(df[group_col].unique())
        rng.shuffle(unique_notes)
        n = len(unique_notes)
        n_val = max(1, int(round(n * val_ratio)))
        n_test = max(1, int(round(n * test_ratio)))
        n_train = n - n_val - n_test

        for note in unique_notes[:n_train]:
            split_map[note] = "train"
        for note in unique_notes[n_train:n_train + n_val]:
            split_map[note] = "val"
        for note in unique_notes[n_train + n_val:]:
            split_map[note] = "test"

    df_out["split"] = df_out[group_col].map(split_map)
    df_out["split"] = df_out["split"].fillna("train")

    is_leak_free, overlaps = verify_zero_leakage(df_out, group_col=group_col)
    if not is_leak_free:
        raise ValueError(f"Data leakage detected across splits: {overlaps}")

    return df_out


def export_ultralytics_dataset(
    splits_df: pd.DataFrame,
    project_root: Path,
    output_dir: Path,
    class_names: Optional[List[str]] = None,
    symlink: bool = False
) -> Path:
    """Exports dataset into standard Ultralytics YOLOv11 directory structure.

    Args:
        splits_df: DataFrame containing columns: image_path, label_path, split.
        project_root: Root directory of the repository.
        output_dir: Target dataset directory (e.g. data/processed/yolo_dataset).
        class_names: List of class names (defaults to 9 VND banknote classes).
        symlink: If True, uses symbolic links instead of copying files.
    """
    output_dir = Path(output_dir)
    class_names = class_names or CLASS_NAMES_DEFAULT

    for split in ["train", "val", "test"]:
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    for _, row in splits_df.iterrows():
        split = row["split"]
        img_src = project_root / row["image_path"]
        lbl_src = project_root / row["label_path"]

        img_dst = output_dir / "images" / split / img_src.name
        lbl_dst = output_dir / "labels" / split / lbl_src.name

        if img_src.exists() and not img_dst.exists():
            shutil.copy2(img_src, img_dst)

        if lbl_src.exists() and not lbl_dst.exists():
            shutil.copy2(lbl_src, lbl_dst)

    # Generate data.yaml configuration for YOLOv11
    yaml_content = {
        "path": output_dir.resolve().as_posix(),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": len(class_names),
        "names": class_names
    }

    yaml_file = output_dir / "data.yaml"
    with open(yaml_file, "w", encoding="utf-8") as f:
        yaml.safe_dump(yaml_content, f, sort_keys=False)

    return yaml_file
