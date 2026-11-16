"""Reproducible YOLOv11 Train/Val/Test Split Script with Zero Data Leakage.

Author: Nguyễn Đức Thịnh (22696331) - Member A
Stack: Stack01 (Stack109, Stack110, Stack111)

Responsibilities:
- Perform Grouped Stratified Split (70/15/15) grouped by physical note_id.
- Guarantee strict Zero Data Leakage (zero note_id overlap, zero md5 hash overlap).
- Export split manifest to data/metadata/splits.csv.
- Export standard Ultralytics YOLO dataset to data/processed/yolo_dataset/ with data.yaml.
"""

import argparse
import sys
from pathlib import Path
import pandas as pd

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Ensure UTF-8 stdout
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
if sys.stderr.encoding and sys.stderr.encoding.lower() != "utf-8":
    sys.stderr.reconfigure(encoding="utf-8")

from src.vndcash.data.split import (
    CLASS_NAMES_DEFAULT,
    stratified_group_split_yolo,
    verify_zero_leakage,
    export_ultralytics_dataset,
)


def create_split(
    metadata_path: Path,
    splits_csv_path: Path,
    yolo_dataset_dir: Path,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    export_yolo: bool = True,
) -> pd.DataFrame:
    """Executes anti-leakage grouped split and exports Ultralytics dataset."""
    metadata_path = Path(metadata_path)
    splits_csv_path = Path(splits_csv_path)
    yolo_dataset_dir = Path(yolo_dataset_dir)

    print(f"Loading metadata from {metadata_path}...")
    df_meta = pd.read_csv(metadata_path)
    print(f"Loaded {len(df_meta)} records across {df_meta['note_id'].nunique()} unique note_ids.")

    # Grouped split
    df_splits = stratified_group_split_yolo(
        df=df_meta,
        group_col="note_id",
        class_col="primary_class",
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )

    # Verification 1: Note ID leakage check
    is_leak_free, note_overlaps = verify_zero_leakage(df_splits, group_col="note_id")
    if not is_leak_free:
        raise RuntimeError(f"DATA LEAKAGE DETECTED! Overlapping note_ids: {note_overlaps}")

    # Verification 2: MD5 hash leakage check
    train_hashes = set(df_splits[df_splits["split"] == "train"]["md5_hash"])
    val_hashes = set(df_splits[df_splits["split"] == "val"]["md5_hash"])
    test_hashes = set(df_splits[df_splits["split"] == "test"]["md5_hash"])

    hash_overlap_tv = train_hashes & val_hashes
    hash_overlap_tt = train_hashes & test_hashes
    hash_overlap_vt = val_hashes & test_hashes

    if hash_overlap_tv or hash_overlap_tt or hash_overlap_vt:
        raise RuntimeError(
            f"MD5 HASH LEAKAGE DETECTED! Overlaps: "
            f"Train-Val={len(hash_overlap_tv)}, Train-Test={len(hash_overlap_tt)}, Val-Test={len(hash_overlap_vt)}"
        )

    print("Zero Data Leakage verification PASSED:")
    print("  - Disjoint note_id check: OK (0 overlapping physical note groups)")
    print("  - Disjoint MD5 hash check: OK (0 duplicate images across splits)")

    # Save splits.csv
    splits_csv_path.parent.mkdir(parents=True, exist_ok=True)
    df_splits.to_csv(splits_csv_path, index=False, encoding="utf-8")
    print(f"Splits manifest saved to: {splits_csv_path}")

    # Summary table
    print("\n" + "=" * 65)
    print("           BẢNG PHÂN CHIA DỮ LIỆU CHUẨN ZERO LEAKAGE           ")
    print("=" * 65)
    split_summary = df_splits.groupby("split").agg(
        num_images=("image_id", "count"),
        num_unique_notes=("note_id", "nunique"),
        total_banknotes=("num_notes", "sum"),
        hard_cases=("is_hard_case", "sum"),
    ).reset_index()
    split_summary["pct_images"] = (split_summary["num_images"] / len(df_splits) * 100).round(1)
    split_summary["pct_notes"] = (split_summary["num_unique_notes"] / df_splits["note_id"].nunique() * 100).round(1)
    print(split_summary.to_string(index=False))
    print("=" * 65)

    # Export Ultralytics dataset
    if export_yolo:
        print(f"\nExporting Ultralytics dataset to {yolo_dataset_dir}...")
        yaml_path = export_ultralytics_dataset(
            splits_df=df_splits,
            project_root=PROJECT_ROOT,
            output_dir=yolo_dataset_dir,
            class_names=CLASS_NAMES_DEFAULT,
        )
        print(f"Ultralytics data.yaml created at: {yaml_path}")
        print("Dataset ready for YOLOv11 training!")

    return df_splits


def main():
    parser = argparse.ArgumentParser(description="Create anti-leakage train/val/test splits.")
    parser.add_argument(
        "--metadata",
        type=str,
        default="data/metadata/metadata.csv",
        help="Path to metadata.csv"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/metadata/splits.csv",
        help="Path to save splits.csv"
    )
    parser.add_argument(
        "--yolo_dir",
        type=str,
        default="data/processed/yolo_dataset",
        help="Path to output Ultralytics YOLO dataset"
    )
    parser.add_argument("--train_ratio", type=float, default=0.70)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--test_ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip_export", action="store_true", help="Skip copying images to yolo_dir")
    args = parser.parse_args()

    create_split(
        metadata_path=PROJECT_ROOT / args.metadata,
        splits_csv_path=PROJECT_ROOT / args.output,
        yolo_dataset_dir=PROJECT_ROOT / args.yolo_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        export_yolo=not args.skip_export,
    )


if __name__ == "__main__":
    main()
