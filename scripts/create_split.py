"""Script to generate reproducible train/val/test splits without data leakage.

Responsibilities (Member A):
- Group split by note_id to guarantee zero data leakage between train/val/test.
- Produce data/metadata/splits.csv.
- Verify zero group overlap and zero hash overlap.
- Report actual split ratio vs target ratio for each denomination.
"""

import argparse
import os
import shutil
import sys
from pathlib import Path

import pandas as pd

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Ensure UTF-8 stdout
if sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
if sys.stderr.encoding.lower() != "utf-8":
    sys.stderr.reconfigure(encoding="utf-8")

from src.data.split import stratified_group_split, verify_no_leakage
from src.utils.config import load_config
from src.utils.logger import setup_logger
from src.utils.paths import (
    DATA_CONFIG_PATH,
    METADATA_CSV,
    PROCESSED_DATA_DIR,
    SPLITS_CSV,
)


def export_physical_splits(df_splits: pd.DataFrame, processed_dir: Path):
    """Copies images into processed/{split}/{denomination} if configured."""
    logger = setup_logger("export_physical_splits")
    logger.info(f"Copying images to physical split folders in {processed_dir}...")
    for _, row in df_splits.iterrows():
        src_path = PROJECT_ROOT / row["rel_path"]
        dst_dir = processed_dir / row["split"] / str(row["denomination"])
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst_path = dst_dir / row["filename"]
        if not dst_path.exists():
            shutil.copy2(src_path, dst_path)


def main():
    parser = argparse.ArgumentParser(description="Generate data splits for VNDCash.")
    parser.add_argument("--config", type=str, default=str(DATA_CONFIG_PATH), help="Path to data config")
    parser.add_argument("--metadata", type=str, default=str(METADATA_CSV), help="Path to metadata.csv")
    parser.add_argument("--output", type=str, default=str(SPLITS_CSV), help="Path to save splits.csv")
    parser.add_argument("--save_physical", action="store_true", help="Copy images to data/processed/{split}/{class}")
    args = parser.parse_args()

    logger = setup_logger("create_split")
    config = load_config(args.config)

    split_cfg = config.get("split", {})
    train_ratio = split_cfg.get("train_ratio", 0.70)
    val_ratio = split_cfg.get("val_ratio", 0.15)
    test_ratio = split_cfg.get("test_ratio", 0.15)
    seed = split_cfg.get("seed", 42)
    group_col = split_cfg.get("group_by", "note_id")
    save_physical = args.save_physical or split_cfg.get("save_physical_splits", False)

    metadata_path = Path(args.metadata)
    if not metadata_path.exists():
        logger.error(f"Metadata file {metadata_path} not found. Please run build_metadata.py first!")
        sys.exit(1)

    df_meta = pd.read_csv(metadata_path)
    logger.info(f"Loaded {len(df_meta)} records from {metadata_path}")

    # Perform stratified group split
    df_splits = stratified_group_split(
        df=df_meta,
        class_col="denomination",
        group_col=group_col,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed
    )

    # Verification: Group leakage check
    is_leak_free, overlaps = verify_no_leakage(df_splits, group_col=group_col)
    if not is_leak_free:
        logger.error(f"DATA LEAKAGE DETECTED! Overlapping note_ids: {overlaps}")
        sys.exit(1)

    # Verification: MD5 hash leakage check
    train_hashes = set(df_splits[df_splits["split"] == "train"]["md5_hash"])
    val_hashes = set(df_splits[df_splits["split"] == "val"]["md5_hash"])
    test_hashes = set(df_splits[df_splits["split"] == "test"]["md5_hash"])
    hash_leak = (train_hashes & val_hashes) | (train_hashes & test_hashes) | (val_hashes & test_hashes)

    if hash_leak:
        logger.error(f"DATA LEAKAGE DETECTED! Identical image hashes across splits: {len(hash_leak)}")
        sys.exit(1)

    # Save splits.csv
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_splits.to_csv(output_path, index=False, encoding="utf-8")
    logger.info(f"Splits saved to {output_path}")

    if save_physical:
        export_physical_splits(df_splits, PROCESSED_DATA_DIR)

    # Print summary report
    print("\n" + "=" * 75)
    print("                KẾT QUẢ PHÂN CHIA TẬP DỮ LIỆU (SPLIT REPORT)             ")
    print("=" * 75)
    print(f"Cấu hình mục tiêu: Train={train_ratio*100:.0f}%, Val={val_ratio*100:.0f}%, Test={test_ratio*100:.0f}%")
    print(f"Random Seed      : {seed}")
    print(f"Grouped by       : {group_col} (Kiểm soát rò rỉ dữ liệu theo tờ tiền)")
    print(f"Kiểm tra rò rỉ   : [PASSED] 0 note_id trùng lặp, 0 hash trùng lặp giữa các tập")
    print("-" * 75)

    total_images = len(df_splits)
    train_count = len(df_splits[df_splits["split"] == "train"])
    val_count = len(df_splits[df_splits["split"] == "val"])
    test_count = len(df_splits[df_splits["split"] == "test"])

    print("TỔNG HỢP THEO TẬP:")
    print(f"  - Tập Train : {train_count:4d} ảnh ({train_count/total_images*100:5.2f}%) | {df_splits[df_splits['split']=='train']['note_id'].nunique()} notes")
    print(f"  - Tập Val   : {val_count:4d} ảnh ({val_count/total_images*100:5.2f}%) | {df_splits[df_splits['split']=='val']['note_id'].nunique()} notes")
    print(f"  - Tập Test  : {test_count:4d} ảnh ({test_count/total_images*100:5.2f}%) | {df_splits[df_splits['split']=='test']['note_id'].nunique()} notes")
    print(f"  - Tổng cộng : {total_images:4d} ảnh (100.00%) | {df_splits['note_id'].nunique()} notes")
    print("-" * 75)

    print("CHI TIẾT PHÂN BỐ TỪNG MỆNH GIÁ (SỐ ẢNH & SỐ TỜ TIỀN):")
    summary = df_splits.pivot_table(
        index=["denomination"],
        columns="split",
        values="image_id",
        aggfunc="count",
        fill_value=0
    )[["train", "val", "test"]]

    summary["total"] = summary.sum(axis=1)
    summary["train_%"] = (summary["train"] / summary["total"] * 100).round(1)
    summary["val_%"] = (summary["val"] / summary["total"] * 100).round(1)
    summary["test_%"] = (summary["test"] / summary["total"] * 100).round(1)

    print(summary.to_string())
    print("=" * 75)


if __name__ == "__main__":
    main()
