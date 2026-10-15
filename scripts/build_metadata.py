"""Build standardized metadata.csv for VNDCash dataset.

Responsibilities (Member A):
- Scan raw dataset and map folders to 9 target denominations.
- Group photos into note_id to guarantee anti-leakage grouped splits.
- Tag denomination, material (cotton/polymer), dimensions, aspect ratio, hash.
- Export to data/metadata/metadata.csv with clean, validated schema.
"""

import argparse
import hashlib
import os
import re
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd
from PIL import Image

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Ensure UTF-8 stdout
if sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
if sys.stderr.encoding.lower() != "utf-8":
    sys.stderr.reconfigure(encoding="utf-8")

from src.utils.config import load_config
from src.utils.logger import setup_logger
from src.utils.paths import DATA_CONFIG_PATH, METADATA_CSV, METADATA_DIR, RAW_DATA_DIR


def compute_md5(file_path: Path) -> str:
    hasher = hashlib.md5()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def extract_index_from_filename(filename: str) -> int:
    """Extracts integer index from filenames like '001000_12.png' -> 12."""
    match = re.search(r"_(\d+)\.", filename)
    if match:
        return int(match.group(1))
    return -1


def build_metadata(
    raw_dir: Path,
    config: dict,
    output_path: Path,
    photos_per_note: int = 10
) -> pd.DataFrame:
    logger = setup_logger("build_metadata")
    logger.info("Building metadata from raw dataset...")

    folder_mapping: Dict[str, str] = config["classes"]["folder_mapping"]
    label_to_idx: Dict[str, int] = config["classes"]["label_to_idx"]

    # Cotton: 1k, 2k, 5k | Polymer: 10k, 20k, 50k, 100k, 200k, 500k
    cotton_denominations = {"1000", "2000", "5000"}

    records: List[Dict] = []
    skipped_count = 0

    for folder_name, denomination in folder_mapping.items():
        folder_path = raw_dir / folder_name
        if not folder_path.exists():
            logger.warning(f"Thư mục {folder_path} không tồn tại, bỏ qua!")
            continue

        label_idx = label_to_idx[denomination]
        material = "cotton" if denomination in cotton_denominations else "polymer"

        image_files = sorted(
            [f for f in folder_path.iterdir() if f.suffix.lower() in [".png", ".jpg", ".jpeg"]],
            key=lambda x: extract_index_from_filename(x.name) if extract_index_from_filename(x.name) >= 0 else x.name
        )

        for img_idx, img_file in enumerate(image_files):
            try:
                file_size = img_file.stat().st_size
                with Image.open(img_file) as im:
                    width, height = im.size
                    mode = im.mode
                    aspect_ratio = round(width / max(height, 1), 3)

                img_seq_idx = extract_index_from_filename(img_file.name)
                seq_val = img_seq_idx if img_seq_idx >= 0 else img_idx
                # Group every N consecutive photos to one distinct physical banknote
                note_num = seq_val // photos_per_note
                note_id = f"NOTE_{denomination}_{note_num:03d}"
                image_id = f"{denomination}_{seq_val:04d}"

                md5_hash = compute_md5(img_file)

                # Store relative path for cross-platform portability
                rel_path = img_file.relative_to(PROJECT_ROOT).as_posix()

                records.append({
                    "image_id": image_id,
                    "filename": img_file.name,
                    "rel_path": rel_path,
                    "denomination": denomination,
                    "label_idx": label_idx,
                    "material": material,
                    "note_id": note_id,
                    "width": width,
                    "height": height,
                    "channels": 3 if mode == "RGB" else (1 if mode == "L" else 4),
                    "color_mode": mode,
                    "aspect_ratio": aspect_ratio,
                    "file_size_bytes": file_size,
                    "md5_hash": md5_hash,
                    "is_valid": True
                })

            except Exception as e:
                logger.error(f"Lỗi khi đọc file {img_file}: {e}")
                skipped_count += 1

    df = pd.DataFrame(records)

    # Validate schema
    expected_cols = [
        "image_id", "filename", "rel_path", "denomination", "label_idx",
        "material", "note_id", "width", "height", "channels", "color_mode",
        "aspect_ratio", "file_size_bytes", "md5_hash", "is_valid"
    ]
    df = df[expected_cols]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8")

    logger.info(f"Đã tạo metadata thành công tại {output_path}")
    logger.info(f"Tổng số ảnh hợp lệ: {len(df)} | Số ảnh bị lỗi: {skipped_count}")
    logger.info(f"Tổng số tờ tiền vật lý (note_id duy nhất): {df['note_id'].nunique()}")

    return df


def main():
    parser = argparse.ArgumentParser(description="Build metadata.csv for VNDCash.")
    parser.add_argument("--config", type=str, default=str(DATA_CONFIG_PATH), help="Path to data config")
    parser.add_argument("--raw_dir", type=str, default=str(RAW_DATA_DIR), help="Path to raw data directory")
    parser.add_argument("--output", type=str, default=str(METADATA_CSV), help="Output metadata.csv path")
    parser.add_argument("--photos_per_note", type=int, default=10, help="Number of photos per physical note_id")
    args = parser.parse_args()

    config = load_config(args.config)
    df = build_metadata(
        raw_dir=Path(args.raw_dir),
        config=config,
        output_path=Path(args.output),
        photos_per_note=args.photos_per_note
    )

    print("\n" + "=" * 60)
    print("           BẢNG THỐNG KÊ METADATA THEO MỆNH GIÁ           ")
    print("=" * 60)
    summary = df.groupby(["denomination", "material"]).agg(
        num_images=("image_id", "count"),
        num_unique_notes=("note_id", "nunique"),
        avg_width=("width", "mean"),
        avg_height=("height", "mean")
    ).reset_index()
    # Sort by denomination integer value
    summary["denom_int"] = summary["denomination"].astype(int)
    summary = summary.sort_values("denom_int").drop(columns=["denom_int"])

    print(summary.to_string(index=False))
    print("=" * 60)


if __name__ == "__main__":
    main()
