"""Audit and inspect raw dataset for Vietnamese Banknote Recognition.

Responsibilities (Member A):
- Audit raw dataset folders, class counts, corrupted images, duplicates, blank images.
- Highlight underrepresented classes (e.g., 5.000, 20.000, 50.000).
- Generate a summary report to console and save structured audit report.
"""

import argparse
import hashlib
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Ensure UTF-8 stdout on Windows
if sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
if sys.stderr.encoding.lower() != "utf-8":
    sys.stderr.reconfigure(encoding="utf-8")

from src.utils.config import load_config
from src.utils.logger import setup_logger
from src.utils.paths import ARTIFACTS_DIR, DATA_CONFIG_PATH, RAW_DATA_DIR, REPORTS_DIR


def calculate_md5(file_path: Path) -> str:
    """Computes MD5 hash of an image file."""
    hasher = hashlib.md5()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def is_blank_or_corrupt(file_path: Path) -> Tuple[bool, bool, Tuple[int, int], str]:
    """Checks if an image is corrupted or nearly blank (variance close to zero).

    Returns:
        is_corrupt, is_blank, (width, height), mode
    """
    try:
        with Image.open(file_path) as img:
            img.verify()
        # Reopen to load pixel data after verify()
        with Image.open(file_path) as img:
            size = img.size
            mode = img.mode
            arr = np.array(img.convert("RGB"))
            is_blank = float(np.var(arr)) < 1.0
            return False, is_blank, size, mode
    except Exception:
        return True, False, (0, 0), "UNKNOWN"


def audit_dataset(raw_dir: Path, config: dict) -> dict:
    logger = setup_logger("inspect_dataset")
    logger.info(f"Auditing raw data directory: {raw_dir}")

    target_folders = config.get("classes", {}).get("folder_mapping", {})
    excluded_folders = config.get("classes", {}).get("excluded_folders", [])

    results = {
        "raw_dir": str(raw_dir),
        "target_classes_stats": {},
        "excluded_classes_stats": {},
        "corrupted_images": [],
        "blank_images": [],
        "duplicates": [],
        "total_target_images": 0,
        "total_excluded_images": 0,
        "resolutions": defaultdict(int),
        "color_modes": defaultdict(int),
    }

    hash_to_paths = defaultdict(list)

    if not raw_dir.exists():
        logger.error(f"Directory {raw_dir} does not exist!")
        return results

    all_subdirs = sorted([d for d in raw_dir.iterdir() if d.is_dir()])

    for subdir in all_subdirs:
        folder_name = subdir.name
        is_target = folder_name in target_folders
        mapped_class = target_folders.get(folder_name, folder_name)

        image_files = [
            f for f in subdir.iterdir()
            if f.suffix.lower() in [".png", ".jpg", ".jpeg", ".bmp", ".webp"]
        ]

        class_stat = {
            "folder": folder_name,
            "mapped_label": mapped_class,
            "count": len(image_files),
            "corrupt_count": 0,
            "blank_count": 0,
            "duplicate_count": 0,
        }

        for img_path in image_files:
            # Check corrupt & blank
            is_corrupt, is_blank, size, mode = is_blank_or_corrupt(img_path)
            if is_corrupt:
                class_stat["corrupt_count"] += 1
                results["corrupted_images"].append(str(img_path))
                continue
            if is_blank:
                class_stat["blank_count"] += 1
                results["blank_images"].append(str(img_path))

            results["resolutions"][f"{size[0]}x{size[1]}"] += 1
            results["color_modes"][mode] += 1

            # Check duplicate hash
            file_hash = calculate_md5(img_path)
            hash_to_paths[file_hash].append(str(img_path))

        if is_target:
            results["target_classes_stats"][mapped_class] = class_stat
            results["total_target_images"] += len(image_files)
        else:
            results["excluded_classes_stats"][folder_name] = class_stat
            results["total_excluded_images"] += len(image_files)

    # Compile duplicate list
    for file_hash, paths in hash_to_paths.items():
        if len(paths) > 1:
            results["duplicates"].append({"hash": file_hash, "paths": paths})
            for p in paths:
                # Mark in stats
                pass

    return results


def print_audit_report(results: dict):
    print("=" * 70)
    print("           AUDIT BỘ DỮ LIỆU TIỀN GIẤY VIỆT NAM (VNDCash)          ")
    print("=" * 70)
    print(f"Thư mục nguồn : {results['raw_dir']}")
    print(f"Tổng số ảnh 9 mệnh giá mục tiêu : {results['total_target_images']}")
    print(f"Tổng số ảnh ngoài phạm vi (excluded) : {results['total_excluded_images']}")
    print(f"Số ảnh lỗi (corrupted)        : {len(results['corrupted_images'])}")
    print(f"Số ảnh rỗng/trắng (blank)      : {len(results['blank_images'])}")
    print(f"Số nhóm ảnh trùng lặp (dups)   : {len(results['duplicates'])}")
    print("-" * 70)
    print("THỐNG KÊ CHI TIẾT 09 MỆNH GIÁ MỤC TIÊU:")
    print(f"{'Mệnh giá':<12} | {'Thư mục':<10} | {'Số lượng':<10} | {'Tình trạng':<25}")
    print("-" * 70)

    target_stats = results.get("target_classes_stats", {})
    counts = [s["count"] for s in target_stats.values()]
    max_count = max(counts) if counts else 0

    for mapped_class, stat in target_stats.items():
        count = stat["count"]
        if count < max_count:
            diff = max_count - count
            status = f"Thiếu mẫu (-{diff} ảnh, {(diff/max_count)*100:.1f}%)"
        else:
            status = "Đủ mẫu chuẩn"
        print(f"{mapped_class + ' VND':<12} | {stat['folder']:<10} | {count:<10} | {status:<25}")

    print("-" * 70)
    print("THỐNG KÊ CÁC LỚP NGOÀI PHẠM VI (EXCLUDED):")
    for folder, stat in results.get("excluded_classes_stats", {}).items():
        print(f"Folder {folder:<10} : {stat['count']} ảnh (Background/Mệnh giá ngoài 9 lớp)")

    print("-" * 70)
    print("PHÂN BỐ ĐỘ PHÂN GIẢI & MODE:")
    for res, count in results.get("resolutions", {}).items():
        print(f"  - Kích thước {res}: {count} ảnh")
    for mode, count in results.get("color_modes", {}).items():
        print(f"  - Hệ màu {mode}: {count} ảnh")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Audit and inspect dataset files.")
    parser.add_argument("--config", type=str, default=str(DATA_CONFIG_PATH), help="Path to data config")
    parser.add_argument("--raw_dir", type=str, default=str(RAW_DATA_DIR), help="Path to raw data directory")
    parser.add_argument("--save_report", action="store_true", default=True, help="Save report to artifacts/reports")
    args = parser.parse_args()

    config = load_config(args.config)
    results = audit_dataset(Path(args.raw_dir), config)
    print_audit_report(results)

    if args.save_report:
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        report_file = REPORTS_DIR / "data_audit_report.json"
        # Convert defaultdicts
        clean_results = dict(results)
        clean_results["resolutions"] = dict(results["resolutions"])
        clean_results["color_modes"] = dict(results["color_modes"])
        with open(report_file, "w", encoding="utf-8") as f:
            json.dump(clean_results, f, indent=2, ensure_ascii=False)
        print(f"Đã lưu báo cáo audit chi tiết tại: {report_file}")


if __name__ == "__main__":
    main()
