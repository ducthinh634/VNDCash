"""Build standardized metadata.csv for VNDCash YOLOv11 dataset.

Responsibilities (Member A - Nguyễn Đức Thịnh, MSSV: 22696331):
- Scan multi-note and raw dataset images.
- Group photos by physical base scene (note_id) to guarantee strict Zero Data Leakage.
- Extract bounding box annotations, unique denominations, bounding box counts.
- Identify Hard-Case samples (multi-denomination, dense overlap, extreme aspect ratio, small bounding boxes).
- Compute image dimensions, aspect ratio, file size, MD5 hash.
- Export to data/metadata/metadata.csv conforming to Stack 01 specifications (Stack107 & Stack108).
"""

import argparse
import hashlib
import sys
from pathlib import Path
from typing import Dict, List, Set, Any
import pandas as pd
from PIL import Image

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Ensure UTF-8 stdout
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
if sys.stderr.encoding and sys.stderr.encoding.lower() != "utf-8":
    sys.stderr.reconfigure(encoding="utf-8")

from src.vndcash.data.validation import CLASS_NAMES_DEFAULT, validate_yolo_label_file


def compute_md5(file_path: Path) -> str:
    """Computes MD5 hash of a file."""
    hasher = hashlib.md5()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def extract_note_id(filename: str) -> str:
    """Extracts physical note / scene identifier from filename.

    Roboflow filenames have pattern: <base_name>.rf.<hash>.jpg
    By extracting base_name, all augmented versions of the same physical scene
    share the same note_id, preventing data leakage.
    """
    if ".rf." in filename:
        return filename.split(".rf.")[0]
    stem = Path(filename).stem
    return stem


def build_metadata(
    images_dir: Path,
    labels_dir: Path,
    output_path: Path,
    class_names: List[str] = CLASS_NAMES_DEFAULT,
) -> pd.DataFrame:
    """Builds metadata.csv from images and YOLO label files."""
    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir)
    output_path = Path(output_path)

    print(f"Scanning images from: {images_dir}")
    print(f"Scanning labels from: {labels_dir}")

    image_files = sorted(
        [f for f in images_dir.iterdir() if f.suffix.lower() in [".jpg", ".jpeg", ".png"]]
    )
    print(f"Found {len(image_files)} image files.")

    records: List[Dict[str, Any]] = []
    skipped = 0

    for img_file in image_files:
        stem = img_file.stem
        lbl_file = labels_dir / f"{stem}.txt"

        if not lbl_file.exists():
            print(f"Warning: Missing label for {img_file.name}, skipping.")
            skipped += 1
            continue

        # Validate label file and extract bounding boxes
        is_valid, errors, boxes = validate_yolo_label_file(
            lbl_file, num_classes=len(class_names)
        )
        if not is_valid:
            print(f"Warning: Invalid label file {lbl_file.name}: {errors}")

        # Image metrics
        try:
            file_size = img_file.stat().st_size
            with Image.open(img_file) as im:
                width, height = im.size
                mode = im.mode
            aspect_ratio = round(width / max(height, 1), 3)
            md5 = compute_md5(img_file)
        except Exception as e:
            print(f"Error reading image {img_file.name}: {e}")
            skipped += 1
            continue

        # Denomination stats
        class_ids = [b["class_id"] for b in boxes]
        denominations_list = [class_names[cid] for cid in class_ids if cid < len(class_names)]
        unique_denoms = sorted(list(set(denominations_list)), key=lambda x: int(x))
        denominations_str = ";".join(unique_denoms) if unique_denoms else "none"
        primary_class = unique_denoms[0] if unique_denoms else "none"

        num_notes = len(boxes)

        # Hard-case criteria:
        # 1. Multi-note image (num_notes >= 2)
        # 2. Contains multiple different denominations (len(unique_denoms) >= 2)
        # 3. Contains very small bounding boxes (width < 0.15 or height < 0.15)
        has_small_box = any(b["width"] < 0.15 or b["height"] < 0.15 for b in boxes)
        is_multi_denom = len(unique_denoms) >= 2
        is_hard_case = (num_notes >= 2) or is_multi_denom or has_small_box

        note_id = extract_note_id(img_file.name)
        image_id = stem

        rel_img_path = img_file.relative_to(PROJECT_ROOT).as_posix()
        rel_lbl_path = lbl_file.relative_to(PROJECT_ROOT).as_posix()

        records.append({
            "image_id": image_id,
            "filename": img_file.name,
            "image_path": rel_img_path,
            "label_path": rel_lbl_path,
            "note_id": note_id,
            "primary_class": primary_class,
            "denominations": denominations_str,
            "num_notes": num_notes,
            "is_multi_denom": is_multi_denom,
            "is_hard_case": is_hard_case,
            "width": width,
            "height": height,
            "aspect_ratio": aspect_ratio,
            "file_size_bytes": file_size,
            "md5_hash": md5,
            "is_valid": is_valid
        })

    df = pd.DataFrame(records)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8")

    print(f"\nMetadata successfully saved to: {output_path}")
    print(f"Total processed records: {len(df)}")
    print(f"Unique note_id groups: {df['note_id'].nunique()}")
    print(f"Hard-case images: {df['is_hard_case'].sum()} ({df['is_hard_case'].mean() * 100:.1f}%)")
    print(f"Multi-denomination images: {df['is_multi_denom'].sum()} ({df['is_multi_denom'].mean() * 100:.1f}%)")
    print(f"Total banknotes annotated: {df['num_notes'].sum()}")

    return df


def main():
    parser = argparse.ArgumentParser(description="Build metadata.csv for VNDCash YOLOv11 dataset.")
    parser.add_argument(
        "--images_dir",
        type=str,
        default="data/raw/multi_note/images",
        help="Path to images directory"
    )
    parser.add_argument(
        "--labels_dir",
        type=str,
        default="data/raw/multi_note/labels",
        help="Path to labels directory"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/metadata/metadata.csv",
        help="Output CSV path"
    )
    args = parser.parse_args()

    build_metadata(
        images_dir=PROJECT_ROOT / args.images_dir,
        labels_dir=PROJECT_ROOT / args.labels_dir,
        output_path=PROJECT_ROOT / args.output,
    )


if __name__ == "__main__":
    main()
