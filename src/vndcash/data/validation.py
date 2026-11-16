"""YOLOv11 Bounding Box and Dataset Validation Module for VNDCash.

Stack: Stack01 (Stack105)
Author: Nguyễn Đức Thịnh (22696331)

Responsibilities:
- Validate YOLO format bounding box coordinates [0.0, 1.0].
- Check class_id boundaries within [0, 8].
- Detect degenerate boxes (zero width/height, out of bounds).
- Enforce strict 1-to-1 matching between images and label files.
"""

from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from PIL import Image


CLASS_NAMES_DEFAULT = [
    "1000", "10000", "100000", "2000", "20000", "200000", "5000", "50000", "500000"
]


def validate_yolo_label_file(
    label_path: Path,
    num_classes: int = 9,
    fix_minor_overflow: bool = False
) -> Tuple[bool, List[str], List[Dict[str, Any]]]:
    """Validates a single YOLO format .txt label file.

    Format: <class_id> <x_center> <y_center> <width> <height> (all normalized in [0, 1])

    Returns:
        (is_valid, list_of_errors, list_of_valid_boxes)
    """
    label_path = Path(label_path)
    if not label_path.exists():
        return False, [f"Label file not found: {label_path}"], []

    errors: List[str] = []
    boxes: List[Dict[str, Any]] = []

    try:
        content = label_path.read_text(encoding="utf-8").strip()
    except Exception as e:
        return False, [f"Failed to read file {label_path}: {e}"], []

    if not content:
        # Empty label file can be valid for background images (zero notes)
        return True, [], []

    lines = content.splitlines()
    for line_idx, line in enumerate(lines, 1):
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 5:
            errors.append(f"Line {line_idx}: Expected at least 5 tokens, got {len(parts)}: '{line}'")
            continue

        try:
            class_id = int(parts[0])
            xc, yc, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
        except ValueError as e:
            errors.append(f"Line {line_idx}: Value parsing error: {e}")
            continue

        if class_id < 0 or class_id >= num_classes:
            errors.append(f"Line {line_idx}: class_id {class_id} outside valid range [0, {num_classes - 1}]")

        if not (0.0 <= xc <= 1.0 and 0.0 <= yc <= 1.0):
            errors.append(f"Line {line_idx}: Center coords out of bounds: xc={xc}, yc={yc}")

        if w <= 0.0 or h <= 0.0:
            errors.append(f"Line {line_idx}: Non-positive dimension: width={w}, height={h}")

        if w > 1.0 or h > 1.0:
            errors.append(f"Line {line_idx}: Dimension exceeds 1.0: width={w}, height={h}")

        xmin = xc - w / 2.0
        ymin = yc - h / 2.0
        xmax = xc + w / 2.0
        ymax = yc + h / 2.0

        boxes.append({
            "class_id": class_id,
            "x_center": xc,
            "y_center": yc,
            "width": w,
            "height": h,
            "x_min": max(0.0, xmin),
            "y_min": max(0.0, ymin),
            "x_max": min(1.0, xmax),
            "y_max": min(1.0, ymax)
        })

    is_valid = len(errors) == 0
    return is_valid, errors, boxes


def validate_image_file(image_path: Path) -> Tuple[bool, Optional[str], Tuple[int, int]]:
    """Checks whether an image file is readable and non-empty.

    Returns:
        (is_valid, error_msg, (width, height))
    """
    image_path = Path(image_path)
    if not image_path.exists():
        return False, f"Image not found: {image_path}", (0, 0)
    try:
        with Image.open(image_path) as img:
            img.verify()
        with Image.open(image_path) as img:
            size = img.size
        return True, None, size
    except Exception as e:
        return False, f"Corrupted image {image_path}: {e}", (0, 0)


def validate_yolo_dataset(
    images_dir: Path,
    labels_dir: Path,
    num_classes: int = 9
) -> Dict[str, Any]:
    """Performs full integrity check on a YOLO directory pair (images & labels)."""
    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir)

    results = {
        "is_valid": True,
        "total_images": 0,
        "total_labels": 0,
        "missing_labels": [],
        "missing_images": [],
        "invalid_label_files": {},
        "corrupted_images": [],
        "total_boxes": 0,
        "per_class_counts": {i: 0 for i in range(num_classes)}
    }

    if not images_dir.exists():
        results["is_valid"] = False
        return results

    image_files = {f.stem: f for f in images_dir.glob("*") if f.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]}
    label_files = {f.stem: f for f in labels_dir.glob("*.txt")} if labels_dir.exists() else {}

    results["total_images"] = len(image_files)
    results["total_labels"] = len(label_files)

    for stem, img_path in image_files.items():
        img_ok, err, _ = validate_image_file(img_path)
        if not img_ok:
            results["corrupted_images"].append(str(img_path))
            results["is_valid"] = False

        if stem not in label_files:
            results["missing_labels"].append(str(img_path))

    for stem, lbl_path in label_files.items():
        if stem not in image_files:
            results["missing_images"].append(str(lbl_path))
            results["is_valid"] = False

        lbl_ok, errs, boxes = validate_yolo_label_file(lbl_path, num_classes=num_classes)
        if not lbl_ok:
            results["invalid_label_files"][str(lbl_path)] = errs
            results["is_valid"] = False

        results["total_boxes"] += len(boxes)
        for b in boxes:
            results["per_class_counts"][b["class_id"]] += 1

    if results["corrupted_images"] or results["missing_images"] or results["invalid_label_files"]:
        results["is_valid"] = False

    return results
