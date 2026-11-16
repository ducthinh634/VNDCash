"""Audit and inspect YOLOv11 dataset for Vietnamese Banknote Recognition.

Author: Nguyễn Đức Thịnh (22696331) - Member A
Stack: Stack01 (Stack112)

Responsibilities:
- Inspect processed YOLO dataset across train/val/test splits.
- Count bounding boxes and occurrences per denomination class.
- Compute Hard-Case ratio, multi-note distribution, and dimension statistics.
- Export full audit report to artifacts/reports/yolo_audit_report.json.
- Provide clean console tables and actionable summary.
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Any
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

from src.vndcash.data.validation import CLASS_NAMES_DEFAULT, validate_yolo_label_file


def audit_yolo_dataset(
    dataset_dir: Path,
    splits_csv: Path,
    report_path: Path,
    class_names: List[str] = CLASS_NAMES_DEFAULT,
) -> Dict[str, Any]:
    dataset_dir = Path(dataset_dir)
    splits_csv = Path(splits_csv)
    report_path = Path(report_path)

    print(f"Auditing YOLO dataset at: {dataset_dir}")
    print(f"Using splits manifest: {splits_csv}")

    df_splits = pd.read_csv(splits_csv) if splits_csv.exists() else None

    splits = ["train", "val", "test"]
    split_stats = {}
    class_counter_total = Counter()
    class_counter_by_split = {s: Counter() for s in splits}
    boxes_per_image_dist = Counter()
    total_images = 0
    total_boxes = 0

    for split in splits:
        img_dir = dataset_dir / "images" / split
        lbl_dir = dataset_dir / "labels" / split

        img_files = list(img_dir.glob("*.*")) if img_dir.exists() else []
        lbl_files = list(lbl_dir.glob("*.txt")) if lbl_dir.exists() else []

        split_boxes_count = 0
        for lbl_path in lbl_files:
            _, _, boxes = validate_yolo_label_file(lbl_path, num_classes=len(class_names))
            n_b = len(boxes)
            boxes_per_image_dist[n_b] += 1
            split_boxes_count += n_b
            for b in boxes:
                c_name = class_names[b["class_id"]]
                class_counter_by_split[split][c_name] += 1
                class_counter_total[c_name] += 1

        total_images += len(img_files)
        total_boxes += split_boxes_count

        split_stats[split] = {
            "num_images": len(img_files),
            "num_labels": len(lbl_files),
            "num_boxes": split_boxes_count,
            "avg_boxes_per_image": round(split_boxes_count / max(len(img_files), 1), 2),
            "classes": dict(class_counter_by_split[split]),
        }

    # Denomination breakdown table
    denom_rows = []
    for c_name in class_names:
        c_train = class_counter_by_split["train"][c_name]
        c_val = class_counter_by_split["val"][c_name]
        c_test = class_counter_by_split["test"][c_name]
        c_tot = class_counter_total[c_name]
        denom_rows.append({
            "Denomination": f"{int(c_name):,} VND",
            "Train": c_train,
            "Val": c_val,
            "Test": c_test,
            "Total": c_tot,
            "Pct_Total": f"{(c_tot / max(total_boxes, 1) * 100):.1f}%"
        })
    df_denom = pd.DataFrame(denom_rows)

    # Print summary
    print("\n" + "=" * 70)
    print("        THỐNG KÊ SỐ LƯỢNG BOUNDING BOX THEO MỆNH GIÁ VÀ SPLIT        ")
    print("=" * 70)
    print(df_denom.to_string(index=False))
    print("=" * 70)

    print("\n" + "=" * 70)
    print("            PHÂN BỐ SỐ TỜ TIỀN TRÊN MỖI BỨC ẢNH (MULTI-NOTE)          ")
    print("=" * 70)
    box_dist_rows = [
        {"Num_Banknotes": k, "Num_Images": v, "Pct": f"{(v / total_images * 100):.1f}%"}
        for k, v in sorted(boxes_per_image_dist.items())
    ]
    print(pd.DataFrame(box_dist_rows).to_string(index=False))
    print("=" * 70)

    hard_case_count = int(df_splits["is_hard_case"].sum()) if df_splits is not None and "is_hard_case" in df_splits.columns else 0
    multi_denom_count = int(df_splits["is_multi_denom"].sum()) if df_splits is not None and "is_multi_denom" in df_splits.columns else 0

    report = {
        "dataset_dir": str(dataset_dir),
        "total_images": total_images,
        "total_boxes": total_boxes,
        "hard_case_images": hard_case_count,
        "hard_case_ratio": round(hard_case_count / max(total_images, 1), 4),
        "multi_denomination_images": multi_denom_count,
        "split_stats": split_stats,
        "boxes_per_image_distribution": dict(boxes_per_image_dist),
        "class_totals": dict(class_counter_total),
    }

    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\nAudit report exported to: {report_path}")

    return report


def main():
    parser = argparse.ArgumentParser(description="Inspect and audit YOLOv11 dataset.")
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default="data/processed/yolo_dataset",
        help="Path to YOLO dataset"
    )
    parser.add_argument(
        "--splits_csv",
        type=str,
        default="data/metadata/splits.csv",
        help="Path to splits.csv"
    )
    parser.add_argument(
        "--report_path",
        type=str,
        default="artifacts/reports/yolo_audit_report.json",
        help="Path to save report"
    )
    args = parser.parse_args()

    audit_yolo_dataset(
        dataset_dir=PROJECT_ROOT / args.dataset_dir,
        splits_csv=PROJECT_ROOT / args.splits_csv,
        report_path=PROJECT_ROOT / args.report_path,
    )


if __name__ == "__main__":
    main()
