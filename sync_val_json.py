#!/usr/bin/env python3
"""Convert dataset/val HOLO txt labels back into app-compatible JSON files.

This mirrors ``json_to_holo.py`` in reverse:
- reads ``dataset/val/*.txt``
- writes ``dataset/json_data/*.json``
- sets a default SKU on every generated box
- ensures the default SKU exists in ``dataset/SKU.json``
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Optional


DEFAULT_SKU = "DEFAULT"
PROJECT = Path(__file__).parent
DATASET = PROJECT / "dataset"
VAL_DIR = DATASET / "val"
JSON_DIR = DATASET / "json_data"
SKU_FILE = DATASET / "SKU.json"
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def find_image_name(image_stem: str) -> str:
    """Best-effort image filename lookup for metadata."""
    images_dir = DATASET / "images"
    for ext in IMAGE_EXTENSIONS:
        candidate = images_dir / f"{image_stem}{ext}"
        if candidate.exists():
            return candidate.name
    return f"{image_stem}.jpg"


def ensure_default_sku(default_sku: str = DEFAULT_SKU) -> None:
    """Ensure dataset/SKU.json exists and contains the default SKU."""
    DATASET.mkdir(parents=True, exist_ok=True)

    payload = {
        "skus": [default_sku],
        "timestamp": datetime.now().isoformat(),
        "total_skus": 1,
    }

    if SKU_FILE.exists():
        try:
            existing = json.loads(SKU_FILE.read_text(encoding="utf-8"))
            skus = existing.get("skus", []) if isinstance(existing, dict) else []
            sku_set = {str(s).strip() for s in skus if str(s).strip()}
            sku_set.add(default_sku)
            payload = {
                "skus": sorted(sku_set),
                "timestamp": datetime.now().isoformat(),
                "total_skus": len(sku_set),
            }
        except Exception:
            pass

    SKU_FILE.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def convert_txt_to_json(txt_file: Path, output_dir: Path, default_sku: str = DEFAULT_SKU) -> Optional[int]:
    """Convert one HOLO txt file into the app's JSON annotation format."""
    try:
        image_stem = txt_file.stem
        lines = txt_file.read_text(encoding="utf-8").splitlines()
        boxes = []

        for line_number, raw_line in enumerate(lines, start=1):
            line = raw_line.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) < 5:
                print(f"Warning: {txt_file.name}:{line_number} skipped (expected 5 values, got {len(parts)})")
                continue

            class_id = int(float(parts[0]))
            center_x = round(float(parts[1]), 6)
            center_y = round(float(parts[2]), 6)
            width = round(float(parts[3]), 6)
            height = round(float(parts[4]), 6)

            boxes.append(
                {
                    "bbox": [class_id, center_x, center_y, width, height],
                    "class_name": "product",
                    "sku": default_sku,
                    "cropped": None,
                }
            )

        annotation = {
            "image_name": find_image_name(image_stem),
            "image_stem": image_stem,
            "image_width": None,
            "image_height": None,
            "timestamp": datetime.now().isoformat(),
            "class_name": "product",
            "boxes": boxes,
        }

        output_path = output_dir / f"{image_stem}.json"
        output_path.write_text(json.dumps(annotation, indent=2), encoding="utf-8")
        print(f"Converted {txt_file.name} -> {output_path.name} ({len(boxes)} boxes, sku={default_sku})")
        return len(boxes)
    except Exception as exc:
        print(f"Failed {txt_file.name}: {exc}")
        return None


def main() -> bool:
    """Convert all dataset/val txt labels into dataset/json_data JSON files."""
    if not VAL_DIR.exists():
        print(f"Error: val directory not found: {VAL_DIR}")
        return False

    JSON_DIR.mkdir(parents=True, exist_ok=True)
    ensure_default_sku(DEFAULT_SKU)

    txt_files = sorted(VAL_DIR.glob("*.txt"))
    if not txt_files:
        print(f"Error: No txt files found in {VAL_DIR}")
        return False

    print(f"Converting {len(txt_files)} txt files from {VAL_DIR} to {JSON_DIR}...\n")

    total_boxes = 0
    success_count = 0
    for txt_file in txt_files:
        result = convert_txt_to_json(txt_file, JSON_DIR, DEFAULT_SKU)
        if result is not None:
            success_count += 1
            total_boxes += result

    print("\n" + "=" * 60)
    print(f"Conversion complete: {success_count}/{len(txt_files)} files processed")
    print(f"Total boxes written: {total_boxes}")
    print(f"Output directory: {JSON_DIR}")
    print(f"Default SKU ensured in: {SKU_FILE}")
    print("=" * 60)
    return success_count == len(txt_files)


if __name__ == "__main__":
    import sys

    sys.exit(0 if main() else 1)
