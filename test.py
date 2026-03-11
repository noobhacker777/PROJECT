#!/usr/bin/env python3
"""
Upload or scan multiple images one-by-one and print a combined JSON summary.

Modes:
  detect     -> POST /detect                 (alias of /api/detect; field name: "image")
  scan       -> GET  /api/scan?image=...     (single image per request)
  raw-upload -> POST /api/dataset/raw-upload (field name: "files")
  sku-upload -> POST /api/dataset/sku/{sku}/upload (field name: "files")
"""
from __future__ import annotations

import argparse
import mimetypes
import sys
import json as json_module
from contextlib import ExitStack
from pathlib import Path
from typing import Iterable

import requests


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}
PROJECT = Path(__file__).resolve().parent
DATASET_IMAGES = PROJECT / "dataset" / "images"
TMP_DIR = PROJECT / "tmp"


def collect_images(paths: list[str], directory: str | None) -> list[Path]:
    images: list[Path] = []

    if directory:
        dir_path = Path(directory)
        if not dir_path.exists() or not dir_path.is_dir():
            raise ValueError(f"Directory not found: {directory}")
        for p in sorted(dir_path.iterdir()):
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
                images.append(p)

    for p in paths:
        path = Path(p)
        if path.is_dir():
            for fp in sorted(path.iterdir()):
                if fp.is_file() and fp.suffix.lower() in IMAGE_EXTS:
                    images.append(fp)
        elif path.is_file():
            images.append(path)
        else:
            raise ValueError(f"File not found: {p}")

    # De-duplicate while preserving order
    seen = set()
    unique_images = []
    for img in images:
        if img not in seen:
            unique_images.append(img)
            seen.add(img)

    return unique_images


def pick_images_via_dialog() -> list[Path]:
    try:
        import tkinter as tk
        from tkinter import filedialog
    except Exception as exc:
        raise RuntimeError(f"Tkinter not available: {exc}")

    root = tk.Tk()
    root.withdraw()
    file_paths = filedialog.askopenfilenames(
        title="Select image files",
        filetypes=[
            ("Image files", "*.jpg *.jpeg *.png *.bmp *.gif *.webp"),
            ("All files", "*.*"),
        ],
    )
    root.destroy()
    return [Path(p) for p in file_paths]


def is_relative_to(path: Path, base: Path) -> bool:
    try:
        path.resolve().relative_to(base.resolve())
        return True
    except Exception:
        return False


def to_scan_ref(path: Path) -> str:
    if is_relative_to(path, DATASET_IMAGES):
        return f"/uploads/image/{path.name}"
    if is_relative_to(path, TMP_DIR):
        return f"/tmp/{path.name}"
    # Fallback: send just the filename
    return path.name


def collect_scan_refs(paths: list[str], directory: str | None) -> list[tuple[str, str]]:
    refs: list[tuple[str, str]] = []

    if directory:
        dir_path = Path(directory)
        if not dir_path.exists() or not dir_path.is_dir():
            raise ValueError(f"Directory not found: {directory}")
        for p in sorted(dir_path.iterdir()):
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
                refs.append((str(p), to_scan_ref(p)))

    for p in paths:
        if p.startswith("http://") or p.startswith("https://"):
            refs.append((p, p))
            continue

        path = Path(p)
        if path.exists():
            if path.is_dir():
                for fp in sorted(path.iterdir()):
                    if fp.is_file() and fp.suffix.lower() in IMAGE_EXTS:
                        refs.append((str(fp), to_scan_ref(fp)))
            else:
                refs.append((str(path), to_scan_ref(path)))
        else:
            # Allow raw filename or /uploads/image/... or /tmp/... refs
            refs.append((p, p))

    # De-duplicate while preserving order
    seen = set()
    unique_refs: list[tuple[str, str]] = []
    for source, ref in refs:
        if ref not in seen:
            unique_refs.append((source, ref))
            seen.add(ref)

    return unique_refs


def build_url(base_url: str, endpoint: str | None, mode: str, sku: str | None) -> str:
    base = base_url.rstrip("/")
    if endpoint:
        path = endpoint if endpoint.startswith("/") else f"/{endpoint}"
        return f"{base}{path}"
    if mode == "scan":
        return f"{base}/api/scan"
    if mode == "detect":
        return f"{base}/detect"
    if mode == "sku-upload" or sku:
        return f"{base}/api/dataset/sku/{sku}/upload"
    return f"{base}/api/dataset/raw-upload"


def infer_field_name(mode: str) -> str:
    if mode == "detect":
        return "image"
    return "files"


def main() -> int:
    parser = argparse.ArgumentParser(description="Upload or scan multiple images.")
    parser.add_argument("images", nargs="*", help="Image file paths (or directories).")
    parser.add_argument("--dir", dest="directory", help="Directory containing images.")
    parser.add_argument("--base-url", default="http://127.0.0.1:5002", help="API base URL.")
    parser.add_argument("--endpoint", help="Override endpoint path (e.g. /api/dataset/raw-upload).")
    parser.add_argument("--mode", choices=["detect", "scan", "raw-upload", "sku-upload"], default="detect",
                        help="Mode: detect (POST /detect), scan (GET /api/scan), raw-upload (POST), sku-upload (POST).")
    parser.add_argument("--sku", help="SKU name for sku-upload mode.")
    parser.add_argument("--timeout", type=int, default=60, help="Request timeout in seconds.")
    parser.add_argument("--field", help="Override multipart field name (default: image or files).")
    parser.add_argument("--accuracy-mode", action="store_true", help="Enable accuracy_mode for /detect.")
    parser.add_argument("--picker", action="store_true", help="Open a file picker for images.")
    args = parser.parse_args()

    mode = args.mode
    if mode == "sku-upload" and not args.sku and not args.endpoint:
        print("Error: --sku is required for sku-upload mode (or set --endpoint).", file=sys.stderr)
        return 2

    url = build_url(args.base_url, args.endpoint, mode, args.sku)
    results = []

    picked_paths: list[Path] = []
    if args.picker or (not args.images and not args.directory):
        try:
            picked_paths = pick_images_via_dialog()
        except RuntimeError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 2

    if mode == "scan":
        try:
            scan_inputs = list(args.images)
            if picked_paths:
                scan_inputs.extend(str(p) for p in picked_paths)
            scan_refs = collect_scan_refs(scan_inputs, args.directory)
        except ValueError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 2

        if not scan_refs:
            print("Error: No images found. Provide refs or --dir.", file=sys.stderr)
            return 2

        print(f"GET {url}")
        print(f"Scanning {len(scan_refs)} image(s) one-by-one...")

        warn_sources = []
        for source, ref in scan_refs:
            src_path = Path(source)
            if src_path.exists() and ref == src_path.name:
                if not is_relative_to(src_path, DATASET_IMAGES) and not is_relative_to(src_path, TMP_DIR):
                    warn_sources.append(source)
        if warn_sources:
            print("Warning: Some local files are not under dataset/images or tmp; /scan may not find them.")

        for source, ref in scan_refs:
            response = requests.get(url, params={"image": ref}, timeout=args.timeout)
            content_type = response.headers.get("content-type", "")
            if "application/json" in content_type.lower():
                try:
                    payload = response.json()
                except Exception:
                    payload = {"_raw": response.text}
            else:
                payload = {"_raw": response.text}

            results.append({
                "input": ref,
                "source": source if source != ref else None,
                "status": response.status_code,
                "ok": response.ok,
                "response": payload,
            })
    else:
        try:
            img_inputs = list(args.images)
            if picked_paths:
                img_inputs.extend(str(p) for p in picked_paths)
            image_paths = collect_images(img_inputs, args.directory)
        except ValueError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 2

        if not image_paths:
            print("Error: No images found. Provide paths or --dir.", file=sys.stderr)
            return 2

        field_name = args.field or infer_field_name(mode)
        print(f"POST {url}")
        print(f"Uploading {len(image_paths)} image(s) one-by-one...")

        params = {}
        if mode == "detect" and args.accuracy_mode:
            params["accuracy_mode"] = "true"

        for img_path in image_paths:
            mime_type, _ = mimetypes.guess_type(str(img_path))
            mime_type = mime_type or "application/octet-stream"

            with ExitStack() as stack:
                f = stack.enter_context(open(img_path, "rb"))
                files = [(field_name, (img_path.name, f, mime_type))]
                response = requests.post(url, files=files, params=params or None, timeout=args.timeout)

            content_type = response.headers.get("content-type", "")
            if "application/json" in content_type.lower():
                try:
                    payload = response.json()
                except Exception:
                    payload = {"_raw": response.text}
            else:
                payload = {"_raw": response.text}

            results.append({
                "input": str(img_path),
                "status": response.status_code,
                "ok": response.ok,
                "response": payload,
            })

    summary = {
        "mode": mode,
        "count": len(results),
        "ok": all(r["ok"] for r in results),
        "results": results,
    }
    print(json_module.dumps(summary, indent=2))
    return 0 if summary["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
