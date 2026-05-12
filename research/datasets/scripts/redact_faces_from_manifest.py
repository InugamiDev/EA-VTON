"""Redact faces using labeled face boxes from a restricted JSONL intake manifest.

Input records may include `source_image_uri` or `raw_image_uri` plus `face_boxes`.
Output records point to redacted images and remove raw/source image paths and
face boxes before the records enter the training manifest.
"""

# intent: make face masking a required preprocessing step before style labels enter training
# status: done
# next: add detector-assisted face-box generation in a separate restricted intake pipeline
# blockers: automatic face detection dependency is intentionally not required here
# confidence: high

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any


def load_pillow():
    try:
        from PIL import Image, ImageDraw, ImageFilter
    except ImportError as exc:
        raise SystemExit("Pillow is required for redaction. Install with `python3 -m pip install pillow`.") from exc
    return Image, ImageDraw, ImageFilter


def nearest_existing_parent(path: Path) -> Path:
    candidate = path if path.exists() else path.parent
    while not candidate.exists() and candidate != candidate.parent:
        candidate = candidate.parent
    return candidate if candidate.exists() else Path(".")


def require_free_space(path: Path, min_free_gb: float) -> None:
    if min_free_gb <= 0:
        return
    check_path = nearest_existing_parent(path)
    free_bytes = shutil.disk_usage(check_path).free
    required_bytes = int(min_free_gb * 1024**3)
    if free_bytes < required_bytes:
        free_gb = free_bytes / 1024**3
        raise SystemExit(
            f"insufficient free disk at {check_path}: {free_gb:.1f}GiB available, "
            f"{min_free_gb:.1f}GiB required. Re-run with more space or lower --min-free-gb."
        )


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            record = json.loads(stripped)
            if not isinstance(record, dict):
                raise ValueError(f"line {line_number}: record must be an object")
            yield line_number, record


def resolve_source(record: dict[str, Any], input_root: Path) -> Path:
    uri = record.get("source_image_uri") or record.get("raw_image_uri")
    if not uri:
        raise ValueError("record missing source_image_uri/raw_image_uri")
    path = Path(str(uri))
    return path if path.is_absolute() else input_root / path


def box_to_pixels(box: dict[str, Any], width: int, height: int) -> tuple[int, int, int, int]:
    x = float(box["x"])
    y = float(box["y"])
    w = float(box["width"])
    h = float(box["height"])
    if 0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1:
        x *= width
        w *= width
        y *= height
        h *= height
    left = max(0, int(round(x)))
    top = max(0, int(round(y)))
    right = min(width, int(round(x + w)))
    bottom = min(height, int(round(y + h)))
    return left, top, right, bottom


def redact_image(source: Path, destination: Path, boxes: list[dict[str, Any]], method: str) -> None:
    Image, ImageDraw, ImageFilter = load_pillow()
    destination.parent.mkdir(parents=True, exist_ok=True)

    if not boxes:
        shutil.copy2(source, destination)
        return

    image = Image.open(source).convert("RGB")
    width, height = image.size

    if method == "blur":
        for box in boxes:
            left, top, right, bottom = box_to_pixels(box, width, height)
            if right <= left or bottom <= top:
                continue
            patch = image.crop((left, top, right, bottom)).filter(ImageFilter.GaussianBlur(radius=24))
            image.paste(patch, (left, top))
    else:
        draw = ImageDraw.Draw(image)
        for box in boxes:
            left, top, right, bottom = box_to_pixels(box, width, height)
            if right <= left or bottom <= top:
                continue
            draw.rectangle((left, top, right, bottom), fill=(0, 0, 0))

    image.save(destination, quality=92)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path, help="Restricted intake JSONL with source image paths and face_boxes")
    parser.add_argument("--input-root", type=Path, default=Path("."))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--method", choices=["black_box", "blur"], default="black_box")
    parser.add_argument("--min-free-gb", type=float, default=2.0, help="minimum free disk required before writing redacted images")
    args = parser.parse_args()

    require_free_space(args.output_root, args.min_free_gb)

    count = 0
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    with args.output_manifest.open("w", encoding="utf-8") as out:
        for line_number, record in iter_jsonl(args.manifest):
            try:
                source = resolve_source(record, args.input_root)
                if not source.exists():
                    raise FileNotFoundError(source)
                image_id = str(record.get("image_id") or source.stem)
                destination = args.output_root / f"{image_id}{source.suffix.lower() or '.jpg'}"
                boxes = record.get("face_boxes") or []
                if not isinstance(boxes, list):
                    raise ValueError("face_boxes must be a list")
                redact_image(source, destination, boxes, args.method)

                next_record = dict(record)
                next_record.pop("source_image_uri", None)
                next_record.pop("raw_image_uri", None)
                next_record.pop("face_boxes", None)
                next_record["redacted_image_uri"] = str(destination)
                next_record["face_redaction"] = {
                    "status": "redacted" if boxes else "no_face_visible",
                    "method": args.method if boxes else "none",
                }
                out.write(json.dumps(next_record, sort_keys=True) + "\n")
                count += 1
            except Exception as exc:  # noqa: BLE001 - batch script should report row context
                print(f"line {line_number}: {exc}", file=sys.stderr)
                return 1

    print(f"redacted {count:,} image(s) -> {args.output_manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
