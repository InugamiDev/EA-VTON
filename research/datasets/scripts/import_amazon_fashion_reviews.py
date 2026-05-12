"""Import sanitized Amazon fashion review signals as a non-person sidecar dataset.

The Amazon review data is useful for product metadata, size/color style fields,
ratings, and review text. It is not a consented people-image dataset. This
importer strips reviewer identifiers and image URLs before writing records.
"""

# intent: extract fashion product/review signals without carrying reviewer identifiers or user images
# status: done
# next: add metadata importer if form-gated product metadata is approved and downloaded
# blockers: full metadata files require request-form access from the dataset maintainers
# confidence: high

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
import shutil
import urllib.error
from collections import Counter
from pathlib import Path
from typing import Any
from urllib.request import urlopen


DEFAULT_SOURCE_URL = "https://jmcauley.ucsd.edu/data/amazon_v2/categoryFilesSmall/AMAZON_FASHION_5.json.gz"
DEFAULT_RAW_PATH = Path("research/datasets/raw/female_clothing_style_v1/amazon_review_2018_fashion/AMAZON_FASHION_5.json.gz")
DEFAULT_OUTPUT = Path("research/datasets/female_clothing_style_v1/amazon_review_2018_fashion_5core_sanitized.jsonl")

EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
PHONE_RE = re.compile(r"(?<!\d)(?:\+?1[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)\d{3}[-.\s]?\d{4}(?!\d)")
URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)


def stable_id(*parts: object) -> str:
    raw = "|".join(str(part or "") for part in parts)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def sanitize_text(value: object, max_length: int) -> str:
    text = str(value or "")
    text = EMAIL_RE.sub("[email_removed]", text)
    text = PHONE_RE.sub("[phone_removed]", text)
    text = URL_RE.sub("[url_removed]", text)
    text = " ".join(text.split())
    return text[:max_length]


def normalize_style(style: object) -> dict[str, str]:
    if not isinstance(style, dict):
        return {}
    normalized: dict[str, str] = {}
    for key, value in style.items():
        clean_key = str(key).strip().lower().rstrip(":")
        if clean_key in {"size", "color", "colour", "fit", "inseam", "waist", "width"}:
            normalized["color" if clean_key == "colour" else clean_key] = sanitize_text(value, 120)
    return normalized


def nearest_existing_parent(path: Path) -> Path:
    candidate = path if path.exists() else path.parent
    while not candidate.exists() and candidate != candidate.parent:
        candidate = candidate.parent
    return candidate if candidate.exists() else Path(".")


def require_free_space(path: Path, min_free_mb: float) -> None:
    if min_free_mb <= 0:
        return
    check_path = nearest_existing_parent(path)
    free_bytes = shutil.disk_usage(check_path).free
    required_bytes = int(min_free_mb * 1024**2)
    if free_bytes < required_bytes:
        free_mb = free_bytes / 1024**2
        raise SystemExit(
            f"insufficient free disk at {check_path}: {free_mb:.0f}MiB available, "
            f"{min_free_mb:.0f}MiB required. Re-run with more space or lower --min-free-mb."
        )


def iter_reviews(path: Path):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                record = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"line {line_number}: invalid JSON: {exc}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"line {line_number}: record must be an object")
            yield record


def download_file(url: str, destination: Path, timeout_seconds: float) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_path = destination.with_suffix(destination.suffix + ".partial")
    try:
        with urlopen(url, timeout=timeout_seconds) as response, temp_path.open("wb") as out:
            shutil.copyfileobj(response, out)
        temp_path.replace(destination)
    except (OSError, TimeoutError, urllib.error.URLError) as exc:
        temp_path.unlink(missing_ok=True)
        raise SystemExit(f"download failed for {url}: {exc}") from exc


def build_record(record: dict[str, Any]) -> dict[str, Any]:
    image_list = record.get("image") if isinstance(record.get("image"), list) else []
    review_text = sanitize_text(record.get("reviewText"), 4000)
    summary = sanitize_text(record.get("summary"), 500)
    asin = sanitize_text(record.get("asin"), 64)
    unix_time = record.get("unixReviewTime")
    return {
        "review_id": f"ar_{stable_id(asin, unix_time, summary, review_text)}",
        "source": "Amazon Review Data 2018 / AMAZON_FASHION_5",
        "rights": {
            "basis": "dataset_license",
            "status": "licensed",
            "scope": "research_only",
            "source_page": "https://nijianmo.github.io/amazon/index.html",
        },
        "asin": asin,
        "overall": record.get("overall"),
        "verified": bool(record.get("verified")) if "verified" in record else None,
        "unix_review_time": unix_time if isinstance(unix_time, int) else None,
        "review_time": sanitize_text(record.get("reviewTime"), 64),
        "style": normalize_style(record.get("style")),
        "summary": summary,
        "review_text": review_text,
        "user_image_count": len(image_list),
        "user_image_urls_removed": bool(image_list),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_RAW_PATH)
    parser.add_argument("--source-url", default=DEFAULT_SOURCE_URL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--download", action="store_true", help="download the small 5-core source file if missing")
    parser.add_argument("--download-timeout", type=float, default=30.0, help="seconds to wait for the direct download connection")
    parser.add_argument("--min-free-mb", type=float, default=512.0, help="minimum free disk required before writing download/output files")
    parser.add_argument("--limit", type=int, default=0, help="maximum records to import; 0 means all")
    args = parser.parse_args()

    require_free_space(args.output, args.min_free_mb)

    if args.download and not args.source.exists():
        require_free_space(args.source, args.min_free_mb)
        print(f"downloading {args.source_url} -> {args.source}")
        download_file(args.source_url, args.source, args.download_timeout)

    if not args.source.exists():
        raise SystemExit(f"source file not found: {args.source}. Re-run with --download or place it there.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    counts: Counter[str] = Counter()
    with args.output.open("w", encoding="utf-8") as out:
        for source_record in iter_reviews(args.source):
            record = build_record(source_record)
            out.write(json.dumps(record, sort_keys=True) + "\n")
            counts["records"] += 1
            if record["user_image_urls_removed"]:
                counts["records_with_user_images_removed"] += 1
            if record["style"]:
                counts["records_with_style"] += 1
            if args.limit and counts["records"] >= args.limit:
                break

    print(f"wrote records={counts['records']:,} -> {args.output}")
    print(f"records_with_style={counts['records_with_style']:,}")
    print(f"records_with_user_images_removed={counts['records_with_user_images_removed']:,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
