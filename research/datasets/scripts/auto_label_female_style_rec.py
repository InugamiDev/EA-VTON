"""Auto-label style recommendation records from redacted manifests.

This does not infer identity or sensitive attributes. It converts existing
source-derived outfit/garment labels into model-training labels, adds confidence
scores, and creates a review queue for low-confidence records.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any


DEFAULT_SOURCE = Path("research/datasets/processed/female_style_rec_v1/images_train.jsonl")
DEFAULT_OUTPUT_DIR = Path("research/datasets/annotation/female_style_rec_v1/auto_labels")

AUTO_LABEL_FIELDS = [
    "task_id",
    "status",
    "confidence",
    "review_reason",
    "person_id",
    "image_id",
    "split",
    "redacted_image_uri",
    "family",
    "category",
    "occasion",
    "formality",
    "aesthetic_tags",
    "color_mood",
    "layering",
    "fit",
    "silhouette",
    "color_family",
    "pattern",
    "style_item_id",
]


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as records:
        for line_number, line in enumerate(records, 1):
            line = line.strip()
            if line:
                yield line_number, json.loads(line)


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as output:
        for record in records:
            output.write(json.dumps(record, sort_keys=True) + "\n")


def write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=AUTO_LABEL_FIELDS)
        writer.writeheader()
        for record in records:
            writer.writerow({field: record.get(field, "") for field in AUTO_LABEL_FIELDS})


def primary_garment(record: dict[str, Any]) -> dict[str, Any]:
    garments = record.get("garments") or []
    for garment in garments:
        if garment.get("visibility") == "primary":
            return garment
    return garments[0] if garments else {}


def style_item_id(label: dict[str, Any]) -> str:
    tags = "+".join(label["aesthetic_tags"])
    return (
        f"style_{label['family']}|{label['category']}|{label['formality']}|"
        f"{label['fit']}|{label['color_family']}|{tags}"
    )


def score_label(label: dict[str, Any]) -> tuple[float, list[str]]:
    confidence = 1.0
    reasons: list[str] = []

    required = ["family", "category", "occasion", "formality", "color_mood", "layering"]
    for field in required:
        if label[field] in {"", "unknown", "not_applicable"}:
            confidence -= 0.10
            reasons.append(f"{field}_unknown")

    attrs = ["fit", "silhouette", "color_family", "pattern"]
    for field in attrs:
        if label[field] in {"", "unknown", "not_applicable"}:
            confidence -= 0.08
            reasons.append(f"{field}_unknown")

    if not label["aesthetic_tags"] or "unknown" in label["aesthetic_tags"]:
        confidence -= 0.12
        reasons.append("aesthetic_unknown")

    if label["family"] == "top" and label["category"] in {"tank", "t_shirt", "blouse", "sweater"}:
        confidence += 0.03
    if label["family"] == "dress_jumpsuit" and "dress" in label["category"]:
        confidence += 0.03
    if label["color_family"] != "unknown" and label["color_mood"] != "unknown":
        confidence += 0.03

    confidence = max(0.0, min(0.99, confidence))
    return round(confidence, 3), reasons


def build_auto_label(record: dict[str, Any], index: int, threshold: float) -> dict[str, Any]:
    garment = primary_garment(record)
    outfit = record.get("outfit") or {}
    attrs = garment.get("attributes") or {}
    label = {
        "task_id": f"auto_style_{index:06d}",
        "person_id": record["person_id"],
        "image_id": record["image_id"],
        "split": record["split"],
        "redacted_image_uri": record["redacted_image_uri"],
        "family": garment.get("family", "unknown"),
        "category": garment.get("category", "unknown"),
        "occasion": outfit.get("occasion", "unknown"),
        "formality": outfit.get("formality", "unknown"),
        "aesthetic_tags": sorted(outfit.get("aesthetic_tags") or ["unknown"]),
        "color_mood": outfit.get("color_mood", "unknown"),
        "layering": outfit.get("layering", "unknown"),
        "fit": attrs.get("fit", "unknown"),
        "silhouette": attrs.get("silhouette", "unknown"),
        "color_family": attrs.get("color_family", "unknown"),
        "pattern": attrs.get("pattern", "unknown"),
    }
    confidence, reasons = score_label(label)
    label["confidence"] = confidence
    label["review_reason"] = "|".join(reasons)
    label["status"] = "auto_labeled" if confidence >= threshold else "needs_review"
    label["style_item_id"] = style_item_id(label)
    return label


def csv_ready(record: dict[str, Any]) -> dict[str, Any]:
    row = dict(record)
    row["aesthetic_tags"] = "|".join(row.get("aesthetic_tags") or [])
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--confidence-threshold", type=float, default=0.72)
    args = parser.parse_args()

    if not 0.0 <= args.confidence_threshold <= 1.0:
        raise SystemExit("--confidence-threshold must be between 0 and 1")

    records = [record for _, record in iter_jsonl(args.source)]
    if not records:
        raise SystemExit(f"no records found in {args.source}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    auto_labels = [
        build_auto_label(record, index + 1, args.confidence_threshold)
        for index, record in enumerate(records)
    ]
    accepted = [record for record in auto_labels if record["status"] == "auto_labeled"]
    review = [record for record in auto_labels if record["status"] == "needs_review"]

    write_jsonl(args.output_dir / "auto_labels.jsonl", auto_labels)
    write_jsonl(args.output_dir / "accepted_labels.jsonl", accepted)
    write_jsonl(args.output_dir / "review_queue.jsonl", review)
    write_csv(args.output_dir / "auto_labels.csv", [csv_ready(record) for record in auto_labels])
    write_csv(args.output_dir / "review_queue.csv", [csv_ready(record) for record in review])

    status_counts = Counter(record["status"] for record in auto_labels)
    reason_counts = Counter(
        reason
        for record in review
        for reason in str(record.get("review_reason", "")).split("|")
        if reason
    )
    category_counts = Counter(record["category"] for record in auto_labels)
    confidence_values = [record["confidence"] for record in auto_labels]
    stats = {
        "source": str(args.source),
        "output_dir": str(args.output_dir),
        "confidence_threshold": args.confidence_threshold,
        "records": len(auto_labels),
        "accepted": len(accepted),
        "needs_review": len(review),
        "status_counts": dict(sorted(status_counts.items())),
        "top_review_reasons": reason_counts.most_common(20),
        "top_categories": category_counts.most_common(20),
        "confidence": {
            "min": min(confidence_values),
            "max": max(confidence_values),
            "mean": round(sum(confidence_values) / len(confidence_values), 4),
        },
    }
    (args.output_dir / "stats.json").write_text(
        json.dumps(stats, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "README.md").write_text(
        "# Auto Labels\n\n"
        "Auto-labeled style recommendation records generated from redacted manifests and existing source-derived labels.\n\n"
        "Files:\n\n"
        "- `auto_labels.jsonl`: all auto-label records with confidence and status.\n"
        "- `accepted_labels.jsonl`: labels accepted at the configured confidence threshold.\n"
        "- `review_queue.jsonl`: low-confidence records for human verification.\n"
        "- `auto_labels.csv` and `review_queue.csv`: spreadsheet-friendly versions.\n"
        "- `stats.json`: counts and review-reason distribution.\n\n"
        "These labels do not infer identity or sensitive traits. They should be treated as weak labels until human QA confirms enough samples per class.\n",
        encoding="utf-8",
    )

    print(f"auto-labeled records={len(auto_labels):,} accepted={len(accepted):,} review={len(review):,}")
    print(f"confidence={stats['confidence']}")
    print(f"top_review_reasons={reason_counts.most_common(8)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
