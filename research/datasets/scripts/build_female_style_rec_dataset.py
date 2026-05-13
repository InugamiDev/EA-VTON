"""Build a style recommendation dataset from redacted clothing manifests.

The source manifests must already contain redacted image paths and non-identifying
labels. This builder creates train-ready files for style recommendation:

- split JSONL records for image-level retrieval/classification
- flat image label CSV
- garment label CSV
- person profile JSONL with repeated-image style aggregates
- positive interaction JSONL for person-to-outfit/item training
- class maps and summary stats
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


DEFAULT_SOURCE = Path(
    "research/datasets/female_clothing_style_v1/"
    "manifest_deepfashion_with_masks_balanced_10_20.jsonl"
)
DEFAULT_OUTPUT_DIR = Path("research/datasets/processed/female_style_rec_v1")

SPLITS = ("train", "val", "test", "holdout")
IMAGE_LABEL_FIELDS = [
    "person_id",
    "image_id",
    "split",
    "redacted_image_uri",
    "primary_family",
    "primary_category",
    "occasion",
    "formality",
    "aesthetic_tags",
    "color_mood",
    "layering",
    "gender_label_source",
    "rights_basis",
]
GARMENT_FIELDS = [
    "person_id",
    "image_id",
    "split",
    "garment_id",
    "family",
    "category",
    "visibility",
    "layer_role",
    "fit",
    "silhouette",
    "color_family",
    "pattern",
    "fabric_weight",
    "structure",
    "material_tags",
]


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as records:
        for line_number, line in enumerate(records, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield line_number, json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{line_number}: invalid JSON: {exc}") from exc


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output:
        for record in records:
            output.write(json.dumps(record, sort_keys=True) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def primary_garment(record: dict[str, Any]) -> dict[str, Any]:
    garments = record.get("garments") or []
    for garment in garments:
        if garment.get("visibility") == "primary":
            return garment
    return garments[0] if garments else {}


def style_key(record: dict[str, Any]) -> str:
    garment = primary_garment(record)
    outfit = record.get("outfit") or {}
    attrs = garment.get("attributes") or {}
    tags = sorted(outfit.get("aesthetic_tags") or ["unknown"])
    key_parts = [
        garment.get("family", "unknown"),
        garment.get("category", "unknown"),
        outfit.get("formality", "unknown"),
        attrs.get("fit", "unknown"),
        attrs.get("color_family", "unknown"),
        "+".join(tags),
    ]
    return "|".join(key_parts)


def image_label_row(record: dict[str, Any]) -> dict[str, Any]:
    garment = primary_garment(record)
    outfit = record.get("outfit") or {}
    return {
        "person_id": record["person_id"],
        "image_id": record["image_id"],
        "split": record["split"],
        "redacted_image_uri": record["redacted_image_uri"],
        "primary_family": garment.get("family", "unknown"),
        "primary_category": garment.get("category", "unknown"),
        "occasion": outfit.get("occasion", "unknown"),
        "formality": outfit.get("formality", "unknown"),
        "aesthetic_tags": "|".join(outfit.get("aesthetic_tags") or ["unknown"]),
        "color_mood": outfit.get("color_mood", "unknown"),
        "layering": outfit.get("layering", "unknown"),
        "gender_label_source": (record.get("gender_label") or {}).get("source", "unknown"),
        "rights_basis": (record.get("rights") or {}).get("basis", "unknown"),
    }


def garment_rows(record: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for garment in record.get("garments") or []:
        attrs = garment.get("attributes") or {}
        rows.append(
            {
                "person_id": record["person_id"],
                "image_id": record["image_id"],
                "split": record["split"],
                "garment_id": garment.get("garment_id", ""),
                "family": garment.get("family", "unknown"),
                "category": garment.get("category", "unknown"),
                "visibility": garment.get("visibility", "unknown"),
                "layer_role": garment.get("layer_role", "unknown"),
                "fit": attrs.get("fit", "unknown"),
                "silhouette": attrs.get("silhouette", "unknown"),
                "color_family": attrs.get("color_family", "unknown"),
                "pattern": attrs.get("pattern", "unknown"),
                "fabric_weight": attrs.get("fabric_weight", "unknown"),
                "structure": attrs.get("structure", "unknown"),
                "material_tags": "|".join(attrs.get("material_tags") or ["unknown"]),
            }
        )
    return rows


def interaction_record(record: dict[str, Any]) -> dict[str, Any]:
    garment = primary_garment(record)
    outfit = record.get("outfit") or {}
    attrs = garment.get("attributes") or {}
    return {
        "person_id": record["person_id"],
        "image_id": record["image_id"],
        "split": record["split"],
        "item_id": f"style_{style_key(record)}",
        "label": 1,
        "event_type": "observed_outfit",
        "redacted_image_uri": record["redacted_image_uri"],
        "family": garment.get("family", "unknown"),
        "category": garment.get("category", "unknown"),
        "fit": attrs.get("fit", "unknown"),
        "silhouette": attrs.get("silhouette", "unknown"),
        "color_family": attrs.get("color_family", "unknown"),
        "pattern": attrs.get("pattern", "unknown"),
        "occasion": outfit.get("occasion", "unknown"),
        "formality": outfit.get("formality", "unknown"),
        "aesthetic_tags": sorted(outfit.get("aesthetic_tags") or ["unknown"]),
    }


def class_maps(rows: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    fields = [
        "primary_family",
        "primary_category",
        "occasion",
        "formality",
        "color_mood",
        "layering",
    ]
    maps: dict[str, dict[str, int]] = {}
    for field in fields:
        values = sorted({str(row[field]) for row in rows})
        maps[field] = {value: index for index, value in enumerate(values)}
    aesthetic_values = sorted(
        {
            tag
            for row in rows
            for tag in str(row["aesthetic_tags"]).split("|")
            if tag
        }
    )
    maps["aesthetic_tags"] = {
        value: index for index, value in enumerate(aesthetic_values)
    }
    return maps


def person_profiles(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[record["person_id"]].append(record)

    profiles = []
    for person_id, person_records in sorted(grouped.items()):
        category_counts: Counter[str] = Counter()
        family_counts: Counter[str] = Counter()
        aesthetic_counts: Counter[str] = Counter()
        color_counts: Counter[str] = Counter()
        formality_counts: Counter[str] = Counter()
        image_ids = []
        splits = Counter()
        for record in person_records:
            garment = primary_garment(record)
            outfit = record.get("outfit") or {}
            attrs = garment.get("attributes") or {}
            image_ids.append(record["image_id"])
            splits[record["split"]] += 1
            family_counts[garment.get("family", "unknown")] += 1
            category_counts[garment.get("category", "unknown")] += 1
            color_counts[attrs.get("color_family", "unknown")] += 1
            formality_counts[outfit.get("formality", "unknown")] += 1
            aesthetic_counts.update(outfit.get("aesthetic_tags") or ["unknown"])

        profiles.append(
            {
                "person_id": person_id,
                "num_images": len(person_records),
                "split_counts": dict(sorted(splits.items())),
                "image_ids": sorted(image_ids),
                "top_families": family_counts.most_common(5),
                "top_categories": category_counts.most_common(8),
                "top_aesthetic_tags": aesthetic_counts.most_common(8),
                "top_color_families": color_counts.most_common(8),
                "top_formality": formality_counts.most_common(5),
            }
        )
    return profiles


def summary_stats(records: list[dict[str, Any]], interactions: list[dict[str, Any]]) -> dict[str, Any]:
    people = {record["person_id"] for record in records}
    split_counts = Counter(record["split"] for record in records)
    split_people: dict[str, set[str]] = defaultdict(set)
    category_counts = Counter()
    family_counts = Counter()
    aesthetic_counts = Counter()
    item_counts = Counter(interaction["item_id"] for interaction in interactions)
    for record in records:
        split_people[record["split"]].add(record["person_id"])
        garment = primary_garment(record)
        family_counts[garment.get("family", "unknown")] += 1
        category_counts[garment.get("category", "unknown")] += 1
        aesthetic_counts.update((record.get("outfit") or {}).get("aesthetic_tags") or ["unknown"])

    return {
        "records": len(records),
        "people": len(people),
        "interactions": len(interactions),
        "unique_style_items": len(item_counts),
        "splits": dict(sorted(split_counts.items())),
        "people_by_split": {split: len(split_people[split]) for split in SPLITS if split_people.get(split)},
        "top_families": family_counts.most_common(12),
        "top_categories": category_counts.most_common(20),
        "top_aesthetic_tags": aesthetic_counts.most_common(20),
        "top_style_items": item_counts.most_common(20),
    }


def validate_no_person_leak(records: list[dict[str, Any]]) -> None:
    person_splits: dict[str, set[str]] = defaultdict(set)
    for record in records:
        person_splits[record["person_id"]].add(record["split"])
    leaked = {person: splits for person, splits in person_splits.items() if len(splits) > 1}
    if leaked:
        sample = dict(list(sorted(leaked.items()))[:10])
        raise SystemExit(f"person-level split leakage detected: {sample}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--allow-person-split-overlap", action="store_true")
    args = parser.parse_args()

    records = [record for _, record in iter_jsonl(args.source)]
    if not records:
        raise SystemExit(f"no records found in {args.source}")
    if not args.allow_person_split_overlap:
        validate_no_person_leak(records)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    by_split: dict[str, list[dict[str, Any]]] = {split: [] for split in SPLITS}
    image_rows: list[dict[str, Any]] = []
    all_garment_rows: list[dict[str, Any]] = []
    interactions: list[dict[str, Any]] = []

    for record in records:
        split = record.get("split", "unknown")
        if split not in by_split:
            raise SystemExit(f"unsupported split '{split}' in {record.get('image_id')}")
        by_split[split].append(record)
        image_rows.append(image_label_row(record))
        all_garment_rows.extend(garment_rows(record))
        interactions.append(interaction_record(record))

    for split, split_records in by_split.items():
        if split_records:
            write_jsonl(args.output_dir / f"images_{split}.jsonl", split_records)

    write_csv(args.output_dir / "image_labels.csv", image_rows, IMAGE_LABEL_FIELDS)
    write_csv(args.output_dir / "garment_labels.csv", all_garment_rows, GARMENT_FIELDS)
    write_jsonl(args.output_dir / "style_interactions.jsonl", interactions)
    write_jsonl(args.output_dir / "person_profiles.jsonl", person_profiles(records))

    maps = class_maps(image_rows)
    (args.output_dir / "class_maps.json").write_text(
        json.dumps(maps, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    stats = summary_stats(records, interactions)
    stats["source_manifest"] = str(args.source)
    stats["output_dir"] = str(args.output_dir)
    stats["person_split_leakage_checked"] = not args.allow_person_split_overlap
    (args.output_dir / "stats.json").write_text(
        json.dumps(stats, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "README.md").write_text(
        "# Female Style Recommendation v1\n\n"
        "Processed dataset package built from redacted clothing style manifests.\n\n"
        "Files:\n\n"
        "- `images_train.jsonl`, `images_val.jsonl`, `images_test.jsonl`: image-level records with redacted paths and labels.\n"
        "- `image_labels.csv`: flattened primary outfit labels for classifier baselines.\n"
        "- `garment_labels.csv`: one row per labeled garment.\n"
        "- `person_profiles.jsonl`: repeated-image style aggregates by person.\n"
        "- `style_interactions.jsonl`: positive person-to-style-item observations for retrieval/recommender training.\n"
        "- `class_maps.json`: stable integer ids for categorical labels.\n"
        "- `stats.json`: source, counts, and label distribution summary.\n\n"
        "Use this package for style-rec pilots only. External dataset rows use `dataset_label` gender provenance and must not be treated as first-party self-identification.\n",
        encoding="utf-8",
    )

    print(f"wrote records={len(records):,} people={stats['people']:,} -> {args.output_dir}")
    print(f"splits={stats['splits']}")
    print(f"unique_style_items={stats['unique_style_items']:,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
