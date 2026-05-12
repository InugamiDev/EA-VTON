"""Audit DeepFashion MultiModal Parts2Whole JSONL metadata without media downloads.

This does not import images. It only counts metadata rows, unique target paths,
gender labels, category labels, and person/outfit groups so we can decide whether
the source can satisfy a requested images-per-person target before downloading
large media archives.
"""

# intent: make Parts2Whole source feasibility reproducible before any large media import
# status: done
# next: use this audit before approving or rejecting a media download/import
# blockers: metadata labels are dataset labels, not first-party self-identification
# confidence: high

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


DEFAULT_RAW_ROOT = Path("research/datasets/raw/female_clothing_style_v1/deepfashion_multimodal_parts2whole")
DEFAULT_OUTPUT = Path("research/datasets/female_clothing_style_v1/parts2whole_metadata_audit.md")
TARGET_RE = re.compile(r"^(?P<gender>MEN|WOMEN)-(?P<category>.+)-(?P<person>id_\d+)-(?P<outfit>\d+)_")


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                record = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_number}: record must be an object")
            yield record


def parse_target_id(target_id: object) -> tuple[str, str, str, str] | None:
    match = TARGET_RE.match(str(target_id or ""))
    if not match:
        return None
    gender = match.group("gender")
    category = match.group("category")
    source_person_key = f"{gender}-{match.group('person')}"
    strict_outfit_key = f"{gender}-{category}-{match.group('person')}-{match.group('outfit')}"
    return gender, category, source_person_key, strict_outfit_key


def audit_file(path: Path, min_images_per_person: int) -> dict[str, Any]:
    rows = 0
    unique_targets: set[str] = set()
    gender_counts: Counter[str] = Counter()
    category_counts: Counter[str] = Counter()
    source_person_targets: defaultdict[str, set[str]] = defaultdict(set)
    strict_outfit_targets: defaultdict[str, set[str]] = defaultdict(set)
    parse_failures = 0

    for record in iter_jsonl(path):
        rows += 1
        target = str(record.get("target") or "")
        target_id = str(record.get("target_id") or "")
        if target:
            unique_targets.add(target)
        parsed = parse_target_id(target_id)
        if parsed is None:
            parse_failures += 1
            continue
        gender, category, source_person_key, strict_outfit_key = parsed
        gender_counts[gender] += 1
        category_counts[f"{gender}/{category}"] += 1
        if target:
            source_person_targets[source_person_key].add(target)
            strict_outfit_targets[strict_outfit_key].add(target)

    source_images_per_person = Counter({person: len(targets) for person, targets in source_person_targets.items()})
    strict_images_per_outfit = Counter({person: len(targets) for person, targets in strict_outfit_targets.items()})
    eligible_source_people = sum(1 for count in source_images_per_person.values() if count >= min_images_per_person)
    eligible_strict_outfits = sum(1 for count in strict_images_per_outfit.values() if count >= min_images_per_person)
    women_source_people = {person: count for person, count in source_images_per_person.items() if person.startswith("WOMEN-")}
    women_strict_outfits = {person: count for person, count in strict_images_per_outfit.items() if person.startswith("WOMEN-")}
    eligible_women_source_people = sum(1 for count in women_source_people.values() if count >= min_images_per_person)
    eligible_women_strict_outfits = sum(1 for count in women_strict_outfits.values() if count >= min_images_per_person)

    return {
        "rows": rows,
        "unique_targets": len(unique_targets),
        "source_people": len(source_person_targets),
        "source_people_min_images": eligible_source_people,
        "women_source_people": len(women_source_people),
        "women_source_people_min_images": eligible_women_source_people,
        "strict_outfits": len(strict_outfit_targets),
        "strict_outfits_min_images": eligible_strict_outfits,
        "women_strict_outfits": len(women_strict_outfits),
        "women_strict_outfits_min_images": eligible_women_strict_outfits,
        "gender_counts": dict(gender_counts),
        "top_categories": dict(category_counts.most_common(12)),
        "min_images_per_person": min_images_per_person,
        "max_images_per_source_person": max(source_images_per_person.values(), default=0),
        "max_images_per_strict_outfit": max(strict_images_per_outfit.values(), default=0),
        "parse_failures": parse_failures,
    }


def render_report(results: dict[str, dict[str, Any]], min_images_per_person: int) -> str:
    lines = [
        "# Parts2Whole Metadata Audit",
        "",
        "This audit uses JSONL metadata only and does not download or inspect media files.",
        "",
        f"Minimum images/person threshold: `{min_images_per_person}`",
        "",
        "| Split | Rows | Unique targets | Source people | Source people >= threshold | WOMEN source people | WOMEN source people >= threshold | Strict outfit groups | WOMEN strict groups >= threshold | Parse failures |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for split, result in results.items():
        lines.append(
            f"| `{split}` | {result['rows']:,} | {result['unique_targets']:,} | {result['source_people']:,} | "
            f"{result['source_people_min_images']:,} | {result['women_source_people']:,} | "
            f"{result['women_source_people_min_images']:,} | {result['strict_outfits']:,} | "
            f"{result['women_strict_outfits_min_images']:,} | {result['parse_failures']:,} |"
        )

    total_women_source_people = sum(result["women_source_people"] for result in results.values())
    total_eligible_women_source = sum(result["women_source_people_min_images"] for result in results.values())
    total_eligible_women_strict = sum(result["women_strict_outfits_min_images"] for result in results.values())
    lines.extend(
        [
            "",
            "## Decision Notes",
            "",
            f"- WOMEN-labeled loose source people across splits: `{total_women_source_people:,}`.",
            f"- WOMEN-labeled loose source people with at least `{min_images_per_person}` images: `{total_eligible_women_source:,}`.",
            f"- WOMEN-labeled strict same-outfit groups with at least `{min_images_per_person}` images: `{total_eligible_women_strict:,}`.",
            "- These are dataset labels, not self-identified gender labels.",
            "- Loose source people may combine multiple outfits and depend on original DeepFashion IDs; strict groups are safer but too sparse.",
            "- This source still cannot satisfy a 10,000-person WOMEN 10+ images/person target by itself.",
            "- Media import still requires rights review, face-redaction audit, and disk preflight.",
            "",
            "## Top Categories",
            "",
        ]
    )
    for split, result in results.items():
        lines.append(f"### `{split}`")
        for category, count in result["top_categories"].items():
            lines.append(f"- `{category}`: {count:,}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--min-images-per-person", type=int, default=10)
    args = parser.parse_args()

    if args.min_images_per_person < 1:
        raise SystemExit("--min-images-per-person must be positive")

    paths = {
        "train": args.raw_root / "train.jsonl",
        "test": args.raw_root / "test.jsonl",
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise SystemExit(f"missing metadata files: {', '.join(missing)}")

    results = {split: audit_file(path, args.min_images_per_person) for split, path in paths.items()}
    report = render_report(results, args.min_images_per_person)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    print(report)
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
