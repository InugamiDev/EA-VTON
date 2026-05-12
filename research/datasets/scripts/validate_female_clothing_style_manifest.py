"""Validate Female Clothing Style Dataset v1 JSONL manifests.

Uses only the Python standard library so validation can run before installing a
training environment. It performs schema-adjacent checks plus semantic checks that
JSON Schema cannot express cleanly, especially category/family compatibility.
"""

# intent: enforce female-clothing, face-redacted, properly classified manifests with honest provenance
# status: done
# next: add optional jsonschema validation in CI if jsonschema becomes a project dependency
# blockers: none
# confidence: high

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


TARGET_PEOPLE = 10_000
IMAGES_PER_PERSON = 10

CATEGORIES_BY_FAMILY: dict[str, set[str]] = {
    "top": {"t_shirt", "shirt", "blouse", "polo", "sweater", "hoodie", "cardigan", "tank", "camisole", "tunic", "bodysuit", "vest_top"},
    "bottom": {"jeans", "trousers", "leggings", "shorts", "skirt", "culottes", "joggers", "sweatpants"},
    "dress_jumpsuit": {"mini_dress", "midi_dress", "maxi_dress", "shift_dress", "wrap_dress", "bodycon_dress", "shirt_dress", "jumpsuit", "romper", "overalls"},
    "outerwear": {"blazer", "denim_jacket", "leather_jacket", "bomber", "coat", "trench", "parka", "puffer", "kimono", "shacket"},
    "footwear": {"sneakers", "flats", "loafers", "boots", "heels", "sandals", "mules", "slides"},
    "accessory": {"bag", "belt", "scarf", "hat", "jewelry", "glasses", "watch", "hair_accessory"},
}

ALLOWED_SPLITS = {"train", "val", "test", "holdout"}
ALLOWED_AGE_BANDS = {"18_24", "25_34", "35_44", "45_54", "55_64", "65_plus", "unknown"}
ALLOWED_GENDER_LABEL_SOURCES = {"self_identified", "dataset_label"}
ALLOWED_RIGHTS_BASIS = {"first_party_consent", "dataset_license"}
ALLOWED_RIGHTS_STATUS = {"granted", "licensed"}
ALLOWED_REDACTION_STATUS = {"redacted", "no_face_visible"}
ALLOWED_LAYER_ROLES = {"base", "mid", "outer", "full_body", "footwear", "accessory"}
ALLOWED_VISIBILITY = {"primary", "secondary", "partial"}
ALLOWED_OCCASIONS = {"casual", "work", "date", "formal", "party", "athleisure", "lounge", "outdoor", "travel", "unknown"}
ALLOWED_FORMALITY = {"very_casual", "casual", "smart_casual", "business", "formal", "unknown"}
ALLOWED_COLOR_MOOD = {"monochrome", "neutral", "warm", "cool", "colorful", "high_contrast", "unknown"}
ALLOWED_LAYERING = {"single_layer", "light_layering", "heavy_layering", "unknown"}

REQUIRED_ATTRIBUTE_KEYS = {"fit", "silhouette", "color_family", "pattern", "fabric_weight", "structure", "material_tags"}
FORBIDDEN_TRAINING_KEYS = {"source_image_uri", "raw_image_uri", "source_pid", "pid", "profile_url", "username", "name", "email", "phone"}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
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
            record["__line_number"] = line_number
            records.append(record)
    return records


def require(condition: bool, errors: list[str], line: int, message: str) -> None:
    if not condition:
        errors.append(f"line {line}: {message}")


def validate_garment(garment: dict[str, Any], errors: list[str], line: int, index: int) -> None:
    prefix = f"garments[{index}]"
    family = garment.get("family")
    category = garment.get("category")
    attributes = garment.get("attributes") or {}

    require(family in CATEGORIES_BY_FAMILY, errors, line, f"{prefix}.family is invalid")
    if family in CATEGORIES_BY_FAMILY:
        require(category in CATEGORIES_BY_FAMILY[str(family)], errors, line, f"{prefix}.category '{category}' is not valid for family '{family}'")
    require(garment.get("visibility") in ALLOWED_VISIBILITY, errors, line, f"{prefix}.visibility is invalid")
    require(garment.get("layer_role") in ALLOWED_LAYER_ROLES, errors, line, f"{prefix}.layer_role is invalid")
    require(isinstance(garment.get("garment_id"), str) and str(garment.get("garment_id")).startswith("g"), errors, line, f"{prefix}.garment_id must be gNN")
    require(isinstance(attributes, dict), errors, line, f"{prefix}.attributes must be an object")

    missing = REQUIRED_ATTRIBUTE_KEYS - set(attributes)
    require(not missing, errors, line, f"{prefix}.attributes missing required keys: {sorted(missing)}")
    material_tags = attributes.get("material_tags")
    require(isinstance(material_tags, list) and len(material_tags) > 0, errors, line, f"{prefix}.attributes.material_tags must be non-empty")

    if family == "dress_jumpsuit":
        require(garment.get("layer_role") == "full_body", errors, line, f"{prefix}.layer_role should be full_body for dress_jumpsuit")
    if family == "footwear":
        require(garment.get("layer_role") == "footwear", errors, line, f"{prefix}.layer_role should be footwear for footwear")
    if family == "accessory":
        require(garment.get("layer_role") == "accessory", errors, line, f"{prefix}.layer_role should be accessory for accessories")


def validate_record(record: dict[str, Any], errors: list[str]) -> None:
    line = int(record.get("__line_number", 0))
    person_id = record.get("person_id")
    image_id = record.get("image_id")
    garments = record.get("garments") or []
    outfit = record.get("outfit") or {}
    quality = record.get("quality") or {}
    rights = record.get("rights") or {}
    gender_label = record.get("gender_label") or {}
    face_redaction = record.get("face_redaction") or {}

    for key in FORBIDDEN_TRAINING_KEYS:
        require(key not in record, errors, line, f"training manifest must not include identifying/raw key '{key}'")

    require(isinstance(person_id, str) and person_id.startswith("p_"), errors, line, "person_id must be a pseudonymous p_* id")
    require(isinstance(image_id, str) and image_id.startswith(f"{person_id}_"), errors, line, "image_id must be scoped under person_id")
    require(record.get("split") in ALLOWED_SPLITS, errors, line, "split is invalid")
    require(isinstance(gender_label, dict), errors, line, "gender_label must be an object")
    require(gender_label.get("value") == "female", errors, line, "gender_label.value must be female")
    require(gender_label.get("source") in ALLOWED_GENDER_LABEL_SOURCES, errors, line, "gender_label.source is invalid")
    require(record.get("age_band") in ALLOWED_AGE_BANDS, errors, line, "age_band is invalid")
    require(bool(record.get("redacted_image_uri")), errors, line, "redacted_image_uri is required")
    require(isinstance(rights, dict), errors, line, "rights must be an object")
    require(rights.get("basis") in ALLOWED_RIGHTS_BASIS, errors, line, "rights.basis is invalid")
    require(rights.get("status") in ALLOWED_RIGHTS_STATUS, errors, line, "rights.status is invalid")
    if rights.get("basis") == "first_party_consent":
        require(rights.get("status") == "granted", errors, line, "first_party_consent rights require status=granted")
    if rights.get("basis") == "dataset_license":
        require(rights.get("status") == "licensed", errors, line, "dataset_license rights require status=licensed")
    require(rights.get("scope") in {"style_rec_training", "research_only"}, errors, line, "rights.scope is invalid")
    require(face_redaction.get("status") in ALLOWED_REDACTION_STATUS, errors, line, "face_redaction.status must be redacted or no_face_visible")
    require(quality.get("usable") is True, errors, line, "quality.usable must be true")
    require(quality.get("garments_visible") is True, errors, line, "quality.garments_visible must be true")
    require(quality.get("face_visible_after_redaction") is False, errors, line, "face must not be visible after redaction")

    require(isinstance(garments, list) and len(garments) > 0, errors, line, "garments must be a non-empty array")
    primary_count = 0
    for index, garment in enumerate(garments):
        if isinstance(garment, dict):
            if garment.get("visibility") == "primary":
                primary_count += 1
            validate_garment(garment, errors, line, index)
        else:
            errors.append(f"line {line}: garments[{index}] must be an object")
    require(primary_count >= 1, errors, line, "at least one garment must be primary")

    require(outfit.get("occasion") in ALLOWED_OCCASIONS, errors, line, "outfit.occasion is invalid")
    require(outfit.get("formality") in ALLOWED_FORMALITY, errors, line, "outfit.formality is invalid")
    require(outfit.get("color_mood") in ALLOWED_COLOR_MOOD, errors, line, "outfit.color_mood is invalid")
    require(outfit.get("layering") in ALLOWED_LAYERING, errors, line, "outfit.layering is invalid")
    require(isinstance(outfit.get("aesthetic_tags"), list) and len(outfit.get("aesthetic_tags", [])) > 0, errors, line, "outfit.aesthetic_tags must be non-empty")


def validate_counts(records: list[dict[str, Any]], errors: list[str], strict_counts: bool) -> None:
    image_ids = [str(record.get("image_id")) for record in records]
    duplicate_images = [image_id for image_id, count in Counter(image_ids).items() if count > 1]
    if duplicate_images:
        errors.append(f"duplicate image_id values: {duplicate_images[:10]}")

    by_person: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_person[str(record.get("person_id"))].append(record)

    if strict_counts:
        if len(by_person) != TARGET_PEOPLE:
            errors.append(f"expected {TARGET_PEOPLE} people, found {len(by_person)}")
        bad_counts = {person_id: len(rows) for person_id, rows in by_person.items() if len(rows) != IMAGES_PER_PERSON}
        if bad_counts:
            sample = dict(list(bad_counts.items())[:10])
            errors.append(f"each person must have {IMAGES_PER_PERSON} images; sample bad counts: {sample}")


def print_summary(records: list[dict[str, Any]]) -> None:
    split_counts = Counter(str(record.get("split")) for record in records)
    family_counts: Counter[str] = Counter()
    category_counts: Counter[str] = Counter()
    gender_sources = Counter(str((record.get("gender_label") or {}).get("source")) for record in records)
    rights_basis = Counter(str((record.get("rights") or {}).get("basis")) for record in records)
    for record in records:
        for garment in record.get("garments") or []:
            family_counts[str(garment.get("family"))] += 1
            category_counts[str(garment.get("category"))] += 1
    people = {str(record.get("person_id")) for record in records}
    print(f"records={len(records):,} people={len(people):,} splits={dict(split_counts)}")
    print(f"gender_label_sources={dict(gender_sources)} rights_basis={dict(rights_basis)}")
    print(f"garment_families={dict(family_counts)}")
    print(f"top_categories={dict(category_counts.most_common(12))}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--strict-counts", action="store_true", help="require 10,000 people and exactly 10 images per person")
    args = parser.parse_args()

    records = load_jsonl(args.manifest)
    errors: list[str] = []
    for record in records:
        validate_record(record, errors)
    validate_counts(records, errors, args.strict_counts)

    print_summary(records)
    if errors:
        print(f"validation failed with {len(errors)} error(s):", file=sys.stderr)
        for error in errors[:50]:
            print(f"- {error}", file=sys.stderr)
        if len(errors) > 50:
            print(f"... {len(errors) - 50} more", file=sys.stderr)
        return 1

    print("validation passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
