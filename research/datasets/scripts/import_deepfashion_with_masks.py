"""Import Hugging Face DeepFashion-with-masks parquet shards into a pilot manifest.

The source uses WOMEN/MEN dataset labels rather than first-party self-ID, so
records are marked as licensed external data with `gender_label.source` set to
`dataset_label`. The importer exports only face-redacted images and never writes
source pids or raw paths into the training manifest.
"""

# intent: convert an approved licensed external fashion dataset into privacy-safe pilot manifests
# status: done
# next: manually audit redaction quality before using these rows for production training
# blockers: source provides no face boxes, so redaction uses a conservative top-band mask
# confidence: medium

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections import Counter, defaultdict
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Any


DEFAULT_RAW_ROOT = Path("research/datasets/raw/female_clothing_style_v1/deepfashion_with_masks")
DEFAULT_DATASET_ROOT = Path("research/datasets/female_clothing_style_v1")
DEFAULT_MANIFEST = DEFAULT_DATASET_ROOT / "manifest_deepfashion_with_masks_pilot.jsonl"
DEFAULT_IMAGE_OUTPUT_ROOT = DEFAULT_DATASET_ROOT / "redacted/deepfashion_with_masks"


def load_dependencies():
    try:
        import pyarrow.parquet as pq
        from PIL import Image, ImageDraw
    except ImportError as exc:
        raise SystemExit(
            "pyarrow and Pillow are required. Use the project service venv or install them before importing."
        ) from exc
    return pq, Image, ImageDraw


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


def stable_hex(value: str, length: int = 12) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()[:length]


def make_person_id(source_pid: str) -> str:
    return f"p_{stable_hex('deepfashion_with_masks:' + source_pid)}"


def split_for_person(person_id: str) -> str:
    bucket = int(stable_hex(person_id, 8), 16) % 100
    if bucket < 80:
        return "train"
    if bucket < 90:
        return "val"
    return "test"


COLOR_KEYWORDS: list[tuple[str, tuple[str, ...]]] = [
    ("black", ("black",)),
    ("white", ("white",)),
    ("gray", ("gray", "grey")),
    ("beige", ("beige", "cream", "tan", "khaki")),
    ("brown", ("brown", "camel")),
    ("navy", ("navy",)),
    ("blue", ("blue",)),
    ("green", ("green",)),
    ("olive", ("olive",)),
    ("yellow", ("yellow",)),
    ("orange", ("orange",)),
    ("red", ("red", "burgundy", "maroon")),
    ("pink", ("pink",)),
    ("purple", ("purple", "lavender", "violet")),
    ("metallic", ("gold", "silver", "metallic")),
]

PATTERN_KEYWORDS: list[tuple[str, tuple[str, ...]]] = [
    ("stripe", ("stripe", "striped")),
    ("plaid", ("plaid", "tartan")),
    ("check", ("check", "checked", "checkered", "gingham")),
    ("floral", ("floral", "flower", "flowers")),
    ("graphic", ("graphic", "printed tee")),
    ("animal", ("leopard", "zebra", "animal")),
    ("logo", ("logo",)),
    ("colorblock", ("colorblock", "color block")),
    ("abstract", ("print", "printed", "pattern")),
]


def caption_words(caption: str) -> str:
    return caption.lower().replace("-", "_")


def color_from_caption(caption: str) -> str:
    lowered = caption_words(caption)
    found = [color for color, tokens in COLOR_KEYWORDS if any(token in lowered for token in tokens)]
    if len(set(found)) > 1:
        return "multi"
    return found[0] if found else "unknown"


def pattern_from_caption(caption: str, cloth_type: str) -> str:
    lowered = caption_words(f"{caption} {cloth_type}")
    for pattern, tokens in PATTERN_KEYWORDS:
        if any(token in lowered for token in tokens):
            return pattern
    return "solid"


def material_tags_for(cloth_type: str, caption: str) -> list[str]:
    lowered = caption_words(f"{caption} {cloth_type}")
    tags: list[str] = []
    if "denim" in lowered or "jeans" in lowered:
        tags.extend(["denim", "cotton"])
    if "sweater" in lowered or "cardigan" in lowered or "knit" in lowered:
        tags.append("knit")
    if "hoodie" in lowered or "sweatshirt" in lowered or "fleece" in lowered:
        tags.append("fleece")
    if "lace" in lowered:
        tags.append("lace")
    if "silk" in lowered or "satin" in lowered:
        tags.append("satin")
    if "leather" in lowered:
        tags.append("leather")
    if "chiffon" in lowered:
        tags.append("chiffon")
    if not tags and any(token in lowered for token in ("tee", "t_shirt", "tank", "shirt", "blouse", "dress")):
        tags.append("cotton")
    return list(dict.fromkeys(tags or ["unknown"]))


def map_category(cloth_type: str, caption: str) -> tuple[str, str, str]:
    lowered = caption_words(f"{caption} {cloth_type}")
    if cloth_type == "Tees_Tanks":
        if "camisole" in lowered:
            return "top", "camisole", "base"
        if "tank" in lowered:
            return "top", "tank", "base"
        return "top", "t_shirt", "base"
    if cloth_type == "Graphic_Tees":
        return "top", "t_shirt", "base"
    if cloth_type == "Blouses_Shirts":
        return "top", "blouse" if "blouse" in lowered else "shirt", "base"
    if cloth_type == "Shirts_Polos":
        return "top", "polo" if "polo" in lowered else "shirt", "base"
    if cloth_type == "Sweaters":
        return "top", "sweater", "mid"
    if cloth_type == "Sweatshirts_Hoodies":
        return "top", "hoodie", "mid"
    if cloth_type == "Cardigans":
        return "top", "cardigan", "mid"
    if cloth_type == "Shorts":
        return "bottom", "shorts", "base"
    if cloth_type == "Skirts":
        return "bottom", "skirt", "base"
    if cloth_type == "Pants":
        return "bottom", "trousers", "base"
    if cloth_type == "Leggings":
        return "bottom", "leggings", "base"
    if cloth_type == "Denim":
        if "jacket" in lowered:
            return "outerwear", "denim_jacket", "outer"
        if "skirt" in lowered:
            return "bottom", "skirt", "base"
        return "bottom", "jeans", "base"
    if cloth_type == "Dresses":
        if "maxi" in lowered:
            return "dress_jumpsuit", "maxi_dress", "full_body"
        if "mini" in lowered:
            return "dress_jumpsuit", "mini_dress", "full_body"
        if "wrap" in lowered:
            return "dress_jumpsuit", "wrap_dress", "full_body"
        return "dress_jumpsuit", "midi_dress", "full_body"
    if cloth_type == "Rompers_Jumpsuits":
        return "dress_jumpsuit", "romper" if "romper" in lowered else "jumpsuit", "full_body"
    if cloth_type == "Jackets_Coats":
        if "denim" in lowered:
            return "outerwear", "denim_jacket", "outer"
        if "leather" in lowered:
            return "outerwear", "leather_jacket", "outer"
        if "trench" in lowered:
            return "outerwear", "trench", "outer"
        if "parka" in lowered:
            return "outerwear", "parka", "outer"
        if "puffer" in lowered:
            return "outerwear", "puffer", "outer"
        return "outerwear", "coat", "outer"
    if cloth_type == "Jackets_Vests":
        if "vest" in lowered:
            return "top", "vest_top", "mid"
        if "denim" in lowered:
            return "outerwear", "denim_jacket", "outer"
        if "leather" in lowered:
            return "outerwear", "leather_jacket", "outer"
        return "outerwear", "bomber", "outer"
    if cloth_type == "Suiting":
        if "pant" in lowered or "trouser" in lowered:
            return "bottom", "trousers", "base"
        return "outerwear", "blazer", "outer"
    return "top", "shirt", "base"


def attributes_for(family: str, category: str, cloth_type: str, caption: str) -> dict[str, Any]:
    lowered = caption_words(f"{caption} {cloth_type}")
    pattern = pattern_from_caption(caption, cloth_type)
    attributes: dict[str, Any] = {
        "fit": "unknown",
        "silhouette": "unknown",
        "color_family": color_from_caption(caption),
        "pattern": pattern,
        "fabric_weight": "medium",
        "structure": "unknown",
        "material_tags": material_tags_for(cloth_type, caption),
    }

    if family == "top":
        attributes.update({"hem_length": "regular", "sleeve_length": "unknown", "closure": "pullover"})
        if category in {"shirt", "blouse", "polo"}:
            attributes.update({"neckline": "collared", "closure": "button", "structure": "structured"})
        elif category == "hoodie":
            attributes.update({"neckline": "hooded", "fabric_weight": "heavy", "structure": "knit"})
        elif category in {"sweater", "cardigan"}:
            attributes.update({"fabric_weight": "heavy", "structure": "knit"})
        elif category in {"tank", "camisole"}:
            attributes.update({"neckline": "scoop", "sleeve_length": "sleeveless", "fabric_weight": "light", "structure": "soft"})
        else:
            attributes.update({"neckline": "crew", "fabric_weight": "light", "structure": "soft"})
    elif family == "bottom":
        attributes.update({"rise": "unknown", "leg_shape": "not_applicable", "hem_length": "regular", "closure": "unknown"})
        if category == "jeans":
            attributes.update({"leg_shape": "straight", "closure": "zip", "structure": "structured"})
        elif category in {"trousers", "leggings"}:
            attributes.update({"leg_shape": "straight" if category == "trousers" else "skinny", "structure": "stretch"})
        elif category == "skirt":
            attributes.update({"silhouette": "a_line", "hem_length": "unknown"})
    elif family == "dress_jumpsuit":
        attributes.update({"hem_length": "midi", "closure": "unknown", "neckline": "unknown", "sleeve_length": "unknown", "structure": "drapey"})
        if category == "mini_dress":
            attributes["hem_length"] = "mini"
        elif category == "maxi_dress":
            attributes["hem_length"] = "maxi"
        if "bodycon" in lowered:
            attributes.update({"fit": "bodycon", "silhouette": "straight"})
        if category == "wrap_dress":
            attributes.update({"silhouette": "wrap", "closure": "wrap"})
    elif family == "outerwear":
        attributes.update({"fabric_weight": "heavy", "structure": "structured", "closure": "zip", "hem_length": "regular"})
        if category in {"blazer", "trench", "coat"}:
            attributes.update({"closure": "button", "silhouette": "tailored"})

    if "oversized" in lowered:
        attributes["fit"] = "oversized"
    elif "fitted" in lowered or "tight" in lowered:
        attributes["fit"] = "fitted"

    if pattern == "graphic":
        attributes["pattern"] = "graphic"
    return attributes


def outfit_for(family: str, category: str, attributes: dict[str, Any]) -> dict[str, Any]:
    tags = ["casual_basic"]
    if category in {"shirt", "blouse", "polo", "blazer", "trousers"}:
        tags = ["classic", "minimal"]
    elif family == "dress_jumpsuit":
        tags = ["classic", "romantic"]
    elif category in {"hoodie", "t_shirt", "jeans"}:
        tags = ["casual_basic", "streetwear"]
    elif category in {"sweater", "cardigan"}:
        tags = ["soft", "casual_basic"]

    color_family = attributes.get("color_family")
    color_mood = "neutral" if color_family in {"black", "white", "gray", "beige", "brown", "navy"} else "unknown"
    if color_family == "multi":
        color_mood = "colorful"

    layering = "light_layering" if family == "outerwear" or category in {"cardigan", "vest_top"} else "single_layer"
    return {
        "occasion": "casual",
        "formality": "casual",
        "aesthetic_tags": tags,
        "color_mood": color_mood,
        "layering": layering,
    }


def redact_and_save(image_bytes: bytes, destination: Path, top_ratio: float, Image: Any, ImageDraw: Any) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    width, height = image.size
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, width, int(round(height * top_ratio))), fill=(0, 0, 0))
    image.save(destination, format="JPEG", quality=92)


def relative_uri(path: Path, dataset_root: Path) -> str:
    try:
        return str(path.relative_to(dataset_root))
    except ValueError:
        return str(path)


def build_record(
    row: dict[str, Any],
    person_id: str,
    image_id: str,
    redacted_uri: str,
    imported_at: str,
) -> dict[str, Any]:
    cloth_type = str(row.get("cloth_type") or "")
    caption = str(row.get("caption") or "")
    family, category, layer_role = map_category(cloth_type, caption)
    attributes = attributes_for(family, category, cloth_type, caption)
    return {
        "person_id": person_id,
        "image_id": image_id,
        "split": split_for_person(person_id),
        "gender_label": {
            "value": "female",
            "source": "dataset_label",
            "notes": "Source row gender label is WOMEN; not first-party self-identification.",
        },
        "age_band": "unknown",
        "redacted_image_uri": redacted_uri,
        "face_redaction": {"status": "redacted", "method": "black_box"},
        "rights": {
            "basis": "dataset_license",
            "status": "licensed",
            "scope": "research_only",
            "version": "hf_metadata_2026-05-07",
            "license": "apache-2.0",
            "source": "SaffalPoosh/deepFashion-with-masks",
        },
        "garments": [
            {
                "garment_id": "g01",
                "family": family,
                "category": category,
                "visibility": "primary",
                "layer_role": layer_role,
                "attributes": attributes,
            }
        ],
        "outfit": outfit_for(family, category, attributes),
        "quality": {
            "usable": True,
            "face_visible_after_redaction": False,
            "garments_visible": True,
            "notes": "Imported from licensed external dataset; face redacted with conservative top-band heuristic; audit before production training.",
        },
        "annotator": {"id": "deepfashion_with_masks_importer_v1", "labeled_at": imported_at},
    }


def iter_rows(parquet_paths: list[Path], batch_size: int):
    pq, _, _ = load_dependencies()
    columns = ["images", "gender", "pose", "cloth_type", "pid", "caption"]
    for parquet_path in parquet_paths:
        parquet_file = pq.ParquetFile(parquet_path)
        for batch in parquet_file.iter_batches(batch_size=batch_size, columns=columns):
            data = batch.to_pydict()
            for index in range(batch.num_rows):
                yield {column: data[column][index] for column in columns}


def count_source_people(parquet_paths: list[Path], batch_size: int, gender: str) -> Counter[str]:
    counts: Counter[str] = Counter()
    for row in iter_rows(parquet_paths, batch_size):
        if str(row.get("gender")) != gender:
            continue
        source_pid = str(row.get("pid") or "")
        if source_pid:
            counts[source_pid] += 1
    return counts


def import_manifest(args: argparse.Namespace) -> int:
    _, Image, ImageDraw = load_dependencies()
    data_dir = args.raw_root / "data"
    parquet_paths = sorted(data_dir.glob("*.parquet"))
    if not parquet_paths:
        raise SystemExit(f"no parquet shards found in {data_dir}")

    eligible_source_pids: set[str] | None = None
    if args.min_source_images_per_person > 1:
        source_counts = count_source_people(parquet_paths, args.batch_size, args.gender)
        eligible_source_pids = {pid for pid, count in source_counts.items() if count >= args.min_source_images_per_person}
        print(
            "eligible_source_people="
            f"{len(eligible_source_pids):,} min_source_images_per_person={args.min_source_images_per_person}"
        )

    imported_at = datetime.now(timezone.utc).isoformat()
    person_counts: defaultdict[str, int] = defaultdict(int)
    category_counts: Counter[str] = Counter()
    split_counts: Counter[str] = Counter()
    skipped_counts: Counter[str] = Counter()
    limit = args.limit if args.limit and args.limit > 0 else None

    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with args.manifest.open("w", encoding="utf-8") as out:
        for row in iter_rows(parquet_paths, args.batch_size):
            if str(row.get("gender")) != args.gender:
                skipped_counts["gender"] += 1
                continue
            source_pid = str(row.get("pid") or "")
            if not source_pid:
                skipped_counts["missing_pid"] += 1
                continue
            if eligible_source_pids is not None and source_pid not in eligible_source_pids:
                skipped_counts["min_source_images_per_person"] += 1
                continue
            person_id = make_person_id(source_pid)
            if person_counts[person_id] >= args.max_per_person:
                skipped_counts["max_per_person"] += 1
                continue

            image_payload = row.get("images") or {}
            image_bytes = image_payload.get("bytes") if isinstance(image_payload, dict) else None
            if not image_bytes:
                skipped_counts["missing_image_bytes"] += 1
                continue

            person_counts[person_id] += 1
            image_id = f"{person_id}_{person_counts[person_id]:02d}"
            destination = args.image_output_root / f"{image_id}.jpg"
            redact_and_save(image_bytes, destination, args.redact_top_ratio, Image, ImageDraw)
            record = build_record(row, person_id, image_id, relative_uri(destination, args.dataset_root), imported_at)
            out.write(json.dumps(record, sort_keys=True) + "\n")
            category_counts[record["garments"][0]["category"]] += 1
            split_counts[record["split"]] += 1
            written += 1
            if limit is not None and written >= limit:
                break

    print(f"wrote records={written:,} people={len(person_counts):,} -> {args.manifest}")
    print(f"splits={dict(split_counts)}")
    print(f"top_categories={dict(category_counts.most_common(12))}")
    print(f"skipped={dict(skipped_counts)}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--image-output-root", type=Path, default=DEFAULT_IMAGE_OUTPUT_ROOT)
    parser.add_argument("--limit", type=int, default=1000, help="maximum rows to import; use 0 for all available rows")
    parser.add_argument("--max-per-person", type=int, default=10)
    parser.add_argument("--min-source-images-per-person", type=int, default=1)
    parser.add_argument("--gender", default="WOMEN")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--redact-top-ratio", type=float, default=0.28, help="top image fraction to black-box for heuristic face redaction")
    parser.add_argument("--min-free-gb", type=float, default=5.0, help="minimum free disk required before writing redacted images")
    args = parser.parse_args()

    if not 0.05 <= args.redact_top_ratio <= 0.5:
        raise SystemExit("--redact-top-ratio must be between 0.05 and 0.5")
    if args.max_per_person < 1 or args.max_per_person > 99:
        raise SystemExit("--max-per-person must be between 1 and 99")
    if args.min_source_images_per_person < 1 or args.min_source_images_per_person > 99:
        raise SystemExit("--min-source-images-per-person must be between 1 and 99")
    require_free_space(args.image_output_root, args.min_free_gb)
    return import_manifest(args)


if __name__ == "__main__":
    raise SystemExit(main())
