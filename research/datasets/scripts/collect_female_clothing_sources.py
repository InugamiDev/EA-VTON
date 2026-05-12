"""Safe source collector for Female Clothing Style Dataset v1.

This script intentionally refuses arbitrary web scraping. It reads the approved
source registry and either prepares manual-access instructions or runs local-only
pilot collection. Direct downloads can be added only when a source has explicit
license approval and a fixed allowlisted URL.
"""

# intent: collect only from approved/consented sources, never from arbitrary people on the web
# status: done
# next: add source-specific importers after licenses/access are approved
# blockers: public fashion datasets usually require registration or manual terms acceptance
# confidence: high

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any


DEFAULT_REGISTRY = Path("research/datasets/female_clothing_style_v1/sources.json")
DEFAULT_OUTPUT_ROOT = Path("research/datasets/raw/female_clothing_style_v1")


def load_registry(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def source_by_id(registry: dict[str, Any], source_id: str) -> dict[str, Any]:
    for source in registry.get("sources", []):
        if source.get("id") == source_id:
            return source
    raise SystemExit(f"unknown source '{source_id}'")


def write_access_required(source: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    content = [
        f"# Access Required: {source.get('name', source.get('id'))}",
        "",
        f"Source ID: `{source.get('id')}`",
        f"Status: `{source.get('status')}`",
        f"Access type: `{source.get('access_type')}`",
        f"License status: `{source.get('license_status')}`",
        f"Homepage: {source.get('homepage', 'n/a')}",
        "",
        "This source requires manual access or license verification before collection.",
        "Do not bypass registration, terms, robots.txt, or download gates.",
        "",
        f"Notes: {source.get('notes', '')}",
        "",
    ]
    (output_dir / "ACCESS_REQUIRED.md").write_text("\n".join(content), encoding="utf-8")


def write_not_approved(source: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    blocked_tasks = source.get("blocked_tasks") or []
    blocked_lines = [f"- `{task}`" for task in blocked_tasks]
    reported_scale = source.get("reported_scale")
    scale_lines = []
    if isinstance(reported_scale, dict) and reported_scale:
        scale_lines = ["", "Reported scale:", *[f"- `{key}`: {value}" for key, value in sorted(reported_scale.items())]]
    content = [
        f"# Not Approved For Collection: {source.get('name', source.get('id'))}",
        "",
        f"Source ID: `{source.get('id')}`",
        f"Status: `{source.get('status')}`",
        f"Access type: `{source.get('access_type')}`",
        f"License status: `{source.get('license_status')}`",
        f"Homepage: {source.get('homepage', 'n/a')}",
        "",
        "This source is tracked for research only. Do not download or import it into the people-image manifest until rights, privacy, gender-label, redaction, and task-fit reviews pass.",
        "",
        "Blocked tasks:",
        *(blocked_lines or ["- n/a"]),
        *scale_lines,
        "",
        f"Notes: {source.get('notes', '')}",
        "",
    ]
    (output_dir / "NOT_APPROVED.md").write_text("\n".join(content), encoding="utf-8")


def collect_local_catalog(output_dir: Path) -> None:
    catalog_path = Path("apps/api/src/data/garments.json")
    if not catalog_path.exists():
        raise SystemExit(f"local catalog not found: {catalog_path}")
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(catalog_path, output_dir / "garments.json")
    (output_dir / "README.md").write_text(
        "# Local Catalog Pilot\n\n"
        "Copied local garment metadata for pipeline smoke tests. This is not a people dataset and cannot satisfy the 10k x 10 target.\n",
        encoding="utf-8",
    )


def collect_source(source: dict[str, Any], output_root: Path) -> None:
    source_id = str(source["id"])
    output_dir = output_root / source_id
    status = source.get("status")

    if source_id == "participant_intake_v1":
        write_access_required(source, output_dir)
        print(f"prepared participant intake instructions -> {output_dir}")
        return

    if source_id == "local_catalog_pilot":
        collect_local_catalog(output_dir)
        print(f"collected local catalog pilot -> {output_dir}")
        return

    if status in {"available_huggingface", "metadata_downloaded_media_requires_storage"}:
        output_dir.mkdir(parents=True, exist_ok=True)
        storage_note = ""
        if status == "metadata_downloaded_media_requires_storage":
            storage_note = "\nFull media requires additional local storage before image import.\n"
        (output_dir / "SOURCE_NOTES.md").write_text(
            f"# {source.get('name', source_id)}\n\n"
            "This approved source can be downloaded from its fixed Hugging Face dataset URL after license review. "
            "Use a source-specific importer rather than arbitrary scraping.\n\n"
            f"{storage_note}"
            f"Homepage: {source.get('homepage', 'n/a')}\n\n"
            f"License status: `{source.get('license_status')}`\n\n"
            f"Notes: {source.get('notes', '')}\n",
            encoding="utf-8",
        )
        print(f"prepared Hugging Face source notes -> {output_dir}")
        return

    if status == "available_partial_direct":
        output_dir.mkdir(parents=True, exist_ok=True)
        direct_files = source.get("direct_files") or {}
        file_lines = [f"- `{name}`: {url}" for name, url in sorted(direct_files.items())]
        (output_dir / "SOURCE_NOTES.md").write_text(
            f"# {source.get('name', source_id)}\n\n"
            "This source has small direct files plus larger form-gated category files. "
            "It is registered for product/review metadata only; user-posted images must not enter the people-image training manifest without separate rights review.\n\n"
            f"Homepage: {source.get('homepage', 'n/a')}\n\n"
            f"License status: `{source.get('license_status')}`\n\n"
            "Direct files:\n"
            + "\n".join(file_lines)
            + "\n\n"
            f"Notes: {source.get('notes', '')}\n",
            encoding="utf-8",
        )
        print(f"prepared partial direct source notes -> {output_dir}")
        return

    if status == "manual_access_required":
        write_access_required(source, output_dir)
        print(f"manual access required -> {output_dir}")
        return

    if status in {
        "candidate_blocked_pending_terms_gender_storage_review",
        "candidate_blocked_pending_terms_privacy_gender_storage_review",
        "rejected_for_target_research_only_candidate",
        "research_index_only",
    }:
        write_not_approved(source, output_dir)
        print(f"not approved for collection -> {output_dir}")
        return

    raise SystemExit(f"source '{source_id}' has unsupported status '{status}'")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_id", help="source id from sources.json, or 'all'")
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args()

    registry = load_registry(args.registry)
    if registry.get("policy", {}).get("no_arbitrary_web_scraping") is not True:
        raise SystemExit("registry policy must explicitly block arbitrary web scraping")

    if args.source_id == "all":
        for source in registry.get("sources", []):
            collect_source(source, args.output_root)
    else:
        collect_source(source_by_id(registry, args.source_id), args.output_root)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
