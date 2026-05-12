"""Create a participant intake roster for Female Clothing Style Dataset v1.

The roster reserves pseudonymous person/image IDs for consented collection. It
does not contain names, contact info, raw images, or any identifying fields.
"""

# intent: prepare the real 10k x 10 collection without fabricating people/images
# status: done
# next: connect generated IDs to a separate restricted consent CRM outside training data
# blockers: actual participants must consent before any image rows are filled
# confidence: high

from __future__ import annotations

import argparse
import csv
from pathlib import Path


DEFAULT_PEOPLE = 10_000
DEFAULT_IMAGES_PER_PERSON = 10


def person_id(index: int) -> str:
    return f"p_{index:06d}"


def image_id(person: str, slot: int) -> str:
    return f"{person}_{slot:02d}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("research/datasets/female_clothing_style_v1/intake_roster.csv"))
    parser.add_argument("--people", type=int, default=DEFAULT_PEOPLE)
    parser.add_argument("--images-per-person", type=int, default=DEFAULT_IMAGES_PER_PERSON)
    args = parser.parse_args()

    if args.people <= 0 or args.images_per_person <= 0:
        raise SystemExit("people and images-per-person must be positive")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "person_id",
                "image_id",
                "slot",
                "collection_status",
                "consent_status",
                "redaction_status",
                "label_status",
            ],
        )
        writer.writeheader()
        for person_index in range(1, args.people + 1):
            pid = person_id(person_index)
            for slot in range(1, args.images_per_person + 1):
                writer.writerow(
                    {
                        "person_id": pid,
                        "image_id": image_id(pid, slot),
                        "slot": slot,
                        "collection_status": "pending",
                        "consent_status": "pending",
                        "redaction_status": "pending",
                        "label_status": "pending",
                    }
                )

    print(f"created roster: people={args.people:,} images={args.people * args.images_per_person:,} -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
