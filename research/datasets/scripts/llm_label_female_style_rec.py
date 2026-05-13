"""Label redacted clothing images with an OpenAI vision model.

This script uses the Responses API with a low-cost vision-capable model. It
never sends raw restricted images; inputs should come from redacted manifests.
The output is JSONL with model labels, confidence, and review flags.
"""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


DEFAULT_SOURCE = Path("research/datasets/processed/female_style_rec_v1/images_train.jsonl")
DEFAULT_OUTPUT = Path("research/datasets/annotation/female_style_rec_v1/llm_labels/labels_gpt5_nano_pilot.jsonl")
DEFAULT_MODEL = "gpt-5-nano"
DEFAULT_FALLBACK_MODEL = "gpt-4.1-nano"
RESPONSES_URL = "https://api.openai.com/v1/responses"

LABEL_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": [
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
        "confidence",
        "needs_review",
        "review_reason",
    ],
    "properties": {
        "family": {"type": "string"},
        "category": {"type": "string"},
        "occasion": {"type": "string"},
        "formality": {"type": "string"},
        "aesthetic_tags": {"type": "array", "items": {"type": "string"}},
        "color_mood": {"type": "string"},
        "layering": {"type": "string"},
        "fit": {"type": "string"},
        "silhouette": {"type": "string"},
        "color_family": {"type": "string"},
        "pattern": {"type": "string"},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "needs_review": {"type": "boolean"},
        "review_reason": {"type": "string"},
    },
}

SYSTEM_PROMPT = """You label redacted fashion images for a style recommendation dataset.
Only describe visible clothing and outfit style. Do not identify people, infer body
shape, infer ethnicity, infer age, infer location, or mention faces. If image quality
or redaction makes the outfit unclear, set needs_review=true and lower confidence.
Use concise snake_case labels compatible with fashion recommendation training."""


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as records:
        for line_number, line in enumerate(records, 1):
            line = line.strip()
            if line:
                yield line_number, json.loads(line)


def resolve_image_path(record: dict[str, Any]) -> Path:
    return Path("research/datasets/female_clothing_style_v1") / record["redacted_image_uri"]


def encode_image(path: Path, max_side: int) -> str:
    data = path.read_bytes()
    mime_type = mimetypes.guess_type(path.name)[0] or "image/jpeg"

    try:
        from PIL import Image
        import io

        image = Image.open(path).convert("RGB")
        image.thumbnail((max_side, max_side))
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=82, optimize=True)
        data = buffer.getvalue()
        mime_type = "image/jpeg"
    except Exception:
        pass

    encoded = base64.b64encode(data).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def prompt_for(record: dict[str, Any]) -> str:
    primary = (record.get("garments") or [{}])[0]
    attrs = primary.get("attributes") or {}
    outfit = record.get("outfit") or {}
    return (
        "Label the visible outfit for style recommendation. Return only JSON.\n"
        "Allowed broad families: top, bottom, dress_jumpsuit, outerwear, footwear, accessory.\n"
        "Common categories: t_shirt, shirt, blouse, sweater, hoodie, cardigan, tank, jeans, "
        "trousers, shorts, skirt, midi_dress, mini_dress, maxi_dress, romper, jumpsuit, coat, "
        "denim_jacket, leather_jacket, blazer, parka.\n"
        "Occasions: casual, work, date, formal, party, athleisure, lounge, outdoor, travel, unknown.\n"
        "Formality: very_casual, casual, smart_casual, business, formal, unknown.\n"
        "Aesthetic tags examples: minimal, classic, streetwear, preppy, romantic, sporty, soft, edgy, "
        "boho, business, luxe, casual_basic, colorful, monochrome, layered, vintage, y2k, modest, "
        "androgynous, glam.\n"
        "Use unknown when not visible.\n\n"
        "Weak existing labels to verify, not blindly copy:\n"
        f"- family={primary.get('family', 'unknown')}\n"
        f"- category={primary.get('category', 'unknown')}\n"
        f"- fit={attrs.get('fit', 'unknown')}\n"
        f"- silhouette={attrs.get('silhouette', 'unknown')}\n"
        f"- color_family={attrs.get('color_family', 'unknown')}\n"
        f"- pattern={attrs.get('pattern', 'unknown')}\n"
        f"- occasion={outfit.get('occasion', 'unknown')}\n"
        f"- formality={outfit.get('formality', 'unknown')}\n"
        f"- aesthetic_tags={outfit.get('aesthetic_tags', ['unknown'])}\n"
        f"- color_mood={outfit.get('color_mood', 'unknown')}\n"
        f"- layering={outfit.get('layering', 'unknown')}\n"
    )


def extract_text(response: dict[str, Any]) -> str:
    texts: list[str] = []
    for item in response.get("output", []):
        for content in item.get("content", []):
            if content.get("type") == "output_text":
                texts.append(content.get("text", ""))
    if texts:
        return "\n".join(texts)
    if "output_text" in response:
        return str(response["output_text"])
    raise RuntimeError(f"response did not include output text: {response}")


def call_openai(
    api_key: str,
    model: str,
    record: dict[str, Any],
    max_image_side: int,
    timeout: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    image_path = resolve_image_path(record)
    if not image_path.exists():
        raise FileNotFoundError(f"image not found: {image_path}")

    payload = {
        "model": model,
        "input": [
            {
                "role": "system",
                "content": [{"type": "input_text", "text": SYSTEM_PROMPT}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt_for(record)},
                    {"type": "input_image", "image_url": encode_image(image_path, max_image_side), "detail": "low"},
                ],
            },
        ],
        "text": {
            "format": {
                "type": "json_schema",
                "name": "style_rec_label",
                "schema": LABEL_SCHEMA,
                "strict": True,
            }
        },
        "max_output_tokens": 500,
    }
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        RESPONSES_URL,
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        raw = response.read().decode("utf-8")
    api_response = json.loads(raw)
    label = json.loads(extract_text(api_response))
    usage = api_response.get("usage") or {}
    return label, usage


def output_record(record: dict[str, Any], label: dict[str, Any], model: str, usage: dict[str, Any]) -> dict[str, Any]:
    return {
        "person_id": record["person_id"],
        "image_id": record["image_id"],
        "split": record["split"],
        "redacted_image_uri": record["redacted_image_uri"],
        "model": model,
        "label_source": "openai_vision_llm",
        "label": label,
        "usage": usage,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--fallback-model", default=DEFAULT_FALLBACK_MODEL)
    parser.add_argument("--limit", type=int, default=25, help="0 means all records")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--max-image-side", type=int, default=512)
    parser.add_argument("--sleep", type=float, default=0.2)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    records = [record for _, record in iter_jsonl(args.source)]
    selected = records[args.start :]
    if args.limit:
        selected = selected[: args.limit]
    if not selected:
        raise SystemExit("no records selected")

    if args.dry_run:
        print(f"would label records={len(selected):,} model={args.model} output={args.output}")
        print(f"first_image={resolve_image_path(selected[0])}")
        return 0

    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise SystemExit(f"{args.api_key_env} is not set")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    total_usage: dict[str, int] = {}
    with args.output.open("a", encoding="utf-8") as output:
        for index, record in enumerate(selected, args.start + 1):
            model = args.model
            try:
                label, usage = call_openai(api_key, model, record, args.max_image_side, args.timeout)
            except urllib.error.HTTPError as exc:
                if args.fallback_model and exc.code in {400, 404} and args.fallback_model != args.model:
                    model = args.fallback_model
                    label, usage = call_openai(api_key, model, record, args.max_image_side, args.timeout)
                else:
                    error_body = exc.read().decode("utf-8", errors="replace")
                    raise SystemExit(f"OpenAI request failed for {record.get('image_id')}: {exc.code} {error_body}") from exc
            result = output_record(record, label, model, usage)
            output.write(json.dumps(result, sort_keys=True) + "\n")
            output.flush()
            written += 1
            for key, value in usage.items():
                if isinstance(value, int):
                    total_usage[key] = total_usage.get(key, 0) + value
            print(f"[{written}/{len(selected)}] {record['image_id']} confidence={label.get('confidence')} model={model}")
            if args.sleep:
                time.sleep(args.sleep)

    stats_path = args.output.with_suffix(".stats.json")
    stats_path.write_text(
        json.dumps(
            {
                "source": str(args.source),
                "output": str(args.output),
                "model": args.model,
                "fallback_model": args.fallback_model,
                "records": written,
                "usage": total_usage,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"wrote {written:,} labels -> {args.output}")
    print(f"usage={total_usage}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
