"""Measure whether face redaction degrades downstream attribute labels.

This validates the v3 paper's central claim — "no downstream artifact derived
from non-redacted images" — by checking whether redaction actually preserves
the information the labels depend on.

Protocol:
  1. Sample N items where both the unredacted and the redacted JPEG exist
  2. Re-derive 4 visual attributes (neckline, silhouette, sleeve, pattern) via CLIP
     zero-shot on each version separately
  3. Compare the per-attribute argmax labels: report agreement rate
  4. Compare CLIP embeddings via cosine similarity: report distribution

If agreement is high, the redaction-first release stance is empirically defensible —
labels derived from redacted images are practically identical to labels derived
from non-redacted ones, so users of the redacted bundle lose no signal.

Sample size N=200 keeps runtime to ~5 minutes on a Mac M-series CPU.
"""

# intent: empirically validate the redaction-first release stance
# status: done
# next: report numbers in §4 of the paper
# confidence: high

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import duckdb
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
DB = ROOT / "research/datasets/processed/deepfashion2/style_catalog.duckdb"
OUT = ROOT / "research/eval/redaction_validity.json"
SAMPLE_N = 200
RNG = np.random.default_rng(42)

# CLIP prompt sets matching the labeling pipeline
PROMPTS = {
    "neckline": [
        ("v_neck", "a top with a V-neckline"),
        ("deep_v_neck", "a top with a deep plunging V-neckline"),
        ("scoop", "a top with a scoop neckline"),
        ("crew", "a top with a crew-neck collar"),
        ("boat", "a top with a wide boat neckline"),
        ("high_neck", "a top with a high turtleneck collar"),
        ("sweetheart", "a top with a sweetheart neckline"),
        ("off_shoulder", "a top with off-the-shoulder neckline"),
    ],
    "silhouette": [
        ("bodycon", "a tight body-conscious fitted top"),
        ("fitted", "a fitted top"),
        ("semi_fitted", "a semi-fitted top"),
        ("a_line", "a flared a-line top"),
        ("loose", "a loose relaxed top"),
        ("oversized", "an oversized baggy top"),
    ],
    "sleeve": [
        ("sleeveless", "a sleeveless top"),
        ("short", "a top with short sleeves"),
        ("three_quarter", "a top with three-quarter sleeves"),
        ("long", "a top with long sleeves"),
    ],
    "pattern": [
        ("solid", "a solid color top"),
        ("vertical_stripe", "a top with vertical stripes"),
        ("horizontal_stripe", "a top with horizontal stripes"),
        ("small_print", "a top with a small repeating print"),
        ("large_print", "a top with a large bold print"),
    ],
}


def load_clip():
    try:
        import open_clip
        import torch
    except ImportError:
        sys.exit("!! pip install open-clip-torch torch")
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="laion2b_s34b_b79k"
    )
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    model.eval()
    return model, preprocess, tokenizer, torch


def encode_image(model, preprocess, torch, img: Image.Image) -> np.ndarray:
    with torch.no_grad():
        x = preprocess(img.convert("RGB")).unsqueeze(0)
        e = model.encode_image(x).squeeze(0).numpy()
        return e / (np.linalg.norm(e) + 1e-8)


def encode_text_batch(model, tokenizer, torch, prompts: list[str]) -> np.ndarray:
    with torch.no_grad():
        tok = tokenizer(prompts)
        te = model.encode_text(tok).numpy()
        te = te / (np.linalg.norm(te, axis=1, keepdims=True) + 1e-8)
        return te


def argmax_label(emb: np.ndarray, label_text_embeds: np.ndarray, label_names: list[str]) -> str:
    sims = label_text_embeds @ emb
    return label_names[int(np.argmax(sims))]


def main() -> None:
    if not DB.exists():
        sys.exit(f"!! catalog DB not found at {DB}")

    con = duckdb.connect(str(DB), read_only=True)
    rows = con.execute(f"""
        SELECT garment_id, image_path
        FROM items
        WHERE redaction_status = 'face_detected_redacted'
        ORDER BY hash(garment_id || 'redactionvalidity42')
        LIMIT {SAMPLE_N * 3}
    """).fetchall()

    print(f"  candidates with redacted version available: {len(rows)}")

    print("  loading CLIP…")
    model, preprocess, tokenizer, torch = load_clip()

    # Pre-encode prompt embeddings
    prompt_embeds = {}
    for dim, prompts in PROMPTS.items():
        names = [p[0] for p in prompts]
        texts = [p[1] for p in prompts]
        prompt_embeds[dim] = (names, encode_text_batch(model, tokenizer, torch, texts))

    agreements = {dim: 0 for dim in PROMPTS}
    cosine_diffs = []
    items_done = 0
    for gid, repo_relative in rows:
        if items_done >= SAMPLE_N:
            break
        # repo_relative is repo-rooted; build absolute file paths
        unredacted_path = ROOT / repo_relative
        redacted_path = ROOT / repo_relative.replace("/images/", "/images_redacted/", 1)
        if not unredacted_path.exists() or not redacted_path.exists():
            continue
        try:
            img_un = Image.open(unredacted_path)
            img_re = Image.open(redacted_path)
        except Exception:
            continue
        emb_un = encode_image(model, preprocess, torch, img_un)
        emb_re = encode_image(model, preprocess, torch, img_re)
        cosine_diffs.append(float(np.dot(emb_un, emb_re)))

        for dim, (names, text_embs) in prompt_embeds.items():
            lab_un = argmax_label(emb_un, text_embs, names)
            lab_re = argmax_label(emb_re, text_embs, names)
            if lab_un == lab_re:
                agreements[dim] += 1

        items_done += 1
        if items_done % 50 == 0:
            print(f"  {items_done}/{SAMPLE_N} items processed…")

    print(f"\n  N processed: {items_done}")
    print()
    print("  ── Per-attribute label agreement (redacted vs original) ──")
    for dim, n in agreements.items():
        rate = n / items_done if items_done else 0.0
        print(f"    {dim:<12s} {n:>4}/{items_done}  =  {rate:.3f}")

    cosine_diffs = np.array(cosine_diffs)
    print()
    print("  ── CLIP image embedding cosine(redacted, original) ──")
    print(f"    mean   = {cosine_diffs.mean():.4f}")
    print(f"    median = {np.median(cosine_diffs):.4f}")
    print(f"    p5/p95 = {np.percentile(cosine_diffs, 5):.4f} / {np.percentile(cosine_diffs, 95):.4f}")
    print(f"    min    = {cosine_diffs.min():.4f}")

    results = {
        "n_items": items_done,
        "per_attribute_agreement": {
            dim: {"n_agreed": n, "rate": n / items_done if items_done else 0.0}
            for dim, n in agreements.items()
        },
        "clip_cosine_redacted_vs_original": {
            "mean":   float(cosine_diffs.mean()),
            "median": float(np.median(cosine_diffs)),
            "p5":     float(np.percentile(cosine_diffs, 5)),
            "p95":    float(np.percentile(cosine_diffs, 95)),
            "min":    float(cosine_diffs.min()),
            "max":    float(cosine_diffs.max()),
        },
        "interpretation": (
            "If per-attribute agreement is >= 0.85 across all 4 dims AND median "
            "CLIP cosine >= 0.95, then redaction-first labels are practically "
            "identical to labels from non-redacted images, validating the "
            "release stance empirically."
        ),
    }
    OUT.write_text(json.dumps(results, indent=2, default=float))
    print(f"\n  wrote {OUT}")


if __name__ == "__main__":
    main()
