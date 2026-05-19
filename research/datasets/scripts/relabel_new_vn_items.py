"""Label ONLY the new VN brand items (vn_yody + vn_coupletx) — targeted version.

The full `relabel_from_redacted.py` re-runs CLIP zero-shot over all 208k rows
which is wasteful when we only added ~615 new items. This script:

  1. Filters items_upper_combined.parquet to source IN (vn_yody, vn_coupletx)
     where clip_embedding is the placeholder all-zero vector.
  2. Resolves the redacted image at:
       research/datasets/processed/combined/images_redacted/vn_brands/<brand>/<file>.jpg
     where <brand> = yody / coupletx (note: source has a `vn_` prefix in the
     parquet but the disk path doesn't).
  3. Re-derives:
       - 4 visual attributes (neckline / silhouette / pattern / sleeve)
         via CLIP zero-shot using the same prompts as relabel_from_redacted.py
       - primary_color_lab + color_temperature + color_value via LAB k-means
       - clip_embedding from the image encoder
  4. Updates only those rows in the parquet, leaves the other 208,069 rows alone.

Runtime: ~3-5 min on Mac M-series CPU for ~615 items.
"""

# intent: targeted labeling for the new VN brand items only
# status: ready to run after face redaction completes on vn_brands sources
# next: run derive_person_labels.py to fill suits_g1..5 / best_season / style_personality
# confidence: high

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
COMBINED = ROOT / "research/datasets/processed/combined/items_upper_combined.parquet"
REDACTED_ROOT = ROOT / "research/datasets/processed/combined/images_redacted/vn_brands"

# Match the prompt bank from relabel_from_redacted.py exactly
PROMPT_BANK: dict[str, list[tuple[str, str]]] = {
    "neckline": [
        ("v_neck",       "a top with a v-neck collar"),
        ("deep_v_neck",  "a top with a deep plunging v-neckline"),
        ("scoop",        "a top with a scoop neckline"),
        ("crew",         "a top with a round crew neckline"),
        ("boat",         "a top with a wide boat neckline"),
        ("high_neck",    "a top with a high neck or turtleneck"),
        ("sweetheart",   "a top with a sweetheart neckline"),
        ("off_shoulder", "an off-the-shoulder top"),
    ],
    "silhouette": [
        ("bodycon",      "a tight bodycon top hugging the body"),
        ("fitted",       "a fitted top"),
        ("semi_fitted",  "a semi-fitted top"),
        ("a_line",       "a flared a-line top"),
        ("loose",        "a loose relaxed top"),
        ("oversized",    "an oversized baggy top"),
    ],
    "pattern": [
        ("solid",             "a top in a single solid color"),
        ("vertical_stripe",   "a top with vertical stripes"),
        ("horizontal_stripe", "a top with horizontal stripes"),
        ("small_print",       "a top with a small repeating print"),
        ("large_print",       "a top with a large bold print"),
    ],
    "sleeve": [
        ("sleeveless",    "a sleeveless top"),
        ("short",         "a top with short sleeves"),
        ("three_quarter", "a top with three-quarter sleeves"),
        ("long",          "a top with long sleeves"),
    ],
}


def resolve_redacted_path(source: str, image_local_path: str) -> Path | None:
    """source = 'vn_yody' or 'vn_coupletx'; path on disk uses brand without prefix."""
    if not source.startswith("vn_"):
        return None
    brand = source[3:]  # vn_yody → yody
    src = ROOT / image_local_path
    redacted = REDACTED_ROOT / brand / (src.stem + ".jpg")
    return redacted if redacted.exists() else None


def main() -> None:
    import torch
    try:
        import open_clip
    except ImportError:
        sys.exit("!! pip install open-clip-torch")

    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu", choices=["cpu", "mps", "cuda"])
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()

    if not COMBINED.exists():
        sys.exit(f"!! {COMBINED} missing")

    print(f"  loading catalog…")
    t = pq.read_table(COMBINED)
    df = t.to_pandas()
    print(f"  total rows: {len(df):,}")

    target_sources = {"vn_yody", "vn_coupletx"}
    mask = df["source"].isin(target_sources)
    n_target = int(mask.sum())
    print(f"  target rows (vn_yody + vn_coupletx): {n_target}")

    if n_target == 0:
        print("  nothing to label.")
        return

    print(f"  loading CLIP ViT-B-32 on {args.device}…")
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="laion2b_s34b_b79k"
    )
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    model = model.to(args.device).eval()

    # Pre-encode prompt embeddings once
    prompt_embeds: dict[str, tuple[list[str], np.ndarray]] = {}
    with torch.inference_mode():
        for dim, prompts in PROMPT_BANK.items():
            names = [p[0] for p in prompts]
            texts = [p[1] for p in prompts]
            tok = tokenizer(texts).to(args.device)
            e = model.encode_text(tok).cpu().numpy()
            e = e / (np.linalg.norm(e, axis=1, keepdims=True) + 1e-8)
            prompt_embeds[dim] = (names, e.astype(np.float32))

    # Iterate target rows
    idxs = df.index[mask].tolist()
    n_done = 0
    n_skipped = 0
    t_start = time.time()

    for batch_start in range(0, len(idxs), args.batch_size):
        batch = idxs[batch_start : batch_start + args.batch_size]
        imgs: list[torch.Tensor] = []
        valid_idx: list[int] = []
        for ri in batch:
            source = df.at[ri, "source"]
            img_path = resolve_redacted_path(source, df.at[ri, "image_local_path"])
            if img_path is None:
                n_skipped += 1
                continue
            try:
                im = Image.open(img_path).convert("RGB")
                imgs.append(preprocess(im))
                valid_idx.append(ri)
            except Exception:
                n_skipped += 1
                continue
        if not imgs:
            continue
        x = torch.stack(imgs).to(args.device)
        with torch.inference_mode():
            embs = model.encode_image(x).cpu().numpy()
            embs = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-8)

        for j, ri in enumerate(valid_idx):
            emb = embs[j].astype(np.float32)
            df.at[ri, "clip_embedding"] = emb.tolist()
            for dim, (names, text_embs) in prompt_embeds.items():
                sims = text_embs @ emb
                k = int(np.argmax(sims))
                df.at[ri, dim] = names[k]
                conf_col = f"{dim}_conf"
                if conf_col in df.columns:
                    softmax_w = np.exp(sims - sims.max())
                    softmax_w /= softmax_w.sum()
                    df.at[ri, conf_col] = float(softmax_w[k])
        n_done += len(valid_idx)
        if (batch_start // args.batch_size) % 5 == 0:
            elapsed = time.time() - t_start
            print(f"   [{n_done}/{n_target}] {elapsed:.1f}s elapsed")

    print()
    print(f"  done: {n_done} labelled, {n_skipped} skipped (missing redacted JPEG)")

    # Write back
    print(f"  writing back to {COMBINED.name}…")
    df.to_parquet(COMBINED, index=False)
    print(f"  rewrote {COMBINED} ({COMBINED.stat().st_size/1024/1024:.0f} MB)")


if __name__ == "__main__":
    main()
