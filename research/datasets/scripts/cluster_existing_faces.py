"""Face-cluster the existing 208k catalog WITHOUT persisting facial biometrics.

User-suggested approach to thread the ethical needle: extract face embeddings
from the un-redacted images on disk → cluster → save ONLY the integer
person_id per row back to the parquet. The embeddings themselves are NEVER
written to any published artifact. The redacted images stay redacted in the
final catalog; the cluster IDs are "anonymised identity tokens" tied to the
redacted item only by row.

Pipeline (multiprocess workers):
  1. Pick a deterministic sample of N items (default 20,000) from rows where
     redaction_status == 'face_detected_redacted'. Faces were already found
     once during the original pipeline, so we know they have faces.
  2. For each, open the UN-redacted JPEG, run MediaPipe FaceLandmarker, crop
     the face bbox + 20% margin, encode with OpenCLIP ViT-B-32.
  3. Cluster the resulting embeddings via Agglomerative (eps=0.15, average
     linkage) in cosine-distance space — same params as the VN pipeline.
  4. Write a SIDECAR file research/datasets/processed/combined/existing_person_clusters.jsonl
     with rows {garment_id, person_id, cluster_size} — NO face embedding.
  5. Run the ingest script (ingest_existing_person_ids.py — separate) to
     merge cluster IDs into items_upper_combined.parquet.

Runtime (~20k items, 8 workers, Mac M-series CPU): ~25-40 min wall-clock.

For the full catalog at 170k items: ~3-4 hours. Override via --sample 170000.

Ethics statement (also documented in DATASHEET.md):
  - Face embeddings are extracted from a per-item un-redacted JPEG already on
    disk, then immediately discarded after the cluster ID is computed.
  - No embedding, bbox, or descriptive face attribute is persisted.
  - The downstream user→model matching feature can compare a user's selfie
    embedding (held in memory at recommendation time) against a *centroid*
    of items in each cluster — but the per-item embedding is gone.
"""

# intent: anonymised face-clustering of existing 208k items (cluster IDs only persist)
# status: ready
# next: run ingest_existing_person_ids.py after this completes
# confidence: high — pattern verified on the 615 new VN items earlier

from __future__ import annotations

import argparse
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
COMBINED = ROOT / "research/datasets/processed/combined/items_upper_combined.parquet"
FACE_MODEL = ROOT / "research/models/face_landmarker.task"
OUT_JSONL = ROOT / "research/datasets/processed/combined/existing_person_clusters.jsonl"

DBSCAN_EPS = 0.15
DBSCAN_MIN_SAMPLES = 2


def init_worker():
    """Each worker initialises its own MediaPipe + CLIP. Models are not picklable."""
    global _face, _clip, _preprocess, _torch, _mp
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
    import torch
    import open_clip

    face_options = vision.FaceLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=str(FACE_MODEL)),
        running_mode=vision.RunningMode.IMAGE,
        num_faces=2,
        min_face_detection_confidence=0.4,
    )
    _face = vision.FaceLandmarker.create_from_options(face_options)
    _clip, _, _preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="laion2b_s34b_b79k"
    )
    _clip.eval()
    _torch = torch
    _mp = mp


def extract_face_embedding(task: tuple[str, str]) -> dict:
    """Worker: open un-redacted image, detect face, return CLIP face embedding.
    Embedding is returned in memory only — never written to disk by this fn.
    The caller (main process) collects + clusters + discards."""
    garment_id, image_local_path = task
    try:
        path = ROOT / image_local_path
        if not path.exists():
            return {"garment_id": garment_id, "ok": False, "reason": "file_missing"}
        img_pil = Image.open(path).convert("RGB")
        img_np = np.array(img_pil)
        H, W = img_np.shape[:2]
        mp_image = _mp.Image(image_format=_mp.ImageFormat.SRGB, data=img_np)
        face_res = _face.detect(mp_image)
        if not face_res.face_landmarks:
            return {"garment_id": garment_id, "ok": False, "reason": "no_face"}
        fl = face_res.face_landmarks[0]
        xs = [lm.x for lm in fl]
        ys = [lm.y for lm in fl]
        x0 = max(0, int((min(xs) - 0.20 * (max(xs) - min(xs))) * W))
        y0 = max(0, int((min(ys) - 0.20 * (max(ys) - min(ys))) * H))
        x1 = min(W, int((max(xs) + 0.20 * (max(xs) - min(xs))) * W))
        y1 = min(H, int((max(ys) + 0.20 * (max(ys) - min(ys))) * H))
        face_crop = img_pil.crop((x0, y0, x1, y1))
        if face_crop.size[0] < 32 or face_crop.size[1] < 32:
            return {"garment_id": garment_id, "ok": False, "reason": "tiny_face"}

        with _torch.inference_mode():
            x = _preprocess(face_crop).unsqueeze(0)
            emb = _clip.encode_image(x).squeeze(0).numpy()
            emb = emb / (np.linalg.norm(emb) + 1e-8)
        return {
            "garment_id": garment_id,
            "ok": True,
            "embedding": emb.astype(np.float32).tolist(),
        }
    except Exception as e:  # noqa: BLE001
        return {"garment_id": garment_id, "ok": False, "reason": f"error:{type(e).__name__}"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=20000,
                    help="0 = process all face_detected items; otherwise random sample size")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch-log", type=int, default=1000)
    args = ap.parse_args()

    if not FACE_MODEL.exists():
        sys.exit(f"!! face model missing at {FACE_MODEL}")

    print(f"  loading catalog…")
    df = pd.read_parquet(COMBINED, columns=["garment_id", "source", "image_local_path", "redaction_status"])
    print(f"  total rows: {len(df):,}")

    # Filter to rows where (a) source is NOT vn_* (those already have person_ids
    # from the VN pipeline) AND (b) face was detected originally
    mask = (df["redaction_status"] == "face_detected_redacted") & (~df["source"].str.startswith("vn_"))
    df_face = df[mask].copy()
    print(f"  candidates (face_detected_redacted, non-VN): {len(df_face):,}")

    if args.sample > 0 and args.sample < len(df_face):
        df_face = df_face.sample(n=args.sample, random_state=args.seed)
        print(f"  random sample size: {len(df_face):,}")

    tasks = list(zip(df_face["garment_id"].tolist(), df_face["image_local_path"].tolist()))

    print(f"  spawning {args.workers} workers…")
    embeddings: list[np.ndarray] = []
    gids: list[str] = []
    drops: dict[str, int] = {}
    t_start = time.time()

    with Pool(processes=args.workers, initializer=init_worker) as pool:
        for i, r in enumerate(pool.imap_unordered(extract_face_embedding, tasks, chunksize=8), 1):
            if r["ok"]:
                embeddings.append(np.array(r["embedding"], dtype=np.float32))
                gids.append(r["garment_id"])
            else:
                drops[r["reason"]] = drops.get(r["reason"], 0) + 1
            if i % args.batch_log == 0:
                elapsed = time.time() - t_start
                eta = elapsed / i * (len(tasks) - i)
                print(f"   [{i}/{len(tasks)}]  faces extracted={len(embeddings)}  drops={dict(drops)}  ETA {eta:.0f}s")

    print()
    print(f"  ── extraction done in {time.time()-t_start:.0f}s ──")
    print(f"  faces extracted: {len(embeddings):,}")
    print(f"  drops: {drops}")

    if not embeddings:
        sys.exit("!! no embeddings extracted; nothing to cluster")

    # Cluster — same params as the VN pipeline
    print(f"\n  clustering {len(embeddings):,} embeddings with agglomerative + cosine distance eps={DBSCAN_EPS}…")
    from sklearn.cluster import AgglomerativeClustering
    embs = np.stack(embeddings)
    # cosine distance via 1 - sim
    sims = embs @ embs.T
    dists = 1.0 - sims
    np.fill_diagonal(dists, 0.0)
    ag = AgglomerativeClustering(
        n_clusters=None, distance_threshold=DBSCAN_EPS,
        metric="precomputed", linkage="average",
    )
    labels = ag.fit_predict(dists)

    # Stats
    unique = sorted(set(labels.tolist()))
    n_clusters = len(unique)
    sizes = sorted([sum(1 for l in labels if l == c) for c in unique], reverse=True)
    print(f"  clusters: {n_clusters}")
    print(f"  cluster sizes (largest 20): {sizes[:20]}")
    print(f"  singletons (cluster of 1): {sum(1 for s in sizes if s == 1)}")
    print(f"  largest cluster: {sizes[0]} ({sizes[0]/len(embeddings)*100:.1f}% of face-extracted items)")

    # Write JSONL — ONLY garment_id, person_id, cluster_size. Embeddings discarded.
    cluster_size = {int(l): sum(1 for x in labels if x == l) for l in unique}
    OUT_JSONL.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSONL, "w", encoding="utf-8") as f:
        for gid, lbl in zip(gids, labels.tolist()):
            f.write(json.dumps({
                "garment_id": gid,
                "person_id": int(lbl),
                "cluster_size": cluster_size[int(lbl)],
            }) + "\n")

    # Embedding numpy array goes out of scope here — never written to disk
    del embs, embeddings, dists, sims
    print(f"\n  wrote {OUT_JSONL} ({OUT_JSONL.stat().st_size/1024:.0f} KB)")
    print(f"  Face embeddings were extracted from on-disk un-redacted images,")
    print(f"  then DISCARDED after clustering. Only integer person_id per row")
    print(f"  is persisted. Run ingest_existing_person_ids.py to merge into parquet.")


if __name__ == "__main__":
    main()
