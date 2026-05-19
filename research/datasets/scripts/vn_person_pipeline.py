"""Pre-redaction pipeline for the 615 new VN brand items.

Runs THREE steps in one pass over each raw image (single load, multiple analyses):
  1. Person-visibility check via MediaPipe Pose: drop items without visible
     shoulders + at least one hip (flatlay/product-only shots).
  2. Face detection via MediaPipe FaceLandmarker: bbox where a face was found.
  3. Face CLIP embedding: crop face region, embed with OpenCLIP ViT-B-32.

Then clusters the face embeddings (DBSCAN on cosine distance) to assign
`person_id` per item — groups of items featuring the same model.

Critical: this script must run BEFORE face redaction. Once faces are blurred,
the embeddings degrade and clustering is meaningless.

Output (one row per VN item that passed person-visibility):
  research/datasets/raw/vn_brands/vn_person_clusters.jsonl

Each row:
  {
    "garment_id":      "vn_yody:abc123:0",
    "brand":           "vn_yody",
    "image_local_path":"research/datasets/raw/vn_brands/images/yody/abc.jpg",
    "title":           "Áo Sơ Mi Nữ ...",
    "person_visible":  true,
    "face_detected":   true,
    "face_bbox":       [x, y, w, h]   // pixel coords
    "face_clip_embedding": [512 floats],
    "person_id":       17,            // cluster index after DBSCAN
    "cluster_size":    9              // how many items share this person_id
  }

Runtime: ~5-10 min for 615 items on a Mac M-series CPU.
"""

# intent: extract per-item person_id BEFORE we redact faces away forever
# status: ready
# next: pipe person_id into items_upper_combined.parquet as a new column
# confidence: high

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
PRODUCTS_JSONL = ROOT / "research/datasets/raw/vn_brands/products.jsonl"
COMBINED_PARQUET = ROOT / "research/datasets/processed/combined/items_upper_combined.parquet"
FACE_MODEL = ROOT / "research/models/face_landmarker.task"
POSE_MODEL = ROOT / "services/feature-service/models/pose_landmarker_heavy.task"
OUT_JSONL = ROOT / "research/datasets/raw/vn_brands/vn_person_clusters.jsonl"

# Pose-visibility criteria — same as filter_person_visible.py
LM_LEFT_SHOULDER = 11
LM_RIGHT_SHOULDER = 12
LM_LEFT_HIP = 23
LM_RIGHT_HIP = 24
POSE_VIS_THRESHOLD = 0.5

# DBSCAN params for face clustering
DBSCAN_EPS = 0.30      # cosine distance threshold (~0.7 cosine similarity)
DBSCAN_MIN_SAMPLES = 2  # need at least 2 items to form a cluster


def init_models():
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
    import torch
    try:
        import open_clip
    except ImportError:
        sys.exit("!! pip install open-clip-torch")

    pose_options = vision.PoseLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=str(POSE_MODEL)),
        running_mode=vision.RunningMode.IMAGE,
        num_poses=1,
        min_pose_detection_confidence=0.4,
        min_pose_presence_confidence=0.4,
        min_tracking_confidence=0.4,
    )
    pose = vision.PoseLandmarker.create_from_options(pose_options)

    face_options = vision.FaceLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=str(FACE_MODEL)),
        running_mode=vision.RunningMode.IMAGE,
        num_faces=3,  # up to 3 faces per image — multiple-model shots
        min_face_detection_confidence=0.4,
    )
    face = vision.FaceLandmarker.create_from_options(face_options)

    clip_model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="laion2b_s34b_b79k"
    )
    clip_model.eval()
    return pose, face, clip_model, preprocess, torch, mp


def analyze_image(path: Path, pose, face, clip_model, preprocess, torch, mp) -> dict:
    if not path.exists():
        return {"ok": False, "reason": "file_missing"}
    try:
        img_pil = Image.open(path).convert("RGB")
    except Exception as e:
        return {"ok": False, "reason": f"image_open:{type(e).__name__}"}
    img_np = np.array(img_pil)
    H, W = img_np.shape[:2]
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_np)

    # ── pose visibility ──
    pose_res = pose.detect(mp_image)
    if not pose_res.pose_landmarks:
        return {"ok": False, "reason": "no_pose"}
    landmarks = pose_res.pose_landmarks[0]
    ls = landmarks[LM_LEFT_SHOULDER].visibility
    rs = landmarks[LM_RIGHT_SHOULDER].visibility
    lh = landmarks[LM_LEFT_HIP].visibility
    rh = landmarks[LM_RIGHT_HIP].visibility
    both_shoulders = ls >= POSE_VIS_THRESHOLD and rs >= POSE_VIS_THRESHOLD
    any_hip = lh >= POSE_VIS_THRESHOLD or rh >= POSE_VIS_THRESHOLD
    if not (both_shoulders and any_hip):
        return {"ok": False, "reason": "person_not_visible"}

    # ── face detection ──
    face_res = face.detect(mp_image)
    if not face_res.face_landmarks:
        return {"ok": True, "person_visible": True, "face_detected": False,
                "face_bbox": None, "face_clip_embedding": None,
                "image_size": [W, H]}

    # Take the first detected face; bound its bbox from the 478 face-mesh landmarks
    fl = face_res.face_landmarks[0]
    xs = [lm.x for lm in fl]
    ys = [lm.y for lm in fl]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    # Expand bbox by 20% margin in image-relative coords
    bw = x_max - x_min
    bh = y_max - y_min
    margin = 0.20
    x0 = max(0, int((x_min - margin * bw) * W))
    y0 = max(0, int((y_min - margin * bh) * H))
    x1 = min(W, int((x_max + margin * bw) * W))
    y1 = min(H, int((y_max + margin * bh) * H))
    face_crop = img_pil.crop((x0, y0, x1, y1))
    if face_crop.size[0] < 32 or face_crop.size[1] < 32:
        # Too tiny to embed meaningfully — treat as no face
        return {"ok": True, "person_visible": True, "face_detected": False,
                "face_bbox": None, "face_clip_embedding": None,
                "image_size": [W, H]}

    # ── CLIP embedding of face crop ──
    with torch.inference_mode():
        x = preprocess(face_crop).unsqueeze(0)
        emb = clip_model.encode_image(x).squeeze(0).numpy()
        emb = emb / (np.linalg.norm(emb) + 1e-8)

    return {
        "ok": True,
        "person_visible": True,
        "face_detected": True,
        "face_bbox": [int(x0), int(y0), int(x1 - x0), int(y1 - y0)],
        "face_clip_embedding": emb.astype(np.float32).tolist(),
        "image_size": [W, H],
    }


def cluster_persons(rows_with_face: list[dict]) -> list[int]:
    """DBSCAN on cosine distance over the face CLIP embeddings."""
    from sklearn.cluster import DBSCAN
    if not rows_with_face:
        return []
    embs = np.stack([np.array(r["face_clip_embedding"]) for r in rows_with_face])
    # cosine distance = 1 - cosine similarity. Embeddings already L2-normed.
    sims = embs @ embs.T
    dists = 1.0 - sims
    np.fill_diagonal(dists, 0.0)
    db = DBSCAN(eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES, metric="precomputed")
    labels = db.fit_predict(dists)
    return labels.tolist()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="0 = process all")
    ap.add_argument("--brands", nargs="+", default=["yody", "coupletx"])
    args = ap.parse_args()

    if not POSE_MODEL.exists() or not FACE_MODEL.exists():
        sys.exit("!! pose or face model missing — check research/models/face_landmarker.task and services/feature-service/models/pose_landmarker_heavy.task")

    if not PRODUCTS_JSONL.exists():
        sys.exit(f"!! {PRODUCTS_JSONL} missing — run scrape_vn_brands_v2 first")

    # Load only the new VN brand items from products.jsonl
    items = []
    with open(PRODUCTS_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            try:
                p = json.loads(line)
            except Exception:
                continue
            if p.get("brand") in args.brands:
                items.append(p)
    if args.limit:
        items = items[:args.limit]
    print(f"  loaded {len(items)} VN items from {args.brands}")

    print("  loading MediaPipe pose + face + OpenCLIP...")
    pose, face, clip_model, preprocess, torch, mp = init_models()
    print("  models ready.")

    results = []
    rows_with_face = []
    drops = {"file_missing": 0, "image_open": 0, "no_pose": 0, "person_not_visible": 0, "ok_no_face": 0, "ok_with_face": 0}

    for i, item in enumerate(items):
        img_path = ROOT / item["image_local_path"]
        ana = analyze_image(img_path, pose, face, clip_model, preprocess, torch, mp)
        if not ana["ok"]:
            reason = ana["reason"].split(":")[0]
            drops[reason] = drops.get(reason, 0) + 1
        elif ana["face_detected"]:
            drops["ok_with_face"] += 1
            # Make a stable garment_id matching the parquet schema
            image_id = hashlib.sha1(item["url"].encode()).hexdigest()[:12]
            gid = f"vn_{item['brand']}:{image_id}:0"
            row = {
                "garment_id":           gid,
                "brand":                f"vn_{item['brand']}",
                "image_local_path":     item["image_local_path"],
                "title":                item.get("title", ""),
                "person_visible":       True,
                "face_detected":        True,
                "face_bbox":            ana["face_bbox"],
                "face_clip_embedding":  ana["face_clip_embedding"],
                "image_size":           ana["image_size"],
            }
            rows_with_face.append(row)
            results.append(row)
        else:
            drops["ok_no_face"] += 1
            image_id = hashlib.sha1(item["url"].encode()).hexdigest()[:12]
            gid = f"vn_{item['brand']}:{image_id}:0"
            results.append({
                "garment_id":          gid,
                "brand":               f"vn_{item['brand']}",
                "image_local_path":    item["image_local_path"],
                "title":               item.get("title", ""),
                "person_visible":      True,
                "face_detected":       False,
                "face_bbox":           None,
                "face_clip_embedding": None,
                "image_size":          ana["image_size"],
            })

        if (i + 1) % 50 == 0:
            print(f"   [{i+1}/{len(items)}]  drops so far: {drops}")

    print()
    print("  ── analysis summary ──")
    for k, v in drops.items():
        print(f"    {k:<25s} {v:>4}  ({v/len(items)*100:.1f}%)")

    # ── DBSCAN on face embeddings ──
    if rows_with_face:
        print(f"\n  clustering {len(rows_with_face)} face embeddings (DBSCAN eps={DBSCAN_EPS})...")
        cluster_labels = cluster_persons(rows_with_face)
        # Convert: -1 = noise (singleton). Map to person_id.
        # Each non-noise label gets a unique person_id; noise items get person_id=-1.
        cluster_size: dict = {}
        for lbl in cluster_labels:
            cluster_size[lbl] = cluster_size.get(lbl, 0) + 1
        for row, lbl in zip(rows_with_face, cluster_labels):
            row["person_id"] = int(lbl)
            row["cluster_size"] = cluster_size[lbl]

        # Cluster distribution
        unique_labels = sorted(set(cluster_labels))
        n_clusters = sum(1 for l in unique_labels if l != -1)
        n_noise = sum(1 for l in cluster_labels if l == -1)
        print(f"   clusters found: {n_clusters}")
        print(f"   singleton items: {n_noise}  (no other item matches)")
        if n_clusters > 0:
            sizes = [v for k, v in cluster_size.items() if k != -1]
            print(f"   cluster sizes — min={min(sizes)} median={int(np.median(sizes))} max={max(sizes)}")

    # ── Items without face embedding ──
    for row in results:
        if "person_id" not in row:
            row["person_id"] = -1
            row["cluster_size"] = 1

    # ── Write JSONL ──
    OUT_JSONL.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSONL, "w", encoding="utf-8") as f:
        for row in results:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"\n  wrote {len(results)} rows to {OUT_JSONL}")
    print(f"  ({sum(1 for r in results if r['face_detected'])} with face, "
          f"{sum(1 for r in results if not r['face_detected'])} without)")


if __name__ == "__main__":
    main()
