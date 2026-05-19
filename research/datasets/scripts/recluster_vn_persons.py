"""Re-cluster the face CLIP embeddings in vn_person_clusters.jsonl with a
configurable eps. Cheap iteration — does not re-run MediaPipe + CLIP.

Default eps tightened from 0.30 to 0.15 after observation that brand
photoshoots produce very similar CLIP face embeddings (median pairwise cosine
distance ~0.30 within the same catalog), making eps=0.30 too loose and
merging all 261 items into one mega-cluster.

Usage:
  python research/datasets/scripts/recluster_vn_persons.py [--eps 0.15] [--min-samples 2]
"""

# intent: cheap re-cluster after seeing the eps=0.30 mega-cluster artifact
# status: ready
# next: pipe person_id back into the parquet via ingest_vn_person_ids.py
# confidence: high

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
JSONL = ROOT / "research/datasets/raw/vn_brands/vn_person_clusters.jsonl"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eps", type=float, default=0.15,
                    help="DBSCAN eps in cosine-distance space")
    ap.add_argument("--min-samples", type=int, default=2)
    ap.add_argument("--method", choices=["dbscan", "agglomerative"], default="dbscan")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(JSONL)]
    face_rows = [r for r in rows if r.get("face_detected")]
    print(f"  total rows: {len(rows)}  with face: {len(face_rows)}")

    if not face_rows:
        print("  nothing to cluster")
        return

    embs = np.stack([np.array(r["face_clip_embedding"]) for r in face_rows])
    embs = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-8)
    sims = embs @ embs.T
    dists = 1.0 - sims
    np.fill_diagonal(dists, 0.0)

    if args.method == "dbscan":
        from sklearn.cluster import DBSCAN
        db = DBSCAN(eps=args.eps, min_samples=args.min_samples, metric="precomputed")
        labels = db.fit_predict(dists)
    else:  # agglomerative
        from sklearn.cluster import AgglomerativeClustering
        ag = AgglomerativeClustering(
            n_clusters=None, distance_threshold=args.eps,
            metric="precomputed", linkage="average",
        )
        labels = ag.fit_predict(dists)

    # Summary
    unique = sorted(set(labels))
    n_clusters = sum(1 for l in unique if l != -1)
    n_noise = sum(1 for l in labels if l == -1)
    print(f"\n  method={args.method}  eps={args.eps}")
    print(f"  clusters: {n_clusters}")
    print(f"  noise (singleton) items: {n_noise}")
    if n_clusters > 0:
        sizes = sorted(
            [sum(1 for l in labels if l == c) for c in unique if c != -1],
            reverse=True,
        )
        print(f"  cluster sizes (largest 15): {sizes[:15]}")
        print(f"  total items in clusters: {sum(sizes)}")
        print(f"  largest cluster: {sizes[0]}  ({sizes[0]/len(face_rows)*100:.1f}% of face-detected items)")

    # Write back
    cluster_size = {}
    for lbl in labels:
        cluster_size[int(lbl)] = cluster_size.get(int(lbl), 0) + 1

    for row, lbl in zip(face_rows, labels):
        row["person_id"] = int(lbl)
        row["cluster_size"] = cluster_size[int(lbl)]

    # Items without face stay person_id=-1, cluster_size=1
    for row in rows:
        if not row.get("face_detected"):
            row["person_id"] = -1
            row["cluster_size"] = 1

    with open(JSONL, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"\n  rewrote {JSONL}")


if __name__ == "__main__":
    main()
