"""
FR-CC-08 — All Identical Embeddings → Single Cluster
Real-image test: use 5 copies of the same face crop (identical file → identical
embedding).  All 5 tracks must chain together into one cluster via greedy
single-linkage (dot(e, e) = 1.0 >= 0.60 at every comparison step).

Expected:
  len(clusters) == 1
  clusters[0]['visit_count'] == 5
  all 5 track_ids present in the single cluster

Run:
  python test_fr_cc_08.py
"""

import sys
import numpy as np
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT   = SCRIPT_DIR.parents[2]
DATA_DIR    = SCRIPT_DIR.parent / "data" / "FR_CC_08"
sys.path.insert(0, str(REPO_ROOT))

from _helpers import load_face_app, get_all_jpg_embeddings, greedy_cluster, report


def run():
    print("=" * 60)
    print("FR-CC-08  All Identical Embeddings → Single Cluster")
    print("=" * 60)

    if not DATA_DIR.exists():
        print(f"  [SKIP] data/FR_CC_08/ not found — run 00_extract_video_data.py first")
        return

    # 5 sub-folders: track_01..track_05, each containing the same face crop
    track_dirs = sorted(DATA_DIR.iterdir())
    track_dirs = [d for d in track_dirs if d.is_dir() and d.name.startswith("track_")]

    print(f"  Source     : {DATA_DIR}")
    print(f"  Tracks     : {len(track_dirs)}")

    if len(track_dirs) < 2:
        print(f"  [SKIP] need ≥ 2 track dirs (got {len(track_dirs)})")
        return

    face_app = load_face_app(det_thresh=0.15)
    if face_app is None:
        print("  [SKIP] InsightFace unavailable")
        return

    # Extract embedding from each track (all identical → same embedding)
    customer_data = []
    for i, td in enumerate(track_dirs):
        embs = get_all_jpg_embeddings(face_app, td)
        if embs:
            avg = np.mean(embs, axis=0).astype(np.float32)
            avg /= (np.linalg.norm(avg) or 1.0)
            customer_data.append((600 + i, avg, {"track_id": 600 + i}))
        else:
            print(f"    [WARN] {td.name}: no embeddings extracted — skipped")

    if len(customer_data) < 2:
        print(f"  [SKIP] only {len(customer_data)} embeddings extracted")
        return

    print(f"  Valid tracks with embeddings : {len(customer_data)}")

    # Verify embeddings are nearly identical (cosine sim ≈ 1.0)
    ref = customer_data[0][1]
    sims = [float(np.dot(ref, e)) for _, e, _ in customer_data[1:]]
    print(f"  Cosine sims to first track   : {[round(s, 4) for s in sims]}")

    clusters = greedy_cluster(customer_data, cluster_threshold=0.60)
    all_track_ids = {tid for c in clusters for tid in c['track_ids']}
    expected_ids  = {cust_id for cust_id, _, _ in customer_data}

    print(f"\n  Clusters    : {len(clusters)}")
    for c in clusters:
        print(f"    cluster {c['cluster_id']}: visits={c['visit_count']}, tracks={c['track_ids']}")

    passed = True
    passed &= report("FR-CC-08-a",
                     "identical embeddings all merge into 1 cluster",
                     len(clusters) == 1,
                     f"got {len(clusters)} clusters")
    if clusters:
        passed &= report("FR-CC-08-b",
                         "visit_count == number of tracks with valid embeddings",
                         clusters[0]['visit_count'] == len(customer_data),
                         f"got {clusters[0]['visit_count']}, expected {len(customer_data)}")
        passed &= report("FR-CC-08-c",
                         "all track IDs present in the single cluster",
                         all_track_ids == expected_ids,
                         f"got {all_track_ids}, expected {expected_ids}")
    print()
    return passed


if __name__ == "__main__":
    run()
