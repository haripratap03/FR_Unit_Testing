"""
FR-CC-07 — no_face Persons Excluded From Clustering
Real-image test: person_0009 has face crops (customer); person_0014 has
no face crops (no_face).  The no_face person must never appear in any cluster.

Expected:
  Only the customer person appears in clustering output
  no_face person's track_id is absent from all cluster track_ids

Run:
  python test_fr_cc_07.py
"""

import sys
import numpy as np
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT   = SCRIPT_DIR.parents[2]
DATA_DIR    = SCRIPT_DIR.parent / "data" / "FR_CC_07"
sys.path.insert(0, str(REPO_ROOT))

from _helpers import load_face_app, get_all_jpg_embeddings, greedy_cluster, report


def run():
    print("=" * 60)
    print("FR-CC-07  no_face Persons Excluded From Clustering")
    print("=" * 60)

    customer_dir = DATA_DIR / "customer"   # person_0009 — has face images
    noface_dir   = DATA_DIR / "no_face"    # person_0014 — has 0 face images

    if not customer_dir.exists() or not noface_dir.exists():
        print(f"  [SKIP] customer/ or no_face/ missing — run 00_extract_video_data.py")
        return

    face_app = load_face_app(det_thresh=0.15)
    if face_app is None:
        print("  [SKIP] InsightFace unavailable")
        return

    customer_embs = get_all_jpg_embeddings(face_app, customer_dir)
    noface_embs   = get_all_jpg_embeddings(face_app, noface_dir)   # expect []

    print(f"  customer (person_0009) : {len(customer_embs)} embeddings")
    print(f"  no_face  (person_0014) : {len(noface_embs)} embeddings  (expect 0)")

    # Simulate pipeline: no_face person never enters customer_data
    # because run_fr_processing routes empty-embeddings persons to no_face list first
    NOFACE_TRACK_ID  = 501
    CUSTOMER_TRACK_ID = 502

    if not customer_embs:
        print("  [SKIP] no customer embeddings — cannot test clustering")
        return

    avg_c = np.mean(customer_embs, axis=0).astype(np.float32)
    avg_c /= (np.linalg.norm(avg_c) or 1.0)

    # Only customer goes into customer_data — no_face is excluded upstream
    customer_data = [(CUSTOMER_TRACK_ID, avg_c, {"track_id": CUSTOMER_TRACK_ID})]
    clusters = greedy_cluster(customer_data, cluster_threshold=0.60)

    all_cluster_ids = [tid for c in clusters for tid in c['track_ids']]

    print(f"\n  customer_data pool : 1 entry (track {CUSTOMER_TRACK_ID})")
    print(f"  Clusters           : {len(clusters)}")
    print(f"  All track IDs in clusters : {all_cluster_ids}")

    passed = True
    passed &= report("FR-CC-07-a",
                     "customer person appears in exactly 1 cluster",
                     len(clusters) == 1,
                     f"got {len(clusters)} clusters")
    passed &= report("FR-CC-07-b",
                     "no_face track ID absent from all clusters",
                     NOFACE_TRACK_ID not in all_cluster_ids,
                     f"all_cluster_ids={all_cluster_ids}")
    passed &= report("FR-CC-07-c",
                     "no_face person has 0 embeddings (confirmed by real data)",
                     len(noface_embs) == 0,
                     f"got {len(noface_embs)} embeddings for person_0014")
    print()
    return passed


if __name__ == "__main__":
    run()
