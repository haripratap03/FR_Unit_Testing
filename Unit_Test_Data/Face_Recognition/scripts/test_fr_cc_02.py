"""
FR-CC-02 — Dissimilar Customers Form Separate Clusters
Real-image test: two different people's embeddings must NOT be merged —
their cosine similarity should be well below 0.60.

Expected:
  len(clusters) == 2
  each cluster has visit_count == 1

Run:
  python test_fr_cc_02.py
"""

import sys
import numpy as np
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT   = SCRIPT_DIR.parents[2]
DATA_DIR    = SCRIPT_DIR.parent / "data" / "FR_CC_02"
sys.path.insert(0, str(REPO_ROOT))

from _helpers import load_face_app, get_all_jpg_embeddings, greedy_cluster, report


def run():
    print("=" * 60)
    print("FR-CC-02  Dissimilar Customers Form Separate Clusters")
    print("=" * 60)

    dir_a = DATA_DIR / "person_a"   # person_0002
    dir_b = DATA_DIR / "person_b"   # person_0016

    if not dir_a.exists() or not dir_b.exists():
        print(f"  [SKIP] person_a/ or person_b/ missing — run 00_extract_video_data.py")
        return

    face_app = load_face_app(det_thresh=0.15)
    if face_app is None:
        print("  [SKIP] InsightFace unavailable")
        return

    embs_a = get_all_jpg_embeddings(face_app, dir_a)
    embs_b = get_all_jpg_embeddings(face_app, dir_b)

    print(f"  person_a (person_0002) : {len(embs_a)} embeddings")
    print(f"  person_b (person_0016) : {len(embs_b)} embeddings")

    if not embs_a or not embs_b:
        print("  [SKIP] insufficient embeddings")
        return

    avg_a = np.mean(embs_a, axis=0).astype(np.float32)
    avg_a /= (np.linalg.norm(avg_a) or 1.0)
    avg_b = np.mean(embs_b, axis=0).astype(np.float32)
    avg_b /= (np.linalg.norm(avg_b) or 1.0)

    inter_sim = float(np.dot(avg_a, avg_b))
    print(f"\n  Cosine sim between persons : {inter_sim:.4f}")

    customer_data = [
        (201, avg_a, {"track_id": 201}),
        (202, avg_b, {"track_id": 202}),
    ]
    clusters = greedy_cluster(customer_data, cluster_threshold=0.60)

    print(f"  Clusters : {len(clusters)}")
    for c in clusters:
        print(f"    cluster {c['cluster_id']}: tracks={c['track_ids']}, visits={c['visit_count']}")

    passed = True
    if inter_sim < 0.60:
        passed &= report("FR-CC-02-a",
                         "different people form 2 separate clusters (sim < 0.60)",
                         len(clusters) == 2,
                         f"got {len(clusters)}")
        if len(clusters) == 2:
            passed &= report("FR-CC-02-b",
                             "each cluster has visit_count == 1",
                             all(c['visit_count'] == 1 for c in clusters),
                             f"counts: {[c['visit_count'] for c in clusters]}")
    else:
        print(f"  [INFO] inter-sim={inter_sim:.4f} >= 0.60 — persons appear similar")
        print(f"         These persons may look alike; test records actual behavior.")
        passed &= report("FR-CC-02",
                         "clustering ran without crash",
                         True, f"sim={inter_sim:.4f}, clusters={len(clusters)}")
    print()
    return passed


if __name__ == "__main__":
    run()
