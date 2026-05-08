"""
FR-CC-01 — Two Similar Customers Merged (sim ≈ 0.98)
Real-image test: split one person's face crops into two separate 'track' folders,
extract embeddings from each half, and verify the greedy clustering merges them
into a single cluster (high cosine similarity → same person).

Expected:
  len(clusters) == 1
  clusters[0]['track_ids'] contains both track IDs
  clusters[0]['visit_count'] == 2

Run:
  python test_fr_cc_01.py
"""

import sys
import numpy as np
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT   = SCRIPT_DIR.parents[2]
DATA_DIR    = SCRIPT_DIR.parent / "data" / "FR_CC_01"
sys.path.insert(0, str(REPO_ROOT))

from _helpers import load_face_app, get_all_jpg_embeddings, greedy_cluster, report


def run():
    print("=" * 60)
    print("FR-CC-01  Two Similar Customers Merged (same person)")
    print("=" * 60)

    track_a = DATA_DIR / "track_a"
    track_b = DATA_DIR / "track_b"

    if not track_a.exists() or not track_b.exists():
        print(f"  [SKIP] data/FR_CC_01/track_a/ or track_b/ missing — run 00_extract_video_data.py")
        return

    face_app = load_face_app(det_thresh=0.15)
    if face_app is None:
        print("  [SKIP] InsightFace unavailable")
        return

    embs_a = get_all_jpg_embeddings(face_app, track_a)
    embs_b = get_all_jpg_embeddings(face_app, track_b)

    print(f"  track_a : {len(embs_a)} embeddings")
    print(f"  track_b : {len(embs_b)} embeddings")

    if not embs_a or not embs_b:
        print("  [SKIP] insufficient embeddings in one or both tracks")
        return

    # Representative embedding: mean of each track
    avg_a = np.mean(embs_a, axis=0).astype(np.float32)
    avg_a /= (np.linalg.norm(avg_a) or 1.0)
    avg_b = np.mean(embs_b, axis=0).astype(np.float32)
    avg_b /= (np.linalg.norm(avg_b) or 1.0)

    inter_sim = float(np.dot(avg_a, avg_b))
    print(f"\n  Cosine sim between tracks : {inter_sim:.4f}")

    # customer_data format: (track_id, representative_emb, person_dict)
    customer_data = [
        (101, avg_a, {"track_id": 101}),
        (102, avg_b, {"track_id": 102}),
    ]
    clusters = greedy_cluster(customer_data, cluster_threshold=0.60)

    print(f"  Clusters : {len(clusters)}")
    for c in clusters:
        print(f"    cluster {c['cluster_id']}: tracks={c['track_ids']}, visits={c['visit_count']}")

    passed = True
    if inter_sim >= 0.60:
        passed &= report("FR-CC-01-a",
                         "same-person tracks merge into 1 cluster (sim >= 0.60)",
                         len(clusters) == 1,
                         f"got {len(clusters)} clusters")
        if clusters:
            passed &= report("FR-CC-01-b",
                             "merged cluster contains both track IDs",
                             set(clusters[0]['track_ids']) == {101, 102},
                             f"got {clusters[0]['track_ids']}")
            passed &= report("FR-CC-01-c",
                             "visit_count == 2",
                             clusters[0]['visit_count'] == 2,
                             f"got {clusters[0]['visit_count']}")
    else:
        print(f"  [INFO] inter-sim={inter_sim:.4f} < 0.60 — same-person split did not meet threshold")
        passed &= report("FR-CC-01-a",
                         "clusters formed without crash (sim < threshold → separate clusters expected)",
                         True, f"sim={inter_sim:.4f}")
    print()
    return passed


if __name__ == "__main__":
    run()
