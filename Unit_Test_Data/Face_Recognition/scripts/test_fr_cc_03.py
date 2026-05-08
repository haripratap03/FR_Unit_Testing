"""
FR-CC-03 — Single Customer Forms Exactly One Cluster
Real-image test: pass a single person's embeddings into greedy clustering.
Must produce exactly one cluster with visit_count == 1.

Expected:
  len(clusters) == 1
  clusters[0]['visit_count'] == 1

Run:
  python test_fr_cc_03.py
"""

import sys
import numpy as np
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT   = SCRIPT_DIR.parents[2]
DATA_DIR    = SCRIPT_DIR.parent / "data" / "FR_CC_03"
sys.path.insert(0, str(REPO_ROOT))

from _helpers import load_face_app, get_all_jpg_embeddings, greedy_cluster, report


def run():
    print("=" * 60)
    print("FR-CC-03  Single Customer Forms Exactly One Cluster")
    print("=" * 60)

    if not DATA_DIR.exists():
        print(f"  [SKIP] data/FR_CC_03/ not found — run 00_extract_video_data.py first")
        return

    face_app = load_face_app(det_thresh=0.15)
    if face_app is None:
        print("  [SKIP] InsightFace unavailable")
        return

    embeddings = get_all_jpg_embeddings(face_app, DATA_DIR)
    print(f"  Source     : {DATA_DIR}")
    print(f"  Embeddings : {len(embeddings)}")

    if not embeddings:
        print(f"  [SKIP] no embeddings extracted")
        return

    avg = np.mean(embeddings, axis=0).astype(np.float32)
    avg /= (np.linalg.norm(avg) or 1.0)

    customer_data = [(301, avg, {"track_id": 301})]
    clusters = greedy_cluster(customer_data, cluster_threshold=0.60)

    print(f"\n  Clusters : {len(clusters)}")
    if clusters:
        print(f"    cluster 1: tracks={clusters[0]['track_ids']}, visits={clusters[0]['visit_count']}")

    passed = True
    passed &= report("FR-CC-03-a", "exactly 1 cluster formed",
                     len(clusters) == 1, f"got {len(clusters)}")
    if clusters:
        passed &= report("FR-CC-03-b", "visit_count == 1",
                         clusters[0]['visit_count'] == 1,
                         f"got {clusters[0]['visit_count']}")
        passed &= report("FR-CC-03-c", "cluster contains only track 301",
                         clusters[0]['track_ids'] == [301],
                         f"got {clusters[0]['track_ids']}")
    print()
    return passed


if __name__ == "__main__":
    run()
