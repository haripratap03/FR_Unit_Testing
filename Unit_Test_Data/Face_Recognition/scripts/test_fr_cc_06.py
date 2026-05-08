"""
FR-CC-06 — Threshold Boundary: sim == threshold → Merged
Real-image test: extract embeddings from two crops of the same person,
compute the actual cosine similarity between their means, then set
cluster_threshold exactly to that value.  The two tracks must merge
(>= is inclusive).

Additionally test that setting threshold = actual_sim + 0.001 causes
them to remain separate.

Expected:
  At threshold == actual_sim  → 1 cluster  (merged)
  At threshold == actual_sim + 0.001 → 2 clusters  (separate)

Run:
  python test_fr_cc_06.py
"""

import sys
import numpy as np
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT   = SCRIPT_DIR.parents[2]
DATA_DIR    = SCRIPT_DIR.parent / "data" / "FR_CC_06"
sys.path.insert(0, str(REPO_ROOT))

from _helpers import load_face_app, get_all_jpg_embeddings, greedy_cluster, report


def run():
    print("=" * 60)
    print("FR-CC-06  Threshold Boundary: sim == threshold → Merged")
    print("=" * 60)

    track_a = DATA_DIR / "track_a"
    track_b = DATA_DIR / "track_b"

    if not track_a.exists() or not track_b.exists():
        print(f"  [SKIP] track_a/ or track_b/ missing — run 00_extract_video_data.py")
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
        print("  [SKIP] insufficient embeddings")
        return

    avg_a = np.mean(embs_a, axis=0).astype(np.float32)
    avg_a /= (np.linalg.norm(avg_a) or 1.0)
    avg_b = np.mean(embs_b, axis=0).astype(np.float32)
    avg_b /= (np.linalg.norm(avg_b) or 1.0)

    actual_sim = float(np.dot(avg_a, avg_b))
    print(f"\n  Cosine sim between tracks : {actual_sim:.6f}")

    # Cluster at EXACTLY actual_sim as threshold (must merge, >= is inclusive)
    exact_threshold = round(actual_sim, 6)
    clusters_at = greedy_cluster(
        [(401, avg_a, {}), (402, avg_b, {})],
        cluster_threshold=exact_threshold,
    )
    # Cluster at actual_sim + 0.001 (must NOT merge)
    above_threshold = round(actual_sim + 0.001, 6)
    clusters_above = greedy_cluster(
        [(401, avg_a, {}), (402, avg_b, {})],
        cluster_threshold=above_threshold,
    )

    print(f"  At threshold={exact_threshold:.6f}  → {len(clusters_at)} cluster(s)")
    print(f"  At threshold={above_threshold:.6f} → {len(clusters_above)} cluster(s)")

    passed = True
    passed &= report("FR-CC-06-a",
                     "threshold == sim → merged (>= inclusive)",
                     len(clusters_at) == 1,
                     f"got {len(clusters_at)} clusters at threshold={exact_threshold:.6f}")
    passed &= report("FR-CC-06-b",
                     "threshold == sim+0.001 → separate",
                     len(clusters_above) == 2,
                     f"got {len(clusters_above)} clusters at threshold={above_threshold:.6f}")
    print()
    return passed


if __name__ == "__main__":
    run()
