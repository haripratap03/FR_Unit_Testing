"""
FR-PD-01 — Selects Most Distant Embeddings From Average
Real-image test: extract real embeddings from a multi-image person folder,
compute the mean embedding, then verify that _pick_diverse selects the
embeddings with the greatest cosine distance from the mean.

Expected:
  result == top-N most distant embeddings (by cosine dist from avg)
  result_dists[:max_extra] == sorted_all_dists[:max_extra]

Run:
  python test_fr_pd_01.py
  (run 00_extract_video_data.py first if data/FR_PD_01/ is missing)
"""

import sys
import numpy as np
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT   = SCRIPT_DIR.parents[2]
DATA_DIR    = SCRIPT_DIR.parent / "data" / "FR_PD_01"
sys.path.insert(0, str(REPO_ROOT))

from _helpers import load_face_app, get_all_jpg_embeddings, report


def run():
    print("=" * 60)
    print("FR-PD-01  Selects Most Distant Embeddings From Average")
    print("=" * 60)

    if not DATA_DIR.exists():
        print(f"  [SKIP] data/FR_PD_01/ not found — run 00_extract_video_data.py first")
        return

    face_app = load_face_app(det_thresh=0.15)
    if face_app is None:
        print("  [SKIP] InsightFace unavailable")
        return

    embeddings = get_all_jpg_embeddings(face_app, DATA_DIR)
    print(f"  Source     : {DATA_DIR}")
    print(f"  Embeddings : {len(embeddings)} extracted from real face crops")

    if len(embeddings) < 4:
        print(f"  [SKIP] need ≥ 4 embeddings for a meaningful diversity test (got {len(embeddings)})")
        return

    from register_staff import _pick_diverse

    avg = np.mean(embeddings, axis=0).astype(np.float32)
    avg_norm = np.linalg.norm(avg)
    if avg_norm > 0:
        avg = avg / avg_norm

    max_extra = 3
    result = _pick_diverse(embeddings, avg, max_extra=max_extra)

    all_dists   = sorted([1.0 - float(np.dot(e, avg)) for e in embeddings],  reverse=True)
    result_dists = sorted([1.0 - float(np.dot(r, avg)) for r in result],      reverse=True)

    print(f"\n  max_extra    : {max_extra}")
    print(f"  all_dists    : {[round(d, 4) for d in all_dists[:6]]}")
    print(f"  result_dists : {[round(d, 4) for d in result_dists]}")

    passed = True
    passed &= report("FR-PD-01-a", "returns exactly max_extra embeddings",
                     len(result) == max_extra,
                     f"got {len(result)}")
    for i in range(max_extra):
        passed &= report(f"FR-PD-01-b{i+1}",
                         f"result[{i}] is the {i+1}. most distant from avg",
                         abs(result_dists[i] - all_dists[i]) < 1e-4,
                         f"result_dist={result_dists[i]:.4f}, all_dist={all_dists[i]:.4f}")
    print()
    return passed


if __name__ == "__main__":
    run()
