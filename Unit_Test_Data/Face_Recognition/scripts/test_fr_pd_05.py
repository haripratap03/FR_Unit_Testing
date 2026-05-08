"""
FR-PD-05 — max_extra=0 Returns Empty
Real-image test: extract real embeddings then call _pick_diverse with max_extra=0.
Result must be [] regardless of pool size.

Expected:
  _pick_diverse(embeddings, avg, max_extra=0) == []

Run:
  python test_fr_pd_05.py
"""

import sys
import numpy as np
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT   = SCRIPT_DIR.parents[2]
DATA_DIR    = SCRIPT_DIR.parent / "data" / "FR_PD_05"
sys.path.insert(0, str(REPO_ROOT))

from _helpers import load_face_app, get_all_jpg_embeddings, report


def run():
    print("=" * 60)
    print("FR-PD-05  max_extra=0 Returns Empty")
    print("=" * 60)

    if not DATA_DIR.exists():
        print(f"  [SKIP] data/FR_PD_05/ not found — run 00_extract_video_data.py first")
        return

    face_app = load_face_app(det_thresh=0.15)
    if face_app is None:
        print("  [SKIP] InsightFace unavailable")
        return

    embeddings = get_all_jpg_embeddings(face_app, DATA_DIR)
    print(f"  Source     : {DATA_DIR}")
    print(f"  Embeddings : {len(embeddings)}")

    if not embeddings:
        print(f"  [SKIP] no embeddings extracted — InsightFace found no faces in crops")
        return

    from register_staff import _pick_diverse

    avg = np.mean(embeddings, axis=0).astype(np.float32)
    avg /= (np.linalg.norm(avg) or 1.0)

    result = _pick_diverse(embeddings, avg, max_extra=0)
    print(f"\n  pool size  : {len(embeddings)}")
    print(f"  max_extra  : 0")
    print(f"  result     : {result!r}")

    passed = report("FR-PD-05",
                    "_pick_diverse with max_extra=0 returns []",
                    result == [],
                    f"got {result!r}")
    print()
    return passed


if __name__ == "__main__":
    run()
