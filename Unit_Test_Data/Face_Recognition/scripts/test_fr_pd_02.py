"""
FR-PD-02 — Returns At Most max_extra Embeddings
Real-image test: extract ≥ 5 real embeddings from a person folder,
call _pick_diverse with max_extra=4, verify len(result) == 4.

Expected:
  len(_pick_diverse(embeddings, avg, max_extra=4)) == 4

Run:
  python test_fr_pd_02.py
"""

import sys
import numpy as np
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT   = SCRIPT_DIR.parents[2]
DATA_DIR    = SCRIPT_DIR.parent / "data" / "FR_PD_02"
sys.path.insert(0, str(REPO_ROOT))

from _helpers import load_face_app, get_all_jpg_embeddings, report


def run():
    print("=" * 60)
    print("FR-PD-02  Returns At Most max_extra Embeddings")
    print("=" * 60)

    if not DATA_DIR.exists():
        print(f"  [SKIP] data/FR_PD_02/ not found — run 00_extract_video_data.py first")
        return

    face_app = load_face_app(det_thresh=0.15)
    if face_app is None:
        print("  [SKIP] InsightFace unavailable")
        return

    embeddings = get_all_jpg_embeddings(face_app, DATA_DIR)
    print(f"  Source     : {DATA_DIR}")
    print(f"  Embeddings : {len(embeddings)}")

    if len(embeddings) < 5:
        print(f"  [SKIP] need ≥ 5 embeddings to test cap (got {len(embeddings)})")
        return

    from register_staff import _pick_diverse

    avg = np.mean(embeddings, axis=0).astype(np.float32)
    avg /= (np.linalg.norm(avg) or 1.0)

    max_extra = 4
    result = _pick_diverse(embeddings, avg, max_extra=max_extra)
    print(f"\n  pool size  : {len(embeddings)}")
    print(f"  max_extra  : {max_extra}")
    print(f"  result     : {len(result)} embeddings returned")

    passed = report("FR-PD-02",
                    f"len(result) == max_extra ({max_extra}) even with pool={len(embeddings)}",
                    len(result) == max_extra,
                    f"got {len(result)}")
    print()
    return passed


if __name__ == "__main__":
    run()
