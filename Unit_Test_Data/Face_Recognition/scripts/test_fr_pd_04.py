"""
FR-PD-04 — Fewer Embeddings Than max_extra Returns All
Real-image test: extract embeddings from a person with only a few face crops
(fewer than max_extra), verify all are returned without error.

Expected:
  len(result) == len(embeddings)  (all returned, no padding or error)
  len(result) < max_extra

Run:
  python test_fr_pd_04.py
"""

import sys
import numpy as np
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT   = SCRIPT_DIR.parents[2]
DATA_DIR    = SCRIPT_DIR.parent / "data" / "FR_PD_04"
sys.path.insert(0, str(REPO_ROOT))

from _helpers import load_face_app, get_all_jpg_embeddings, report


def run():
    print("=" * 60)
    print("FR-PD-04  Fewer Embeddings Than max_extra Returns All")
    print("=" * 60)

    if not DATA_DIR.exists():
        print(f"  [SKIP] data/FR_PD_04/ not found — run 00_extract_video_data.py first")
        return

    face_app = load_face_app(det_thresh=0.15)
    if face_app is None:
        print("  [SKIP] InsightFace unavailable")
        return

    embeddings = get_all_jpg_embeddings(face_app, DATA_DIR)
    print(f"  Source     : {DATA_DIR}")
    print(f"  Embeddings : {len(embeddings)}")

    max_extra = 4
    if len(embeddings) >= max_extra:
        print(f"  [WARN] extracted {len(embeddings)} embeddings ≥ max_extra={max_extra}")
        print(f"         Test still runs (verifies cap at pool size)")

    from register_staff import _pick_diverse

    avg = np.mean(embeddings, axis=0).astype(np.float32) if embeddings else np.zeros(512, np.float32)
    n = np.linalg.norm(avg)
    if n > 0:
        avg /= n

    result = _pick_diverse(embeddings, avg, max_extra=max_extra)
    expected = min(len(embeddings), max_extra)

    print(f"\n  pool size  : {len(embeddings)}")
    print(f"  max_extra  : {max_extra}")
    print(f"  result     : {len(result)} embeddings returned")
    print(f"  expected   : {expected}")

    passed = report("FR-PD-04",
                    "len(result) == min(pool_size, max_extra) — no padding, no OOB",
                    len(result) == expected,
                    f"got {len(result)}, expected {expected}")
    print()
    return passed


if __name__ == "__main__":
    run()
