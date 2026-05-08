"""
FR-PD-06 — Result Is Deterministic for Same Inputs
Real-image test: extract real embeddings, call _pick_diverse twice with identical
inputs, verify both calls return the same result in the same order.

Expected:
  call1 == call2  (element-by-element)

Run:
  python test_fr_pd_06.py
"""

import sys
import numpy as np
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT   = SCRIPT_DIR.parents[2]
DATA_DIR    = SCRIPT_DIR.parent / "data" / "FR_PD_06"
sys.path.insert(0, str(REPO_ROOT))

from _helpers import load_face_app, get_all_jpg_embeddings, report


def run():
    print("=" * 60)
    print("FR-PD-06  Result Is Deterministic for Same Inputs")
    print("=" * 60)

    if not DATA_DIR.exists():
        print(f"  [SKIP] data/FR_PD_06/ not found — run 00_extract_video_data.py first")
        return

    face_app = load_face_app(det_thresh=0.15)
    if face_app is None:
        print("  [SKIP] InsightFace unavailable")
        return

    embeddings = get_all_jpg_embeddings(face_app, DATA_DIR)
    print(f"  Source     : {DATA_DIR}")
    print(f"  Embeddings : {len(embeddings)}")

    if len(embeddings) < 2:
        print(f"  [SKIP] need ≥ 2 embeddings for determinism check (got {len(embeddings)})")
        return

    from register_staff import _pick_diverse

    avg = np.mean(embeddings, axis=0).astype(np.float32)
    avg /= (np.linalg.norm(avg) or 1.0)

    result1 = _pick_diverse(embeddings, avg, max_extra=4)
    result2 = _pick_diverse(embeddings, avg, max_extra=4)

    print(f"\n  Call 1 result: {len(result1)} embeddings")
    print(f"  Call 2 result: {len(result2)} embeddings")

    passed = True
    passed &= report("FR-PD-06-a", "same number of embeddings returned both calls",
                     len(result1) == len(result2),
                     f"call1={len(result1)}, call2={len(result2)}")
    if result1 and result2 and len(result1) == len(result2):
        all_match = all(np.allclose(r1, r2, atol=1e-6) for r1, r2 in zip(result1, result2))
        passed &= report("FR-PD-06-b", "all returned embeddings are identical element-by-element",
                         all_match,
                         "element-wise allclose comparison")
    print()
    return passed


if __name__ == "__main__":
    run()
