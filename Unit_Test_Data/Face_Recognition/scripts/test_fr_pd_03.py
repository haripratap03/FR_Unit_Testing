"""
FR-PD-03 — Empty Input Returns Empty List
Real-image test: pass an empty embeddings list to _pick_diverse.
This test requires no real images (pure math) but is run here alongside
the other FR-PD tests for completeness.

Expected:
  _pick_diverse([], any_avg, max_extra=4) == []
  No exception raised

Run:
  python test_fr_pd_03.py
"""

import sys
import numpy as np
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT   = SCRIPT_DIR.parents[2]
sys.path.insert(0, str(REPO_ROOT))

from _helpers import report


def run():
    print("=" * 60)
    print("FR-PD-03  Empty Input Returns Empty List")
    print("=" * 60)
    print("  (pure math test — no InsightFace or real images needed)")

    from register_staff import _pick_diverse

    # Use a random unit-norm avg embedding (simulates a real mean embedding)
    rng = np.random.default_rng(seed=42)
    raw = rng.random(512).astype(np.float32)
    avg = raw / np.linalg.norm(raw)

    result = _pick_diverse([], avg, max_extra=4)
    print(f"\n  pool      : 0 embeddings")
    print(f"  result    : {result}")

    passed = True
    passed &= report("FR-PD-03-a", "_pick_diverse([]) returns []",
                     result == [],
                     f"got {result!r}")
    passed &= report("FR-PD-03-b", "no exception raised",
                     True, "reached this point without exception")
    print()
    return passed


if __name__ == "__main__":
    run()
