"""
FR-SM-07 — Out-of-Range Registry Index → No Crash
Real-image test: FAISS is built with 1 real embedding but the registry JSON
has only 0 entries (or the registry is intentionally mismatched), so the
FAISS index returns index 0 which is out of range → pipeline must not crash
and must fall back to 'customer'.

Verifies the guard at pipeline.py:1225:
    if 0 <= idx < len(staff_registry):
        best_name = staff_registry[idx]['name']

Uses the ACTUAL pipeline.run_fr_processing.

Expected:
  No IndexError / KeyError
  update_type == 'customer'  (best_name stays None → condition fails)

Run:
  python test_fr_sm_07.py
"""

import sys
import numpy as np
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT   = SCRIPT_DIR.parents[2]
DATA_DIR    = SCRIPT_DIR.parent / "data" / "FR_SM_07"
sys.path.insert(0, str(REPO_ROOT))

from _helpers import build_faiss_index, run_fr_with_real_index, report


def run():
    print("=" * 60)
    print("FR-SM-07  Out-of-Range Registry Index → No Crash")
    print("=" * 60)

    if not DATA_DIR.exists():
        print(f"  [SKIP] data/FR_SM_07/ not found — run 00_extract_video_data.py first")
        return

    face_files = [f for f in DATA_DIR.iterdir()
                  if f.suffix == ".jpg" and f.name != "body_snapshot.jpg"]
    print(f"  Source    : {DATA_DIR}")
    print(f"  Face files: {len(face_files)}")

    if not face_files:
        print(f"  [SKIP] no face images")
        return

    # Build FAISS index with 1 real-looking embedding
    # but provide EMPTY registry — index 0 is out of range for an empty list
    dummy = np.zeros(512, dtype=np.float32)
    dummy[0] = 1.0
    index    = build_faiss_index([dummy])   # ntotal == 1
    registry = []                            # length 0 → any FAISS result index is OOB

    print(f"  FAISS ntotal : {index.ntotal}")
    print(f"  Registry len : {len(registry)}  (intentionally empty → index 0 is OOB)")

    person_dict = {
        "track_id": 701,
        "images_folder": str(DATA_DIR),
        "face_images_count": len(face_files),
        "first_seen": "2026-04-29T09:00:00",
        "last_seen":  "2026-04-29T09:05:00",
        "fr_processed": False,
    }
    state = {"date": "2026-04-29", "persons": [person_dict], "fr_results": {}}

    no_crash = True
    try:
        run_fr_with_real_index(state, index, registry)
    except (IndexError, KeyError) as exc:
        no_crash = False
        print(f"  [FAIL] exception raised: {exc}")

    person = state["persons"][0]
    label = person.get("update_type")
    print(f"\n  no_crash     : {no_crash}")
    print(f"  update_type  : {label}")

    passed = True
    passed &= report("FR-SM-07-a",
                     "no IndexError/KeyError when FAISS index is out of range for registry",
                     no_crash, "guard at pipeline.py:1225")
    passed &= report("FR-SM-07-b",
                     "update_type == 'customer' (best_name=None → staff condition fails)",
                     label == "customer",
                     f"got '{label}'")
    print()
    return passed


if __name__ == "__main__":
    run()
