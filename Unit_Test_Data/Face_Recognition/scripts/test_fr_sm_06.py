"""
FR-SM-06 — Empty FAISS Index → All Customers
Real-image test: real face crops from person_0022 processed by run_fr_processing
against an empty FAISS index (no staff registered).  All persons must be customer.

Uses the ACTUAL pipeline.run_fr_processing (with an empty FAISS index).

Expected:
  update_type == 'customer'  (no crash, no staff match)

Run:
  python test_fr_sm_06.py
"""

import sys
import faiss
import numpy as np
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT   = SCRIPT_DIR.parents[2]
DATA_DIR    = SCRIPT_DIR.parent / "data" / "FR_SM_06"
sys.path.insert(0, str(REPO_ROOT))

from _helpers import run_fr_with_real_index, report


def run():
    print("=" * 60)
    print("FR-SM-06  Empty FAISS Index → All Customers")
    print("=" * 60)

    if not DATA_DIR.exists():
        print(f"  [SKIP] data/FR_SM_06/ not found — run 00_extract_video_data.py first")
        return

    face_files = [f for f in DATA_DIR.iterdir()
                  if f.suffix == ".jpg" and f.name != "body_snapshot.jpg"]
    print(f"  Source    : {DATA_DIR}")
    print(f"  Face files: {len(face_files)}")

    if not face_files:
        print(f"  [SKIP] no face images in FR_SM_06/")
        return

    # Empty FAISS index — no staff registered
    empty_index = faiss.IndexFlatIP(512)
    registry    = []
    print(f"  FAISS ntotal: {empty_index.ntotal}  (empty)")

    person_dict = {
        "track_id": 601,
        "images_folder": str(DATA_DIR),
        "face_images_count": len(face_files),
        "first_seen": "2026-04-29T09:00:00",
        "last_seen":  "2026-04-29T09:05:00",
        "fr_processed": False,
    }
    state = {"date": "2026-04-29", "persons": [person_dict], "fr_results": {}}
    run_fr_with_real_index(state, empty_index, registry)

    person = state["persons"][0]
    print(f"\n  update_type  : {person.get('update_type')}")
    print(f"  fr_processed : {person.get('fr_processed')}")

    passed = True
    passed &= report("FR-SM-06-a",
                     "empty FAISS → update_type == 'customer'",
                     person.get("update_type") == "customer",
                     f"got '{person.get('update_type')}'")
    passed &= report("FR-SM-06-b",
                     "no crash with empty FAISS index",
                     True, "reached end without exception")
    print()
    return passed


if __name__ == "__main__":
    run()
