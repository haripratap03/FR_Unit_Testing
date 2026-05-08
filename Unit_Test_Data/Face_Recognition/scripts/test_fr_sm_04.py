"""
FR-SM-04 — No Embeddings Extracted → no_face
Real-image test: person_0014 has 0 face image files.
run_fr_processing must route this person to update_type='no_face'.

Uses the ACTUAL pipeline.run_fr_processing.

Expected:
  state['persons'][0]['update_type'] == 'no_face'
  person appears in state['fr_results']['no_face']
  person does NOT appear in any customer cluster

Run:
  python test_fr_sm_04.py
"""

import sys
import numpy as np
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT   = SCRIPT_DIR.parents[2]
DATA_DIR    = SCRIPT_DIR.parent / "data" / "FR_SM_04"
sys.path.insert(0, str(REPO_ROOT))

from _helpers import build_faiss_index, run_fr_with_real_index, report


def run():
    print("=" * 60)
    print("FR-SM-04  No Embeddings Extracted → no_face")
    print("=" * 60)

    if not DATA_DIR.exists():
        print(f"  [SKIP] data/FR_SM_04/ not found — run 00_extract_video_data.py first")
        return

    face_files = [f for f in DATA_DIR.iterdir()
                  if f.suffix == ".jpg" and f.name != "body_snapshot.jpg"]
    print(f"  Source    : {DATA_DIR}")
    print(f"  Face files: {len(face_files)}  (expect 0 for person_0014)")

    # Build a non-empty FAISS index so we know the empty-embeddings path fires,
    # not the empty-index path.
    dummy = np.zeros(512, dtype=np.float32)
    dummy[0] = 1.0
    index    = build_faiss_index([dummy])
    registry = [{"name": "SomeStaff"}]

    person_dict = {
        "track_id": 401,
        "images_folder": str(DATA_DIR),
        "face_images_count": 0,
        "first_seen": "2026-04-29T09:00:00",
        "last_seen":  "2026-04-29T09:01:00",
        "fr_processed": False,
    }
    state = {"date": "2026-04-29", "persons": [person_dict], "fr_results": {}}
    run_fr_with_real_index(state, index, registry)

    person = state["persons"][0]
    fr = state.get("fr_results", {})
    no_face_ids    = [x.get("track_id") for x in fr.get("no_face", [])]
    cluster_ids    = [tid for c in fr.get("customer_clusters", []) for tid in c["track_ids"]]

    print(f"\n  update_type  : {person.get('update_type')}")
    print(f"  fr_processed : {person.get('fr_processed')}")
    print(f"  no_face list : {no_face_ids}")
    print(f"  cluster IDs  : {cluster_ids}")

    passed = True
    passed &= report("FR-SM-04-a", "update_type == 'no_face'",
                     person.get("update_type") == "no_face",
                     f"got '{person.get('update_type')}'")
    passed &= report("FR-SM-04-b", "track_id 401 in fr_results['no_face']",
                     401 in no_face_ids,
                     f"no_face list: {no_face_ids}")
    passed &= report("FR-SM-04-c", "track_id 401 NOT in any customer cluster",
                     401 not in cluster_ids,
                     f"cluster_ids: {cluster_ids}")
    print()
    return passed


if __name__ == "__main__":
    run()
