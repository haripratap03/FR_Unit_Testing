"""
FR-SM-02 — Similarity Below Threshold → Customer
Real-image test: register person_0002 in FAISS, run run_fr_processing on
person_0007 (different person).  Best sim must be < 0.80 → update_type='customer'.

Uses the ACTUAL pipeline.run_fr_processing.

Expected:
  state['persons'][0]['update_type'] == 'customer'
  similarity_score < 0.80

Run:
  python test_fr_sm_02.py
"""

import sys
import numpy as np
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT   = SCRIPT_DIR.parents[2]
DATA_DIR    = SCRIPT_DIR.parent / "data" / "FR_SM_02"
sys.path.insert(0, str(REPO_ROOT))

from _helpers import (load_face_app, get_all_jpg_embeddings,
                       build_faiss_index, run_fr_with_real_index, report)


def _make_person(track_id, folder):
    return {
        "track_id": track_id,
        "images_folder": str(folder),
        "face_images_count": len(list(Path(folder).glob("*.jpg"))),
        "first_seen": "2026-04-29T09:00:00",
        "last_seen":  "2026-04-29T09:05:00",
        "fr_processed": False,
    }


def run():
    print("=" * 60)
    print("FR-SM-02  Similarity Below Threshold → Customer")
    print("=" * 60)

    gallery_dir = DATA_DIR / "gallery"   # person_0002
    probe_dir   = DATA_DIR / "probe"     # person_0007 (different person)

    if not gallery_dir.exists() or not probe_dir.exists():
        print(f"  [SKIP] gallery/ or probe/ missing — run 00_extract_video_data.py")
        return

    face_app = load_face_app(det_thresh=0.15)
    if face_app is None:
        print("  [SKIP] InsightFace unavailable")
        return

    gallery_embs = get_all_jpg_embeddings(face_app, gallery_dir)
    print(f"  Gallery (person_0002): {len(gallery_embs)} embeddings")

    if not gallery_embs:
        print("  [SKIP] no gallery embeddings")
        return

    avg = np.mean(gallery_embs, axis=0).astype(np.float32)
    avg /= (np.linalg.norm(avg) or 1.0)
    index    = build_faiss_index([avg])
    registry = [{"name": "Person0002"}]

    state = {"date": "2026-04-29", "persons": [_make_person(2, probe_dir)], "fr_results": {}}
    run_fr_with_real_index(state, index, registry)

    person = state["persons"][0]
    print(f"\n  update_type      : {person.get('update_type')}")
    print(f"  similarity_score : {person.get('similarity_score')}")
    print(f"  recognized_name  : {person.get('recognized_name')}")

    passed = True
    passed &= report("FR-SM-02-a",
                     "different-person probe → update_type == 'customer'",
                     person.get("update_type") == "customer",
                     f"got '{person.get('update_type')}'")
    sim = person.get("similarity_score", 1.0)
    passed &= report("FR-SM-02-b",
                     "similarity_score < 0.80 (different identity)",
                     sim < 0.80,
                     f"got {sim}")
    print()
    return passed


if __name__ == "__main__":
    run()
