"""
FR-SM-05 — Multiple Embeddings: Maximum Score Wins
Real-image test: a person folder with ≥ 3 face crops produces multiple
embeddings during run_fr_processing.  The max similarity across all embeddings
determines classification — not the first or last.

Uses the ACTUAL pipeline.run_fr_processing.

Strategy:
  - Gallery: mean of person_0021 crops (registered as staff)
  - Probe folder: same person_0021 crops (3 files → 3 FAISS queries)
  - The max of the 3 search results determines update_type

Expected:
  update_type == 'staff' or 'customer' depending on max similarity
  similarity_score == max individual similarity (not mean)

Run:
  python test_fr_sm_05.py
"""

import sys
import numpy as np
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT   = SCRIPT_DIR.parents[2]
DATA_DIR    = SCRIPT_DIR.parent / "data" / "FR_SM_05"
sys.path.insert(0, str(REPO_ROOT))

from _helpers import (load_face_app, get_all_jpg_embeddings,
                       build_faiss_index, run_fr_with_real_index, report)


def run():
    print("=" * 60)
    print("FR-SM-05  Multiple Embeddings: Maximum Score Wins")
    print("=" * 60)

    if not DATA_DIR.exists():
        print(f"  [SKIP] data/FR_SM_05/ not found — run 00_extract_video_data.py first")
        return

    face_files = [f for f in DATA_DIR.iterdir()
                  if f.suffix == ".jpg" and f.name != "body_snapshot.jpg"]
    print(f"  Source    : {DATA_DIR}")
    print(f"  Face files: {len(face_files)}")

    if len(face_files) < 2:
        print(f"  [SKIP] need ≥ 2 face crops (got {len(face_files)})")
        return

    face_app = load_face_app(det_thresh=0.15)
    if face_app is None:
        print("  [SKIP] InsightFace unavailable")
        return

    # Extract all embeddings to measure per-file similarities
    embeddings = get_all_jpg_embeddings(face_app, DATA_DIR)
    print(f"  Embeddings extracted: {len(embeddings)}")

    if len(embeddings) < 2:
        print(f"  [SKIP] need ≥ 2 embeddings (got {len(embeddings)})")
        return

    # Gallery: mean of first half
    half = max(1, len(embeddings) // 2)
    avg = np.mean(embeddings[:half], axis=0).astype(np.float32)
    avg /= (np.linalg.norm(avg) or 1.0)
    index    = build_faiss_index([avg])
    registry = [{"name": "Person0021"}]

    # Compute individual sims to verify the pipeline picks the max
    per_sims = []
    for emb in embeddings:
        D, _ = index.search(np.expand_dims(emb, 0).astype(np.float32), 1)
        per_sims.append(round(float(D[0][0]), 4))
    expected_max = max(per_sims)

    person_dict = {
        "track_id": 501,
        "images_folder": str(DATA_DIR),
        "face_images_count": len(face_files),
        "first_seen": "2026-04-29T09:00:00",
        "last_seen":  "2026-04-29T09:05:00",
        "fr_processed": False,
    }
    state = {"date": "2026-04-29", "persons": [person_dict], "fr_results": {}}
    run_fr_with_real_index(state, index, registry)

    person = state["persons"][0]
    actual_sim = person.get("similarity_score", -1.0)
    label = person.get("update_type")

    print(f"\n  Per-embedding sims : {per_sims}")
    print(f"  Expected max sim   : {expected_max}")
    print(f"  Pipeline reported  : {actual_sim}")
    print(f"  update_type        : {label}")

    passed = True
    # Pipeline must report the best (max) similarity, not an average or random one
    passed &= report("FR-SM-05-a",
                     "similarity_score == max of individual similarities",
                     abs(actual_sim - expected_max) < 0.01,
                     f"got {actual_sim}, expected {expected_max}")
    # Classification depends on whether max >= threshold (0.80 default)
    import pipeline as _pl
    threshold = _pl.FR_CFG.get("similarity_threshold", 0.95)
    expected_label = "staff" if expected_max >= threshold else "customer"
    passed &= report("FR-SM-05-b",
                     f"update_type == '{expected_label}' (max_sim vs threshold {threshold})",
                     label == expected_label,
                     f"got '{label}'")
    print()
    return passed


if __name__ == "__main__":
    run()
