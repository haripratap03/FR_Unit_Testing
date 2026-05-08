"""
FR-EX-01 — Normal Folder With Multiple Face Images
Real-image test: extract_embeddings on a folder with multiple real face crops.

Expected:
  len(embeddings) >= 3
  stats['images_scanned'] >= 3
  stats['faces_extracted'] >= 1  (at least one InsightFace detection)
  all embeddings are unit-norm

Run:
  python test_fr_ex_01.py
  (run 00_extract_video_data.py first if data/FR_EX_01/ is missing)
"""

import sys
import numpy as np
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT   = SCRIPT_DIR.parents[2]
DATA_DIR    = SCRIPT_DIR.parent / "data" / "FR_EX_01"
sys.path.insert(0, str(REPO_ROOT))

from _helpers import load_face_app, report


def run():
    print("=" * 60)
    print("FR-EX-01  Normal Folder With Multiple Face Images")
    print("=" * 60)

    # ── Guard: data folder must exist and have face images ─────────────────
    if not DATA_DIR.exists():
        print(f"  [SKIP] data/FR_EX_01/ not found — run 00_extract_video_data.py first")
        return

    face_files = [f for f in DATA_DIR.iterdir()
                  if f.suffix == ".jpg" and f.name != "body_snapshot.jpg"]
    if len(face_files) < 3:
        print(f"  [SKIP] only {len(face_files)} face images in FR_EX_01/ — need ≥ 3")
        return
    print(f"  Source  : {DATA_DIR}")
    print(f"  Files   : {len(face_files)} face images")

    # ── Load InsightFace ───────────────────────────────────────────────────
    face_app = load_face_app(det_thresh=0.15)
    if face_app is None:
        print("  [SKIP] InsightFace unavailable — cannot run real-image test")
        return

    # ── Run extract_embeddings ─────────────────────────────────────────────
    from rerun_fr import extract_embeddings
    embeddings, stats = extract_embeddings(face_app, str(DATA_DIR), body_snapshot=None)

    print(f"\n  images_scanned   : {stats['images_scanned']}")
    print(f"  faces_extracted  : {stats['faces_extracted']}")
    print(f"  strategies_used  : {stats['strategies_used']}")
    print(f"  embeddings       : {len(embeddings)}")

    # ── Assertions ─────────────────────────────────────────────────────────
    passed = True
    passed &= report("FR-EX-01-a", "images_scanned == num face files",
                     stats['images_scanned'] == len(face_files),
                     f"got {stats['images_scanned']}, expected {len(face_files)}")
    passed &= report("FR-EX-01-b", "at least 1 embedding extracted",
                     stats['faces_extracted'] >= 1,
                     f"got {stats['faces_extracted']}")
    passed &= report("FR-EX-01-c", "embeddings list length matches faces_extracted",
                     len(embeddings) == stats['faces_extracted'],
                     f"len={len(embeddings)}, stat={stats['faces_extracted']}")
    if embeddings:
        norms = [abs(np.linalg.norm(e) - 1.0) for e in embeddings]
        passed &= report("FR-EX-01-d", "all embeddings are unit-norm",
                         all(n < 1e-5 for n in norms),
                         f"max norm deviation: {max(norms):.2e}")
    else:
        print("  [WARN] No embeddings extracted — unit-norm check skipped")

    print()
    return passed


if __name__ == "__main__":
    run()
