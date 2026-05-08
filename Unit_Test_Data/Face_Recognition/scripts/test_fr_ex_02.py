"""
FR-EX-02 — Empty Folder Returns Empty List
Real-image test: extract_embeddings on a folder with no face images.

Expected:
  embeddings == []
  stats['images_scanned'] == 0
  stats['faces_extracted'] == 0

Run:
  python test_fr_ex_02.py
  (run 00_extract_video_data.py first to create the empty data/FR_EX_02/ folder)
"""

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT   = SCRIPT_DIR.parents[2]
DATA_DIR    = SCRIPT_DIR.parent / "data" / "FR_EX_02"
sys.path.insert(0, str(REPO_ROOT))

from _helpers import load_face_app, report


def run():
    print("=" * 60)
    print("FR-EX-02  Empty Folder Returns Empty List")
    print("=" * 60)

    # ── Guard: data folder must exist and be empty ─────────────────────────
    if not DATA_DIR.exists():
        print(f"  [SKIP] data/FR_EX_02/ not found — run 00_extract_video_data.py first")
        return
    face_files = [f for f in DATA_DIR.iterdir()
                  if f.suffix == ".jpg" and f.name != "body_snapshot.jpg"]
    if face_files:
        print(f"  [WARN] FR_EX_02/ is not empty ({len(face_files)} face files found)")
        print(f"         Test continues — expects 0 face images to scan")
    print(f"  Source  : {DATA_DIR}")
    print(f"  Files   : {len(face_files)} face images (should be 0)")

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
    print(f"  embeddings       : {len(embeddings)}")

    # ── Assertions ─────────────────────────────────────────────────────────
    passed = True
    passed &= report("FR-EX-02-a", "embeddings list is empty",
                     embeddings == [],
                     f"got {len(embeddings)} embeddings")
    passed &= report("FR-EX-02-b", "images_scanned == 0",
                     stats['images_scanned'] == 0,
                     f"got {stats['images_scanned']}")
    passed &= report("FR-EX-02-c", "faces_extracted == 0",
                     stats['faces_extracted'] == 0,
                     f"got {stats['faces_extracted']}")
    print()
    return passed


if __name__ == "__main__":
    run()
