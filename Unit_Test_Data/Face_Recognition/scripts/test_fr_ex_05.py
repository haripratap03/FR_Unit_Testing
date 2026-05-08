"""
FR-EX-05 — Non-JPEG Files Ignored in Scan
Real-image test: a folder containing real face crops (.jpg) alongside non-JPEG
files (.png, .npy, .txt) — only the .jpg files should be scanned.

Expected:
  stats['images_scanned'] == number of .jpg files only
  .png / .npy / .txt files are completely ignored

Run:
  python test_fr_ex_05.py
  (run 00_extract_video_data.py first if data/FR_EX_05/ is missing)
"""

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT   = SCRIPT_DIR.parents[2]
DATA_DIR    = SCRIPT_DIR.parent / "data" / "FR_EX_05"
sys.path.insert(0, str(REPO_ROOT))

from _helpers import load_face_app, report


def run():
    print("=" * 60)
    print("FR-EX-05  Non-JPEG Files Ignored in Scan")
    print("=" * 60)

    if not DATA_DIR.exists():
        print(f"  [SKIP] data/FR_EX_05/ not found — run 00_extract_video_data.py first")
        return

    all_files  = list(DATA_DIR.iterdir())
    face_jpgs  = [f for f in all_files
                  if f.suffix == ".jpg" and f.name != "body_snapshot.jpg"]
    non_jpgs   = [f for f in all_files
                  if f.suffix not in (".jpg",) or f.name == "body_snapshot.jpg"]

    print(f"  Source      : {DATA_DIR}")
    print(f"  .jpg files  : {len(face_jpgs)}")
    print(f"  other files : {[f.name for f in non_jpgs]}")

    if not face_jpgs:
        print(f"  [SKIP] no .jpg face files present")
        return

    # ── Load InsightFace ───────────────────────────────────────────────────
    face_app = load_face_app(det_thresh=0.15)
    if face_app is None:
        print("  [SKIP] InsightFace unavailable")
        return

    # ── Run extract_embeddings ─────────────────────────────────────────────
    from rerun_fr import extract_embeddings
    embeddings, stats = extract_embeddings(face_app, str(DATA_DIR), body_snapshot=None)

    print(f"\n  images_scanned  : {stats['images_scanned']}")
    print(f"  faces_extracted : {stats['faces_extracted']}")

    passed = True
    passed &= report("FR-EX-05-a",
                     "images_scanned equals only .jpg face files (not .png/.npy/.txt)",
                     stats['images_scanned'] == len(face_jpgs),
                     f"got {stats['images_scanned']}, expected {len(face_jpgs)}")
    passed &= report("FR-EX-05-b",
                     "non-JPEG files do not cause a crash",
                     True,  # reaching here means no crash
                     "no exception raised")
    print()
    return passed


if __name__ == "__main__":
    run()
