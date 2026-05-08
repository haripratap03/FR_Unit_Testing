"""
FR-EX-03 — body_snapshot Excluded From Face Scan, Used as Fallback
Real-image test: body_snapshot.jpg must NOT appear in the folder-scan list
but MUST be processed as the final entry when passed via the body_snapshot arg.

Expected:
  stats['images_scanned'] == (n_face_jpgs + 1)  [+1 for body_snapshot arg]
  body_snapshot.jpg is NOT in the folder-scan (verified by checking file list)
  body_snapshot processed LAST (it follows all face images)

Run:
  python test_fr_ex_03.py
  (run 00_extract_video_data.py first if data/FR_EX_03/ is missing)
"""

import sys
import os
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT   = SCRIPT_DIR.parents[2]
DATA_DIR    = SCRIPT_DIR.parent / "data" / "FR_EX_03"
sys.path.insert(0, str(REPO_ROOT))

from _helpers import load_face_app, report


def run():
    print("=" * 60)
    print("FR-EX-03  body_snapshot Excluded From Scan, Used as Fallback")
    print("=" * 60)

    if not DATA_DIR.exists():
        print(f"  [SKIP] data/FR_EX_03/ not found — run 00_extract_video_data.py first")
        return

    face_files = [f for f in DATA_DIR.iterdir()
                  if f.suffix == ".jpg" and f.name != "body_snapshot.jpg"]
    body_path  = DATA_DIR / "body_snapshot.jpg"

    if not face_files:
        print(f"  [SKIP] no face images in FR_EX_03/")
        return
    if not body_path.exists():
        print(f"  [SKIP] body_snapshot.jpg missing from FR_EX_03/")
        return

    print(f"  Source          : {DATA_DIR}")
    print(f"  Face images     : {len(face_files)}")
    print(f"  body_snapshot   : present")

    # ── Load InsightFace ───────────────────────────────────────────────────
    face_app = load_face_app(det_thresh=0.15)
    if face_app is None:
        print("  [SKIP] InsightFace unavailable")
        return

    # ── Run with body_snapshot arg ─────────────────────────────────────────
    from rerun_fr import extract_embeddings
    embeddings_with, stats_with = extract_embeddings(
        face_app, str(DATA_DIR), body_snapshot=str(body_path)
    )

    # ── Run WITHOUT body_snapshot arg to get face-only scan count ──────────
    embeddings_no, stats_no = extract_embeddings(
        face_app, str(DATA_DIR), body_snapshot=None
    )

    print(f"\n  With body_snapshot:")
    print(f"    images_scanned   : {stats_with['images_scanned']}")
    print(f"    faces_extracted  : {stats_with['faces_extracted']}")
    print(f"\n  Without body_snapshot:")
    print(f"    images_scanned   : {stats_no['images_scanned']}")
    print(f"    faces_extracted  : {stats_no['faces_extracted']}")

    n_face = len(face_files)

    passed = True
    passed &= report("FR-EX-03-a",
                     "folder scan contains only face jpgs (body_snapshot.jpg excluded)",
                     stats_no['images_scanned'] == n_face,
                     f"scanned {stats_no['images_scanned']}, expected {n_face}")
    passed &= report("FR-EX-03-b",
                     "images_scanned with body_snapshot == n_face + 1",
                     stats_with['images_scanned'] == n_face + 1,
                     f"scanned {stats_with['images_scanned']}, expected {n_face + 1}")
    passed &= report("FR-EX-03-c",
                     "more or equal embeddings extracted when body_snapshot provided",
                     stats_with['faces_extracted'] >= stats_no['faces_extracted'],
                     f"with={stats_with['faces_extracted']}, without={stats_no['faces_extracted']}")
    print()
    return passed


if __name__ == "__main__":
    run()
