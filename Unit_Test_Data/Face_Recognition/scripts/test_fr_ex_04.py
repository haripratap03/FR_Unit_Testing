"""
FR-EX-04 — All Faces Fail, Body Snapshot Succeeds
Real-image test: face crops are tiny (32×32 video-extracted crops that InsightFace
cannot detect a face in).  The body_snapshot.jpg is a full-body image from a real
person that InsightFace can successfully extract an embedding from as fallback.

Expected:
  stats['faces_extracted'] >= 1  (at least body_snapshot produces one embedding)
  The extracted embedding(s) come from the body_snapshot path, not the face crops
  If InsightFace truly fails on all tiny crops, faces_extracted == 1 (body only)

Run:
  python test_fr_ex_04.py
  (run 00_extract_video_data.py first — it extracts tiny crops from video)
"""

import sys
import os
import numpy as np
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT   = SCRIPT_DIR.parents[2]
DATA_DIR    = SCRIPT_DIR.parent / "data" / "FR_EX_04"
sys.path.insert(0, str(REPO_ROOT))

from _helpers import load_face_app, report


def run():
    print("=" * 60)
    print("FR-EX-04  All Faces Fail, Body Snapshot Succeeds")
    print("=" * 60)

    if not DATA_DIR.exists():
        print(f"  [SKIP] data/FR_EX_04/ not found — run 00_extract_video_data.py first")
        return

    face_files = sorted(
        [f for f in DATA_DIR.iterdir()
         if f.suffix == ".jpg" and f.name != "body_snapshot.jpg"]
    )
    body_path = DATA_DIR / "body_snapshot.jpg"

    if not face_files:
        print(f"  [SKIP] no face crop files in FR_EX_04/")
        return
    if not body_path.exists():
        print(f"  [SKIP] body_snapshot.jpg missing from FR_EX_04/")
        return

    print(f"  Source        : {DATA_DIR}")
    print(f"  Face crops    : {len(face_files)} (tiny / low quality — expected to fail InsightFace)")
    print(f"  body_snapshot : present")

    # Report file sizes to confirm crops are tiny
    for f in face_files:
        import cv2
        img = cv2.imread(str(f))
        h, w = img.shape[:2] if img is not None else (0, 0)
        print(f"    {f.name}  {w}×{h} px")
    body_img = cv2.imread(str(body_path))
    if body_img is not None:
        bh, bw = body_img.shape[:2]
        print(f"    body_snapshot.jpg  {bw}×{bh} px")

    # ── Load InsightFace ───────────────────────────────────────────────────
    face_app = load_face_app(det_thresh=0.15)
    if face_app is None:
        print("  [SKIP] InsightFace unavailable")
        return

    # ── Run extract_embeddings with body_snapshot fallback ─────────────────
    from rerun_fr import extract_embeddings
    embeddings, stats = extract_embeddings(
        face_app, str(DATA_DIR), body_snapshot=str(body_path)
    )

    # Also run without body_snapshot to confirm face crops failed
    embs_no_body, stats_no = extract_embeddings(
        face_app, str(DATA_DIR), body_snapshot=None
    )

    print(f"\n  With body_snapshot:")
    print(f"    images_scanned  : {stats['images_scanned']}")
    print(f"    faces_extracted : {stats['faces_extracted']}")
    print(f"    embeddings      : {len(embeddings)}")
    print(f"\n  Without body_snapshot (face crops only):")
    print(f"    images_scanned  : {stats_no['images_scanned']}")
    print(f"    faces_extracted : {stats_no['faces_extracted']} (tiny crops; expect 0 or few)")

    n_face = len(face_files)
    passed = True

    # Core assertion: with body_snapshot, we get at least one embedding
    passed &= report("FR-EX-04-a",
                     "body_snapshot provides at least 1 embedding when face crops fail",
                     stats['faces_extracted'] >= 1,
                     f"faces_extracted={stats['faces_extracted']}")

    # images_scanned should include all face files + body_snapshot (n_face + 1)
    passed &= report("FR-EX-04-b",
                     "images_scanned == n_face_crops + 1 (body_snapshot appended)",
                     stats['images_scanned'] == n_face + 1,
                     f"got {stats['images_scanned']}, expected {n_face + 1}")

    # The face crops should produce fewer/equal embeddings than body fallback adds
    passed &= report("FR-EX-04-c",
                     "body_snapshot fallback increases or maintains extraction count",
                     stats['faces_extracted'] >= stats_no['faces_extracted'],
                     f"with={stats['faces_extracted']}, without={stats_no['faces_extracted']}")

    if embeddings:
        norms = [abs(np.linalg.norm(e) - 1.0) for e in embeddings]
        passed &= report("FR-EX-04-d", "extracted embeddings are unit-norm",
                         all(n < 1e-5 for n in norms),
                         f"max norm deviation: {max(norms):.2e}")
    print()
    return passed


if __name__ == "__main__":
    run()
