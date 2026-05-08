"""
Master Data Extraction Script — FR Test Cases
Extracts real face crops from video footage for all 25 extended FR test cases.

Usage:
    python 00_extract_video_data.py [--force]

    --force  : Re-extract even if data folders already exist

Source date: 2026-05-03 (remapped from 2026-04-29 — original IDs lacked the
required face-crop counts on this date; richest folder is person_0037 with 9 crops).

Outputs (written to Unit_Test_Data/Face_Recognition/data/):
    FR_EX_01/   person_0037 face crops   (richest folder — 9 crops)
    FR_EX_02/   empty folder             (no files)
    FR_EX_03/   face crops + body_snapshot from person_0265 (7 crops + body)
    FR_EX_04/   tiny/blurry crops from video + body_snapshot from person_0001
    FR_EX_05/   face crops from person_0177 + .png/.npy/.txt noise files
    FR_PD_01/   up to 10 crops from person_0037 (9 available)
    FR_PD_02/   crops from person_0037 (9 available — cannot exceed cap=10 on this date)
    FR_PD_03/   empty                    (pure math, no images needed)
    FR_PD_04/   3 crops from person_0069 (fewer than max_extra)
    FR_PD_05/   5 crops from person_0088 (max_extra=0 test)
    FR_PD_06/   5 crops from person_0025 (determinism test)
    FR_SM_01/   person_0037 crops        (self-match → staff)
    FR_SM_02/   gallery=person_0037 / probe=person_0088 (different person → customer)
    FR_SM_03/   person_0238 crops        (boundary sim test)
    FR_SM_04/   person_0014/             (max_files=0 → no face images → no_face)
    FR_SM_05/   3 crops from person_0036 (multiple embeddings, max wins)
    FR_SM_06/   person_0022 crops        (empty FAISS → customer)
    FR_SM_07/   person_0264 crops        (out-of-range index)
    FR_CC_01/   person_0037 split 5+4    (same person → merge)
    FR_CC_02/   person_0265 + person_0177 (different persons, low sim → separate)
    FR_CC_03/   person_0009 crops        (one cluster)
    FR_CC_04/   empty                    (no customers → empty clusters)
    FR_CC_06/   person_0238 split        (sim == threshold → merge)
    FR_CC_07/   customer=person_0009 / no_face=person_0014 (no_face excluded)
    FR_CC_08/   five copies of one person_0037 crop (identical embeddings → one cluster)
"""

import sys
import os
import shutil
import argparse
import cv2
import numpy as np
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT   = SCRIPT_DIR.parents[2]
FACES_DIR   = REPO_ROOT / "data" / "2026-05-03" / "faces"
VIDEO_DIR   = REPO_ROOT / "data" / "2026-05-03" / "videos" / "videos"
DATA_DIR    = SCRIPT_DIR.parent / "data"

sys.path.insert(0, str(REPO_ROOT))


# ── Helpers ───────────────────────────────────────────────────────────────────

def copy_face_folder(src: Path, dst: Path, max_files: int = 40,
                     include_body: bool = True, suffix: str = ""):
    """Copy .jpg face crops from src to dst (optionally capped + renamed)."""
    dst.mkdir(parents=True, exist_ok=True)
    files = sorted(
        [f for f in src.iterdir()
         if f.suffix == ".jpg" and f.name != "body_snapshot.jpg"],
        reverse=True,
    )[:max_files]
    for f in files:
        name = f.stem + suffix + f.suffix
        shutil.copy2(str(f), str(dst / name))
    if include_body and (src / "body_snapshot.jpg").exists():
        shutil.copy2(str(src / "body_snapshot.jpg"), str(dst / "body_snapshot.jpg"))
    return len(files)


def extract_tiny_crops_from_video(video_path: Path, dst: Path, n_frames: int = 3):
    """
    Sample n_frames from the video and save very small (32×32) centre crops
    as 'face' images that InsightFace will almost certainly fail to detect a
    clear face in.  These simulate bad quality captures for FR-EX-04.
    """
    dst.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  [WARN] Cannot open {video_path.name} — skipping tiny crop extraction")
        return 0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    saved = 0
    for i in range(n_frames):
        frame_idx = int(total * (i + 1) / (n_frames + 1))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok:
            continue
        h, w = frame.shape[:2]
        # Grab a tiny 32×32 crop from the far corner (very unlikely to contain a usable face)
        crop = frame[:32, :32]
        fname = dst / f"face_{frame_idx:06d}_q12.jpg"   # quality 12 (below reject floor 25)
        cv2.imwrite(str(fname), crop)
        saved += 1
    cap.release()
    return saved


def pick_video(n: int = 0) -> Path:
    """Return the nth video clip, sorted by name. Falls back to other date dirs
    under data/ if VIDEO_DIR (data/2026-05-03/videos/videos/) has no clips —
    that folder is sometimes faces-only (e.g., 2026-05-03), so we look up other
    available date directories."""
    clips = sorted(VIDEO_DIR.glob("*.mp4"))
    if not clips:
        data_root = REPO_ROOT / "data"
        for date_dir in sorted(data_root.iterdir()):
            alt = date_dir / "videos" / "videos"
            if alt.is_dir():
                alt_clips = sorted(alt.glob("*.mp4"))
                if alt_clips:
                    clips = alt_clips
                    break
    if not clips:
        raise RuntimeError(f"No .mp4 clips found under {REPO_ROOT / 'data'}")
    return clips[n % len(clips)]


def ok(tag: str, msg: str = ""):
    line = f"  [OK ] {tag}"
    if msg:
        line += f"  — {msg}"
    print(line)


def skip(tag: str, reason: str):
    print(f"  [SKIP] {tag}  — {reason}")


# ── Per-test data preparation ─────────────────────────────────────────────────

def prep_fr_ex_01(force: bool):
    dst = DATA_DIR / "FR_EX_01"
    if dst.exists() and not force:
        ok("FR_EX_01", "already exists"); return
    shutil.rmtree(dst, ignore_errors=True)
    n = copy_face_folder(FACES_DIR / "person_0037", dst, max_files=10, include_body=False)
    ok("FR_EX_01", f"copied {n} face crops from person_0037")


def prep_fr_ex_02(force: bool):
    dst = DATA_DIR / "FR_EX_02"
    if dst.exists() and not force:
        ok("FR_EX_02", "already exists"); return
    shutil.rmtree(dst, ignore_errors=True)
    dst.mkdir(parents=True)
    ok("FR_EX_02", "empty folder created")


def prep_fr_ex_03(force: bool):
    dst = DATA_DIR / "FR_EX_03"
    if dst.exists() and not force:
        ok("FR_EX_03", "already exists"); return
    shutil.rmtree(dst, ignore_errors=True)
    n = copy_face_folder(FACES_DIR / "person_0265", dst, max_files=6, include_body=True)
    ok("FR_EX_03", f"copied {n} face crops + body_snapshot from person_0265")


def prep_fr_ex_04(force: bool):
    """
    FR-EX-04 needs a folder where face crops fail InsightFace detection
    but body_snapshot succeeds.  Strategy:
      1. Extract tiny (32×32) crops from the first video clip → 'face_*.jpg' files
         (too small/blurry for InsightFace at det_thresh=0.15)
      2. Copy a real body_snapshot from person_0001 as the fallback image.
    """
    dst = DATA_DIR / "FR_EX_04"
    # Skip only if the folder already has both face crops AND a body_snapshot
    # — an empty pre-existing folder (e.g. from an interrupted prior run) must
    # be repopulated, otherwise the test SKIPs forever.
    has_faces = dst.exists() and any(
        f.suffix == ".jpg" and f.name != "body_snapshot.jpg" for f in dst.iterdir()
    ) if dst.exists() else False
    has_body = dst.exists() and (dst / "body_snapshot.jpg").exists()
    if has_faces and has_body and not force:
        ok("FR_EX_04", "already exists"); return
    shutil.rmtree(dst, ignore_errors=True)
    dst.mkdir(parents=True)

    # Step 1: tiny/blurry crops from first video clip
    video = pick_video(0)
    n = extract_tiny_crops_from_video(video, dst, n_frames=3)
    if n == 0:
        skip("FR_EX_04/tiny_crops", f"video {video.name} could not be read")
    else:
        ok("FR_EX_04", f"extracted {n} tiny crops from {video.name}")

    # Step 2: copy body_snapshot from person_0001 (good quality — InsightFace will succeed)
    body_src = FACES_DIR / "person_0001" / "body_snapshot.jpg"
    if body_src.exists():
        shutil.copy2(str(body_src), str(dst / "body_snapshot.jpg"))
        ok("FR_EX_04/body_snapshot", "copied from person_0001")
    else:
        skip("FR_EX_04/body_snapshot", "person_0001/body_snapshot.jpg not found")


def prep_fr_ex_05(force: bool):
    dst = DATA_DIR / "FR_EX_05"
    if dst.exists() and not force:
        ok("FR_EX_05", "already exists"); return
    shutil.rmtree(dst, ignore_errors=True)
    n = copy_face_folder(FACES_DIR / "person_0177", dst, max_files=3, include_body=False)
    # Add non-JPEG noise files
    (dst / "thumbnail.png").write_bytes(b"")
    (dst / "embedding.npy").write_bytes(b"")
    (dst / "metadata.txt").write_text("track_id: 1\n")
    ok("FR_EX_05", f"copied {n} face crops + 3 non-jpg files")


def prep_fr_pd_01(force: bool):
    dst = DATA_DIR / "FR_PD_01"
    if dst.exists() and not force:
        ok("FR_PD_01", "already exists"); return
    shutil.rmtree(dst, ignore_errors=True)
    n = copy_face_folder(FACES_DIR / "person_0037", dst, max_files=10, include_body=False)
    ok("FR_PD_01", f"copied {n} crops from person_0037 (diversity selection test)")


def prep_fr_pd_02(force: bool):
    dst = DATA_DIR / "FR_PD_02"
    if dst.exists() and not force:
        ok("FR_PD_02", "already exists"); return
    shutil.rmtree(dst, ignore_errors=True)
    # NOTE: 2026-05-03's richest folder has only 9 crops, so the cap=10 path
    # cannot be exercised on this date. Test will pass-through without hitting cap.
    n = copy_face_folder(FACES_DIR / "person_0037", dst, max_files=10, include_body=False)
    ok("FR_PD_02", f"copied {n} crops from person_0037 (max_extra cap test — cap not reached)")


def prep_fr_pd_03(force: bool):
    dst = DATA_DIR / "FR_PD_03"
    if dst.exists() and not force:
        ok("FR_PD_03", "already exists"); return
    shutil.rmtree(dst, ignore_errors=True)
    dst.mkdir(parents=True)
    ok("FR_PD_03", "empty folder (pure math test)")


def prep_fr_pd_04(force: bool):
    dst = DATA_DIR / "FR_PD_04"
    if dst.exists() and not force:
        ok("FR_PD_04", "already exists"); return
    shutil.rmtree(dst, ignore_errors=True)
    n = copy_face_folder(FACES_DIR / "person_0069", dst, max_files=3, include_body=False)
    ok("FR_PD_04", f"copied {n} crops from person_0069 (fewer than max_extra)")


def prep_fr_pd_05(force: bool):
    dst = DATA_DIR / "FR_PD_05"
    if dst.exists() and not force:
        ok("FR_PD_05", "already exists"); return
    shutil.rmtree(dst, ignore_errors=True)
    n = copy_face_folder(FACES_DIR / "person_0088", dst, max_files=5, include_body=False)
    ok("FR_PD_05", f"copied {n} crops from person_0088 (max_extra=0 test)")


def prep_fr_pd_06(force: bool):
    dst = DATA_DIR / "FR_PD_06"
    if dst.exists() and not force:
        ok("FR_PD_06", "already exists"); return
    shutil.rmtree(dst, ignore_errors=True)
    n = copy_face_folder(FACES_DIR / "person_0025", dst, max_files=5, include_body=False)
    ok("FR_PD_06", f"copied {n} crops from person_0025 (determinism test)")


def prep_fr_sm_01(force: bool):
    dst = DATA_DIR / "FR_SM_01"
    if dst.exists() and not force:
        ok("FR_SM_01", "already exists"); return
    shutil.rmtree(dst, ignore_errors=True)
    # gallery: first 5 crops; probe: next 3 crops (same person → self-match)
    gallery = dst / "gallery"
    probe   = dst / "probe"
    src = FACES_DIR / "person_0037"
    files = sorted([f for f in src.iterdir()
                    if f.suffix == ".jpg" and f.name != "body_snapshot.jpg"],
                   reverse=True)
    gallery.mkdir(parents=True)
    probe.mkdir(parents=True)
    for f in files[:5]:
        shutil.copy2(str(f), str(gallery / f.name))
    for f in files[5:8]:
        shutil.copy2(str(f), str(probe / f.name))
    ok("FR_SM_01", f"gallery=5, probe=3 crops from person_0037 (self-match → staff)")


def prep_fr_sm_02(force: bool):
    dst = DATA_DIR / "FR_SM_02"
    if dst.exists() and not force:
        ok("FR_SM_02", "already exists"); return
    shutil.rmtree(dst, ignore_errors=True)
    # gallery: person_0037; probe: person_0088 (different person → customer)
    gallery = dst / "gallery"
    probe   = dst / "probe"
    copy_face_folder(FACES_DIR / "person_0037", gallery, max_files=5, include_body=False)
    copy_face_folder(FACES_DIR / "person_0088", probe,   max_files=3, include_body=False)
    ok("FR_SM_02", "gallery=person_0037, probe=person_0088 (different → customer)")


def prep_fr_sm_03(force: bool):
    dst = DATA_DIR / "FR_SM_03"
    if dst.exists() and not force:
        ok("FR_SM_03", "already exists"); return
    shutil.rmtree(dst, ignore_errors=True)
    gallery = dst / "gallery"
    probe   = dst / "probe"
    copy_face_folder(FACES_DIR / "person_0238", gallery, max_files=5, include_body=False)
    copy_face_folder(FACES_DIR / "person_0238", probe,   max_files=3, include_body=False)
    ok("FR_SM_03", "gallery + probe from person_0238 (boundary threshold test)")


def prep_fr_sm_04(force: bool):
    """no_face test — relies on max_files=0 to skip face copy regardless of source count."""
    dst = DATA_DIR / "FR_SM_04"
    if dst.exists() and not force:
        ok("FR_SM_04", "already exists"); return
    shutil.rmtree(dst, ignore_errors=True)
    copy_face_folder(FACES_DIR / "person_0014", dst, max_files=0, include_body=True)
    ok("FR_SM_04", "copied 0 face crops from person_0014 (no_face test)")


def prep_fr_sm_05(force: bool):
    dst = DATA_DIR / "FR_SM_05"
    if dst.exists() and not force:
        ok("FR_SM_05", "already exists"); return
    shutil.rmtree(dst, ignore_errors=True)
    n = copy_face_folder(FACES_DIR / "person_0036", dst, max_files=3, include_body=False)
    ok("FR_SM_05", f"copied {n} crops from person_0036 (multiple embeddings, max wins)")


def prep_fr_sm_06(force: bool):
    dst = DATA_DIR / "FR_SM_06"
    if dst.exists() and not force:
        ok("FR_SM_06", "already exists"); return
    shutil.rmtree(dst, ignore_errors=True)
    n = copy_face_folder(FACES_DIR / "person_0022", dst, max_files=3, include_body=False)
    ok("FR_SM_06", f"copied {n} crops from person_0022 (empty FAISS → customer)")


def prep_fr_sm_07(force: bool):
    dst = DATA_DIR / "FR_SM_07"
    if dst.exists() and not force:
        ok("FR_SM_07", "already exists"); return
    shutil.rmtree(dst, ignore_errors=True)
    n = copy_face_folder(FACES_DIR / "person_0264", dst, max_files=3, include_body=False)
    ok("FR_SM_07", f"copied {n} crops from person_0264 (out-of-range registry index)")


def prep_fr_cc_01(force: bool):
    """Same person split across two sub-folders → high cosine sim → merge."""
    dst = DATA_DIR / "FR_CC_01"
    if dst.exists() and not force:
        ok("FR_CC_01", "already exists"); return
    shutil.rmtree(dst, ignore_errors=True)
    src = FACES_DIR / "person_0037"
    files = sorted([f for f in src.iterdir()
                    if f.suffix == ".jpg" and f.name != "body_snapshot.jpg"],
                   reverse=True)
    (dst / "track_a").mkdir(parents=True)
    (dst / "track_b").mkdir(parents=True)
    for f in files[:5]:
        shutil.copy2(str(f), str(dst / "track_a" / f.name))
    for f in files[5:10]:
        shutil.copy2(str(f), str(dst / "track_b" / f.name))
    ok("FR_CC_01", "person_0037 split 5+4 into track_a / track_b (same person → merge)")


def prep_fr_cc_02(force: bool):
    """Two different people → low sim → separate clusters."""
    dst = DATA_DIR / "FR_CC_02"
    if dst.exists() and not force:
        ok("FR_CC_02", "already exists"); return
    shutil.rmtree(dst, ignore_errors=True)
    copy_face_folder(FACES_DIR / "person_0265", dst / "person_a", max_files=5, include_body=False)
    copy_face_folder(FACES_DIR / "person_0177", dst / "person_b", max_files=5, include_body=False)
    ok("FR_CC_02", "person_0265 vs person_0177 (different people → separate clusters)")


def prep_fr_cc_03(force: bool):
    dst = DATA_DIR / "FR_CC_03"
    if dst.exists() and not force:
        ok("FR_CC_03", "already exists"); return
    shutil.rmtree(dst, ignore_errors=True)
    n = copy_face_folder(FACES_DIR / "person_0009", dst, max_files=3, include_body=False)
    ok("FR_CC_03", f"copied {n} crops from person_0009 (single customer → one cluster)")


def prep_fr_cc_04(force: bool):
    dst = DATA_DIR / "FR_CC_04"
    if dst.exists() and not force:
        ok("FR_CC_04", "already exists"); return
    shutil.rmtree(dst, ignore_errors=True)
    dst.mkdir(parents=True)
    ok("FR_CC_04", "empty folder (no customers → empty cluster list)")


def prep_fr_cc_06(force: bool):
    """Boundary: sim exactly at cluster_threshold → merges."""
    dst = DATA_DIR / "FR_CC_06"
    if dst.exists() and not force:
        ok("FR_CC_06", "already exists"); return
    shutil.rmtree(dst, ignore_errors=True)
    # Use two crops from the same person to get high sim, then we'll lower the threshold
    # dynamically in the test so that sim == threshold
    copy_face_folder(FACES_DIR / "person_0238", dst / "track_a", max_files=3, include_body=False)
    copy_face_folder(FACES_DIR / "person_0238", dst / "track_b", max_files=3, include_body=False)
    ok("FR_CC_06", "person_0238 split (boundary threshold test)")


def prep_fr_cc_07(force: bool):
    """Customer with face + no_face person — no_face must be excluded from clustering."""
    dst = DATA_DIR / "FR_CC_07"
    if dst.exists() and not force:
        ok("FR_CC_07", "already exists"); return
    shutil.rmtree(dst, ignore_errors=True)
    copy_face_folder(FACES_DIR / "person_0009", dst / "customer",
                     max_files=3, include_body=False)
    copy_face_folder(FACES_DIR / "person_0014", dst / "no_face",
                     max_files=0, include_body=True)
    ok("FR_CC_07", "customer=person_0009, no_face=person_0014")


def prep_fr_cc_08(force: bool):
    """5 copies of the same crop → identical embeddings → single cluster."""
    dst = DATA_DIR / "FR_CC_08"
    if dst.exists() and not force:
        ok("FR_CC_08", "already exists"); return
    shutil.rmtree(dst, ignore_errors=True)
    src_files = sorted(
        [f for f in (FACES_DIR / "person_0037").iterdir()
         if f.suffix == ".jpg" and f.name != "body_snapshot.jpg"],
        reverse=True,
    )
    if not src_files:
        skip("FR_CC_08", "person_0037 has no face images")
        return
    best = src_files[0]
    for i in range(5):
        d = dst / f"track_{i+1:02d}"
        d.mkdir(parents=True)
        shutil.copy2(str(best), str(d / best.name))
    ok("FR_CC_08", f"5 copies of {best.name} in track_01..05 (identical embeddings)")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Extract test data for FR extended test cases.")
    parser.add_argument("--force", action="store_true", help="Re-extract even if data exists")
    args = parser.parse_args()
    force = args.force

    print(f"\nData root  : {DATA_DIR}")
    print(f"Faces root : {FACES_DIR}")
    print(f"Video root : {VIDEO_DIR}")
    print(f"Force mode : {force}")
    print()

    preps = [
        prep_fr_ex_01, prep_fr_ex_02, prep_fr_ex_03, prep_fr_ex_04, prep_fr_ex_05,
        prep_fr_pd_01, prep_fr_pd_02, prep_fr_pd_03, prep_fr_pd_04, prep_fr_pd_05,
        prep_fr_pd_06,
        prep_fr_sm_01, prep_fr_sm_02, prep_fr_sm_03, prep_fr_sm_04, prep_fr_sm_05,
        prep_fr_sm_06, prep_fr_sm_07,
        prep_fr_cc_01, prep_fr_cc_02, prep_fr_cc_03, prep_fr_cc_04, prep_fr_cc_06,
        prep_fr_cc_07, prep_fr_cc_08,
    ]

    errors = []
    for prep_fn in preps:
        tag = prep_fn.__name__.replace("prep_", "").upper().replace("_", "-")
        try:
            prep_fn(force)
        except Exception as exc:
            errors.append((tag, exc))
            print(f"  [ERR ] {tag}  — {exc}")

    print(f"\nDone.  {len(preps) - len(errors)}/{len(preps)} test data sets prepared.")
    if errors:
        print("Errors:")
        for tag, exc in errors:
            print(f"  {tag}: {exc}")


if __name__ == "__main__":
    main()
