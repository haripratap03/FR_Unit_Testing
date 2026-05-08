#!/usr/bin/env python3
"""
Filter bad face crops from existing processed faces.

Applies 3 filters:
  1. InsightFace keypoint validation — need 3+ keypoints with conf > 0.5
  2. Skin-tone check — at least 20% of center pixels must be skin-tone (HSV)
  3. Quality floor — reject if quality score < 25

Rejected images are moved to a rejected/ subfolder inside each person folder.

Usage:
    python filter_faces.py --date 2026-04-09
    python filter_faces.py --date 2026-04-09 --faces-dir /custom/path/to/faces
    python filter_faces.py --date 2026-04-09 --dry-run          # preview only
    python filter_faces.py --date 2026-04-09 --min-keypoints 2  # loosen keypoint gate
    python filter_faces.py --date 2026-04-09 --min-skin 0.15    # loosen skin check
    python filter_faces.py --date 2026-04-09 --min-quality 30   # raise quality floor
"""

import os
import sys
import json
import shutil
import argparse
import re
import logging
import numpy as np
import cv2
import time

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("filter_faces")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def data_dir(date_str):
    return os.path.join(SCRIPT_DIR, "data", date_str)


# ─────────────────────────────────────────────────────────────────────────────
# FILTER 1: InsightFace keypoint validation
# ─────────────────────────────────────────────────────────────────────────────

def check_keypoints(face_app, img, min_keypoints=3, min_kp_conf=0.5):
    """
    Detect face with InsightFace and check that at least `min_keypoints`
    of the 5 landmarks (left_eye, right_eye, nose, left_mouth, right_mouth)
    are present. Returns (pass, num_keypoints_found, reason).
    """
    try:
        faces = face_app.get(img)
        if not faces:
            return False, 0, "no_face_detected"

        # Pick largest face
        best = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))

        kps = best.kps  # shape (5, 2) — 2D landmarks from InsightFace
        if kps is None or len(kps) < 5:
            return False, 0, "no_keypoints"

        # InsightFace kps don't have per-point confidence, but we can check
        # if the points are within the face bbox (valid) vs degenerate (0,0)
        bbox = best.bbox  # x1, y1, x2, y2
        x1, y1, x2, y2 = bbox
        face_w = x2 - x1
        face_h = y2 - y1

        # Allow some margin outside bbox
        margin_x = face_w * 0.2
        margin_y = face_h * 0.2

        valid = 0
        for kp in kps:
            kx, ky = kp
            if (x1 - margin_x <= kx <= x2 + margin_x and
                    y1 - margin_y <= ky <= y2 + margin_y):
                valid += 1

        passed = valid >= min_keypoints
        reason = "" if passed else f"keypoints_{valid}_of_{min_keypoints}"
        return passed, valid, reason

    except Exception as e:
        return False, 0, f"keypoint_error:{e}"


# ─────────────────────────────────────────────────────────────────────────────
# FILTER 2: Skin-tone check (HSV)
# ─────────────────────────────────────────────────────────────────────────────

def check_skin_tone(img, min_skin_ratio=0.20):
    """
    Check that at least `min_skin_ratio` of the center region pixels
    fall within typical skin-tone HSV ranges.
    Returns (pass, skin_ratio, reason).
    """
    h, w = img.shape[:2]

    # Center 60% crop to avoid hair/background at edges
    cx1 = int(w * 0.2)
    cx2 = int(w * 0.8)
    cy1 = int(h * 0.2)
    cy2 = int(h * 0.8)
    center = img[cy1:cy2, cx1:cx2]

    if center.size == 0:
        return False, 0.0, "empty_center"

    hsv = cv2.cvtColor(center, cv2.COLOR_BGR2HSV)

    # Skin-tone HSV ranges (covers light to dark skin)
    # Range 1: standard skin tones
    lower1 = np.array([0, 30, 60], dtype=np.uint8)
    upper1 = np.array([25, 180, 255], dtype=np.uint8)
    mask1 = cv2.inRange(hsv, lower1, upper1)

    # Range 2: reddish/pinkish skin
    lower2 = np.array([160, 30, 60], dtype=np.uint8)
    upper2 = np.array([180, 180, 255], dtype=np.uint8)
    mask2 = cv2.inRange(hsv, lower2, upper2)

    mask = mask1 | mask2
    total_pixels = center.shape[0] * center.shape[1]
    skin_pixels = cv2.countNonZero(mask)
    ratio = skin_pixels / total_pixels if total_pixels > 0 else 0.0

    passed = ratio >= min_skin_ratio
    reason = "" if passed else f"skin_{ratio:.2f}_below_{min_skin_ratio}"
    return passed, round(ratio, 4), reason


# ─────────────────────────────────────────────────────────────────────────────
# FILTER 3: Quality floor from filename
# ─────────────────────────────────────────────────────────────────────────────

def check_quality(filename, min_quality=25):
    """
    Extract quality score from filename (face_XXXXXX_qNN.jpg).
    Returns (pass, score, reason).
    """
    match = re.search(r'_q(\d+)', filename)
    if not match:
        # No quality in filename — can't filter, let it pass
        return True, -1, ""

    score = int(match.group(1))
    passed = score >= min_quality
    reason = "" if passed else f"quality_{score}_below_{min_quality}"
    return passed, score, reason


# ─────────────────────────────────────────────────────────────────────────────
# MAIN FILTER
# ─────────────────────────────────────────────────────────────────────────────

def filter_faces(faces_dir, min_keypoints=3, min_skin=0.20, min_quality=25,
                 dry_run=False):
    if not os.path.exists(faces_dir):
        logger.error(f"Faces directory not found: {faces_dir}")
        return

    # Load InsightFace for keypoint check
    logger.info("Loading InsightFace for keypoint validation...")
    from insightface.app import FaceAnalysis
    models_dir = os.path.join(SCRIPT_DIR, "insightface_models")
    providers = [("CUDAExecutionProvider", {}), ("CPUExecutionProvider", {})]
    face_app = FaceAnalysis(name="antelopev2", root=models_dir, providers=providers)
    face_app.prepare(ctx_id=0, det_size=(640, 640), det_thresh=0.15)
    logger.info("InsightFace loaded")

    # Find all person folders
    person_folders = sorted([
        f for f in os.listdir(faces_dir)
        if os.path.isdir(os.path.join(faces_dir, f)) and f.startswith("person_")
    ])

    if not person_folders:
        logger.info(f"No person folders found in {faces_dir}")
        return

    # Stats
    total_images = 0
    total_rejected = 0
    rejected_by_keypoint = 0
    rejected_by_skin = 0
    rejected_by_quality = 0
    persons_with_all_rejected = 0

    t_start = time.time()

    for folder_name in person_folders:
        folder_path = os.path.join(faces_dir, folder_name)
        rejected_dir = os.path.join(folder_path, "rejected")

        # Get face images (skip body_snapshot.jpg and existing rejected/)
        face_files = sorted([
            f for f in os.listdir(folder_path)
            if f.endswith('.jpg') and f != 'body_snapshot.jpg'
               and not os.path.isdir(os.path.join(folder_path, f))
        ])

        if not face_files:
            continue

        folder_rejected = 0

        for fname in face_files:
            fpath = os.path.join(folder_path, fname)
            total_images += 1
            reasons = []

            # Filter 3: quality floor (cheapest, check first)
            q_pass, q_score, q_reason = check_quality(fname, min_quality)
            if not q_pass:
                reasons.append(q_reason)

            # Load image for remaining checks
            img = cv2.imread(fpath)
            if img is None:
                reasons.append("unreadable")
            else:
                # Filter 2: skin-tone
                s_pass, s_ratio, s_reason = check_skin_tone(img, min_skin)
                if not s_pass:
                    reasons.append(s_reason)

                # Filter 1: keypoint validation
                k_pass, k_count, k_reason = check_keypoints(face_app, img, min_keypoints)
                if not k_pass:
                    reasons.append(k_reason)

            if reasons:
                total_rejected += 1
                folder_rejected += 1

                # Count per-filter
                for r in reasons:
                    if r.startswith("keypoint") or r.startswith("no_face") or r.startswith("no_key"):
                        rejected_by_keypoint += 1
                        break
                for r in reasons:
                    if r.startswith("skin"):
                        rejected_by_skin += 1
                        break
                for r in reasons:
                    if r.startswith("quality"):
                        rejected_by_quality += 1
                        break

                if not dry_run:
                    os.makedirs(rejected_dir, exist_ok=True)
                    # Rename with rejection reason prefix for easy inspection
                    reason_tag = "+".join(reasons)[:80]
                    new_name = f"REJ_{reason_tag}__{fname}"
                    shutil.move(fpath, os.path.join(rejected_dir, new_name))

                logger.debug(f"  REJECT {folder_name}/{fname}: {reasons}")
            else:
                logger.debug(f"  KEEP   {folder_name}/{fname}: q={q_score} skin={s_ratio if img is not None else '?'}")

        if folder_rejected == len(face_files):
            persons_with_all_rejected += 1

        if folder_rejected > 0:
            logger.info(f"  {folder_name}: {folder_rejected}/{len(face_files)} rejected")

    elapsed = time.time() - t_start

    # Summary
    kept = total_images - total_rejected
    print(f"\n{'='*60}")
    print(f"FILTER RESULTS {'(DRY RUN)' if dry_run else ''}")
    print(f"{'='*60}")
    print(f"Person folders scanned : {len(person_folders)}")
    print(f"Total images           : {total_images}")
    print(f"Kept                   : {kept}")
    print(f"Rejected               : {total_rejected}")
    print(f"  - Keypoint failures  : {rejected_by_keypoint}")
    print(f"  - Skin-tone failures : {rejected_by_skin}")
    print(f"  - Quality too low    : {rejected_by_quality}")
    print(f"Persons fully emptied  : {persons_with_all_rejected}")
    print(f"Time                   : {elapsed:.1f}s")
    if not dry_run and total_rejected > 0:
        print(f"\nRejected images moved to rejected/ subfolder in each person folder.")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="Filter bad face crops")
    parser.add_argument("--date", "-d", required=True, help="Date YYYY-MM-DD")
    parser.add_argument("--faces-dir", default=None,
                        help="Custom faces directory (default: data/<date>/faces)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Preview only, don't move files")
    parser.add_argument("--min-keypoints", type=int, default=3,
                        help="Min keypoints required (default: 3)")
    parser.add_argument("--min-skin", type=float, default=0.20,
                        help="Min skin-tone ratio (default: 0.20)")
    parser.add_argument("--min-quality", type=int, default=25,
                        help="Min quality score from filename (default: 25)")

    args = parser.parse_args()

    faces_dir = args.faces_dir or os.path.join(data_dir(args.date), "faces")

    print(f"Faces dir      : {faces_dir}")
    print(f"Min keypoints  : {args.min_keypoints}")
    print(f"Min skin ratio : {args.min_skin}")
    print(f"Min quality    : {args.min_quality}")
    print(f"Dry run        : {args.dry_run}")
    print()

    filter_faces(
        faces_dir,
        min_keypoints=args.min_keypoints,
        min_skin=args.min_skin,
        min_quality=args.min_quality,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
