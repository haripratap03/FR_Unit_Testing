#!/usr/bin/env python3
"""
FD-21: check_keypoints [filter_faces] — Non-face Region → passed=False, count==0, reason=='no_face_detected'
Target : check_keypoints(filter_faces.py)
Extract: Find a background/non-face frame crop (reuse FD-17 background if available).
         Requires InsightFace.
Test   : Call check_keypoints(app, background_crop, min_keypoints=3).
Expect : passed=False, count==0, reason=='no_face_detected'
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import utils
import numpy as np

TEST_ID = "FD-21"
TITLE   = "check_keypoints [filter_faces]: non-face region → no_face_detected"
IMG     = "background_crop.jpg"


def extract():
    cached = utils.load_img(TEST_ID, IMG)
    if cached is not None and utils.is_clean_nonface_crop(cached):
        print(f"[{TEST_ID}] Cached crop still clean (skin-free, face-free).")
        return True
    if cached is not None:
        print(f"[{TEST_ID}] Cached crop fails validation; regenerating.")
    # Try FD-17 — only adopt it if it is itself clean.
    src = utils.load_img("FD-17", "background_crop.jpg")
    if src is not None and utils.is_clean_nonface_crop(src):
        utils.save_img(TEST_ID, IMG, src)
        print(f"[{TEST_ID}] Reused validated FD-17 background crop.")
        return True
    model = utils.load_yolo_face()
    if model is not None:
        print(f"[{TEST_ID}] Scanning for background frame...")
        frame = utils.find_empty_frame(model, n_vids=3, step=15)
        if frame is not None:
            h, w = frame.shape[:2]
            crop = frame[:h//4, w//4:3*w//4].copy()
            if crop.size > 0 and min(crop.shape[:2]) >= 10 and utils.is_clean_nonface_crop(crop):
                utils.save_img(TEST_ID, IMG, crop)
                print(f"[{TEST_ID}] Saved validated background crop: {crop.shape}")
                return True
    # Synthetic guaranteed non-face fallback.
    crop = utils.synth_nonface_crop(size=160)
    utils.save_img(TEST_ID, IMG, crop)
    print(f"[{TEST_ID}] Saved synthetic non-face crop.")
    return True


def test():
    from filter_faces import check_keypoints
    crop = utils.load_img(TEST_ID, IMG)
    if crop is None:
        print(f"[{TEST_ID}] SKIP — no extracted data.")
        return
    app = utils.load_insightface(det_thresh=0.15)
    if app is None:
        print(f"[{TEST_ID}] SKIP — InsightFace not available.")
        return
    passed_flag, count, reason = check_keypoints(app, crop, min_keypoints=3)
    passed = (passed_flag is False and count == 0 and reason == 'no_face_detected')
    utils.report(TEST_ID, TITLE, passed,
                 actual=f"passed={passed_flag}, count={count}, reason={reason!r}",
                 expected="(False, 0, 'no_face_detected')")


if __name__ == '__main__':
    ok = extract()
    if ok:
        test()
