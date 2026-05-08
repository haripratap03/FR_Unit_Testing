#!/usr/bin/env python3
"""
FD-17: check_skin_tone [filter_faces] — Non-face background crop → passed=False
Target : check_skin_tone(filter_faces.py)
Extract: Find a frame with zero face detections (background/ceiling/floor).
         Crop a region that is clearly non-skin (e.g. floor, wall, sky).
Test   : Call check_skin_tone(background_crop, min_skin_ratio=0.20).
Expect : passed=False, ratio < 0.20
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import utils
import numpy as np

TEST_ID = "FD-17"
TITLE   = "check_skin_tone [filter_faces]: non-face background → passed=False"
IMG     = "background_crop.jpg"


def extract():
    cached = utils.load_img(TEST_ID, IMG)
    if cached is not None and utils.is_clean_nonface_crop(cached):
        print(f"[{TEST_ID}] Cached crop still clean (skin-free, face-free).")
        return True
    if cached is not None:
        print(f"[{TEST_ID}] Cached crop fails skin/face validation; regenerating.")
    # Real-footage attempt: scan for an empty frame and validate the cropped region.
    model = utils.load_yolo_face()
    if model is not None:
        print(f"[{TEST_ID}] Scanning for empty frame (no faces)...")
        frame = utils.find_empty_frame(model, n_vids=3, step=15)
        if frame is not None:
            h, w = frame.shape[:2]
            crop = frame[:h//3, w//4:3*w//4].copy()
            if crop.size > 0 and min(crop.shape[:2]) >= 10 and utils.is_clean_nonface_crop(crop):
                utils.save_img(TEST_ID, IMG, crop)
                print(f"[{TEST_ID}] Saved validated background crop: {crop.shape}, brightness={utils.brightness(crop):.1f}")
                return True
            print(f"[{TEST_ID}] Background crop has skin/face content; falling back to synthetic.")
    # Synthetic guaranteed-clean fallback.
    crop = utils.synth_nonface_crop(size=160)
    utils.save_img(TEST_ID, IMG, crop)
    print(f"[{TEST_ID}] Saved synthetic pure-blue non-skin crop.")
    return True


def test():
    from filter_faces import check_skin_tone
    crop = utils.load_img(TEST_ID, IMG)
    if crop is None:
        print(f"[{TEST_ID}] SKIP — no extracted data.")
        return
    passed_flag, ratio, reason = check_skin_tone(crop, min_skin_ratio=0.20)
    passed = (passed_flag is False and ratio < 0.20)
    utils.report(TEST_ID, TITLE, passed,
                 actual=f"passed={passed_flag}, ratio={ratio:.3f}, reason={reason!r}",
                 expected="passed=False, ratio < 0.20")


if __name__ == '__main__':
    ok = extract()
    if ok:
        test()
