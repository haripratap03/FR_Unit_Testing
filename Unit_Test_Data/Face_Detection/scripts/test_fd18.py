#!/usr/bin/env python3
"""
FD-18: check_skin_tone [filter_faces] — Center 60% Painted Non-Skin → passed=False
Target : check_skin_tone(filter_faces.py)
Extract: Take a real face crop. Create a MODIFIED copy: paint the CENTER 60%
         region with pure blue BGR(255,0,0). Leave the outer border as-is.
         check_skin_tone only evaluates the center 60% crop, so it will fail.
Test   : Call check_skin_tone(modified_crop, min_skin_ratio=0.20).
Expect : passed=False  (blue center overrides any skin in the border)
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import utils
import numpy as np

TEST_ID = "FD-18"
TITLE   = "check_skin_tone [filter_faces]: blue center 60% → passed=False"
IMG     = "face_blue_center.jpg"


def extract():
    if utils.load_img(TEST_ID, IMG) is not None:
        print(f"[{TEST_ID}] Data already extracted.")
        return True
    # Get a real face crop
    crop = utils.load_img("FD-05", "face_best.jpg")
    if crop is None:
        crop = utils.load_img("FD-01", "face_crop.jpg")
    if crop is None:
        model = utils.load_yolo_face()
        if model is None:
            return False
        dets = utils.scan_faces(model, max_faces=200, step=15, n_vids=3)
        if not dets:
            return False
        crop = utils.best_face(dets)['crop']
    # Ensure minimum size for meaningful center painting
    import cv2
    if min(crop.shape[:2]) < 50:
        crop = cv2.resize(crop, (100, 100), interpolation=cv2.INTER_LINEAR)
    h, w = crop.shape[:2]
    modified = crop.copy()
    # Compute center 60% region (same logic as check_skin_tone in filter_faces.py)
    margin_h = int(h * 0.20)
    margin_w = int(w * 0.20)
    y1, y2 = margin_h, h - margin_h
    x1, x2 = margin_w, w - margin_w
    # Paint center with pure blue (outside all skin HSV ranges)
    modified[y1:y2, x1:x2] = [255, 0, 0]
    utils.save_img(TEST_ID, IMG, modified)
    print(f"[{TEST_ID}] Saved {w}×{h} face with center {x2-x1}×{y2-y1} painted pure blue.")
    return True


def test():
    from filter_faces import check_skin_tone
    crop = utils.load_img(TEST_ID, IMG)
    if crop is None:
        print(f"[{TEST_ID}] SKIP — no extracted data.")
        return
    passed_flag, ratio, reason = check_skin_tone(crop, min_skin_ratio=0.20)
    passed = (passed_flag is False)
    utils.report(TEST_ID, TITLE, passed,
                 actual=f"passed={passed_flag}, ratio={ratio:.3f}, reason={reason!r}",
                 expected="passed=False (blue center defeats skin check)")


if __name__ == '__main__':
    ok = extract()
    if ok:
        test()
