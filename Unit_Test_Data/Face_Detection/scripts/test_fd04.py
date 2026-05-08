#!/usr/bin/env python3
"""
FD-04: score_face — Crop Exactly 40×40 (Boundary)
Target : score_face(pipeline.py:593)
Extract: Resize any real face crop to exactly 40×40.
         Pipeline gate is (face_w < 40 OR face_h < 40) so 40×40 is exactly
         on the passing side of the boundary.
Test   : Call score_face(crop_40x40, yolo_conf=0.80, keypoints=None).
Expect : score > 0 (gate is strictly < 40, so 40 passes).
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import utils
import cv2

TEST_ID = "FD-04"
TITLE   = "score_face: Boundary 40×40 crop passes (gate is < 40)"
IMG     = "face_40x40.jpg"


def extract():
    if utils.load_img(TEST_ID, IMG) is not None:
        print(f"[{TEST_ID}] Data already extracted.")
        return True
    # Try reusing FD-01 crop and resize
    src = utils.load_img("FD-01", "face_crop.jpg")
    if src is None:
        model = utils.load_yolo_face()
        if model is None:
            return False
        dets = utils.scan_faces(model, max_faces=100, step=20, n_vids=2)
        if not dets:
            return False
        src = utils.best_face(dets)['crop']
    crop_40 = cv2.resize(src, (40, 40), interpolation=cv2.INTER_AREA)
    utils.save_img(TEST_ID, IMG, crop_40)
    print(f"[{TEST_ID}] Saved 40×40 crop (resized from real face).")
    return True


def test():
    from pipeline import score_face
    crop = utils.load_img(TEST_ID, IMG)
    if crop is None:
        print(f"[{TEST_ID}] SKIP — no extracted data.")
        return
    h, w = crop.shape[:2]
    score, _ = score_face(crop, yolo_conf=0.80, keypoints=None)
    passed = (score > 0)
    utils.report(TEST_ID, TITLE, passed,
                 actual=f"crop={w}×{h}, score={score}",
                 expected="score > 0")


if __name__ == '__main__':
    ok = extract()
    if ok:
        test()
