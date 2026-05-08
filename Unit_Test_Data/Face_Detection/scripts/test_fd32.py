#!/usr/bin/env python3
"""
FD-32: _detect_face — Real Non-Face/Background → returns None
Target : _detect_face(rerun_fr.py)
Extract: Background/non-face crop (reuse FD-21 or FD-17). Requires InsightFace.
Test   : Call _detect_face(app, background_crop). Expect None.
Expect : None
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import utils
import numpy as np

TEST_ID = "FD-32"
TITLE   = "_detect_face: non-face/background → None"
IMG     = "background_crop.jpg"


def extract():
    cached = utils.load_img(TEST_ID, IMG)
    if cached is not None and utils.is_clean_nonface_crop(cached):
        print(f"[{TEST_ID}] Cached crop still clean.")
        return True
    if cached is not None:
        print(f"[{TEST_ID}] Cached crop fails validation; regenerating.")
    for src_id, src_file in [("FD-21", "background_crop.jpg"), ("FD-17", "background_crop.jpg")]:
        src = utils.load_img(src_id, src_file)
        if src is not None and utils.is_clean_nonface_crop(src):
            utils.save_img(TEST_ID, IMG, src)
            print(f"[{TEST_ID}] Reused validated {src_id} background crop.")
            return True
    model = utils.load_yolo_face()
    if model is not None:
        print(f"[{TEST_ID}] Scanning for empty frame...")
        frame = utils.find_empty_frame(model, n_vids=3, step=15)
        if frame is not None:
            h, w = frame.shape[:2]
            crop = frame[:h//4, w//4:3*w//4].copy()
            if crop.size > 0 and min(crop.shape[:2]) >= 10 and utils.is_clean_nonface_crop(crop):
                utils.save_img(TEST_ID, IMG, crop)
                print(f"[{TEST_ID}] Saved validated background crop: {crop.shape}")
                return True
    crop = utils.synth_nonface_crop(size=160)
    utils.save_img(TEST_ID, IMG, crop)
    print(f"[{TEST_ID}] Saved synthetic non-face crop.")
    return True


def test():
    from rerun_fr import _detect_face
    crop = utils.load_img(TEST_ID, IMG)
    if crop is None:
        print(f"[{TEST_ID}] SKIP — no extracted data.")
        return
    app = utils.load_insightface(det_thresh=0.15)
    if app is None:
        print(f"[{TEST_ID}] SKIP — InsightFace not available.")
        return
    result = _detect_face(app, crop)
    passed = (result is None)
    utils.report(TEST_ID, TITLE, passed,
                 actual=f"result={result}",
                 expected="None")


if __name__ == '__main__':
    ok = extract()
    if ok:
        test()
