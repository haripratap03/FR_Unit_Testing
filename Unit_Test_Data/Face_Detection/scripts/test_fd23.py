#!/usr/bin/env python3
"""
FD-23: _check_skin_tone [pipeline] — Non-face Crop → result[0] == False
Target : _check_skin_tone(pipeline.py:636)
Extract: Background/non-skin crop (reuse FD-17 background or FD-18 blue-center).
Test   : Call pipeline._check_skin_tone(non_face_crop).
Expect : result[0] == False
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import utils
import numpy as np

TEST_ID = "FD-23"
TITLE   = "_check_skin_tone [pipeline]: non-face crop → False"
IMG     = "non_face_crop.jpg"


def extract():
    cached = utils.load_img(TEST_ID, IMG)
    if cached is not None and utils.is_clean_nonface_crop(cached):
        print(f"[{TEST_ID}] Cached crop still clean.")
        return True
    if cached is not None:
        print(f"[{TEST_ID}] Cached crop fails validation; regenerating.")
    src = utils.load_img("FD-17", "background_crop.jpg")
    if src is not None and utils.is_clean_nonface_crop(src):
        utils.save_img(TEST_ID, IMG, src)
        print(f"[{TEST_ID}] Reused validated FD-17 background crop.")
        return True
    crop = utils.synth_nonface_crop(size=160)
    utils.save_img(TEST_ID, IMG, crop)
    print(f"[{TEST_ID}] Saved synthetic pure-blue non-skin crop.")
    return True


def test():
    from pipeline import _check_skin_tone as pipeline_check_skin_tone
    crop = utils.load_img(TEST_ID, IMG)
    if crop is None:
        print(f"[{TEST_ID}] SKIP — no extracted data.")
        return
    result = pipeline_check_skin_tone(crop)
    passed = (result[0] is False)
    utils.report(TEST_ID, TITLE, passed,
                 actual=f"result[0]={result[0]}, ratio={result[1]:.3f}",
                 expected="result[0] == False")


if __name__ == '__main__':
    ok = extract()
    if ok:
        test()
