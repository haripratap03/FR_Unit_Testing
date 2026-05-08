#!/usr/bin/env python3
"""
FD-29: _enhance_contrast — No In-Place Mutation
Target : _enhance_contrast(rerun_fr.py)
Extract: Any real face crop (reuse FD-28 dark face or FD-05 best face).
Test   : Make a copy of the crop, call _enhance_contrast(crop), verify crop unchanged.
Expect : crop after call is identical to the copy made before the call
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import utils
import numpy as np

TEST_ID = "FD-29"
TITLE   = "_enhance_contrast: input image not mutated in-place"
IMG     = "face_crop.jpg"


def extract():
    if utils.load_img(TEST_ID, IMG) is not None:
        print(f"[{TEST_ID}] Data already extracted.")
        return True
    for src_id, src_file in [("FD-28", "face_dark.jpg"), ("FD-05", "face_best.jpg"), ("FD-01", "face_crop.jpg")]:
        src = utils.load_img(src_id, src_file)
        if src is not None:
            utils.save_img(TEST_ID, IMG, src)
            print(f"[{TEST_ID}] Reused {src_id} crop.")
            return True
    model = utils.load_yolo_face()
    if model is None:
        return False
    dets = utils.scan_faces(model, max_faces=100, step=20, n_vids=2)
    if not dets:
        return False
    utils.save_img(TEST_ID, IMG, utils.best_face(dets)['crop'])
    return True


def test():
    from rerun_fr import _enhance_contrast
    crop = utils.load_img(TEST_ID, IMG)
    if crop is None:
        print(f"[{TEST_ID}] SKIP — no extracted data.")
        return
    original = crop.copy()
    _enhance_contrast(crop)
    not_mutated = np.array_equal(crop, original)
    passed = not_mutated
    utils.report(TEST_ID, TITLE, passed,
                 actual=f"input unchanged: {not_mutated}",
                 expected="input image NOT modified after _enhance_contrast call")


if __name__ == '__main__':
    ok = extract()
    if ok:
        test()
