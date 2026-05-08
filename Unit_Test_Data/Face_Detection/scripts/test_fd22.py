#!/usr/bin/env python3
"""
FD-22: _check_skin_tone [pipeline] — Real Face Crop → result[0] == True
Target : _check_skin_tone(pipeline.py:636)
Extract: Best face crop from CCTV footage (reuse FD-16 or FD-05).
Test   : Call pipeline._check_skin_tone(real_face_crop).
Expect : result[0] == True
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import utils

TEST_ID = "FD-22"
TITLE   = "_check_skin_tone [pipeline]: real face crop → True"
IMG     = "face_crop.jpg"


def extract():
    if utils.load_img(TEST_ID, IMG) is not None:
        print(f"[{TEST_ID}] Data already extracted.")
        return True
    for src_id, src_file in [("FD-16", "face_skin.jpg"), ("FD-05", "face_best.jpg"), ("FD-01", "face_crop.jpg")]:
        src = utils.load_img(src_id, src_file)
        if src is not None:
            utils.save_img(TEST_ID, IMG, src)
            print(f"[{TEST_ID}] Reused {src_id} face crop.")
            return True
    model = utils.load_yolo_face()
    if model is None:
        return False
    dets = utils.scan_faces(model, max_faces=200, step=15, n_vids=3)
    if not dets:
        return False
    utils.save_img(TEST_ID, IMG, utils.best_face(dets)['crop'])
    return True


def test():
    from pipeline import _check_skin_tone as pipeline_check_skin_tone
    crop = utils.load_img(TEST_ID, IMG)
    if crop is None:
        print(f"[{TEST_ID}] SKIP — no extracted data.")
        return
    result = pipeline_check_skin_tone(crop)
    passed = (result[0] is True)
    utils.report(TEST_ID, TITLE, passed,
                 actual=f"result[0]={result[0]}, ratio={result[1]:.3f}",
                 expected="result[0] == True")


if __name__ == '__main__':
    ok = extract()
    if ok:
        test()
