#!/usr/bin/env python3
"""
FD-31: _detect_face — Single Real Face → ndarray shape (512,), norm ≈ 1.0
Target : _detect_face(rerun_fr.py)
Extract: Best face crop (reuse FD-05 or FD-01). Requires InsightFace.
Test   : Call _detect_face(app, face_crop).
Expect : returns ndarray shape (512,), np.linalg.norm(result) ≈ 1.0
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import utils
import numpy as np

TEST_ID = "FD-31"
TITLE   = "_detect_face: single real face → ndarray(512,), norm ≈ 1.0"
IMG     = "face_crop.jpg"


def extract():
    if utils.load_img(TEST_ID, IMG) is not None:
        print(f"[{TEST_ID}] Data already extracted.")
        return True
    for src_id, src_file in [("FD-05", "face_best.jpg"), ("FD-01", "face_crop.jpg")]:
        src = utils.load_img(src_id, src_file)
        if src is not None:
            utils.save_img(TEST_ID, IMG, src)
            print(f"[{TEST_ID}] Reused {src_id} crop.")
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
    if result is None:
        utils.report(TEST_ID, TITLE, False,
                     actual="result is None (InsightFace found no face in crop)",
                     expected="ndarray shape (512,), norm ≈ 1.0")
        return
    norm = float(np.linalg.norm(result))
    passed = (result.shape == (512,) and abs(norm - 1.0) < 1e-4)
    utils.report(TEST_ID, TITLE, passed,
                 actual=f"shape={result.shape}, norm={norm:.6f}",
                 expected="shape=(512,), norm ≈ 1.0")


if __name__ == '__main__':
    ok = extract()
    if ok:
        test()
