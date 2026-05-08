#!/usr/bin/env python3
"""
FD-02: score_face — YOLO Confidence Exactly 0.4 (Boundary)
Target : score_face(pipeline.py:593)
Extract: Reuses FD-01 face crop (same clear frontal face).
Test   : Call score_face with yolo_conf=0.40 (exactly at boundary gate <0.4).
Expect : score > 0 (0.40 passes because gate is strictly < 0.4).
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import utils

TEST_ID = "FD-02"
TITLE   = "score_face: Boundary yolo_conf=0.40 passes (gate is < 0.4)"
IMG     = "face_crop.jpg"


def extract():
    if utils.load_img(TEST_ID, IMG) is not None:
        print(f"[{TEST_ID}] Data already extracted.")
        return True
    # Reuse FD-01 crop if available
    src = utils.load_img("FD-01", "face_crop.jpg")
    if src is not None:
        utils.save_img(TEST_ID, IMG, src)
        print(f"[{TEST_ID}] Reused FD-01 face crop.")
        return True
    # Otherwise extract fresh
    model = utils.load_yolo_face()
    if model is None:
        return False
    print(f"[{TEST_ID}] Scanning for face crop...")
    dets = utils.scan_faces(model, max_faces=100, step=20, n_vids=2)
    if not dets:
        return False
    utils.save_img(TEST_ID, IMG, utils.best_face(dets)['crop'])
    return True


def test():
    from pipeline import score_face
    crop = utils.load_img(TEST_ID, IMG)
    if crop is None:
        print(f"[{TEST_ID}] SKIP — no extracted data.")
        return
    score, scores_or_reason = score_face(crop, yolo_conf=0.40, keypoints=None)
    passed = (score > 0)
    utils.report(TEST_ID, TITLE, passed,
                 actual=f"score={score}",
                 expected="score > 0")


if __name__ == '__main__':
    ok = extract()
    if ok:
        test()
