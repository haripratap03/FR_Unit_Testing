#!/usr/bin/env python3
"""
FD-01: score_face — Hard Reject: YOLO Confidence Below 0.4
Target : score_face(pipeline.py:593)
Extract: Clear frontal face crop from real CCTV footage.
Test   : Call score_face with yolo_conf=0.35 (below 0.4 gate).
Expect : score==0, reason contains 'low_yolo'.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import utils
import cv2

TEST_ID = "FD-01"
TITLE   = "score_face: Hard Reject yolo_conf=0.35 < 0.4"
IMG     = "face_crop.jpg"


def extract():
    if utils.load_img(TEST_ID, IMG) is not None:
        print(f"[{TEST_ID}] Data already extracted — skipping video scan.")
        return True
    model = utils.load_yolo_face()
    if model is None:
        return False
    print(f"[{TEST_ID}] Scanning videos for a clear face crop...")
    dets = utils.scan_faces(model, max_faces=200, step=15, n_vids=3)
    if not dets:
        print(f"[{TEST_ID}] No faces found in video.")
        return False
    chosen = utils.best_face(dets)
    utils.save_img(TEST_ID, IMG, chosen['crop'])
    bv = utils.lap_var(chosen['crop'])
    print(f"[{TEST_ID}] Saved crop {chosen['crop'].shape} | blur_var={bv:.1f} | conf={chosen['conf']:.3f}")
    return True


def test():
    from pipeline import score_face
    crop = utils.load_img(TEST_ID, IMG)
    if crop is None:
        print(f"[{TEST_ID}] SKIP — no extracted data found. Run extract() first.")
        return
    score, reason = score_face(crop, yolo_conf=0.35, keypoints=None)
    passed = (score == 0 and isinstance(reason, str) and "low_yolo" in reason)
    utils.report(TEST_ID, TITLE, passed,
                 actual=f"score={score}, reason={reason!r}",
                 expected="score=0, reason contains 'low_yolo'")


if __name__ == '__main__':
    ok = extract()
    if ok:
        test()
