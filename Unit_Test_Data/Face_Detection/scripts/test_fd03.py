#!/usr/bin/env python3
"""
FD-03: score_face — Hard Reject: Crop Too Small (35×35)
Target : score_face(pipeline.py:593)
Extract: Search for a naturally tiny face detection (distant person in frame).
         If smallest found is still > 40px on either side, resize to 35×35 to
         simulate a distant-person crop — this is realistic since the pipeline
         will encounter such crops from far-away persons.
Test   : Call score_face(crop_35x35, yolo_conf=0.90, keypoints=None).
Expect : score==0, reason contains 'too_small'.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import utils
import cv2

TEST_ID = "FD-03"
TITLE   = "score_face: Hard Reject 35×35 crop (too_small)"
IMG     = "face_tiny.jpg"


def extract():
    if utils.load_img(TEST_ID, IMG) is not None:
        print(f"[{TEST_ID}] Data already extracted.")
        return True
    model = utils.load_yolo_face()
    if model is None:
        return False
    print(f"[{TEST_ID}] Scanning for smallest face detection...")
    dets = utils.scan_faces(model, max_faces=400, step=10, n_vids=5)
    if not dets:
        return False
    smallest = utils.find_small_face(dets, max_px=50)
    crop = smallest['crop']
    h, w = crop.shape[:2]
    print(f"[{TEST_ID}] Smallest natural detection: {w}×{h}")
    # Resize to exactly 35×35 to guarantee the hard-gate condition
    tiny = cv2.resize(crop, (35, 35), interpolation=cv2.INTER_AREA)
    utils.save_img(TEST_ID, IMG, tiny)
    print(f"[{TEST_ID}] Saved 35×35 crop (resized from {w}×{h} natural detection).")
    return True


def test():
    from pipeline import score_face
    crop = utils.load_img(TEST_ID, IMG)
    if crop is None:
        print(f"[{TEST_ID}] SKIP — no extracted data.")
        return
    h, w = crop.shape[:2]
    score, reason = score_face(crop, yolo_conf=0.90, keypoints=None)
    passed = (score == 0 and isinstance(reason, str) and "too_small" in reason)
    utils.report(TEST_ID, TITLE, passed,
                 actual=f"crop={w}×{h}, score={score}, reason={reason!r}",
                 expected="score=0, reason contains 'too_small'")


if __name__ == '__main__':
    ok = extract()
    if ok:
        test()
