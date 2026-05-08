#!/usr/bin/env python3
"""
FD-10: score_face — keypoints=None → pose component == 10 (neutral fallback)
Target : score_face(pipeline.py:593)
Extract: Any real face crop (reuse FD-01 or FD-05 if available).
Test   : Call score_face(crop, yolo_conf=0.80, keypoints=None).
Expect : scores['pose'] == 10
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import utils

TEST_ID = "FD-10"
TITLE   = "score_face: keypoints=None → pose component == 10"
IMG     = "face_crop.jpg"


def extract():
    if utils.load_img(TEST_ID, IMG) is not None:
        print(f"[{TEST_ID}] Data already extracted.")
        return True
    # Try reusing existing crops
    for src_id, src_file in [("FD-05", "face_best.jpg"), ("FD-01", "face_crop.jpg")]:
        src = utils.load_img(src_id, src_file)
        if src is not None:
            utils.save_img(TEST_ID, IMG, src)
            print(f"[{TEST_ID}] Reused {src_id} face crop.")
            return True
    model = utils.load_yolo_face()
    if model is None:
        return False
    print(f"[{TEST_ID}] Scanning for a face crop...")
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
    score, scores = score_face(crop, yolo_conf=0.80, keypoints=None)
    pose_comp = scores.get('pose', -1) if isinstance(scores, dict) else -1
    passed = (pose_comp == 10)
    utils.report(TEST_ID, TITLE, passed,
                 actual=f"score={score}, pose_component={pose_comp}",
                 expected="pose_component == 10")


if __name__ == '__main__':
    ok = extract()
    if ok:
        test()
