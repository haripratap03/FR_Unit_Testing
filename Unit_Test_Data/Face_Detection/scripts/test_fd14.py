#!/usr/bin/env python3
"""
FD-14: check_quality — No _qNN tag in filename → (True, -1, '')
Target : check_quality(filter_faces.py)
Extract: Save any real face crop as 'face_real.jpg' (no quality tag).
Test   : Call check_quality("face_real.jpg", min_quality=25).
Expect : (True, -1, '')  — files without _qNN pass unconditionally
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import utils

TEST_ID = "FD-14"
TITLE   = "check_quality: no _qNN tag → (True, -1, '')"
IMG     = "face_real.jpg"


def extract():
    p = utils.data_path(TEST_ID, IMG)
    if os.path.exists(p):
        print(f"[{TEST_ID}] Data already extracted.")
        return True
    crop = utils.load_img("FD-05", "face_best.jpg")
    if crop is None:
        crop = utils.load_img("FD-01", "face_crop.jpg")
    if crop is None:
        model = utils.load_yolo_face()
        if model is None:
            return False
        dets = utils.scan_faces(model, max_faces=100, step=20, n_vids=2)
        if not dets:
            return False
        crop = utils.best_face(dets)['crop']
    utils.save_img(TEST_ID, IMG, crop)
    print(f"[{TEST_ID}] Saved {IMG} (no quality tag in filename).")
    return True


def test():
    from filter_faces import check_quality
    p = utils.data_path(TEST_ID, IMG)
    if not os.path.exists(p):
        print(f"[{TEST_ID}] SKIP — no extracted data.")
        return
    passed_flag, score, reason = check_quality(IMG, min_quality=25)
    passed = (passed_flag is True and score == -1 and reason == '')
    utils.report(TEST_ID, TITLE, passed,
                 actual=f"passed={passed_flag}, score={score}, reason={reason!r}",
                 expected="(True, -1, '')")


if __name__ == '__main__':
    ok = extract()
    if ok:
        test()
