#!/usr/bin/env python3
"""
FD-19: check_keypoints [filter_faces] — Real Face + InsightFace → passed=True, count >= 3
Target : check_keypoints(filter_faces.py)
Extract: Best face crop from CCTV footage (reuse FD-05 if available).
         Requires InsightFace. Skips gracefully if models not present.
Test   : Call check_keypoints(app, real_face_crop, min_keypoints=3).
Expect : passed=True, count >= 3, reason == ''
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import utils

TEST_ID = "FD-19"
TITLE   = "check_keypoints [filter_faces]: real face → passed=True, count >= 3"
IMG     = "face_kp.jpg"


def extract():
    if utils.load_img(TEST_ID, IMG) is not None:
        print(f"[{TEST_ID}] Data already extracted.")
        return True
    crop = utils.load_img("FD-05", "face_best.jpg")
    if crop is None:
        crop = utils.load_img("FD-01", "face_crop.jpg")
    if crop is None:
        model = utils.load_yolo_face()
        if model is None:
            return False
        dets = utils.scan_faces(model, max_faces=200, step=15, n_vids=3)
        if not dets:
            return False
        crop = utils.best_face(dets)['crop']
    utils.save_img(TEST_ID, IMG, crop)
    print(f"[{TEST_ID}] Saved face crop: {crop.shape}")
    return True


def test():
    from filter_faces import check_keypoints
    crop = utils.load_img(TEST_ID, IMG)
    if crop is None:
        print(f"[{TEST_ID}] SKIP — no extracted data.")
        return
    app = utils.load_insightface(det_thresh=0.15)
    if app is None:
        print(f"[{TEST_ID}] SKIP — InsightFace not available.")
        return
    passed_flag, count, reason = check_keypoints(app, crop, min_keypoints=3)
    passed = (passed_flag is True and count >= 3 and reason == '')
    utils.report(TEST_ID, TITLE, passed,
                 actual=f"passed={passed_flag}, count={count}, reason={reason!r}",
                 expected="passed=True, count >= 3, reason=''")


if __name__ == '__main__':
    ok = extract()
    if ok:
        test()
