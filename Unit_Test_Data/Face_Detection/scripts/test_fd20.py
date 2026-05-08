#!/usr/bin/env python3
"""
FD-20: check_keypoints [filter_faces] — Partial/Edge Face → passed=False, count < 3
Target : check_keypoints(filter_faces.py)
Extract: Take a real face crop and slice only the left half (losing right eye + right ear).
         InsightFace may detect a face but will have fewer landmarks inside the bbox.
         Requires InsightFace.
Test   : Call check_keypoints(app, partial_crop, min_keypoints=3).
Expect : passed=False, count < 3
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import utils
import cv2

TEST_ID = "FD-20"
TITLE   = "check_keypoints [filter_faces]: partial face crop → passed=False, count < 3"
IMG     = "face_partial.jpg"


def extract():
    if utils.load_img(TEST_ID, IMG) is not None:
        print(f"[{TEST_ID}] Data already extracted.")
        return True
    # Load a good face and take a narrow side strip
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
    h, w = crop.shape[:2]
    # Crop left 35% only — lose most of face structure
    partial = crop[:, :max(w//3, 30)].copy()
    # Upscale so InsightFace can process it at all (but landmarks mostly outside)
    partial = cv2.resize(partial, (max(partial.shape[1], 80), max(partial.shape[0], 80)),
                         interpolation=cv2.INTER_LINEAR)
    utils.save_img(TEST_ID, IMG, partial)
    print(f"[{TEST_ID}] Saved partial face: {partial.shape} (left third of {w}×{h} original).")
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
    passed = (passed_flag is False and count < 3)
    utils.report(TEST_ID, TITLE, passed,
                 actual=f"passed={passed_flag}, count={count}, reason={reason!r}",
                 expected="passed=False, count < 3")


if __name__ == '__main__':
    ok = extract()
    if ok:
        test()
