#!/usr/bin/env python3
"""
FD-13: check_quality — q < 25 filename → (False, 18, reason has 'quality_18_below_25')
Target : check_quality(filter_faces.py)
Extract: Take any real face crop. Save a copy with q=18 in the filename,
         simulating a low-quality face file from the pipeline.
Test   : Call check_quality("face_001234_q18.jpg", min_quality=25).
Expect : (False, 18, reason containing 'quality_18_below_25')
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import utils

TEST_ID = "FD-13"
TITLE   = "check_quality: q=18 < 25 filename → (False, 18, 'quality_18_below_25')"
IMG     = "face_001234_q18.jpg"


def extract():
    # check_quality only uses the filename; just need the file to exist
    p = utils.data_path(TEST_ID, IMG)
    if os.path.exists(p):
        print(f"[{TEST_ID}] Data already extracted.")
        return True
    # Reuse any existing crop
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
    print(f"[{TEST_ID}] Saved {IMG} (q=18 forced in filename).")
    return True


def test():
    from filter_faces import check_quality
    p = utils.data_path(TEST_ID, IMG)
    if not os.path.exists(p):
        print(f"[{TEST_ID}] SKIP — no extracted data.")
        return
    passed_flag, score, reason = check_quality(IMG, min_quality=25)
    passed = (passed_flag is False and score == 18 and 'quality_18_below_25' in reason)
    utils.report(TEST_ID, TITLE, passed,
                 actual=f"passed={passed_flag}, score={score}, reason={reason!r}",
                 expected="(False, 18, reason contains 'quality_18_below_25')")


if __name__ == '__main__':
    ok = extract()
    if ok:
        test()
