#!/usr/bin/env python3
"""
FD-12: check_quality — Real face with q >= 25 filename → (True, score, '')
Target : check_quality(filter_faces.py)
Extract: Take a real face crop (reuse FD-05). Compute a good-quality score via
         score_face. Save a copy with filename face_001234_q{score}.jpg.
         Only the filename is used by check_quality (no image I/O).
Test   : Call check_quality("face_001234_q{score}.jpg", min_quality=25).
Expect : (True, score, '')
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import utils

TEST_ID = "FD-12"
TITLE   = "check_quality: q >= 25 filename → (True, score, '')"
META_F  = "meta.json"


def extract():
    meta = utils.load_json_file(TEST_ID, META_F)
    if meta is not None and 'fname' in meta:
        print(f"[{TEST_ID}] Data already extracted.")
        return True
    # Get a good face crop — reuse FD-05 or FD-01
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
    # Compute quality score
    from pipeline import score_face
    score, _ = score_face(crop, yolo_conf=0.80, keypoints=None)
    # If score too low, clamp to a safe value for testing
    q = max(int(score), 42)
    fname = f"face_001234_q{q}.jpg"
    utils.save_img(TEST_ID, fname, crop)
    utils.save_json(TEST_ID, META_F, {'fname': fname, 'q': q})
    print(f"[{TEST_ID}] Saved {fname} (score={score:.1f}, using q={q})")
    return True


def test():
    from filter_faces import check_quality
    meta = utils.load_json_file(TEST_ID, META_F)
    if meta is None:
        print(f"[{TEST_ID}] SKIP — no extracted data.")
        return
    fname = meta['fname']
    q = meta['q']
    passed_flag, score, reason = check_quality(fname, min_quality=25)
    passed = (passed_flag is True and score == q and reason == '')
    utils.report(TEST_ID, TITLE, passed,
                 actual=f"passed={passed_flag}, score={score}, reason={reason!r}",
                 expected=f"(True, {q}, '')")


if __name__ == '__main__':
    ok = extract()
    if ok:
        test()
