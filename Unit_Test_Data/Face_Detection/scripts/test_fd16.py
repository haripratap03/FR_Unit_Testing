#!/usr/bin/env python3
"""
FD-16: check_skin_tone [filter_faces] — Real face crop → passed=True, ratio >= 0.20
Target : check_skin_tone(filter_faces.py)
Extract: Best face crop from CCTV footage — should contain real skin pixels.
Test   : Call check_skin_tone(real_face_crop, min_skin_ratio=0.20).
Expect : passed=True, ratio >= 0.20
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import utils

TEST_ID = "FD-16"
TITLE   = "check_skin_tone [filter_faces]: real face crop → passed=True, ratio >= 0.20"
IMG     = "face_skin.jpg"


def extract():
    if utils.load_img(TEST_ID, IMG) is not None:
        print(f"[{TEST_ID}] Data already extracted.")
        return True
    crop = utils.load_img("FD-05", "face_best.jpg")
    if crop is not None:
        utils.save_img(TEST_ID, IMG, crop)
        print(f"[{TEST_ID}] Reused FD-05 best face.")
        return True
    crop = utils.load_img("FD-01", "face_crop.jpg")
    if crop is not None:
        utils.save_img(TEST_ID, IMG, crop)
        print(f"[{TEST_ID}] Reused FD-01 face crop.")
        return True
    model = utils.load_yolo_face()
    if model is None:
        return False
    print(f"[{TEST_ID}] Scanning for a real face crop with skin tone...")
    dets = utils.scan_faces(model, max_faces=300, step=12, n_vids=4)
    if not dets:
        return False
    chosen = utils.best_face(dets)
    utils.save_img(TEST_ID, IMG, chosen['crop'])
    print(f"[{TEST_ID}] Saved face crop: {chosen['crop'].shape}, brightness={utils.brightness(chosen['crop']):.1f}")
    return True


def test():
    from filter_faces import check_skin_tone
    crop = utils.load_img(TEST_ID, IMG)
    if crop is None:
        print(f"[{TEST_ID}] SKIP — no extracted data.")
        return
    passed_flag, ratio, reason = check_skin_tone(crop, min_skin_ratio=0.20)
    passed = (passed_flag is True and ratio >= 0.20)
    utils.report(TEST_ID, TITLE, passed,
                 actual=f"passed={passed_flag}, ratio={ratio:.3f}, reason={reason!r}",
                 expected="passed=True, ratio >= 0.20")


if __name__ == '__main__':
    ok = extract()
    if ok:
        test()
