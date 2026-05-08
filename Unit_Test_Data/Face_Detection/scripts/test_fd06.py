#!/usr/bin/env python3
"""
FD-06: score_face — Blurry Crop: Score > 0 but blur component < 1.0
Target : score_face(pipeline.py:593)
Extract: Scan CCTV footage for a naturally blurry face (low Laplacian variance).
         If none found below lap_var=80, take the blurriest available.
Test   : Call score_face(blurry_crop, yolo_conf=0.75). Expect score > 0 and
         scores['blur'] < 1.0.
Expect : score > 0; scores['blur'] < 1.0
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import utils
import cv2

TEST_ID = "FD-06"
TITLE   = "score_face: blurry crop → score > 0 and blur component < 1.0"
IMG     = "face_blurry.jpg"


def extract():
    if utils.load_img(TEST_ID, IMG) is not None:
        print(f"[{TEST_ID}] Data already extracted.")
        return True
    model = utils.load_yolo_face()
    if model is None:
        return False
    print(f"[{TEST_ID}] Scanning for blurry face crop (max_lap_var=80)...")
    dets = utils.scan_faces(model, max_faces=400, step=10, n_vids=5)
    if not dets:
        print(f"[{TEST_ID}] No faces found in video.")
        return False
    chosen = utils.find_blurry_face(dets, max_lap_var=80)
    if chosen is None:
        chosen = utils.find_blurry_face(dets)
    lv = utils.lap_var(chosen['crop'])
    print(f"[{TEST_ID}] Blurriest face: {chosen['crop'].shape}, lap_var={lv:.1f}, conf={chosen['conf']:.3f}")
    # If still very sharp, apply a blur to guarantee the component test
    crop = chosen['crop']
    if lv > 80:
        print(f"[{TEST_ID}] Applying GaussianBlur to ensure blurry condition.")
        crop = cv2.GaussianBlur(crop, (15, 15), 0)
    utils.save_img(TEST_ID, IMG, crop)
    print(f"[{TEST_ID}] Saved blurry crop. Final lap_var={utils.lap_var(crop):.1f}")
    return True


def test():
    from pipeline import score_face
    crop = utils.load_img(TEST_ID, IMG)
    if crop is None:
        print(f"[{TEST_ID}] SKIP — no extracted data.")
        return
    lv = utils.lap_var(crop)
    score, scores = score_face(crop, yolo_conf=0.75, keypoints=None)
    blur_comp = scores.get('blur', -1) if isinstance(scores, dict) else -1
    passed = (score > 0 and blur_comp < 1.0)
    utils.report(TEST_ID, TITLE, passed,
                 actual=f"score={score}, blur_component={blur_comp:.4f}, lap_var={lv:.1f}",
                 expected="score > 0, blur_component < 1.0")


if __name__ == '__main__':
    ok = extract()
    if ok:
        test()
