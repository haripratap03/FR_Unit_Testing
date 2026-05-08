#!/usr/bin/env python3
"""
FD-09: score_face — All Keypoint Confs Forced to 0.1 → pose component == 6.0
Target : score_face(pipeline.py:593)
Extract: Take the best frontal face (reuse FD-05 if available). Load its YOLO kps.
         Force ALL kp confidences to 0.1 (below the 0.3 threshold).
Test   : Call score_face(crop, yolo_conf=0.80, keypoints=modified_kps).
Expect : scores['pose'] == 6.0  (the fixed fallback value for all-low-conf kps)
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import utils
import numpy as np

TEST_ID = "FD-09"
TITLE   = "score_face: all kp confs < 0.3 → pose component == 6.0"
IMG     = "face_crop.jpg"
KPS_F   = "keypoints.json"


def extract():
    if utils.load_img(TEST_ID, IMG) is not None:
        print(f"[{TEST_ID}] Data already extracted.")
        return True
    # Reuse FD-05 best face if available
    src = utils.load_img("FD-05", "face_best.jpg")
    if src is not None:
        utils.save_img(TEST_ID, IMG, src)
        kps_raw = utils.load_json_file("FD-05", "keypoints.json")
        if kps_raw is not None:
            kps = np.array(kps_raw, dtype=np.float32)
            # Force all confs to 0.1
            kps[:, 2] = 0.1
            utils.save_json(TEST_ID, KPS_F, kps)
        else:
            # Synthetic low-conf kps
            kps = np.array([[50, 40, 0.1], [30, 35, 0.1], [70, 35, 0.1],
                            [30, 65, 0.1], [70, 65, 0.1]], dtype=np.float32)
            utils.save_json(TEST_ID, KPS_F, kps)
        print(f"[{TEST_ID}] Reused FD-05 face crop, forced all kp confs to 0.1.")
        return True
    # Fall back to scanning
    model = utils.load_yolo_face()
    if model is None:
        return False
    print(f"[{TEST_ID}] Scanning for a clear face crop...")
    dets = utils.scan_faces(model, max_faces=300, step=15, n_vids=3)
    if not dets:
        return False
    chosen = utils.best_face(dets)
    crop = chosen['crop']
    utils.save_img(TEST_ID, IMG, crop)
    kps = chosen.get('kps')
    if kps is not None:
        kps_arr = np.array(kps, dtype=np.float32)
        kps_arr[:, 2] = 0.1
        utils.save_json(TEST_ID, KPS_F, kps_arr)
    else:
        kps_arr = np.array([[50, 40, 0.1], [30, 35, 0.1], [70, 35, 0.1],
                             [30, 65, 0.1], [70, 65, 0.1]], dtype=np.float32)
        utils.save_json(TEST_ID, KPS_F, kps_arr)
    print(f"[{TEST_ID}] Saved face crop with forced-low kp confs.")
    return True


def test():
    from pipeline import score_face
    crop = utils.load_img(TEST_ID, IMG)
    if crop is None:
        print(f"[{TEST_ID}] SKIP — no extracted data.")
        return
    kps_raw = utils.load_json_file(TEST_ID, KPS_F)
    kps = np.array(kps_raw, dtype=np.float32) if kps_raw else None
    score, scores = score_face(crop, yolo_conf=0.80, keypoints=kps)
    pose_comp = scores.get('pose', -1) if isinstance(scores, dict) else -1
    passed = (abs(pose_comp - 6.0) < 1e-5)
    utils.report(TEST_ID, TITLE, passed,
                 actual=f"score={score}, pose_component={pose_comp}",
                 expected="pose_component == 6.0")


if __name__ == '__main__':
    ok = extract()
    if ok:
        test()
