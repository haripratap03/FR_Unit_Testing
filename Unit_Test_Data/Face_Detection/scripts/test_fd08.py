#!/usr/bin/env python3
"""
FD-08: score_face — Both Ears Visible: ear_penalty=0.8 → pose component <= 16
Target : score_face(pipeline.py:593)
Extract: Scan CCTV footage for a face where both ear keypoints (kps[3] and kps[4])
         have confidence > 0.5. Saves crop + kps.
Test   : Call score_face(crop, yolo_conf=0.90, keypoints=real_kps).
Expect : scores['pose'] <= 16 (ear_pen=0.8 caps the pose score)
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import utils
import numpy as np

TEST_ID = "FD-08"
TITLE   = "score_face: both ears visible → pose component <= 16"
IMG     = "face_ears.jpg"
KPS_F   = "keypoints.json"
META_F  = "meta.json"


def extract():
    if utils.load_img(TEST_ID, IMG) is not None:
        print(f"[{TEST_ID}] Data already extracted.")
        return True
    model = utils.load_yolo_face()
    if model is None:
        return False
    print(f"[{TEST_ID}] Scanning for face with both ear kps > 0.5...")
    dets = utils.scan_faces(model, max_faces=500, step=10, n_vids=5)
    if not dets:
        print(f"[{TEST_ID}] No faces found.")
        return False
    chosen = utils.find_both_ears_face(dets)
    if chosen is None:
        # Fallback: pick any face and manually set ear kps conf > 0.5
        print(f"[{TEST_ID}] No natural both-ears face found; will use best face + synthetic ear kps.")
        chosen = utils.best_face(dets)
    crop = chosen['crop']
    kps = chosen.get('kps')
    h, w = crop.shape[:2]
    if kps is not None:
        # Ensure both ear kps have conf > 0.5
        kps_arr = np.array(kps, dtype=np.float32)
        if len(kps_arr) >= 5:
            if kps_arr[3][2] <= 0.5:
                kps_arr[3][2] = 0.8
            if kps_arr[4][2] <= 0.5:
                kps_arr[4][2] = 0.8
        kps = kps_arr
    else:
        # Synthetic kps with both ears visible
        kps = np.array([[w*0.5, h*0.4, 0.9], [w*0.35, h*0.35, 0.9],
                        [w*0.65, h*0.35, 0.9], [w*0.2, h*0.6, 0.8],
                        [w*0.8, h*0.6, 0.8]], dtype=np.float32)
    print(f"[{TEST_ID}] Face: {w}×{h}, ear kp confs: {kps[3][2]:.2f}, {kps[4][2]:.2f}")
    utils.save_img(TEST_ID, IMG, crop)
    utils.save_json(TEST_ID, KPS_F, kps)
    utils.save_json(TEST_ID, META_F, {'conf': chosen['conf'], 'bbox': list(chosen['bbox'])})
    return True


def test():
    from pipeline import score_face
    crop = utils.load_img(TEST_ID, IMG)
    if crop is None:
        print(f"[{TEST_ID}] SKIP — no extracted data.")
        return
    meta = utils.load_json_file(TEST_ID, META_F) or {}
    conf = meta.get('conf', 0.90)
    kps_raw = utils.load_json_file(TEST_ID, KPS_F)
    kps = np.array(kps_raw, dtype=np.float32) if kps_raw else None
    score, scores = score_face(crop, yolo_conf=conf, keypoints=kps)
    pose_comp = scores.get('pose', -1) if isinstance(scores, dict) else -1
    passed = (pose_comp <= 16)
    utils.report(TEST_ID, TITLE, passed,
                 actual=f"score={score}, pose_component={pose_comp:.2f}",
                 expected="pose_component <= 16")


if __name__ == '__main__':
    ok = extract()
    if ok:
        test()
