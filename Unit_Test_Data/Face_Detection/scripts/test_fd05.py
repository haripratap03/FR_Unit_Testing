#!/usr/bin/env python3
"""
FD-05: score_face — Perfect Frontal Face, Maximum Score
Target : score_face(pipeline.py:593)
Extract: Best face from CCTV footage — scan for: size ≥ 120px, high blur
         (sharp), brightness ~130, symmetric nose keypoints (frontal pose).
         Saves crop + YOLO keypoints JSON.
Test   : Call score_face(best_crop, conf, real_kps).
Expect : score >= 70; all five components (yolo, blur, size, brightness,
         pose) > 0.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import utils
import numpy as np

TEST_ID = "FD-05"
TITLE   = "score_face: Perfect frontal face → score >= 70"
IMG     = "face_best.jpg"
KPS_F   = "keypoints.json"
META_F  = "meta.json"


def extract():
    # Skip re-extraction only if the saved meta carries a usable yolo_conf.
    # score_face's hard `low_yolo` gate trips at conf < 0.4, so anything
    # below that means an older extract() (pre-conf>=0.5 filter) saved a junk
    # crop and the test will FAIL until we refresh the data. Keypoints are NOT
    # required: yolov11s-face.pt is non-pose and never returns kps, in which
    # case score_face falls back to pose=10 — still positive, still passes.
    if utils.load_img(TEST_ID, IMG) is not None:
        meta = utils.load_json_file(TEST_ID, META_F) or {}
        if meta.get('conf', 0.0) >= 0.5:
            print(f"[{TEST_ID}] Data already extracted (conf={meta.get('conf'):.2f}).")
            return True
        print(f"[{TEST_ID}] Stale data (conf={meta.get('conf', 0):.2f}); re-extracting.")
    model = utils.load_yolo_face()
    if model is None:
        return False
    print(f"[{TEST_ID}] Scanning for best frontal face (≥120px, sharp, symmetric)...")
    dets = utils.scan_faces(model, max_faces=500, step=10, n_vids=5)
    # score_face hard-rejects any detection with yolo_conf < 0.4 (the FD-01 gate),
    # so this test must only ever save a high-confidence crop. Pre-filter accordingly.
    dets = [d for d in dets if d['conf'] >= 0.5]
    if not dets:
        return False

    # Score each detection by: size * sharpness * brightness_quality * keypoint_symmetry
    def face_score(d):
        c = d['crop']
        h, w = c.shape[:2]
        if min(h, w) < 60:
            return 0
        bv = utils.lap_var(c)
        bright = utils.brightness(c)
        bright_sc = max(1 - abs(bright - 130) / 130, 0)
        size_sc = min(min(h, w) / 120, 1.0)
        sym_sc = 0.5  # default if no kps
        kps = d.get('kps')
        if kps is not None and len(kps) >= 3:
            nose, leye, reye = kps[0], kps[1], kps[2]
            if nose[2] >= 0.3 and leye[2] >= 0.3 and reye[2] >= 0.3:
                ld = abs(nose[0] - leye[0])
                rd = abs(nose[0] - reye[0])
                if max(ld, rd) > 0:
                    sym_sc = min(ld, rd) / max(ld, rd)
        return bv * size_sc * bright_sc * sym_sc

    best = max(dets, key=face_score)
    c = best['crop']
    h, w = c.shape[:2]
    bv = utils.lap_var(c)
    br = utils.brightness(c)
    print(f"[{TEST_ID}] Best face: {w}×{h}, blur_var={bv:.1f}, brightness={br:.1f}, conf={best['conf']:.3f}")
    utils.save_img(TEST_ID, IMG, c)
    kps = best.get('kps')
    if kps is not None:
        utils.save_json(TEST_ID, KPS_F, kps)
    utils.save_json(TEST_ID, META_F, {'conf': best['conf'], 'bbox': list(best['bbox'])})
    return True


def test():
    from pipeline import score_face
    crop = utils.load_img(TEST_ID, IMG)
    if crop is None:
        print(f"[{TEST_ID}] SKIP — no extracted data.")
        return
    meta = utils.load_json_file(TEST_ID, META_F) or {}
    conf = meta.get('conf', 0.80)
    kps_raw = utils.load_json_file(TEST_ID, KPS_F)
    kps = np.array(kps_raw, dtype=np.float32) if kps_raw else None

    score, scores = score_face(crop, yolo_conf=conf, keypoints=kps)

    all_positive = isinstance(scores, dict) and all(v > 0 for v in scores.values())
    passed = (score >= 70 and all_positive)
    comp_str = str({k: round(v, 2) for k, v in scores.items()}) if isinstance(scores, dict) else str(scores)
    utils.report(TEST_ID, TITLE, passed,
                 actual=f"score={score}, components={comp_str}",
                 expected="score >= 70, all components > 0")


if __name__ == '__main__':
    ok = extract()
    if ok:
        test()
