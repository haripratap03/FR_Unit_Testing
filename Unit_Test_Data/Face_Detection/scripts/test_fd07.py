#!/usr/bin/env python3
"""
FD-07: score_face — Profile Face: Asymmetric Keypoints → pose component < 5
Target : score_face(pipeline.py:593)
Extract: Scan CCTV footage for the most profile/asymmetric face detection
         (nose closest to one eye relative to the other). Saves crop + kps JSON.
Test   : Call score_face(profile_crop, yolo_conf=0.80, keypoints=real_kps).
Expect : scores['pose'] < 5
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import utils
import numpy as np

TEST_ID = "FD-07"
TITLE   = "score_face: profile face (asymmetric kps) → pose component < 5"
IMG     = "face_profile.jpg"
KPS_F   = "keypoints.json"
META_F  = "meta.json"


def extract():
    # Profile-pose test depends on real keypoints — without them score_face
    # falls back to pose=10 which violates the pose<5 expectation. Re-extract
    # whenever keypoints.json is missing or empty.
    if utils.load_img(TEST_ID, IMG) is not None:
        kps_raw = utils.load_json_file(TEST_ID, KPS_F)
        if kps_raw is not None and len(kps_raw) >= 3:
            print(f"[{TEST_ID}] Data already extracted (kps OK).")
            return True
        print(f"[{TEST_ID}] Stale data (no keypoints.json); re-extracting.")
    model = utils.load_yolo_face()
    if model is None:
        return False
    print(f"[{TEST_ID}] Scanning for any face (kps will come from InsightFace, not YOLO)...")
    dets = utils.scan_faces(model, max_faces=500, step=10, n_vids=5, validate=False)
    dets = [d for d in dets if d['conf'] >= 0.4]
    if not dets:
        print(f"[{TEST_ID}] No faces found.")
        return False

    # The YOLO face model in this repo (yolov11s-face.pt) is non-pose: every
    # detection has kps=None. So we can't pick the most-asymmetric YOLO face
    # by keypoints. Two-step strategy instead:
    #   1) Use InsightFace.kps (5 landmarks: leye, reye, nose, lmouth, rmouth)
    #      to compute asymmetry on each YOLO crop;
    #   2) Pick the most asymmetric crop and convert InsightFace kps to
    #      score_face's expected layout: [nose, leye, reye, lear, rear] each
    #      [x, y, conf]. InsightFace doesn't expose per-kp confidence, so we
    #      mark the visible landmarks with conf=1.0 and the missing ear pair
    #      with conf=0.0 (they aren't required by score_face's pose math).
    app = utils.load_insightface(det_thresh=0.30)
    if app is None:
        print(f"[{TEST_ID}] InsightFace unavailable — cannot extract real keypoints.")
        return False

    def asym_score_from_if(crop):
        """Run InsightFace on a crop, return (faces_obj, asym_score 0..1) or None."""
        try:
            faces = app.get(crop)
        except Exception:
            return None
        if not faces:
            return None
        f = max(faces, key=lambda fc: fc.bbox[2] - fc.bbox[0])  # widest face
        if f.kps is None or len(f.kps) < 3:
            return None
        leye, reye, nose = f.kps[0], f.kps[1], f.kps[2]
        ld = abs(nose[0] - leye[0])
        rd = abs(nose[0] - reye[0])
        if max(ld, rd) == 0:
            return None
        return f, 1.0 - (min(ld, rd) / max(ld, rd))

    scored = []
    for d in dets:
        result = asym_score_from_if(d['crop'])
        if result is not None:
            f, asym = result
            scored.append((d, f, asym))
    if not scored:
        print(f"[{TEST_ID}] No InsightFace landmarks found in any YOLO face crop.")
        return False
    chosen_d, chosen_f, chosen_asym = max(scored, key=lambda t: t[2])

    # Convert InsightFace kps order [leye, reye, nose, lmouth, rmouth] →
    # score_face order [nose, leye, reye, lear, rear] with conf column.
    leye, reye, nose = chosen_f.kps[0], chosen_f.kps[1], chosen_f.kps[2]
    kps = [
        [float(nose[0]), float(nose[1]), 1.0],
        [float(leye[0]), float(leye[1]), 1.0],
        [float(reye[0]), float(reye[1]), 1.0],
        [0.0, 0.0, 0.0],   # left ear — InsightFace 5-kps doesn't include ears
        [0.0, 0.0, 0.0],   # right ear
    ]

    crop = chosen_d['crop']
    h, w = crop.shape[:2]
    print(f"[{TEST_ID}] Profile face: {w}×{h}, conf={chosen_d['conf']:.3f}, asym={chosen_asym:.3f}")
    utils.save_img(TEST_ID, IMG, crop)
    utils.save_json(TEST_ID, KPS_F, kps)
    utils.save_json(TEST_ID, META_F, {'conf': chosen_d['conf'], 'bbox': list(chosen_d['bbox'])})
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
    pose_comp = scores.get('pose', -1) if isinstance(scores, dict) else -1
    passed = (pose_comp < 5)
    utils.report(TEST_ID, TITLE, passed,
                 actual=f"score={score}, pose_component={pose_comp:.2f}",
                 expected="pose_component < 5")


if __name__ == '__main__':
    ok = extract()
    if ok:
        test()
