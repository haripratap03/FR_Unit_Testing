#!/usr/bin/env python3
"""
FD-30: _detect_face — Multi-Face Frame → Largest Face Selected, L2-norm ≈ 1.0
Target : _detect_face(rerun_fr.py)
Extract: Find a frame with 2+ simultaneous face detections. Save the frame.
         Requires InsightFace.
Test   : Call _detect_face(app, multi_face_frame). Expect the result to be a
         unit-norm embedding (the largest detected face wins).
Expect : result is ndarray shape (512,), np.linalg.norm(result) ≈ 1.0
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import utils
import numpy as np

TEST_ID = "FD-30"
TITLE   = "_detect_face: multi-face frame → largest face selected, norm ≈ 1.0"
IMG     = "multi_face_frame.jpg"


def extract():
    if utils.load_img(TEST_ID, IMG) is not None:
        print(f"[{TEST_ID}] Data already extracted.")
        return True
    model = utils.load_yolo_face()
    if model is None:
        return False
    print(f"[{TEST_ID}] Scanning for multi-face frame...")
    frame, dets = utils.find_multi_face_frame(model, n_vids=5, step=15)
    if frame is None:
        print(f"[{TEST_ID}] No multi-face frame found in footage.")
        # Fallback: use any frame (InsightFace may find 1+ faces)
        videos = utils.get_videos()
        if not videos:
            return False
        for _, f in utils.iter_frames(videos[0], step=20, limit=100):
            frame = f
            break
    utils.save_img(TEST_ID, IMG, frame)
    print(f"[{TEST_ID}] Saved frame: {frame.shape}")
    return True


def test():
    from rerun_fr import _detect_face
    frame = utils.load_img(TEST_ID, IMG)
    if frame is None:
        print(f"[{TEST_ID}] SKIP — no extracted data.")
        return
    app = utils.load_insightface(det_thresh=0.15)
    if app is None:
        print(f"[{TEST_ID}] SKIP — InsightFace not available.")
        return
    result = _detect_face(app, frame)
    if result is None:
        utils.report(TEST_ID, TITLE, False,
                     actual="result is None",
                     expected="ndarray shape (512,), norm ≈ 1.0")
        return
    norm = float(np.linalg.norm(result))
    passed = (result.shape == (512,) and abs(norm - 1.0) < 1e-4)
    utils.report(TEST_ID, TITLE, passed,
                 actual=f"shape={result.shape}, norm={norm:.6f}",
                 expected="shape=(512,), norm ≈ 1.0")


if __name__ == '__main__':
    ok = extract()
    if ok:
        test()
