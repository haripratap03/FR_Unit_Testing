#!/usr/bin/env python3
"""
FD-41: _batch_capture_faces — Empty Frame (No Face in It) → No Face Saved
Target : ClipProcessor._batch_capture_faces(pipeline.py)
Extract: Find a frame with zero face detections in the footage (background only).
         Use that frame as the person crop input.
Test   : Pass the empty frame as person crop with real YOLO face model.
         YOLO will detect 0 faces → nothing submitted.
Expect : submit not called, face_images empty
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import utils
import tempfile
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import MagicMock

TEST_ID = "FD-41"
TITLE   = "_batch_capture_faces: empty frame (0 faces) → no face saved"
IMG     = "empty_frame.jpg"

_TS = datetime(2026, 4, 29, 9, 0, 0)


def make_processor(tracked_persons, face_model):
    proc = MagicMock(spec=[])
    proc.tracked_persons = tracked_persons
    proc.face_model = face_model
    proc._face_save_pool = MagicMock()
    proc.frame_count = 1000
    proc.stats = {'faces_captured': 0}
    return proc


def make_person(last_capture_offset_s=5.0):
    folder = tempfile.mkdtemp()
    p = {'track_id': 1, 'images_folder': folder,
         'face_images': [], 'best_face_image': None}
    last_dt = _TS - timedelta(seconds=last_capture_offset_s)
    p['_last_face_capture'] = last_dt.isoformat()
    return p


def extract():
    if utils.load_img(TEST_ID, IMG) is not None:
        print(f"[{TEST_ID}] Data already extracted.")
        return True
    # Reuse FD-17 background if available
    src = utils.load_img("FD-17", "background_crop.jpg")
    if src is not None:
        # Upscale so it's a believable frame size
        import cv2
        h, w = src.shape[:2]
        if w < 100:
            src = cv2.resize(src, (200, 200), interpolation=cv2.INTER_LINEAR)
        utils.save_img(TEST_ID, IMG, src)
        print(f"[{TEST_ID}] Reused FD-17 background (resized to frame).")
        return True
    model = utils.load_yolo_face()
    if model is None:
        return False
    print(f"[{TEST_ID}] Scanning for empty frame (0 face detections)...")
    frame = utils.find_empty_frame(model, n_vids=4, step=10)
    if frame is not None:
        utils.save_img(TEST_ID, IMG, frame)
        h, w = frame.shape[:2]
        print(f"[{TEST_ID}] Saved empty frame: {w}×{h}")
        return True
    # Synthetic fallback — solid gray frame, no faces
    print(f"[{TEST_ID}] No empty frame found; using synthetic gray frame.")
    frame = np.full((480, 640, 3), 110, dtype=np.uint8)
    utils.save_img(TEST_ID, IMG, frame)
    return True


def test():
    from pipeline import ClipProcessor
    frame = utils.load_img(TEST_ID, IMG)
    if frame is None:
        print(f"[{TEST_ID}] SKIP — no extracted data.")
        return
    face_model = utils.load_yolo_face()
    if face_model is None:
        print(f"[{TEST_ID}] SKIP — YOLO face model not available.")
        return
    uid = 1
    person = make_person(last_capture_offset_s=5.0)
    h, w = frame.shape[:2]
    proc = make_processor({uid: person}, face_model)
    face_targets = [(uid, frame, (0, 0), [0, 0, w, h])]
    ClipProcessor._batch_capture_faces(proc, face_targets, _TS)
    passed = (len(person['face_images']) == 0 and not proc._face_save_pool.submit.called)
    utils.report(TEST_ID, TITLE, passed,
                 actual=f"face_images={len(person['face_images'])}, submit_called={proc._face_save_pool.submit.called}",
                 expected="face_images=0, submit not called")


if __name__ == '__main__':
    ok = extract()
    if ok:
        test()
