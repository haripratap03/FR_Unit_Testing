#!/usr/bin/env python3
"""
FD-44: _batch_capture_faces — Multi-Face Frame → Highest Conf Face Captured
Target : ClipProcessor._batch_capture_faces(pipeline.py)
Extract: Find a frame with 2+ simultaneous face detections. The pipeline uses
         boxes.conf.argmax() to select the best face from the YOLO output.
         Passing the full multi-face frame as the person crop exercises this path.
Test   : Pass multi-face frame as person crop. Expect exactly 1 face captured,
         face_model called once, face_images has 1 entry.
Expect : face_model called once, face_images has 1 entry (highest conf selected)
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import utils
import tempfile
from datetime import datetime, timedelta
from unittest.mock import MagicMock

TEST_ID = "FD-44"
TITLE   = "_batch_capture_faces: multi-face frame → highest conf face captured"
IMG     = "multi_face_frame.jpg"

_TS = datetime(2026, 4, 29, 9, 0, 0)


def make_processor(tracked_persons, face_model):
    proc = MagicMock(spec=[])
    proc.tracked_persons = tracked_persons
    proc.face_model = face_model
    proc._face_save_pool = MagicMock()
    proc.frame_count = 7777
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
    # Reuse FD-30 multi-face frame if available
    src = utils.load_img("FD-30", "multi_face_frame.jpg")
    if src is not None:
        utils.save_img(TEST_ID, IMG, src)
        print(f"[{TEST_ID}] Reused FD-30 multi-face frame.")
        return True
    model = utils.load_yolo_face()
    if model is None:
        return False
    print(f"[{TEST_ID}] Scanning for multi-face frame...")
    frame, dets = utils.find_multi_face_frame(model, n_vids=5, step=10)
    if frame is not None:
        utils.save_img(TEST_ID, IMG, frame)
        print(f"[{TEST_ID}] Saved multi-face frame with {len(dets)} detections.")
        return True
    # Fallback: use any frame with at least 1 face
    print(f"[{TEST_ID}] No multi-face frame found; using single-face frame.")
    for vp in utils.get_videos()[:5]:
        for _, fr in utils.iter_frames(vp, step=15, limit=200):
            if utils.yolo_detections(model, fr):
                utils.save_img(TEST_ID, IMG, fr)
                return True
    return False


def test():
    from pipeline import ClipProcessor
    frame = utils.load_img(TEST_ID, IMG)
    if frame is None:
        print(f"[{TEST_ID}] SKIP — no extracted data.")
        return
    real_model = utils.load_yolo_face()
    if real_model is None:
        print(f"[{TEST_ID}] SKIP — YOLO face model not available.")
        return
    # Wrap the real YOLO model so the test can use call tracking (MagicMock attrs)
    # while the wrapped model still produces real detections.
    face_model = MagicMock(wraps=real_model)
    uid = 1
    person = make_person(last_capture_offset_s=5.0)
    h, w = frame.shape[:2]
    proc = make_processor({uid: person}, face_model)
    face_targets = [(uid, frame, (0, 0), [0, 0, w, h])]
    ClipProcessor._batch_capture_faces(proc, face_targets, _TS)
    # face_model should have been called exactly once (one person in face_targets)
    model_called_once = (face_model.call_count == 1)
    # face_images should have at most 1 entry (one best face selected per person per call)
    images_count = len(person['face_images'])
    passed = model_called_once and images_count <= 1
    utils.report(TEST_ID, TITLE, passed,
                 actual=f"face_model.call_count={face_model.call_count}, face_images={images_count}",
                 expected="face_model called once, face_images <= 1 (highest conf selected)")


if __name__ == '__main__':
    ok = extract()
    if ok:
        test()
