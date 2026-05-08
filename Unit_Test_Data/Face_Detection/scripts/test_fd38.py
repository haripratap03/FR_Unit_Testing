#!/usr/bin/env python3
"""
FD-38: _batch_capture_faces — Valid Detection, Cooldown Elapsed → Face Saved
Target : ClipProcessor._batch_capture_faces(pipeline.py)
Extract: Load a real CCTV video frame. Use it as the 'person crop'. The real
         YOLO face model will detect a face inside the frame.
Test   : Set up proc with real face_model, last_capture 5s ago, cooldown elapsed.
         Call _batch_capture_faces. Expect: face_images has 1 entry, submit called.
Expect : len(person['face_images']) == 1, _face_save_pool.submit called
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import utils
import tempfile
from datetime import datetime, timedelta
from unittest.mock import MagicMock

TEST_ID = "FD-38"
TITLE   = "_batch_capture_faces: valid det, cooldown elapsed → face saved"
IMG     = "person_frame.jpg"

_TS = datetime(2026, 4, 29, 9, 0, 0)


def make_processor(tracked_persons, face_model):
    proc = MagicMock(spec=[])
    proc.tracked_persons = tracked_persons
    proc.face_model = face_model
    proc._face_save_pool = MagicMock()
    proc.frame_count = 1000
    proc.stats = {'faces_captured': 0}
    return proc


def make_person(last_capture_offset_s=None, face_images=None):
    folder = tempfile.mkdtemp()
    p = {'track_id': 1, 'images_folder': folder,
         'face_images': face_images or [], 'best_face_image': None}
    if last_capture_offset_s is not None:
        last_dt = _TS - timedelta(seconds=last_capture_offset_s)
        p['_last_face_capture'] = last_dt.isoformat()
    return p


def extract():
    if utils.load_img(TEST_ID, IMG) is not None:
        print(f"[{TEST_ID}] Data already extracted.")
        return True
    videos = utils.get_videos()
    if not videos:
        print(f"[{TEST_ID}] No videos found.")
        return False
    model = utils.load_yolo_face()
    if model is None:
        return False
    print(f"[{TEST_ID}] Scanning for a frame with at least one face...")
    for vp in videos[:5]:
        for _, frame in utils.iter_frames(vp, step=20, limit=200):
            dets = utils.yolo_detections(model, frame, conf=0.20)
            if dets:
                utils.save_img(TEST_ID, IMG, frame)
                h, w = frame.shape[:2]
                print(f"[{TEST_ID}] Saved frame with {len(dets)} face(s): {w}×{h}")
                return True
    print(f"[{TEST_ID}] No frame with faces found.")
    return False


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
    saved = len(person['face_images']) >= 1
    submitted = proc._face_save_pool.submit.called
    passed = (saved and submitted)
    utils.report(TEST_ID, TITLE, passed,
                 actual=f"face_images count={len(person['face_images'])}, submit_called={submitted}",
                 expected="face_images has >= 1 entry, submit called")


if __name__ == '__main__':
    ok = extract()
    if ok:
        test()
