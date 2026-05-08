#!/usr/bin/env python3
"""
FD-39: _batch_capture_faces — Cooldown NOT Elapsed → face_model NOT Called
Target : ClipProcessor._batch_capture_faces(pipeline.py)
Extract: Reuse FD-38 person frame (or any frame — cooldown gate fires before model).
Test   : Set up proc with last_capture 0.5s ago (cooldown=1.5s).
         Call _batch_capture_faces. Expect: face_model NOT called, no face saved.
Expect : face_model.assert_not_called(), len(person['face_images']) == 0
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import utils
import tempfile
from datetime import datetime, timedelta
from unittest.mock import MagicMock

TEST_ID = "FD-39"
TITLE   = "_batch_capture_faces: cooldown not elapsed → face_model not called"
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


def make_person(last_capture_offset_s=None):
    folder = tempfile.mkdtemp()
    p = {'track_id': 1, 'images_folder': folder,
         'face_images': [], 'best_face_image': None}
    if last_capture_offset_s is not None:
        last_dt = _TS - timedelta(seconds=last_capture_offset_s)
        p['_last_face_capture'] = last_dt.isoformat()
    return p


def extract():
    if utils.load_img(TEST_ID, IMG) is not None:
        print(f"[{TEST_ID}] Data already extracted.")
        return True
    src = utils.load_img("FD-38", "person_frame.jpg")
    if src is not None:
        utils.save_img(TEST_ID, IMG, src)
        print(f"[{TEST_ID}] Reused FD-38 frame.")
        return True
    videos = utils.get_videos()
    if not videos:
        return False
    model = utils.load_yolo_face()
    if model is None:
        return False
    for vp in videos[:3]:
        for _, frame in utils.iter_frames(vp, step=30, limit=50):
            dets = utils.yolo_detections(model, frame, conf=0.20)
            if dets:
                utils.save_img(TEST_ID, IMG, frame)
                return True
    return False


def test():
    from pipeline import ClipProcessor
    frame = utils.load_img(TEST_ID, IMG)
    if frame is None:
        print(f"[{TEST_ID}] SKIP — no extracted data.")
        return
    # The face_model is a MagicMock — we verify it is NOT called
    face_model_mock = MagicMock()
    uid = 1
    person = make_person(last_capture_offset_s=0.5)  # 0.5s ago < 1.5s cooldown
    h, w = frame.shape[:2]
    proc = make_processor({uid: person}, face_model_mock)
    face_targets = [(uid, frame, (0, 0), [0, 0, w, h])]
    ClipProcessor._batch_capture_faces(proc, face_targets, _TS)
    model_not_called = not face_model_mock.called
    no_faces = (len(person['face_images']) == 0)
    no_submit = not proc._face_save_pool.submit.called
    passed = model_not_called and no_faces and no_submit
    utils.report(TEST_ID, TITLE, passed,
                 actual=f"model_called={face_model_mock.called}, face_images={len(person['face_images'])}, submit_called={proc._face_save_pool.submit.called}",
                 expected="face_model not called, face_images=0, submit not called")


if __name__ == '__main__':
    ok = extract()
    if ok:
        test()
