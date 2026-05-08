#!/usr/bin/env python3
"""
FD-42: _batch_capture_faces — At MAX_FACES Capacity, Worse Quality → Not Saved
Target : ClipProcessor._batch_capture_faces(pipeline.py)
Extract: Reuse FD-38 person frame. Pre-fill face_images with MAX_FACES entries
         all at score=50. Patch score_face to return 45 (below min).
Test   : Call _batch_capture_faces. New face (score=45) should NOT evict/save.
Expect : len(face_images) == MAX_FACES, submit not called
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import utils
import tempfile
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

TEST_ID = "FD-42"
TITLE   = "_batch_capture_faces: at capacity, worse quality → not saved"
IMG     = "person_frame.jpg"

_TS = datetime(2026, 4, 29, 9, 0, 0)


def make_processor(tracked_persons, face_model):
    proc = MagicMock(spec=[])
    proc.tracked_persons = tracked_persons
    proc.face_model = face_model
    proc._face_save_pool = MagicMock()
    proc.frame_count = 9999
    proc.stats = {'faces_captured': 0}
    return proc


def make_person_at_capacity(max_faces):
    folder = tempfile.mkdtemp()
    existing = [(50.0, f"{folder}/old_q50_{i}.jpg") for i in range(max_faces)]
    p = {'track_id': 1, 'images_folder': folder,
         'face_images': list(existing), 'best_face_image': None}
    last_dt = _TS - timedelta(seconds=5.0)
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
        for _, frame in utils.iter_frames(vp, step=20, limit=100):
            if utils.yolo_detections(model, frame):
                utils.save_img(TEST_ID, IMG, frame)
                return True
    return False


def test():
    from pipeline import ClipProcessor, MAX_FACES
    frame = utils.load_img(TEST_ID, IMG)
    if frame is None:
        print(f"[{TEST_ID}] SKIP — no extracted data.")
        return
    face_model = utils.load_yolo_face()
    if face_model is None:
        print(f"[{TEST_ID}] SKIP — YOLO face model not available.")
        return
    uid = 1
    person = make_person_at_capacity(MAX_FACES)
    h, w = frame.shape[:2]
    proc = make_processor({uid: person}, face_model)
    face_targets = [(uid, frame, (0, 0), [0, 0, w, h])]
    with patch('pipeline.score_face', return_value=(45.0, {})):
        ClipProcessor._batch_capture_faces(proc, face_targets, _TS)
    count = len(person['face_images'])
    submit_called = proc._face_save_pool.submit.called
    passed = (count == MAX_FACES and not submit_called)
    utils.report(TEST_ID, TITLE, passed,
                 actual=f"face_images={count}/{MAX_FACES}, submit_called={submit_called}",
                 expected=f"face_images={MAX_FACES}, submit not called")


if __name__ == '__main__':
    ok = extract()
    if ok:
        test()
