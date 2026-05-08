#!/usr/bin/env python3
"""
FD-43: _batch_capture_faces — At MAX_FACES Capacity, Better Quality → Evicts Lowest, Saves New
Target : ClipProcessor._batch_capture_faces(pipeline.py)
Extract: Reuse FD-38 person frame. Pre-fill face_images with MAX_FACES entries:
         one at score=30 (weakest), rest at 50. Patch score_face to return 65.
Test   : New face (65) evicts the weakest (30). Saved. submit called >= 2 times.
Expect : score 30 gone, score 65 present, len == MAX_FACES, submit call count >= 2
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import utils
import tempfile
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

TEST_ID = "FD-43"
TITLE   = "_batch_capture_faces: at capacity, better quality → eviction + saved"
IMG     = "person_frame.jpg"

_TS = datetime(2026, 4, 29, 9, 0, 0)


def make_processor(tracked_persons, face_model):
    proc = MagicMock(spec=[])
    proc.tracked_persons = tracked_persons
    proc.face_model = face_model
    proc._face_save_pool = MagicMock()
    proc.frame_count = 5555
    proc.stats = {'faces_captured': 0}
    return proc


def make_person_at_capacity(max_faces):
    folder = tempfile.mkdtemp()
    existing = [(30.0, f"{folder}/old_q30.jpg")] + \
               [(50.0, f"{folder}/other_{i}.jpg") for i in range(max_faces - 1)]
    p = {'track_id': 1, 'images_folder': folder,
         'face_images': list(existing), 'best_face_image': None}
    last_dt = _TS - timedelta(seconds=5.0)
    p['_last_face_capture'] = last_dt.isoformat()
    return p


def extract():
    if utils.load_img(TEST_ID, IMG) is not None:
        print(f"[{TEST_ID}] Data already extracted.")
        return True
    for src_id in ["FD-38", "FD-42"]:
        src = utils.load_img(src_id, "person_frame.jpg")
        if src is not None:
            utils.save_img(TEST_ID, IMG, src)
            print(f"[{TEST_ID}] Reused {src_id} frame.")
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
    with patch('pipeline.score_face', return_value=(65.0, {})):
        ClipProcessor._batch_capture_faces(proc, face_targets, _TS)
    scores = [s for s, _ in person['face_images']]
    evicted = (30.0 not in scores)
    added = (65.0 in scores)
    count_ok = (len(person['face_images']) == MAX_FACES)
    submit_ok = (proc._face_save_pool.submit.call_count >= 2)
    passed = evicted and added and count_ok and submit_ok
    utils.report(TEST_ID, TITLE, passed,
                 actual=f"scores={sorted(scores)}, count={len(scores)}, submit_count={proc._face_save_pool.submit.call_count}",
                 expected=f"30.0 evicted, 65.0 added, count={MAX_FACES}, submit >= 2")


if __name__ == '__main__':
    ok = extract()
    if ok:
        test()
