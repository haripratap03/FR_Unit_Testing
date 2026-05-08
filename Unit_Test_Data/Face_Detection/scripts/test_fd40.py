#!/usr/bin/env python3
"""
FD-40: _batch_capture_faces — Tiny Person Crop (< 20×20) → No Face Saved
Target : ClipProcessor._batch_capture_faces(pipeline.py)
Extract: A tiny 15×15 crop (the pipeline rejects person crops smaller than
         min_crop_w=32 or min_crop_h=64, and YOLO face detection also fails
         on very tiny input). Create synthetically.
Test   : Pass a 15×15 crop as the person region. Expect no face saved.
Expect : submit not called, face_images empty
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import utils
import tempfile
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import MagicMock

TEST_ID = "FD-40"
TITLE   = "_batch_capture_faces: tiny 15×15 crop → no face saved"
IMG     = "tiny_crop.jpg"

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
    p = utils.data_path(TEST_ID, IMG)
    if os.path.exists(p):
        print(f"[{TEST_ID}] Data already extracted.")
        return True
    # A tiny synthetic crop — face model will find nothing in 15×15
    tiny = np.random.randint(80, 180, (15, 15, 3), dtype=np.uint8)
    utils.save_img(TEST_ID, IMG, tiny)
    print(f"[{TEST_ID}] Saved synthetic tiny crop: 15×15")
    return True


def test():
    from pipeline import ClipProcessor
    crop = utils.load_img(TEST_ID, IMG)
    if crop is None:
        print(f"[{TEST_ID}] SKIP — no extracted data.")
        return
    face_model = utils.load_yolo_face()
    if face_model is None:
        print(f"[{TEST_ID}] SKIP — YOLO face model not available.")
        return
    uid = 1
    person = make_person(last_capture_offset_s=5.0)
    h, w = crop.shape[:2]
    proc = make_processor({uid: person}, face_model)
    face_targets = [(uid, crop, (0, 0), [0, 0, w, h])]
    ClipProcessor._batch_capture_faces(proc, face_targets, _TS)
    passed = (len(person['face_images']) == 0 and not proc._face_save_pool.submit.called)
    utils.report(TEST_ID, TITLE, passed,
                 actual=f"face_images={len(person['face_images'])}, submit_called={proc._face_save_pool.submit.called}",
                 expected="face_images=0, submit not called")


if __name__ == '__main__':
    ok = extract()
    if ok:
        test()
