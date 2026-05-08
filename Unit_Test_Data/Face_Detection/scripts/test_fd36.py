#!/usr/bin/env python3
"""
FD-36: _extract_with_strategies — Any Real Face → (embedding, strategy) where strategy is not None
Target : _extract_with_strategies(rerun_fr.py)
Extract: Best face crop (reuse FD-05 or FD-01). Requires InsightFace.
Test   : Call _extract_with_strategies(app, real_face_crop).
Expect : returns (embedding, strategy) where both are not None; strategy in
         ('direct', 'upscaled', 'enhanced', 'padded')
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import utils
import numpy as np

TEST_ID = "FD-36"
TITLE   = "_extract_with_strategies: real face → (embedding, strategy) both not None"
IMG     = "face_crop.jpg"
VALID_STRATEGIES = ('direct', 'upscaled', 'enhanced', 'padded')


def extract():
    if utils.load_img(TEST_ID, IMG) is not None:
        print(f"[{TEST_ID}] Data already extracted.")
        return True
    for src_id, src_file in [("FD-05", "face_best.jpg"), ("FD-01", "face_crop.jpg")]:
        src = utils.load_img(src_id, src_file)
        if src is not None:
            utils.save_img(TEST_ID, IMG, src)
            print(f"[{TEST_ID}] Reused {src_id} crop.")
            return True
    model = utils.load_yolo_face()
    if model is None:
        return False
    dets = utils.scan_faces(model, max_faces=200, step=15, n_vids=3)
    if not dets:
        return False
    utils.save_img(TEST_ID, IMG, utils.best_face(dets)['crop'])
    return True


def test():
    from rerun_fr import _extract_with_strategies
    crop = utils.load_img(TEST_ID, IMG)
    if crop is None:
        print(f"[{TEST_ID}] SKIP — no extracted data.")
        return
    app = utils.load_insightface(det_thresh=0.15)
    if app is None:
        print(f"[{TEST_ID}] SKIP — InsightFace not available.")
        return
    emb, strategy = _extract_with_strategies(app, crop)
    if emb is None:
        utils.report(TEST_ID, TITLE, False,
                     actual="emb=None (InsightFace found no face in crop)",
                     expected=f"(embedding, strategy) where strategy in {VALID_STRATEGIES}")
        return
    passed = (emb is not None and strategy in VALID_STRATEGIES)
    utils.report(TEST_ID, TITLE, passed,
                 actual=f"emb.shape={emb.shape}, strategy={strategy!r}",
                 expected=f"strategy in {VALID_STRATEGIES}")


if __name__ == '__main__':
    ok = extract()
    if ok:
        test()
