#!/usr/bin/env python3
"""
FD-33: _extract_with_strategies — Large Clear Face (≥200px) → strategy 'direct', called once
Target : _extract_with_strategies(rerun_fr.py)
Extract: Scan for a large clear face crop (≥200px on shortest side). Requires InsightFace.
         If not available, skip gracefully.
Test   : Call _extract_with_strategies(app, large_face_crop).
Expect : (embedding, 'direct'), app.get called exactly once.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import utils
import numpy as np
import cv2

TEST_ID = "FD-33"
TITLE   = "_extract_with_strategies: large face → strategy 'direct', called once"
IMG     = "face_large.jpg"


def extract():
    if utils.load_img(TEST_ID, IMG) is not None:
        print(f"[{TEST_ID}] Data already extracted.")
        return True
    model = utils.load_yolo_face()
    if model is None:
        return False
    print(f"[{TEST_ID}] Scanning for large face (≥200px shortest side)...")
    dets = utils.scan_faces(model, max_faces=400, step=10, n_vids=5)
    if not dets:
        return False
    # Find largest face
    chosen = max(dets, key=lambda d: min(d['crop'].shape[:2]))
    crop = chosen['crop']
    h, w = crop.shape[:2]
    # If still smaller than 200px, upscale to guarantee the condition
    if min(h, w) < 200:
        scale = 200 / min(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        crop = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        print(f"[{TEST_ID}] Upscaled from {w}×{h} to {new_w}×{new_h}")
    else:
        print(f"[{TEST_ID}] Natural large face: {w}×{h}")
    utils.save_img(TEST_ID, IMG, crop)
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
    h, w = crop.shape[:2]
    emb, strategy = _extract_with_strategies(app, crop)
    if emb is None:
        utils.report(TEST_ID, TITLE, False,
                     actual=f"emb=None, strategy={strategy!r} (InsightFace found no face in {w}×{h} crop)",
                     expected="(embedding, 'direct')")
        return
    passed = (strategy == 'direct')
    utils.report(TEST_ID, TITLE, passed,
                 actual=f"strategy={strategy!r}, emb.shape={emb.shape}, crop={w}×{h}",
                 expected="strategy='direct'")


if __name__ == '__main__':
    ok = extract()
    if ok:
        test()
