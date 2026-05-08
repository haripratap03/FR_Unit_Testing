#!/usr/bin/env python3
"""
FD-34: _extract_with_strategies — Small Face Crop (<200px) → strategy 'upscaled' or 'direct'
Target : _extract_with_strategies(rerun_fr.py)
Extract: Find a small face crop (max dim < 200px) from footage. Requires InsightFace.
Test   : Call _extract_with_strategies(app, small_crop).
Expect : returns (embedding, strategy) where strategy in ('direct', 'upscaled')
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import utils
import numpy as np

TEST_ID = "FD-34"
TITLE   = "_extract_with_strategies: small face (<200px) → strategy 'upscaled' or 'direct'"
IMG     = "face_small.jpg"


def extract():
    if utils.load_img(TEST_ID, IMG) is not None:
        print(f"[{TEST_ID}] Data already extracted.")
        return True
    model = utils.load_yolo_face()
    if model is None:
        return False
    print(f"[{TEST_ID}] Scanning for small face (max dim < 200px)...")
    dets = utils.scan_faces(model, max_faces=400, step=10, n_vids=5)
    if not dets:
        return False
    # Find small faces — prefer < 200px max dim
    small_dets = [d for d in dets if max(d['crop'].shape[:2]) < 200]
    if small_dets:
        chosen = utils.best_face(small_dets)
    else:
        chosen = utils.find_small_face(dets)
    crop = chosen['crop']
    h, w = crop.shape[:2]
    print(f"[{TEST_ID}] Small face: {w}×{h}, conf={chosen['conf']:.3f}")
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
        # Acceptable: very small crops may not yield embeddings at all
        utils.report(TEST_ID, TITLE, False,
                     actual=f"emb=None, strategy={strategy!r} (too small for InsightFace)",
                     expected="strategy in ('direct', 'upscaled')")
        return
    passed = strategy in ('direct', 'upscaled')
    utils.report(TEST_ID, TITLE, passed,
                 actual=f"strategy={strategy!r}, emb.shape={emb.shape}, crop={w}×{h}",
                 expected="strategy in ('direct', 'upscaled')")


if __name__ == '__main__':
    ok = extract()
    if ok:
        test()
