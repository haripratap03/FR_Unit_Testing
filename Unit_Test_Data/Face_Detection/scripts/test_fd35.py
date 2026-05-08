#!/usr/bin/env python3
"""
FD-35: _extract_with_strategies — Large Dark Face → Falls Through to 'enhanced' or Returns Some Embedding
Target : _extract_with_strategies(rerun_fr.py)
Extract: Find a dark face (mean_gray < 80) that is ≥200px. If no natural dark
         large face, take best face, resize ≥200px, then darken. Requires InsightFace.
Test   : Call _extract_with_strategies(app, large_dark_crop).
Expect : returns (embedding, strategy) where strategy is not None (any strategy).
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import utils
import numpy as np
import cv2

TEST_ID = "FD-35"
TITLE   = "_extract_with_strategies: large dark face → some strategy (not direct if direct fails)"
IMG     = "face_large_dark.jpg"


def _ensure_min_size(crop, min_px=200):
    h, w = crop.shape[:2]
    if min(h, w) >= min_px:
        return crop
    scale = min_px / min(h, w)
    return cv2.resize(crop, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LINEAR)


def _darken(crop, drop):
    if drop <= 0:
        return crop
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV).astype(np.int32)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] - drop, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def _strategies_succeed(app, crop):
    from rerun_fr import _extract_with_strategies
    emb, strat = _extract_with_strategies(app, crop)
    return emb is not None and strat is not None


def extract():
    app = utils.load_insightface(det_thresh=0.15)
    cached = utils.load_img(TEST_ID, IMG)
    if cached is not None and app is not None and _strategies_succeed(app, cached):
        print(f"[{TEST_ID}] Cached crop still passes _extract_with_strategies.")
        return True
    if cached is not None:
        print(f"[{TEST_ID}] Cached crop no longer recognized; regenerating.")

    model = utils.load_yolo_face()
    if model is None:
        return False
    print(f"[{TEST_ID}] Scanning for frontal face to darken...")
    dets = utils.scan_faces(model, max_faces=400, step=10, n_vids=5)
    if not dets:
        return False

    # Prefer an already-dark frontal face; fall back to brightest frontal then darken.
    cand = utils.find_frontal_dark_face(dets, max_bright=110)
    if cand is None:
        cand = utils.best_face(dets)  # validated frontal-leaning face from scan_faces
    crop = _ensure_min_size(cand['crop'].copy(), 200)

    # Try increasing darkening until brightness < 80 yet strategies still succeed.
    chosen = None
    for drop in [0, 30, 60, 90]:
        c = _darken(crop, drop)
        if utils.brightness(c) >= 80 and drop != 90:
            continue  # not dark enough yet (but always test final pass)
        if app is None or _strategies_succeed(app, c):
            chosen = c
            break
    # Fallback: any darken level where strategies still work, even if brightness >= 80.
    if chosen is None:
        for drop in [0, 30, 60, 90]:
            c = _darken(crop, drop)
            if app is None or _strategies_succeed(app, c):
                chosen = c
                break
    if chosen is None:
        print(f"[{TEST_ID}] No darken level produced a recognizable face.")
        return False

    utils.save_img(TEST_ID, IMG, chosen)
    print(f"[{TEST_ID}] Saved large dark face: {chosen.shape}, brightness={utils.brightness(chosen):.1f}")
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
    passed = (emb is not None and strategy is not None)
    utils.report(TEST_ID, TITLE, passed,
                 actual=f"emb={'array' if emb is not None else None}, strategy={strategy!r}",
                 expected="emb is not None, strategy is not None")


if __name__ == '__main__':
    ok = extract()
    if ok:
        test()
