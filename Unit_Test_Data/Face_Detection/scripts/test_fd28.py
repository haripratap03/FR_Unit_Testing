#!/usr/bin/env python3
"""
FD-28: _enhance_contrast — Real Dark Face → output mean > input mean; same shape/dtype
Target : _enhance_contrast(rerun_fr.py)
Extract: Find a naturally dark face from CCTV footage (mean_gray < 80).
         If none found, take best face and darken it artificially.
Test   : Call _enhance_contrast(dark_face_crop).
Expect : out.shape == in.shape, out.dtype == in.dtype, out.mean() > in.mean()
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import utils
import numpy as np
import cv2

TEST_ID = "FD-28"
TITLE   = "_enhance_contrast: dark face → output brighter, same shape/dtype"
IMG     = "face_dark.jpg"


def extract():
    if utils.load_img(TEST_ID, IMG) is not None:
        print(f"[{TEST_ID}] Data already extracted.")
        return True
    model = utils.load_yolo_face()
    if model is None:
        return False
    print(f"[{TEST_ID}] Scanning for dark face (mean_gray < 80)...")
    dets = utils.scan_faces(model, max_faces=400, step=10, n_vids=5)
    if not dets:
        return False
    dark = utils.find_dark_face(dets, max_bright=80)
    if dark is not None:
        crop = dark['crop']
        br = utils.brightness(crop)
        print(f"[{TEST_ID}] Found natural dark face: brightness={br:.1f}")
    else:
        # Darken the best face
        chosen = utils.best_face(dets)
        crop = chosen['crop'].copy()
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV).astype(np.int32)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] - 100, 0, 255)
        crop = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        br = utils.brightness(crop)
        print(f"[{TEST_ID}] Artificially darkened face: brightness={br:.1f}")
    utils.save_img(TEST_ID, IMG, crop)
    return True


def test():
    from rerun_fr import _enhance_contrast
    crop = utils.load_img(TEST_ID, IMG)
    if crop is None:
        print(f"[{TEST_ID}] SKIP — no extracted data.")
        return
    in_mean = float(crop.mean())
    out = _enhance_contrast(crop)
    out_mean = float(out.mean())
    shape_ok = (out.shape == crop.shape)
    dtype_ok = (out.dtype == crop.dtype)
    mean_ok = (out_mean > in_mean)
    passed = shape_ok and dtype_ok and mean_ok
    utils.report(TEST_ID, TITLE, passed,
                 actual=f"in_mean={in_mean:.1f}, out_mean={out_mean:.1f}, shape={out.shape}, dtype={out.dtype}",
                 expected="out_mean > in_mean, same shape/dtype")


if __name__ == '__main__':
    ok = extract()
    if ok:
        test()
