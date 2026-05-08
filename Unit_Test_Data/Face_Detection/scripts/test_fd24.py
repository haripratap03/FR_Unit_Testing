#!/usr/bin/env python3
"""
FD-24: _count_kps_in_bbox — All 5 YOLO Kps Inside Bbox → returns 5
Target : _count_kps_in_bbox(pipeline.py:660)
Extract: Scan for a good frontal face detection. Save the YOLO kps + bbox.
         Verify all 5 kps are inside the bbox (with margin). If the real kps
         don't all fall inside, also test with synthetic values.
Test   : Call _count_kps_in_bbox(kps, bbox). Expect 5.
Expect : 5
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import utils
import numpy as np

TEST_ID = "FD-24"
TITLE   = "_count_kps_in_bbox: all 5 kps inside bbox → 5"
KPS_F   = "kps.json"
META_F  = "meta.json"


def extract():
    if utils.load_json_file(TEST_ID, KPS_F) is not None:
        print(f"[{TEST_ID}] Data already extracted.")
        return True
    # Reuse FD-05 kps if available
    kps_raw = utils.load_json_file("FD-05", "keypoints.json")
    meta = utils.load_json_file("FD-05", "meta.json")
    if kps_raw is not None and meta is not None:
        kps = np.array(kps_raw, dtype=np.float32)
        bbox = tuple(meta['bbox'])
        utils.save_json(TEST_ID, KPS_F, kps)
        utils.save_json(TEST_ID, META_F, {'bbox': list(bbox)})
        print(f"[{TEST_ID}] Reused FD-05 kps. bbox={bbox}")
        return True
    # Scan for a frontal face
    model = utils.load_yolo_face()
    if model is None:
        return False
    print(f"[{TEST_ID}] Scanning for frontal face with all kps inside bbox...")
    dets = utils.scan_faces(model, max_faces=400, step=10, n_vids=4)
    if not dets:
        return False
    chosen = utils.best_face(dets)
    kps = chosen.get('kps')
    bbox = chosen['bbox']
    if kps is None:
        # Use synthetic kps that are definitely inside the bbox
        x1, y1, x2, y2 = bbox
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        kps = np.array([[cx, cy, 0.9], [cx-10, cy-10, 0.9], [cx+10, cy-10, 0.9],
                        [cx-10, cy+10, 0.9], [cx+10, cy+10, 0.9]], dtype=np.float32)
    utils.save_json(TEST_ID, KPS_F, kps)
    utils.save_json(TEST_ID, META_F, {'bbox': list(bbox)})
    print(f"[{TEST_ID}] Saved kps + bbox={bbox}")
    return True


def test():
    from pipeline import _count_kps_in_bbox
    kps_raw = utils.load_json_file(TEST_ID, KPS_F)
    meta = utils.load_json_file(TEST_ID, META_F)
    if kps_raw is None or meta is None:
        print(f"[{TEST_ID}] SKIP — no extracted data.")
        return
    kps = kps_raw  # list of [x,y,conf]
    bbox = tuple(meta['bbox'])
    count = _count_kps_in_bbox(kps, bbox)
    # kps from real detection inside their own bbox should all be inside
    passed = (count == 5)
    utils.report(TEST_ID, TITLE, passed,
                 actual=f"count={count}",
                 expected="count == 5")


if __name__ == '__main__':
    ok = extract()
    if ok:
        test()
