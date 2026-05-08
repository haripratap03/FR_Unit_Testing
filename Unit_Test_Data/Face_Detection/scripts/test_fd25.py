#!/usr/bin/env python3
"""
FD-25: _count_kps_in_bbox — Same Kps Shifted Far Outside Bbox → returns 0
Target : _count_kps_in_bbox(pipeline.py:660)
Extract: Reuse FD-24 kps. Shift all kps to be far outside the bbox (+500, +500).
Test   : Call _count_kps_in_bbox(shifted_kps, bbox). Expect 0.
Expect : 0
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import utils
import numpy as np

TEST_ID = "FD-25"
TITLE   = "_count_kps_in_bbox: all kps far outside bbox → 0"
KPS_F   = "kps_outside.json"
META_F  = "meta.json"


def extract():
    if utils.load_json_file(TEST_ID, KPS_F) is not None:
        print(f"[{TEST_ID}] Data already extracted.")
        return True
    # Reuse FD-24 kps and bbox, then shift kps outside
    kps_raw = utils.load_json_file("FD-24", "kps.json")
    meta = utils.load_json_file("FD-24", "meta.json")
    if kps_raw is None or meta is None:
        # Fallback: hardcoded outside kps
        kps = [[200, 200], [220, 200], [230, 200], [200, 250], [230, 250]]
        bbox = [10, 10, 90, 90]
    else:
        kps = np.array(kps_raw, dtype=np.float32)
        kps[:, 0] += 500
        kps[:, 1] += 500
        kps = kps.tolist()
        bbox = meta['bbox']
    utils.save_json(TEST_ID, KPS_F, kps)
    utils.save_json(TEST_ID, META_F, {'bbox': bbox})
    print(f"[{TEST_ID}] Saved shifted-outside kps. bbox={bbox}")
    return True


def test():
    from pipeline import _count_kps_in_bbox
    kps_raw = utils.load_json_file(TEST_ID, KPS_F)
    meta = utils.load_json_file(TEST_ID, META_F)
    if kps_raw is None or meta is None:
        print(f"[{TEST_ID}] SKIP — no extracted data.")
        return
    bbox = tuple(meta['bbox'])
    count = _count_kps_in_bbox(kps_raw, bbox)
    passed = (count == 0)
    utils.report(TEST_ID, TITLE, passed,
                 actual=f"count={count}",
                 expected="count == 0")


if __name__ == '__main__':
    ok = extract()
    if ok:
        test()
