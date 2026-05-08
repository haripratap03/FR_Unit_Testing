#!/usr/bin/env python3
"""
FD-27: _count_kps_in_bbox — 3 of 5 Kps Inside Bbox → returns 3
Target : _count_kps_in_bbox(pipeline.py:660)
Extract: No real footage needed — uses synthetic mixed-position kps.
         bbox=(10,10,90,90). First 3 kps inside, last 2 far outside.
Test   : Call _count_kps_in_bbox(mixed_kps, (10,10,90,90)). Expect 3.
Expect : 3
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import utils

TEST_ID = "FD-27"
TITLE   = "_count_kps_in_bbox: 3 of 5 inside → 3"
KPS_F   = "kps_partial.json"
META_F  = "meta.json"


def extract():
    if utils.load_json_file(TEST_ID, KPS_F) is not None:
        print(f"[{TEST_ID}] Data already extracted.")
        return True
    # bbox=(10,10,90,90): margin = int(80 * 0.15) = 12
    # Expanded: (10-12, 10-12, 90+12, 90+12) = (-2, -2, 102, 102)
    # Inside (considering expanded): [50,50], [30,30], [70,30] → inside
    # Outside: [200,300], [300,400] → definitely outside
    kps = [
        [50, 50, 0.9],    # inside
        [30, 30, 0.9],    # inside
        [70, 30, 0.8],    # inside
        [200, 300, 0.7],  # outside
        [300, 400, 0.6],  # outside
    ]
    bbox = [10, 10, 90, 90]
    utils.save_json(TEST_ID, KPS_F, kps)
    utils.save_json(TEST_ID, META_F, {'bbox': bbox})
    print(f"[{TEST_ID}] Saved mixed kps: 3 inside, 2 far outside. bbox={bbox}")
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
    passed = (count == 3)
    utils.report(TEST_ID, TITLE, passed,
                 actual=f"count={count}",
                 expected="count == 3")


if __name__ == '__main__':
    ok = extract()
    if ok:
        test()
