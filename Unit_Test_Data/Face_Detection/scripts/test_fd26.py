#!/usr/bin/env python3
"""
FD-26: _count_kps_in_bbox — Kp Exactly on Margin Boundary → counted as inside
Target : _count_kps_in_bbox(pipeline.py:660)
Extract: No real footage needed — uses synthetic boundary values.
         bbox=(20,20,80,80); pipeline margin = 15% of (x2-x1) and (y2-y1).
         margin_x = 0.15 * 60 = 9.0 (rounds to int 9 or uses float).
         Expanded bbox: x1_exp = 20 - 9 = 11, so a kp at x=11 is on the boundary.
Test   : Call _count_kps_in_bbox([[11, 50, 0.9], ...], (20, 20, 80, 80)).
         All 5 kps are at or just inside the expanded boundary.
Expect : 5
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import utils

TEST_ID = "FD-26"
TITLE   = "_count_kps_in_bbox: kp exactly on margin boundary → counted as inside"
KPS_F   = "kps_boundary.json"
META_F  = "meta.json"


def extract():
    if utils.load_json_file(TEST_ID, KPS_F) is not None:
        print(f"[{TEST_ID}] Data already extracted.")
        return True
    # bbox=(20,20,80,80): width=60, height=60
    # pipeline uses margin = int(w * 0.15) = int(9.0) = 9
    # Expanded x1 = 20 - 9 = 11, expanded x2 = 80 + 9 = 89
    # Expanded y1 = 20 - 9 = 11, expanded y2 = 80 + 9 = 89
    # Place kps exactly at/near the boundary edges:
    kps = [
        [11, 50, 0.9],   # exactly at left x boundary
        [89, 50, 0.9],   # exactly at right x boundary
        [50, 11, 0.9],   # exactly at top y boundary
        [50, 89, 0.9],   # exactly at bottom y boundary
        [50, 50, 0.9],   # center — safely inside
    ]
    bbox = [20, 20, 80, 80]
    utils.save_json(TEST_ID, KPS_F, kps)
    utils.save_json(TEST_ID, META_F, {'bbox': bbox})
    print(f"[{TEST_ID}] Saved boundary-edge kps: {kps}")
    print(f"[{TEST_ID}] bbox={bbox} → margin≈9 → expanded (11,11,89,89)")
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
    passed = (count == 5)
    utils.report(TEST_ID, TITLE, passed,
                 actual=f"count={count}",
                 expected="count == 5 (boundary kps counted as inside)")


if __name__ == '__main__':
    ok = extract()
    if ok:
        test()
