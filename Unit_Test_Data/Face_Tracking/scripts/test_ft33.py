"""
test_ft33.py  —  FT-33
Test:  PersonTracker.match_or_create  (pipeline.py:425)
Check: greedy assignment correctly cross-pairs detections with the right tracks
       even when each detection is closest to a different track than its
       neighbour box-position would suggest.

Output: Unit_Test_Data/Face_Tracking/data/test_ft33/
"""

from datetime import datetime
from unittest.mock import patch

import numpy as np

from _ft_helpers import (
    load_osnet, make_tracker, make_active_track, make_emb_at_dist, random_unit_vec,
    prepare_data_dir, save_single_test_result, print_verdict,
)

TEST_ID = "FT-33"
DATA_OUT = prepare_data_dir(TEST_ID)


def run():
    print(f"\n{'='*60}\n{TEST_ID} — match_or_create greedy assignment  "
          f"({datetime.now():%H:%M:%S})\n{'='*60}")

    e_a = random_unit_vec(seed=10)
    e_b = random_unit_vec(seed=11)
    det_1 = make_emb_at_dist(e_b, 0.05, seed=12)   # close to B
    det_2 = make_emb_at_dist(e_a, 0.05, seed=13)   # close to A

    tracker = make_tracker(osnet=load_osnet(), frame_wh=(2688, 1520))
    tracker.active = {
        1: make_active_track(e_a, [100, 100, 200, 300]),
        2: make_active_track(e_b, [400, 100, 500, 300]),
    }
    tracker.next_id = 3

    frame = np.zeros((1520, 2688, 3), dtype=np.uint8)
    boxes = [[105, 105, 200, 295], [405, 105, 495, 295]]
    with patch.object(tracker.osnet, "extract_batch", return_value=[det_1, det_2]):
        result = tracker.match_or_create(frame, boxes, datetime.now(), gap_seconds=0.05)

    matched_ids = {r[0] for r in result}
    ok = matched_ids == {1, 2}
    status = "PASS" if ok else "FAIL"
    print(f"  matched_ids={matched_ids}  expected={{1,2}}  | {status}")
    print_verdict(TEST_ID, status)

    save_single_test_result(TEST_ID, DATA_OUT, {
        "status": status,
        "matched_ids": sorted(matched_ids),
    })


if __name__ == "__main__":
    run()
