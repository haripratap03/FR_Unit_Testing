"""
test_ft39.py  —  FT-39
Test:  PersonTracker.match_or_create  (pipeline.py:425)
Check: an empty boxes list returns [] and leaves the active pool unchanged.

Output: Unit_Test_Data/Face_Tracking/data/test_ft39/
"""

from datetime import datetime

import numpy as np

from _ft_helpers import (
    load_osnet, make_tracker, make_active_track, random_unit_vec,
    prepare_data_dir, save_single_test_result, print_verdict,
)

TEST_ID = "FT-39"
DATA_OUT = prepare_data_dir(TEST_ID)


def run():
    print(f"\n{'='*60}\n{TEST_ID} — match_or_create empty boxes list  "
          f"({datetime.now():%H:%M:%S})\n{'='*60}")

    e = random_unit_vec(seed=22)
    tracker = make_tracker(osnet=load_osnet(), frame_wh=(2688, 1520))
    tracker.active = {1: make_active_track(e), 2: make_active_track(e)}
    tracker.next_id = 3

    frame = np.zeros((1520, 2688, 3), dtype=np.uint8)
    result = tracker.match_or_create(frame, [], datetime.now(), gap_seconds=0.05)

    ok = (result == []) and (set(tracker.active.keys()) == {1, 2})
    status = "PASS" if ok else "FAIL"
    print(f"  result=[]={result==[]}  active_ids={sorted(tracker.active.keys())}  | {status}")
    print_verdict(TEST_ID, status)

    save_single_test_result(TEST_ID, DATA_OUT, {
        "status": status,
        "result_empty": result == [],
        "active_unchanged": sorted(tracker.active.keys()) == [1, 2],
    })


if __name__ == "__main__":
    run()
