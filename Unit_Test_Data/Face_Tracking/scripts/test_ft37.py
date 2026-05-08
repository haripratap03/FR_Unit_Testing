"""
test_ft37.py  —  FT-37
Test:  PersonTracker.match_or_create  (pipeline.py:425)
Check: an unmatched detection on an empty tracker creates a new PendingTrack
       with frames_seen == 1.

Output: Unit_Test_Data/Face_Tracking/data/test_ft37/
"""

from datetime import datetime
from unittest.mock import patch

import numpy as np

from _ft_helpers import (
    load_osnet, make_tracker, random_unit_vec,
    prepare_data_dir, save_single_test_result, print_verdict,
)

TEST_ID = "FT-37"
DATA_OUT = prepare_data_dir(TEST_ID)


def run():
    print(f"\n{'='*60}\n{TEST_ID} — match_or_create new PendingTrack  "
          f"({datetime.now():%H:%M:%S})\n{'='*60}")

    tracker = make_tracker(osnet=load_osnet(), frame_wh=(2688, 1520))
    e = random_unit_vec(seed=20)
    frame = np.zeros((1520, 2688, 3), dtype=np.uint8)

    with patch.object(tracker.osnet, "extract_batch", return_value=[e]):
        result = tracker.match_or_create(frame, [[100, 100, 200, 300]], datetime.now(), gap_seconds=0.05)

    pending_count = len(tracker.pending)
    frames_seen = list(tracker.pending.values())[0].frames_seen if tracker.pending else None
    ok = (result == []) and (pending_count == 1) and (frames_seen == 1)
    status = "PASS" if ok else "FAIL"
    print(f"  result=[]={result==[]}  pending={pending_count}  "
          f"frames_seen={frames_seen}  | {status}")
    print_verdict(TEST_ID, status)

    save_single_test_result(TEST_ID, DATA_OUT, {
        "status": status,
        "new_pending": pending_count,
        "frames_seen": frames_seen,
    })


if __name__ == "__main__":
    run()
