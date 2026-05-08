"""
test_ft38.py  —  FT-38
Test:  PersonTracker.match_or_create  (pipeline.py:425)
Check: when extract_batch returns None for an undersized crop, that detection
       is silently skipped (no crash); valid detections still produce a pending track.

Output: Unit_Test_Data/Face_Tracking/data/test_ft38/
"""

from datetime import datetime
from unittest.mock import patch

import numpy as np

from _ft_helpers import (
    load_osnet, make_tracker, random_unit_vec,
    prepare_data_dir, save_single_test_result, print_verdict,
)

TEST_ID = "FT-38"
DATA_OUT = prepare_data_dir(TEST_ID)


def run():
    print(f"\n{'='*60}\n{TEST_ID} — match_or_create None embedding skip  "
          f"({datetime.now():%H:%M:%S})\n{'='*60}")

    tracker = make_tracker(osnet=load_osnet(), frame_wh=(2688, 1520))
    e_valid = random_unit_vec(seed=21)
    frame = np.zeros((1520, 2688, 3), dtype=np.uint8)

    with patch.object(tracker.osnet, "extract_batch", return_value=[None, e_valid]):
        tracker.match_or_create(
            frame, [[0, 0, 20, 40], [100, 100, 300, 500]],
            datetime.now(), gap_seconds=0.05,
        )

    none_skipped = all(p.bank[0] is not None for p in tracker.pending.values())
    ok = (len(tracker.pending) == 1) and none_skipped
    status = "PASS" if ok else "FAIL"
    print(f"  pending={len(tracker.pending)}  none_skipped={none_skipped}  | {status}")
    print_verdict(TEST_ID, status)

    save_single_test_result(TEST_ID, DATA_OUT, {
        "status": status,
        "pending_count": len(tracker.pending),
        "none_skipped": none_skipped,
    })


if __name__ == "__main__":
    run()
