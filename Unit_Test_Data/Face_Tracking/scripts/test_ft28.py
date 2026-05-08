"""
test_ft28.py  —  FT-28
Test:  PersonTracker.match_or_create  (pipeline.py:425)
Check: an embedding far enough from any active track is not matched, and a
       new pending track is created instead.

Output: Unit_Test_Data/Face_Tracking/data/test_ft28/
"""

from datetime import datetime
from unittest.mock import patch

import numpy as np

from _ft_helpers import (
    load_osnet, make_tracker, make_active_track, make_emb_at_dist, random_unit_vec,
    prepare_data_dir, save_single_test_result, print_verdict,
)

TEST_ID = "FT-28"
DATA_OUT = prepare_data_dir(TEST_ID)


def run():
    import pipeline
    REID = pipeline.REID_DISTANCE
    print(f"\n{'='*60}\n{TEST_ID} — match_or_create distant embedding "
          f"(REID={REID})  ({datetime.now():%H:%M:%S})\n{'='*60}")

    ref_emb = random_unit_vec(seed=42)
    far_emb = make_emb_at_dist(ref_emb, REID + 0.1, seed=100)

    tracker = make_tracker(osnet=load_osnet(), frame_wh=(2688, 1520))
    tracker.active = {99: make_active_track(ref_emb, [400, 200, 600, 600])}
    tracker.next_id = 100

    frame = np.zeros((1520, 2688, 3), dtype=np.uint8)
    box   = [410, 210, 590, 580]
    with patch.object(tracker.osnet, "extract_batch", return_value=[far_emb]):
        result = tracker.match_or_create(frame, [box], datetime.now(), gap_seconds=0.05)

    ok = (len(result) == 0) and (len(tracker.pending) == 1)
    status = "PASS" if ok else "FAIL"
    cos_dist = float(1.0 - np.dot(far_emb, ref_emb))
    print(f"  cos_dist={cos_dist:.4f}  matched={len(result)}  "
          f"new_pending={len(tracker.pending)}  | {status}")
    print_verdict(TEST_ID, status)

    save_single_test_result(TEST_ID, DATA_OUT, {
        "status": status,
        "cos_dist": round(cos_dist, 4),
        "matched": len(result),
        "new_pending": len(tracker.pending),
    })


if __name__ == "__main__":
    run()
