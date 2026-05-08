"""
test_ft29.py  —  FT-29
Test:  PersonTracker.match_or_create  (pipeline.py:425)
Check: a close ReID match is blocked by the spatial gate when the candidate
       box is at the opposite corner of the frame.

Output: Unit_Test_Data/Face_Tracking/data/test_ft29/
"""

from datetime import datetime
from unittest.mock import patch

import numpy as np

from _ft_helpers import (
    load_osnet, make_tracker, make_active_track, make_emb_at_dist, random_unit_vec,
    prepare_data_dir, save_single_test_result, print_verdict,
)

TEST_ID = "FT-29"
DATA_OUT = prepare_data_dir(TEST_ID)


def run():
    import pipeline
    print(f"\n{'='*60}\n{TEST_ID} — match_or_create spatial gate  "
          f"(MAX_BOX_JUMP={pipeline.MAX_BOX_JUMP})  "
          f"({datetime.now():%H:%M:%S})\n{'='*60}")

    ref_emb = random_unit_vec(seed=1)
    close_emb = make_emb_at_dist(ref_emb, 0.05, seed=2)

    tracker = make_tracker(osnet=load_osnet(), frame_wh=(2688, 1520))
    tracker.active = {1: make_active_track(ref_emb, [10, 10, 100, 100])}
    tracker.next_id = 2

    frame = np.zeros((1520, 2688, 3), dtype=np.uint8)
    far_box = [2500, 1400, 2688, 1520]
    with patch.object(tracker.osnet, "extract_batch", return_value=[close_emb]):
        result = tracker.match_or_create(frame, [far_box], datetime.now(), gap_seconds=0.05)

    ok = len(result) == 0
    status = "PASS" if ok else "FAIL"
    cos_dist = float(1.0 - np.dot(close_emb, ref_emb))
    box_dist = tracker._box_distance(far_box, [10, 10, 100, 100])
    print(f"  cos_dist={cos_dist:.4f}  box_dist={box_dist:.4f}  "
          f"matched={len(result)}  | {status}")
    print_verdict(TEST_ID, status)

    save_single_test_result(TEST_ID, DATA_OUT, {
        "status": status,
        "cos_dist": round(cos_dist, 4),
        "box_dist": round(box_dist, 4),
        "matched": len(result),
    })


if __name__ == "__main__":
    run()
