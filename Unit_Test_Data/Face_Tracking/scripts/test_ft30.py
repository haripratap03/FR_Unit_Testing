"""
test_ft30.py  —  FT-30
Test:  PersonTracker.match_or_create  (pipeline.py:425)
Check: a large gap_seconds widens the spatial gate so a mid-frame box can
       still match an active track whose last box was near the corner.

Output: Unit_Test_Data/Face_Tracking/data/test_ft30/
"""

from datetime import datetime
from unittest.mock import patch

import numpy as np

from _ft_helpers import (
    load_osnet, make_tracker, make_active_track, make_emb_at_dist, random_unit_vec,
    prepare_data_dir, save_single_test_result, print_verdict,
)

TEST_ID = "FT-30"
DATA_OUT = prepare_data_dir(TEST_ID)


def run():
    import pipeline
    MAX_JUMP = pipeline.MAX_BOX_JUMP
    print(f"\n{'='*60}\n{TEST_ID} — match_or_create gap-widened gate  "
          f"(MAX_BOX_JUMP={MAX_JUMP})  ({datetime.now():%H:%M:%S})\n{'='*60}")

    ref_emb = random_unit_vec(seed=3)
    close_emb = make_emb_at_dist(ref_emb, 0.15, seed=4)

    tracker = make_tracker(osnet=load_osnet(), frame_wh=(2688, 1520))
    tracker.active = {1: make_active_track(ref_emb, [10, 10, 100, 100])}
    tracker.next_id = 2

    frame = np.zeros((1520, 2688, 3), dtype=np.uint8)
    mid_box = [1300, 700, 1400, 900]
    with patch.object(tracker.osnet, "extract_batch", return_value=[close_emb]):
        result = tracker.match_or_create(frame, [mid_box], datetime.now(), gap_seconds=35)

    ok = len(result) == 1
    status = "PASS" if ok else "FAIL"
    capped_gate = min(MAX_JUMP * max(1, 35 / 0.5), 0.8)
    box_dist = tracker._box_distance(mid_box, [10, 10, 100, 100])
    print(f"  gap=35s  gate={capped_gate:.2f}  box_dist={box_dist:.4f}  "
          f"matched={len(result)}  | {status}")
    print_verdict(TEST_ID, status)

    save_single_test_result(TEST_ID, DATA_OUT, {
        "status": status,
        "gate": round(capped_gate, 4),
        "box_dist": round(box_dist, 4),
        "matched": len(result),
    })


if __name__ == "__main__":
    run()
