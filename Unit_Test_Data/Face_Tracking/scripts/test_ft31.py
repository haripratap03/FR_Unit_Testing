"""
test_ft31.py  —  FT-31
Test:  PersonTracker.match_or_create  (pipeline.py:425)
Check: a close match (cos_dist < REID_DISTANCE * UPDATE_GATE) extends the
       embedding bank by 1 entry.

Output: Unit_Test_Data/Face_Tracking/data/test_ft31/
"""

from datetime import datetime
from unittest.mock import patch

import numpy as np

from _ft_helpers import (
    load_osnet, make_tracker, make_active_track, make_emb_at_dist, random_unit_vec,
    prepare_data_dir, save_single_test_result, print_verdict,
)

TEST_ID = "FT-31"
DATA_OUT = prepare_data_dir(TEST_ID)


def run():
    import pipeline
    REID, UG = pipeline.REID_DISTANCE, pipeline.UPDATE_GATE
    print(f"\n{'='*60}\n{TEST_ID} — match_or_create bank update on close match  "
          f"(threshold={REID*UG:.3f})  ({datetime.now():%H:%M:%S})\n{'='*60}")

    ref_emb = random_unit_vec(seed=5)
    update_thresh = REID * UG
    close_emb = make_emb_at_dist(ref_emb, update_thresh * 0.5, seed=6)

    tracker = make_tracker(osnet=load_osnet(), frame_wh=(2688, 1520))
    tracker.active = {1: make_active_track(ref_emb)}
    tracker.next_id = 2
    bank_before = len(tracker.active[1].bank)

    frame = np.zeros((1520, 2688, 3), dtype=np.uint8)
    with patch.object(tracker.osnet, "extract_batch", return_value=[close_emb]):
        tracker.match_or_create(frame, [[100, 100, 200, 300]], datetime.now(), gap_seconds=0.05)

    bank_after = len(tracker.active[1].bank)
    ok = bank_after == bank_before + 1
    status = "PASS" if ok else "FAIL"
    cos_dist = float(1.0 - np.dot(close_emb, ref_emb))
    print(f"  cos_dist={cos_dist:.4f}  bank: {bank_before} -> {bank_after}  | {status}")
    print_verdict(TEST_ID, status)

    save_single_test_result(TEST_ID, DATA_OUT, {
        "status": status,
        "cos_dist": round(cos_dist, 4),
        "bank_before": bank_before,
        "bank_after": bank_after,
    })


if __name__ == "__main__":
    run()
