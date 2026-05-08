"""
test_ft32.py  —  FT-32
Test:  PersonTracker.match_or_create  (pipeline.py:425)
Check: a borderline match (REID*UG <= cos_dist < REID) still produces a track
       hit but does NOT extend the embedding bank.

Output: Unit_Test_Data/Face_Tracking/data/test_ft32/
"""

from datetime import datetime
from unittest.mock import patch

import numpy as np

from _ft_helpers import (
    load_osnet, make_tracker, make_active_track, make_emb_at_dist, random_unit_vec,
    prepare_data_dir, save_single_test_result, print_verdict,
)

TEST_ID = "FT-32"
DATA_OUT = prepare_data_dir(TEST_ID)


def run():
    import pipeline
    REID = pipeline.REID_DISTANCE
    print(f"\n{'='*60}\n{TEST_ID} — match_or_create borderline (no bank update)  "
          f"({datetime.now():%H:%M:%S})\n{'='*60}")

    ref_emb = random_unit_vec(seed=7)
    borderline = make_emb_at_dist(ref_emb, REID * 0.92, seed=8)

    tracker = make_tracker(osnet=load_osnet(), frame_wh=(2688, 1520))
    tracker.active = {1: make_active_track(ref_emb)}
    tracker.next_id = 2
    bank_before = len(tracker.active[1].bank)

    frame = np.zeros((1520, 2688, 3), dtype=np.uint8)
    with patch.object(tracker.osnet, "extract_batch", return_value=[borderline]):
        result = tracker.match_or_create(frame, [[100, 100, 200, 300]], datetime.now(), gap_seconds=0.05)

    bank_after = len(tracker.active[1].bank)
    matched = len(result) == 1
    no_update = bank_after == bank_before
    ok = matched and no_update
    status = "PASS" if ok else "FAIL"
    cos_dist = float(1.0 - np.dot(borderline, ref_emb))
    print(f"  cos_dist={cos_dist:.4f}  matched={matched}  bank_unchanged={no_update}  | {status}")
    print_verdict(TEST_ID, status)

    save_single_test_result(TEST_ID, DATA_OUT, {
        "status": status,
        "cos_dist": round(cos_dist, 4),
        "matched": matched,
        "bank_unchanged": no_update,
    })


if __name__ == "__main__":
    run()
