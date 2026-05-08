"""
test_ft35.py  —  FT-35
Test:  PersonTracker.match_or_create  (pipeline.py:425)
Check: a pending track is promoted to active once frames_seen reaches WARMUP_FRAMES.

Strategy: feed exactly WARMUP_FRAMES close-match frames and verify the result
contains an ID and tracker.active gains the track.

Output: Unit_Test_Data/Face_Tracking/data/test_ft35/
"""

from datetime import datetime
from unittest.mock import patch

import numpy as np

from _ft_helpers import (
    load_osnet, make_tracker, make_emb_at_dist, random_unit_vec,
    prepare_data_dir, save_single_test_result, print_verdict,
)

TEST_ID = "FT-35"
DATA_OUT = prepare_data_dir(TEST_ID)


def run():
    import pipeline
    WARMUP = pipeline.WARMUP_FRAMES
    print(f"\n{'='*60}\n{TEST_ID} — match_or_create promotion at WARMUP={WARMUP}  "
          f"({datetime.now():%H:%M:%S})\n{'='*60}")

    ref_emb = random_unit_vec(seed=14)
    close   = make_emb_at_dist(ref_emb, 0.08, seed=15)

    tracker = make_tracker(osnet=load_osnet(), frame_wh=(2688, 1520))
    frame = np.zeros((1520, 2688, 3), dtype=np.uint8)
    box   = [100, 100, 300, 500]

    # Frame 1
    with patch.object(tracker.osnet, "extract_batch", return_value=[ref_emb]):
        tracker.match_or_create(frame, [box], datetime.now(), gap_seconds=0.05)
    # Frames 2..WARMUP-1: still pending
    for _ in range(WARMUP - 2):
        with patch.object(tracker.osnet, "extract_batch", return_value=[close]):
            tracker.match_or_create(frame, [box], datetime.now(), gap_seconds=0.05)
    # Frame WARMUP: should promote
    with patch.object(tracker.osnet, "extract_batch", return_value=[close]):
        result = tracker.match_or_create(frame, [box], datetime.now(), gap_seconds=0.05)

    promoted = (len(tracker.active) >= 1) and (len(result) >= 1)
    ok = promoted
    status = "PASS" if ok else "FAIL"
    promoted_id = result[0][0] if result else None
    print(f"  active_count={len(tracker.active)}  result_len={len(result)}  "
          f"promoted_id={promoted_id}  | {status}")
    print_verdict(TEST_ID, status)

    save_single_test_result(TEST_ID, DATA_OUT, {
        "status": status,
        "active_count": len(tracker.active),
        "promoted_id": promoted_id,
    })


if __name__ == "__main__":
    run()
