"""
test_ft34.py  —  FT-34
Test:  PersonTracker.match_or_create  (pipeline.py:425)
Check: a pending track's frames_seen increments each frame; the track is NOT
       promoted before WARMUP_FRAMES is reached.

Strategy: feed the tracker WARMUP_FRAMES-1 close-match frames and verify the
state stays in tracker.pending with frames_seen == WARMUP_FRAMES-1.

Output: Unit_Test_Data/Face_Tracking/data/test_ft34/
"""

from datetime import datetime
from unittest.mock import patch

import numpy as np

from _ft_helpers import (
    load_osnet, make_tracker, make_emb_at_dist, random_unit_vec,
    prepare_data_dir, save_single_test_result, print_verdict,
)

TEST_ID = "FT-34"
DATA_OUT = prepare_data_dir(TEST_ID)


def run():
    import pipeline
    WARMUP = pipeline.WARMUP_FRAMES
    print(f"\n{'='*60}\n{TEST_ID} — match_or_create pending warmup  "
          f"(WARMUP_FRAMES={WARMUP})  ({datetime.now():%H:%M:%S})\n{'='*60}")

    ref_emb = random_unit_vec(seed=14)
    close   = make_emb_at_dist(ref_emb, 0.08, seed=15)

    tracker = make_tracker(osnet=load_osnet(), frame_wh=(2688, 1520))
    frame = np.zeros((1520, 2688, 3), dtype=np.uint8)
    box   = [100, 100, 300, 500]

    # Frame 1: creates pending
    with patch.object(tracker.osnet, "extract_batch", return_value=[ref_emb]):
        tracker.match_or_create(frame, [box], datetime.now(), gap_seconds=0.05)
    # Frames 2..WARMUP-1: still pending
    for _ in range(WARMUP - 2):
        with patch.object(tracker.osnet, "extract_batch", return_value=[close]):
            tracker.match_or_create(frame, [box], datetime.now(), gap_seconds=0.05)

    still_pending = len(tracker.pending) > 0
    not_promoted  = len(tracker.active) == 0
    pend = list(tracker.pending.values())[0] if tracker.pending else None
    frames_seen = pend.frames_seen if pend else None

    ok = still_pending and not_promoted and (frames_seen == WARMUP - 1)
    status = "PASS" if ok else "FAIL"
    print(f"  frames_seen={frames_seen}  still_pending={still_pending}  "
          f"not_promoted={not_promoted}  | {status}")
    print_verdict(TEST_ID, status)

    save_single_test_result(TEST_ID, DATA_OUT, {
        "status": status,
        "frames_seen": frames_seen,
        "still_pending": still_pending,
    })


if __name__ == "__main__":
    run()
