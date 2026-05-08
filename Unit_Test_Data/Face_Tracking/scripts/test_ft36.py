"""
test_ft36.py  —  FT-36
Test:  PersonTracker.match_or_create  (pipeline.py:425)
Check: when a pending track is promoted, the new active track's bank is seeded
       from the pending history and contains >= WARMUP_FRAMES embeddings.

Output: Unit_Test_Data/Face_Tracking/data/test_ft36/
"""

from datetime import datetime
from unittest.mock import patch

import numpy as np

from _ft_helpers import (
    load_osnet, make_tracker, make_emb_at_dist, random_unit_vec,
    prepare_data_dir, save_single_test_result, print_verdict,
)

TEST_ID = "FT-36"
DATA_OUT = prepare_data_dir(TEST_ID)


def run():
    import pipeline
    WARMUP = pipeline.WARMUP_FRAMES
    print(f"\n{'='*60}\n{TEST_ID} — match_or_create promoted-track bank seeding  "
          f"(WARMUP_FRAMES={WARMUP})  ({datetime.now():%H:%M:%S})\n{'='*60}")

    ref_emb = random_unit_vec(seed=14)
    close   = make_emb_at_dist(ref_emb, 0.08, seed=15)

    tracker = make_tracker(osnet=load_osnet(), frame_wh=(2688, 1520))
    frame = np.zeros((1520, 2688, 3), dtype=np.uint8)
    box   = [100, 100, 300, 500]

    with patch.object(tracker.osnet, "extract_batch", return_value=[ref_emb]):
        tracker.match_or_create(frame, [box], datetime.now(), gap_seconds=0.05)
    for _ in range(WARMUP - 2):
        with patch.object(tracker.osnet, "extract_batch", return_value=[close]):
            tracker.match_or_create(frame, [box], datetime.now(), gap_seconds=0.05)
    with patch.object(tracker.osnet, "extract_batch", return_value=[close]):
        result = tracker.match_or_create(frame, [box], datetime.now(), gap_seconds=0.05)

    if not result:
        save_single_test_result(TEST_ID, DATA_OUT, {
            "status": "SKIP",
            "reason": "Promotion did not happen — FT-35 must pass first",
        })
        print_verdict(TEST_ID, "SKIP")
        return

    promoted_id = result[0][0]
    bank_size = len(tracker.active[promoted_id].bank)
    ok = bank_size >= WARMUP
    status = "PASS" if ok else "FAIL"
    print(f"  bank_size={bank_size}  >= {WARMUP}: {ok}  | {status}")
    print_verdict(TEST_ID, status)

    save_single_test_result(TEST_ID, DATA_OUT, {
        "status": status,
        "bank_size": bank_size,
    })


if __name__ == "__main__":
    run()
