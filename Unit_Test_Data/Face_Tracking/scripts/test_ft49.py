"""
test_ft49.py  —  FT-49
Test:  PersonTracker.remove  (pipeline.py:558)
Check: remove(2) drops track 2 from the active pool while leaving 1 and 3 intact.

Output: Unit_Test_Data/Face_Tracking/data/test_ft49/
"""

from datetime import datetime

from _ft_helpers import (
    collect_embeddings, make_tracker, make_active_track,
    prepare_data_dir, save_single_test_result, print_verdict,
)

TEST_ID = "FT-49"
DATA_OUT = prepare_data_dir(TEST_ID)


def run():
    print(f"\n{'='*60}\n{TEST_ID} — PersonTracker.remove existing  "
          f"({datetime.now():%H:%M:%S})\n{'='*60}")

    embeddings, boxes = collect_embeddings(n=3)
    assert len(embeddings) >= 3, "Need >=3 real embeddings"

    tracker = make_tracker()
    for uid, (e, b) in zip([1, 2, 3], zip(embeddings, boxes)):
        tracker.active[uid] = make_active_track(e, b)

    before = sorted(tracker.active.keys())
    tracker.remove(2)
    after = sorted(tracker.active.keys())

    ok = (2 not in tracker.active) and (1 in tracker.active) and (3 in tracker.active)
    status = "PASS" if ok else "FAIL"
    print(f"  before={before}  after={after}  | {status}")
    print_verdict(TEST_ID, status)

    save_single_test_result(TEST_ID, DATA_OUT, {
        "status": status,
        "before": before,
        "after": after,
        "2_removed": 2 not in tracker.active,
        "1_intact": 1 in tracker.active,
        "3_intact": 3 in tracker.active,
    })


if __name__ == "__main__":
    run()
