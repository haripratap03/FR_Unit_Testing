"""
test_ft50.py  —  FT-50
Test:  PersonTracker.remove  (pipeline.py:558)
Check: remove(99) on a pool that doesn't contain id 99 raises no exception
       and leaves the pool unchanged.

Output: Unit_Test_Data/Face_Tracking/data/test_ft50/
"""

from datetime import datetime

from _ft_helpers import (
    collect_embeddings, make_tracker, make_active_track,
    prepare_data_dir, save_single_test_result, print_verdict,
)

TEST_ID = "FT-50"
DATA_OUT = prepare_data_dir(TEST_ID)


def run():
    print(f"\n{'='*60}\n{TEST_ID} — PersonTracker.remove non-existent  "
          f"({datetime.now():%H:%M:%S})\n{'='*60}")

    embeddings, boxes = collect_embeddings(n=2)
    assert len(embeddings) >= 2, "Need >=2 real embeddings"

    tracker = make_tracker()
    tracker.active[1] = make_active_track(embeddings[0], boxes[0])
    tracker.active[3] = make_active_track(embeddings[1], boxes[1])
    before = sorted(tracker.active.keys())

    exception_raised = False
    try:
        tracker.remove(99)
    except Exception as exc:
        exception_raised = True
        print(f"  EXCEPTION: {exc}")

    after = sorted(tracker.active.keys())
    ok = (not exception_raised) and (after == [1, 3])
    status = "PASS" if ok else "FAIL"
    print(f"  before={before}  after={after}  exception={exception_raised}  | {status}")
    print_verdict(TEST_ID, status)

    save_single_test_result(TEST_ID, DATA_OUT, {
        "status": status,
        "before": before,
        "after": after,
        "pool_unchanged": after == [1, 3],
        "exception_raised": exception_raised,
    })


if __name__ == "__main__":
    run()
