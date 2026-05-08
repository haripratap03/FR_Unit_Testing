"""
test_ft41.py  —  FT-41
Test:  PersonTracker.get_lost  (pipeline.py:530)
Check: with two tracks (30s ago, 100s ago) and max_lost=90, only the 100s-old
       track is returned in the lost list.

Output: Unit_Test_Data/Face_Tracking/data/test_ft41/
"""

from datetime import datetime, timedelta

from _ft_helpers import (
    collect_embeddings, make_tracker, make_active_track,
    prepare_data_dir, save_single_test_result, print_verdict,
)

TEST_ID = "FT-41"
DATA_OUT = prepare_data_dir(TEST_ID)


def run():
    print(f"\n{'='*60}\n{TEST_ID} — get_lost stale vs fresh  "
          f"({datetime.now():%H:%M:%S})\n{'='*60}")

    embeddings, boxes = collect_embeddings(n=2)
    assert len(embeddings) >= 2, "Need >=2 real embeddings"

    now = datetime(2026, 4, 9, 10, 0, 0)
    tracker = make_tracker()
    tracker.active[1] = make_active_track(embeddings[0], boxes[0],
                                          timestamp=now - timedelta(seconds=30))
    tracker.active[2] = make_active_track(embeddings[1], boxes[1],
                                          timestamp=now - timedelta(seconds=100))

    lost = tracker.get_lost(now, max_lost_seconds=90)
    ok = lost == [2]
    status = "PASS" if ok else "FAIL"
    print(f"  lost={lost}  expected=[2]  | {status}")
    print_verdict(TEST_ID, status)

    save_single_test_result(TEST_ID, DATA_OUT, {
        "status": status,
        "lost": lost,
        "expected": [2],
    })


if __name__ == "__main__":
    run()
