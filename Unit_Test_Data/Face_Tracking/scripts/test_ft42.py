"""
test_ft42.py  —  FT-42
Test:  PersonTracker.get_lost  (pipeline.py:530)
Check: with three fresh tracks (2s, 4s, 6s old) and max_lost=90, the lost list
       is empty.

Output: Unit_Test_Data/Face_Tracking/data/test_ft42/
"""

from datetime import datetime, timedelta

from _ft_helpers import (
    collect_embeddings, make_tracker, make_active_track,
    prepare_data_dir, save_single_test_result, print_verdict,
)

TEST_ID = "FT-42"
DATA_OUT = prepare_data_dir(TEST_ID)


def run():
    print(f"\n{'='*60}\n{TEST_ID} — get_lost all-fresh  "
          f"({datetime.now():%H:%M:%S})\n{'='*60}")

    embeddings, boxes = collect_embeddings(n=3)
    assert len(embeddings) >= 3, "Need >=3 real embeddings"

    now = datetime(2026, 4, 9, 10, 0, 0)
    tracker = make_tracker()
    for i in range(3):
        tracker.active[i + 1] = make_active_track(
            embeddings[i], boxes[i], timestamp=now - timedelta(seconds=(i + 1) * 2))

    lost = tracker.get_lost(now, max_lost_seconds=90)
    ok = lost == []
    status = "PASS" if ok else "FAIL"
    print(f"  lost={lost}  expected=[]  | {status}")
    print_verdict(TEST_ID, status)

    save_single_test_result(TEST_ID, DATA_OUT, {
        "status": status,
        "lost": lost,
    })


if __name__ == "__main__":
    run()
