"""
test_ft45.py  —  FT-45
Test:  PersonTracker.get_lost  (pipeline.py:530)
Check: a track with last_timestamp exactly max_lost_seconds in the past is
       NOT considered lost (strict greater-than).

Output: Unit_Test_Data/Face_Tracking/data/test_ft45/
"""

from datetime import datetime, timedelta

from _ft_helpers import (
    collect_embeddings, make_tracker, make_active_track,
    prepare_data_dir, save_single_test_result, print_verdict,
)

TEST_ID = "FT-45"
DATA_OUT = prepare_data_dir(TEST_ID)


def run():
    print(f"\n{'='*60}\n{TEST_ID} — get_lost boundary not lost  "
          f"({datetime.now():%H:%M:%S})\n{'='*60}")

    embeddings, boxes = collect_embeddings(n=1)
    assert embeddings, "Need >=1 real embedding"

    now = datetime(2026, 4, 9, 10, 0, 0)
    tracker = make_tracker()
    tracker.active[7] = make_active_track(embeddings[0], boxes[0],
                                          timestamp=now - timedelta(seconds=90))

    lost = tracker.get_lost(now, max_lost_seconds=90)
    ok = 7 not in lost
    status = "PASS" if ok else "FAIL"
    print(f"  lost={lost}  7_in_lost={7 in lost}  | {status}")
    print_verdict(TEST_ID, status)

    save_single_test_result(TEST_ID, DATA_OUT, {
        "status": status,
        "lost": lost,
        "id_not_in_lost": 7 not in lost,
    })


if __name__ == "__main__":
    run()
