"""
test_ft44.py  —  FT-44
Test:  PersonTracker.get_lost  (pipeline.py:530)
Check: an ISO-string last_timestamp (the form persisted to state.json) is
       parsed correctly; a 120-second gap is reported as lost when max_lost=90.

Output: Unit_Test_Data/Face_Tracking/data/test_ft44/
"""

from datetime import datetime

from _ft_helpers import (
    collect_embeddings, make_tracker, make_active_track,
    prepare_data_dir, save_single_test_result, print_verdict,
)

TEST_ID = "FT-44"
DATA_OUT = prepare_data_dir(TEST_ID)


def run():
    print(f"\n{'='*60}\n{TEST_ID} — get_lost with ISO string timestamp  "
          f"({datetime.now():%H:%M:%S})\n{'='*60}")

    embeddings, boxes = collect_embeddings(n=1)
    assert embeddings, "Need >=1 real embedding"

    tracker = make_tracker()
    track = make_active_track(embeddings[0], boxes[0])
    track.last_timestamp = "2026-04-09T09:00:00"   # state.json form
    tracker.active[5] = track

    current = datetime(2026, 4, 9, 9, 2, 0)        # 120s after
    exception_raised = False
    lost = []
    try:
        lost = tracker.get_lost(current, max_lost_seconds=90)
    except Exception as exc:
        exception_raised = True
        print(f"  EXCEPTION: {exc}")

    ok = (5 in lost) and not exception_raised
    status = "PASS" if ok else "FAIL"
    print(f"  lost={lost}  5_in_lost={5 in lost}  exception={exception_raised}  | {status}")
    print_verdict(TEST_ID, status)

    save_single_test_result(TEST_ID, DATA_OUT, {
        "status": status,
        "lost": lost,
        "id_in_lost": 5 in lost,
        "exception_raised": exception_raised,
    })


if __name__ == "__main__":
    run()
