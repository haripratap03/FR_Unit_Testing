"""
test_ft43.py  —  FT-43
Test:  PersonTracker.get_lost  (pipeline.py:530)
Check: an empty active pool returns [] without raising.

Output: Unit_Test_Data/Face_Tracking/data/test_ft43/
"""

from datetime import datetime

from _ft_helpers import (
    make_tracker,
    prepare_data_dir, save_single_test_result, print_verdict,
)

TEST_ID = "FT-43"
DATA_OUT = prepare_data_dir(TEST_ID)


def run():
    print(f"\n{'='*60}\n{TEST_ID} — get_lost empty pool  "
          f"({datetime.now():%H:%M:%S})\n{'='*60}")

    tracker = make_tracker()
    exception_raised = False
    lost = []
    try:
        lost = tracker.get_lost(datetime.now(), max_lost_seconds=90)
    except Exception as exc:
        exception_raised = True
        print(f"  EXCEPTION: {exc}")

    ok = (lost == []) and not exception_raised
    status = "PASS" if ok else "FAIL"
    print(f"  lost={lost}  exception_raised={exception_raised}  | {status}")
    print_verdict(TEST_ID, status)

    save_single_test_result(TEST_ID, DATA_OUT, {
        "status": status,
        "lost": lost,
        "exception_raised": exception_raised,
    })


if __name__ == "__main__":
    run()
