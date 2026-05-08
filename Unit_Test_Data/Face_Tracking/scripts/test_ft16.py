"""
test_ft16.py  —  FT-16
Test:  ActiveTrack.deserialize  (pipeline.py:368)
Check: when the .npy file referenced by the state dict is missing, deserialize
       raises an exception (rather than silently constructing a bad track).

Output: Unit_Test_Data/Face_Tracking/data/test_ft16/
"""

import shutil
from datetime import datetime

from _ft_helpers import (
    prepare_data_dir, save_single_test_result, print_verdict,
)

TEST_ID = "FT-16"
DATA_OUT = prepare_data_dir(TEST_ID)
TRACKS  = DATA_OUT / "tracks"


def run():
    import pipeline
    print(f"\n{'='*60}\n{TEST_ID} — deserialize with missing .npy  "
          f"({datetime.now():%H:%M:%S})\n{'='*60}")

    if TRACKS.exists():
        shutil.rmtree(TRACKS)
    TRACKS.mkdir(parents=True)

    bad_state = {
        "bank_file": "track_9999_bank.npy",
        "last_box": [0, 0, 100, 200],
        "last_timestamp": "2026-04-09T09:00:00",
        "lost_seconds": 0.0,
    }

    exception_raised = False
    exc_type = ""
    try:
        pipeline.ActiveTrack.deserialize(bad_state, str(TRACKS))
    except Exception as exc:
        exception_raised = True
        exc_type = type(exc).__name__

    ok = exception_raised
    status = "PASS" if ok else "FAIL"
    print(f"  exception_raised={exception_raised}  type={exc_type}  | {status}")
    print_verdict(TEST_ID, status)

    save_single_test_result(TEST_ID, DATA_OUT, {
        "status": status,
        "exception_raised": exception_raised,
        "exception_type": exc_type,
    })


if __name__ == "__main__":
    run()
