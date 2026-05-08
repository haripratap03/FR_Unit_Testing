"""
test_ft47.py  —  FT-47
Test:  PersonTracker.serialize_state  (pipeline.py:560)
Check: an empty tracker serialises to {active_tracks={}, next_track_id=1} and
       writes no .npy files.

Output: Unit_Test_Data/Face_Tracking/data/test_ft47/
"""

import shutil
from datetime import datetime

from _ft_helpers import (
    make_tracker,
    prepare_data_dir, save_single_test_result, print_verdict,
)

TEST_ID = "FT-47"
DATA_OUT = prepare_data_dir(TEST_ID)
TRACKS  = DATA_OUT / "tracks"


def run():
    print(f"\n{'='*60}\n{TEST_ID} — serialize_state empty tracker  "
          f"({datetime.now():%H:%M:%S})\n{'='*60}")

    if TRACKS.exists():
        shutil.rmtree(TRACKS)
    TRACKS.mkdir(parents=True)

    tracker = make_tracker()
    tracker.next_id = 1
    state = tracker.serialize_state(str(TRACKS))
    npy_files = list(TRACKS.glob("*.npy"))

    active_empty = (state.get("active_tracks") == {})
    nid_ok       = (state.get("next_track_id") == 1)
    no_npy       = (len(npy_files) == 0)
    ok = active_empty and nid_ok and no_npy
    status = "PASS" if ok else "FAIL"
    print(f"  active_empty={active_empty}  next_track_id={state.get('next_track_id')}  "
          f"npy_files={len(npy_files)}  | {status}")
    print_verdict(TEST_ID, status)

    save_single_test_result(TEST_ID, DATA_OUT, {
        "status": status,
        "active_tracks_empty": active_empty,
        "next_track_id": state.get("next_track_id"),
        "npy_files_written": len(npy_files),
    })


if __name__ == "__main__":
    run()
