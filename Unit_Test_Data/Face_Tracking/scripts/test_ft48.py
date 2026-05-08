"""
test_ft48.py  —  FT-48
Test:  PersonTracker.restore_state  (pipeline.py:560)
Check: when one of the .npy bank files is missing on disk, restore_state skips
       that track gracefully and recovers the others without raising.

Output: Unit_Test_Data/Face_Tracking/data/test_ft48/
"""

import shutil
from datetime import datetime

from _ft_helpers import (
    collect_embeddings, make_tracker, make_active_track,
    prepare_data_dir, save_single_test_result, print_verdict,
)

TEST_ID = "FT-48"
DATA_OUT = prepare_data_dir(TEST_ID)
TRACKS  = DATA_OUT / "tracks"


def run():
    print(f"\n{'='*60}\n{TEST_ID} — restore_state with missing .npy  "
          f"({datetime.now():%H:%M:%S})\n{'='*60}")

    if TRACKS.exists():
        shutil.rmtree(TRACKS)
    TRACKS.mkdir(parents=True)

    embeddings, boxes = collect_embeddings(n=2)
    assert len(embeddings) >= 2, "Need >=2 real embeddings"

    tracker = make_tracker()
    tracker.active[1] = make_active_track(embeddings[0], boxes[0])
    tracker.active[2] = make_active_track(embeddings[1], boxes[1])
    tracker.next_id = 3
    state = tracker.serialize_state(str(TRACKS))

    target = TRACKS / "track_0002_bank.npy"
    if target.exists():
        target.unlink()
        print(f"  Deleted {target.name} to simulate missing .npy")

    restored = make_tracker()
    exception_raised = False
    try:
        restored.restore_state(state, str(TRACKS))
    except Exception as exc:
        exception_raised = True
        print(f"  EXCEPTION: {exc}")

    ok = (len(restored.active) == 1 and 1 in restored.active and not exception_raised)
    status = "PASS" if ok else "FAIL"
    print(f"  active_count={len(restored.active)}  track1_present={1 in restored.active}  "
          f"exception={exception_raised}  | {status}")
    print_verdict(TEST_ID, status)

    save_single_test_result(TEST_ID, DATA_OUT, {
        "status": status,
        "active_count": len(restored.active),
        "track1_present": 1 in restored.active,
        "exception_raised": exception_raised,
    })


if __name__ == "__main__":
    run()
