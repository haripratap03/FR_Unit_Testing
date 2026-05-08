"""
test_ft27.py  —  FT-27
Test:  PersonTracker.match_or_create  (pipeline.py:425)
Check: a person tracked across consecutive frames keeps the same track ID.

Strategy: run a live tracking session on real footage and look for at least one
track ID that appears in two consecutive frames' matched output.

Output: Unit_Test_Data/Face_Tracking/data/test_ft27/
"""

import json
from datetime import datetime

from _ft_helpers import (
    VIDEO_PATH, load_yolo, load_osnet, read_frames, detect_persons,
    prepare_data_dir, save_single_test_result, print_verdict,
)

TEST_ID = "FT-27"
DATA_OUT = prepare_data_dir(TEST_ID)


def run():
    import pipeline
    print(f"\n{'='*60}\n{TEST_ID} — match_or_create stable IDs  "
          f"({datetime.now():%H:%M:%S})\n{'='*60}")

    yolo    = load_yolo()
    osnet   = load_osnet()
    tracker = pipeline.PersonTracker(osnet)
    frames  = read_frames(VIDEO_PATH, start=0, count=40, step=1)
    print(f"[SETUP] tracking {len(frames)} frames")

    session = []
    for fi, frame in enumerate(frames):
        boxes = detect_persons(yolo, frame)
        ts    = datetime(2026, 4, 21, 13, 0, fi)
        matched = tracker.match_or_create(frame, boxes, ts, gap_seconds=0.05)
        session.append([uid for uid, _ in matched])

    stable = set()
    for i in range(1, len(session)):
        stable |= set(session[i]) & set(session[i-1])

    ok = len(stable) > 0
    status = "PASS" if ok else "FAIL"
    print(f"  stable IDs across consecutive frames: {sorted(stable)}  | {status}")
    print_verdict(TEST_ID, status)

    with open(DATA_OUT / "session_log.json", "w") as f:
        json.dump([{"frame": i, "matched_ids": ids} for i, ids in enumerate(session)],
                  f, indent=2)

    save_single_test_result(TEST_ID, DATA_OUT, {
        "status": status,
        "stable_ids": sorted(stable),
    })


if __name__ == "__main__":
    run()
