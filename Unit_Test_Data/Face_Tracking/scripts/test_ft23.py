"""
test_ft23.py  —  FT-23
Test:  PersonTracker._box_distance  (pipeline.py:417)
Check: identical boxes have distance 0.0.

Output: Unit_Test_Data/Face_Tracking/data/test_ft23/
"""

from datetime import datetime

from _ft_helpers import (
    VIDEO_PATH, load_yolo, detect_persons, sample_frame, make_tracker,
    prepare_data_dir, save_single_test_result, print_verdict,
)

TEST_ID = "FT-23"
DATA_OUT = prepare_data_dir(TEST_ID)


def run():
    print(f"\n{'='*60}\n{TEST_ID} — _box_distance identical boxes  "
          f"({datetime.now():%H:%M:%S})\n{'='*60}")

    yolo = load_yolo()
    frame = sample_frame(VIDEO_PATH, frame_idx=40)
    H, W = frame.shape[:2]
    boxes = detect_persons(yolo, frame)
    assert boxes, "No persons detected"

    tracker = make_tracker(frame_wh=(W, H))
    box = boxes[0]
    dist = tracker._box_distance(box, box)

    ok = abs(dist) < 1e-9
    status = "PASS" if ok else "FAIL"
    print(f"  box={box}  distance={dist}  | {status}")
    print_verdict(TEST_ID, status)

    save_single_test_result(TEST_ID, DATA_OUT, {
        "status": status,
        "box": box,
        "distance": dist,
    })


if __name__ == "__main__":
    run()
