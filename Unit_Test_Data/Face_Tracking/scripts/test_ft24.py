"""
test_ft24.py  —  FT-24
Test:  PersonTracker._box_distance  (pipeline.py:417)
Check: top-left vs bottom-right of the frame -> normalised distance > 1.0.

Output: Unit_Test_Data/Face_Tracking/data/test_ft24/
"""

from datetime import datetime

from _ft_helpers import (
    VIDEO_PATH, sample_frame, make_tracker,
    prepare_data_dir, save_single_test_result, print_verdict,
)

TEST_ID = "FT-24"
DATA_OUT = prepare_data_dir(TEST_ID)


def run():
    print(f"\n{'='*60}\n{TEST_ID} — _box_distance frame corners  "
          f"({datetime.now():%H:%M:%S})\n{'='*60}")

    frame = sample_frame(VIDEO_PATH, frame_idx=40)
    H, W = frame.shape[:2]

    tracker = make_tracker(frame_wh=(W, H))
    box_tl = [0, 0, 100, 100]
    box_br = [W - 100, H - 100, W, H]
    dist = tracker._box_distance(box_tl, box_br)

    ok = dist > 1.0
    status = "PASS" if ok else "FAIL"
    print(f"  box_tl={box_tl}  box_br={box_br}  distance={dist:.4f}  | {status}")
    print_verdict(TEST_ID, status)

    save_single_test_result(TEST_ID, DATA_OUT, {
        "status": status,
        "box_tl": box_tl,
        "box_br": box_br,
        "distance": round(dist, 6),
    })


if __name__ == "__main__":
    run()
