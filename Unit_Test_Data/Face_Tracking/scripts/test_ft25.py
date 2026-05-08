"""
test_ft25.py  —  FT-25
Test:  PersonTracker._box_distance  (pipeline.py:417)
Check: the same physical 200-px shift produces a larger normalised distance
       on a smaller frame than on a larger one (scale invariance).

Output: Unit_Test_Data/Face_Tracking/data/test_ft25/
"""

from datetime import datetime

from _ft_helpers import (
    VIDEO_PATH, sample_frame, make_tracker,
    prepare_data_dir, save_single_test_result, print_verdict,
)

TEST_ID = "FT-25"
DATA_OUT = prepare_data_dir(TEST_ID)


def run():
    print(f"\n{'='*60}\n{TEST_ID} — _box_distance scale invariance  "
          f"({datetime.now():%H:%M:%S})\n{'='*60}")

    frame = sample_frame(VIDEO_PATH, frame_idx=40)
    H, W = frame.shape[:2]

    box_a = [200, 100, 400, 400]
    box_b = [400, 100, 600, 400]   # 200-px horizontal shift

    tracker_small = make_tracker(frame_wh=(640, 360))
    tracker_large = make_tracker(frame_wh=(W, H))
    dist_small = tracker_small._box_distance(box_a, box_b)
    dist_large = tracker_large._box_distance(box_a, box_b)

    ok = dist_small > dist_large
    status = "PASS" if ok else "FAIL"
    print(f"  small={dist_small:.4f}  large={dist_large:.4f}  small>large={ok}  | {status}")
    print_verdict(TEST_ID, status)

    save_single_test_result(TEST_ID, DATA_OUT, {
        "status": status,
        "dist_640x360": round(dist_small, 6),
        "dist_full_res": round(dist_large, 6),
        "frame_full": [W, H],
    })


if __name__ == "__main__":
    run()
