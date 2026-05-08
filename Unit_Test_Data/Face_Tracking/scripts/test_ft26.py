"""
test_ft26.py  —  FT-26
Test:  PersonTracker._box_distance  (pipeline.py:417)
Check: a slow-moving person between two consecutive frames yields a small
       normalised distance (< 0.05).

Output: Unit_Test_Data/Face_Tracking/data/test_ft26/
"""

from datetime import datetime

import cv2

from _ft_helpers import (
    VIDEO_PATH, load_yolo, detect_persons, make_tracker,
    prepare_data_dir, save_single_test_result, print_verdict,
)

TEST_ID = "FT-26"
DATA_OUT = prepare_data_dir(TEST_ID)


def run():
    print(f"\n{'='*60}\n{TEST_ID} — _box_distance consecutive-frame match  "
          f"({datetime.now():%H:%M:%S})\n{'='*60}")

    yolo = load_yolo()
    cap = cv2.VideoCapture(str(VIDEO_PATH))
    cap.set(cv2.CAP_PROP_POS_FRAMES, 40)
    ok_a, frame_a = cap.read()
    cap.set(cv2.CAP_PROP_POS_FRAMES, 42)
    ok_b, frame_b = cap.read()
    cap.release()
    assert ok_a and ok_b, "Could not read both frames"
    H, W = frame_a.shape[:2]

    boxes_a = detect_persons(yolo, frame_a)
    boxes_b = detect_persons(yolo, frame_b)
    assert boxes_a, "No persons detected in frame_a"

    if boxes_b:
        best, pair = float("inf"), (boxes_a[0], boxes_a[0])
        for ba in boxes_a:
            cxa, cya = (ba[0]+ba[2])/2, (ba[1]+ba[3])/2
            for bb in boxes_b:
                cxb, cyb = (bb[0]+bb[2])/2, (bb[1]+bb[3])/2
                d = ((cxa-cxb)**2 + (cya-cyb)**2)**0.5
                if d < best:
                    best, pair = d, (ba, bb)
        box_a, box_b = pair
    else:
        b = boxes_a[0]
        box_a, box_b = b, [b[0]+10, b[1]+5, b[2]+10, b[3]+5]

    tracker = make_tracker(frame_wh=(W, H))
    dist = tracker._box_distance(box_a, box_b)
    ok = dist < 0.05
    status = "PASS" if ok else "FAIL"
    print(f"  box_a={box_a}  box_b={box_b}  distance={dist:.4f}  | {status}")
    print_verdict(TEST_ID, status)

    save_single_test_result(TEST_ID, DATA_OUT, {
        "status": status,
        "box_a": box_a,
        "box_b": box_b,
        "distance": round(dist, 6),
    })


if __name__ == "__main__":
    run()
