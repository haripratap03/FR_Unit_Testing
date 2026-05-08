"""
test_ft01.py  —  FT-01
Test:  OSNetExtractor.extract_batch  (pipeline.py:254)
Check: list length matches box count; each embedding shape (512,).

Output: Unit_Test_Data/Face_Tracking/data/test_ft01/
"""

from datetime import datetime

import cv2
import numpy as np

from _ft_helpers import (
    VIDEO_PATH, load_yolo, load_osnet, detect_persons, sample_frame,
    prepare_data_dir, save_single_test_result, print_verdict,
)

TEST_ID = "FT-01"
DATA_OUT = prepare_data_dir(TEST_ID)


def run():
    print(f"\n{'='*60}\n{TEST_ID} — OSNetExtractor.extract_batch  "
          f"({datetime.now():%H:%M:%S})\n{'='*60}")

    yolo  = load_yolo()
    osnet = load_osnet()

    frame = sample_frame(VIDEO_PATH, frame_idx=50)
    boxes_full = detect_persons(yolo, frame)
    print(f"[SETUP] persons detected = {len(boxes_full)}")
    assert len(boxes_full) >= 2, "Need >=2 person detections in frame 50"

    boxes_3 = boxes_full[:3] if len(boxes_full) >= 3 else boxes_full

    result = osnet.extract_batch(frame, boxes_3)
    ok = len(result) == len(boxes_3)
    for r in result:
        if r is not None:
            ok &= r.shape == (512,)

    # save annotated frame for reference
    vis = frame.copy()
    for i, b in enumerate(boxes_3):
        x1, y1, x2, y2 = b
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(vis, f"box{i}", (x1, max(y1-8, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.imwrite(str(DATA_OUT / "frame.jpg"), vis)

    status = "PASS" if ok else "FAIL"
    print(f"  len(result)={len(result)}  expected={len(boxes_3)}  | {status}")
    print_verdict(TEST_ID, status)

    save_single_test_result(TEST_ID, DATA_OUT, {
        "status": status,
        "len_result": len(result),
        "len_boxes": len(boxes_3),
    })


if __name__ == "__main__":
    run()
