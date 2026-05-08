"""
test_ft02.py  —  FT-02
Test:  OSNetExtractor.extract_batch  (pipeline.py:254)
Check: undersized crop returns None at the correct index; valid crop returns 512-d.

Output: Unit_Test_Data/Face_Tracking/data/test_ft02/
"""

from datetime import datetime

from _ft_helpers import (
    VIDEO_PATH, load_yolo, load_osnet, detect_persons, sample_frame,
    prepare_data_dir, save_single_test_result, print_verdict,
)

TEST_ID = "FT-02"
DATA_OUT = prepare_data_dir(TEST_ID)


def run():
    print(f"\n{'='*60}\n{TEST_ID} — OSNetExtractor.extract_batch (mixed sizes)  "
          f"({datetime.now():%H:%M:%S})\n{'='*60}")

    yolo  = load_yolo()
    osnet = load_osnet()
    frame = sample_frame(VIDEO_PATH, frame_idx=50)

    boxes_full = detect_persons(yolo, frame)
    assert boxes_full, "No person detections in frame 50"

    tiny_box  = [0, 0, 25, 55]   # 25x55 — below 32x64 threshold
    valid_box = boxes_full[0]
    mixed = [tiny_box, valid_box]

    result = osnet.extract_batch(frame, mixed)
    ok = (result[0] is None) and (result[1] is not None) and (result[1].shape == (512,))

    status = "PASS" if ok else "FAIL"
    print(f"  result[0] is None={result[0] is None}  "
          f"result[1].shape={getattr(result[1],'shape','N/A')}  | {status}")
    print_verdict(TEST_ID, status)

    save_single_test_result(TEST_ID, DATA_OUT, {
        "status": status,
        "index0_is_none": result[0] is None,
        "index1_shape": str(getattr(result[1], "shape", None)),
    })


if __name__ == "__main__":
    run()
