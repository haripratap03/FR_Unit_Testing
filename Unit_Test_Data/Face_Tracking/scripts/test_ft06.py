"""
test_ft06.py  —  FT-06
Test:  OSNetExtractor.extract  (pipeline.py:240)
Check: a single crop below MIN_CROP_W=32 / MIN_CROP_H=64 returns None.

Output: Unit_Test_Data/Face_Tracking/data/test_ft06/
"""

from datetime import datetime

from _ft_helpers import (
    VIDEO_PATH, load_osnet, sample_frame,
    prepare_data_dir, save_single_test_result, print_verdict,
)

TEST_ID = "FT-06"
DATA_OUT = prepare_data_dir(TEST_ID)


def run():
    print(f"\n{'='*60}\n{TEST_ID} — extract on undersized crop  "
          f"({datetime.now():%H:%M:%S})\n{'='*60}")

    osnet = load_osnet()
    frame = sample_frame(VIDEO_PATH, frame_idx=20)

    tiny_box = [100, 100, 130, 150]      # 30 wide x 50 tall, both below threshold
    result = osnet.extract(frame, tiny_box)

    ok = result is None
    status = "PASS" if ok else "FAIL"
    crop_w = tiny_box[2] - tiny_box[0]
    crop_h = tiny_box[3] - tiny_box[1]
    print(f"  box={tiny_box}  crop={crop_w}x{crop_h}  result is None={ok}  | {status}")

    # informational: a valid box returns 512-d
    valid_box = [200, 200, 350, 500]
    valid_res = osnet.extract(frame, valid_box)
    print(f"  valid box {valid_box}: shape={getattr(valid_res, 'shape', 'N/A')}")

    print_verdict(TEST_ID, status)

    save_single_test_result(TEST_ID, DATA_OUT, {
        "status": status,
        "box": tiny_box,
        "crop_wh": [crop_w, crop_h],
        "result_is_none": ok,
        "valid_box_shape": str(getattr(valid_res, "shape", None)),
    })


if __name__ == "__main__":
    run()
