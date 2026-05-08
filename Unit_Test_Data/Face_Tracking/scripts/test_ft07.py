"""
test_ft07.py  —  FT-07
Test:  OSNetExtractor.extract  (pipeline.py:240)
Check: a bbox that extends past the frame edge does not raise; a bbox fully
       outside the frame also does not raise.

Output: Unit_Test_Data/Face_Tracking/data/test_ft07/
"""

from datetime import datetime

from _ft_helpers import (
    VIDEO_PATH, load_osnet, sample_frame,
    prepare_data_dir, save_single_test_result, print_verdict,
)

TEST_ID = "FT-07"
DATA_OUT = prepare_data_dir(TEST_ID)


def run():
    print(f"\n{'='*60}\n{TEST_ID} — extract with out-of-bounds bbox  "
          f"({datetime.now():%H:%M:%S})\n{'='*60}")

    osnet = load_osnet()
    frame = sample_frame(VIDEO_PATH, frame_idx=20)
    h, w = frame.shape[:2]

    oob_box = [w - 50, h - 80, w + 200, h + 400]
    exception_raised = False
    result = None
    try:
        result = osnet.extract(frame, oob_box)
    except Exception as exc:
        exception_raised = True
        print(f"  EXCEPTION: {exc}")

    outside_box = [w + 100, h + 100, w + 300, h + 400]
    fully_outside_no_exc = True
    try:
        osnet.extract(frame, outside_box)
    except Exception as exc:
        fully_outside_no_exc = False
        print(f"  fully-outside raised: {exc}")

    ok = (not exception_raised) and fully_outside_no_exc
    status = "PASS" if ok else "FAIL"
    print(f"  oob_box ok = {not exception_raised}  "
          f"fully_outside ok = {fully_outside_no_exc}  | {status}")
    print_verdict(TEST_ID, status)

    save_single_test_result(TEST_ID, DATA_OUT, {
        "status": status,
        "frame_wh": [w, h],
        "oob_box": oob_box,
        "exception_raised": exception_raised,
        "result_type": type(result).__name__,
        "fully_outside_no_exception": fully_outside_no_exc,
    })


if __name__ == "__main__":
    run()
