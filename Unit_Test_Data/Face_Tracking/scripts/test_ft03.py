"""
test_ft03.py  —  FT-03
Test:  OSNetExtractor.extract_batch  (pipeline.py:254)
Check: when every crop is undersized, every returned slot is None.

Output: Unit_Test_Data/Face_Tracking/data/test_ft03/
"""

from datetime import datetime

from _ft_helpers import (
    VIDEO_PATH, load_osnet, sample_frame,
    prepare_data_dir, save_single_test_result, print_verdict,
)

TEST_ID = "FT-03"
DATA_OUT = prepare_data_dir(TEST_ID)


def run():
    print(f"\n{'='*60}\n{TEST_ID} — extract_batch all-undersized  "
          f"({datetime.now():%H:%M:%S})\n{'='*60}")

    osnet = load_osnet()
    frame = sample_frame(VIDEO_PATH, frame_idx=50)

    tiny_boxes = [[0, 0, 20, 40], [10, 10, 25, 50]]
    result = osnet.extract_batch(frame, tiny_boxes)

    ok = all(r is None for r in result)
    status = "PASS" if ok else "FAIL"
    print(f"  results all None = {ok}  | {status}")
    print_verdict(TEST_ID, status)

    save_single_test_result(TEST_ID, DATA_OUT, {
        "status": status,
        "all_none": ok,
        "result_count": len(result),
    })


if __name__ == "__main__":
    run()
