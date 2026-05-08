"""
test_ft53.py  —  FT-53
Test:  LineCrossDetector.get_side  (pipeline.py:320)
Check: with a diagonal line (y = x) and the point (0.5, 0.3), the interpolated
       line_y at nx=0.5 is 0.5; ny=0.3 is therefore 'above'.

Output: Unit_Test_Data/Face_Tracking/data/test_ft53/
"""

from datetime import datetime

from _ft_helpers import (
    prepare_data_dir, save_single_test_result, print_verdict,
)

TEST_ID = "FT-53"
DATA_OUT = prepare_data_dir(TEST_ID)


def run():
    import pipeline
    print(f"\n{'='*60}\n{TEST_ID} — get_side diagonal line  "
          f"({datetime.now():%H:%M:%S})\n{'='*60}")

    lcd = pipeline.LineCrossDetector({"x": 0, "y": 0}, {"x": 1, "y": 1}, "below")
    side = lcd.get_side(0.5, 0.3)

    ok = side == "above"
    status = "PASS" if ok else "FAIL"
    print(f"  nx=0.5  ny=0.3  line_y=0.5  -> '{side}'  expected='above'  | {status}")
    print_verdict(TEST_ID, status)

    save_single_test_result(TEST_ID, DATA_OUT, {
        "status": status,
        "nx": 0.5,
        "ny": 0.3,
        "interpolated_line_y": 0.5,
        "side": side,
    })


if __name__ == "__main__":
    run()
