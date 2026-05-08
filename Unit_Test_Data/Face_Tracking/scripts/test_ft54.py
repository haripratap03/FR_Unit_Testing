"""
test_ft54.py  —  FT-54
Test:  LineCrossDetector.get_side  (pipeline.py:320)
Check: a vertical line (xa == xb == 0.5) does not raise ZeroDivisionError;
       a point above the line range (ny=0.1) returns 'above'.

Output: Unit_Test_Data/Face_Tracking/data/test_ft54/
"""

from datetime import datetime

from _ft_helpers import (
    prepare_data_dir, save_single_test_result, print_verdict,
)

TEST_ID = "FT-54"
DATA_OUT = prepare_data_dir(TEST_ID)


def run():
    import pipeline
    print(f"\n{'='*60}\n{TEST_ID} — get_side vertical line  "
          f"({datetime.now():%H:%M:%S})\n{'='*60}")

    lcd = pipeline.LineCrossDetector({"x": 0.5, "y": 0.3},
                                     {"x": 0.5, "y": 0.8}, "below")
    exception_raised = False
    side = None
    try:
        side = lcd.get_side(0.5, 0.1)
    except Exception as exc:
        exception_raised = True
        print(f"  EXCEPTION ({type(exc).__name__}): {exc}")

    ok = (side == "above") and not exception_raised
    status = "PASS" if ok else "FAIL"
    print(f"  vertical line  ny=0.1 -> '{side}'  exception={exception_raised}  | {status}")
    print_verdict(TEST_ID, status)

    save_single_test_result(TEST_ID, DATA_OUT, {
        "status": status,
        "side": side,
        "exception_raised": exception_raised,
    })


if __name__ == "__main__":
    run()
