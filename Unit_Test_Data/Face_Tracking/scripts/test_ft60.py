"""
test_ft60.py  —  FT-60
Test:  LineCrossDetector.cleanup  (pipeline.py:340)
Check: cleanup(99) on {3:'above'} raises no exception and leaves the dict
       unchanged.

Output: Unit_Test_Data/Face_Tracking/data/test_ft60/
"""

from datetime import datetime

from _ft_helpers import (
    prepare_data_dir, save_single_test_result, print_verdict,
)

TEST_ID = "FT-60"
DATA_OUT = prepare_data_dir(TEST_ID)


def run():
    import pipeline
    print(f"\n{'='*60}\n{TEST_ID} — cleanup unknown id  "
          f"({datetime.now():%H:%M:%S})\n{'='*60}")

    lcd = pipeline.LineCrossDetector({"x": 0, "y": 0.5}, {"x": 1, "y": 0.5}, "below")
    lcd.person_sides = {3: "above"}
    before = dict(lcd.person_sides)

    exception_raised = False
    try:
        lcd.cleanup(99)
    except Exception as exc:
        exception_raised = True
        print(f"  EXCEPTION: {exc}")

    after = dict(lcd.person_sides)
    ok = (not exception_raised) and (lcd.person_sides == {3: "above"})
    status = "PASS" if ok else "FAIL"
    print(f"  before={before}  after={after}  exception={exception_raised}  | {status}")
    print_verdict(TEST_ID, status)

    save_single_test_result(TEST_ID, DATA_OUT, {
        "status": status,
        "before": before,
        "after": after,
        "unchanged": lcd.person_sides == {3: "above"},
        "exception_raised": exception_raised,
    })


if __name__ == "__main__":
    run()
