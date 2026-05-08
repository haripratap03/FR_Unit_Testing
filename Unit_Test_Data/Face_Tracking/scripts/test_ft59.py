"""
test_ft59.py  —  FT-59
Test:  LineCrossDetector.cleanup  (pipeline.py:340)
Check: cleanup(3) on {3:'above', 7:'below'} leaves {7:'below'}.

Output: Unit_Test_Data/Face_Tracking/data/test_ft59/
"""

from datetime import datetime

from _ft_helpers import (
    prepare_data_dir, save_single_test_result, print_verdict,
)

TEST_ID = "FT-59"
DATA_OUT = prepare_data_dir(TEST_ID)


def run():
    import pipeline
    print(f"\n{'='*60}\n{TEST_ID} — cleanup existing entry  "
          f"({datetime.now():%H:%M:%S})\n{'='*60}")

    lcd = pipeline.LineCrossDetector({"x": 0, "y": 0.5}, {"x": 1, "y": 0.5}, "below")
    lcd.person_sides = {3: "above", 7: "below"}
    before = dict(lcd.person_sides)

    lcd.cleanup(3)
    after = dict(lcd.person_sides)

    ok = (3 not in lcd.person_sides) and (lcd.person_sides == {7: "below"})
    status = "PASS" if ok else "FAIL"
    print(f"  before={before}  after={after}  | {status}")
    print_verdict(TEST_ID, status)

    save_single_test_result(TEST_ID, DATA_OUT, {
        "status": status,
        "before": before,
        "after": after,
        "3_removed": 3 not in lcd.person_sides,
        "7_intact": lcd.person_sides.get(7) == "below",
    })


if __name__ == "__main__":
    run()
