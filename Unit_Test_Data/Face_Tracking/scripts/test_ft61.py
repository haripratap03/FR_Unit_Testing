"""
test_ft61.py  —  FT-61
Test:  LineCrossDetector.get_state / restore_state  (pipeline.py:344)
Check: get_state -> restore_state preserves person_sides exactly.

Output: Unit_Test_Data/Face_Tracking/data/test_ft61/
"""

from datetime import datetime

from _ft_helpers import (
    prepare_data_dir, save_single_test_result, print_verdict,
)

TEST_ID = "FT-61"
DATA_OUT = prepare_data_dir(TEST_ID)


def run():
    import pipeline
    print(f"\n{'='*60}\n{TEST_ID} — get_state/restore_state round trip  "
          f"({datetime.now():%H:%M:%S})\n{'='*60}")

    src = "synthetic"
    person_sides = {1: "above", 5: "below", 12: "above"}

    lcd = pipeline.LineCrossDetector({"x": 0, "y": 0.5}, {"x": 1, "y": 0.5}, "below")
    lcd.person_sides = dict(person_sides)
    state = lcd.get_state()

    restored = pipeline.LineCrossDetector({"x": 0, "y": 0.5}, {"x": 1, "y": 0.5}, "below")
    restored.restore_state(state)

    ok = restored.person_sides == lcd.person_sides
    status = "PASS" if ok else "FAIL"
    print(f"  original={lcd.person_sides}  restored={restored.person_sides}  | {status}")
    print_verdict(TEST_ID, status)

    save_single_test_result(TEST_ID, DATA_OUT, {
        "status": status,
        "original": dict(lcd.person_sides),
        "restored": dict(restored.person_sides),
        "match": ok,
        "source": src,
    })


if __name__ == "__main__":
    run()
