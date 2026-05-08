"""
test_ft55.py  —  FT-55
Test:  LineCrossDetector.update  (pipeline.py:330)
Check: the very first observation for a person returns None and the side is
       recorded in person_sides.

Output: Unit_Test_Data/Face_Tracking/data/test_ft55/
"""

from datetime import datetime

from _ft_helpers import (
    extract_person_centroids,
    prepare_data_dir, save_single_test_result, print_verdict,
)

TEST_ID = "FT-55"
DATA_OUT = prepare_data_dir(TEST_ID)


def run():
    import pipeline
    print(f"\n{'='*60}\n{TEST_ID} — update first observation  "
          f"({datetime.now():%H:%M:%S})\n{'='*60}")

    centroids, _ = extract_person_centroids(n_frames=80)
    above = [c for c in centroids if c[1] < 0.5]
    if above:
        nx, ny = above[0]
        src = "real"
    else:
        nx, ny = 0.5, 0.3
        src = "synthetic"

    lcd = pipeline.LineCrossDetector({"x": 0, "y": 0.5}, {"x": 1, "y": 0.5}, "below")
    result = lcd.update(5, nx, ny)
    side_recorded = lcd.person_sides.get(5)

    ok = (result is None) and (side_recorded == "above")
    status = "PASS" if ok else "FAIL"
    print(f"  [{src}] result={result}  person_sides[5]='{side_recorded}'  | {status}")
    print_verdict(TEST_ID, status)

    save_single_test_result(TEST_ID, DATA_OUT, {
        "status": status,
        "result": result,
        "side_recorded": side_recorded,
        "centroid_source": src,
    })


if __name__ == "__main__":
    run()
