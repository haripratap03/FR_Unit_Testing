"""
test_ft56.py  —  FT-56
Test:  LineCrossDetector.update  (pipeline.py:330)
Check: when a person previously recorded as 'above' is now seen 'below'
       (with in_side='below'), update returns 'enter'.

Output: Unit_Test_Data/Face_Tracking/data/test_ft56/
"""

from datetime import datetime

from _ft_helpers import (
    extract_person_centroids,
    prepare_data_dir, save_single_test_result, print_verdict,
)

TEST_ID = "FT-56"
DATA_OUT = prepare_data_dir(TEST_ID)


def run():
    import pipeline
    print(f"\n{'='*60}\n{TEST_ID} — update enter event  "
          f"({datetime.now():%H:%M:%S})\n{'='*60}")

    centroids, _ = extract_person_centroids(n_frames=80)
    below = [c for c in centroids if c[1] > 0.5]
    if below:
        nx, ny = below[0]
        src = "real"
    else:
        nx, ny = 0.5, 0.7
        src = "synthetic"

    lcd = pipeline.LineCrossDetector({"x": 0, "y": 0.5}, {"x": 1, "y": 0.5}, "below")
    lcd.person_sides[5] = "above"
    result = lcd.update(5, nx, ny)

    ok = result == "enter"
    status = "PASS" if ok else "FAIL"
    print(f"  [{src}] prev='above'  ny={ny:.4f}  -> '{result}'  | {status}")
    print_verdict(TEST_ID, status)

    save_single_test_result(TEST_ID, DATA_OUT, {
        "status": status,
        "result": result,
        "centroid_source": src,
    })


if __name__ == "__main__":
    run()
