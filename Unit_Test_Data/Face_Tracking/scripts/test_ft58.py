"""
test_ft58.py  —  FT-58
Test:  LineCrossDetector.update  (pipeline.py:330)
Check: a person who stays on the same side as last time produces no event
       (update returns None).

Output: Unit_Test_Data/Face_Tracking/data/test_ft58/
"""

from datetime import datetime

from _ft_helpers import (
    extract_person_centroids,
    prepare_data_dir, save_single_test_result, print_verdict,
)

TEST_ID = "FT-58"
DATA_OUT = prepare_data_dir(TEST_ID)


def run():
    import pipeline
    print(f"\n{'='*60}\n{TEST_ID} — update no transition  "
          f"({datetime.now():%H:%M:%S})\n{'='*60}")

    centroids, _ = extract_person_centroids(n_frames=80)
    below = [c for c in centroids if c[1] > 0.5]
    if len(below) >= 2:
        nx, ny = below[1]
        src = "real"
    elif below:
        nx, ny = below[0]
        src = "real (same centroid)"
    else:
        nx, ny = 0.6, 0.8
        src = "synthetic"

    lcd = pipeline.LineCrossDetector({"x": 0, "y": 0.5}, {"x": 1, "y": 0.5}, "below")
    lcd.person_sides[5] = "below"
    result = lcd.update(5, nx, ny)

    ok = result is None
    status = "PASS" if ok else "FAIL"
    print(f"  [{src}] prev='below'  ny={ny:.4f}  -> {result}  | {status}")
    print_verdict(TEST_ID, status)

    save_single_test_result(TEST_ID, DATA_OUT, {
        "status": status,
        "result": result,
        "centroid_source": src,
    })


if __name__ == "__main__":
    run()
