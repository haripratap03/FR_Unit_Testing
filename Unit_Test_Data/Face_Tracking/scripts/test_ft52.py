"""
test_ft52.py  —  FT-52
Test:  LineCrossDetector.get_side  (pipeline.py:320)
Check: a centroid below the horizontal mid-line (ny > 0.5) returns 'below'.

Output: Unit_Test_Data/Face_Tracking/data/test_ft52/
"""

from datetime import datetime

from _ft_helpers import (
    extract_person_centroids,
    prepare_data_dir, save_single_test_result, print_verdict,
)

TEST_ID = "FT-52"
DATA_OUT = prepare_data_dir(TEST_ID)


def run():
    import pipeline
    print(f"\n{'='*60}\n{TEST_ID} — get_side below midline  "
          f"({datetime.now():%H:%M:%S})\n{'='*60}")

    centroids, _ = extract_person_centroids(n_frames=80)
    below = [(cx, cy) for cx, cy in centroids if cy > 0.5]

    if below:
        nx, ny = below[0]
        src = "real"
    else:
        nx, ny = 0.5, 0.7
        src = "synthetic"

    lcd = pipeline.LineCrossDetector({"x": 0, "y": 0.5}, {"x": 1, "y": 0.5}, "below")
    side = lcd.get_side(nx, ny)

    ok = side == "below"
    status = "PASS" if ok else "FAIL"
    print(f"  [{src}] nx={nx:.4f}  ny={ny:.4f}  -> '{side}'  | {status}")
    print_verdict(TEST_ID, status)

    save_single_test_result(TEST_ID, DATA_OUT, {
        "status": status,
        "nx": round(nx, 4),
        "ny": round(ny, 4),
        "side": side,
        "centroid_source": src,
    })


if __name__ == "__main__":
    run()
