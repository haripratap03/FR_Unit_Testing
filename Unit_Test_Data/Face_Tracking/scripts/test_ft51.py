"""
test_ft51.py  —  FT-51
Test:  LineCrossDetector.get_side  (pipeline.py:320)
Check: a centroid above the horizontal mid-line (ny < 0.5) returns 'above'.

Output: Unit_Test_Data/Face_Tracking/data/test_ft51/
"""

from datetime import datetime

from _ft_helpers import (
    extract_person_centroids,
    prepare_data_dir, save_single_test_result, print_verdict,
)

TEST_ID = "FT-51"
DATA_OUT = prepare_data_dir(TEST_ID)


def run():
    import pipeline
    print(f"\n{'='*60}\n{TEST_ID} — get_side above midline  "
          f"({datetime.now():%H:%M:%S})\n{'='*60}")

    centroids, _ = extract_person_centroids(n_frames=80)
    above = [(cx, cy) for cx, cy in centroids if cy < 0.5]

    if above:
        nx, ny = above[0]
        src = "real"
    else:
        nx, ny = 0.5, 0.3
        src = "synthetic"

    lcd = pipeline.LineCrossDetector({"x": 0, "y": 0.5}, {"x": 1, "y": 0.5}, "below")
    side = lcd.get_side(nx, ny)

    ok = side == "above"
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
