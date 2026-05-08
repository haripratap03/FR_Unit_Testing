"""
test_ft04.py  —  FT-04
Test:  OSNetExtractor.extract_batch  (pipeline.py:254)
Check: every real embedding is L2-normalised (||e|| ~= 1.0).

Output: Unit_Test_Data/Face_Tracking/data/test_ft04/
"""

from datetime import datetime

import numpy as np

from _ft_helpers import (
    VIDEO_PATH, load_yolo, load_osnet, detect_persons, sample_frame,
    prepare_data_dir, save_single_test_result, print_verdict,
)

TEST_ID = "FT-04"
DATA_OUT = prepare_data_dir(TEST_ID)


def run():
    print(f"\n{'='*60}\n{TEST_ID} — extract_batch L2 normalisation  "
          f"({datetime.now():%H:%M:%S})\n{'='*60}")

    yolo  = load_yolo()
    osnet = load_osnet()
    frame = sample_frame(VIDEO_PATH, frame_idx=50)

    boxes = detect_persons(yolo, frame)[:5]
    assert boxes, "No person detections in frame 50"

    result = osnet.extract_batch(frame, boxes)
    norms = [float(np.linalg.norm(r)) for r in result if r is not None]
    ok = bool(norms) and all(abs(n - 1.0) < 1e-4 for n in norms)

    valid_embs = [r for r in result if r is not None]
    if valid_embs:
        np.save(str(DATA_OUT / "embeddings.npy"),
                np.array(valid_embs, dtype=np.float32))

    status = "PASS" if ok else "FAIL"
    print(f"  norms = {[round(n,6) for n in norms[:5]]}  | {status}")
    print_verdict(TEST_ID, status)

    save_single_test_result(TEST_ID, DATA_OUT, {
        "status": status,
        "norms": [round(n, 6) for n in norms],
    })


if __name__ == "__main__":
    run()
