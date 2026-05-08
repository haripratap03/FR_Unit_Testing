"""
test_ft22.py  —  FT-22
Test:  PendingTrack  (pipeline.py:392)
Check: ref updates dynamically after add() — appending another embedding moves
       the reference vector.

Output: Unit_Test_Data/Face_Tracking/data/test_ft22/
"""

from datetime import datetime

import numpy as np

from _ft_helpers import (
    collect_person_detections,
    prepare_data_dir, save_single_test_result, print_verdict,
)

TEST_ID = "FT-22"
DATA_OUT = prepare_data_dir(TEST_ID)


def run():
    import pipeline
    print(f"\n{'='*60}\n{TEST_ID} — PendingTrack.ref dynamic update  "
          f"({datetime.now():%H:%M:%S})\n{'='*60}")

    embeddings, boxes = collect_person_detections(n=12)
    assert len(embeddings) >= 4, "Need >=4 detections"

    pend = pipeline.PendingTrack(embeddings[0], boxes[0])
    ref_before = pend.ref.copy()
    target_idx = 10 % len(embeddings)
    pend.add(embeddings[target_idx], boxes[target_idx])
    ref_after = pend.ref

    delta = float(np.linalg.norm(ref_after - ref_before))
    changed = not np.allclose(ref_before, ref_after, atol=1e-5)
    ok = changed
    status = "PASS" if ok else "FAIL"
    print(f"  ||ref_after - ref_before||={delta:.6f}  changed={changed}  | {status}")
    print_verdict(TEST_ID, status)

    save_single_test_result(TEST_ID, DATA_OUT, {
        "status": status,
        "ref_delta": round(delta, 6),
        "changed": changed,
    })


if __name__ == "__main__":
    run()
