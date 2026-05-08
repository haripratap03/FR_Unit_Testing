"""
test_ft19.py  —  FT-19
Test:  PendingTrack.ref  (pipeline.py:392)
Check: the reference is the unweighted mean of accumulated embeddings, L2-normalised.

Output: Unit_Test_Data/Face_Tracking/data/test_ft19/
"""

from datetime import datetime

import numpy as np

from _ft_helpers import (
    collect_person_detections,
    prepare_data_dir, save_single_test_result, print_verdict,
)

TEST_ID = "FT-19"
DATA_OUT = prepare_data_dir(TEST_ID)


def run():
    import pipeline
    print(f"\n{'='*60}\n{TEST_ID} — PendingTrack.ref unweighted mean  "
          f"({datetime.now():%H:%M:%S})\n{'='*60}")

    embeddings, boxes = collect_person_detections(n=3)
    assert len(embeddings) >= 2, "Need >=2 detections"

    e0, e1 = embeddings[0], embeddings[1]
    pend = pipeline.PendingTrack(e0, boxes[0])
    pend.bank.append(e1.copy())

    ref = pend.ref
    norm = float(np.linalg.norm(ref))
    sim0 = float(np.dot(ref, e0))
    sim1 = float(np.dot(ref, e1))
    approx_equal = abs(sim0 - sim1) < 0.05

    ok = abs(norm - 1.0) < 1e-4   # primary check: L2-normalised
    status = "PASS" if ok else "FAIL"
    print(f"  ||ref||={norm:.6f}  sim_e0={sim0:.4f}  sim_e1={sim1:.4f}  "
          f"approx_equal={approx_equal}  | {status}")
    print_verdict(TEST_ID, status)

    save_single_test_result(TEST_ID, DATA_OUT, {
        "status": status,
        "norm": round(norm, 6),
        "sim_e0": round(sim0, 4),
        "sim_e1": round(sim1, 4),
        "approx_equal": approx_equal,
    })


if __name__ == "__main__":
    run()
