"""
test_ft08.py  —  FT-08
Test:  ActiveTrack.ref  (pipeline.py:360)
Check: with a single embedding in the bank, ref ~= that embedding and ||ref|| ~= 1.

Output: Unit_Test_Data/Face_Tracking/data/test_ft08/
"""

from datetime import datetime

import numpy as np

from _ft_helpers import (
    collect_embeddings, make_active_track,
    prepare_data_dir, save_single_test_result, print_verdict,
)

TEST_ID = "FT-08"
DATA_OUT = prepare_data_dir(TEST_ID)


def run():
    print(f"\n{'='*60}\n{TEST_ID} — ActiveTrack.ref single-embedding  "
          f"({datetime.now():%H:%M:%S})\n{'='*60}")

    embeddings, boxes = collect_embeddings(n=1)
    assert len(embeddings) >= 1, "Need >=1 real embedding from video"
    e1 = embeddings[0]

    track = make_active_track(e1, box=boxes[0])
    ref = track.ref
    norm = float(np.linalg.norm(ref))
    cos_sim = float(np.dot(ref, e1))

    ok = (abs(norm - 1.0) < 1e-4) and (cos_sim > 0.9999)
    status = "PASS" if ok else "FAIL"
    print(f"  ||ref||={norm:.6f}  cos_sim(ref,e1)={cos_sim:.6f}  | {status}")
    print_verdict(TEST_ID, status)

    save_single_test_result(TEST_ID, DATA_OUT, {
        "status": status,
        "norm": round(norm, 6),
        "cosine_sim_to_e1": round(cos_sim, 6),
    })


if __name__ == "__main__":
    run()
