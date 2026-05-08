"""
test_ft09.py  —  FT-09
Test:  ActiveTrack.ref  (pipeline.py:360)
Check: recency weighting — the most recently appended embedding has at least
       as high a cosine similarity to ref as the oldest does.

Output: Unit_Test_Data/Face_Tracking/data/test_ft09/
"""

from datetime import datetime

import numpy as np

from _ft_helpers import (
    collect_embeddings, make_active_track,
    prepare_data_dir, save_single_test_result, print_verdict,
)

TEST_ID = "FT-09"
DATA_OUT = prepare_data_dir(TEST_ID)


def run():
    print(f"\n{'='*60}\n{TEST_ID} — ActiveTrack.ref recency weighting  "
          f"({datetime.now():%H:%M:%S})\n{'='*60}")

    embeddings, boxes = collect_embeddings(n=3)
    assert len(embeddings) >= 3, "Need >=3 real embeddings from video"

    e_old, e_mid, e_new = embeddings[0], embeddings[1], embeddings[2]
    track = make_active_track(e_old, box=boxes[0])
    track.bank.append(e_mid.copy())
    track.bank.append(e_new.copy())

    ref = track.ref
    sim_old = float(np.dot(ref, e_old))
    sim_mid = float(np.dot(ref, e_mid))
    sim_new = float(np.dot(ref, e_new))

    ok = sim_new >= sim_old
    status = "PASS" if ok else "FAIL"
    print(f"  cos_sim: old={sim_old:.4f}  mid={sim_mid:.4f}  new={sim_new:.4f}  | {status}")
    print_verdict(TEST_ID, status)

    save_single_test_result(TEST_ID, DATA_OUT, {
        "status": status,
        "sim_old": round(sim_old, 6),
        "sim_mid": round(sim_mid, 6),
        "sim_new": round(sim_new, 6),
    })


if __name__ == "__main__":
    run()
