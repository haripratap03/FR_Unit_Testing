"""
test_ft13.py  —  FT-13
Test:  ActiveTrack.update_embedding  (pipeline.py:365)
Check: ref is recomputed (changes) after a sufficiently distant embedding is
       added.  We pick the embedding least similar to the current ref to make
       the change visible above floating-point noise.

Output: Unit_Test_Data/Face_Tracking/data/test_ft13/
"""

from datetime import datetime

import numpy as np

from _ft_helpers import (
    collect_embeddings, make_active_track,
    prepare_data_dir, save_single_test_result, print_verdict,
)

TEST_ID = "FT-13"
DATA_OUT = prepare_data_dir(TEST_ID)


def run():
    print(f"\n{'='*60}\n{TEST_ID} — update_embedding ref recompute  "
          f"({datetime.now():%H:%M:%S})\n{'='*60}")

    embeddings, boxes = collect_embeddings(n=10)
    assert len(embeddings) >= 4, f"Need >=4 embeddings, got {len(embeddings)}"

    track = make_active_track(embeddings[0], box=boxes[0])
    track.bank.append(embeddings[1].copy())
    ref_before = track.ref.copy()

    sims = [float(np.dot(ref_before, e)) for e in embeddings]
    distant_idx = int(np.argmin(sims))
    track.update_embedding(embeddings[distant_idx])
    ref_after = track.ref

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
