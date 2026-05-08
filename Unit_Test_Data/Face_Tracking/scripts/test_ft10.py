"""
test_ft10.py  —  FT-10
Test:  ActiveTrack.ref  (pipeline.py:360)
Check: with a full bank (10 embeddings), the reference vector is L2-normalised.

Output: Unit_Test_Data/Face_Tracking/data/test_ft10/
"""

from datetime import datetime

import numpy as np

from _ft_helpers import (
    collect_embeddings, make_active_track,
    prepare_data_dir, save_single_test_result, print_verdict,
)

TEST_ID = "FT-10"
DATA_OUT = prepare_data_dir(TEST_ID)


def run():
    print(f"\n{'='*60}\n{TEST_ID} — ActiveTrack.ref full-bank normalisation  "
          f"({datetime.now():%H:%M:%S})\n{'='*60}")

    import pipeline
    BANK_SIZE = pipeline.BANK_SIZE

    embeddings, boxes = collect_embeddings(n=BANK_SIZE)
    assert len(embeddings) >= 3, "Need >=3 real embeddings"
    n_use = min(BANK_SIZE, len(embeddings))

    track = make_active_track(embeddings[0], box=boxes[0])
    for e in embeddings[1:n_use]:
        track.bank.append(e.copy())

    ref = track.ref
    norm = float(np.linalg.norm(ref))
    ok = abs(norm - 1.0) < 1e-4
    status = "PASS" if ok else "FAIL"
    print(f"  bank_size={len(track.bank)}  ||ref||={norm:.6f}  | {status}")
    print_verdict(TEST_ID, status)

    np.save(str(DATA_OUT / "ref.npy"), ref)
    save_single_test_result(TEST_ID, DATA_OUT, {
        "status": status,
        "bank_size": len(track.bank),
        "norm": round(norm, 6),
    })


if __name__ == "__main__":
    run()
