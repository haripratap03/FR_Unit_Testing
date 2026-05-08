"""
test_ft12.py  —  FT-12
Test:  ActiveTrack.update_embedding  (pipeline.py:365)
Check: when the bank is full, the oldest entry is evicted on the next add.

Output: Unit_Test_Data/Face_Tracking/data/test_ft12/
"""

from datetime import datetime

import numpy as np

from _ft_helpers import (
    collect_embeddings, make_active_track,
    prepare_data_dir, save_single_test_result, print_verdict,
)

TEST_ID = "FT-12"
DATA_OUT = prepare_data_dir(TEST_ID)


def run():
    print(f"\n{'='*60}\n{TEST_ID} — update_embedding eviction  "
          f"({datetime.now():%H:%M:%S})\n{'='*60}")

    embeddings, boxes = collect_embeddings(n=4)
    assert len(embeddings) >= 4, f"Need >=4 embeddings, got {len(embeddings)}"

    e1, e2, e3, e4 = embeddings[:4]
    track = make_active_track(e1, box=boxes[0], bank_maxlen=3)
    track.bank.append(e2.copy())
    track.bank.append(e3.copy())
    # bank now [e1, e2, e3] — full
    track.update_embedding(e4)
    # expect [e2, e3, e4] — e1 evicted

    bank = list(track.bank)
    e1_gone = not any(np.allclose(b, e1, atol=1e-5) for b in bank)
    e4_present = any(np.allclose(b, e4, atol=1e-5) for b in bank)
    ok = (len(bank) == 3) and e1_gone and e4_present

    status = "PASS" if ok else "FAIL"
    print(f"  bank_len={len(bank)}  e1_gone={e1_gone}  e4_present={e4_present}  | {status}")
    print_verdict(TEST_ID, status)

    save_single_test_result(TEST_ID, DATA_OUT, {
        "status": status,
        "bank_len": len(bank),
        "e1_evicted": e1_gone,
        "e4_present": e4_present,
    })


if __name__ == "__main__":
    run()
