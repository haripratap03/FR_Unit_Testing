"""
test_ft46.py  —  FT-46
Test:  PersonTracker.serialize_state / restore_state  (pipeline.py:560)
Check: full round-trip — next_id, active track keys, and bank embeddings all
       survive a serialize -> restore cycle.

Output: Unit_Test_Data/Face_Tracking/data/test_ft46/
"""

import shutil
from datetime import datetime

import numpy as np

from _ft_helpers import (
    collect_embeddings, make_tracker, make_active_track,
    prepare_data_dir, save_single_test_result, print_verdict,
)

TEST_ID = "FT-46"
DATA_OUT = prepare_data_dir(TEST_ID)
TRACKS  = DATA_OUT / "tracks"


def run():
    print(f"\n{'='*60}\n{TEST_ID} — serialize_state / restore_state round trip  "
          f"({datetime.now():%H:%M:%S})\n{'='*60}")

    if TRACKS.exists():
        shutil.rmtree(TRACKS)
    TRACKS.mkdir(parents=True)

    embeddings, boxes = collect_embeddings(n=5)
    assert len(embeddings) >= 4, "Need >=4 real embeddings"

    tracker = make_tracker()
    for tid, start in [(3, 0), (7, 2)]:
        tr = make_active_track(embeddings[start], boxes[start])
        end = min(start + 3, len(embeddings))
        for i in range(start + 1, end):
            tr.bank.append(embeddings[i].copy())
        tracker.active[tid] = tr
    tracker.next_id = 10
    tracker.pending_next = 22

    state = tracker.serialize_state(str(TRACKS))

    restored = make_tracker()
    restored.restore_state(state, str(TRACKS))

    next_id_ok = (restored.next_id == 10)
    keys_ok    = (set(restored.active.keys()) == {3, 7})
    emb_ok = True
    for tid in [3, 7]:
        if tid not in restored.active:
            emb_ok = False
            continue
        for a, b in zip(tracker.active[tid].bank, restored.active[tid].bank):
            if not np.allclose(a, b, atol=1e-5):
                emb_ok = False

    ok = next_id_ok and keys_ok and emb_ok
    status = "PASS" if ok else "FAIL"
    print(f"  next_id_ok={next_id_ok}  keys_ok={keys_ok}  embeddings_ok={emb_ok}  | {status}")
    print_verdict(TEST_ID, status)

    save_single_test_result(TEST_ID, DATA_OUT, {
        "status": status,
        "next_id_ok": next_id_ok,
        "keys_ok": keys_ok,
        "embeddings_ok": emb_ok,
    })


if __name__ == "__main__":
    run()
