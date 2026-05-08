"""
test_ft15.py  —  FT-15
Test:  ActiveTrack.serialize / deserialize  (pipeline.py:368)
Check: round-trip fidelity — restored bank embeddings, last_box, and
       lost_seconds match the originals.

Output: Unit_Test_Data/Face_Tracking/data/test_ft15/
"""

import shutil
from datetime import datetime

import numpy as np

from _ft_helpers import (
    collect_embeddings, make_active_track,
    prepare_data_dir, save_single_test_result, print_verdict,
)

TEST_ID = "FT-15"
DATA_OUT = prepare_data_dir(TEST_ID)
TRACKS  = DATA_OUT / "tracks"


def run():
    import pipeline
    print(f"\n{'='*60}\n{TEST_ID} — serialize/deserialize round trip  "
          f"({datetime.now():%H:%M:%S})\n{'='*60}")

    if TRACKS.exists():
        shutil.rmtree(TRACKS)
    TRACKS.mkdir(parents=True)

    embeddings, boxes = collect_embeddings(n=5)
    assert len(embeddings) >= 5, f"Need >=5 embeddings, got {len(embeddings)}"

    LOST_S = 2.5
    box15 = boxes[0]
    track = make_active_track(embeddings[0], box=box15, lost_seconds=LOST_S)
    for e in embeddings[1:5]:
        track.bank.append(e.copy())
    state = track.serialize(str(TRACKS), track_id=15)

    restored = pipeline.ActiveTrack.deserialize(state, str(TRACKS))

    orig = list(track.bank)
    rest = list(restored.bank)
    emb_ok = (len(orig) == len(rest)) and all(
        np.allclose(a, b, atol=1e-5) for a, b in zip(orig, rest))
    box_ok  = np.allclose(restored.last_box, box15, atol=1e-3)
    lost_ok = abs(restored.lost_seconds - LOST_S) < 1e-4

    ok = emb_ok and box_ok and lost_ok
    status = "PASS" if ok else "FAIL"
    print(f"  embeddings_ok={emb_ok}  box_ok={box_ok}  lost_seconds_ok={lost_ok}  | {status}")
    print_verdict(TEST_ID, status)

    save_single_test_result(TEST_ID, DATA_OUT, {
        "status": status,
        "embeddings_ok": emb_ok,
        "box_ok": box_ok,
        "lost_seconds_ok": lost_ok,
    })


if __name__ == "__main__":
    run()
