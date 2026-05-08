"""
test_ft17.py  —  FT-17
Test:  ActiveTrack.serialize / deserialize  (pipeline.py:368)
Check: every bank entry is dtype float32 after a round-trip.

Output: Unit_Test_Data/Face_Tracking/data/test_ft17/
"""

import shutil
from datetime import datetime

from _ft_helpers import (
    collect_embeddings, make_active_track,
    prepare_data_dir, save_single_test_result, print_verdict,
)

TEST_ID = "FT-17"
DATA_OUT = prepare_data_dir(TEST_ID)
TRACKS  = DATA_OUT / "tracks"


def run():
    import pipeline
    print(f"\n{'='*60}\n{TEST_ID} — bank dtype after deserialize  "
          f"({datetime.now():%H:%M:%S})\n{'='*60}")

    if TRACKS.exists():
        shutil.rmtree(TRACKS)
    TRACKS.mkdir(parents=True)

    embeddings, boxes = collect_embeddings(n=3)
    assert len(embeddings) >= 3, "Need >=3 embeddings"

    track = make_active_track(embeddings[0], box=boxes[0])
    for e in embeddings[1:3]:
        track.bank.append(e.copy())
    state = track.serialize(str(TRACKS), track_id=17)
    restored = pipeline.ActiveTrack.deserialize(state, str(TRACKS))

    dtypes = [str(list(restored.bank)[i].dtype) for i in range(len(restored.bank))]
    ok = all(d == "float32" for d in dtypes)
    status = "PASS" if ok else "FAIL"
    print(f"  dtypes={dtypes}  all_float32={ok}  | {status}")
    print_verdict(TEST_ID, status)

    save_single_test_result(TEST_ID, DATA_OUT, {
        "status": status,
        "dtypes": dtypes,
        "all_float32": ok,
    })


if __name__ == "__main__":
    run()
