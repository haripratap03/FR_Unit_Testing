"""
test_ft14.py  —  FT-14
Test:  ActiveTrack.serialize  (pipeline.py:368)
Check: serialize writes a zero-padded track_NNNN_bank.npy and the returned
       state dict's "bank_file" entry matches that filename.

Output: Unit_Test_Data/Face_Tracking/data/test_ft14/
"""

import shutil
from datetime import datetime

from _ft_helpers import (
    collect_embeddings, make_active_track,
    prepare_data_dir, save_single_test_result, print_verdict,
)

TEST_ID = "FT-14"
DATA_OUT = prepare_data_dir(TEST_ID)
TRACKS  = DATA_OUT / "tracks"


def run():
    print(f"\n{'='*60}\n{TEST_ID} — serialize filename  "
          f"({datetime.now():%H:%M:%S})\n{'='*60}")

    if TRACKS.exists():
        shutil.rmtree(TRACKS)
    TRACKS.mkdir(parents=True)

    embeddings, boxes = collect_embeddings(n=3)
    assert len(embeddings) >= 1, "Need at least 1 real embedding"

    track = make_active_track(embeddings[0], box=boxes[0])
    for e in embeddings[1:3]:
        track.bank.append(e.copy())

    state = track.serialize(str(TRACKS), track_id=7)
    expected = "track_0007_bank.npy"
    file_exists = (TRACKS / expected).exists()
    key_match = state["bank_file"] == expected
    ok = file_exists and key_match
    status = "PASS" if ok else "FAIL"
    print(f"  bank_file={state['bank_file']}  file_exists={file_exists}  | {status}")
    print_verdict(TEST_ID, status)

    save_single_test_result(TEST_ID, DATA_OUT, {
        "status": status,
        "bank_file": state["bank_file"],
        "expected": expected,
        "file_exists": file_exists,
    })


if __name__ == "__main__":
    run()
