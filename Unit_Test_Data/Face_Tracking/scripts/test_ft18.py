"""
test_ft18.py  —  FT-18
Test:  PendingTrack.__init__  (pipeline.py:392)
Check: a freshly constructed PendingTrack has frames_seen=1, bank length 1,
       and last_box equal to the constructor box.

Output: Unit_Test_Data/Face_Tracking/data/test_ft18/
"""

from datetime import datetime

from _ft_helpers import (
    collect_person_detections,
    prepare_data_dir, save_single_test_result, print_verdict,
)

TEST_ID = "FT-18"
DATA_OUT = prepare_data_dir(TEST_ID)


def run():
    import pipeline
    print(f"\n{'='*60}\n{TEST_ID} — PendingTrack constructor  "
          f"({datetime.now():%H:%M:%S})\n{'='*60}")

    embeddings, boxes = collect_person_detections(n=2)
    assert len(embeddings) >= 1, "Need at least 1 detection"

    pend = pipeline.PendingTrack(embeddings[0], boxes[0])
    ok = (pend.frames_seen == 1) and \
         (len(pend.bank) == 1) and \
         (pend.last_box == list(boxes[0]))
    status = "PASS" if ok else "FAIL"
    print(f"  frames_seen={pend.frames_seen}  bank_len={len(pend.bank)}  "
          f"last_box={pend.last_box}  | {status}")
    print_verdict(TEST_ID, status)

    save_single_test_result(TEST_ID, DATA_OUT, {
        "status": status,
        "frames_seen": pend.frames_seen,
        "bank_len": len(pend.bank),
        "last_box": pend.last_box,
    })


if __name__ == "__main__":
    run()
