"""
test_ft20.py  —  FT-20
Test:  PendingTrack.add  (pipeline.py:392)
Check: add() increments frames_seen, updates last_box, and grows the bank.

Output: Unit_Test_Data/Face_Tracking/data/test_ft20/
"""

from datetime import datetime

from _ft_helpers import (
    collect_person_detections,
    prepare_data_dir, save_single_test_result, print_verdict,
)

TEST_ID = "FT-20"
DATA_OUT = prepare_data_dir(TEST_ID)


def run():
    import pipeline
    print(f"\n{'='*60}\n{TEST_ID} — PendingTrack.add  "
          f"({datetime.now():%H:%M:%S})\n{'='*60}")

    embeddings, boxes = collect_person_detections(n=4)
    assert len(embeddings) >= 3, "Need >=3 detections"

    pend = pipeline.PendingTrack(embeddings[0], boxes[0])
    pend.add(embeddings[1], boxes[1])
    pend.add(embeddings[2], boxes[2])

    ok = (pend.frames_seen == 3) and \
         (pend.last_box == list(boxes[2])) and \
         (len(pend.bank) == 3)
    status = "PASS" if ok else "FAIL"
    print(f"  frames_seen={pend.frames_seen}  last_box={pend.last_box}  "
          f"bank_len={len(pend.bank)}  | {status}")
    print_verdict(TEST_ID, status)

    save_single_test_result(TEST_ID, DATA_OUT, {
        "status": status,
        "frames_seen": pend.frames_seen,
        "bank_len": len(pend.bank),
        "last_box_matches_last_input": pend.last_box == list(boxes[2]),
    })


if __name__ == "__main__":
    run()
