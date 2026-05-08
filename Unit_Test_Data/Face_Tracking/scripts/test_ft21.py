"""
test_ft21.py  —  FT-21
Test:  PendingTrack  (pipeline.py:392)
Check: the pending bank is unbounded — 16 sequential adds keep all 16 entries.

Output: Unit_Test_Data/Face_Tracking/data/test_ft21/
"""

from datetime import datetime

from _ft_helpers import (
    collect_person_detections,
    prepare_data_dir, save_single_test_result, print_verdict,
)

TEST_ID = "FT-21"
DATA_OUT = prepare_data_dir(TEST_ID)


def run():
    import pipeline
    print(f"\n{'='*60}\n{TEST_ID} — PendingTrack unbounded bank  "
          f"({datetime.now():%H:%M:%S})\n{'='*60}")

    embeddings, boxes = collect_person_detections(n=16)
    assert len(embeddings) >= 4, "Need >=4 detections"
    n = len(embeddings)

    pend = pipeline.PendingTrack(embeddings[0], boxes[0])
    for i in range(1, 16):
        pend.add(embeddings[i % n], boxes[i % n])

    ok = len(pend.bank) == 16
    status = "PASS" if ok else "FAIL"
    print(f"  bank_len={len(pend.bank)}  expected=16  | {status}")
    print_verdict(TEST_ID, status)

    save_single_test_result(TEST_ID, DATA_OUT, {
        "status": status,
        "bank_len": len(pend.bank),
    })


if __name__ == "__main__":
    run()
