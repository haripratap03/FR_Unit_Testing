"""
test_ft11.py  —  FT-11
Test:  ActiveTrack.update_embedding  (pipeline.py:365)
Check: bank grows up to BANK_SIZE then stops growing as more embeddings arrive.

Output: Unit_Test_Data/Face_Tracking/data/test_ft11/
"""

from datetime import datetime

from _ft_helpers import (
    collect_embeddings, make_active_track,
    prepare_data_dir, save_single_test_result, print_verdict,
)

TEST_ID = "FT-11"
DATA_OUT = prepare_data_dir(TEST_ID)


def run():
    import pipeline
    BANK_SIZE = pipeline.BANK_SIZE
    print(f"\n{'='*60}\n{TEST_ID} — update_embedding bank growth (BANK_SIZE={BANK_SIZE})  "
          f"({datetime.now():%H:%M:%S})\n{'='*60}")

    embeddings, boxes = collect_embeddings(n=BANK_SIZE + 5)
    assert len(embeddings) >= BANK_SIZE + 3, \
        f"Need >={BANK_SIZE+3} embeddings, got {len(embeddings)}"

    track = make_active_track(embeddings[0], box=boxes[0], bank_maxlen=BANK_SIZE)
    sizes = [1]
    for i in range(1, BANK_SIZE + 3):
        track.update_embedding(embeddings[i % len(embeddings)])
        sizes.append(len(track.bank))

    ok = sizes[BANK_SIZE - 1] == BANK_SIZE and sizes[BANK_SIZE] == BANK_SIZE
    status = "PASS" if ok else "FAIL"
    print(f"  sizes={sizes}  | {status}")
    print_verdict(TEST_ID, status)

    save_single_test_result(TEST_ID, DATA_OUT, {
        "status": status,
        "bank_size": BANK_SIZE,
        "sizes_log": sizes,
    })


if __name__ == "__main__":
    run()
