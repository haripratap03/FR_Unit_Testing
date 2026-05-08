"""
test_ft05.py  —  FT-05
Test:  OSNetExtractor.extract_batch  (pipeline.py:254)
Check: when the underlying batch call raises, the extractor falls back to
       processing crops one at a time and still returns a valid list.

Output: Unit_Test_Data/Face_Tracking/data/test_ft05/
"""

from datetime import datetime

from _ft_helpers import (
    VIDEO_PATH, load_yolo, load_osnet, detect_persons, sample_frame,
    prepare_data_dir, save_single_test_result, print_verdict,
)

TEST_ID = "FT-05"
DATA_OUT = prepare_data_dir(TEST_ID)


def run():
    print(f"\n{'='*60}\n{TEST_ID} — extract_batch sequential fallback  "
          f"({datetime.now():%H:%M:%S})\n{'='*60}")

    yolo  = load_yolo()
    osnet = load_osnet()
    frame = sample_frame(VIDEO_PATH, frame_idx=50)

    boxes = detect_persons(yolo, frame)[:2]
    assert len(boxes) >= 2, "Need >=2 detections to trigger batch path"

    underlying = getattr(getattr(osnet._model, "model", None),
                         "get_features", None) if osnet._model else None
    if underlying is None:
        result = {
            "status": "SKIP",
            "reason": "OSNet model not loaded; sequential fallback path can't be triggered",
        }
        print(f"  SKIP — {result['reason']}")
        print_verdict(TEST_ID, "SKIP")
        save_single_test_result(TEST_ID, DATA_OUT, result)
        return

    call_count = {"n": 0}

    def patched_features(arr, frm):
        call_count["n"] += 1
        if call_count["n"] == 1 and len(arr) > 1:
            raise RuntimeError("simulated batch OOM")
        return underlying(arr, frm)

    osnet._model.model.get_features = patched_features

    result = osnet.extract_batch(frame, boxes)
    valid = sum(r is not None for r in result)
    ok = (len(result) == 2) and (valid >= 1)
    status = "PASS" if ok else "FAIL"
    print(f"  call_count={call_count['n']}  valid={valid}/2  | {status}")
    print_verdict(TEST_ID, status)

    save_single_test_result(TEST_ID, DATA_OUT, {
        "status": status,
        "call_count": call_count["n"],
        "valid": valid,
    })


if __name__ == "__main__":
    run()
