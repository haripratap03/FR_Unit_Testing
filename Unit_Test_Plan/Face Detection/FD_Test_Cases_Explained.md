# Face Detection Unit Test Cases — FD-01 to FD-44

**File:** `Unit_Test_Test_Cases/Face Detection/test_face_detection.py`  
**Last run:** 2026-04-30 — 44 passed, 0 failed, 0 errors  
**Runtime:** ~0.11 s (no GPU, no model weights required)

---

## Overview

These 44 tests cover every function in the face detection pipeline:
the quality scorer, two skin-tone checkers, two keypoint counters,
a contrast enhancer, an embedding extractor, a multi-strategy embedder,
and the main face-capture batch method.

No video, no YOLO weights, and no InsightFace models are loaded.
Every ML call is replaced by a `MagicMock`; images are synthesised
with NumPy.

---

## Test Infrastructure

### Synthetic Image Helpers

| Helper | What it produces | Used by |
|---|---|---|
| `sharp_image(h, w)` | Seeded RNG noise clipped to a target mean brightness — simulates a well-focused face crop | FD-01..11, FD-22..27, FD-33..37 |
| `blurry_image(h, w)` | Flat array passed through `cv2.GaussianBlur(kernel 21×21)` — Laplacian variance < 10 | FD-06 |
| `skin_tone_image(h, w)` | Solid `BGR(60,100,160)` → HSV ≈ (168°, 47%, 63%) — falls in skin-tone range 2 | FD-16, FD-22 |
| `non_skin_image(h, w)` | Solid `BGR(255,0,0)` (pure blue) → HSV H=120° — outside every skin range | FD-17, FD-18, FD-23 |

### Mock Stubs

| Stub | Replaces | Key attributes |
|---|---|---|
| `_MockFace` | InsightFace `Face` object | `.bbox` (float32 array), `.kps` (5×2 array), `.embedding` (512-d array) |
| `_MockYoloBoxes` | YOLO `Results[0].boxes` | `.conf` (argmax-able), `.xyxy` (indexed rows) |
| `make_face_model(...)` | Callable YOLO model | Returns `[result0]` where `result0.boxes` is a `_MockYoloBoxes` or `None` |
| `make_processor(...)` | `ClipProcessor` instance | `.tracked_persons`, `.face_model`, `._face_save_pool`, `.frame_count`, `.stats` |

### `run_batch(processor, face_targets, ts)`

Calls `ClipProcessor._batch_capture_faces(processor, face_targets, ts)` — binds the real
method to a fake `self` so it runs without constructing the full `ClipProcessor`.

---

## Function Groups and Test Cases

---

### Group 1 — `score_face` (`pipeline.py:593`)

`score_face(face_crop, yolo_conf, keypoints=None)` returns `(total_score, component_dict)`.

It sums five weighted components — **yolo**, **blur**, **size**, **brightness**, **pose** —
and applies two hard gates that short-circuit to `score=0` before any component is computed.

#### Hard Gates

| Gate | Condition | Reason tag |
|---|---|---|
| Low YOLO confidence | `yolo_conf < 0.4` | `"low_yolo"` |
| Crop too small | `min(h, w) < 40` | `"too_small"` |

---

**FD-01 — Hard reject: YOLO confidence below 0.4**

- **Input:** 80×80 sharp BGR image; `yolo_conf=0.35`; `keypoints=None`
- **Why it matters:** The YOLO conf gate is the cheapest possible reject — if triggered, no
  scoring math runs at all. The test confirms the gate fires and the reason string is
  `"low_yolo"`.
- **Expected:** `score == 0`, `"low_yolo" in reason`
- **Result:** PASS

---

**FD-02 — YOLO confidence exactly 0.4 (boundary)**

- **Input:** 80×80 sharp BGR; `yolo_conf=0.40`
- **Why it matters:** The gate condition is `< 0.4` (strict less-than), so 0.40 must pass.
  Off-by-one fence-post errors here would silently drop borderline frames.
- **Expected:** `score > 0`
- **Result:** PASS

---

**FD-03 — Hard reject: crop too small (35×35)**

- **Input:** 35×35 BGR; `yolo_conf=0.90`
- **Why it matters:** A 35×35 face crop does not contain enough pixels for reliable quality
  scoring. The size gate short-circuits before any computation.
- **Expected:** `score == 0`, `"too_small" in reason`
- **Result:** PASS

---

**FD-04 — Crop exactly 40×40 (boundary)**

- **Input:** 40×40 sharp BGR; `yolo_conf=0.80`
- **Why it matters:** Same fence-post concern as FD-02. Gate is `< 40`, so 40 must pass.
- **Expected:** `score > 0`
- **Result:** PASS

---

**FD-05 — Perfect frontal face, maximum score**

- **Input:** 120×120 sharp image at mean brightness 130; `yolo_conf=0.95`; frontal keypoints
  ```
  [[60,50,1.0], [40,40,1.0], [80,40,1.0], [40,70,0.9], [80,70,0.9]]
  ```
  (nose centred, both eyes symmetric, both mouth corners symmetric)
- **Why it matters:** The only test that exercises all five components simultaneously.
  Confirms that a textbook-quality crop actually scores well (≥ 70) and that none of the
  components are accidentally zeroed.
- **Expected:** `score >= 70`; all of `yolo`, `blur`, `size`, `brightness`, `pose` > 0
- **Result:** PASS

---

**FD-06 — Blurry image, low blur component**

- **Input:** 80×80 `blurry_image` (GaussianBlur 21×21); `yolo_conf=0.75`
- **Why it matters:** The blur component uses Laplacian variance as a sharpness proxy. A
  heavily blurred image produces near-zero variance, so the blur contribution collapses.
  The total score should still be positive (other components fire), but the blur component
  must be < 1.0.
- **Expected:** `score > 0`; `scores['blur'] < 1.0`
- **Result:** PASS

---

**FD-07 — Profile pose, low pose symmetry score**

- **Input:** 80×80 sharp; `yolo_conf=0.80`; keypoints:
  ```
  [[20,40,0.9], [15,35,0.9], [70,35,0.8], [0,0,0.0], [0,0,0.0]]
  ```
  Nose (kp[0]) is almost on top of the left eye (kp[1]) — a strong left-profile.
- **Why it matters:** Pose symmetry is scored by the horizontal distance from the nose to
  the midpoint between both eyes. A profile face has near-zero symmetry, so the pose
  component should drop below 5.
- **Expected:** `scores['pose'] < 5`
- **Result:** PASS

---

**FD-08 — Both ears visible, ear penalty 0.8 applied**

- **Input:** 120×120 sharp; `yolo_conf=0.90`; keypoints all with conf ≥ 0.3, both mouth
  corners (kp[3], kp[4]) with conf ≥ 0.5 (ear-proxy heuristic):
  ```
  [[60,50,0.9], [40,40,0.9], [80,40,0.9], [25,65,0.8], [95,65,0.8]]
  ```
- **Why it matters:** When both ear-proxy keypoints are confident, `pipeline.py` applies
  `ear_pen = 0.8` to the pose component, capping it at `20 × 0.8 = 16`. This test confirms
  the penalty is applied and not accidentally skipped.
- **Expected:** `scores['pose'] <= 16`
- **Result:** PASS

---

**FD-09 — All keypoint confidences < 0.3, fallback score**

- **Input:** 80×80 sharp; `yolo_conf=0.80`; all five keypoint confidences = 0.1
- **Why it matters:** When no keypoint is confident enough, the pose component falls back
  to a neutral constant (6.0) rather than zero or an undefined value.
- **Expected:** `scores['pose'] == 6.0`
- **Result:** PASS

---

**FD-10 — `keypoints=None`, neutral fallback pose score**

- **Input:** 80×80 well-lit; `yolo_conf=0.80`; `keypoints=None`
- **Why it matters:** Not every detection provides keypoints. When the argument is `None`,
  the fallback must produce a stable neutral value (10) rather than raising an exception.
- **Expected:** `scores['pose'] == 10`
- **Result:** PASS

---

**FD-11 — Overexposed image, low brightness component**

- **Input:** 80×80 filled with `BGR(240,240,240)`; `yolo_conf=0.85`
- **Why it matters:** The brightness component uses `max(1 − |mean_gray − 130| / 130, 0) × 15`.
  At mean_gray ≈ 240 the result is ≈ 0. Overexposed crops should score near zero on
  brightness so the total score stays low.
- **Expected:** `scores['brightness'] < 5`
- **Result:** PASS

---

### Group 2 — `check_quality` (`filter_faces.py`)

`check_quality(filename, min_quality)` parses the `_qNN` suffix from the filename and
returns `(passed: bool, score: int, reason: str)`.
Files without the suffix unconditionally pass with `score = -1`.

---

**FD-12 — Score above threshold passes**

- **Input:** `"face_001234_q42.jpg"`, `min_quality=25`
- **Why it matters:** Standard passing case — confirms the parser extracts 42 and the
  comparison is correct.
- **Expected:** `(True, 42, "")`
- **Result:** PASS

---

**FD-13 — Score below threshold fails**

- **Input:** `"face_001234_q18.jpg"`, `min_quality=25`
- **Why it matters:** Confirms a low-quality face is rejected and the reason string encodes
  the actual score and threshold for downstream debugging.
- **Expected:** `(False, 18, "quality_18_below_25")`
- **Result:** PASS

---

**FD-14 — No `_qNN` tag, unconditional pass**

- **Input:** `"body_snapshot.jpg"`, `min_quality=25`
- **Why it matters:** Body snapshot files and other tagless filenames must always pass —
  the quality gate only applies to `face_*` crops. This guards against accidentally
  rejecting non-face images in the filter pipeline.
- **Expected:** `(True, -1, "")`
- **Result:** PASS

---

**FD-15 — Score exactly at threshold (boundary)**

- **Input:** `"face_001234_q25.jpg"`, `min_quality=25`
- **Why it matters:** The gate is `score < min_quality` (strict), so `score == min_quality`
  must pass. An off-by-one error here would silently drop all boundary-quality faces.
- **Expected:** `(True, 25, "")`
- **Result:** PASS

---

### Group 3 — `check_skin_tone` (`filter_faces.py`)

`check_skin_tone(img, min_skin_ratio)` evaluates the **centre 60% × 60% crop** of the
image only. Returns `(passed: bool, ratio: float, reason: str)`.

---

**FD-16 — Solid skin-tone image passes**

- **Input:** 100×100 `np.full(BGR(60,100,160))` (HSV ≈ H168, S47%, V63% — skin range 2);
  `min_skin_ratio=0.20`
- **Why it matters:** Confirms the HSV skin mask fires for a genuine skin colour.
- **Expected:** `passed == True`, `ratio >= 0.20`
- **Result:** PASS

---

**FD-17 — Non-skin image rejected**

- **Input:** 100×100 solid `BGR(255,0,0)` (pure blue, HSV H=120°); `min_skin_ratio=0.20`
- **Why it matters:** Pure blue has no overlap with the skin HSV ranges. The filter must
  reject it cleanly.
- **Expected:** `passed == False`, `ratio < 0.20`
- **Result:** PASS

---

**FD-18 — Centre crop isolation**

- **Input:** 100×100 image where the outer 20% border is warm skin tone but the inner 60%
  is pure blue; `min_skin_ratio=0.20`
- **Why it matters:** `check_skin_tone` is supposed to evaluate only the centre of the
  image, so peripheral skin pixels should not rescue an otherwise non-skin crop. This test
  would catch any regression where the function evaluates the full image instead.
- **Expected:** `passed == False` (only centre is evaluated; centre is blue)
- **Result:** PASS

---

### Group 4 — `check_keypoints` (`filter_faces.py`)

`check_keypoints(img, face_app, min_keypoints)` runs InsightFace `face_app.get(img)` and
counts how many of the returned landmarks fall inside the detected bounding box (with a
10% margin). Returns `(passed: bool, n_valid: int, reason: str)`.

---

**FD-19 — All 5 landmarks inside bbox**

- **Input:** `face_app.get()` returns a `_MockFace` with `bbox=(10,10,90,90)` and all five
  keypoints well inside: `[[50,50],[30,30],[70,30],[30,70],[70,70]]`; `min_keypoints=3`
- **Why it matters:** The happy-path baseline. Five-of-five should clearly pass.
- **Expected:** `(True, 5, "")`
- **Result:** PASS

---

**FD-20 — Insufficient landmarks inside bbox**

- **Input:** Same bbox; three keypoints far outside (`[200,300]`, `[250,300]`, `[-5,-5]`);
  only two valid; `min_keypoints=3`
- **Why it matters:** Confirms the count-based rejection logic and that the reason string
  encodes both the actual count and the threshold.
- **Expected:** `(False, 2, "keypoints_2_of_3")`
- **Result:** PASS

---

**FD-21 — No face detected by InsightFace**

- **Input:** `face_app.get()` returns `[]`; `min_keypoints=3`
- **Why it matters:** If InsightFace finds no face at all, `check_keypoints` must return a
  distinct `"no_face_detected"` reason rather than crashing on an empty list or returning
  an ambiguous zero count.
- **Expected:** `(False, 0, "no_face_detected")`
- **Result:** PASS

---

### Group 5 — `_check_skin_tone` (`pipeline.py:636`)

The inline pipeline variant of the skin-tone check. Same HSV logic as
`filter_faces.check_skin_tone` but returns a plain `bool`.

---

**FD-22 — Skin-tone image passes**

- **Input:** 100×100 solid `BGR(60,100,160)`
- **Expected:** `True`
- **Result:** PASS

---

**FD-23 — Non-skin image fails**

- **Input:** 100×100 solid `BGR(255,0,0)` (pure blue)
- **Expected:** `False`
- **Result:** PASS

---

### Group 6 — `_count_kps_in_bbox` (`pipeline.py:660`)

`_count_kps_in_bbox(bbox, kps)` expands the bbox by 20% of the shorter dimension
(min(w,h) × 0.20) on each side, then counts how many of the five keypoints fall inside
the expanded box. Boundary is **inclusive**.

---

**FD-24 — All 5 keypoints inside bbox**

- **Input:** `bbox=(10,10,90,90)`; all five kps well inside
- **Expected:** `5`
- **Result:** PASS

---

**FD-25 — All keypoints outside bbox**

- **Input:** `bbox=(10,10,90,90)`; all five kps at (200+, 200+) — far outside even with margin
- **Expected:** `0`
- **Result:** PASS

---

**FD-26 — Keypoint exactly on margin boundary (inclusive)**

- **Input:** `bbox=(20,20,80,80)` — width=60, height=60, margin=12; expanded box = `[8,8,92,92]`;
  first keypoint at `[8,50]` — exactly on the left boundary
- **Why it matters:** Confirms the boundary check is `>=` and `<=` (inclusive), not strict.
  A keypoint sitting exactly on the expanded edge must be counted.
- **Expected:** `5` (all five counted, first one at boundary)
- **Result:** PASS

---

**FD-27 — Partial match: 3 of 5 inside**

- **Input:** `bbox=(10,10,90,90)`; first three kps inside, last two at (200,300) and (300,400)
- **Expected:** `3`
- **Result:** PASS

---

### Group 7 — `_enhance_contrast` (`rerun_fr.py`)

`_enhance_contrast(img)` applies CLAHE only to the L channel in LAB colour space and
returns the result. The original image must not be mutated.

---

**FD-28 — CLAHE brightens a dark image**

- **Input:** 100×100 `BGR(30,30,30)` (very dark)
- **Why it matters:** CLAHE redistributes the L-channel histogram, so a dark image's mean
  brightness should increase. Also confirms output shape and dtype are preserved.
- **Expected:** `output.mean() > input.mean()`; same `shape` and `dtype` as input
- **Result:** PASS

---

**FD-29 — No in-place mutation of input**

- **Input:** 80×80 arbitrary BGR; copy taken before call
- **Why it matters:** If `_enhance_contrast` mutates in place, callers that reuse the
  original image for other strategies (upscale, padded) would receive corrupted input.
- **Expected:** `np.array_equal(img_before, img_after_call)` — original unchanged
- **Result:** PASS

---

### Group 8 — `_detect_face` (`rerun_fr.py:158`)

`_detect_face(img, face_app)` calls `face_app.get(img)`, selects the largest face by
bbox area, L2-normalises its embedding, and returns it. Returns `None` if no face found.

---

**FD-30 — Largest face selected when multiple detected**

- **Input:** `face_app.get()` returns two faces:
  - Face A: `bbox=(0,0,20,20)` area = 400
  - Face B: `bbox=(0,0,80,80)` area = 6400
- **Why it matters:** In multi-face crops the pipeline should use the most prominent face.
  Verifies the argmax-by-area selection, not just "first in list".
- **Expected:** returned embedding matches `normalize(Face_B.embedding)` within 1e-6
- **Result:** PASS

---

**FD-31 — L2 normalisation applied to raw embedding**

- **Input:** `face_app.get()` returns one face with `embedding = np.ones(512) * 5.0`
  (magnitude 5.0, far from unit-norm)
- **Why it matters:** FAISS uses inner product on unit vectors as cosine similarity.
  If the embedding is not normalised before storage, all similarity scores will be wrong.
- **Expected:** `|‖result‖ − 1.0| < 1e-6`
- **Result:** PASS

---

**FD-32 — Returns `None` when no face detected**

- **Input:** `face_app.get()` returns `[]`
- **Why it matters:** Callers check for `None` to skip FR. An exception or a zero vector
  here would break the embedding pipeline.
- **Expected:** `result is None`
- **Result:** PASS

---

### Group 9 — `_extract_with_strategies` (`rerun_fr.py:125`)

`_extract_with_strategies(img, face_app)` tries up to four strategies in order:

| # | Strategy | Triggered when |
|---|---|---|
| 1 | **direct** | always attempted first |
| 2 | **upscaled** | `min(h,w) < 200` |
| 3 | **enhanced** (CLAHE) | always attempted if direct fails |
| 4 | **padded** | always attempted last if all above fail |

Returns `(embedding, strategy_name)` or `(None, None)`.

---

**FD-33 — Direct strategy succeeds first**

- **Input:** 200×200 image; `face_app.get()` returns a valid face on the first call
- **Expected:** `strategy == "direct"`, `face_app.get.call_count == 1` (no fallback)
- **Result:** PASS

---

**FD-34 — Upscale triggered for small image**

- **Input:** 60×80 image (both dims < 200); `face_app.get.side_effect = [[], [mock_face]]`
  — direct fails, upscaled succeeds
- **Why it matters:** Small crops are common when a face is captured at distance. Upscaling
  gives InsightFace a larger input to work with.
- **Expected:** `strategy == "upscaled"`, call_count == 2
- **Result:** PASS

---

**FD-35 — Upscale skipped for large image, falls to CLAHE**

- **Input:** 250×250 (≥ 200 — upscale condition is `min(h,w) < 200`, so skipped);
  `face_app.get.side_effect = [[], [mock_face]]` — direct fails, upscale skipped, CLAHE wins
- **Why it matters:** Confirms the size gate correctly suppresses the upscale strategy for
  large images and that the fall-through continues to CLAHE.
- **Expected:** `strategy == "enhanced"`, call_count == 2
- **Result:** PASS

---

**FD-36 — Padded strategy as last resort**

- **Input:** 200×200; `side_effect = [[], [], [mock_face]]` — direct fails, enhanced fails,
  padded succeeds
- **Why it matters:** Padding adds a white border around the crop to help InsightFace
  handle edge-cropped faces. If it fires only as a true last resort this test catches any
  premature ordering.
- **Expected:** `strategy == "padded"`
- **Result:** PASS

---

**FD-37 — All four strategies fail**

- **Input:** 60×60 small image; `face_app.get()` always returns `[]`
- **Why it matters:** The function must degrade gracefully — no exception, no crash.
- **Expected:** `(None, None)`, no exception raised
- **Result:** PASS

---

### Group 10 — `_batch_capture_faces` (`pipeline.py:940` — `ClipProcessor`)

The main per-frame face capture loop. For each `(uid, crop, offset, box)` tuple in
`face_targets` it:

1. Checks cooldown (skip if last capture was < 1.5 s ago)
2. Runs YOLO face detection on the crop
3. Scores the detected face with `score_face`
4. Evicts the lowest-quality stored face if at capacity, else appends
5. Submits the save job to `_face_save_pool`

All tests use `make_processor()` + `run_batch()` instead of a real `ClipProcessor`.

---

**FD-38 — Valid detection: face captured**

- **Input:** `face_targets=[(uid, 100×100 crop, (0,0), [0,0,100,100])]`; YOLO returns
  `conf=0.85`, `bbox=(10,10,60,60)`; cooldown fully elapsed (3 s ago); `tracked_persons`
  injected with a person record
- **Expected:** `len(person.face_images) == 1`; `_face_save_pool.submit` called once;
  `best_face_image` populated
- **Result:** PASS

---

**FD-39 — Cooldown gate skips capture (< 1.5 s)**

- **Input:** `_last_face_capture[uid]` set to 0.5 s ago (cooldown = 1.5 s not elapsed)
- **Why it matters:** The cooldown prevents burst captures of the same person in a tight
  time window, which would waste disk and clog the face pool with near-duplicate frames.
- **Expected:** `face_model` never called; `submit` never called; `face_images` unchanged
- **Result:** PASS

---

**FD-40 — Quality gate: score 0 face discarded**

- **Input:** YOLO returns `conf=0.30` (< 0.4 hard gate → `score_face` returns 0)
- **Why it matters:** A face with `score == 0` must be silently dropped — not saved, not
  appended to the person record.
- **Expected:** `submit` not called; `face_images` empty
- **Result:** PASS

---

**FD-41 — No detection: YOLO returns empty boxes**

- **Input:** YOLO result has `boxes = None`
- **Why it matters:** A frame may have a person bounding box but no face inside it
  (person looking away, occluded, etc.). The loop must skip gracefully.
- **Expected:** `submit` not called; `face_images` empty
- **Result:** PASS

---

**FD-42 — Capacity full: new face below minimum stored score**

- **Input:** `face_images` already at `MAX_FACES`; minimum existing quality score = 50;
  new detection quality = 45
- **Why it matters:** The pool keeps only the top-N faces. A new face worse than the worst
  stored face should be discarded without evicting anything.
- **Expected:** `submit` not called; `len(face_images) == MAX_FACES` (unchanged)
- **Result:** PASS

---

**FD-43 — Capacity full: new face evicts lowest-quality**

- **Input:** `face_images` at `MAX_FACES`; lowest existing entry has quality 30 (filename
  `"old_q30.jpg"`); new detection quality = 65
- **Why it matters:** The pool upgrade — a better face arrives and displaces the worst
  stored crop. `submit` should be called twice: once to save the new face, once to delete
  the old one via `os.remove`.
- **Expected:** `submit` called twice; `len(face_images) == MAX_FACES`; score-30 entry
  gone; score-65 entry present
- **Result:** PASS

---

**FD-44 — Multiple detections: highest confidence wins**

- **Input:** YOLO returns two detections: `conf=0.60` at `bbox=(5,5,30,30)` and `conf=0.90`
  at `bbox=(10,10,60,60)`; `argmax()` returns index 1
- **Why it matters:** `_batch_capture_faces` uses `boxes.conf.argmax()` to pick one face
  per crop. Confirms that index 1 (the higher-confidence detection) drives the quality
  score, not index 0.
- **Expected:** Face from detection index 1 (bbox 10,10,60,60) saved; detection 0 ignored
- **Result:** PASS

---

## Coverage Matrix

| Function | Source file | Test IDs | Cases |
|---|---|---|---|
| `score_face` | `pipeline.py:593` | FD-01 to FD-11 | 11 |
| `check_quality` | `filter_faces.py` | FD-12 to FD-15 | 4 |
| `check_skin_tone` | `filter_faces.py` | FD-16 to FD-18 | 3 |
| `check_keypoints` | `filter_faces.py` | FD-19 to FD-21 | 3 |
| `_check_skin_tone` | `pipeline.py:636` | FD-22 to FD-23 | 2 |
| `_count_kps_in_bbox` | `pipeline.py:660` | FD-24 to FD-27 | 4 |
| `_enhance_contrast` | `rerun_fr.py` | FD-28 to FD-29 | 2 |
| `_detect_face` | `rerun_fr.py:158` | FD-30 to FD-32 | 3 |
| `_extract_with_strategies` | `rerun_fr.py:125` | FD-33 to FD-37 | 5 |
| `_batch_capture_faces` | `pipeline.py:940` | FD-38 to FD-44 | 7 |
| **Total** | | | **44** |

---

## Running the Tests

```bash
# From FR_Thor/ root
pytest "Unit_Test_Test_Cases/Face Detection/test_face_detection.py" -v

# Single test by ID substring
pytest "Unit_Test_Test_Cases/Face Detection/test_face_detection.py" -v -k "fd05"

# Stop on first failure
pytest "Unit_Test_Test_Cases/Face Detection/test_face_detection.py" -v -x
```

**Requirements:** `numpy`, `opencv-python`, `pytest` — no GPU or model weights needed.

---

## Known Gaps

The 44 tests above cover the minimum required set. Extended cases FD-045 to FD-077
(in `Unit_Test_Test_Cases/feature_test_cases/face_detection_test_cases.md`) are currently
`State=Design, Verdict=Not Run` — test code has not yet been written for them.
