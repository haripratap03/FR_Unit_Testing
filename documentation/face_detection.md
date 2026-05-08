# Face Detection Module Documentation

## Overview

Face Detection is the stage in the FR pipeline responsible for detecting faces from tracked person crops, scoring their quality, and saving the best candidates for downstream recognition. It operates on person-level crops (not full frames), runs per tracked person at configurable cooldown intervals, and applies a multi-component quality score to ensure only useful face images reach the InsightFace embedding stage.

The same detection logic is extended in `rerun_fr.py` with four fallback strategies to recover faces that the primary detector misses due to small image size, low contrast, or faces partially cut at crop edges.

---

## Pipeline Position

```
Recording Jetson (SSH/SCP)
        ↓
  YOLO v11l Person Detection       ← full frame inference
        ↓
  OSNet ReID Tracking              ← cross-frame identity continuity
        ↓
  ➡ Face Detection (YOLOv11s-face) ← per-person crop inference   [THIS MODULE]
  ➡ Face Quality Scoring           ← blur / size / brightness / pose
  ➡ Async Face Saving              ← ThreadPoolExecutor, keep top-N
        ↓
  Post-hoc Face Filtering          ← quality / skin / keypoint gates
        ↓
  InsightFace Embedding            ← antelopev2 ONNX
        ↓
  FAISS Staff Matching + Clustering
        ↓
  Excel Report
```

---

## Models Used

| Model                  | File                      | Role                                                                       |
| ---------------------- | ------------------------- | -------------------------------------------------------------------------- |
| YOLOv11s-face          | `models/yolov11s-face.pt` | Face detection with 5-point keypoints on person crops                      |
| InsightFace antelopev2 | `insightface_models/`     | Used in post-filtering and re-run embedding — **not** in primary detection |

YOLOv11s-face is loaded by `ClipProcessor.initialize()` at pipeline startup (`pipeline.py:735`) and optionally exported to TensorRT for maximum throughput. It runs in half-precision (`half=True`) on GPU when available.

---

## Workflow

```
Per video frame (sampled at PROCESSING_FPS):
  ┌─────────────────────────────────────────────────────┐
  │  YOLO Person Detection → N bounding boxes           │
  │  OSNet Batch Embedding + Tracker Match              │
  │  For each tracked person:                           │
  │    crop = frame[y1:y2, x1:x2]                      │
  │    face_targets.append((uid, crop, offset, box))    │
  └───────────────────┬─────────────────────────────────┘
                      │
                      ▼
  _batch_capture_faces(face_targets, timestamp)
    ├── cooldown check (1.5s between captures)
    ├── capacity check (MAX_FACES gate)
    ├── YOLO face model on crop (conf=0.3)
    │     → pick highest-confidence detection
    │     → extract 5-point keypoints
    │     → apply 10px bbox padding
    ├── score_face(face_crop, yolo_conf, keypoints)
    │     → hard reject: conf < 0.4, size < 40×40
    │     → score = yolo(20) + blur(25) + size(20) + brightness(15) + pose(20)
    ├── quality competition (replace lowest if at capacity)
    └── async save to data/<date>/faces/person_XXXX/face_XXXXXX_qNN.jpg
```

---

## Detailed Process

### Step 1 — Person Crop Extraction

At each processed frame, YOLO detects all persons. For each person that is actively tracked (`uid in self.tracked_persons`), a crop is sliced from the full frame:

```python
# pipeline.py:881-901
x1, y1, x2, y2 = [int(v) for v in box]
x1, y1 = max(0, x1), max(0, y1)
x2, y2 = min(w, x2), min(h, y2)
crop = frame[y1:y2, x1:x2]
face_targets.append((uid, crop, (x1, y1), box))
```

Crops are passed to `_batch_capture_faces` as a list. Despite the name, processing is sequential per crop — the face model's TensorRT engine operates at batch size 1.

### Step 2 — Cooldown and Capacity Gates

Before running the face model, two gates avoid redundant inference:

1. **Cooldown gate**: 1.5 seconds must pass since the last face capture for the same person. This prevents capturing multiple nearly identical frames.
2. **Capacity gate**: If `len(person['face_images']) >= MAX_FACES` and fewer than 3 seconds have elapsed since the last capture, skip entirely. Once 3 seconds have passed, a new capture attempt may replace the current worst face.

### Step 3 — YOLO Face Detection on Crop

The face model runs on the person crop:

```python
# pipeline.py:963
face_results = self.face_model(crop, conf=0.3, verbose=False, half=True)
```

- **Input**: BGR person crop (variable size, at least `MIN_CROP_W × MIN_CROP_H` pixels).
- **Confidence threshold**: 0.3 — relatively permissive to tolerate partially occluded faces. Hard quality gates downstream handle false positives.
- **Multiple detections**: If the crop contains more than one detected face (e.g., two people walking closely), only the detection with the **highest confidence** is used (`best_idx = face_boxes.conf.argmax()`).
- **Keypoints**: YOLOv11s-face outputs 5 facial landmarks (nose, left eye, right eye, left ear, right ear) for each detection. These are extracted from `face_results[0].keypoints[best_idx]`.

### Step 4 — Bounding Box Padding

A 10-pixel symmetric pad is added to the detected face bbox before cropping:

```python
# pipeline.py:981-986
pad = 10
fx1 = max(0, fx1 - pad)
fy1 = max(0, fy1 - pad)
fx2 = min(crop.shape[1], fx2 + pad)
fy2 = min(crop.shape[0], fy2 + pad)
face_crop = crop[fy1:fy2, fx1:fx2]
```

This gives the face recognition model a small margin of context around the face boundary, which improves InsightFace detection on tight crops.

### Step 5 — Quality Scoring

`score_face(face_crop, yolo_conf, keypoints)` returns a scalar score (0–100). A score of 0 indicates hard rejection. See the [Face Quality Scoring](#face-quality-scoring) section for full details.

### Step 6 — Quality Competition and Async Save

```python
# pipeline.py:995-1016
if len(person['face_images']) >= MAX_FACES:
    min_score = min(person['face_images'], key=lambda x: x[0])[0]
    if quality <= min_score:
        continue   # new face is not better — discard

filename = f"face_{self.frame_count:06d}_q{quality:.0f}.jpg"
filepath = os.path.join(person['images_folder'], filename)

face_copy = face_crop.copy()
self._face_save_pool.submit(cv2.imwrite, filepath, face_copy)   # async

person['face_images'].append((quality, filepath))
...
if len(person['face_images']) > MAX_FACES:
    person['face_images'].sort(key=lambda x: x[0], reverse=True)
    removed = person['face_images'].pop()       # remove lowest
    self._face_save_pool.submit(os.remove, removed[1])  # async delete
```

- **File saves are asynchronous** via a `ThreadPoolExecutor(max_workers=4)`. The GPU pipeline does not block on disk I/O.
- The face list is kept sorted by quality; the worst image is evicted when the list exceeds `MAX_FACES`.
- `best_face_image` always points to the highest-quality face path for the current person.
- The quality score is embedded in the filename (`_qNN`) so downstream scripts can re-rank or filter without reloading images.

---

## Key Functions

### `score_face`

**Location**: `pipeline.py:593`

- **Purpose**: Compute a quality score (0–100) for a face crop to decide if it is worth keeping and how it ranks among other captured faces for the same person.
- **Input**:
  - `face_crop` — BGR crop of the face (already padded, sliced from person crop)
  - `yolo_conf` — YOLO detection confidence (float 0–1)
  - `keypoints` — Optional array of shape `(5, 3)`: `(x, y, conf)` per landmark from YOLOv11s-face
- **Output**: `(score: float, reason: dict | str)` — score 0 means hard reject
- **Logic**:

  **Hard gates** (return 0 immediately):
  - `yolo_conf < 0.4` → `"low_yolo"`
  - Face crop narrower or shorter than 40px → `"too_small"`

  **Soft component scores** (all contribute to the total):

  | Component    | Calculation                               | Max             |
  | ------------ | ----------------------------------------- | --------------- | --------------- | --- |
  | `yolo`       | `yolo_conf × 20`                          | 20              |
  | `blur`       | `min(Laplacian_variance / 300, 1.0) × 25` | 25              |
  | `size`       | `min(min(w, h) / 120, 1.0) × 20`          | 20              |
  | `brightness` | `max(1 −                                  | mean_gray − 130 | / 130, 0) × 15` | 15  |
  | `pose`       | `symmetry × ear_penalty × 20`             | 20              |

  **Pose scoring detail**:
  - Requires nose, left eye, right eye keypoints with confidence ≥ 0.3.
  - Symmetry = `min(|nose_x − leye_x|, |nose_x − reye_x|) / max(...)` — penalizes heavily off-centre or profile poses.
  - `ear_penalty = 0.8` if both ears are visible (full frontal, ears present), else 1.0. Slight frontal preference.
  - If any critical keypoint confidence < 0.3, pose score = 6 (0.3 × 20).

- **Role in Face Detection**: Acts as the primary quality gate. Only faces with score > 0 enter the saved pool. Faces compete by score — the lowest-quality face in the pool is evicted when `MAX_FACES` is reached.

---

### `ClipProcessor._batch_capture_faces`

**Location**: `pipeline.py:940`

- **Purpose**: Orchestrate the face capture loop for all tracked persons in a single frame, applying the cooldown, detection, scoring, and async save logic.
- **Input**:
  - `face_targets` — list of `(uid, crop, offset, box)` tuples, one per tracked person with a valid crop in the current frame
  - `ts` — `datetime` wall-clock timestamp for the current frame
- **Output**: None — modifies `self.tracked_persons[uid]['face_images']` and triggers async disk writes
- **Logic** (per target):
  1. Skip if person no longer tracked.
  2. Enforce 1.5s cooldown; enforce 3.0s wait if at capacity.
  3. Run `self.face_model(crop, conf=0.3)` on the person crop.
  4. Select highest-confidence detection if multiple faces found.
  5. Pad the face bbox by 10px.
  6. Call `score_face` — skip if quality == 0.
  7. If at capacity and new quality ≤ current minimum, skip.
  8. Submit async `cv2.imwrite` to the thread pool.
  9. Append to `face_images` list; evict lowest if over capacity.
  10. Update `best_face_image` and `_last_face_capture` timestamp.
- **Role in Face Detection**: Central coordinator between the per-frame tracking loop and disk persistence. Decouples GPU inference from disk I/O through async saves.

---

### `_detect_face`

**Location**: `rerun_fr.py:158`

- **Purpose**: Detect the largest face in an image using InsightFace and return its L2-normalized embedding.
- **Input**: `face_app` (InsightFace `FaceAnalysis` instance), `img` (BGR image)
- **Output**: L2-normalized `float32` embedding array (512-dim), or `None` if no face detected
- **Logic**:
  1. Call `face_app.get(img)` — runs antelopev2 ONNX detector.
  2. If no faces returned, return `None`.
  3. Select the face with the largest bounding box area: `(bbox[2]-bbox[0]) × (bbox[3]-bbox[1])`.
  4. L2-normalize the embedding; return as `float32`.
- **Role in Face Detection**: The atomic detection unit used by all four strategies in `_extract_with_strategies`. Encapsulates InsightFace's detection + embedding into a single nullable call.

**Note**: In `rerun_fr.py`, InsightFace is initialized with `det_thresh=0.15` (vs `det_thresh=0.3` in `pipeline.py`). This lower threshold is intentional — the re-run targets face images that the pipeline already failed on, so higher recall is prioritized over precision.

---

### `_extract_with_strategies`

**Location**: `rerun_fr.py:125`

- **Purpose**: Try up to four progressively more aggressive detection strategies on a single image, returning the embedding from the first strategy that succeeds.
- **Input**: `face_app`, `img` (BGR image)
- **Output**: `(embedding: float32 array | None, strategy_name: str | None)`
- **Logic**:

  | Strategy   | Condition                      | Transformation                                                                |
  | ---------- | ------------------------------ | ----------------------------------------------------------------------------- |
  | `direct`   | Always tried first             | None — original image                                                         |
  | `upscaled` | `w < 200` or `h < 200`         | `cv2.resize` with `INTER_CUBIC`; scale factor = `max(200/w, 200/h)`           |
  | `enhanced` | Always tried if direct fails   | CLAHE on L channel in LAB color space (`clipLimit=3.0`, `tileGridSize=(8,8)`) |
  | `padded`   | Always tried if enhanced fails | `cv2.copyMakeBorder` with 40px on all sides, `BORDER_REFLECT_101`             |

  Each strategy calls `_detect_face`. On first success, the strategy name and embedding are returned. If all four fail, returns `(None, None)`.

- **Role in Face Detection**: Recovers face embeddings from images that primary YOLO+pipeline detection missed — typically due to very small face crops, poor lighting, or faces partially cut at the edge of the person bounding box.

---

## Input / Output

### Primary Detection (`pipeline.py`)

| Stage                   | Data                                                                  |
| ----------------------- | --------------------------------------------------------------------- |
| **Input to face model** | BGR person crop, variable size (≥ `MIN_CROP_W` × `MIN_CROP_H` pixels) |
| **Face model output**   | Bounding boxes + confidences + 5 keypoints per detection              |
| **Saved face image**    | BGR JPEG, padded face crop                                            |
| **Filename convention** | `face_XXXXXX_qNN.jpg` — frame counter + quality score                 |
| **Destination**         | `data/<date>/faces/person_XXXX/`                                      |
| **Quality range**       | 0–100 (hard rejected < score_face threshold; typically 25 floor)      |

### Re-run Detection (`rerun_fr.py`)

| Stage        | Data                                                                                                |
| ------------ | --------------------------------------------------------------------------------------------------- |
| **Input**    | Pre-saved JPEG face images in `data/<date>/faces/person_XXXX/`                                      |
| **Process**  | `_extract_with_strategies` on each JPEG, sorted by filename (reverse order = highest quality first) |
| **Output**   | List of L2-normalized embeddings per person; strategy name logged for diagnostics                   |
| **Fallback** | `body_snapshot.jpg` appended as last-resort input if no face images work                            |

---

## Face Quality Scoring

Quality scoring runs in `score_face` (`pipeline.py:593`) and gates all face saves during pipeline processing.

### Hard Gates (score = 0, immediate reject)

| Gate            | Threshold                      | Reason                                             |
| --------------- | ------------------------------ | -------------------------------------------------- |
| YOLO confidence | `< 0.4`                        | Detection too uncertain                            |
| Face crop size  | `< 40 × 40 px`                 | Too small for reliable recognition                 |
| Empty crop      | `size == 0` or `shape < 20×20` | Degenerate bbox (checked before `score_face` call) |

### Soft Score Components

```
Total score = yolo + blur + size + brightness + pose    (max ≈ 100)
```

**Blur** (max 25 pts) — Laplacian variance of grayscale crop:

- `score = min(variance / 300, 1.0) × 25`
- Variance 300+ → full score; variance 0 → 0 (completely blurred)

**YOLO confidence** (max 20 pts):

- `score = yolo_conf × 20`
- Confident detections (0.9+) score near 18 pts

**Face size** (max 20 pts) — based on the shorter face dimension:

- `score = min(min(w, h) / 120, 1.0) × 20`
- A 120×120 face scores the full 20 pts; a 60×60 face scores 10 pts

**Brightness** (max 15 pts) — penalizes both over- and under-exposure:

- `score = max(1 − |mean_gray − 130| / 130, 0) × 15`
- Mean gray of 130 (target) → 15 pts; mean 0 or 260 → 0 pts

**Pose / symmetry** (max 20 pts) — requires YOLOv11s-face keypoints:

- `symmetry = min(|nose_x − leye_x|, |nose_x − reye_x|) / max(...)`
- Profile views score near 0; frontal views score near 1.0
- `ear_penalty = 0.8` if both ears detected (discourages over-scoring frontal faces at the edge of profile)
- Falls back to 10 pts (half max) when keypoints unavailable

### Post-capture Quality Floor

During FR processing (`run_fr_processing` in `pipeline.py:1088`), faces with encoded quality `< FILTER_MIN_QUALITY` (default 25) are moved to `rejected/` before any embeddings are computed. This is the cheapest filter and runs first.

---

## Edge Cases & Failure Handling

| Scenario                                 | Behaviour                                                                                             |
| ---------------------------------------- | ----------------------------------------------------------------------------------------------------- |
| No face detected in crop                 | `_batch_capture_faces` skips silently; person may still get faces from later frames                   |
| Multiple faces in one crop               | Only the highest-confidence detection is used (argmax on confidences)                                 |
| Face bbox extends outside crop boundary  | Padding clamped to crop dimensions before slicing                                                     |
| Resulting face crop is < 20×20 after pad | Discarded before `score_face` (guards against zero-sized crops at frame edges)                        |
| All faces rejected for a person          | Person recorded as "no-face" in `state.json`; body snapshot used as fallback in report                |
| Face model fails (exception)             | No face captured for that frame; processing continues                                                 |
| Async save failure                       | Silent — `cv2.imwrite` returns False; entry still in `face_images` list pointing to non-existent file |
| Re-run: all 4 strategies fail            | `no_face_list` entry created; images_scanned count logged for diagnostics                             |
| Re-run: body snapshot as last resort     | `body_snapshot.jpg` appended to image list; strategies tried on it                                    |
| Cooldown not elapsed                     | Detection skipped entirely for that frame; no YOLO face inference                                     |

---

## Performance Considerations

### GPU Usage

- The face model runs on GPU (`half=True`, FP16) for each person crop per frame.
- The model call is sequential per crop, not batched. The TensorRT engine (if enabled) runs optimally at batch size 1.
- Face detection typically accounts for ~20–30% of total per-frame GPU time (alongside person detection and OSNet).

### Async Face Saving

```python
self._face_save_pool = ThreadPoolExecutor(max_workers=4)
self._face_save_pool.submit(cv2.imwrite, filepath, face_copy)
```

- File writes happen on 4 background threads, decoupled from the GPU processing loop.
- `face_crop.copy()` is called before submitting to avoid race conditions with the main thread's frame buffer.
- File deletions (lowest-quality evictions) are also async.

### Sampling Rate

The pipeline processes frames at `PROCESSING_FPS` (default 10 FPS), skipping every `skip = native_fps / PROCESSING_FPS` frames. At 30 FPS native and 10 FPS processing, every third frame is analyzed. Combined with the 1.5s cooldown, typical captures per person are 1 every 1.5–3 seconds of screen time.

### Top-N Retention

`MAX_FACES` (config: `face_capture.max_faces_per_person`, currently 10) bounds memory. After processing, `POST_FILTER_KEEP_TOP_N` (code default 20) further trims to the top-scoring faces before FR runs, reducing embedding computation.

---

## Integration with Other Modules

### Upstream: PersonTracker

Face detection only runs on persons promoted to `active` tracks (past `WARMUP_FRAMES`). Pending tracks do not have face capture attempted, avoiding wasting inference on transient detections.

### Downstream: Post-hoc Filtering (`filter_faces.py`)

`filter_faces.py` applies three filters as a standalone cleanup pass on the saved face images:

1. **Quality floor** — reads `_qNN` from filename; rejects if below threshold (cheapest, runs first)
2. **Skin-tone check** — HSV analysis on center 60% of face crop; two ranges cover light-to-dark skin tones
3. **Keypoint validation** — InsightFace detects face and checks that ≥ 3 of 5 landmarks fall inside the bbox (with 20% margin); guards against misdetections where landmarks cluster outside the face region

Rejected images are moved to `rejected/` within the person folder, not deleted — allowing manual inspection.

The same filter logic is embedded directly in `run_fr_processing` inside `pipeline.py` (applied before embedding, without requiring a separate pass).

### Downstream: FR Processing

The filtered face images' paths and quality scores are carried in `state.json` under each person's `face_images` list. The FR step iterates the top-N remaining images, extracts InsightFace embeddings, and searches the FAISS staff index.

### Downstream: `rerun_fr.py`

`rerun_fr.py` bypasses the saved quality metadata and re-runs detection from scratch on the image files, using `_extract_with_strategies` with `det_thresh=0.15` for higher recall. This is the recommended path when the pipeline's first-pass FR yields too many no-face results — typically because face crops were captured from difficult angles or in poor lighting.
