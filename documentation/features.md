# Features Documentation — Offline Face Recognition (FR) Pipeline

---

## Overview

The Offline FR Pipeline is an AI-powered computer vision system that processes recorded video footage from a pharmacy/retail store. It detects persons, tracks their movement, captures face crops, recognizes enrolled staff, clusters unknown customers into unique identities, and produces a structured daily Excel report — all without requiring internet connectivity.

The pipeline is designed to run either as a continuous background service or as a one-shot processor for a specific date. Core AI components are YOLO (person + face detection), OSNet (re-identification tracking), InsightFace/AntelopeV2 (face embeddings), and FAISS (fast nearest-neighbour search).

---

## Feature List

| #   | Feature                                                               | Primary File                                 |
| --- | --------------------------------------------------------------------- | -------------------------------------------- |
| 1   | [Video Fetching](#1-video-fetching)                                   | `pipeline.py`                                |
| 2   | [Frame Reading](#2-frame-reading)                                     | `pipeline.py`                                |
| 3   | [Person Detection](#3-person-detection)                               | `pipeline.py`                                |
| 4   | [Person Tracking (Re-ID)](#4-person-tracking-re-id)                   | `pipeline.py`                                |
| 5   | [Face Detection](#5-face-detection)                                   | `pipeline.py`                                |
| 6   | [Face Quality Scoring](#6-face-quality-scoring)                       | `pipeline.py`                                |
| 7   | [Face Filtering](#7-face-filtering)                                   | `filter_faces.py`, `pipeline.py`             |
| 8   | [Face Recognition](#8-face-recognition)                               | `pipeline.py`, `rerun_fr.py`, `fr_report.py` |
| 9   | [Customer Clustering](#9-customer-clustering)                         | `pipeline.py`, `rerun_fr.py`, `fr_report.py` |
| 10  | [Staff Registration](#10-staff-registration)                          | `register_staff.py`                          |
| 11  | [Re-run FR Processing](#11-re-run-fr-processing)                      | `rerun_fr.py`                                |
| 12  | [Report Generation](#12-report-generation)                            | `pipeline.py`, `rerun_fr.py`, `fr_report.py` |
| 13  | [State Management](#13-state-management)                              | `pipeline.py`                                |
| 14  | [Body Snapshot](#14-body-snapshot)                                    | `pipeline.py`                                |
| 15  | [Line Cross Detection](#15-line-cross-detection)                      | `pipeline.py`                                |
| 16  | [Pipeline Service Mode](#16-pipeline-service-mode)                    | `pipeline.py`                                |
| 17  | [Standalone Report (fr_report.py)](#17-standalone-report-fr_reportpy) | `fr_report.py`                               |
| 18  | [TensorRT Acceleration](#18-tensorrt-acceleration)                    | `pipeline.py`                                |

---

---

### 1. Video Fetching

#### Description

Retrieves recorded video clips from a remote Jetson device over SSH/SCP before processing begins.

#### Purpose

The recording Jetson stores `.mp4` clips on its local filesystem. The pipeline host has no direct filesystem access, so it must pull clips over the network. This step ensures all raw video is available locally before any GPU processing starts.

#### Workflow

1. Read connection details (host, user, password/SSH key, remote base path) from `config.json`.
2. SSH into the Jetson and list all `.mp4` files under `<remote_base>/<date>/`.
3. Compare the remote file list against `state.json`'s `processed_clips` list to skip already-fetched clips.
4. SCP each new file to `data/<date>/videos/` on the local machine.
5. Return the sorted list of newly fetched filenames for the pipeline to process.

#### Components Involved

- **File:** `pipeline.py`
- **Function:** `fetch_videos(date_str, state)`
- **Config keys:** `recording_jetson.host`, `.user`, `.password`, `.ssh_key`, `.video_base_path`, `.min_file_age_seconds`
- **External tools:** `sshpass`, `ssh`, `scp` (called via `subprocess`)

#### Input

- `date_str` — target date in `YYYY-MM-DD` format
- `state` — current `state.json` dict (used to skip already-processed clips)

#### Output

- Sorted list of newly downloaded `.mp4` filenames
- Video files written to `data/<date>/videos/`

#### Dependencies

- `sshpass` (optional, for password auth)
- SSH key or password configured in `config.json`
- Network access to the recording Jetson

#### Edge Cases

- If the Jetson is unreachable or the remote directory is empty, returns an empty list and logs a debug message.
- If a file already exists locally (partial download), it is treated as successfully fetched without re-downloading.
- SCP timeout is 120 seconds per file; failures are logged and the file is skipped.

#### Notes

- Use `--local-dir` to bypass this step entirely and process videos already on disk.
- The `min_file_age_seconds` guard prevents fetching clips that are still being written.

---

### 2. Frame Reading

#### Description

A background thread that reads and decodes video frames while the GPU simultaneously processes previously read frames.

#### Purpose

Video decoding (CPU) and inference (GPU) can run concurrently. Without threading, the GPU would be idle while each frame is decoded, halving throughput. The `FrameReader` thread pre-fills a queue so the GPU pipeline never stalls waiting for frames.

#### Workflow

1. `FrameReader` is initialized with a video path, a skip ratio, and a maximum queue depth.
2. On `start()`, it opens the video with `cv2.VideoCapture` and reads frames in a loop.
3. Every `skip`-th frame is placed into a `queue.Queue`; the rest are discarded to meet the target processing FPS.
4. A `None` sentinel is enqueued when the video ends.
5. The main processing loop calls `reader.queue.get()` and processes each frame on the GPU.

#### Components Involved

- **File:** `pipeline.py`
- **Class:** `FrameReader(threading.Thread)`
- **Config keys:** `video.processing_fps`, `video.expected_fps`

#### Input

- A local `.mp4` file path
- `skip` — integer frame skip ratio (e.g., skip=7 processes every 7th frame → 4 FPS from a 30 FPS source)
- `max_queue` — maximum buffered frames (default 64)

#### Output

- A queue of `(frame_index, frame)` tuples, consumed by the clip processor

#### Dependencies

- `opencv-python` (`cv2`)
- `threading`, `queue` (Python standard library)

#### Edge Cases

- If the video cannot be opened, the thread immediately enqueues the sentinel and exits.
- If the queue is full (GPU is slower than decode), the thread blocks on `queue.put()`, naturally throttling decode.
- The number of `queue empty` events is logged as a performance indicator.

#### Notes

- The skip ratio is computed as `max(1, int(native_fps / PROCESSING_FPS))` to adapt to any source FPS.
- Profiling breaks down time spent waiting on the queue versus actively processing.

---

### 3. Person Detection

#### Description

Detects all persons present in each video frame using a YOLO object detection model, returning bounding boxes for all detected individuals.

#### Purpose

Person bounding boxes are the entry point for all downstream processing. Only pixels within these boxes are passed to the re-ID tracker and face detector, keeping compute focused on people rather than the full frame.

#### Workflow

1. Load `yolo11l.pt` (YOLO11 Large) at startup; move to GPU if available.
2. For each processed frame, call `model.track()` with `classes=[0]` (person class only), confidence threshold `CONF_THRESHOLD` (default 0.45), and `half=True` for FP16 inference.
3. Extract bounding boxes from the result as `xyxy` coordinates.
4. Pass boxes to the tracker's `match_or_create()` method.

#### Components Involved

- **File:** `pipeline.py`
- **Class:** `ClipProcessor`
- **Method:** `initialize()`, `process_clip()`
- **Model:** `models/yolo11l.pt`
- **Library:** `ultralytics` (YOLO)
- **Config keys:** `tracking.conf_threshold`

#### Input

- Video frame (BGR NumPy array, typically 1920×1080 or 1280×720)

#### Output

- List of bounding boxes `[x1, y1, x2, y2]` for all detected persons in the frame

#### Dependencies

- `ultralytics` package
- CUDA GPU (falls back to CPU)
- `models/yolo11l.pt` — must be present before running

#### Edge Cases

- If no persons are detected in a frame, the tracker's lost-track logic runs to finalize persons who have disappeared.
- Very small detections (below `min_crop_w` × `min_crop_h`) are discarded downstream by the OSNet extractor.

#### Notes

- Half-precision (`half=True`) roughly doubles GPU throughput with negligible accuracy loss.
- The model can optionally be exported to a TensorRT `.engine` for further speedup (see [TensorRT Acceleration](#18-tensorrt-acceleration)).
- YOLO's built-in `.track()` mode is used but the pipeline's own `PersonTracker` overrides identity assignment.

---

### 4. Person Tracking (Re-ID)

#### Description

Assigns consistent identity IDs to detected persons across frames and across video clips using appearance-based re-identification with OSNet embeddings.

#### Purpose

YOLO detection alone treats each frame independently and loses identity between clips. The custom `PersonTracker` maintains identity continuity throughout the day by matching new detections to existing tracks using appearance embeddings rather than positional heuristics alone.

#### Workflow

1. **OSNet extraction:** For all detected bounding boxes in a frame, extract 512-dimensional L2-normalized appearance embeddings in a single GPU batch call.
2. **Phase 1 — match to active tracks:** Build a cosine distance matrix between new embeddings and existing track reference embeddings. Greedily assign the best match (lowest distance) below `REID_DISTANCE` threshold (default 0.65), subject to a bounding-box jump gate.
3. **Phase 2 — match to pending tracks:** Unmatched detections are compared against pending (warmup) tracks. If a pending track's `frames_seen` reaches `WARMUP_FRAMES` (default 10), it is promoted to active.
4. **Phase 3 — new pending:** Any remaining unmatched detection creates a new pending track.
5. **Lost track pruning:** Tracks not matched for `MAX_LOST_SECONDS` (default 25s) are finalized and removed.
6. **Cross-clip continuity:** At the end of each clip, the tracker state (embedding bank `.npy` files + metadata) is serialized to `data/<date>/tracks/`. The next clip deserializes this state, so a person who appears across multiple clips gets the same track ID.

#### Components Involved

- **File:** `pipeline.py`
- **Classes:** `OSNetExtractor`, `PersonTracker`, `ActiveTrack`, `PendingTrack`
- **Model:** `models/osnet_x1_0_msmt17.pt`
- **Library:** `boxmot` (ReidAutoBackend)
- **Config keys:** `tracking.reid_distance` (0.65), `.warmup_frames` (10), `.bank_size` (10), `.update_gate` (0.8), `.max_box_jump` (0.3), `.max_lost_seconds` (25)

#### Input

- Frame (BGR NumPy array) + list of person bounding boxes for that frame
- Timestamp of the current frame
- Optional `gap_seconds` between clips (relaxes the box-jump gate during long gaps)

#### Output

- List of `(track_id, bounding_box)` pairs for all matched or newly promoted persons
- Updated tracker state serialized to `tracks/` after each clip

#### Dependencies

- `boxmot` package (OSNet Re-ID backend)
- `torch` (GPU inference)
- `numpy`

#### Edge Cases

- If OSNet fails to load, the extractor falls back to random L2-normalized vectors (tracking degrades gracefully but meaningfully).
- The box-jump gate is scaled up proportionally for large inter-frame gaps (e.g., across clips) to avoid rejecting valid matches.
- Embeddings are only added to a track's bank when the distance is below `REID_DISTANCE × UPDATE_GATE`, preventing drift from low-quality detections.

#### Notes

- Each active track maintains a rolling bank of up to `BANK_SIZE` (default 10) embeddings; the reference vector is a recency-weighted average (weights from 0.5 to 1.0, oldest to newest).
- The warmup gate (10 frames minimum) filters spurious YOLO detections that appear only briefly.

---

### 5. Face Detection

#### Description

Detects faces within each tracked person's body crop, extracts the highest-confidence face, and captures it as a JPEG image for later recognition.

#### Purpose

Full-frame face detection would generate many false positives and be computationally expensive. Running a dedicated face detector on tight person crops isolates the search space, improves accuracy, and produces clean face images ready for embedding extraction.

#### Workflow

1. For each tracked person in the current frame, extract the person body crop from the frame.
2. Apply a 1.5-second cooldown between face capture attempts per person to avoid near-duplicate crops.
3. Run `yolov11s-face.pt` on the body crop with confidence 0.3 and `half=True`.
4. Select the highest-confidence detected face.
5. Extract 5-point facial keypoints (landmarks) if available.
6. Pad the face bounding box by 10 pixels on all sides.
7. Score the face crop (see [Face Quality Scoring](#6-face-quality-scoring)).
8. Save asynchronously to `data/<date>/faces/person_XXXX/face_XXXXXX_qNN.jpg` where `NN` is the quality score.
9. If the person already has `MAX_FACES` (default 10) crops, replace the lowest-quality one if the new crop scores higher.

#### Components Involved

- **File:** `pipeline.py`
- **Class:** `ClipProcessor`
- **Method:** `_batch_capture_faces()`
- **Model:** `models/yolov11s-face.pt` (YOLO face detector with keypoint heads)
- **Config keys:** `face_capture.max_faces_per_person`

#### Input

- Person body crop (BGR NumPy array)
- Current timestamp

#### Output

- JPEG face images saved under `data/<date>/faces/person_XXXX/`
- Updated `person['face_images']` list of `(quality_score, filepath)` tuples

#### Dependencies

- `ultralytics` (YOLO face model)
- `opencv-python` (`cv2.imwrite`)
- `ThreadPoolExecutor` (async file writes)

#### Edge Cases

- If no face is detected in a body crop, the attempt is skipped silently.
- If the face crop is smaller than 20×20 pixels, it is discarded.
- File writes are done asynchronously (`_face_save_pool`) to prevent blocking the GPU pipeline.

#### Notes

- The quality score `NN` is embedded in the filename so that downstream tools (FR, filter, report) can parse it without reloading the image.
- The face detector uses the small YOLO variant (`yolov11s`) for speed on tight crops where high recall matters more than detecting distant faces.

---

### 6. Face Quality Scoring

#### Description

Assigns a composite numeric quality score (0–100) to each captured face crop based on multiple image-level metrics.

#### Purpose

Not all captured face crops are usable for recognition — some are blurry, too dark, too small, or at extreme angles. The quality score drives which faces are kept when storage is limited and which are given priority during FR.

#### Workflow

The `score_face()` function computes a weighted sum of five sub-scores:

| Sub-score    | Max pts | Metric                                             |
| ------------ | ------- | -------------------------------------------------- |
| `yolo`       | 20      | YOLO detection confidence × 20                     |
| `blur`       | 25      | Laplacian variance / 300, capped at 1.0            |
| `size`       | 20      | `min(face_w, face_h) / 120`, capped at 1.0         |
| `brightness` | 15      | Proximity to ideal brightness (130/255)            |
| `pose`       | 20      | Nose-to-eye symmetry ratio × ear occlusion penalty |

Hard gates reject the face outright (score = 0) if YOLO confidence < 0.4 or face dimensions < 40×40 px.

#### Components Involved

- **File:** `pipeline.py`
- **Function:** `score_face(face_crop, yolo_conf, keypoints=None)`

#### Input

- `face_crop` — BGR NumPy array of the detected face
- `yolo_conf` — YOLO detection confidence for this face
- `keypoints` — optional array of 5 facial landmarks with per-point confidence

#### Output

- `(score, breakdown)` — total score (float 0–100) and a dict of per-metric scores

#### Dependencies

- `opencv-python` (`cv2.Laplacian`, `cv2.cvtColor`)
- `numpy`

#### Edge Cases

- If keypoints are not available or have low confidence (< 0.3), the pose sub-score defaults to 10 (neutral, no penalty).
- Extremely bright or dark faces get reduced brightness scores; the ideal is 130/255 brightness.

#### Notes

- The scoring is "soft" by design: no sub-score alone causes rejection except the hard gates. This ensures the top-N selection logic (not hard cutoffs) governs what is kept at capture time.
- The score is embedded in the filename (`_qNN.jpg`) for fast retrieval without re-loading the image.

---

### 7. Face Filtering

#### Description

A multi-gate filtering pass that removes low-quality, non-face, or anatomically invalid face crops from the captured image set, either at FR time (integrated) or as a standalone post-processing step.

#### Purpose

YOLO face detection occasionally captures non-face regions (hands, clothing, background objects) that pass the quality scoring stage. Filtering before embedding extraction prevents these from corrupting recognition results.

#### Workflow

Three filters are applied in order from cheapest to most expensive:

1. **Quality floor (filename-based):** Parse the `_qNN` quality score from the filename. Reject if `q < min_quality` (default 25). No image load needed.
2. **Skin-tone check (HSV, no model):** Load the image. Crop the center 60% region. Convert to HSV. Compute the fraction of pixels matching skin-tone HSV ranges (H: 0–25 or 160–180, S: 30–180, V: 60–255). Reject if below `min_skin_ratio` (default 0.20).
3. **InsightFace keypoint validation (model-based):** Run InsightFace on the crop. Require at least 3 of the 5 facial landmarks to fall within the face bounding box (with a 20% margin). Reject if fewer landmarks are valid.

Rejected images are moved to a `rejected/` subfolder inside each person's folder, renamed with the rejection reason prefix (`REJ_<reason>__<original_name>.jpg`).

When run inside the pipeline (`run_fr_processing`), a fourth gate also applies: 4. **Top-N cap:** After filtering, only the top `POST_FILTER_KEEP_TOP_N` (default 20) faces by quality score are kept; excess images are moved to `rejected/` as well.

#### Components Involved

- **File:** `filter_faces.py` (standalone), `pipeline.py` (`run_fr_processing`)
- **Functions:** `check_quality()`, `check_skin_tone()`, `check_keypoints()`, `_check_skin_tone()`, `_count_kps_in_bbox()`
- **Library:** `insightface` (AntelopeV2), `opencv-python`
- **Config keys:** `face_capture.min_keypoints`, `.min_kp_conf`, `.min_skin_ratio`, `.min_quality`, `.post_filter_enabled`, `.post_filter_keep_top_n`

#### Input

- Directory of face JPEG images for a given date
- Tunable thresholds (min keypoints, min skin ratio, min quality)
- `--dry-run` flag to preview without moving files

#### Output

- Valid face images remain in-place
- Rejected images moved to `person_XXXX/rejected/REJ_<reason>__<filename>.jpg`
- Console summary: kept / rejected counts broken down by filter type

#### Dependencies

- `insightface` (AntelopeV2 ONNX model, downloaded on first run)
- `opencv-python`
- `numpy`

#### Edge Cases

- If InsightFace cannot detect a face in a crop at all, the crop is rejected as `no_face_detected`.
- `--dry-run` mode logs all decisions but moves nothing — safe for previewing impact before committing.
- The skin-tone check covers light to dark skin tones by using two HSV ranges and focuses only on the center crop to avoid hair/background bias.

#### Notes

- Running `filter_faces.py` standalone (after the pipeline) is useful when you want to tune thresholds without rerunning the full pipeline.
- The integrated version inside `run_fr_processing` runs at FR time, not at capture time, so all raw crops are preserved until FR.

---

### 8. Face Recognition

#### Description

Matches each person's best available face embeddings against a pre-built FAISS index of known staff embeddings to determine whether the person is a recognized staff member or an unknown customer.

#### Purpose

The core classification step of the pipeline. It separates enrolled staff (who have known identities) from customers (who are clustered by visual similarity). This drives the report's Staff Log vs. Customer Analysis sheets.

#### Workflow

1. Load the InsightFace AntelopeV2 model (`det_thresh=0.3` during pipeline; `0.15` during rerun for better recall).
2. Load the FAISS flat inner-product index from `vector_db/faiss.index` and the corresponding `staff_registry.json`.
3. For each unprocessed person, apply the [Face Filtering](#7-face-filtering) pipeline to select the best embeddings.
4. For each valid face image, run InsightFace to extract a 512-dimensional L2-normalized embedding.
5. Search the FAISS index for the nearest neighbor using inner product (equivalent to cosine similarity for L2-normalized vectors).
6. If the best similarity across all embeddings exceeds `similarity_threshold` (default 0.80, from `config.json`), classify as **Staff** with the matched name.
7. Otherwise, classify as **Customer** and pass the embedding to the clustering stage.
8. Persons with no usable embeddings are tagged `no_face`.

#### Components Involved

- **File:** `pipeline.py` (`run_fr_processing`), `rerun_fr.py` (`rerun_fr`), `fr_report.py` (`run_fr`)
- **Library:** `insightface` (AntelopeV2), `faiss-cpu`
- **Data:** `vector_db/faiss.index`, `vector_db/staff_registry.json`
- **Config keys:** `fr.similarity_threshold`, `fr.customer_cluster_threshold`

#### Input

- Directory of filtered face JPEG images per person
- Pre-built FAISS index with staff embeddings

#### Output

- Each person record annotated with `update_type` (staff/customer/no_face), `recognized_name`, and `similarity_score`
- `fr_results` dict in `state.json` with `staff`, `customer_clusters`, `no_face` lists

#### Dependencies

- `insightface` (AntelopeV2 ONNX)
- `faiss-cpu` (or `faiss-gpu`)
- `opencv-python`
- `numpy`

#### Edge Cases

- If the FAISS index is empty or missing, all persons are classified as customers.
- Multiple embeddings per person are all searched; the highest similarity score wins.
- A person appearing across multiple track IDs (due to re-ID failures) may be recognized independently under the same name and merged in the Staff results.

#### Notes

- Both pipeline and rerun use `similarity_threshold` from `config.json` (default 0.80). Rerun additionally lowers `det_thresh` to 0.15 and uses 4-strategy embedding for higher recall.
- The index uses `IndexFlatIP` (exact search); no quantization, so accuracy is lossless.

---

### 9. Customer Clustering

#### Description

Groups all persons classified as customers (unknown to the staff registry) into unique visitor identities by comparing their face embeddings using cosine similarity.

#### Purpose

A single customer may visit the store multiple times in a day, generating multiple track IDs. Clustering merges these detections into one "Visitor #N" identity, enabling accurate unique-visitor counts and repeat-visit detection.

#### Workflow

1. Collect all `(track_id, embedding, person_record)` tuples for persons classified as customers.
2. Use a greedy single-linkage approach:
   - Iterate over customers in order.
   - For each unassigned customer, start a new cluster.
   - Extend the cluster by adding any unassigned customer whose maximum cosine similarity to any existing cluster embedding exceeds `cluster_threshold` (default 0.60).
3. For each cluster, record: `cluster_id`, all `track_ids`, `first_seen`, `last_seen`, `visit_count`, and the best available face image.
4. Clusters are sorted by `visit_count` descending (most-seen visitor first).
5. Customers with `visit_count > 1` are flagged as "Repeated" in the report.

#### Components Involved

- **File:** `pipeline.py` (`run_fr_processing`), `rerun_fr.py` (`rerun_fr`), `fr_report.py` (`run_fr`)
- **Config keys:** `fr.customer_cluster_threshold` (0.60)

#### Input

- List of customer embeddings (one representative embedding per track ID)

#### Output

- List of cluster dicts, each containing `cluster_id`, `track_ids`, `visit_count`, `first_seen`, `last_seen`, `best_face`

#### Dependencies

- `numpy` (cosine similarity via `np.dot`)

#### Edge Cases

- If only one person visits, they form a single cluster with `visit_count=1`.
- A customer with a poor-quality embedding (from a blurry or partial face) may be incorrectly clustered as a separate visitor — this is the primary source of over-counting.
- Lowering `cluster_threshold` merges more aggressively; raising it splits more conservatively.

#### Notes

- The greedy algorithm is O(n²) in the number of customers. For very high-traffic days this may become slow; the current implementation is adequate for typical retail footfall.
- The representative embedding per customer is the one that had the highest similarity score during staff matching (proxy for best face quality).

---

### 10. Staff Registration

#### Description

Enrolls known staff members into the FAISS vector database by extracting face embeddings from their pipeline-captured images and storing them for future recognition.

#### Purpose

The FR stage compares unknown persons against enrolled staff. Registration is the one-time (or periodic) setup step that builds the knowledge base. It must be done after the pipeline has run and captured face images for the person to be enrolled.

#### Workflow

**CLI mode:**

1. Provide `--name "Name"` and `--ids <track_id1> <track_id2> ...`.
2. Load `state.json` for the given date to find matching person records.
3. Run InsightFace on all face images from those person folders.
4. Compute an average embedding across all extracted embeddings.
5. If the staff member already exists in the registry, rebuild the FAISS index merging old and new embeddings.
6. Add the average embedding plus up to 4 maximally diverse extra embeddings to the FAISS index for robustness.
7. Save the updated `faiss.index` and `staff_registry.json`.

**Interactive mode:**

- Display each person's best face image in an OpenCV window.
- Prompt for a name, 's' to skip, 'q' to quit, or `merge <id>` to combine multiple IDs under one name.

#### Components Involved

- **File:** `register_staff.py`
- **Functions:** `register_staff_from_ids()`, `interactive_mode()`, `extract_embeddings()`, `_pick_diverse()`
- **Data:** `vector_db/faiss.index`, `vector_db/staff_registry.json`
- **Library:** `insightface`, `faiss`

#### Input

- Date string + one or more track IDs (CLI mode)
- Or visual inspection of face images (interactive mode)

#### Output

- Updated `vector_db/faiss.index` with new staff embedding vectors
- Updated `vector_db/staff_registry.json` with staff name metadata

#### Dependencies

- `insightface` (AntelopeV2)
- `faiss-cpu`
- `opencv-python` (for interactive display)

#### Edge Cases

- If no face embeddings can be extracted from the specified person's images, registration is aborted with an error.
- Re-registering an existing name merges embeddings rather than duplicating, preventing index bloat.
- Removing a staff member (`--remove "Name"`) rebuilds the entire FAISS index from scratch, since FAISS flat indices do not support in-place deletion.

#### Notes

- Multiple track IDs per name handle the case where the same staff member was assigned different IDs across clips due to long absences.
- Up to 4 "diverse" extra embeddings (those most distant from the average) are stored alongside the mean to improve matching from different angles.

---

### 11. Re-run FR Processing

#### Description

Reprocesses face recognition for a given date using improved extraction strategies and relaxed thresholds, targeting persons who were missed or misclassified during the original pipeline run.

#### Purpose

The pipeline prioritizes speed, so FR runs with conservative settings. After the day's processing is complete, `rerun_fr.py` provides a second pass with higher recall — useful for recovering missed staff identities or improving customer clustering quality.

#### Workflow

1. Load `state.json` for the target date.
2. Select persons to reprocess — all persons (full rerun) or only `no_face` / unprocessed ones (`--retry-only`).
3. For full rerun, clear all `fr_processed`, `update_type`, `recognized_name`, `similarity_score` flags.
4. Load InsightFace with `det_thresh=0.15` (vs. 0.3 in pipeline) for higher recall.
5. For each person, attempt embedding extraction with four progressive strategies:
   - **Direct** — run InsightFace on the image as-is.
   - **Upscaled** — if image is < 200px wide/tall, upscale 2× with cubic interpolation before detection.
   - **CLAHE enhanced** — apply LAB-space CLAHE contrast enhancement for dark or washed-out images.
   - **Padded** — add 40px reflective border for faces that were cut off at the crop edge.
6. Also attempt embedding from the body snapshot as a last resort.
7. Match against FAISS with `staff_threshold=0.60` (vs. 0.80 in pipeline).
8. Cluster customers at `cluster_threshold=0.55`.
9. Save updated `state.json` and generate a new report at `data/<date>/report/fr_rerun_report_<date>.xlsx`.

#### Components Involved

- **File:** `rerun_fr.py`
- **Functions:** `rerun_fr()`, `extract_embeddings()`, `_extract_with_strategies()`, `_enhance_contrast()`
- **Library:** `insightface`, `faiss-cpu`, `opencv-python`

#### Input

- `--date` — the date to reprocess
- `--retry-only` — limit to failed persons only
- `--staff-threshold`, `--cluster-threshold` — override thresholds
- `--dry-run` — print results without saving

#### Output

- Updated `state.json` with reprocessed FR results
- `data/<date>/report/fr_rerun_report_<date>.xlsx`

#### Dependencies

- `insightface`, `faiss-cpu`, `opencv-python`, `openpyxl`

#### Edge Cases

- `--dry-run` mode prints the full classification summary without modifying `state.json` — safe for evaluating threshold sensitivity.
- If no FAISS index exists, all persons are classified as customers (no staff matching possible).
- Body snapshot fallback ensures persons who had only a body captured still have a chance of recognition.

#### Notes

- The rerun's lower similarity threshold (0.60) has higher false-positive risk. Use `--dry-run` to verify before committing.
- CLAHE operates in LAB color space to avoid affecting color (only luminance channel is enhanced).

---

### 12. Report Generation

#### Description

Produces a formatted multi-sheet Excel workbook summarizing the day's face recognition results, including face thumbnail images, timestamps, dwell times, and hourly footfall analytics.

#### Purpose

Provides a human-readable record of who was present, when, and for how long. The report is the primary deliverable for store managers and stakeholders.

#### Workflow

1. Gather `fr_results` from `state.json` (or pass results directly from FR functions).
2. Build a lookup table mapping each `track_id` to its classification (Staff / Customer / No-Face), label, and best face image path.
3. Create an Excel workbook (`openpyxl`) with the following sheets:

| Sheet                 | Content                                                                                                    |
| --------------------- | ---------------------------------------------------------------------------------------------------------- |
| **Track IDs**         | Flat list of every tracked person — ID, type, label, first/last seen, duration, face count                 |
| **Staff Log**         | One row per recognized staff member — name, first/last seen, total duration, per-detection log             |
| **Customer Analysis** | One row per unique visitor cluster — visitor label, times seen, track IDs, first/last seen, total duration |
| **No-Face Persons**   | Persons with no usable face — body snapshot, track ID, first/last seen, duration, folder path              |
| **Hourly Footfall**   | Detections bucketed by hour of day — staff / customer / no-face counts per hour                            |
| **Summary**           | Key metrics: total persons, staff count, unique customers, repeated visitors, avg dwell time, peak hour    |

4. Embed face thumbnails (80×80 JPEG) into the face column cells using `openpyxl.drawing.image`.
5. Apply color-coded row fills (green = staff, orange = customer, gray = no-face).
6. Save to `data/<date>/report/fr_report_<date>.xlsx`.
7. Clean up temporary thumbnail files.

#### Components Involved

- **File:** `pipeline.py` (`generate_report`), `rerun_fr.py` (`_generate_report`), `fr_report.py` (`generate_report`)
- **Library:** `openpyxl`, `Pillow` (PIL for thumbnail generation)

#### Input

- `state.json` contents (persons list, fr_results dict)
- Date string

#### Output

- `data/<date>/report/fr_report_<date>.xlsx` (from pipeline)
- `data/<date>/report/fr_rerun_report_<date>.xlsx` (from rerun)
- `data/<date>/report/fr_faces_report_<date>.xlsx` (from standalone fr_report.py)

#### Dependencies

- `openpyxl`
- `Pillow` (PIL) — for thumbnail generation
- Face image files must exist on disk for thumbnails to render

#### Edge Cases

- If `openpyxl` is not installed, the report step is skipped with a warning — the pipeline still completes.
- If a face image path no longer exists, its cell is left blank (no crash).
- The hourly footfall sheet is only populated for hours that have at least one detection.

#### Notes

- Three separate report generators exist (in `pipeline.py`, `rerun_fr.py`, `fr_report.py`) with slightly different sheet layouts. They all share the same visual style (blue headers, color-coded rows, 80px thumbnails).
- The `pipeline.py` report includes a "Track IDs" sheet and "Hourly Footfall" that the standalone report does not.

---

### 13. State Management

#### Description

A JSON-based persistence layer that records the complete processing state between video clips and between pipeline runs, enabling incremental processing and crash recovery.

#### Purpose

Processing a full day's footage may take hours. State management ensures the pipeline can be stopped and restarted without reprocessing already-completed clips, and that tracker identity continuity is maintained across clips.

#### Workflow

1. On startup, `load_state(date_str)` reads `data/<date>/state.json` or initializes a fresh state dict.
2. After each clip is processed, `save_state()` writes the updated state to disk atomically.
3. State includes:
   - `processed_clips` — list of already-processed filenames (skip on restart)
   - `persons` — list of finalized person records with FR results
   - `active_tracks` — serialized tracker state (embedding banks as `.npy` files in `tracks/`)
   - `next_track_id` — counter for assigning new track IDs
   - `fr_results` — final classification results

#### Components Involved

- **File:** `pipeline.py`
- **Functions:** `load_state()`, `save_state()`, `ClipProcessor.serialize_state()`, `ClipProcessor.restore_from_state()`
- **Data:** `data/<date>/state.json`, `data/<date>/tracks/track_XXXX_bank.npy`

#### Input

- Date string

#### Output

- `data/<date>/state.json` (updated after each clip)
- `data/<date>/tracks/track_XXXX_bank.npy` (embedding banks per active track)

#### Dependencies

- Python standard library (`json`, `os`)
- `numpy` (for `.npy` embedding banks)

#### Edge Cases

- If the pipeline crashes mid-clip, the state written after the previous clip is still valid — restarting will skip completed clips and resume from where processing broke off.
- The `active_tracks` in state represents in-progress (not yet finalized) persons; `persons` contains finalized ones.
- Do not hand-edit `state.json` — the `active_tracks` references `.npy` files by filename; editing either side inconsistently will corrupt tracker restore.

#### Notes

- `state.json` is the source of truth for `register_staff.py` and `rerun_fr.py`. Both tools load it to find person records and image folder paths.

---

### 14. Body Snapshot

#### Description

Captures and saves a body-level crop image for every tracked person, used as a visual fallback when no face crop was successfully captured.

#### Purpose

Some individuals may never face the camera, may wear masks, or may be too far away for face detection to succeed. The body snapshot provides a visual reference for review and is used as a last-resort input for face recognition in rerun mode.

#### Workflow

1. When a person track is first created (`_start_person`), the first valid body crop is saved immediately as `body_snapshot.jpg` in their image folder.
2. After 2 seconds of tracking the same person, the snapshot is updated once with a new crop (typically better framing after the person has settled into a position).
3. In FR processing, if no face embeddings are extracted, the body snapshot path is stored in the `no_face` list.
4. In rerun mode, the body snapshot is passed to InsightFace as a last-resort detection attempt.
5. In the Excel report, body snapshots appear in the "No-Face Persons" sheet.

#### Components Involved

- **File:** `pipeline.py`
- **Methods:** `ClipProcessor.process_clip()`, `ClipProcessor._start_person()`

#### Input

- Person body crop extracted from the detection bounding box

#### Output

- `data/<date>/faces/person_XXXX/body_snapshot.jpg`

#### Edge Cases

- Body snapshots are excluded from face filtering and FR face lists — only face crops are filtered.
- If a person is in frame for less than 2 seconds, only the initial snapshot is saved.

---

### 15. Line Cross Detection

#### Description

Detects when a tracked person crosses a configured virtual line (e.g., the store entrance), classifying the direction as "enter" or "exit".

#### Purpose

The line provides a spatial gate for entry/exit counting. Rather than tracking every person in the camera's full field of view, the pipeline can restrict attention to persons who cross the defined threshold — useful for stores with a specific entrance boundary.

#### Workflow

1. A virtual line is defined by two points (`line_a`, `line_b`) in normalized frame coordinates, configured in `config.json` under the `line` key.
2. `in_side` specifies which side of the line is "inside" the store ('above' or 'below').
3. For each tracked person, compute which side of the line their foot position (bottom-center of bounding box) falls on.
4. When the side changes between frames, emit an 'enter' or 'exit' event.
5. Line state per track ID is serialized alongside tracker state for cross-clip continuity.

#### Components Involved

- **File:** `pipeline.py`
- **Class:** `LineCrossDetector`
- **Config keys:** `line`, `in_side`

#### Input

- Normalized person foot position (cx/w, cy/h) per frame
- Track ID

#### Output

- `'enter'` or `'exit'` event when a crossing is detected, or `None` otherwise

#### Dependencies

- None (pure Python geometry)

#### Edge Cases

- Currently, the pipeline tracks all detected persons regardless of line crossing (the line gate is wired up in the class but the conditional `if uid not in self.tracked_persons` fires for all persons). The `LineCrossDetector` class is production-ready but the line-gating logic in `process_clip()` is currently configured to track all persons.
- For vertical lines (x1 == x2), the line equation uses the y-intercept directly.

---

### 16. Pipeline Service Mode

#### Description

Runs the pipeline as a long-lived background service that continuously polls for new video clips from the recording Jetson and processes them as they arrive throughout the day.

#### Purpose

The recording Jetson continuously produces video clips (one per minute or per configured interval). Service mode eliminates the need for manual intervention — the pipeline picks up and processes new clips automatically as the day progresses.

#### Workflow

1. Install SIGINT/SIGTERM signal handlers so the service exits cleanly on Ctrl+C or kill.
2. Enter an infinite loop:
   a. Compute today's date.
   b. Call `process_day()` — fetches new clips, processes them, runs FR, generates report.
   c. Sleep for `poll_interval_seconds` (default 30s).
3. Each iteration only processes clips not already in `state.json`'s `processed_clips` list.

The alternative modes are:

- `--once` — process today's clips once and exit
- `--date YYYY-MM-DD` — process a specific date's clips once and exit
- `--local-dir <path>` — skip fetching and process a local directory of videos

#### Components Involved

- **File:** `pipeline.py`
- **Function:** `main()`, `process_day()`
- **Config keys:** `recording_jetson.poll_interval_seconds`

#### Input

- CLI flags: `--once`, `--date`, `--local-dir`, `--no-fr`

#### Output

- Runs the full pipeline (fetch → track → FR → report) for each new batch of clips

#### Edge Cases

- If `process_day()` raises an unhandled exception, the error is logged as a warning and the service continues to the next poll cycle rather than crashing.
- Date rollover (midnight) is handled naturally — the loop recomputes `datetime.now()` each iteration.

---

### 17. Standalone Report (fr_report.py)

#### Description

Generates a complete FR report by scanning the `faces/` directory directly, without requiring `state.json`. It re-runs InsightFace embedding extraction and FAISS matching from scratch on whatever face images are present.

#### Purpose

Useful when `state.json` is missing, corrupted, or when the FR results in state are outdated (e.g., after adding new staff to the registry). It provides a clean-slate report from the raw face image artifacts.

#### Workflow

1. Scan `data/<date>/faces/` for `person_XXXX/` subdirectories.
2. For each folder, build a person record from the face images found (skipping folders with no face images).
3. Optionally load timestamps from `state.json` (if it exists) to populate `first_seen`/`last_seen`.
4. Run the same FR pipeline as `rerun_fr.py` (InsightFace at det_thresh=0.15, four extraction strategies, FAISS matching, customer clustering).
5. Generate the Excel report at `data/<date>/report/fr_faces_report_<date>.xlsx`.

#### Components Involved

- **File:** `fr_report.py`
- **Functions:** `scan_faces()`, `extract_embeddings()`, `run_fr()`, `generate_report()`

#### Input

- `--date` — date to process
- `--faces-dir` — optional override for faces directory
- `--staff-threshold`, `--cluster-threshold`

#### Output

- `data/<date>/report/fr_faces_report_<date>.xlsx`

#### Edge Cases

- If `state.json` does not exist, timestamps are left empty (face images are still processed).
- Body-only folders (no face images) are skipped; body snapshots are used only as fallback during embedding extraction.

#### Notes

- This tool is the most portable report generator — it only needs the `faces/` directory and the FAISS index.

---

### 18. TensorRT Acceleration

#### Description

Optionally exports the YOLO person and face detection models to NVIDIA TensorRT `.engine` format for significantly faster GPU inference.

#### Purpose

TensorRT compiles ONNX models into optimized GPU execution plans for the specific hardware on hand. For YOLO models, this typically yields 2–4× speedup over standard PyTorch FP16 inference, directly reducing pipeline processing time.

#### Workflow

1. Check `config.json` for `use_tensorrt: true`.
2. On the first run with TensorRT enabled, call `model.export(format="engine", half=True)` for both the person and face models. This takes several minutes.
3. The resulting `.engine` files are saved alongside the original `.pt` files.
4. On all subsequent runs, load the `.engine` files directly — export is skipped.
5. If TensorRT export fails (e.g., wrong CUDA toolkit version), the pipeline falls back to PyTorch inference with a warning.

#### Components Involved

- **File:** `pipeline.py`
- **Method:** `ClipProcessor.initialize()`
- **Config keys:** `use_tensorrt`

#### Input

- `config.json` with `"use_tensorrt": true`
- CUDA-capable GPU with compatible TensorRT installation

#### Output

- `models/yolo11l.engine`
- `models/yolov11s-face.engine`

#### Dependencies

- `nvidia-tensorrt` (matching CUDA version)
- `ultralytics` (TensorRT export support)

#### Edge Cases

- TensorRT engines are hardware-specific — an engine built on one GPU cannot be used on another model.
- The first-run export can take 5–15 minutes. Do not interrupt it.
- If the `.engine` file exists but is corrupt, it will fail to load; delete the file to force a rebuild.

#### Notes

- TensorRT is enabled by default (`use_tensorrt: true` in `config.json`). Disable by setting `use_tensorrt: false` or removing the key.
- Even without TensorRT, FP16 (`half=True`) inference is active by default and provides substantial speedup over FP32.

---

## Data Flow Between Features

```
Video Fetching
      │
      ▼
Frame Reading (threaded)
      │
      ▼
Person Detection (YOLO)
      │
      ▼
Person Tracking / Re-ID (OSNet)  ←──── State Management (cross-clip restore)
      │                                          │
      ├──── Body Snapshot (saved per person)     │
      │                                          │
      ▼                                          │
Face Detection (YOLO-face on body crops)         │
      │                                          │
      ▼                                          │
Face Quality Scoring                             │
      │                                          │
      ▼                                          │
Face Images saved to data/<date>/faces/          │
      │                                          │
      │            State Management (save) ◄─────┘
      ▼
Face Filtering (quality / skin-tone / keypoints)
      │
      ▼
Face Recognition (InsightFace + FAISS)
      │
      ├──── Staff match ──► Staff Registration (builds FAISS index)
      │
      └──── No match ──────► Customer Clustering
                                     │
                                     ▼
                             Report Generation (Excel)
                                     │
                             Re-run FR Processing
                             (if first pass missed faces)
```

---

_Document generated from source: `pipeline.py` v1.7.0, `register_staff.py`, `rerun_fr.py`, `filter_faces.py`, `fr_report.py`._
_Last updated: 2026-04-28._
