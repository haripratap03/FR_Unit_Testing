# Face Recognition (FR) Pipeline — Technical Documentation

**Version:** 1.7.0  
**Tech Stack:** Python · OpenCV · YOLO (Ultralytics) · OSNet (BoxMOT) · InsightFace (antelopev2) · FAISS · ONNX Runtime · openpyxl  
**Platform:** Linux / Windows · GPU recommended (CUDA, optional TensorRT)

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Architecture Overview](#2-architecture-overview)
3. [Pipeline Workflow](#3-pipeline-workflow)
4. [File-by-File Explanation](#4-file-by-file-explanation)
5. [Key Functionalities](#5-key-functionalities)
6. [Installation Guide](#6-installation-guide)
7. [Usage Instructions](#7-usage-instructions)
8. [Configuration Reference (`config.json`)](#8-configuration-reference-configjson)
9. [Module Descriptions](#9-module-descriptions)
10. [Function Responsibilities](#10-function-responsibilities)
11. [Input / Output Formats](#11-input--output-formats)
12. [Error Handling Strategy](#12-error-handling-strategy)
13. [Performance Considerations](#13-performance-considerations)
14. [Testing Strategy](#14-testing-strategy)
15. [Future Improvements](#15-future-improvements)

---

## 1. Project Overview

This project is an **offline Face Recognition pipeline** designed for retail / pharmacy store environments. It runs on a dedicated processing machine and:

- Fetches recorded video clips from a remote NVIDIA Jetson device over SSH/SCP.
- Detects and tracks every person who appears in the footage.
- Captures high-quality face crops per tracked person.
- Runs face recognition to classify each person as **staff** (named) or **customer** (anonymous, clustered by identity).
- Generates a formatted Excel report per day with face thumbnails, timestamps, dwell time, and hourly footfall.

The pipeline is fully **offline** — no cloud API calls for recognition. All models run locally via ONNX Runtime or PyTorch.

---

## 2. Architecture Overview

```
┌──────────────────────────────────────────────────────────────────────────┐
│                        OFFLINE FR PIPELINE                               │
│                                                                          │
│  Remote Jetson                  Processing Machine                       │
│  ─────────────                  ──────────────────                       │
│  /recordings/                                                            │
│    YYYYMMDD/       ──SCP──►  data/YYYY-MM-DD/videos/                     │
│    *.mp4                                                                 │
│                                    │                                     │
│                           ┌────────▼────────┐                           │
│                           │  FrameReader    │  (threaded IO)             │
│                           │  (background)   │                           │
│                           └────────┬────────┘                           │
│                                    │ frames                              │
│                           ┌────────▼────────┐                           │
│                           │  YOLO Person    │  yolo11l.pt                │
│                           │  Detection      │  GPU · half precision       │
│                           └────────┬────────┘                           │
│                                    │ bounding boxes                      │
│                           ┌────────▼────────┐                           │
│                           │  OSNet Re-ID    │  osnet_x1_0_msmt17.pt      │
│                           │  + PersonTracker│  embedding bank · 3-phase  │
│                           └────────┬────────┘                           │
│                                    │ track_id + box                      │
│                           ┌────────▼────────┐                           │
│                           │  YOLO Face      │  yolov11s-face.pt          │
│                           │  Detection      │  + quality scoring         │
│                           └────────┬────────┘                           │
│                                    │ face crops (per person folder)      │
│                                    │                                     │
│         ┌──────────────────────────▼──────────────────────────────┐     │
│         │               POST-PROCESSING                           │     │
│         │  ┌────────────┐  ┌──────────────┐  ┌────────────────┐  │     │
│         │  │filter_faces│  │InsightFace   │  │ FAISS Lookup   │  │     │
│         │  │(3 filters) │  │antelopev2    │  │ staff_registry │  │     │
│         │  └────────────┘  │(embedding)   │  └────────────────┘  │     │
│         │                  └──────────────┘                       │     │
│         │     Staff match ──► staff_results                       │     │
│         │     No match    ──► customer clustering ──► clusters    │     │
│         │     No face     ──► no_face_list                        │     │
│         └──────────────────────────┬────────────────────────────┘     │
│                                    │                                     │
│                           ┌────────▼────────┐                           │
│                           │  Excel Report   │  openpyxl                  │
│                           │  (6 sheets)     │  with face thumbnails       │
│                           └─────────────────┘                           │
└──────────────────────────────────────────────────────────────────────────┘
```

### Storage Layout

```
offline_fr/
├── config.json
├── pipeline.py
├── filter_faces.py
├── fr_report.py
├── register_staff.py
├── rerun_fr.py
├── models/
│   ├── yolo11l.pt              # Person detector
│   ├── yolov11s-face.pt        # Face detector
│   └── osnet_x1_0_msmt17.pt    # Re-ID model
├── insightface_models/
│   └── antelopev2/             # ONNX embedding models
├── vector_db/
│   ├── faiss.index             # Cosine similarity index (512-dim)
│   └── staff_registry.json    # Name → embedding metadata
├── data/
│   └── YYYY-MM-DD/
│       ├── videos/             # Fetched MP4 clips
│       ├── faces/
│       │   └── person_XXXX/   # Face crops per track
│       │       ├── face_XXXXXX_qNN.jpg
│       │       ├── body_snapshot.jpg
│       │       └── rejected/   # Filtered-out images
│       ├── tracks/             # Serialized embedding banks (.npy)
│       ├── state.json          # Full processing state
│       └── report/             # Excel output
└── logs/
```

---

## 3. Pipeline Workflow

The pipeline runs in five sequential stages per day:

### Stage 1 — Video Fetch
`fetch_videos()` lists MP4 files on the remote Jetson via SSH, then SCPs any files not yet in `state["processed_clips"]`. Supports SSH key or password authentication, configurable polling interval.

### Stage 2 — Person Detection & Tracking
For each video clip (sorted chronologically):
1. A background thread (`FrameReader`) decodes frames while the GPU processes the previous batch.
2. YOLOv11l detects persons (class 0) at configurable FPS (decimating native FPS).
3. OSNet extracts 512-dim re-ID embeddings for all detected bounding boxes in a single GPU batch call.
4. `PersonTracker` matches embeddings to existing tracks using **3-phase Hungarian-style matching**:
   - **Phase 1:** Match to active tracks (cosine distance + spatial jump gate).
   - **Phase 2:** Match to pending tracks (warmup buffer — requires `warmup_frames` observations before promoting to active).
   - **Phase 3:** Create new pending tracks for unmatched detections.
5. Tracks not seen for `max_lost_seconds` are finalized and their person record stored in state.

### Stage 3 — Face Capture
Per tracked person, per frame (with a 1.5s capture cooldown):
1. YOLOv11s-face detects faces inside the person's body crop.
2. `score_face()` computes a composite quality score (YOLO confidence, Laplacian blur, size, brightness, facial symmetry via keypoints).
3. The best-N face images (by quality score) are kept. Lower-quality images are asynchronously deleted.
4. A body snapshot is saved immediately and updated at 2 seconds for a better-quality body image.

### Stage 4 — Face Recognition
`run_fr_processing()` (inside `pipeline.py`) processes all finalized persons:

1. **Filter 1 — Quality floor:** Reject faces below `min_quality` score (parsed from filename).
2. **Filter 2 — Skin-tone check:** Center-crop HSV analysis; reject if skin pixel ratio < `min_skin_ratio`.
3. **Filter 3 — InsightFace keypoint validation:** Reject if fewer than `min_keypoints` landmarks fall within the face bounding box.
4. **Embedding:** InsightFace (antelopev2) extracts a 512-dim L2-normalized embedding per accepted face.
5. **Staff matching:** FAISS `IndexFlatIP` inner-product search (≡ cosine similarity) against the staff registry. Threshold: `similarity_threshold` (default 0.80).
6. **Customer clustering:** Unmatched persons are greedy-clustered by pairwise cosine similarity ≥ `customer_cluster_threshold` (default 0.60). Each cluster = one unique visitor identity.

### Stage 5 — Report Generation
`generate_report()` writes a 6-sheet Excel workbook:

| Sheet | Contents |
|---|---|
| Track IDs | Flat list of all tracked persons with face thumbnail, type, timestamps, duration |
| Staff Log | Per-staff detection log with total dwell time |
| Customer Analysis | Unique visitors, repeated-visitor flag, detection log |
| No-Face Persons | Body snapshots only, no FR possible |
| Hourly Footfall | Hour-by-hour breakdown of staff/customer/no-face counts |
| Summary | Aggregate metrics: total persons, staff, customers, avg dwell, peak hour |

---

## 4. File-by-File Explanation

### `pipeline.py`
The main orchestrator. Runs as either a **continuous service** (polling for new clips every N seconds) or a **one-shot** processor for a specific date or local directory. Contains all core classes and the full FR + reporting logic inline. Self-contained — can run end-to-end without any other script.

### `filter_faces.py`
A standalone post-processing tool that applies the same 3-stage face filter to an already-processed date's face folder. Useful for re-filtering after tuning thresholds without re-running the full pipeline. Supports `--dry-run` to preview rejections before moving files.

### `register_staff.py`
An admin tool for building and maintaining the staff registry. Extracts InsightFace embeddings from face images in a person's folder, computes an average embedding, and adds it to the FAISS index. Supports:
- Direct mode (`--name` + `--ids`)
- Interactive mode with face preview window
- Merge (multiple track IDs → same staff member)
- Update existing entries (rebuilds FAISS)
- Remove staff

### `rerun_fr.py`
An improved re-run of just the FR stage on an existing processed date. Uses more aggressive detection strategies (lower threshold, upscaling, CLAHE, padding) than the inline pipeline FR. Also generates its own Excel report with a dedicated **No Face** tab.

### `fr_report.py`
A completely standalone report generator that requires only the `faces/` directory — does **not** need `state.json`. Directly scans person folders, runs FR, and outputs an Excel report. Useful when state is missing or corrupted.

### `config.json`
All runtime parameters in one place. See [Section 8](#8-configuration-reference-configjson) for full details.

### `requirements.txt`
Minimal production dependencies. Does not pin versions — install with `pip install -r requirements.txt` into a virtual environment.

---

## 5. Key Functionalities

### Re-ID Based Cross-Clip Tracking
Tracker state (embedding bank `.npy` files + metadata) is serialized to `state.json` after each clip. The next clip restores state and continues tracking the same track IDs, even across a 90-second gap between recordings.

### Warmup Buffer (Ghost Prevention)
New detections enter a `PendingTrack` buffer. Only after `warmup_frames` consecutive observations does a detection get promoted to an `ActiveTrack` with a stable ID. This prevents spurious one-frame detections from polluting the report.

### Embedding Bank with Recency Weighting
Each active track maintains a rolling bank of the last N embeddings (`bank_size`). The reference embedding used for matching is a recency-weighted average (linear weights 0.5 → 1.0, newest = highest weight), making the tracker robust to appearance changes.

### Tiered Face Quality Capture
Face capture uses a competition model: only the top-N quality faces (by composite score) are kept per person. If a higher-quality face arrives and the buffer is full, the lowest-quality image is evicted and deleted. Quality score encodes YOLO confidence, sharpness (Laplacian variance), face size, brightness, and frontal pose symmetry.

### Multi-Strategy Embedding Extraction (rerun/fr_report)
When the standard detection fails (angled faces, small crops, poor lighting), the re-run scripts try: direct → upscale → CLAHE contrast enhancement → edge padding. This substantially increases FR recall on difficult images.

### Diverse Embedding Storage in FAISS
When registering a staff member, the pipeline stores an average embedding plus up to 4 maximally-diverse additional embeddings per person. This improves recall for faces captured at different angles or lighting conditions.

---

## 6. Installation Guide

### Prerequisites
- Python 3.9+
- NVIDIA GPU with CUDA 11.8+ (strongly recommended; CPU fallback available)
- `sshpass` on the processing machine (for password-based SCP)
- Jetson recording device accessible over LAN

### Step 1 — Clone / Copy Project

```bash
git clone <repo-url> offline_fr
cd offline_fr
```

### Step 2 — Create Virtual Environment

```bash
python -m venv myenv
# Linux/Mac
source myenv/bin/activate
# Windows
myenv\Scripts\activate
```

### Step 3 — Install Dependencies

```bash
pip install -r requirements.txt
# GPU: install matching torch version first
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Step 4 — Download Models

Place the following files in the `models/` directory:
- `yolo11l.pt` — Ultralytics YOLO11 large person detector
- `yolov11s-face.pt` — YOLO face detector with keypoints
- `osnet_x1_0_msmt17.pt` — OSNet Re-ID model (BoxMOT)

Place InsightFace models in `insightface_models/antelopev2/`:
- `det_10g.onnx` — detection
- `w600k_r50.onnx` — recognition embedding

InsightFace will auto-download these on first run if internet is available.


### Step 5 — Configure

Edit `config.json` with your Jetson SSH credentials and store name. See [Section 8](#8-configuration-reference-configjson).

### Step 6 — Register Staff (Optional, but needed for FR to label staff)

```bash
python register_staff.py --date 2026-04-09 --list        # See who was detected
python register_staff.py --date 2026-04-09 --name "Alice" --ids 3 7
```

---

## 7. Usage Instructions

### Run the Pipeline

```bash
# Continuous service (polls Jetson every 30s)
python pipeline.py

# Process once for today and exit
python pipeline.py --once

# Process a specific past date
python pipeline.py --date 2026-04-09 --once

# Process local video directory (skip fetch)
python pipeline.py --local-dir ./videos/2026-04-09

# Skip FR processing (tracking + face capture only)
python pipeline.py --once --no-fr
```

### Filter Face Crops

```bash
# Default thresholds
python filter_faces.py --date 2026-04-09

# Preview mode (no file moves)
python filter_faces.py --date 2026-04-09 --dry-run

# Relax thresholds
python filter_faces.py --date 2026-04-09 --min-keypoints 2 --min-skin 0.15 --min-quality 20
```

### Register Staff

```bash
# Interactive: shows each person's face
python register_staff.py --date 2026-04-09 --interactive

# Direct assignment
python register_staff.py --date 2026-04-09 --name "Ravi" --ids 3 7 12

# Merge two track IDs under same name
python register_staff.py --date 2026-04-09 --interactive
# (then type: merge <id> at the prompt)

# Show current registry
python register_staff.py --show-registry

# Remove staff member
python register_staff.py --remove "Ravi"
```

### Re-run FR with Improved Settings

```bash
# Full re-run on all persons
python rerun_fr.py --date 2026-04-09

# Only retry persons that failed FR (no_face / unprocessed)
python rerun_fr.py --date 2026-04-09 --retry-only

# Custom thresholds
python rerun_fr.py --date 2026-04-09 --staff-threshold 0.55 --cluster-threshold 0.50

# Preview without saving
python rerun_fr.py --date 2026-04-09 --dry-run

# List current FR state
python rerun_fr.py --date 2026-04-09 --list
```

### Standalone Report (no state.json required)

```bash
python fr_report.py --date 2026-04-09
python fr_report.py --date 2026-04-09 --faces-dir /path/to/faces
```

---

## 8. Configuration Reference (`config.json`)

```jsonc
{
  "store_name": "StoreName",              // Label in Excel reports

  "recording_jetson": {
    "host": "192.168.1.100",              // Jetson IP
    "user": "jetson",                    // SSH username
    "password": "",                      // SSH password (use ssh_key instead if possible)
    "ssh_key": "",                       // Path to private key file
    "video_base_path": "/home/.../",     // Remote parent directory of YYYY-MM-DD folders
    "poll_interval_seconds": 30,         // Continuous mode polling interval
    "min_file_age_seconds": 10           // Skip files still being written
  },

  "video": {
    "expected_fps": 25,                  // Native camera FPS (fallback if unreadable)
    "processing_fps": 10                 // Frames actually processed (decimation)
  },

  "tracking": {
    "reid_distance": 0.65,              // Max cosine distance to match a detection to track
    "warmup_frames": 10,                // Frames before pending → active
    "bank_size": 10,                    // Embedding bank rolling window size
    "update_gate": 0.8,                 // Only update bank if distance < threshold * gate
    "max_box_jump": 0.3,                // Max normalized bbox centre displacement between frames
    "max_lost_seconds": 25,             // Finalize track after this many seconds unseen
    "conf_threshold": 0.45,             // YOLO person detection confidence floor
    "min_crop_w": 32,                   // Minimum body crop width for OSNet
    "min_crop_h": 64                    // Minimum body crop height for OSNet
  },

  "face_capture": {
    "max_faces_per_person": 10          // Competition buffer size (cap per track)
    // post_filter_keep_top_n, min_keypoints, min_kp_conf, min_skin_ratio, min_quality
    // use code defaults (20, 3, 0.5, 0.20, 25) — omitted from config
  },

  "fr": {
    "similarity_threshold": 0.80,       // Cosine similarity to match as staff
    "customer_cluster_threshold": 0.60  // Cosine similarity to merge customer tracks
  },

  "line": {                             // Virtual line for entry direction (reserved)
    "A": {"x": 0.0, "y": 0.5},
    "B": {"x": 1.0, "y": 0.5}
  },
  "in_side": "below",                   // Which side of line = "inside" (reserved)

  "telegram": {                         // Push notifications (reserved, not yet wired)
    "bot_token": "",
    "chat_id": ""
  }
}
```

---

## 9. Module Descriptions

### `OSNetExtractor`
Wraps the BoxMOT `ReidAutoBackend` to extract 512-dim L2-normalized body embeddings. Falls back to random unit vectors if the model fails to load (for testing without GPU). Supports single-crop and batch extraction; batch extraction is used in the main pipeline for GPU efficiency.

### `PersonTracker`
Stateful multi-object tracker using Re-ID embeddings. Maintains two pools:
- `pending`: unconfirmed detections (warmup buffer)
- `active`: confirmed tracks with rolling embedding banks

Matching is 3-phase greedy Hungarian (not true linear assignment, but correct for typical store densities < 20 persons in frame).

### `ActiveTrack`
Stores the rolling embedding bank, last bounding box, last timestamp, and lost-seconds counter. Serializes/deserializes to `.npy` + JSON for cross-clip state persistence.

### `PendingTrack`
Lightweight buffer for unconfirmed detections. Promoted to `ActiveTrack` after `warmup_frames` observations.

### `ClipProcessor`
Loads models, holds tracker state, and drives per-clip frame processing. Manages the async face-save thread pool to avoid GPU stalls on disk IO.

### `LineCrossDetector`
Detects when a tracked person crosses a configurable virtual line (configured but currently bypassed — every detected person is tracked regardless of line crossing).

### `FrameReader`
Background thread that decodes video frames and puts them in a bounded queue. Decouples disk IO from GPU inference, preventing pipeline stalls.

---

## 10. Function Responsibilities

### `pipeline.py`

| Function | Responsibility |
|---|---|
| `load_config()` | Read `config.json`, expose all subsections as module-level constants |
| `parse_video_timestamp(filename)` | Parse `YYYYMMDD_HHMMSS.mp4` filename into `datetime` |
| `frame_timestamp(clip_start, frame_idx, fps)` | Compute wall-clock time for any frame index |
| `fetch_videos(date_str, state)` | SSH list + SCP new clips from Jetson |
| `load_state / save_state` | Read/write `state.json` (tracks, persons, FR results) |
| `score_face(face_crop, yolo_conf, keypoints)` | Multi-factor face quality score (0–100) |
| `_check_skin_tone(face_crop)` | HSV-based skin pixel ratio check |
| `_count_kps_in_bbox(kps, bbox)` | Count InsightFace landmarks inside bounding box |
| `_move_to_rejected(folder, fname, reason)` | Move rejected image to `rejected/` subfolder |
| `ClipProcessor.initialize()` | Load YOLO + OSNet models; optional TensorRT export |
| `ClipProcessor.process_clip()` | Main per-clip loop: detect → track → capture faces |
| `ClipProcessor._batch_capture_faces()` | Run face model on all person crops; manage quality buffer |
| `ClipProcessor._finalize_person()` | Close a track, return serializable person record |
| `run_fr_processing(date_str, state)` | Filter → embed → FAISS match → cluster → save results |
| `generate_report(date_str, state)` | Build 6-sheet Excel workbook with face thumbnails |
| `process_day(date_str, ...)` | Top-level orchestrator: fetch → process clips → FR → report |
| `main()` | CLI entry point; continuous or one-shot mode |

### `filter_faces.py`

| Function | Responsibility |
|---|---|
| `check_keypoints(face_app, img, ...)` | InsightFace detect; count valid landmarks |
| `check_skin_tone(img, min_skin_ratio)` | HSV mask; compute skin pixel fraction |
| `check_quality(filename, min_quality)` | Parse quality score from filename |
| `filter_faces(faces_dir, ...)` | Walk all person folders; apply 3 filters; move rejects |

### `register_staff.py`

| Function | Responsibility |
|---|---|
| `extract_embeddings(face_app, folder)` | Extract + L2-normalize InsightFace embeddings from folder |
| `register_staff_from_ids(date_str, name, ids)` | Load state → extract embeddings → upsert FAISS + registry |
| `_pick_diverse(embeddings, avg, max_extra)` | Select most angle-diverse embeddings for robustness |
| `list_persons(date_str)` | Print tabular summary of all persons from a date |
| `show_registry()` | Print current staff registry grouped by name |
| `remove_staff(name)` | Rebuild FAISS index without specified staff member |
| `interactive_mode(date_str)` | CV2 window face review + CLI naming loop |

### `rerun_fr.py`

| Function | Responsibility |
|---|---|
| `extract_embeddings(face_app, folder, body_snap)` | Try 4 detection strategies per image |
| `_extract_with_strategies(face_app, img)` | Direct → upscale → CLAHE → padding cascade |
| `_detect_face(face_app, img)` | InsightFace detect; return normalized embedding of largest face |
| `_enhance_contrast(img)` | CLAHE on LAB L-channel |
| `rerun_fr(date_str, ...)` | Load state → re-embed → match → cluster → save → report |
| `list_persons(date_str)` | Tabular print of current FR state |

### `fr_report.py`

| Function | Responsibility |
|---|---|
| `scan_faces(faces_dir, date_str)` | Build person list from `person_XXXX/` directories |
| `load_state_timestamps(date_str)` | Merge first_seen/last_seen from `state.json` into person dicts |
| `extract_embeddings(face_app, person)` | Multi-strategy embedding extraction per person |
| `run_fr(persons, ...)` | FAISS match + greedy customer clustering |
| `generate_report(date_str, ...)` | Write 4-sheet Excel report |

---

## 11. Input / Output Formats

### Input

| Item | Format | Notes |
|---|---|---|
| Video clips | MP4 (H.264) | Named `YYYYMMDD_HHMMSS.mp4` |
| config.json | JSON | Must exist in script directory |
| YOLO models | `.pt` or `.engine` | Ultralytics format |
| OSNet model | `.pt` | BoxMOT ReidAutoBackend |
| InsightFace models | `.onnx` | antelopev2 pack |
| FAISS index | `faiss.index` | `IndexFlatIP`, 512-dim float32 |
| Staff registry | `staff_registry.json` | List of `{name, person_ids, embedding_count, ...}` |

### Output

| Item | Format | Location |
|---|---|---|
| Face crop images | JPEG | `data/<date>/faces/person_XXXX/face_XXXXXX_qNN.jpg` |
| Body snapshot | JPEG | `data/<date>/faces/person_XXXX/body_snapshot.jpg` |
| Rejected faces | JPEG | `data/<date>/faces/person_XXXX/rejected/REJ_<reason>__<filename>.jpg` |
| Embedding banks | `.npy` | `data/<date>/tracks/track_XXXX_bank.npy` |
| Processing state | JSON | `data/<date>/state.json` |
| Pipeline report | Excel `.xlsx` | `data/<date>/report/fr_report_<date>.xlsx` |
| Rerun report | Excel `.xlsx` | `data/<date>/report/fr_rerun_report_<date>.xlsx` |
| Standalone report | Excel `.xlsx` | `data/<date>/report/fr_faces_report_<date>.xlsx` |

### `state.json` Schema (key fields)

```json
{
  "date": "YYYY-MM-DD",
  "processed_clips": ["20260409_093000.mp4", "..."],
  "next_track_id": 42,
  "active_tracks": { "1": { "bank_file": "...", "last_box": [...], ... } },
  "persons": [
    {
      "track_id": 1,
      "first_seen": "2026-04-09T09:30:05",
      "last_seen": "2026-04-09T09:32:10",
      "images_folder": "data/2026-04-09/faces/person_0001",
      "face_images_count": 12,
      "best_face_image": "data/.../face_000123_q78.jpg",
      "body_snapshot": "data/.../body_snapshot.jpg",
      "fr_processed": true,
      "update_type": "staff",
      "recognized_name": "Alice",
      "similarity_score": 0.8423
    }
  ],
  "fr_results": {
    "staff": [...],
    "customer_clusters": [...],
    "no_face": [...],
    "no_face_count": 5
  }
}
```

---

## 12. Error Handling Strategy

| Scenario | Handling |
|---|---|
| Cannot open video | Log warning, skip clip — continue to next |
| SSH/SCP failure | Log warning, return empty list — retry on next poll |
| Cannot parse clip timestamp | Log warning, skip clip |
| YOLO model not found | Exception at startup — hard fail (pipeline cannot run) |
| OSNet model not found | Fall back to random embeddings (tracking still works, Re-ID quality degraded) |
| InsightFace model missing | Exception during FR — caught, FR skipped, report still generated |
| Face image unreadable | Skip that image, continue with others |
| openpyxl missing | Log warning, skip report generation — state still saved |
| Corrupted track bank file | Log warning, skip that track on restore |
| Track finalization error | Silently skip (no `None` persons appended) |
| FR processing exception | Caught at `process_day` level — state saved, report attempted |
| Report generation exception | Caught at `process_day` level — pipeline continues |

All critical model loading happens in `ClipProcessor.initialize()`. Model failures are separated from data failures so that partial results are always persisted to `state.json`.

---

## 13. Performance Considerations

### GPU Utilization
- All YOLO inference runs with `half=True` (FP16) on CUDA, roughly 2× speedup over FP32.
- OSNet batch extraction processes all persons in a frame in one GPU call.
- TensorRT export is optionally available (`"use_tensorrt": true` in config) for a further 1.5–2× speedup on Jetson/RTX; the first run triggers a one-time export.

### CPU/GPU Overlap
- `FrameReader` runs in a background daemon thread, decoding the next batch of frames while the GPU processes the current batch. On disk-bound systems this eliminates read stalls.
- Face file writes are submitted to a 4-thread `ThreadPoolExecutor`, keeping GPU processing unblocked by disk IO.

### Frame Decimation
The pipeline processes only `processing_fps` frames per second of video (e.g., 10 from 30-fps source), skipping every `skip=3` frames. This reduces GPU load 3× with minimal tracking quality loss for typical human movement speeds.

### Processing FPS Benchmark
At 10 processing FPS on a mid-range GPU (RTX 3060):
- Frame decode queue wait: ~15% of time
- YOLO person detection: ~40% of time [GPU]
- OSNet Re-ID: ~25% of time [GPU+CPU]
- Track/crop overhead: ~10% of time [CPU]
- Face detection: ~10% of time [GPU]

### Embedding Bank Sizing
`bank_size=10` balances memory per track against identity stability. Increasing it improves long-term identity consistency at the cost of more RAM per active track.

---

## 15. Future Improvements

| Priority | Improvement |
|---|---|
| High | Replace greedy customer clustering with proper DBSCAN or hierarchical clustering for more accurate unique visitor counts |
| High | True linear assignment (scipy `linear_sum_assignment`) in `PersonTracker` for correctness at high crowd densities |
| High | Unit test coverage for all pure functions (see Section 14) |
| Medium | Telegram bot integration (config keys already present) to push daily summary after report generation |
| Medium | TensorRT auto-detection and pre-export pipeline so first-run penalty is explicit |
| Medium | Multi-camera support — correlate track IDs across overlapping camera views using shared FAISS space |
| Medium | Web dashboard (FastAPI + React) to view daily reports without opening Excel |
| Low | Replace JSON state with SQLite for concurrent access safety and faster queries on large person lists |
| Low | Configurable line-crossing gate (currently bypassed — every person is tracked) |
| Low | Age/gender attribute extraction from InsightFace for richer customer analytics |
| Low | Automated model version management (download + checksum verification on startup) |
