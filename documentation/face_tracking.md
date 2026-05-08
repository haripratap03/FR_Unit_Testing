# Face Tracking Module Documentation

**Project:** Offline Face Recognition (FR) Pipeline  
**Version:** 1.7.0  
**Source:** `pipeline.py`  
**Audience:** Computer Vision Engineers · Developers · QA Engineers

---

## Table of Contents

1. [Overview](#1-overview)
2. [Pipeline Position](#2-pipeline-position)
3. [Core Concepts](#3-core-concepts)
4. [Tracking Architecture](#4-tracking-architecture)
5. [Workflow](#5-workflow)
6. [Detailed Tracking Logic](#6-detailed-tracking-logic)
7. [Key Classes](#7-key-classes)
8. [Key Functions](#8-key-functions)
9. [Track Lifecycle](#9-track-lifecycle)
10. [Matching Strategy](#10-matching-strategy)
11. [Embedding Management](#11-embedding-management)
12. [Input / Output](#12-input--output)
13. [Edge Cases & Failure Handling](#13-edge-cases--failure-handling)
14. [Performance Considerations](#14-performance-considerations)
15. [Integration with Other Modules](#15-integration-with-other-modules)

---

## 1. Overview

The Face Tracking module assigns and maintains consistent integer identities (**track IDs**) for all persons detected in a video stream. Its core responsibility is identity continuity: ensuring that a person walking through a pharmacy store receives the same `track_id` across every frame they appear in, across multiple one-minute video clips, and even across pipeline restarts mid-day.

Tracking is built entirely on **Re-Identification (Re-ID) embeddings** extracted by OSNet — a 512-dimensional appearance representation that encodes a person's clothing, body shape, and posture. There is no optical flow, Kalman filter, or motion prediction involved. Identity matching is done by comparing embedding similarity (cosine distance).

This module produces the `track_id` that every downstream component — face capture, FR processing, report generation — uses as its primary key for a person.

---

## 2. Pipeline Position

```
Video Clips (MP4)
        │
        ▼
 [FrameReader Thread]        ← Background IO: decode frames while GPU works
        │ frames
        ▼
 Person Detection (YOLO v11l)
        │ bounding boxes
        ▼
 ┌─────────────────────────────┐
 │   FACE TRACKING MODULE      │  ← This document
 │                             │
 │   OSNetExtractor            │  ← GPU: extract Re-ID embeddings
 │        │                   │
 │   PersonTracker             │  ← CPU: 3-phase greedy matching
 │   ├── ActiveTrack[]         │
 │   └── PendingTrack[]        │
 └─────────────────────────────┘
        │ (track_id, bbox) pairs
        ▼
 Face Detection (YOLO v11s-face)
        │
        ▼
 Face Recognition (InsightFace + FAISS)
        │
        ▼
 Excel Report
```

The tracking module sits between person detection and face capture. It receives raw YOLO bounding boxes and outputs `(track_id, bbox)` pairs. Everything after it operates on the track ID as an identity handle.

---

## 3. Core Concepts

### Re-ID Embeddings (not pixel tracking)

Traditional trackers (SORT, DeepSORT) combine motion prediction (Kalman filter) with appearance. This pipeline uses **appearance only** via OSNet embeddings. Cosine distance between a detection embedding and a track's reference embedding determines whether they belong to the same person.

**Why appearance-only?** The clips are ~1 minute each with 30-second gaps between them. Motion prediction cannot bridge those gaps. Appearance embeddings can.

### Cosine Similarity as the Matching Metric

All OSNet embeddings are L2-normalized before use. On unit-norm vectors, dot product equals cosine similarity:

```
cosine_similarity(a, b) = a · b
cosine_distance(a, b)   = 1 - a · b
```

A distance of `0` = identical appearance. A distance of `1` = completely dissimilar. The match threshold is `REID_DISTANCE = 0.65`, meaning detections and tracks with distance ≥ 0.65 are never matched.

### Two-Phase Track State: Pending → Active

New detections are not immediately given a track ID. They first spend `WARMUP_FRAMES` (default: 10) frames in a **pending** state. Only after surviving warmup — i.e., being consistently detected and matched across multiple frames — does a `PendingTrack` graduate to an `ActiveTrack` with a permanent integer ID. This suppresses false positives from brief occlusions or detector noise.

### Embedding Bank: Rolling Appearance Memory

Each `ActiveTrack` maintains a fixed-size sliding window of its last `BANK_SIZE` (default: 10) embeddings. The **reference embedding** used for matching is a recency-weighted average of this bank — recent observations count up to 2× more than older ones. This allows the track to adapt to gradual appearance changes (lighting shifts, slight rotation) without discarding historical context.

### Cross-Clip Continuity via Serialization

After each clip, the full tracker state (embedding banks, ID counters) is serialized to disk: banks as `.npy` files in `data/<date>/tracks/`, metadata in `state.json`. On the next clip, state is restored, and the same `PersonTracker` instance effectively continues where it left off. Persons who appear across multiple clips keep the same `track_id`.

---

## 4. Tracking Architecture

```
ClipProcessor
│
├── FrameReader (thread)
│       └── queues (frame_idx, frame) tuples
│
├── person_model (YOLO v11l)
│       └── detects bounding boxes each frame
│
├── OSNetExtractor
│       └── batch GPU inference → 512-d L2-normalized embeddings
│
└── PersonTracker
        ├── active: dict[track_id → ActiveTrack]
        │       └── embedding bank (deque, maxlen=BANK_SIZE)
        │       └── last_box, last_timestamp, lost_seconds
        │
        └── pending: dict[pending_key → PendingTrack]
                └── plain list bank (no size limit)
                └── frames_seen counter
```

**Class ownership:**
- `ClipProcessor` owns all models and the `PersonTracker` instance.
- `PersonTracker` owns all `ActiveTrack` and `PendingTrack` objects.
- `OSNetExtractor` is stateless after init; it is injected into `PersonTracker` at construction.

One `ClipProcessor` instance is created per processing day and is reused across all clips (not re-initialized per clip). The tracker state persists inside it across clips.

---

## 5. Workflow

### Per-Clip Workflow

```
Start of clip
│
├── Parse clip timestamp from filename (YYYYMMDD_HHMMSS.mp4)
├── Read native FPS, compute skip = native_fps / PROCESSING_FPS
├── Start FrameReader thread
│
│   ── for each sampled frame ──
│   │
│   ├── YOLO person detect → raw bounding boxes
│   │
│   ├── PersonTracker.match_or_create(frame, boxes, timestamp)
│   │   ├── OSNet batch extract embeddings (GPU)
│   │   ├── Phase 1: match to active tracks
│   │   ├── Phase 2: match remaining to pending tracks
│   │   └── Phase 3: create new pending tracks
│   │
│   ├── For each (track_id, box) returned:
│   │   ├── Create person record if first appearance (_start_person)
│   │   ├── Update last_seen timestamp
│   │   ├── Save body snapshot (first frame + update at ≥2s)
│   │   └── Queue for face detection (_batch_capture_faces)
│   │
│   └── Check and finalize lost tracks (get_lost → _finalize_person)
│
├── End of frames (sentinel received)
│
└── Return list of finalized person records
```

### Per-Day Workflow

```
process_day()
│
├── Load state.json (or create empty)
├── Fetch new video clips (SSH/SCP from Jetson, or local)
├── Initialize ClipProcessor (load models)
├── Restore tracker state from state.json (if resuming)
│
├── For each new clip in chronological order:
│   ├── Check gap from previous clip
│   │   └── If gap > MAX_LOST_SECONDS: finalize_all()
│   ├── process_clip() → finalized persons appended to state["persons"]
│   ├── get_save_state() → serialize tracker to state.json
│   └── save_state()
│
├── finalize_all() on last clip
├── run_fr_processing()
└── generate_report()
```

---

## 6. Detailed Tracking Logic

### Phase 1: Matching Detections to Active Tracks

This runs on every frame where YOLO detects at least one person and active tracks exist.

**Step 1 — Batch embedding extraction**

```python
embeddings = self.osnet.extract_batch(frame, boxes)
```

All detection boxes for the current frame are sent to OSNet in a single GPU call. The result is a list aligned with `boxes`: `embeddings[i]` is the 512-d embedding for `boxes[i]`, or `None` if the crop was too small (< 32×64 px).

**Step 2 — Build cost matrix**

```python
det_mat = np.stack([emb for _, emb in valid_dets])   # shape: (D, 512)
ref_mat = np.stack(active_refs)                        # shape: (A, 512)
dist_matrix = 1.0 - det_mat @ ref_mat.T               # shape: (D, A)
```

`active_refs` is `[track.ref for track in active_tracks]`. `track.ref` is the recency-weighted, L2-normalized average of the embedding bank. The matrix entry `[d, a]` is the cosine distance between detection `d` and active track `a`.

**Step 3 — Greedy assignment with spatial gate**

```python
for _ in range(min(len(valid_dets), len(active_ids))):
    # Zero out already-matched rows and columns
    for r in range(len(valid_dets)):
        if used_d[r]: tmp[r, :] = 1.0
    for c in range(len(active_ids)):
        if used_a[c]: tmp[:, c] = 1.0

    # Apply spatial gate: if box moved too far, set cost to 1.0
    for r, (di, _) in enumerate(valid_dets):
        if used_d[r]: continue
        for c, uid in enumerate(active_ids):
            if used_a[c]: continue
            if self._box_distance(boxes[di], self.active[uid].last_box) > box_jump:
                tmp[r, c] = 1.0

    # Pick globally best remaining match
    best_idx = int(np.argmin(tmp))
    r, c = divmod(best_idx, len(active_ids))
    if tmp[r, c] >= REID_DISTANCE:   # 0.65
        break  # no more valid matches exist
    ...
```

This is a **greedy Hungarian-style** algorithm. Each iteration picks the globally cheapest unassigned pair, marks both as used, and records the match. The loop terminates when the best remaining match has distance ≥ `REID_DISTANCE` (0.65).

**The spatial gate** (`box_jump`) rejects matches that would require a person to move more than `MAX_BOX_JUMP` (0.30) of the frame diagonal in one frame. This prevents embedding-similar persons on opposite sides of the frame from being merged. The gate is widened for frames with large time gaps:

```python
box_jump = MAX_BOX_JUMP * max(1, gap_seconds / 0.5)  # capped at 0.8
```

**Step 4 — Selective bank update**

```python
if tmp[r, c] < REID_DISTANCE * UPDATE_GATE:   # < 0.65 × 0.80 = 0.52
    self.active[uid].update_embedding(det_emb)
```

Only close matches (distance < 0.32) update the embedding bank. Borderline matches (0.32–0.40) are still valid assignments but do not pollute the bank with potentially noisy embeddings.

**Step 5 — Bookkeeping**

```python
self.active[uid].last_box = list(boxes[di])
self.active[uid].last_timestamp = timestamp
self.active[uid].lost_seconds = 0
results.append((uid, boxes[di]))
```

Every matched active track gets its position and timestamp updated and its lost counter reset to 0.

---

### Phase 2: Matching Remaining Detections to Pending Tracks

Detections not consumed by Phase 1 are compared against pending tracks using the same greedy cosine-distance matching (no spatial gate in Phase 2).

```python
pd = 1.0 - det_mat @ ref_mat.T   # pending distance matrix
```

`PendingTrack.ref` is a simple unweighted mean of all observed embeddings. When a pending track is matched:

```python
pend.add(unmatched[r][1], boxes[di])

if pend.frames_seen >= WARMUP_FRAMES:   # default: 10
    uid = self.next_id
    self.next_id += 1
    track = ActiveTrack(pend.ref, boxes[di], timestamp)
    for e in pend.bank:
        track.bank.append(e.copy())   # seed ActiveTrack bank with all pending history
    self.active[uid] = track
    del self.pending[pk]
    results.append((uid, boxes[di]))
```

Promotion transfers all accumulated embeddings from the pending bank into the new `ActiveTrack.bank`. The `next_id` counter is monotonically increasing and is persisted in `state.json`, guaranteeing unique IDs across the entire day.

---

### Phase 3: Creating New Pending Tracks

Any detection still unmatched after Phases 1 and 2 spawns a new `PendingTrack`:

```python
for di in range(len(boxes)):
    if di not in used_det and embeddings[di] is not None:
        pk = self.pending_next
        self.pending_next += 1
        self.pending[pk] = PendingTrack(embeddings[di], boxes[di])
```

`pending_next` is a separate counter from `next_id` — pending keys are not person IDs. Pending tracks that never reach `WARMUP_FRAMES` are silently discarded when the clip ends (they are never serialized).

---

## 7. Key Classes

### Class: `OSNetExtractor`

- **Purpose:** GPU-accelerated Re-ID embedding extractor wrapping `osnet_x1_0_msmt17.pt` via BoxMOT's `ReidAutoBackend`.
- **Responsibilities:**
  - Load OSNet weights onto GPU (CUDA) or CPU.
  - Validate crop sizes before inference (min 32×64 px).
  - Extract and L2-normalize 512-dimensional embeddings.
  - Support both single-crop (`extract`) and batch (`extract_batch`) extraction.
  - Provide a non-crashing fallback (random unit vectors) when the model fails to load.
- **Key Methods:** `extract(frame, bbox)`, `extract_batch(frame, boxes)`
- **Role in Tracking:** Produces the embedding vectors that are the sole basis for all matching decisions. Called once per frame in batch mode for maximum GPU throughput.

---

### Class: `ActiveTrack`

- **Purpose:** Represents a fully confirmed, active person identity with a rolling embedding bank.
- **Responsibilities:**
  - Maintain a `deque(maxlen=BANK_SIZE)` of recent L2-normalized embeddings.
  - Compute a recency-weighted reference embedding (`ref` property) for matching.
  - Record the last known bounding box and timestamp.
  - Accumulate `lost_seconds` to trigger eventual finalization.
  - Serialize/deserialize the embedding bank to `.npy` files for cross-clip persistence.
- **Key Methods:** `ref` (property), `update_embedding(emb)`, `serialize(tracks_dir, track_id)`, `deserialize(data, tracks_dir)` (staticmethod)
- **Role in Tracking:** The persistent identity object. One `ActiveTrack` per confirmed person; its `ref` is compared against every incoming detection embedding.

---

### Class: `PendingTrack`

- **Purpose:** Lightweight candidate track for newly observed, unconfirmed persons.
- **Responsibilities:**
  - Accumulate embeddings and bounding boxes for a new detection.
  - Count how many frames this candidate has been consistently matched (`frames_seen`).
  - Provide a simple mean reference embedding for Phase 2 matching.
  - Graduate to `ActiveTrack` when `frames_seen >= WARMUP_FRAMES`.
- **Key Methods:** `ref` (property), `add(emb, box)`
- **Role in Tracking:** Acts as a noise filter. A person must be seen in at least `WARMUP_FRAMES` frames before receiving a track ID. Never serialized — always discarded at clip boundaries.

---

### Class: `PersonTracker`

- **Purpose:** The central tracking engine managing both pools (active and pending) and running the 3-phase matching algorithm.
- **Responsibilities:**
  - Batch-extract embeddings via `OSNetExtractor`.
  - Run 3-phase greedy matching on every frame.
  - Promote pending tracks to active when warmup threshold is met.
  - Gate embedding bank updates to prevent noise accumulation.
  - Apply bounding-box spatial constraint to prevent cross-frame teleportation matches.
  - Detect and report lost tracks for finalization.
  - Serialize and restore full state for cross-clip continuity.
- **Key Methods:** `match_or_create(frame, boxes, timestamp, gap_seconds)`, `get_lost(current_time)`, `remove(uid)`, `serialize_state(tracks_dir)`, `restore_state(state_data, tracks_dir)`
- **Role in Tracking:** The orchestrator. Every tracking decision — match, promote, create, finalize — passes through this class.

---

### Class: `ClipProcessor`

- **Purpose:** Top-level pipeline orchestrator per clip and per day. Owns all models and drives the frame loop.
- **Responsibilities:**
  - Initialize YOLO person/face models and OSNet.
  - Restore tracker state between clips.
  - Run the per-frame detect → track → capture pipeline.
  - Manage in-progress person records in `tracked_persons`.
  - Finalize persons when lost or at clip boundaries.
  - Expose tracker state for serialization after each clip.
- **Key Methods:** `initialize()`, `restore_from_state(state, date_str)`, `process_clip(video_path, date_str, state)`, `_finalize_person(track_id, ts)`, `finalize_all(ts)`, `get_save_state(date_str)`
- **Role in Tracking:** The entry point from the pipeline's perspective. One `ClipProcessor` instance runs for the entire day; the `PersonTracker` inside it maintains continuity across all clips.

---

## 8. Key Functions

### `PersonTracker.match_or_create`

- **Purpose:** Main per-frame tracking step — resolves YOLO detections to track identities.
- **Input:**
  - `frame` (`np.ndarray`, BGR) — current video frame
  - `boxes` (`list` of `[x1, y1, x2, y2]`) — YOLO person detections in pixel space
  - `timestamp` (`datetime`) — wall-clock time of this frame
  - `gap_seconds` (`float`) — time since the previous frame; widens the spatial gate for large gaps
- **Output:** `list` of `(track_id: int, box: list)` for every active-track assignment. Pending tracks are updated but not returned.
- **Logic:**
  1. Batch-extract embeddings for all detection boxes (one GPU call).
  2. Phase 1: build cosine-distance matrix between valid detections and active refs; greedy assignment with spatial gate; update banks and timestamps.
  3. Phase 2: greedy matching of remaining detections against pending refs; promote tracks that reach `WARMUP_FRAMES`.
  4. Phase 3: create new `PendingTrack` for every still-unmatched detection.
- **Role in Tracking:** The only function that runs on every processed frame. Everything else is triggered by its results.

---

### `PersonTracker._match_tracks` (inline logic in `match_or_create`)

The matching loop is implemented inline within `match_or_create`. The greedy assignment strategy:

1. Copy the distance matrix.
2. Set rows and columns of already-matched detections/tracks to 1.0.
3. Apply spatial gate: set matrix cells to 1.0 where box distance exceeds `box_jump`.
4. `argmin` over the remaining matrix to find the globally cheapest pair.
5. If the minimum distance ≥ `REID_DISTANCE`, stop — no more valid matches.
6. Record the match, mark row and column as used.
7. Repeat until all detections or tracks are exhausted.

---

### `ActiveTrack.ref` (property)

- **Purpose:** Compute the recency-weighted reference embedding for this track.
- **Input:** None (reads `self.bank`)
- **Output:** `np.ndarray (512,)` float32, L2-normalized
- **Logic:**
  ```python
  n = len(self.bank)
  w = np.linspace(0.5, 1.0, n)   # weight ramp: oldest=0.5, newest=1.0
  return _l2(np.average(list(self.bank), axis=0, weights=w))
  ```
- **Role in Tracking:** This is the vector used in every cosine-distance comparison. It is recomputed on each access — not cached — so it always reflects the current bank state.

---

### `ActiveTrack.get_reference_embedding` (via `ref` property)

The `ref` property **is** the reference embedding function. It is accessed as `track.ref` in the matching loop.

---

### `OSNetExtractor.extract`

- **Purpose:** Single-crop embedding extraction (used for fallback or isolated cases).
- **Input:** `frame` (BGR `np.ndarray`), `bbox` (4-tuple pixel coords)
- **Output:** `np.ndarray (512,)` L2-normalized float32, or `None` if crop is too small or inference fails
- **Logic:** Clamps bbox to frame boundaries → validates crop size (≥ 32×64 px) → runs `model.get_features([[x1,y1,x2,y2]], frame)` → L2-normalize.
- **Role in Tracking:** Available for single-crop use. In practice, `extract_batch` is used in the hot path.

---

### `OSNetExtractor.extract_batch`

- **Purpose:** Batch GPU embedding extraction — single forward pass for all persons in a frame.
- **Input:** `frame` (BGR `np.ndarray`), `boxes` (list of 4-tuples)
- **Output:** `list` of length `len(boxes)`; each element is either `np.ndarray (512,)` or `None`
- **Logic:**
  1. Filter boxes by minimum crop size; record valid indices.
  2. Stack valid boxes into `np.array` and call `model.get_features(valid_boxes, frame)` — one GPU call.
  3. L2-normalize each result and map back to original indices.
  4. On batch failure, fall back to sequential single-crop extraction.
- **Role in Tracking:** Primary performance optimization — reduces N GPU calls per frame to 1.

---

### `ClipProcessor.process_clip`

- **Purpose:** Full per-clip pipeline orchestrator.
- **Input:** `video_path` (str), `date_str` (str), `state` (dict)
- **Output:** `list[dict]` — finalized person records for persons who left the scene during this clip
- **Logic:**
  1. Parse clip start timestamp from filename.
  2. Compute `skip = max(1, int(native_fps / PROCESSING_FPS))`.
  3. Start `FrameReader` thread.
  4. For each frame: YOLO detect → `match_or_create` → body snapshot → face capture → lost-track finalization.
  5. Stop reader; log per-stage profiling breakdown.
- **Role in Tracking:** The outermost loop. Coordinates all tracking sub-components and produces the finalized person records that accumulate in `state["persons"]`.

---

## 9. Track Lifecycle

```
Frame N: Person first detected by YOLO
        │
        ▼
[PendingTrack created]
  frames_seen = 1
  bank = [emb₁]
        │
        │  ← matched in frames N+1, N+2, N+3, N+4
        ▼
[PendingTrack.frames_seen = 10 = WARMUP_FRAMES]
        │
        ▼
[Promotion to ActiveTrack]
  next_id assigned (monotonically increasing integer)
  bank seeded with all 5 pending embeddings
  last_box, last_timestamp set
  result returned to ClipProcessor: (track_id, box)
        │
        │  ← person appears in subsequent frames
        │  bank updated (if match dist < REID_DISTANCE × UPDATE_GATE)
        │  last_box, last_timestamp reset each match
        │  lost_seconds = 0 while matched
        │
        │  ← person temporarily occluded or walks out of YOLO's view
        │  no match in current frame → lost_seconds not explicitly incremented
        │  (finalization check uses timestamp delta via get_lost())
        │
        ▼
[get_lost() detects: (current_time - last_timestamp).total_seconds() > MAX_LOST_SECONDS]
        │
        ▼
[_finalize_person(track_id, ts)]
  person record moved from tracked_persons → finalized list
  PersonTracker.remove(track_id) called
  person dict returned with first_seen, last_seen, face_images, body_snapshot
        │
        ▼
[Appended to state["persons"]]
  Available for FR processing and report generation
```

### Cross-Clip Lifecycle

```
End of clip N:
  ClipProcessor.get_save_state()
    → PersonTracker.serialize_state(tracks_dir)
      → ActiveTrack.serialize() for each active track
        → bank saved to track_XXXX_bank.npy
        → metadata (last_box, last_timestamp, lost_seconds) → state.json

Start of clip N+1:
  ClipProcessor.restore_from_state(state, date_str)
    → PersonTracker.restore_state(state_data, tracks_dir)
      → ActiveTrack.deserialize() for each serialized track
        → bank loaded from .npy file
  PersonTracker.next_id restored (guarantees no ID reuse)

  PendingTracks are NOT restored — always start fresh each clip.
  Persons who were pending at end of clip N are discarded.
  Persons who were active at end of clip N continue seamlessly.
```

---

## 10. Matching Strategy

### Why Greedy Instead of Optimal (Hungarian)?

The greedy approach is O(min(D,A)²) per frame — acceptable for typical store densities (< 20 persons/frame). True Hungarian algorithm would be O(n³) and offers marginally better assignment quality, but for this use case (where persons rarely swap apparent identities), greedy with the spatial gate is sufficiently accurate and significantly simpler to implement without an external dependency.

The greedy algorithm is sequentially optimal: each iteration picks the globally best remaining pair. This avoids many of the failure modes of naive row-by-row or column-by-column greedy assignment.

### REID_DISTANCE Threshold (0.65)

A cosine distance of 0.65 means the angle between the two embedding vectors is arccos(0.35) ≈ 69°. This wider threshold accommodates more appearance variation (lighting, partial occlusion, different clip angles) while still excluding clearly different persons. Cross-person matches typically exceed 0.7–0.8 in this embedding space.

### UPDATE_GATE (0.80)

Only matches with distance < `REID_DISTANCE × UPDATE_GATE` = 0.52 update the embedding bank. This selective update prevents borderline matches (0.52–0.65) from degrading the reference embedding with uncertain data. If a match is borderline, it is likely the person is at a difficult angle or partially occluded — including that embedding could shift the reference toward a bad direction.

### Spatial Gate (`box_jump`)

The normalized box center distance constraint prevents two embedding-similar persons (e.g., staff in similar uniforms) on opposite sides of the frame from being incorrectly merged. The gate is dynamically scaled:

```
box_jump = MAX_BOX_JUMP × max(1, gap_seconds / 0.5)   # capped at 0.8
```

At `PROCESSING_FPS = 10` (25 FPS native), gap_seconds ≈ 0.1s per frame, so `box_jump ≈ MAX_BOX_JUMP = 0.30`. Between clips (gap can be 30+ seconds), the gate opens to 0.8, allowing the same person to reappear anywhere in the frame.

### Phase 1 vs. Phase 2 Matching Differences

| Property | Phase 1 (Active) | Phase 2 (Pending) |
|---|---|---|
| Target pool | `active` tracks | `pending` tracks |
| Reference embedding | Recency-weighted bank mean | Simple bank mean |
| Spatial gate | Applied | Not applied |
| On match | Update bank (if gated), reset lost | Increment `frames_seen`, possibly promote |
| Returns track_id | Yes (if already active) | Only on promotion |

---

## 11. Embedding Management

### OSNet Embedding: 512-d, L2-Normalized Float32

All embeddings are L2-normalized immediately after extraction by the `_l2()` helper:

```python
def _l2(v):
    n = np.linalg.norm(v)
    return v / n if n > 0 else v
```

This ensures cosine similarity = dot product, and that all distance computations are numerically stable.

### Embedding Bank: Sliding Window in `ActiveTrack`

```python
self.bank = deque(maxlen=BANK_SIZE)   # default BANK_SIZE = 10
```

The bank is a fixed-capacity circular buffer. When full, the oldest embedding is automatically dropped when a new one is appended. At `PROCESSING_FPS = 10` (from 25 FPS native), a full bank of 10 embeddings represents the last ~1 second of observations, subject to the `UPDATE_GATE` filter.

### Recency Weighting in `ActiveTrack.ref`

```python
n = len(self.bank)
w = np.linspace(0.5, 1.0, n)   # [0.5, ..., 1.0] for n entries
ref = _l2(np.average(list(self.bank), axis=0, weights=w))
```

With `n=10` embeddings, weights are: `[0.5, 0.556, 0.611, ..., 1.0]`. The most recent embedding carries 2× the weight of the oldest. This makes the reference adaptive to recent appearance changes (person turned around, moved under different lighting) without completely discarding historical context.

**Effect on tracking robustness:** If a person's appearance shifts gradually (clothing wrinkle, lighting change), the reference shifts with them, maintaining match quality over long dwell times.

### PendingTrack.ref: Simple Mean

```python
return _l2(np.mean(self.bank, axis=0))
```

No recency weighting for pending tracks. With only 1–4 observations (fewer than `WARMUP_FRAMES`), each frame is equally informative and there is not enough history to meaningfully differentiate old from new.

### Bank Seeding on Promotion

When a `PendingTrack` is promoted to `ActiveTrack`, all accumulated pending embeddings are copied into the new bank:

```python
track = ActiveTrack(pend.ref, boxes[di], timestamp)
for e in pend.bank:
    track.bank.append(e.copy())
```

This ensures the new `ActiveTrack` starts with a richer reference than a single-embedding bank, improving its matching accuracy immediately after promotion.

### Serialization Format

After each clip, each `ActiveTrack`'s bank is saved as:

```
data/<date>/tracks/track_XXXX_bank.npy
```

Format: `np.ndarray` of shape `(n, 512)` where `n` is the current bank size (≤ `BANK_SIZE`). The file is overwritten on each serialization.

Metadata stored in `state.json`:

```json
{
  "1": {
    "bank_file": "track_0001_bank.npy",
    "last_box": [120.0, 45.0, 280.0, 430.0],
    "last_timestamp": "2026-04-09T10:15:33.200000",
    "lost_seconds": 0.0
  }
}
```

---

## 12. Input / Output

### Inputs to the Tracking Module

| Input | Source | Format |
|---|---|---|
| Video frame | `FrameReader` queue | `np.ndarray` (H, W, 3) BGR uint8 |
| YOLO detections | `person_model.track()` | `np.ndarray` (N, 4) float32, pixel coords `[x1,y1,x2,y2]` |
| Frame timestamp | Derived from clip filename + `frame_idx / native_fps` | `datetime` |
| Gap seconds | Computed from consecutive frame timestamps | `float` |
| Saved state | `state.json` + `tracks/*.npy` (on clip restore) | `dict` + numpy files |

### Outputs from the Tracking Module

| Output | Consumer | Format |
|---|---|---|
| `(track_id, box)` pairs per frame | `ClipProcessor` per-frame loop | `list` of `(int, list)` |
| Lost track IDs | `ClipProcessor._finalize_person()` | `list[int]` |
| Finalized person records | `state["persons"]`, FR processing | `list[dict]` |
| Serialized tracker state | `state.json`, `tracks/*.npy` | `dict` + numpy files |

### Finalized Person Record Schema

```python
{
    'track_id':           int,            # e.g. 7
    'first_seen':         str,            # ISO datetime, e.g. "2026-04-09T09:14:22.1"
    'last_seen':          str,            # ISO datetime
    'crossing_timestamp': str,            # same as first_seen (legacy field)
    'images_folder':      str,            # "data/2026-04-09/faces/person_0007/"
    'face_images':        list[(float, str)],  # [(quality, filepath), ...]
    'best_face_image':    str | None,     # highest quality face path
    'body_snapshot':      str | None,     # "data/.../faces/person_0007/body_snapshot.jpg"
    'face_images_count':  int,
}
```

---

## 13. Edge Cases & Failure Handling

### OSNet Model Fails to Load

`OSNetExtractor.__init__` catches all exceptions and sets `self._model = None`. Subsequent calls to `extract` and `extract_batch` return random L2-normalized 512-d vectors. The tracking system continues to function structurally (IDs are assigned, state is serialized), but matching quality degrades to random — effectively creating a new track ID for every detection. This is a deliberate non-crashing fallback, appropriate for smoke-testing without GPU.

### Person Crop Too Small (< 32×64 px)

`extract_batch` returns `None` for the corresponding index. In Phase 1 matching, only `valid_dets` where `emb is not None` are included. In Phase 3, `if embeddings[di] is not None` guards new pending creation. Small crops produce no embedding and are silently skipped — no ID is assigned and no pending track is created.

### Occlusion Within a Clip

A person temporarily behind an obstacle will not be detected by YOLO for some frames. Their `ActiveTrack.last_timestamp` is not updated during the occluded frames. `get_lost()` checks elapsed time against `MAX_LOST_SECONDS` (default 25 seconds). If the occlusion is shorter, the track survives and re-matches when the person re-emerges.

**Re-appearance after occlusion:** On re-detection, the embedding is compared against all active track references. If the person's embedding is still close enough (< `REID_DISTANCE = 0.65`) to their stored bank average, the match succeeds and the same track ID is maintained. The spatial gate is relaxed proportionally to the gap duration.

### Long Gaps Between Clips

When consecutive clips are separated by more than `MAX_LOST_SECONDS`, `process_day()` explicitly calls `finalize_all()` before processing the new clip. This prevents persons from clip N being incorrectly matched to persons in clip N+1. After finalization, active tracks are empty and all persons in the new clip start fresh.

### Pending Track Never Reaches WARMUP_FRAMES

If a person appears in fewer than `WARMUP_FRAMES` (10) frames (e.g., very brief entry and exit), the `PendingTrack` is never promoted to `ActiveTrack`. At clip end, all pending tracks are discarded. These persons receive no track ID and are not captured in `state["persons"]`. They are effectively invisible to the pipeline — a deliberate trade-off to avoid tracking transient false positives.

### Cross-Clip State Restoration Failure

`PersonTracker.restore_state()` handles individual track deserialization failures with a `try/except`:

```python
try:
    self.active[int(uid_str)] = ActiveTrack.deserialize(data, tracks_dir)
except Exception as e:
    logger.warning(f"Failed to restore track {uid_str}: {e}")
```

If a `.npy` bank file is missing or corrupted, that track is skipped. The remaining tracks restore correctly. The `next_id` counter is still restored accurately, so ID uniqueness is never compromised.

### Same Person Gets Multiple Track IDs

Can occur if the person leaves the frame for longer than `MAX_LOST_SECONDS`, is finalized, and then re-enters. They will be detected as a new person and receive a new pending → active promotion cycle. The FR clustering stage (downstream) can merge these into one cluster if the face embeddings match — tracking does not handle this case.

### Identical Clothing / High Embedding Similarity Between Different People

The spatial gate limits false merges. Two embedding-similar persons on opposite sides of a 1280×720 frame will have a normalized box-center distance of ~1.0, well above `MAX_BOX_JUMP = 0.30`, so their costs are set to 1.0 in the distance matrix and no match occurs. Both receive separate track IDs.

---

## 14. Performance Considerations

### Batch GPU Extraction

The most critical optimization: `extract_batch` issues one GPU call per frame regardless of how many persons are detected. At 10 FPS with 5 persons per frame, this replaces 50 individual GPU calls per second with 10 batch calls — a significant reduction in GPU kernel launch overhead.

### Frame Skipping

```python
skip = max(1, int(native_fps / PROCESSING_FPS))
```

At native FPS = 30, `skip = 3`. The `FrameReader` thread decodes all frames but only enqueues every 3rd one. This reduces the tracking workload to `PROCESSING_FPS = 10` effective frames per second without losing temporal resolution for appearance matching.

### Double-Buffered IO: FrameReader Thread

```
FrameReader thread:        [decode frame₁] [decode frame₂] [decode frame₃] ...
Main thread (GPU):    ... [YOLO+OSNet on frame₀] [YOLO+OSNet on frame₁] ...
```

The queue (size 64) buffers up to ~6 seconds of decoded frames. The profiler tracks how often the main thread waits on an empty queue (`_read_waits`). High wait count = disk bottleneck; low wait count = GPU bottleneck.

### Matching Complexity

The greedy matching loop is O(min(D, A) × D × A) per frame where D = number of detections, A = number of active tracks. For typical store densities (D ≤ 10, A ≤ 20), this is entirely CPU-bound and takes negligible time compared to GPU inference. The profiler logs this as "Track/crop" stage.

### Memory: Bounded by BANK_SIZE

Each `ActiveTrack` stores at most `BANK_SIZE = 10` embeddings of 512 float32 values = 10 × 512 × 4 bytes ≈ 20 KB per track. With 100 concurrent active tracks, this is ~2 MB — entirely negligible.

### Profiling Output (per clip)

After each clip, `ClipProcessor.process_clip()` logs a breakdown:

```
PERF [300 frames, 30.2s, 9.9 fps]
  Frame read/wait : 1.23s  4.1%  (queue empty 3x)
  YOLO person     : 14.50s 48.0%  [GPU]
  OSNet re-ID     : 8.12s  26.9%  [GPU+CPU]
  Track/crop      : 1.05s  3.5%   [CPU]
  Face detect     : 4.80s  15.9%  [GPU]
  Misc/finalize   : 0.50s  1.7%   [CPU]
```

YOLO person detection is typically the dominant GPU stage (~48%). OSNet is second (~27%). Tracking itself (matching logic) is CPU-only and takes < 4% of total time.

---

## 15. Integration with Other Modules

### From: Person Detection (YOLO v11l)

`ClipProcessor.process_clip()` calls:

```python
results = self.person_model.track(
    frame, classes=[0], conf=CONF_THRESHOLD,
    persist=True, verbose=False, half=True
)
boxes = results[0].boxes.xyxy.cpu().numpy()
```

`CONF_THRESHOLD = 0.45`. Class 0 = person. `persist=True` maintains YOLO's internal ID state (used for its internal SORT tracker), though the pipeline ignores YOLO track IDs and uses its own embedding-based tracking. Only the bounding boxes (`.xyxy`) are passed to `PersonTracker`.

### To: Face Capture (`_batch_capture_faces`)

`process_clip()` collects `(uid, crop, offset, box)` tuples for each active track and passes them to `_batch_capture_faces`. The crop is the person bounding box from the current frame. Face detection runs on this crop, not the full frame. The `uid` (track ID) links the captured face image to the correct `tracked_persons` record.

### To: State Serialization (`state.json`)

After each clip, `get_save_state()` serializes the entire tracker state into the `state.json` structure under:

```json
{
  "next_track_id": 42,
  "active_tracks": { "1": {...}, "7": {...} },
  "pending_next": 88,
  "persons_in_progress": [...]
}
```

This is the bridge for cross-clip and cross-run continuity.

### To: FR Processing (`run_fr_processing`)

FR processing operates on `state["persons"]` — the list of finalized person records. Each record's `track_id` determines the folder `data/<date>/faces/person_XXXX/` where face images were saved by the tracking loop. FR does not interact with `PersonTracker` directly.

### To: Report Generation (`generate_report`)

The report uses `state["persons"]` for timing data (first_seen, last_seen per track_id) and `state["fr_results"]` for classification. Track IDs appear in the Excel report as row identifiers in the Track IDs sheet.

### Configuration (`config.json` keys consumed by tracking)

| Key | Current (`config.json`) | Effect |
|---|---|---|
| `tracking.reid_distance` | `0.65` | Max cosine distance for valid match |
| `tracking.update_gate` | `0.80` | Fraction of threshold below which bank updates |
| `tracking.warmup_frames` | `10` | Frames before pending → active promotion |
| `tracking.bank_size` | `10` | Embedding bank sliding window size |
| `tracking.max_lost_seconds` | `25` | Seconds of inactivity before track finalization |
| `tracking.conf_threshold` | `0.45` | YOLO person detection confidence floor |
| `tracking.max_box_jump` | `0.30` | Max normalized box center distance for match |
| `tracking.min_crop_w` | `32` | Minimum person crop width for OSNet |
| `tracking.min_crop_h` | `64` | Minimum person crop height for OSNet |
| `video.processing_fps` | `10` | Effective frames per second processed |
| `video.expected_fps` | `25` | Native camera FPS (fallback if OpenCV reports 0) |

> **Note on defaults:** Values above are hardcoded defaults in `pipeline.py` when the key is absent from `config.json`. The deployed `config.json` may override any of these.

---

*Generated: 2026-04-28 | Source: `pipeline.py` v1.7.0 | Read-only: do not modify `pipeline.py`*
