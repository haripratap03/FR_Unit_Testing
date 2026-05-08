# Face Recognition Module Documentation

## Overview

The Face Recognition (FR) module identifies whether a detected person is a **known staff member** or an **unknown customer**. It operates on face images captured during video tracking, extracts 512-dimensional embeddings using InsightFace (antelopev2), and performs cosine similarity search against a pre-built FAISS staff index. Persons that do not match any staff member are routed to a greedy clustering step that groups repeat visitors.

FR runs as the final processing stage before report generation. It is implemented in three separate scripts depending on context:

| Script                              | When to use                                                       |
| ----------------------------------- | ----------------------------------------------------------------- |
| `pipeline.py` (`run_fr_processing`) | End of main pipeline run, after all clips are tracked             |
| `rerun_fr.py` (`rerun_fr`)          | Re-running FR on an already-processed day; higher recall settings |
| `fr_report.py` (`run_fr`)           | Standalone scan of `faces/` without requiring `state.json`        |

---

## Pipeline Position

```
Face Detection (YOLOv11s-face)
        ↓
Face Quality Scoring (score_face)
        ↓
Face Images saved → faces/person_XXXX/
        ↓
➡ Face Recognition
    ├─ Pre-Recognition Filtering (quality / skin-tone / keypoints)
    ├─ Embedding Extraction (InsightFace antelopev2)
    ├─ Staff Matching (FAISS IndexFlatIP cosine similarity)
    └─ Customer Clustering (greedy cosine grouping)
        ↓
Report Generation (Excel)
```

FR does not re-process tracking output in real time. It reads the finalized `faces/person_XXXX/` folders produced by the tracker and runs as a batch step.

---

## Core Concepts

### Embeddings

An **embedding** is a 512-dimensional floating-point vector that encodes a face's identity. Two photos of the same person will produce embeddings with high cosine similarity; two different people will produce embeddings with low similarity. InsightFace (antelopev2) is the model used to produce these vectors.

### L2 Normalization

Before any comparison, every embedding is **L2-normalized** — divided by its own Euclidean magnitude so that its length becomes exactly 1.0. After normalization, the **dot product** of two vectors equals their **cosine similarity**:

```
cosine_similarity(a, b) = dot(a, b)   [when ||a|| = ||b|| = 1]
```

This is exploited by the FAISS index (see below).

### Cosine Similarity vs. Euclidean Distance

| Property                               | Cosine similarity                | Euclidean distance                      |
| -------------------------------------- | -------------------------------- | --------------------------------------- |
| Range                                  | −1 to +1 (higher = more similar) | 0 to ∞ (lower = more similar)           |
| Sensitive to magnitude?                | No                               | Yes                                     |
| Appropriate for normalized embeddings? | Yes                              | No (all lengths = 1, so it's redundant) |

FR uses cosine similarity throughout. All embedding vectors are L2-normalized before storage or search.

---

## Models and Tools Used

| Model / Tool                 | Role                                                                                          |
| ---------------------------- | --------------------------------------------------------------------------------------------- |
| **InsightFace `antelopev2`** | Face detection (bounding box + 5 landmarks) and 512-dim embedding extraction                  |
| **FAISS `IndexFlatIP`**      | Exact nearest-neighbor search using inner product (= cosine similarity on normalized vectors) |
| **NumPy**                    | L2 normalization (`np.linalg.norm`), dot products (`np.dot`), matrix operations               |
| **OpenCV**                   | Image preprocessing: CLAHE contrast enhancement, upscaling, border padding                    |

### Why FAISS IndexFlatIP?

`IndexFlatIP` computes the **inner product** (dot product) between a query vector and every stored vector. Because all vectors are L2-normalized, this is exactly cosine similarity. `IndexFlatIP` performs exhaustive search — there is no approximate quantization — which is appropriate for the typical registry size of fewer than 50 staff vectors. A query with `k=1` returns the single most similar stored vector.

---

## Workflow

```
1. Load InsightFace (antelopev2)
2. Load FAISS index + staff_registry.json
3. For each unprocessed person:
   a. Walk face images in faces/person_XXXX/
   b. Filter 1: quality score floor (from filename)
   c. Filter 2: skin-tone HSV check
   d. Filter 3: InsightFace detection + keypoint containment
   e. Extract L2-normalized 512-dim embedding
   f. Keep top-N embeddings by quality score
   g. Search each embedding against FAISS index
   h. Take best cosine similarity score across all embeddings
   i. If score ≥ staff_threshold → STAFF
      Else → CUSTOMER (added to clustering pool)
      If no embedding extracted → NO FACE
4. Greedy customer clustering
5. Write results to state["fr_results"]
```

---

## Detailed Recognition Process

### Step 1 — Load Models

```python
face_app = FaceAnalysis(name="antelopev2", root=models_dir, providers=providers)
face_app.prepare(ctx_id=0, det_size=(640, 640), det_thresh=0.3)
```

- `det_thresh=0.3` in `pipeline.py`; reduced to `0.15` in `rerun_fr.py` and `fr_report.py` to catch partial or angled faces at the cost of more false detections.
- `det_size=(640, 640)` sets the internal detection resolution.
- `providers` tries CUDA first, falls back to CPU.

```python
faiss_index = faiss.read_index(faiss_path)        # IndexFlatIP, 512-dim
staff_registry = json.load(open(registry_path))   # list of {name, person_ids, ...}
```

The FAISS index and registry JSON are loaded from `vector_db/`. The registry at position `i` corresponds to the vector at index `i` in FAISS. They are always kept in sync.

---

## Face Filtering (Pre-Recognition)

Three filters are applied in `run_fr_processing` (`pipeline.py`) before any model inference. They are ordered cheapest-first to short-circuit early.

### Filter 1 — Quality Score Floor

```python
q = _parse_quality_from_filename(jpg)   # extracts NN from face_XXXXXX_qNN.jpg
if 0 <= q < FILTER_MIN_QUALITY:         # default: 25
    _move_to_rejected(folder, jpg, f"quality_{q}")
    continue
```

- Quality scores are encoded in the face filename at capture time by `score_face()`.
- Images where the score is known and below 25 are moved to `rejected/` immediately, avoiding the cost of loading them.
- Images with an unknown score (filename doesn't match pattern, `q = -1`) are **not** rejected by this filter.

### Filter 2 — Skin-Tone HSV Check

```python
skin_ok, skin_ratio = _check_skin_tone(img, FILTER_MIN_SKIN)  # min_ratio=0.20
if not skin_ok:
    _move_to_rejected(folder, jpg, f"skin_{skin_ratio:.2f}")
    continue
```

The `_check_skin_tone` function:

1. Crops the **central 60%** of the face image (20%–80% on each axis) to avoid background edges.
2. Converts to HSV color space.
3. Checks membership in two HSV skin-tone ranges:
   - Range 1: H 0–25, S 30–180, V 60–255 (light to medium skin)
   - Range 2: H 160–180, S 30–180, V 60–255 (red-channel wraparound)
4. If fewer than 20% of center pixels fall in either range, the image is rejected.

This filters out:

- Images where the face region is a wall, garment, or background object
- Very dark images with no visible skin tone

### Filter 3 — InsightFace Detection + Keypoint Containment

```python
faces = face_app.get(img)
if not faces:
    _move_to_rejected(folder, jpg, "no_face")
    continue
f = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))

kp_valid = _count_kps_in_bbox(f.kps, f.bbox)
if kp_valid < FILTER_MIN_KEYPOINTS:     # default: 3
    _move_to_rejected(folder, jpg, f"kps_{kp_valid}")
    continue
```

`_count_kps_in_bbox`:

- InsightFace outputs 5 facial landmarks (nose tip, left eye, right eye, left ear, right ear) with no per-point confidence scores.
- Validates landmark positions against the detected bounding box, allowing a 20% margin on each side.
- If fewer than 3 of 5 landmarks fall within the expanded bbox, the face crop is likely a false detection (back of head, profile extreme, object).

After passing all three filters, the embedding is extracted from the detected face.

---

## Embedding Extraction

### Standard Extraction (pipeline.py)

```python
emb = f.embedding           # 512-dim float32 from InsightFace ArcFace backbone
norm = np.linalg.norm(emb)
if norm > 0:
    emb = emb / norm        # L2 normalize
candidates.append((q, emb.astype(np.float32), jpg))
```

- Only the **largest detected face** in the image is used (selected by bounding box area).
- The embedding is L2-normalized before storage and search.
- All passing images for a person contribute an embedding to the `candidates` list.

### Top-N Selection

```python
candidates.sort(key=lambda x: -x[0])                    # sort by quality, descending
if len(candidates) > POST_FILTER_KEEP_TOP_N:             # default: 20
    # move excess to rejected/
    candidates = candidates[:POST_FILTER_KEEP_TOP_N]
```

The top 20 face images by quality score are kept. The rest are moved to `rejected/`. This prevents a person with 40 captured faces from inflating match time, while ensuring the best-quality embeddings are used for recognition.

### Multi-Strategy Extraction (rerun_fr.py, fr_report.py)

When the standard detection fails (e.g., very small face, poor lighting, face at crop border), `_extract_with_strategies()` tries four methods in sequence, stopping at the first success:

| Strategy   | Trigger                       | Method                                                           |
| ---------- | ----------------------------- | ---------------------------------------------------------------- |
| `direct`   | Always tried first            | `face_app.get(img)` on original image                            |
| `upscaled` | Image width or height < 200px | Bicubic upscale to at least 200px on shortest side               |
| `enhanced` | Direct failed                 | CLAHE (clipLimit=3.0, tileGridSize=8×8) on LAB lightness channel |
| `padded`   | Enhanced failed               | 40px reflective border on all sides                              |

The stat `strategies_used` records which strategies produced successful embeddings, useful for debugging.

---

## Staff Matching (FAISS)

```python
for emb in all_embs:
    D, I = faiss_index.search(np.expand_dims(emb, 0).astype(np.float32), 1)
    sim = float(D[0][0])    # cosine similarity (inner product of normalized vectors)
    idx = int(I[0][0])      # index into staff_registry
    if sim > best_sim:
        best_sim = sim
        best_name = staff_registry[idx]['name']
        best_emb = emb
```

- All embeddings extracted from a person's face images are searched independently.
- The **maximum cosine similarity across all embeddings** is taken as the person's match score.
- The staff name is retrieved from `staff_registry[idx]`.

### Classification Decision

```python
if best_sim >= threshold and best_name:
    person['update_type'] = 'staff'
    person['recognized_name'] = best_name
    person['similarity_score'] = round(best_sim, 4)
else:
    person['update_type'] = 'customer'
    customer_data.append((pid, best_emb, person))
```

| Outcome  | Condition                                                  |
| -------- | ---------------------------------------------------------- |
| STAFF    | `best_sim ≥ staff_threshold` and a registry name was found |
| CUSTOMER | `best_sim < staff_threshold` or no registry loaded         |
| NO FACE  | No embeddings could be extracted from any image            |

### Threshold Values

| Context        | Default `staff_threshold`                           | Notes                                                                          |
| -------------- | --------------------------------------------------- | ------------------------------------------------------------------------------ |
| `pipeline.py`  | `0.80` (from `config.json fr.similarity_threshold`) | Balanced precision/recall — avoids false staff matches during live processing  |
| `rerun_fr.py`  | `0.60`                                              | Higher recall; use when pipeline misses staff detections                       |
| `fr_report.py` | `0.60`                                              | Same as rerun                                                                  |

A higher threshold reduces false positives (customers misidentified as staff) at the cost of more false negatives (staff not recognized). A lower threshold improves recall at the cost of potential misidentification.

---

## Customer Clustering (Overview)

After staff matching, persons not recognized as staff enter a greedy single-linkage clustering step to identify repeat visitors.

```python
for i, (pid_i, emb_i, p_i) in enumerate(customer_data):
    if i in assigned:
        continue
    cluster = {'embeddings': [emb_i], ...}
    assigned.add(i)
    for j, (pid_j, emb_j, p_j) in enumerate(customer_data):
        if j in assigned:
            continue
        max_sim = max(float(np.dot(emb_j, ce)) for ce in cluster['embeddings'])
        if max_sim >= cluster_threshold:
            cluster['embeddings'].append(emb_j)
            assigned.add(j)
```

- Clusters are grown by checking the cosine similarity between a candidate and **every embedding already in the cluster** (single-linkage / max-link).
- The `cluster_threshold` default is `0.55` in `rerun_fr.py` / `fr_report.py`, and `fr.customer_cluster_threshold` (`0.60` from `config.json`) in `pipeline.py`.
- A cluster with `visit_count > 1` is a **repeated visitor** — the same person was tracked across multiple time windows.
- Clusters are sorted descending by `visit_count` in the output.
- No-face persons are intentionally excluded from clustering (they have no embedding to compare).

---

## Key Functions

### `run_fr_processing` (`pipeline.py:1087`)

- **Purpose:** Run the full FR pipeline on all unprocessed persons in `state["persons"]`.
- **Input:** `date_str` (str), `state` (dict from `state.json`)
- **Output:** Writes `state["fr_results"]` in-place; no return value.
- **Logic:**
  1. Load InsightFace and FAISS index.
  2. For each person without `fr_processed=True`: apply 3 filters, extract embeddings, search FAISS, classify.
  3. Cluster all customers.
  4. Build `staff_results`, `customer_clusters`, `no_face` lists.
  5. Write results to `state["fr_results"]`.
- **Role in Recognition:** The primary FR entry point in the main pipeline. Runs once at end of day's clip processing.

---

### `extract_embeddings` (`rerun_fr.py:86`)

- **Purpose:** Extract face embeddings from all images in a person's folder, using multi-strategy detection.
- **Input:** `face_app` (InsightFace), `images_folder` (str path), `body_snapshot` (optional fallback path)
- **Output:** `(embeddings: list[np.ndarray], stats: dict)` — list of L2-normalized 512-dim vectors, and a dict with `images_scanned`, `faces_extracted`, `strategies_used`.
- **Logic:**
  1. List all `.jpg` files in the folder (excluding `body_snapshot.jpg`), sorted in reverse order.
  2. Append body snapshot path at the end as a last-resort image.
  3. For each image: call `_extract_with_strategies`; if successful, append embedding.
- **Role in Recognition:** Provides embeddings for FAISS search in `rerun_fr.py` and `fr_report.py`.

---

### `_extract_with_strategies` (`rerun_fr.py:125`)

- **Purpose:** Try up to four preprocessing strategies to detect a face in a single image.
- **Input:** `face_app` (InsightFace), `img` (BGR numpy array)
- **Output:** `(embedding: np.ndarray | None, strategy_name: str | None)`
- **Logic:** Tries `direct → upscaled → enhanced → padded` in sequence, returning at first success. Returns `(None, None)` if all strategies fail.
- **Role in Recognition:** Maximizes embedding extraction rate for difficult face crops.

---

### `_detect_face` (`rerun_fr.py:158`, `fr_report.py:202`)

- **Purpose:** Run InsightFace detection on an image, select the largest face, return a normalized embedding.
- **Input:** `face_app`, `img` (BGR numpy array)
- **Output:** `np.ndarray` (512-dim float32, L2-normalized) or `None`
- **Logic:**
  1. `face_app.get(img)` — returns all detected faces with bounding boxes, landmarks, and embeddings.
  2. Select face with maximum bounding box area: `max(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))`.
  3. L2-normalize the embedding.
- **Role in Recognition:** The innermost detection call; called by `_extract_with_strategies` for each preprocessing variant.

---

### `register_staff_from_ids` (`register_staff.py:117`)

- **Purpose:** Register a staff member into the FAISS index using face images from one or more person track IDs.
- **Input:** `date_str`, `name` (str), `person_ids` (list[int])
- **Output:** `True` on success, `False` on failure. Side effect: updates `vector_db/faiss.index` and `vector_db/staff_registry.json`.
- **Logic:**
  1. Load face images from all matching person folders.
  2. Extract L2-normalized embeddings from each image.
  3. Compute **average embedding** across all extracted embeddings, then L2-normalize the result.
  4. If name already exists: rebuild the index without the old entry, then add the merged average.
  5. Add the average embedding to FAISS and the name to the registry.
  6. Call `_pick_diverse` to select up to 4 extra diverse embeddings; add each to the index with the same name entry.
- **Role in Recognition:** Builds the staff reference database. More embeddings per staff member increases recognition robustness.

---

### `_pick_diverse` (`register_staff.py:232`)

- **Purpose:** Select the most diverse face embeddings relative to the staff average, to maximize coverage of different angles and lighting.
- **Input:** `embeddings` (list[np.ndarray]), `avg` (L2-normalized average embedding), `max_extra` (int, default 4)
- **Output:** List of up to `max_extra` embeddings.
- **Logic:**
  1. For each embedding, compute `1.0 - dot(emb, avg)` as the distance from the average.
  2. Sort descending by distance (most different from average comes first).
  3. Return the top `max_extra` embeddings.
- **Role in Recognition:** The FAISS index stores multiple vectors per staff member: one average + up to 4 diverse. This lets the search succeed across varied face orientations.

---

## Input / Output

### Input

| Item                            | Description                                                               |
| ------------------------------- | ------------------------------------------------------------------------- |
| `faces/person_XXXX/*.jpg`       | Face crop images captured during tracking; filenames encode quality score |
| `vector_db/faiss.index`         | Pre-built FAISS IndexFlatIP(512) with staff embeddings                    |
| `vector_db/staff_registry.json` | Parallel registry mapping FAISS index positions to staff names            |
| `state.json`                    | Processing state with person metadata (`pipeline.py`, `rerun_fr.py` only) |
| `config.json`                   | Thresholds: `fr.similarity_threshold`, `fr.customer_cluster_threshold`    |

### Output (written to `state["fr_results"]`)

```json
{
  "staff": [
    {
      "name": "Ravi",
      "track_ids": [3, 7],
      "visit_count": 2,
      "first_seen": "...",
      "last_seen": "...",
      "best_face": "path/to/best_face.jpg",
      "detections": [...]
    }
  ],
  "customer_clusters": [
    {
      "cluster_id": 1,
      "track_ids": [12, 15],
      "visit_count": 2,
      "first_seen": "...",
      "last_seen": "...",
      "best_face": "path/to/best_face.jpg"
    }
  ],
  "no_face": [...],
  "no_face_count": 5
}
```

---

## Thresholds and Configuration

| Config Key                            | Default                                    | Effect                                                                                                          |
| ------------------------------------- | ------------------------------------------ | --------------------------------------------------------------------------------------------------------------- |
| `fr.similarity_threshold`             | `0.80` (pipeline) / `0.60` (rerun, report) | Minimum cosine similarity to classify as staff. Higher → fewer staff matches, fewer false positives.            |
| `fr.customer_cluster_threshold`       | `0.60` (pipeline) / `0.55` (rerun, report) | Minimum cosine similarity to merge two persons into the same customer cluster. Lower → more aggressive merging. |
| `face_capture.min_quality`            | `25`                                       | Quality floor for pre-recognition filter 1.                                                                     |
| `face_capture.min_skin_ratio`         | `0.20`                                     | Minimum skin-tone pixel fraction for pre-recognition filter 2.                                                  |
| `face_capture.min_keypoints`          | `3`                                        | Minimum landmarks inside bbox for pre-recognition filter 3.                                                     |
| `face_capture.post_filter_keep_top_n` | `20`                                       | Max embeddings retained per person after quality filtering.                                                     |
| `face_capture.post_filter_enabled`    | `true`                                     | If false, all three pre-recognition filters are disabled.                                                       |

### Threshold Tuning Impact

Raising `similarity_threshold`:

- Fewer staff recognitions (higher confidence required)
- Fewer false positives (customers labeled as staff)
- More persons fall into customer pool

Lowering `similarity_threshold`:

- More staff recognitions (partial/angled faces may match)
- Risk of false positives if two people have moderately similar appearance

Lowering `customer_cluster_threshold`:

- More aggressive grouping — distinct persons may be merged into one cluster
- Inflates `visit_count` for a cluster; repeated-visitor flag triggers more often

---

## Edge Cases & Failure Handling

| Condition                                          | Behavior                                                                                   |
| -------------------------------------------------- | ------------------------------------------------------------------------------------------ |
| No `faiss.index` file                              | FAISS search is skipped; all persons classified as customers                               |
| Image cannot be opened                             | Skipped silently (returns `None` from `cv2.imread`)                                        |
| All 4 extraction strategies fail                   | Person is added to `no_face_list`; body snapshot recorded but not clustered                |
| Face detected but embedding is `None`              | Skipped; not added to candidates                                                           |
| Zero-norm embedding                                | Skip normalization step (guard: `if norm > 0`)                                             |
| FAISS index is empty (`ntotal == 0`)               | Search is skipped; everyone is a customer                                                  |
| Staff name not found in registry at returned index | `best_name` stays `None`; person is classified as customer even if `best_sim >= threshold` |
| Person folder does not exist                       | Skipped; `os.path.exists(folder)` guard                                                    |
| Face detected outside expanded keypoint bbox       | Rejected by Filter 3 as likely non-frontal or phantom detection                            |

---

## Performance Considerations

- **Filtering order matters**: quality score (parse filename, no I/O) → skin-tone (image load + HSV, no model) → InsightFace (GPU model). This minimizes GPU usage by eliminating bad images early.
- **Multi-embedding search is O(n × k)** where n = embeddings per person (max 20) and k = FAISS index size. With a small staff registry (< 50 vectors), this is negligible.
- **FAISS IndexFlatIP is exact** (no approximation). For >1000 staff vectors, consider `IndexIVFFlat` with `nlist` partitions for speed.
- **insightface CUDA** inference on GPU is used when available. CPU fallback is significantly slower for high face counts.
- **Body snapshot fallback** (`rerun_fr.py`): InsightFace's face detection on a full-body image has low accuracy. Embeddings from body snapshots are used as last resort and may produce lower-quality matches.
- **CLAHE enhancement** converts to LAB color space, applies contrast equalization only to the lightness channel, then converts back — preserving hue and saturation so skin-tone detection still works.

---

## Integration with Other Modules

| Module                                    | Relationship                                                                                  |
| ----------------------------------------- | --------------------------------------------------------------------------------------------- |
| `pipeline.py` tracking (`PersonTracker`)  | Produces `faces/person_XXXX/` folders and `state["persons"]` list that FR reads               |
| `filter_faces.py`                         | Standalone version of the same 3-filter logic; can be run before FR to pre-clean face folders |
| `register_staff.py`                       | Writes `vector_db/faiss.index` and `vector_db/staff_registry.json` that FR reads              |
| `pipeline.py` report (`generate_report`)  | Reads `state["fr_results"]` written by FR to populate Excel sheets                            |
| `rerun_fr.py` report (`_generate_report`) | Same, but generates a separate `fr_rerun_report_<date>.xlsx` with a dedicated No Face sheet   |

### Staff Registration → FR Matching Flow

```
register_staff.py
  extract_embeddings (InsightFace, det_thresh=0.3)
    → average embedding + up to 4 diverse embeddings
    → L2 normalize all
    → faiss.IndexFlatIP.add(embedding)    # stored as inner-product vectors
    → staff_registry.json entry           # name + metadata at same index position

pipeline / rerun_fr / fr_report
  InsightFace extracts query embedding from face image
    → L2 normalize
    → faiss_index.search(query, k=1)      # inner product = cosine similarity
    → staff_registry[result_index]['name'] → recognized name
```

The symmetry between registration (L2-normalized, stored in IndexFlatIP) and query (L2-normalized, searched via IndexFlatIP) is what makes the inner product mathematically equivalent to cosine similarity.
