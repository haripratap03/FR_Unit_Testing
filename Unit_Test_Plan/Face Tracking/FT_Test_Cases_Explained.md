# Face Tracking Unit Test Report

**Date:** 2026-04-30
**Test file:** `Unit_Test_Test_Cases/Face Tracking/test_face_tracking.py`
**Runner:** `pytest` (verbosity=2)
**Total:** 61 tests — **61 passed, 0 failed, 0 errors**
**Duration:** < 1s (no GPU, no disk I/O except tmp_path .npy fixtures)
**Result:** PASS

---

## What Is Being Tested

The Face Tracking feature spans five classes in `pipeline.py`:

| Class | Role in pipeline |
|---|---|
| `OSNetExtractor` | Wraps the OSNet ReID model; produces 512-d L2-normalized embeddings from person crops |
| `ActiveTrack` | Confirmed person track; holds a rolling embedding bank, last bounding box, and timestamp |
| `PendingTrack` | Unconfirmed track candidate; buffers observations until `WARMUP_FRAMES` is reached |
| `PersonTracker` | Orchestrates per-frame matching (3-phase greedy), track promotion, and state persistence |
| `LineCrossDetector` | Optional entry/exit line logic (wired to `None` in production v1.7.0; tested in isolation) |

No actual ML model inference runs in any test. OSNet is bypassed via `MagicMock`; `ActiveTrack` and `PendingTrack` instances are built by calling `__new__` and setting attributes directly, which eliminates all GPU and I/O dependencies.

---

## Summary by Test Class

| Class | Test IDs | Tests | Passed | Scope |
|---|---|---|---|---|
| `TestExtractBatch` | FT-01..05 | 5 | 5 | `OSNetExtractor.extract_batch` — `pipeline.py:280` |
| `TestExtract` | FT-06..07 | 2 | 2 | `OSNetExtractor.extract` — `pipeline.py:265` |
| `TestActiveTrackRef` | FT-08..10 | 3 | 3 | `ActiveTrack.ref` property — `pipeline.py:360` |
| `TestActiveTrackUpdateEmbedding` | FT-11..13 | 3 | 3 | `ActiveTrack.update_embedding` — `pipeline.py:368` |
| `TestActiveTrackSerialize` | FT-14..17 | 4 | 4 | `ActiveTrack.serialize / deserialize` — `pipeline.py:370` |
| `TestPendingTrack` | FT-18..22 | 5 | 5 | `PendingTrack.__init__`, `.ref`, `.add` — `pipeline.py:392` |
| `TestBoxDistance` | FT-23..26 | 4 | 4 | `PersonTracker._box_distance` — `pipeline.py:428` |
| `TestMatchOrCreate` | FT-27..40 | 14 | 14 | `PersonTracker.match_or_create` — `pipeline.py:440` |
| `TestGetLost` | FT-41..45 | 5 | 5 | `PersonTracker.get_lost` — `pipeline.py:530` |
| `TestSerializeRestoreState` | FT-46..48 | 3 | 3 | `PersonTracker.serialize_state / restore_state` — `pipeline.py:560` |
| `TestPersonTrackerRemove` | FT-49..50 | 2 | 2 | `PersonTracker.remove` — `pipeline.py:558` |
| `TestLineCrossGetSide` | FT-51..54 | 4 | 4 | `LineCrossDetector.get_side` — `pipeline.py:320` |
| `TestLineCrossUpdate` | FT-55..58 | 4 | 4 | `LineCrossDetector.update` — `pipeline.py:330` |
| `TestLineCrossCleanup` | FT-59..60 | 2 | 2 | `LineCrossDetector.cleanup` — `pipeline.py:340` |
| `TestLineCrossState` | FT-61 | 1 | 1 | `LineCrossDetector.get_state / restore_state` — `pipeline.py:344` |

---

## Detailed Results

---

### TestExtractBatch — `OSNetExtractor.extract_batch` (pipeline.py:280)

**What `extract_batch` does:** Takes a full video frame and a list of bounding boxes. It slices person crops, filters crops below the minimum size threshold (32×64 px), batches the valid crops through the OSNet ReID model, L2-normalizes every output, and returns a list aligned to the input boxes — with `None` at positions where the crop was too small or failed.

| ID | Test Function | Setup | Assertion | Result |
|---|---|---|---|---|
| FT-01 | `test_ft01_returns_list_aligned_with_boxes` | 3 valid boxes (all ≥ 32×64); mock returns a `(3, 512)` feature matrix | `len(result)==3`; each entry is a non-None `ndarray` of shape `(512,)` | PASS |
| FT-02 | `test_ft02_undersized_crop_returns_none_at_correct_index` | box[0] is 25×55 (too small); box[1] is valid; mock returns `(1, 512)` | `result[0] is None`; `result[1].shape==(512,)` | PASS |
| FT-03 | `test_ft03_all_undersized_no_gpu_call` | Both boxes are below 32×64 | `result==[None, None]`; `get_features` never called | PASS |
| FT-04 | `test_ft04_output_embeddings_are_l2_normalized` | Mock returns constant-3.5 vectors (magnitude ≠ 1) | `‖result[i]‖ ≈ 1.0` for all entries | PASS |
| FT-05 | `test_ft05_batch_exception_triggers_sequential_fallback` | `get_features` raises `RuntimeError` on first (batch) call, then succeeds for each individual crop | `len(result)==2`; both entries are valid unit vectors; no exception propagated | PASS |

**Why these matter:**
- FT-02/03 confirm that the `None`-guard prevents downstream crash when a person crop is too small for ReID.
- FT-04 ensures FAISS cosine comparisons are valid (inner product on unit vectors = cosine similarity).
- FT-05 confirms the sequential fallback path recovers from GPU OOM without raising.

---

### TestExtract — `OSNetExtractor.extract` (pipeline.py:265)

**What `extract` does:** Single-box variant of `extract_batch`. Clamps the box to frame boundaries, checks the crop size, then calls the model.

| ID | Test Function | Setup | Assertion | Result |
|---|---|---|---|---|
| FT-06 | `test_ft06_crop_below_size_threshold_returns_none` | 30×50 crop on 480×640 frame (below 32×64 threshold) | `result is None` | PASS |
| FT-07 | `test_ft07_bbox_clamped_to_frame_boundaries_no_exception` | Box extends beyond frame edges | No exception raised; result is `None` or `ndarray` | PASS |

---

### TestActiveTrackRef — `ActiveTrack.ref` property (pipeline.py:360)

**What `ActiveTrack.ref` does:** Computes a recency-weighted mean over all embeddings in the bank. Weights are `np.linspace(0.5, 1.0, len(bank))`, so the newest embedding gets weight `1.0` and the oldest gets `0.5`. The result is L2-normalized before returning, so `ref` is always a unit vector usable directly in cosine comparisons.

**Design intent:** A recency-weighted reference adapts faster to appearance changes (e.g., lighting, viewpoint) while retaining older observations for robustness. A flat-mean would dilute recent information equally with stale history.

| ID | Test Function | Setup | Key Assertion | Result |
|---|---|---|---|---|
| FT-08 | `test_ft08_single_embedding_returns_itself` | `bank=[e1]` where `e1=basis_emb(0)=[1,0,...,0]` | `track.ref ≈ e1`; `‖ref‖ ≈ 1.0`. With a single entry `linspace(0.5,1.0,1)=[1.0]`, so the weighted mean is just `1.0 × e1`. | PASS |
| FT-09 | `test_ft09_recency_weighting_newest_dominates` | `bank=[basis(0) oldest, basis(1), basis(2) newest]`; weights `[0.5, 0.75, 1.0]` | After normalization: `ref[2] > ref[1] > ref[0]` — axis-2 (newest) component dominates | PASS |
| FT-10 | `test_ft10_always_returns_l2_normalized` | Full bank of 10 random unit vectors | `‖track.ref‖ ≈ 1.0` (within `1e-5`) | PASS |

**FT-08 in depth:** `linspace(0.5, 1.0, 1)` returns `[1.0]`, so the single-element bank produces `weighted_sum = 1.0 × e1 = e1`, and after L2 normalization (which is a no-op on a unit vector) `ref == e1`. This confirms the edge case where the bank has only one entry behaves identically to a direct lookup.

**FT-09 in depth:** With `weights=[0.5, 0.75, 1.0]` and orthogonal basis vectors, the unnormalized mean is `0.5·e0 + 0.75·e1 + 1.0·e2`. After L2-normalization the direction is preserved, so the axis-2 component is largest (weight 1.0), followed by axis-1 (0.75), then axis-0 (0.5).

---

### TestActiveTrackUpdateEmbedding — `ActiveTrack.update_embedding` (pipeline.py:368)

**What `update_embedding` does:** Appends the new embedding to `bank` (a `deque(maxlen=BANK_SIZE)`). Because `deque` with `maxlen` automatically evicts the left (oldest) entry when full, the bank always holds at most `BANK_SIZE` embeddings in insertion order (oldest on the left, newest on the right).

**Design intent:** Bounding the bank to `BANK_SIZE=10` prevents unbounded memory growth during long tracking sequences and keeps `ref` computation O(BANK_SIZE).

| ID | Test Function | Setup | Key Assertion | Result |
|---|---|---|---|---|
| FT-11 | `test_ft11_bank_grows_up_to_bank_size` | `BANK_SIZE=5`; start with 1 embedding; call `update_embedding` 4 more times | `len(bank)==5`; all 5 distinct embeddings present | PASS |
| FT-12 | `test_ft12_evicts_oldest_when_full` | `BANK_SIZE=3`; `bank=[e1,e2,e3]`; call `update_embedding(e4)` | `list(bank)==[e2,e3,e4]`; `e1` is gone | PASS |
| FT-13 | `test_ft13_ref_recomputed_after_update` | `bank=[e1,e2]`; capture `ref_before`; call `update_embedding(basis(2))` | `not np.allclose(ref_before, track.ref)` — ref reflects the updated bank | PASS |

**FT-12 in depth (CRITICAL):** This is the memory-bounding invariant. If eviction failed, the bank would grow unboundedly and `ref` computation would slow over time. The test directly checks that `e1` (the oldest) is gone after a fourth insertion into a size-3 bank.

**FT-13 in depth:** `ref` is a property (computed on access), not stored. This test confirms there is no stale cached value — every `.ref` access recomputes over the current bank contents.

---

### TestActiveTrackSerialize — `ActiveTrack.serialize / deserialize` (pipeline.py:370)

**What serialize/deserialize does:** `serialize(tracks_dir, track_id)` saves `bank` as a NumPy `.npy` file (`track_XXXX_bank.npy` with zero-padded 4-digit ID), and returns a dict with `bank_file`, `last_box`, `last_timestamp`, and `lost_seconds`. `deserialize(data, tracks_dir)` reads the `.npy` file back and reconstructs the `ActiveTrack`.

**Design intent:** Cross-clip continuity. After every clip, each active track is serialized so that at the start of the next clip, the same track ID resumes with its full embedding history intact. Without serialization, every person would get a new ID at each clip boundary.

| ID | Test Function | Setup | Key Assertion | Result |
|---|---|---|---|---|
| FT-14 | `test_ft14_creates_zero_padded_npy_filename` | `track_id=7`; `tracks_dir=tmp_path` | File `track_0007_bank.npy` exists; `result["bank_file"]=="track_0007_bank.npy"` | PASS |
| FT-15 | `test_ft15_round_trip_fidelity` | 5 unit-norm embeddings; `last_box=[100,50,200,300]`; `lost_seconds=2.5`; `track_id=3` | All 5 bank entries identical within `1e-6`; `last_box` and `lost_seconds` preserved | PASS |
| FT-16 | `test_ft16_missing_npy_raises_exception` | `data={"bank_file":"track_9999_bank.npy",...}`; no `.npy` files in `tmp_path` | `FileNotFoundError` (or similar) is raised | PASS |
| FT-17 | `test_ft17_bank_embeddings_restored_as_float32` | Serialize then deserialize one track | `list(restored.bank)[0].dtype == np.float32` | PASS |

**FT-14 in depth:** The zero-padded filename (`%04d`) ensures file ordering is consistent and human-readable. A track ID of 7 must produce `track_0007_bank.npy`, not `track_7_bank.npy`.

**FT-15 in depth (CRITICAL):** This is the cross-clip continuity test. If any embedding value drifts during serialization (e.g., float64 upcast in NumPy save/load), cosine similarity computations at the next clip start would produce wrong distances. The `atol=1e-6` tolerance on `float32` is tight enough to catch any dtype conversion issue.

**FT-16 in depth:** If the `.npy` file is missing (e.g., corrupt write, accidental delete), `deserialize` must fail loudly rather than silently producing a track with an empty bank.

---

### TestPendingTrack — `PendingTrack.__init__ / .ref / .add` (pipeline.py:392)

**What `PendingTrack` does:** Buffers a new detection candidate until it has been observed for `WARMUP_FRAMES` consecutive frames. This prevents transient detections (shadows, reflections, brief occlusions) from immediately creating track IDs. Once `frames_seen >= WARMUP_FRAMES` and a new match occurs, the pending track is promoted to an `ActiveTrack`.

**Key differences from `ActiveTrack`:**
- `PendingTrack.ref` uses **unweighted** mean (no recency bias) — it's a simpler accumulator.
- The bank is **unbounded** (a plain list, not a `deque`) because pending tracks are short-lived.
- `frames_seen` is incremented on every `add()` call, unlike `ActiveTrack` which does not count frames.

| ID | Test Function | Setup | Key Assertion | Result |
|---|---|---|---|---|
| FT-18 | `test_ft18_init_frames_seen_starts_at_1` | `PendingTrack(e1, [10,10,50,80])` | `frames_seen==1`; `len(bank)==1`; `last_box==[10,10,50,80]` | PASS |
| FT-19 | `test_ft19_ref_unweighted_mean_l2_normalized` | `bank=[basis(0), basis(1)]` | `ref[0] ≈ ref[1]` (equal weights); `‖ref‖ ≈ 1.0` | PASS |
| FT-20 | `test_ft20_add_increments_frames_seen_and_updates_box` | `PendingTrack(e1,[10,10,50,80])`; two `add()` calls | `frames_seen==3`; `last_box==[20,20,60,90]`; `len(bank)==3` | PASS |
| FT-21 | `test_ft21_bank_is_unbounded` | 15 `add()` calls | `len(bank)==16` (1 initial + 15 added); no eviction | PASS |
| FT-22 | `test_ft22_ref_updates_dynamically_after_add` | `bank=[basis(0)]`; `add(basis(1))` | `not np.allclose(ref_1, ref_2)` | PASS |

**FT-18 in depth:** A newly constructed `PendingTrack` must start with `frames_seen=1` (not 0), because construction itself counts as the first observation. If it started at 0, a `WARMUP_FRAMES=1` config would require an extra match to promote, off by one.

**FT-19 in depth:** `PendingTrack.ref` is an unweighted mean. With two orthogonal basis vectors `[1,0,...,0]` and `[0,1,...,0]`, the unnormalized mean is `[0.5, 0.5, ..., 0]`. After L2 normalization both components are equal. This is distinct from `ActiveTrack.ref` which would weight the second entry higher.

**FT-21 in depth:** The unbounded bank is intentional — pending tracks are short-lived (at most `WARMUP_FRAMES` frames), so there's no need for an eviction policy. The test confirms that no silent truncation occurs.

---

### TestBoxDistance — `PersonTracker._box_distance` (pipeline.py:428)

**What `_box_distance` does:** Computes the normalized Euclidean distance between the centroids of two bounding boxes, dividing by `sqrt(frame_w² + frame_h²)` (the diagonal of the frame). The result is a frame-size-agnostic distance in `[0, 1]` (approximately — can slightly exceed 1 for extreme cases).

**Design intent:** Used as a spatial gate in `match_or_create`. A detection that teleports across the frame is rejected even if its ReID embedding happens to be close to an active track. The normalization ensures the gate threshold (`MAX_BOX_JUMP=0.30`) works consistently across different camera resolutions.

| ID | Test Function | Setup | Key Assertion | Result |
|---|---|---|---|---|
| FT-23 | `test_ft23_identical_boxes_produce_distance_0` | `box_a == box_b == [200,100,400,500]` on 1280×720 | `distance == 0.0` | PASS |
| FT-24 | `test_ft24_opposite_corners_produce_large_distance` | `box_a=[0,0,100,100]` (top-left); `box_b=[1180,620,1280,720]` (bottom-right) | `distance > 1.0` | PASS |
| FT-25 | `test_ft25_normalized_by_frame_dimensions` | Same 200 px horizontal shift on 640×360 vs 3840×2160 frame | `dist_small > dist_large` (same absolute shift = larger relative shift on a small frame) | PASS |
| FT-26 | `test_ft26_small_displacement_below_spatial_gate` | ~10 px shift on 1280×720 frame | `distance < 0.05` (well below `MAX_BOX_JUMP=0.30`) | PASS |

**FT-25 in depth:** This test confirms the normalization is frame-aware. A person walking 200 px on a 640-wide camera has moved ~31% of the frame width; the same 200 px on a 3840-wide camera is only ~5%. If the gate were pixel-based rather than normalized, it would behave incorrectly on high-resolution feeds.

---

### TestMatchOrCreate — `PersonTracker.match_or_create` (pipeline.py:440)

**What `match_or_create` does:** The core per-frame matching function. It runs in three phases:

- **Phase 1:** Build a cost matrix (cosine distance) between all active track `ref` vectors and all detection embeddings. Apply spatial gate to invalidate teleports. Run greedy assignment (lowest-cost pair first). Matched detections update the active track; unmatched go to Phase 2.
- **Phase 2:** Try to match unmatched detections against pending tracks. Matches increment `frames_seen`; when `frames_seen >= WARMUP_FRAMES`, the pending track is promoted to an active track with the accumulated embedding history.
- **Phase 3:** Any detection still unmatched after Phase 2 creates a new `PendingTrack` with `frames_seen=1`.

**Test patching:** FT-27..40 patch `REID_DISTANCE=0.40` (instead of the config value `0.65`) and `UPDATE_GATE=0.80`, so test distances stated in the spec (0.10, 0.35, 0.45…) behave exactly as described. The effective bank-update threshold is `0.40 × 0.80 = 0.32`.

| ID | Test Function | Setup | Key Assertion | Result |
|---|---|---|---|---|
| FT-27 | `test_ft27_single_detection_matches_active_track` | 1 active track (`e_ref`); detection `dist=0.10` from `e_ref`; `gap_seconds=0.1` | `result==[(1, box)]`; `active[1].last_box` updated to new box | PASS |
| FT-28 | `test_ft28_distance_gte_reid_not_matched` | 1 active track; detection `dist=0.45` (≥ `REID=0.40`) | `result==[]`; 1 new pending created | PASS |
| FT-29 | `test_ft29_spatial_gate_blocks_cross_frame_teleport` | Track at top-left `[10,10,100,100]`; detection at bottom-right `[1180,620,1280,720]`; embedding `dist=0.05` | Track NOT matched despite close embedding; detection goes to pending | PASS |
| FT-30 | `test_ft30_large_gap_seconds_widens_spatial_gate` | Track at top-left; detection mid-frame with `center_dist≈0.62`; `dist=0.15`; `gap_seconds=35` | Matched (`gate` widens with time, capped at 0.8; `0.62 < 0.8`) | PASS |
| FT-31 | `test_ft31_close_match_updates_embedding_bank` | Active track `bank=[e_ref]`; detection `dist=0.20` (< update threshold `0.32`) | `len(bank)==2` after match | PASS |
| FT-32 | `test_ft32_borderline_match_no_bank_update` | Active track; detection `dist=0.35` (`0.32 ≤ dist < 0.40`) | Match returned; `len(bank)` unchanged; `last_box` IS updated | PASS |
| FT-33 | `test_ft33_greedy_selects_globally_best_pair` | Track A (`basis(0)`), Track B (`basis(1)`); `d1` close to B, `d2` close to A | Both A and B matched (cross-assignment by global greedy) | PASS |
| FT-34 | `test_ft34_phase2_increments_frames_seen` | `WARMUP=5`; 1 pending `frames_seen=2`; detection `dist=0.08` from pending | `result==[]`; `pending.frames_seen==3`; not yet promoted | PASS |
| FT-35 | `test_ft35_pending_promoted_to_active_at_warmup` | `WARMUP=3`; 1 pending `frames_seen=2`; detection matches | 1 active track; `len(result)==1`; `len(pending)==0` | PASS |
| FT-36 | `test_ft36_promoted_track_seeded_with_pending_history` | `WARMUP=3`; pending `bank=[ep1,ep2]` (`frames_seen=2`); 3rd match triggers promotion | `len(new_active.bank) >= 3` (pending history preserved) | PASS |
| FT-37 | `test_ft37_unmatched_detection_creates_pending` | Empty active and pending; 1 detection | `result==[]`; `len(pending)==1`; `pending.frames_seen==1` | PASS |
| FT-38 | `test_ft38_none_embedding_silently_skipped` | `osnet` returns `[None, near_emb]`; 1 active track close to `near_emb` | Only `det[1]` matched; `None`-embedding detection NOT added to pending; no crash | PASS |
| FT-39 | `test_ft39_empty_boxes_returns_empty_result` | 2 active tracks; `boxes=[]` | `result==[]`; active pool unchanged | PASS |
| FT-40 | `test_ft40_more_active_tracks_than_detections` | 3 active tracks; 1 detection close to track 2 only | `result==[(2, box)]`; all 3 tracks remain in `active` | PASS |

**Critical cases explained:**

**FT-27 (CRITICAL):** The happy path — confirms a basic match works and both `last_box` and the return list are updated correctly. The foundation of the entire tracking system.

**FT-29 (CRITICAL):** The spatial gate is the only defense against ReID false positives when two people have similar appearance (e.g., same uniform). Even if cosine distance is 0.05 (very close), a cross-frame jump must be blocked. Without this gate, ID switches across the room would occur whenever staff appearances coincide.

**FT-31 vs FT-32 (CRITICAL):** These two cases together define the bank-update policy: update only when the match is confident (`dist < REID × UPDATE_GATE = 0.32`), not just any match below `REID_DISTANCE=0.40`. A borderline match at `dist=0.35` is valid for continuity but not trustworthy enough to update the appearance model. Including low-confidence embeddings in the bank would drift the reference vector toward noisy observations.

**FT-33 (CRITICAL — greedy):** The cost matrix is solved globally (sort all (track, detection) pairs by cost, assign the cheapest unassigned pair first). If matching were done sequentially per-track, track A might "steal" the detection that would have been the global optimum for track B, causing track B to go unmatched unnecessarily.

**FT-38 (CRITICAL):** `None` embeddings come from undersized crops (see FT-02). They must be silently skipped — the person is still detected, just not embeddable. If `None` were forwarded to the cost matrix, it would cause a NumPy error. If it were added to pending, it would create a `PendingTrack` with no embedding, breaking all future `ref` computations.

**FT-35/36 (CRITICAL — promotion):** The pending→active transition must carry forward all accumulated embeddings so the new `ActiveTrack` starts with a meaningful bank. Without history seeding, the promoted track's `ref` would be computed from only one embedding, making it immediately vulnerable to ID switches.

**FT-30 (gap_seconds gate):** When a clip boundary has a long timestamp gap (e.g., 35s), a person who was at the store entrance may have moved significantly. The spatial gate widens proportionally (capped at `MAX_BOX_JUMP=0.8`) to allow re-matching across this gap. This prevents creating a duplicate track ID after brief recording interruptions.

---

### TestGetLost — `PersonTracker.get_lost` (pipeline.py:530)

**What `get_lost` does:** Scans all active tracks and returns a list of IDs where `elapsed_seconds > max_lost_seconds`. `elapsed_seconds` is computed from `track.last_timestamp` (which can be either a `datetime` object or an ISO-8601 string, depending on whether it was restored from `state.json`).

**Design intent:** Tracks that haven't been seen for `max_lost_seconds` (default 25s from config) are finalized — their data is written to `state["persons"]` and they are removed from `active`. This prevents stale track IDs from being re-assigned to new people and bounds memory usage.

| ID | Test Function | Setup | Key Assertion | Result |
|---|---|---|---|---|
| FT-41 | `test_ft41_returns_only_stale_track_ids` | Track 1: `last_seen = now - 30s`; Track 2: `last_seen = now - 100s`; `max_lost=90` | `[2]` only (100s > 90s; 30s < 90s) | PASS |
| FT-42 | `test_ft42_returns_empty_when_all_fresh` | 3 tracks all within 10s; `max_lost=90` | `result==[]` | PASS |
| FT-43 | `test_ft43_empty_active_pool_returns_empty_list` | `tracker.active={}` | `result==[]`; no exception | PASS |
| FT-44 | `test_ft44_iso_string_timestamp_parsed_correctly` | `last_timestamp="2026-04-09T09:00:00"` (string); `current_time=datetime(2026,4,9,9,2,0)` → 120s later; `max_lost=90` | Track ID in returned list (120s > 90s) | PASS |
| FT-45 | `test_ft45_track_exactly_at_threshold_is_not_lost` | Track exactly 90s ago; `max_lost=90` | Track ID NOT in returned list (boundary: `elapsed == threshold` is not lost, gate is strict `>`) | PASS |

**FT-44 in depth (CRITICAL):** When tracks are restored from `state.json` via `restore_state`, `last_timestamp` is a plain string (JSON doesn't have a datetime type). `get_lost` must parse the ISO string before computing elapsed time. If it compared a string to a datetime, Python would raise `TypeError`. This test ensures the parser path is exercised.

**FT-45 in depth:** The boundary condition confirms the gate is strict-greater-than (`elapsed > max_lost`), not greater-than-or-equal. A track seen exactly `max_lost_seconds` ago is still considered active. This prevents edge-case finalization at exactly the threshold (e.g., a person who steps away for exactly 25s then returns).

---

### TestSerializeRestoreState — `PersonTracker.serialize_state / restore_state` (pipeline.py:560)

**What serialize_state / restore_state does:** `serialize_state(tracks_dir)` calls `track.serialize()` for every active track, writes each bank to `tracks_dir/track_XXXX_bank.npy`, and returns a dict containing `active_tracks`, `next_track_id`, and `pending_next`. `restore_state(state_data, tracks_dir)` reconstructs each `ActiveTrack` from its serialized dict, skipping gracefully if the `.npy` file is missing.

**Design intent:** The tracker state is written to `state.json` after every clip. On the next clip, `restore_state` re-populates `tracker.active` so tracking continues seamlessly. This is the mechanism that allows a person to keep their ID across clip boundaries.

| ID | Test Function | Setup | Key Assertion | Result |
|---|---|---|---|---|
| FT-46 | `test_ft46_full_round_trip` | 2 active tracks (IDs 3, 7) each with 5 embeddings; `next_id=10`; `pending_next=22` | `restored.next_id==10`; `set(active.keys())=={3,7}`; bank entries identical within `1e-6` | PASS |
| FT-47 | `test_ft47_empty_tracker_produces_empty_state` | `tracker.active={}` | `state["active_tracks"]=={}` ; `state["next_track_id"]==1`; no `.npy` files written | PASS |
| FT-48 | `test_ft48_restore_skips_missing_npy` | 2 tracks serialized; `track_0002_bank.npy` deleted before restore | `len(restored.active)==1`; track 1 present; no exception raised | PASS |

**FT-46 in depth (CRITICAL):** This is the end-to-end cross-clip continuity test. It verifies that `next_id` is preserved (prevents duplicate IDs on the next clip), that both track IDs survive the round-trip, and that embedding values are bit-for-bit identical (within float32 tolerance) — any drift would corrupt ReID comparisons at the next clip start.

**FT-48 in depth:** A `.npy` file can be missing if a write was interrupted by a crash or if the file system is full. Rather than raising and halting the entire pipeline, `restore_state` logs a warning and skips that track. The remaining tracks are still recovered correctly. This test confirms the partial-recovery path.

---

### TestPersonTrackerRemove — `PersonTracker.remove` (pipeline.py:558)

**What `remove` does:** Deletes a track ID from `tracker.active`. Used when a track has been finalized (lost for too long) to free the slot.

| ID | Test Function | Setup | Key Assertion | Result |
|---|---|---|---|---|
| FT-49 | `test_ft49_removes_track_from_active_pool` | `active={1,2,3}`; `remove(2)` | `2 not in active`; `1` and `3` still present | PASS |
| FT-50 | `test_ft50_remove_nonexistent_is_noop` | `active={1,3}`; `remove(99)` | No exception; pool unchanged at `{1,3}` | PASS |

**FT-50 in depth:** The pipeline may call `remove(tid)` inside a finalization loop. If the ID was already removed by an earlier call (e.g., gap detection finalized all tracks before the clip's normal finalization pass), a second `remove` must be a no-op. A `KeyError` here would abort clip processing.

---

### TestLineCrossGetSide — `LineCrossDetector.get_side` (pipeline.py:320)

**Note:** `LineCrossDetector` is wired to `None` in the live pipeline (v1.7.0). These tests exercise the class logic in isolation as specified in the test plan.

**What `get_side` does:** Given a normalized point `(nx, ny)` (relative to frame `[0,1]`), determines whether the point is above or below the detector's configured line by linear interpolation.

| ID | Test Function | Setup | Key Assertion | Result |
|---|---|---|---|---|
| FT-51 | `test_ft51_point_above_horizontal_line` | Line `y=0.5`; point `(0.5, 0.3)` | `'above'` | PASS |
| FT-52 | `test_ft52_point_below_horizontal_line` | Line `y=0.5`; point `(0.5, 0.7)` | `'below'` | PASS |
| FT-53 | `test_ft53_diagonal_line_interpolation` | Diagonal line `y=x`; point `(0.5, 0.3)` → line at `x=0.5` is `y=0.5`; point is below line_y | `'above'` | PASS |
| FT-54 | `test_ft54_vertical_line_no_zerodivision` | Vertical line `xa==xb=0.5`; point `(0.5, 0.1)` | `'above'`; no `ZeroDivisionError` | PASS |

---

### TestLineCrossUpdate — `LineCrossDetector.update` (pipeline.py:330)

**What `update` does:** Checks whether a person's current position crosses the configured line relative to their previous position. Returns `'enter'`, `'exit'`, or `None`.

| ID | Test Function | Setup | Key Assertion | Result |
|---|---|---|---|---|
| FT-55 | `test_ft55_first_observation_returns_none_records_side` | Fresh detector; `ny=0.3` (above) | `None`; `person_sides[5]=='above'` | PASS |
| FT-56 | `test_ft56_outside_to_inside_emits_enter` | `prev='above'` (outside); `ny=0.7` (inside=`'below'`) | `'enter'` | PASS |
| FT-57 | `test_ft57_inside_to_outside_emits_exit` | `prev='below'` (inside); `ny=0.2` (outside=`'above'`) | `'exit'` | PASS |
| FT-58 | `test_ft58_no_transition_returns_none` | `prev='below'`; `ny=0.8` (still below) | `None` | PASS |

---

### TestLineCrossCleanup — `LineCrossDetector.cleanup` (pipeline.py:340)

**What `cleanup` does:** Removes a track ID's side record when the person is finalized, preventing memory leak from stale entries.

| ID | Test Function | Setup | Key Assertion | Result |
|---|---|---|---|---|
| FT-59 | `test_ft59_removes_specified_track` | `person_sides={3:'above', 7:'below'}`; `cleanup(3)` | `3 not in person_sides`; `{7:'below'}` remains | PASS |
| FT-60 | `test_ft60_cleanup_nonexistent_is_noop` | `person_sides={3:'above'}`; `cleanup(99)` | No exception; unchanged | PASS |

---

### TestLineCrossState — `LineCrossDetector.get_state / restore_state` (pipeline.py:344)

| ID | Test Function | Setup | Key Assertion | Result |
|---|---|---|---|---|
| FT-61 | `test_ft61_round_trip_fidelity` | `person_sides={1:'above', 5:'below', 12:'above'}`; `get_state` then `restore_state` into a new instance | `lcd2.person_sides == {1:'above', 5:'below', 12:'above'}` | PASS |

---

## Coverage Notes

| Function | File | Tests | Coverage |
|---|---|---|---|
| `OSNetExtractor.extract_batch` | `pipeline.py:280` | FT-01..05 | Valid batch, undersized rejection, L2 norm, batch-failure fallback |
| `OSNetExtractor.extract` | `pipeline.py:265` | FT-06..07 | Size rejection, boundary-clamped box |
| `ActiveTrack.ref` | `pipeline.py:360` | FT-08..10 | Single entry, recency weighting, L2 normalization |
| `ActiveTrack.update_embedding` | `pipeline.py:368` | FT-11..13 | Bank growth, eviction of oldest, ref recomputation |
| `ActiveTrack.serialize` | `pipeline.py:370` | FT-14..15 | Filename format, round-trip fidelity |
| `ActiveTrack.deserialize` | `pipeline.py:370` | FT-15..17 | Round-trip, missing file, dtype preservation |
| `PendingTrack.__init__` | `pipeline.py:392` | FT-18 | Initial `frames_seen`, bank, box |
| `PendingTrack.ref` | `pipeline.py:398` | FT-19, FT-22 | Unweighted mean, dynamic recomputation |
| `PendingTrack.add` | `pipeline.py:402` | FT-20..22 | Counter increment, box update, unbounded bank |
| `PersonTracker._box_distance` | `pipeline.py:428` | FT-23..26 | Zero distance, large distance, frame normalization, gate calibration |
| `PersonTracker.match_or_create` | `pipeline.py:440` | FT-27..40 | All 3 phases, spatial gate, bank update policy, greedy assignment, None-embedding guard, gap-seconds widening |
| `PersonTracker.get_lost` | `pipeline.py:530` | FT-41..45 | Stale detection, all-fresh, empty pool, ISO string parse, boundary condition |
| `PersonTracker.serialize_state` | `pipeline.py:560` | FT-46..47 | Round-trip, empty state |
| `PersonTracker.restore_state` | `pipeline.py:560` | FT-46, FT-48 | Full restore, missing-.npy recovery |
| `PersonTracker.remove` | `pipeline.py:558` | FT-49..50 | Present and absent ID |
| `LineCrossDetector.get_side` | `pipeline.py:320` | FT-51..54 | Above/below, diagonal, vertical line |
| `LineCrossDetector.update` | `pipeline.py:330` | FT-55..58 | First obs, enter, exit, no-transition |
| `LineCrossDetector.cleanup` | `pipeline.py:340` | FT-59..60 | Present and absent ID |
| `LineCrossDetector.get_state / restore_state` | `pipeline.py:344` | FT-61 | Round-trip fidelity |

---

## Key Helpers and Mock Strategy

```
unit_vec(seed)           — reproducible L2-normalized 512-d float32 vector
basis_emb(axis)          — standard basis vector e_axis (unit vector along one axis)
make_emb_with_dist(ref, d) — construct a unit vector at exact cosine distance d from ref
mock_osnet()             — OSNetExtractor instance with ._model replaced by MagicMock
make_active_track(emb)   — build ActiveTrack via __new__; set bank/box/timestamp directly
make_tracker()           — build PersonTracker via __new__; set active={}, pending={}
```

`basis_emb` vectors are used when axis ordering in `ref` is being asserted (FT-08, FT-09, FT-19). `unit_vec` vectors with distinct seeds are used elsewhere to avoid accidental cosine coincidences between unrelated embeddings.

FT-27..40 patch `pipeline.REID_DISTANCE=0.40` and `pipeline.UPDATE_GATE=0.80` (instead of the config values `0.65` / `0.80`) so the exact distance values quoted in the spec (`dist=0.10`, `dist=0.35`, `dist=0.45`) map cleanly to pass/fail/bank-update decisions. FT-34..36 additionally patch `WARMUP_FRAMES` to 3 or 5 so pending-track promotion can be triggered within a single test.

---

## Config Values Used (as of 2026-04-29)

| Key | Live value | Patched in tests |
|---|---|---|
| `tracking.reid_distance` | `0.65` | `0.40` in FT-27..40 |
| `tracking.update_gate` | `0.80` | `0.80` (unchanged) |
| `tracking.warmup_frames` | `10` | `3` (FT-35,36) / `5` (FT-34) |
| `tracking.bank_size` | `10` | `3` (FT-12) / `5` (FT-11) |
| `tracking.max_lost_seconds` | `25` | `90` in FT-41..45 (for clarity) |
| `tracking.max_box_jump` | `0.30` | unpatched (FT-29/30 rely on live value) |
| `tracking.min_crop_w` | `32` | unpatched |
| `tracking.min_crop_h` | `64` | unpatched |
