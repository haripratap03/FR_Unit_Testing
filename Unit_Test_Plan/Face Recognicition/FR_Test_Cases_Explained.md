# Face Recognition — Extended Test Case Documentation

**Date:** 2026-04-30
**Scope:** FR-EX, FR-PD, FR-SM, FR-CC series (28 test cases)
**Test file:** `Unit_Test_Test_Cases/Face Recognicition/test_face_recognition.py`
**Functions under test:** `extract_embeddings`, `_pick_diverse`, `run_fr_processing` (classification), greedy clustering loop
**Source files (read-only):** `rerun_fr.py`, `register_staff.py`, `pipeline.py`
**State:** All 28 Active — all PASS

---

## Table of Contents

1. [Group FR-EX — `extract_embeddings`](#group-fr-ex--extract_embeddings)
2. [Group FR-PD — `_pick_diverse`](#group-fr-pd--_pick_diverse)
3. [Group FR-SM — Staff Classification (FAISS matching)](#group-fr-sm--staff-classification-faiss-matching)
4. [Group FR-CC — Greedy Customer Clustering](#group-fr-cc--greedy-customer-clustering)
5. [Coverage Summary](#coverage-summary)

---

## Group FR-EX — `extract_embeddings`

### What the function does

```
extract_embeddings(face_app, images_folder, body_snapshot=None)
                                                        rerun_fr.py : 86
```

This function builds the list of face embeddings for a single person before FR matching.

**Step-by-step flow:**

1. **Scan the folder** — lists all `.jpg` files in `images_folder`, sorted descending
   (highest quality first because filenames carry `_qNN`). `body_snapshot.jpg` is
   explicitly excluded from this list even if present in the folder.
2. **Append body snapshot** — if `body_snapshot` path is provided and the file
   exists, it is appended at the end of the scan list as a last-resort image.
3. **Record `images_scanned`** — the length of the combined list is stored before
   any extraction attempt.
4. **Extract per image** — calls `_extract_with_strategies(face_app, img)` for each
   path. On success, appends the L2-normalized 512-d embedding and increments
   `faces_extracted`.
5. **Return** `(embeddings, stats)` where `stats` has keys `images_scanned`,
   `faces_extracted`, `strategies_used`.

**Key design point:** `body_snapshot.jpg` is intentionally skipped in the folder scan
(`f != 'body_snapshot.jpg'`) so it is never double-counted. It enters the list only via
the explicit `body_snapshot` argument, ensuring it is always processed last.

---

### Test Cases

#### FR-EX-01 — Normal Folder With Multiple Face Images

| Field | Value |
|---|---|
| **Target** | `extract_embeddings` — `rerun_fr.py:86` |
| **Mocking** | `face_app.get()` returns a valid mock face per call; `cv2.imread` returns a real synthetic image; `body_snapshot=None` |
| **Setup** | `tmp_path` contains three face images: `face_001200_q42.jpg`, `face_000800_q55.jpg`, `face_000600_q30.jpg` |
| **Expected** | `len(embeddings) == 3`; `stats['images_scanned'] == 3`; `stats['faces_extracted'] == 3`; all embeddings are unit-norm (`np.linalg.norm(e) ≈ 1.0`) |
| **Verdict** | PASS |
| **Why it matters** | Happy-path baseline — confirms the scan, extraction loop, and stats counter all work end-to-end for the normal case |

---

#### FR-EX-02 — Empty Folder Returns Empty List

| Field | Value |
|---|---|
| **Target** | `extract_embeddings` — `rerun_fr.py:86` |
| **Mocking** | `face_app.get()` never called; `body_snapshot=None` |
| **Setup** | `tmp_path` is an empty directory |
| **Expected** | `embeddings == []`; `stats['images_scanned'] == 0`; `stats['faces_extracted'] == 0`; `face_app.get.call_count == 0` |
| **Verdict** | PASS |
| **Why it matters** | Persons with no saved face images (e.g. tracked but no face capture happened) must produce an empty embedding list without crashing. The caller (`run_fr_processing`) then routes this person to `no_face` |

---

#### FR-EX-03 — `body_snapshot` Excluded From Face Scan, Used as Fallback

| Field | Value |
|---|---|
| **Target** | `extract_embeddings` — `rerun_fr.py:86` |
| **Mocking** | `face_app.get()` returns a valid mock face; `body_snapshot` path provided |
| **Setup** | `tmp_path` contains `face_001200_q42.jpg`, `body_snapshot.jpg`, `face_000600_q30.jpg`; `body_snapshot` arg points to `tmp_path/body_snapshot.jpg` |
| **Expected** | `stats['images_scanned'] == 3`; face images are processed first (two face images + one body snapshot at end); `body_snapshot.jpg` is NOT in the folder-scan list but IS appended as the final entry |
| **Verdict** | PASS |
| **Why it matters** | Verifies the filter `f != 'body_snapshot.jpg'` in the folder scan works correctly. If body_snapshot.jpg were in both the folder scan AND the explicit argument, it would be embedded twice — a subtle overcounting bug this test guards against |

---

#### FR-EX-04 — All Faces Fail, Body Snapshot Succeeds

| Field | Value |
|---|---|
| **Target** | `extract_embeddings` — `rerun_fr.py:86` |
| **Mocking** | `face_app.get()` `side_effect=[[], [mock_face]]` — first call (face image) returns no faces; second call (body snapshot) returns a face |
| **Setup** | `tmp_path`: one face image + `body_snapshot.jpg`; `body_snapshot` arg provided |
| **Expected** | `len(embeddings) == 1`; `stats['faces_extracted'] == 1`; the single embedding originates from the body snapshot |
| **Verdict** | PASS |
| **Why it matters** | The body snapshot is the ultimate fallback for persons where InsightFace cannot detect a face in any of the captured crops. This test proves the fallback actually fires and yields a usable embedding rather than routing to `no_face` unnecessarily |

---

#### FR-EX-05 — Non-JPEG Files Ignored in Scan

| Field | Value |
|---|---|
| **Target** | `extract_embeddings` — `rerun_fr.py:86` |
| **Mocking** | `face_app.get()` called once only |
| **Setup** | `tmp_path`: `face_001_q40.jpg`, `thumbnail.png`, `embedding.npy`, `metadata.txt` |
| **Expected** | `stats['images_scanned'] == 1`; `face_app.get.call_count == 1`; `.png`, `.npy`, `.txt` files are ignored |
| **Verdict** | PASS |
| **Why it matters** | The `faces/` folder may contain auxiliary files (numpy bank files, thumbnails, metadata). The scan must be selective — only `.jpg` files are face images. This test verifies the filter `f.endswith('.jpg')` correctly excludes all non-JPEG files |

---

## Group FR-PD — `_pick_diverse`

### What the function does

```
_pick_diverse(embeddings, avg, max_extra=4)
                                    register_staff.py : 232
```

Used during staff registration to select the most *diverse* embeddings to store in the FAISS index alongside the mean embedding. The goal is better coverage of the staff member's pose and lighting variation.

**Algorithm:**

```
for each emb in embeddings:
    distance = 1.0 - dot(emb, avg)      # cosine distance from mean
sort by distance descending             # most different from avg first
return first max_extra embeddings
```

**Why cosine distance from mean?** The mean embedding represents the "typical" appearance.
Embeddings that are farthest from the mean capture the most unusual poses/lighting — exactly
what improves recall at the FAISS search boundary.

**FAISS implication:** `register_staff_from_ids()` stores: `[mean_embedding] + _pick_diverse(...)`.
So the FAISS index ends up with `1 + min(4, n)` vectors per staff member. These must stay
in sync with `staff_registry.json` — one entry per index slot with `name`.

---

### Test Cases

#### FR-PD-01 — Selects Most Distant Embeddings From Average

| Field | Value |
|---|---|
| **Target** | `_pick_diverse` — `register_staff.py:232` |
| **Mocking** | None (pure numpy math) |
| **Setup** | `avg = [1, 0, ...]`; 5 unit vectors with cosine distances from avg: `0.02, 0.20, 0.85, 0.88, 0.91`; `max_extra=3` |
| **Expected** | Returns the 3 embeddings with distances `0.91, 0.88, 0.85` (most distant); `e1` (dist 0.02) and `e2` (dist 0.20) excluded |
| **Verdict** | PASS |
| **Why it matters** | Core selection logic — confirms the sort-descending-by-distance rule picks the embeddings most unlike the mean |

---

#### FR-PD-02 — Returns At Most `max_extra` Embeddings

| Field | Value |
|---|---|
| **Target** | `_pick_diverse` — `register_staff.py:232` |
| **Mocking** | None |
| **Setup** | 10 distinct unit vectors; `avg` = random unit vector; `max_extra=4` |
| **Expected** | `len(result) == 4` |
| **Verdict** | PASS |
| **Why it matters** | Caps how many diverse vectors are stored. Without this cap, staff with many images would bloat the FAISS index and slow down search |

---

#### FR-PD-03 — Empty Input Returns Empty List

| Field | Value |
|---|---|
| **Target** | `_pick_diverse` — `register_staff.py:232` |
| **Mocking** | None |
| **Setup** | `embeddings = []`; `avg` = any unit vector; `max_extra=4` |
| **Expected** | `[] ` returned; no exception raised |
| **Verdict** | PASS |
| **Why it matters** | Edge case — staff member with zero valid face embeddings. The caller must not crash. Downstream: `register_staff_from_ids` still stores just the mean embedding |

---

#### FR-PD-04 — Fewer Embeddings Than `max_extra` Returns All

| Field | Value |
|---|---|
| **Target** | `_pick_diverse` — `register_staff.py:232` |
| **Mocking** | None |
| **Setup** | `embeddings` = 2 distinct unit vectors; `avg` = random unit vector; `max_extra=4` |
| **Expected** | `len(result) == 2` (both returned, no padding) |
| **Verdict** | PASS |
| **Why it matters** | Staff with very few photos must not cause an out-of-bounds error or silent truncation. Python's `list[:n]` slicing handles this naturally — this test verifies that assumption |

---

#### FR-PD-05 — `max_extra=0` Returns Empty

| Field | Value |
|---|---|
| **Target** | `_pick_diverse` — `register_staff.py:232` |
| **Mocking** | None |
| **Setup** | 5 unit vectors; `avg` = unit vector; `max_extra=0` |
| **Expected** | `[]` |
| **Verdict** | PASS |
| **Why it matters** | `dists[:0]` must return empty, not throw. Caller controls whether diverse embeddings are stored at all — `max_extra=0` is a valid config |

---

#### FR-PD-06 — Result Is Deterministic for Same Inputs

| Field | Value |
|---|---|
| **Target** | `_pick_diverse` — `register_staff.py:232` |
| **Mocking** | None |
| **Setup** | Fixed seeded 6-vector set (`seed=42`); `max_extra=4`; called twice with identical inputs |
| **Expected** | Both calls return identical lists in identical order |
| **Verdict** | PASS |
| **Why it matters** | FAISS index position `i` maps directly to `registry[i]`. If `_pick_diverse` were non-deterministic, re-running staff registration would produce a different index ordering than the existing registry — corrupting all matches. Determinism is a correctness requirement |

---

## Group FR-SM — Staff Classification (FAISS matching)

### What the logic does

This is the classification block inside `run_fr_processing()` at `pipeline.py:1213`.
After embeddings are extracted for a person, the pipeline decides: **staff**, **customer**, or **no_face**.

**Full decision flow:**

```
all_embs = [embeddings from extraction]

if all_embs is empty:
    → update_type = 'no_face'

else:
    best_sim = -1.0
    for each emb in all_embs:
        D, I = faiss_index.search(emb, k=1)     # returns (similarity, index)
        if D[0][0] > best_sim:
            best_sim = D[0][0]
            best_name = registry[I[0][0]]['name']

    if best_sim >= threshold (0.80):
        → update_type = 'staff',   recognized_name = best_name
    else:
        → update_type = 'customer', appended to customer_data pool
```

**FAISS inner product = cosine similarity** because all embeddings are L2-normalized.
A score of `1.0` is a perfect match; `0.80` is the staff recognition threshold.

**Multi-embedding strategy:** searching all embeddings and taking the max gives the best
chance of matching a staff member who may have been captured in varied poses.

---

### Test Cases

#### FR-SM-01 — Similarity Above Threshold → Staff

| Field | Value |
|---|---|
| **Target** | `run_fr_processing` classification — `pipeline.py:1213` |
| **Mocking** | `faiss_index.search()` returns `([[0.75]], [[0]])`; `staff_registry = [{'name': 'Ravi'}, {'name': 'Priya'}]`; `threshold=0.60` |
| **Setup** | Person has 1 embedding |
| **Expected** | `update_type == 'staff'`; `recognized_name == 'Ravi'`; `similarity_score == 0.75` |
| **Verdict** | PASS |
| **Note** | In production the threshold is `0.80`; the test uses `0.60` to exercise the branching logic independently of the threshold value |

---

#### FR-SM-02 — Similarity Below Threshold → Customer

| Field | Value |
|---|---|
| **Target** | `run_fr_processing` — `pipeline.py:1213` |
| **Mocking** | `faiss_index.search()` returns `([[0.45]], [[0]])`; `threshold=0.60` |
| **Setup** | Person has 1 embedding |
| **Expected** | `update_type == 'customer'`; person appended to `customer_data`; not in `staff_by_name` |
| **Verdict** | PASS |

---

#### FR-SM-03 — Similarity Exactly at Threshold → Staff (Boundary)

| Field | Value |
|---|---|
| **Target** | `run_fr_processing` — `pipeline.py:1213` |
| **Mocking** | `faiss_index.search()` returns `([[0.60]], [[1]])`; `registry = [Ravi, Priya]`; `threshold=0.60` |
| **Expected** | `update_type == 'staff'`; `recognized_name == 'Priya'` (index 1) |
| **Verdict** | PASS |
| **Why it matters** | The condition is `best_sim >= threshold` (inclusive). A person exactly on the boundary must be classified as staff, not customer. Off-by-one here would cause a staff member to appear as an unknown customer in the report |

---

#### FR-SM-04 — No Embeddings Extracted → `no_face`

| Field | Value |
|---|---|
| **Target** | `run_fr_processing` — `pipeline.py:1213` |
| **Mocking** | All extraction returns empty; `person_embeddings = []` |
| **Expected** | `update_type == 'no_face'`; not added to `customer_data`; `faiss_index.search` never called |
| **Verdict** | PASS |
| **Note** | Covered by `test_fr25_no_face_classification` in `TestNoFaceAndSchema`. No FAISS call is made — short-circuits before the search loop |

---

#### FR-SM-05 — Multiple Embeddings: Maximum Score Wins

| Field | Value |
|---|---|
| **Target** | `run_fr_processing` — `pipeline.py:1213` |
| **Mocking** | `faiss_index.search()` `side_effect` returns sims `0.42, 0.73, 0.55` on successive calls; `threshold=0.60` |
| **Setup** | Person has 3 embeddings |
| **Expected** | `update_type == 'staff'`; `similarity_score == 0.73` (max of the three); `search.call_count == 3` |
| **Verdict** | PASS |
| **Why it matters** | A person photographed at an unusual angle may have one very good match and two poor ones. Taking the max ensures the best available embedding drives the classification decision |

---

#### FR-SM-06 — Empty FAISS Index → All Customers

| Field | Value |
|---|---|
| **Target** | `run_fr_processing` — `pipeline.py:1213` |
| **Mocking** | `faiss_index.ntotal == 0`; valid embeddings available |
| **Expected** | `update_type == 'customer'`; `faiss_index.search` never called |
| **Verdict** | PASS |
| **Why it matters** | Fresh deployments before any staff are registered must not crash. The guard `if faiss_index and faiss_index.ntotal > 0` prevents a search on an empty index (which would return index `-1`) |

---

#### FR-SM-07 — Out-of-Range Registry Index → No Crash

| Field | Value |
|---|---|
| **Target** | `run_fr_processing` — `pipeline.py:1213` |
| **Mocking** | `registry = [Ravi, Priya]` (valid indices 0, 1); `faiss_index.search()` returns `([[0.85]], [[99]])` (index 99 out of range) |
| **Expected** | No `IndexError` or `KeyError`; person classified as `'customer'` (safe fallback because `best_name` stays `None`) |
| **Verdict** | PASS |
| **Why it matters** | FAISS index and registry can drift out of sync if the registry file is hand-edited or partially written. The guard `if 0 <= idx < len(staff_registry)` prevents a crash and gracefully falls back to customer rather than corrupting the report |

---

## Group FR-CC — Greedy Customer Clustering

### What the algorithm does

After all persons are classified as staff or customer, the customer pool goes through a
**greedy single-linkage clustering** loop (`pipeline.py:1243`).

The goal is to merge multiple track IDs that belong to the same physical person — someone
who re-entered the store, was lost by the tracker, and got a new `track_id`.

**Algorithm walkthrough:**

```python
clusters = []
assigned = set()

for i, (pid_i, emb_i, p_i) in enumerate(customer_data):
    if i in assigned:
        continue                          # already merged into an earlier cluster

    # Seed a new cluster with this person
    cluster = { track_ids: [pid_i], embeddings: [emb_i], ... }
    assigned.add(i)

    # Scan all remaining unassigned persons
    for j, (pid_j, emb_j, p_j) in enumerate(customer_data):
        if j in assigned:
            continue
        # Compare emb_j against EVERY embedding already in the cluster
        max_sim = max(dot(emb_j, ce) for ce in cluster['embeddings'])
        if max_sim >= cluster_threshold (0.60):
            cluster.track_ids.append(pid_j)
            cluster.embeddings.append(emb_j)    # grows the cluster
            assigned.add(j)

    clusters.append(cluster)

clusters.sort(key=lambda c: -c['visit_count'])  # most frequent visitor first
```

**Key properties:**

| Property | Detail |
|---|---|
| **Order-dependent** | The first unassigned person always seeds a new cluster. Changing `customer_data` order can change which persons end up merged |
| **Growing cluster** | Once a person joins, their embedding is added to the cluster set. New candidates are compared against all cluster embeddings, not just the seed |
| **Single-pass** | No iterative re-assignment. One linear scan through `customer_data` |
| **Threshold** | `cluster_threshold = 0.60` (cosine similarity; `>= 0.60` merges) |
| **`no_face` persons** | Never reach this loop — they were routed out before `customer_data` is populated |

---

### Test Cases

#### FR-CC-01 — Two Similar Customers Merged (sim ≈ 0.98)

| Field | Value |
|---|---|
| **Target** | Greedy clustering loop — `pipeline.py:1245` |
| **Mocking** | None (pure numpy math) |
| **Setup** | `customer_data = [(1, e1, p1), (2, e2, p2)]`; `np.dot(e1, e2) ≈ 0.98`; `cluster_threshold=0.55` |
| **Expected** | `len(clusters) == 1`; `track_ids == {1, 2}`; `visit_count == 2` |
| **Verdict** | PASS |
| **Interpretation** | Two appearances of the same person (same face embedding) → one customer cluster with two visits |

---

#### FR-CC-02 — Dissimilar Customers Form Separate Clusters

| Field | Value |
|---|---|
| **Target** | Greedy clustering loop — `pipeline.py:1245` |
| **Mocking** | None |
| **Setup** | `np.dot(e1, e2) ≈ 0.02` (near-orthogonal, completely different people); `cluster_threshold=0.55` |
| **Expected** | `len(clusters) == 2`; each `visit_count == 1` |
| **Verdict** | PASS |
| **Interpretation** | Two different people → two separate clusters, each with one visit |

---

#### FR-CC-03 — Single Customer Forms Exactly One Cluster

| Field | Value |
|---|---|
| **Target** | Greedy clustering loop — `pipeline.py:1245` |
| **Mocking** | None |
| **Setup** | `customer_data = [(1, unit_vector(seed=5), p1)]`; `cluster_threshold=0.55` |
| **Expected** | `len(clusters) == 1`; `visit_count == 1`; `track_ids == [1]` |
| **Verdict** | PASS |
| **Why it matters** | Degenerate single-element input must not crash and must produce exactly one cluster |

---

#### FR-CC-04 — Empty Customer Pool → Empty Cluster List

| Field | Value |
|---|---|
| **Target** | Greedy clustering loop — `pipeline.py:1245` |
| **Mocking** | None |
| **Setup** | `customer_data = []`; `cluster_threshold=0.55` |
| **Expected** | `clusters == []`; no exception |
| **Verdict** | PASS |
| **Why it matters** | A day where all persons are staff or no-face must produce an empty customer section in the report without crashing |

---

#### FR-CC-06 — Threshold Boundary: `sim == threshold` → Merged

| Field | Value |
|---|---|
| **Target** | Greedy clustering loop — `pipeline.py:1245` |
| **Mocking** | None |
| **Setup** | `e1 = [1, 0, ...]`; `e2 = [0.55, sqrt(1-0.55²), 0, ...]` so that `np.dot(e1, e2) == 0.55` exactly; `threshold=0.55` |
| **Expected** | `len(clusters) == 1` (the `>=` comparison is inclusive; exact boundary merges) |
| **Verdict** | PASS |
| **Why it matters** | `max_sim >= cluster_threshold` — the `=` case merges. A `>` only condition would silently split a customer who appears twice with an exactly-threshold embedding pair |

---

#### FR-CC-07 — `no_face` Persons Excluded From Clustering

| Field | Value |
|---|---|
| **Target** | Greedy clustering loop — `pipeline.py:1245` |
| **Mocking** | None |
| **Setup** | `person_1` valid → in `customer_data`; `person_2` no-face → routed to `no_face_list` before clustering; `customer_data = [(1, e1, p1)]` only |
| **Expected** | `len(clusters) == 1`; `person_2`'s `track_id` absent from all cluster `track_ids` |
| **Verdict** | PASS |
| **Why it matters** | Confirms that the no-face filter upstream correctly prevents no-face persons from polluting the clustering step. A no-face person has no meaningful embedding — they must never be merged with a real customer |

---

#### FR-CC-08 — All Identical Embeddings → Single Cluster

| Field | Value |
|---|---|
| **Target** | Greedy clustering loop — `pipeline.py:1245` |
| **Mocking** | None |
| **Setup** | `e = unit_vector(seed=0)` (single unit vector); `customer_data = [(i, e.copy(), {}) for i in range(5)]`; `threshold=0.55` |
| **Expected** | `len(clusters) == 1`; `visit_count == 5`; all 5 track IDs in the single cluster |
| **Verdict** | PASS |
| **Why it matters** | Extreme case — one person re-tracked 5 times. The greedy loop must chain all of them together: person 0 seeds cluster 1, persons 1–4 all score `dot(e, e) = 1.0 >= 0.55` and join. Tests the full merging chain rather than just a pair |

---

## Coverage Summary

| Group | Function | File | Cases | Verdict |
|---|---|---|---|---|
| FR-EX | `extract_embeddings` | `rerun_fr.py:86` | EX-01..05 | 5 PASS |
| FR-PD | `_pick_diverse` | `register_staff.py:232` | PD-01..06 | 6 PASS |
| FR-SM | `run_fr_processing` (classification block) | `pipeline.py:1213` | SM-01..07 | 7 PASS |
| FR-CC | Greedy clustering loop | `pipeline.py:1245` | CC-01..08 (no CC-05) | 7 PASS |
| **Total** | | | **25 cases** | **25 PASS** |

### What each group covers

| Group | Boundary tested | Failure mode guarded |
|---|---|---|
| FR-EX | Empty folder, non-JPEG files, body_snapshot double-count | No crash on zero images; correct fallback order |
| FR-PD | `max_extra` cap, empty pool, zero cap, determinism | Registry/index sync corruption from non-determinism |
| FR-SM | `>= threshold` inclusive, empty FAISS, out-of-range index | Staff shown as customer; crash on missing registry entry |
| FR-CC | `>= threshold` inclusive, empty pool, all-identical chain | Threshold off-by-one splits real re-visits; no-face in cluster |

### What is NOT covered by these groups

| Gap | Nearest existing coverage |
|---|---|
| Multi-staff same-name accumulation in `staff_by_name` | FR-SM-01 (single staff match) |
| Cluster `first_seen`/`last_seen` timestamps | FR-34 in `TestGreedyClustering` |
| `clusters.sort` by `visit_count` descending | FR-38 |
| `state["fr_results"]` schema write | FR-50 in `TestNoFaceAndSchema` |
| 4-strategy fallback chain inside `_extract_with_strategies` | FD-33..37, FR-ES-01..06 |

---

*Config values used by these tests (as of 2026-04-29):*
`similarity_threshold=0.80`, `customer_cluster_threshold=0.60`, `min_quality=25`,
`min_skin_ratio=0.20`, `min_keypoints=3`.
*rerun_fr thresholds differ:* `staff_threshold=0.60`, `cluster_threshold=0.55`.
