# FR_Thor — Offline Face Recognition Pipeline

## Project Overview

Offline FR pipeline for pharmacy footfall analysis. Pulls recorded CCTV clips from a remote Jetson, tracks persons across clips, captures faces, runs FR against a staff registry, clusters unknown visitors, and produces multi-sheet Excel reports. All inference is local — no cloud APIs. `pipeline.py` v1.7.0.

---

## Tech Stack

Pure-Python CV pipeline. One outbound SSH/SCP channel to the Jetson; everything else local.

- **YOLO (Ultralytics)** — `yolo11l.pt` person detection (`conf=0.45`); `yolov11s-face.pt` face detection (`conf=0.30`). Optional TensorRT (`.pt → .onnx → .engine`) when `use_tensorrt: true`.
- **OSNet** (via `boxmot`) — 512-d L2-normalized ReID embeddings for `PersonTracker`.
- **InsightFace** `antelopev2` (ArcFace `w600k_r50.onnx`) — 512-d embeddings + 5 landmarks. `FaceAnalysis(root="insightface_models/")`.
- **FAISS** `IndexFlatIP(512)` on L2-normalized vectors → cosine similarity. Thresholds 0.55–0.95 are similarities, not distances.
- **OpenCV**, **NumPy<2**, **Pillow**, **Matplotlib**, **openpyxl**, **onnxruntime-gpu** (swap to CPU `onnxruntime` if no CUDA).
- **Runtime** — Python 3, `ThreadPoolExecutor` for async face saves, background `FrameReader` per clip.
- **Persistence** — JSON only: `state.json` (per-date), `staff_registry.json` (FAISS-aligned), `config.json`.
- **Models on disk** — `models/` (YOLO/OSNet weights), `insightface_models/antelopev2/` (nested `antelopev2/antelopev2/` is an extraction artifact), `vector_db/faiss.index` + `staff_registry.json` (rebuild together).
- **Testing** — plain `python test_*.py` under `Unit_Test_Data/`, master runner `run_all_and_report.py` (run → save state JSON → render Excel). Markdown plans/reports under gitignored `Unit_Test_Test_Cases/`.
- **Tooling** — Conventional Commits; `main` (prod) ← `dev` (staging) ← `feat/*` / `fix/*` / `testing`. Custom Claude agents `git-flow`, `fr-debugger` under `.claude/agents/`.

---

## Repository Structure

```
FR_Thor/
├── pipeline.py          # Main pipeline — fetch, track, capture, FR, report
├── fr_report.py         # Standalone FR+report from faces/ folder (no state.json)
├── rerun_fr.py          # Re-run FR with improved embedding strategies
├── register_staff.py    # Builds FAISS index + registry
├── filter_faces.py      # Post-hoc quality/skin/keypoint filter
├── config.json          # All runtime config
├── requirements.txt     # 10 packages (no version pins; see Dependencies)
├── README.txt
├── models/              # YOLO + OSNet weights (.pt / .onnx / .engine)
├── insightface_models/  # antelopev2 ONNX pack
├── vector_db/           # faiss.index + staff_registry.json
├── data/YYYY-MM-DD/     # Per-date: videos/, faces/, tracks/, state.json, report/
├── documentation/       # Analysis docs
├── Unit_Test_Test_Cases/       # Test plans, extended cases, run reports (markdown only, gitignored)
│   ├── test_plan/              # Master + per-feature plans
│   ├── feature_test_cases/     # Extended/additive cases
│   ├── required_test_cases.md  # Pruned 143-of-276 set
│   └── Face Detection|Recognicition|Tracking/  # Per-feature run reports
├── Unit_Test_Data/             # Real-footage integration scripts + extracted data
│   ├── run_all_and_report.py   # Master runner → state JSON + Excel
│   ├── reports/                # test_state_<ts>.json + FR_Thor_VideoTest_Report_<ts>.xlsx
│   ├── Annotation/             # annotate_pipeline.py + output mp4s
│   ├── Face_Detection/scripts/ # test_fd01..fd44.py + utils.py + run_all.py
│   ├── Face_Tracking/scripts/  # test_ft01.py..test_ft61.py + _ft_helpers.py + annotate_tracking.py
│   └── Face_Recognition/scripts/ # test_fr_*.py + _helpers.py + 00_extract_video_data.py
├── scripts/             # Auto-generated (gitignored): convert_to_trt.py
└── .claude/             # Settings + agents (git-flow, fr-debugger)
```

Local data fixtures (gitignored `data/`):
- `data/2026-04-29/` — primary fixture for FR/FT tests (faces/, tracks/, **`videos/videos/`** nested, state.json, report/)
- `data/2026-04-21/videos/` — raw clips only
- `data/2026-05-03/` — additional pipeline output

Test scripts (`_helpers.py:16`, `_ft_helpers.py:26`) resolve videos from the nested `videos/videos/` path.

`Unit_Test_Test_Cases/` is gitignored — all plans/cases/reports are local-only; not on `origin`.

---

## Branch Flow & Commits

- `main` (prod) ← `dev` (staging) ← `feat/<name>` | `fix/<name>` | `testing`.
- Hotfixes: `fix/<name>` off `main` → PR into `main`, then merge `main` → `dev`.
- Conventional Commits: `feat`/`fix`/`chore`/`docs`/`refactor`/`perf`/`test`/`build`/`ci`. Imperative subject ≤72 chars, no period. One logical change per commit. Body explains *why*.

---

## Architecture & Pipeline Flow

Four phases: **Tracking** (YOLO+OSNet) → **FR** (3-pass filter → InsightFace → FAISS → cluster) → **Staff Registration** (avg+diverse embeddings) → **Reporting** (multi-sheet Excel). `state.json` is the sole persistent store, written after every clip.

```
Recording Jetson (SSH/SCP) → data/<date>/videos/*.mp4
  ClipProcessor (per clip, timestamp order)
  ├─ FrameReader (background thread)
  ├─ YOLO person detection (yolo11l.pt, conf=0.45)
  ├─ OSNetExtractor.extract_batch() → ReID embeddings
  ├─ PersonTracker.match_or_create()
  │    ├─ Phase 1: greedy match → active tracks (cosine dist < 0.65 + box gate)
  │    ├─ Phase 2: match → pending tracks; promote after WARMUP_FRAMES=10
  │    └─ Phase 3: new pending entries
  ├─ YOLO face detection (yolov11s-face.pt, conf=0.30) → score_face()
  ├─ Async face save → face_XXXXXX_qNN.jpg
  └─ state.json saved after every clip
  run_fr_processing()
  ├─ Filter 1: quality < 25 (filename parse)
  ├─ Filter 2: skin-tone HSV (center 60%, ≥20% skin)
  ├─ Filter 3: InsightFace keypoints (3+ of 5 inside bbox)
  ├─ Embed → FAISS.search() → sim ≥ 0.80 staff; < 0.80 customer pool
  └─ Greedy cluster customers (sim ≥ 0.60)
  generate_report() → data/<date>/report/fr_report_<date>.xlsx (6 sheets)
```

---

## File-Level Context & Key Functions

| Script | Purpose | Entry point |
|---|---|---|
| `pipeline.py` | Full pipeline: fetch → track → FR → report | `process_day():1710` |
| `fr_report.py` | Standalone FR from `faces/` (no state.json); 4-strategy embedding | `run_fr()` |
| `rerun_fr.py` | Re-run FR on existing state.json; `--retry-only` skips matched | `rerun_fr():226` |
| `register_staff.py` | Build/update FAISS staff index | `register_staff_from_ids():117` |
| `filter_faces.py` | Post-hoc face filter; rejects → `rejected/` | `filter_faces():159` |
| `config.json` | All runtime config — no hardcoded thresholds in source | — |

**Key functions:**

| Function | File:line | Responsibility |
|---|---|---|
| `fetch_videos()` | `pipeline.py:123` | SSH `ls` + SCP; skip already-processed |
| `score_face()` | `pipeline.py:593` | Quality 0–100+: blur+size+brightness+pose symmetry |
| `_check_skin_tone()` | `pipeline.py:636` | HSV skin ratio on center 60% crop |
| `_count_kps_in_bbox()` | `pipeline.py:660` | InsightFace landmarks inside bbox+margin |
| `run_fr_processing()` | `pipeline.py:1087` | Filter→embed→FAISS→cluster→write `fr_results` |
| `generate_report()` | `pipeline.py:1367` | 6-sheet Excel with thumbnails |
| `PersonTracker.match_or_create()` | `pipeline.py:425` | 3-phase greedy matching |
| `ActiveTrack.serialize()` | `pipeline.py:368` | Save bank `.npy` for cross-clip restore |
| `PersonTracker.restore_state()` | `pipeline.py:577` | Restore tracks at next-clip start |
| `_extract_with_strategies()` | `rerun_fr.py:125` | 4-strategy: direct → upscale → CLAHE → pad |
| `extract_embeddings()` | `rerun_fr.py:86` | Per-folder embedding with strategy stats |
| `remove_staff()` | `register_staff.py:316` | Rebuild FAISS without named staff |
| `check_keypoints()` | `filter_faces.py:46` | InsightFace bbox containment (no per-point conf) |

**Alternative run paths:** `fr_report.py` (FR from faces/, no state.json) · `rerun_fr.py` (re-FR existing state.json) · `register_staff.py` (build vector_db/) · `filter_faces.py` (move bad crops to rejected/).

---

## `state.json` Schema (key fields)

```json
{
  "date": "2026-04-09",
  "processed_clips": ["20260409_090000.mp4"],
  "next_track_id": 42,
  "active_tracks": {"5": {"bank_file": "track_0005_bank.npy", "last_box": [...], "lost_seconds": 0}},
  "persons": [{
    "track_id": 3,
    "first_seen": "2026-04-09T09:02:15", "last_seen": "2026-04-09T09:04:30",
    "images_folder": "data/2026-04-09/faces/person_0003",
    "face_images": [[72.5, "path/face_001200_q72.jpg"]],
    "face_images_count": 12,
    "best_face_image": "path/face_001200_q72.jpg",
    "body_snapshot": "path/body_snapshot.jpg",
    "fr_processed": true, "update_type": "staff",
    "recognized_name": "Ravi", "similarity_score": 0.7832
  }],
  "fr_results": {"staff": [...], "customer_clusters": [...], "no_face": [...], "no_face_count": 7}
}
```

---

## Configuration (`config.json`)

| Key | Default | Description |
|---|---|---|
| `store_name` | `"StoreName"` | Report title label |
| `recording_jetson.*` | — | SSH host/user/password/key + `video_base_path` |
| `recording_jetson.poll_interval_seconds` | `30` | Service mode poll |
| `recording_jetson.min_file_age_seconds` | `10` | Skip files being written |
| `video.expected_fps` / `processing_fps` | `25` / `10` | Skip = expected/processing |
| `tracking.reid_distance` | `0.65` | Max cosine dist for ReID match |
| `tracking.warmup_frames` | `10` | Pending → active threshold |
| `tracking.bank_size` | `10` | Embedding bank depth |
| `tracking.update_gate` | `0.8` | Update bank if dist < threshold × gate |
| `tracking.max_box_jump` | `0.3` | Max normalized centroid displacement |
| `tracking.max_lost_seconds` | `25` | Lost-track finalize timeout |
| `tracking.conf_threshold` | `0.45` | YOLO person conf floor |
| `tracking.min_crop_w/h` | `32/64` | Min person crop for ReID |
| `face_capture.max_faces_per_person` | `10` | Max stored faces |
| `fr.similarity_threshold` | `0.80` | FAISS cosine cutoff for staff |
| `fr.customer_cluster_threshold` | `0.60` | Cosine sim to group same customer |
| `use_tensorrt` | `true` | Compile YOLO to TensorRT on first run |

**Code-default constants** (not in config.json): `post_filter_keep_top_n=20`, `min_keypoints=3`, `min_kp_conf=0.5` (per-point), `min_skin_ratio=0.20`, `min_quality=25`.

---

## Important Design Decisions

- **FAISS IndexFlatIP + L2-normalized = cosine similarity.** Thresholds 0.55–0.95 are similarities, not distances.
- **Two-tier tracking (Pending → Active).** `WARMUP_FRAMES` buffer suppresses spurious IDs; very brief detections never get a track ID.
- **Cross-clip continuity via `.npy`.** `ActiveTrack.bank` saved to `tracks/track_XXXX_bank.npy` after each clip, restored at start of next.
- **Gap detection.** If filename-timestamp gap exceeds `MAX_LOST_SECONDS`, all active tracks finalize first to prevent stale ID re-use.
- **4-strategy embedding** (`rerun_fr.py`, `fr_report.py`): direct → upscale → CLAHE → padded border. Improves yield without altering disk images.
- **Staff registry: mean + diverse.** Stores mean plus up to 4 most-distant embeddings per staff. **JSON consequence:** one staff member produces up to 5 rows in `staff_registry.json` — one primary row with `embedding_count: N`, up to 4 tagged `is_extra_embedding: true`. `registry[i]` mirrors FAISS row `i`. `--show-registry` collapses these.
- **Three-filter pipeline in cost order.** Quality (filename, ~0) → skin-tone (HSV) → keypoints (InsightFace). Early-exit on failure.
- **`state.json` is the only store.** Crash-safe, written after every clip.
- **Filenames encode quality.** `face_XXXXXX_qNN.jpg` — `_qNN` is integer score (can exceed 100).

---

## Quick Reference

```bash
# Pipeline
python pipeline.py --date 2026-04-09 --once
python pipeline.py --date 2026-04-09 --local-dir ./data/2026-04-09/videos --once
python pipeline.py --date 2026-04-09 --once --no-fr   # tracking only

# Re-run FR (lower det_thresh=0.15, 4-strategy embedding)
python rerun_fr.py --date 2026-04-09
python rerun_fr.py --date 2026-04-09 --retry-only     # only no-face persons
python rerun_fr.py --date 2026-04-09 --dry-run

# Standalone FR report (no state.json needed)
python fr_report.py --date 2026-04-09

# Staff
python register_staff.py --date 2026-04-09 --list
python register_staff.py --date 2026-04-09 --name "Ravi" --ids 3 7
python register_staff.py --show-registry
python register_staff.py --remove "Ravi"

# Clean bad crops
python filter_faces.py --date 2026-04-09 --dry-run
python filter_faces.py --date 2026-04-09
```

---

## Dependencies & Models

`requirements.txt`: `numpy`, `ultralytics`, `boxmot`, `opencv-python`, `insightface`, `faiss-cpu`, `onnxruntime-gpu`, `openpyxl`, `Pillow`, `matplotlib`. No version pins — README.txt pins `numpy<2`, `faiss-cpu<1.8`. Swap `onnxruntime-gpu` → `onnxruntime` if no CUDA. `httpx` (in README only) needed for SSH-over-HTTP.

**Models:** all gitignored, machine-local, must exist before any script runs.
- **`det_thresh`:** `pipeline.py`/`register_staff.py` use `0.30`; `filter_faces.py`/`fr_report.py`/`rerun_fr.py` use `0.15` (higher recall).
- **TensorRT:** `use_tensorrt: true` exports `.pt → .onnx → .engine` on first run; subsequent runs load `.engine`. Manual: `python scripts/convert_to_trt.py`.
- **Replacing `w600k_r50.onnx`** (primary ArcFace) invalidates the staff registry — rebuild from scratch.

---

## Common Gotchas

1. **FAISS index ↔ registry must stay in sync.** `registry[i]` = index row `i`. Rebuild together.
2. **Two keypoint sources.** YOLO face: 5 kps with per-point conf. InsightFace: 5 landmarks validated by bbox containment, no per-point conf.
3. **`det_thresh` differs by script.** Pipeline/register `0.3`; filter/rerun/report `0.15`.
4. **Quality score is NOT a percentage.** `_qNN` is integer sum, can exceed 100. Reject floor 25; good ~60+.
5. **`body_snapshot.jpg` excluded from FR.** Only `face_*.jpg` processed. Snapshot is fallback in `rerun_fr.py`/`fr_report.py` only.
6. **Greedy clustering is order-dependent.** First sorted person seeds each cluster. Deterministic, not globally optimal.
7. **`active_tracks` empty after `finalize_all()`.** Holds only tracks alive at end-of-last-clip.
8. **Empty staff registry → all `CUSTOMER (-1.0000)`.** FAISS returns no neighbours; sentinel sim `-1.0000`. Fix: register staff, re-run FR.
9. **Re-registering an existing name merges, doesn't duplicate.** Combines `person_ids`, rebuilds avg+diverse, rewrites index.
10. **Registering staff does NOT retroactively reclassify past dates.** `state.json` stores results, not embeddings. Run `python rerun_fr.py --date <date>` to re-score.
11. **`rerun_fr.py` thresholds differ.** Staff `0.60` / cluster `0.55` (vs pipeline `0.80` / `0.60`). Higher recall, more borderline staff hits.

**Critical vs utility:** Critical = `PersonTracker`, `run_fr_processing()`, `generate_report()`, state.json schema, FAISS/registry alignment. Inactive = `LineCrossDetector` (wired to `None` in `ClipProcessor`).

---

## Working Constraints

**Source files are read-only.** Never edit `pipeline.py`, `fr_report.py`, `rerun_fr.py`, `register_staff.py`, `filter_faces.py`. Work is analysis and documentation only. If a task seems to need a source change, stop and confirm.

**Edits go to:** `documentation/`, `Unit_Test_Test_Cases/`, `Unit_Test_Data/`, `CLAUDE.md`.

**Do NOT touch:** `vector_db/*` (managed by `register_staff.py`) · `data/<date>/state.json`, `faces/`, `tracks/` (pipeline-managed) · model files (per-machine, gitignored) · `config.json` credentials (never commit `password`/`ssh_key`/`bot_token`).

---

## Unit Test Plans (`Unit_Test_Test_Cases/test_plan/`)

| File | Feature | Test IDs |
|---|---|---|
| `test_plan.md` | Master | overview/strategy/coverage |
| `face_detection_test_plan.md` | FD | TC-SF-01–18, TC-BCF-01–13, TC-DF-01–06, TC-EWS-01–08, TC-CQ-01–08, TC-CST-01–07, TC-CKP-01–08 |
| `face_tracking_test_plan.md` | FT | TC-OE-01–07, TC-AT-01–08, TC-PT-01–06, TC-TR-01–19 (40 cases) |
| `face_recognition_test_plan.md` | FR | FR-01..FR-50 |

**Extended cases** (`feature_test_cases/`, additive only): `face_detection_test_cases.md` (SF-EXT-01+), `face_tracking_test_cases.md` (FT-OE-08+), `face_recognition_feature_tests.md` (FR-51+).

**`required_test_cases.md`** — pruned 143-of-276 minimum-required set (FD 50, FT 46, FR 47). Each retained case has a justification; dropped cases note coverage by other tests.

**Config snapshot for plans (2026-04-29):** `reid_distance=0.65`, `warmup_frames=10`, `max_lost_seconds=25`, `conf_threshold=0.45`, `max_faces_per_person=10`, `similarity_threshold=0.80`, `customer_cluster_threshold=0.60`.

---

## Unit Testing Status

The mocked pytest layer (formerly `unit_testing/` and `test_cases/`) is **removed**. Test plans/cases/reports survive as markdown in `Unit_Test_Test_Cases/`. The active test layer is the real-footage suite under `Unit_Test_Data/`.

### Known Source Bugs (carried over from deleted pytest layer)

| ID | Verdict | Bug |
|---|---|---|
| `FR-EE-05` | XFAIL | `_detect_face` (`rerun_fr.py`) returns zero vector instead of `None` when norm is 0 — `if norm > 0` skips math but never returns. Fix: add `else: return None`. |
| `FR-75` | XFAIL | Same `_detect_face`: NaN embedding has `norm == NaN`, `NaN > 0` False, corrupted vector flows through. Same fix. |
| `EC-01` | XFAIL | Independent re-confirmation of FR-EE-05. |
| `FR-SM-08` | XFAIL | `classify_with_faiss` does `registry[idx]["name"]` — `KeyError` if entry missing `name`, kills FR loop. Fix: `.get("name")`. |
| `FR-90` | XFAIL | Empty persons list → `run_fr_processing` early-returns without writing `fr_results`; `generate_report()` crashes on `KeyError`. Fix: write empty `{staff:[], customer_clusters:[], no_face:[], no_face_count:0}` before early return. |
| `FR-73` | SKIP | Cannot construct three 512-d embeddings satisfying chain-linkage similarity. Fixture limit, not a code bug. |

Extended cases FD-045..077, FT-062..103, FR-51..90: all `State=Design, Verdict=Not Run` — code not yet written.

---

## Real-Footage Test Scripts (`Unit_Test_Data/`)

Active test layer. Real CCTV + real models (YOLO/OSNet/InsightFace). Requires `data/2026-04-29/` pipeline output.

| Directory | What it tests |
|---|---|
| `Face_Detection/scripts/` | Real YOLO crops → `score_face`, `check_skin_tone`, `check_keypoints`, `_batch_capture_faces`, `_detect_face` |
| `Face_Tracking/scripts/` | OSNet + `PersonTracker.match_or_create`; FT-01..07, FT-23..27 and the FT-41..61 family touch real video, others mock embeddings via `patch.object` |
| `Face_Recognition/scripts/` | Full `run_fr_processing` with real InsightFace; reads pipeline `faces/person_XXXX/` folders |

**FD pattern (`utils.py`):** `scan_faces(model, n_vids, step, validate=True)` scans first N clips every S frames. With `validate=True`, each crop runs `is_human_face()` (HSV skin ≥0.08 in center 60% AND non-empty `InsightFace.get()`) — filters bike/wall false positives before `best_face()`. `best_face()` picks max `blur_var × size × brightness`. Each `test_fdXX.py` has `extract()` + `test()`. **Cache check is content-aware**, not just `os.path.exists()` — tests requiring specific properties (FD-05 conf, FD-07 keypoints.json, FD-11 mean_gray) re-extract on property failure.

**FT pattern:** one script per test ID (`test_ft01.py`..`test_ft61.py`), each writing `Face_Tracking/data/test_ftNN/results.json` with only its own outcome. Shared YOLO/OSNet/centroid helpers live in `_ft_helpers.py` (paths, `load_yolo`, `load_osnet`, `collect_embeddings`, `collect_person_detections`, `extract_person_centroids`, `make_active_track`, `make_tracker`, `make_emb_at_dist`, `prepare_data_dir`, `save_single_test_result`). Real-footage tests read video → YOLO → OSNet; the synthetic-embedding family (FT-28..40) mocks `extract_batch` via `patch.object`.

**FR pattern:** `00_extract_video_data.py` first — copies face crops `data/2026-04-29/faces/person_XXXX/` → `Face_Recognition/data/FR_XX_YY/`. `_helpers.py` exports `load_face_app`, `build_faiss_index`, `classify_with_faiss` (replica of `pipeline.py:1213`), `run_fr_with_real_index` (calls real `run_fr_processing`, backs up vector_db/), `greedy_cluster` (replica of pipeline loop).

### Known Issues (resolved)

- **FT hardcoded video filenames** → `_ft_helpers.py` resolves `VIDEO_PATH`/`VIDEO_PATH2` via `sorted(VIDEO_DIR.glob("*.mp4"))` so no per-script filename is hardcoded.
- **FD bike/wall false positives** → `scan_faces` defaults `validate=True` with `is_human_face` two-stage filter. Pass `validate=False` only when raw YOLO is needed (e.g. profile-face hunts where InsightFace's frontal bias rejects valid crops).
- **FD stale extract data (2026-05-04 hardening):**
  - `test_fd05.py:24-38` re-extract triggers when `meta.json[conf] < 0.5` (the score_face low-yolo gate).
  - `test_fd07.py:32-100` — `yolov11s-face.pt` is non-pose (`r[0].keypoints is None`), so the test now runs InsightFace on YOLO crops, computes asymmetry from 5 landmarks, picks the most asymmetric, then maps kps to `score_face`'s `[nose, leye, reye, lear, rear]` layout (synthetic `conf=1.0` for 3 visible, `0.0` for absent ear pair).
  - `test_fd11.py:21-58` overexposure synthesis switched from `+160` HSV V-bump to a 0.2/0.8 white blend with post-load re-blend if `mean_gray ≤ 217` (satisfies score_face: `b > 217`).
- **FR-EX-04 permanent SKIP** → `prep_fr_ex_04` (`00_extract_video_data.py:166-184`) now requires both face crops AND `body_snapshot.jpg` before short-circuiting; empty/partial folders re-extract.

**Not an issue — FR scripts** read pre-processed pipeline output, not raw video. `_helpers.py` matches `pipeline.py:1213` exactly.

### Master Runner: `run_all_and_report.py`

Mirrors pipeline pattern: run → `save_state()` → `generate_report(state)`.

```bash
# From FR_Thor/ root — PYTHONUTF8=1 required on Windows (Unicode box chars)
PYTHONUTF8=1 python Unit_Test_Data/run_all_and_report.py            # full run
PYTHONUTF8=1 python Unit_Test_Data/run_all_and_report.py --fd-only  # | --fr-only | --ft-only
PYTHONUTF8=1 python Unit_Test_Data/run_all_and_report.py --skip-run # rebuild Excel from FT results.json
PYTHONUTF8=1 python Unit_Test_Data/run_all_and_report.py --from-json reports/test_state_<ts>.json
PYTHONUTF8=1 python Unit_Test_Data/run_all_and_report.py --timeout 600  # per-script timeout (s); default 300
```

**Output (`Unit_Test_Data/reports/`):**
- `test_state_<YYYYMMDD_HHMMSS>.json` — run metadata + per-feature counts + per-test verdicts/actual/notes.
- `FR_Thor_VideoTest_Report_<ts>.xlsx` — 4 sheets: Summary, FD (44), FR (25), FT (61). Columns: Test ID | Target Function | Description | Expected | Actual | Verdict | Notes. Auto-fallbacks to `_v2.xlsx` if locked open.

`--from-json <path>` re-renders Excel from any saved state JSON without re-running scripts. Bare filenames resolve against `reports/`. Replay preserves original `run_ts`.

**Verdicts:** PASS / FAIL / SKIP (model unavailable) / ERROR (crash or timeout) / NOT RUN.

**FT persistence:** Each FT script writes `Face_Tracking/data/<group>/results.json`. `--skip-run` reads those without re-running.

**Timing (RTX 3050):** FD ~5 min · FR ~1 min · FT ~30–90 min (14 scripts, each loads YOLO+OSNet+video).

### Run Status (snapshot 2026-05-04 12:54)

| Feature | Total | PASS | FAIL | SKIP | ERROR |
|---|---|---|---|---|---|
| Face Detection | 44 | 44 | 0 | 0 | 0 |
| Face Recognition | 25 | 25 | 0 | 0 | 0 |
| Face Tracking | 61 | 61 | 0 | 0 | 0 |

All three suites green. State files: `test_state_20260504_125141.json` (FD), `test_state_20260504_125145.json` (FR).

### Individual Script Usage

```bash
python Unit_Test_Data/Face_Detection/scripts/test_fd01.py
python Unit_Test_Data/Face_Detection/scripts/run_all.py        # all 44 sequentially, no Excel
python Unit_Test_Data/Face_Tracking/scripts/test_ft01.py
python Unit_Test_Data/Face_Tracking/scripts/test_ft27.py
python Unit_Test_Data/Face_Recognition/scripts/00_extract_video_data.py  # FR: extract first
python Unit_Test_Data/Face_Recognition/scripts/run_all_tests.py
python Unit_Test_Data/Face_Recognition/scripts/test_fr_sm_01.py
```

Requires all model weights + `data/2026-04-29/` pipeline output. GPU optional; CPU works.

---

## Claude Code Agents

- `.claude/agents/git-flow` — multi-commit splits, Conventional Commit messages
- `.claude/agents/fr-debugger` — diagnoses missed faces, wrong staff matches, report issues
- `.claude/settings.json` — shared; personal overrides in `.claude/settings.local.json` (gitignored)
