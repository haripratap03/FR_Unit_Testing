#!/usr/bin/env python3
"""
run_all_and_report.py — FR_Thor Video-Based Test Runner & Report Generator
===========================================================================
Executes all real-footage test scripts for Face Detection (FD-01..44),
Face Recognition (FR-EX/PD/SM/CC — 25 tests), and Face Tracking (FT-01..61),
then writes — into  Unit_Test_Data/reports/ :
  test_state_<YYYYMMDD_HHMMSS>.json     <- pipeline-style intermediate state
  FR_Thor_VideoTest_Report_<ts>.xlsx    <- rendered FROM that JSON

Mirrors the pipeline.py pattern:  save_state(state) → generate_report(state).
The Excel is always built from the JSON, so any state file can be replayed
later with --from-json without re-running the scripts.

Unlike the (now-removed) mocked pytest layer formerly under unit_testing/,
these scripts use real CCTV video, actual YOLO / InsightFace / OSNet models,
and real face embeddings — no NumPy preset arrays.

Prerequisites:
  • ML models in  models/  and  insightface_models/
  • CCTV footage at  data/2026-04-29/videos/videos/
  • FR data prepared (run once):
      python Unit_Test_Data/Face_Recognition/scripts/00_extract_video_data.py

Usage (run from FR_Thor/ root):
  python Unit_Test_Data/run_all_and_report.py               # full run → JSON + Excel
  python Unit_Test_Data/run_all_and_report.py --skip-run    # rebuild from FT results.json
  python Unit_Test_Data/run_all_and_report.py --from-json reports/test_state_<ts>.json
                                                       # render Excel from saved JSON
  python Unit_Test_Data/run_all_and_report.py --fd-only
  python Unit_Test_Data/run_all_and_report.py --fr-only
  python Unit_Test_Data/run_all_and_report.py --ft-only
  python Unit_Test_Data/run_all_and_report.py --timeout 300 # per-script timeout (s)

Timing estimates (GPU machine):
  FD  44 scripts   ~  5-15 min  (seconds each if data already extracted)
  FR  25 tests     ~  3-10 min  (InsightFace on real crops)
  FT  14 scripts   ~ 30-90 min  (YOLO+OSNet on video per script)
  Use --skip-run to rebuild Excel from prior results without re-running.
"""

import argparse
import json
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# Force UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import openpyxl
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

# ─── Paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent           # Unit_Test_Data/
REPO_ROOT  = SCRIPT_DIR.parent                         # FR_Thor/
FD_SCRIPTS = SCRIPT_DIR / "Face_Detection"  / "scripts"
FR_SCRIPTS = SCRIPT_DIR / "Face_Recognition" / "scripts"
FT_SCRIPTS = SCRIPT_DIR / "Face_Tracking"   / "scripts"
FT_DATA    = SCRIPT_DIR / "Face_Tracking"   / "data"
REPORTS_DIR = SCRIPT_DIR / "reports"                   # JSON + Excel land here
PYTHON     = sys.executable
RUN_DATE   = datetime.now().strftime("%Y-%m-%d")
RUN_TS     = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
RUN_TAG    = datetime.now().strftime("%Y%m%d_%H%M%S")  # filename-safe stamp
STATE_JSON = REPORTS_DIR / f"test_state_{RUN_TAG}.json"
REPORT_OUT = REPORTS_DIR / f"FR_Thor_VideoTest_Report_{RUN_TAG}.xlsx"
STATE_VERSION = 1

# ─── Colour palette ──────────────────────────────────────────────────────────
VERDICT_BG = {
    "PASS":    "FF92D050",
    "FAIL":    "FFFF4040",
    "SKIP":    "FFD9D9D9",
    "ERROR":   "FFFFC000",
    "NOT RUN": "FFFFFFFF",
}
FD_COLOR = "FF2F5496"   # blue
FR_COLOR = "FF375623"   # green
FT_COLOR = "FF7B2C2C"   # dark red
ALT_A    = "FFD9E1F2"
ALT_B    = "FFFFFFFF"


# ─── Style helpers ───────────────────────────────────────────────────────────
def _side():
    return Side(border_style="thin", color="AAAAAA")

def _border():
    s = _side()
    return Border(left=s, right=s, top=s, bottom=s)

def _style(cell, *, bold=False, fc="000000", bg=None, sz=10,
           wrap=True, ha="left", va="top"):
    cell.font      = Font(name="Calibri", size=sz, bold=bold, color=fc)
    cell.alignment = Alignment(wrap_text=wrap, vertical=va, horizontal=ha)
    if bg:
        cell.fill  = PatternFill(fill_type="solid", fgColor=bg)
    cell.border    = _border()

def _hdr(cell, text, *, sz=10, bg="FF1F3864", fc="FFFFFFFF"):
    cell.value = text
    _style(cell, bold=True, sz=sz, bg=bg, fc=fc, ha="center", va="center", wrap=False)

def _verdict_cell(cell, verdict):
    v  = verdict or "NOT RUN"
    bg = VERDICT_BG.get(v, "FFFFFFFF")
    fc = "FFFFFFFF" if v == "FAIL" else "FF000000"
    cell.value = v
    _style(cell, bold=True, bg=bg, fc=fc, ha="center", va="center", wrap=False)

def _title_row(ws, text, ncols, row, *, bg="FF2F5496", sz=13):
    ws.merge_cells(start_row=row, start_column=1, end_row=row, end_column=ncols)
    c = ws.cell(row=row, column=1, value=text)
    _style(c, bold=True, sz=sz, bg=bg, fc="FFFFFFFF",
           ha="center", va="center", wrap=False)
    ws.row_dimensions[row].height = 30

def _meta_row(ws, key, value, row, ncols):
    kc = ws.cell(row=row, column=1, value=key)
    _style(kc, bold=True, bg=ALT_A, va="center", wrap=False)
    ws.merge_cells(start_row=row, start_column=2, end_row=row, end_column=ncols)
    vc = ws.cell(row=row, column=2, value=value)
    _style(vc, va="center")
    ws.row_dimensions[row].height = 16


# ─── Test metadata: (target_function, description, expected_result) ───────────

FD_META = {
    # score_face — FD-01..11
    "FD-01": ("score_face",              "Low-confidence face detection should be rejected outright",                  "Score is 0 (rejected as low-confidence)"),
    "FD-02": ("score_face",              "Face just barely passing the confidence threshold should still be scored",   "Score is greater than 0"),
    "FD-03": ("score_face",              "Tiny face crop (35x35) should be rejected as too small",                     "Score is 0 (rejected as too small)"),
    "FD-04": ("score_face",              "Face crop right at the size limit (40x40) should be scored",                 "Score is greater than 0"),
    "FD-05": ("score_face",              "A clean frontal face image should score high",                               "Score is 70 or higher"),
    "FD-06": ("score_face",              "A blurry face crop should get a low blur sub-score",                         "Blur sub-score is below 1.0"),
    "FD-07": ("score_face",              "A side-profile face should get a low pose sub-score",                        "Pose sub-score is below 5"),
    "FD-08": ("score_face",              "Both ears visible should reduce the pose sub-score",                         "Pose sub-score is 16 or lower"),
    "FD-09": ("score_face",              "When all 5 face keypoints have very low confidence, fall back to a fixed pose score", "Pose sub-score equals 6.0"),
    "FD-10": ("score_face",              "When no face keypoints are detected at all, use a neutral pose score",       "Pose sub-score equals 10"),
    "FD-11": ("score_face",              "An overexposed (all-bright) image should get a low brightness sub-score",    "Brightness sub-score is below 5"),
    # check_quality — FD-12..15
    "FD-12": ("check_quality",           "Quality score 42 (above the 25 floor) should pass",                          "Pass; score 42; no rejection reason"),
    "FD-13": ("check_quality",           "Quality score 18 (below the 25 floor) should fail",                          "Fail; score 18; reason mentions 'below_25'"),
    "FD-14": ("check_quality",           "Filename without a quality tag passes automatically",                        "Pass; score reported as -1 (no tag)"),
    "FD-15": ("check_quality",           "Quality score exactly at the 25 threshold passes (strict less-than gate)",   "Pass; score 25"),
    # check_skin_tone — FD-16..18
    "FD-16": ("check_skin_tone",         "Skin-coloured pixels should pass the skin-tone check",                       "Skin check passes"),
    "FD-17": ("check_skin_tone",         "Pure blue image (no skin pixels) should fail",                               "Skin check fails"),
    "FD-18": ("check_skin_tone",         "Skin only at the corners (not centre) should fail (centre is checked)",      "Skin check fails"),
    # check_keypoints — FD-19..21
    "FD-19": ("check_keypoints",         "All 5 face keypoints inside the bounding box should pass",                   "Pass; 5 keypoints inside"),
    "FD-20": ("check_keypoints",         "Only 2 keypoints inside (need at least 3) should fail",                      "Fail; 2 keypoints inside"),
    "FD-21": ("check_keypoints",         "InsightFace finds no face -> keypoint check fails",                          "Fail; no face detected"),
    # _check_skin_tone (pipeline) — FD-22..23
    "FD-22": ("_check_skin_tone",        "Skin-coloured pixels pass the pipeline's skin check",                        "Returns True"),
    "FD-23": ("_check_skin_tone",        "Pure blue pixels fail the pipeline's skin check",                            "Returns False"),
    # _count_kps_in_bbox — FD-24..27
    "FD-24": ("_count_kps_in_bbox",      "All 5 keypoints clearly inside the box should be counted",                   "Count is 5"),
    "FD-25": ("_count_kps_in_bbox",      "All keypoints far outside the box should not be counted",                    "Count is 0"),
    "FD-26": ("_count_kps_in_bbox",      "A keypoint exactly on the margin boundary still counts",                     "Count is 1"),
    "FD-27": ("_count_kps_in_bbox",      "First 3 keypoints inside, last 2 outside should give a count of 3",          "Count is 3"),
    # _enhance_contrast — FD-28..29
    "FD-28": ("_enhance_contrast",       "A dark image becomes brighter after contrast enhancement",                   "Output is brighter than input"),
    "FD-29": ("_enhance_contrast",       "Contrast enhancement does not modify the original image in place",           "Original image unchanged"),
    # _detect_face — FD-30..32
    "FD-30": ("_detect_face",            "When two faces are present, the larger one is picked and the embedding is unit-length", "Larger face selected; embedding length about 1.0"),
    "FD-31": ("_detect_face",            "A raw embedding gets normalized to unit length",                             "Embedding length about 1.0"),
    "FD-32": ("_detect_face",            "If no face is detected, return None (not a zero vector)",                    "Returns None"),
    # _extract_with_strategies — FD-33..37
    "FD-33": ("_extract_with_strategies","Normal 200x200 image: 'direct' strategy works on the first try",             "Strategy used: direct"),
    "FD-34": ("_extract_with_strategies","Small 60x80 image: triggers the 'upscaled' strategy",                        "Strategy used: upscaled"),
    "FD-35": ("_extract_with_strategies","When direct fails on a 250x250 image, falls back to CLAHE-enhanced",         "Strategy used: enhanced"),
    "FD-36": ("_extract_with_strategies","When both direct and CLAHE fail, falls back to padded",                      "Strategy used: padded"),
    "FD-37": ("_extract_with_strategies","When all 4 strategies fail, return (None, None) without crashing",           "Returns (None, None)"),
    # _batch_capture_faces — FD-38..44
    "FD-38": ("_batch_capture_faces",    "Valid 100x100 face with high confidence and cooldown elapsed should be captured", "Face submitted for saving"),
    "FD-39": ("_batch_capture_faces",    "If last capture was 0.5s ago and cooldown is 1.5s, skip capture",            "Capture skipped"),
    "FD-40": ("_batch_capture_faces",    "Detection with low confidence (0.30) gets a score of 0 and is discarded",    "Not submitted for saving"),
    "FD-41": ("_batch_capture_faces",    "If YOLO returns no boxes, no capture is attempted",                          "Not submitted for saving"),
    "FD-42": ("_batch_capture_faces",    "Already at max faces; new face quality (45) lower than worst (50) -> not saved", "New face not saved"),
    "FD-43": ("_batch_capture_faces",    "At max faces; new face quality (65) higher than worst (30) -> worst evicted, new saved", "Quality-30 face evicted; quality-65 saved"),
    "FD-44": ("_batch_capture_faces",    "Two detections (confidence 0.60 and 0.90) -> highest-confidence one wins",   "Detection at index 1 (conf 0.90) saved"),
}

FR_META = {
    # extract_embeddings — FR-EX
    "FR-EX-01": ("extract_embeddings", "Folder with multiple face images returns at least one unit-length embedding", "At least 1 embedding; length about 1.0"),
    "FR-EX-02": ("extract_embeddings", "Empty folder returns no embeddings",                                          "Zero embeddings"),
    "FR-EX-03": ("extract_embeddings", "body_snapshot.jpg in folder is excluded from the face scan",                  "body_snapshot.jpg skipped"),
    "FR-EX-04": ("extract_embeddings", "When face images are tiny, fall back to body_snapshot",                       "body_snapshot embedding used"),
    "FR-EX-05": ("extract_embeddings", "Non-image junk files in folder are ignored (only .jpg is processed)",         "Junk files ignored"),
    # pick_diverse — FR-PD
    "FR-PD-01": ("pick_diverse",       "Diversity selection picks the most spread-out embeddings",                    "Diverse subset selected"),
    "FR-PD-02": ("pick_diverse",       "Result length never exceeds the requested cap",                               "Length is at most max_extra"),
    "FR-PD-03": ("pick_diverse",       "Pure-math test: cosine distances are computed correctly",                     "Distances correct"),
    "FR-PD-04": ("pick_diverse",       "If fewer embeddings exist than the cap, return all of them",                  "Length equals what is available"),
    "FR-PD-05": ("pick_diverse",       "Cap of 0 returns an empty list",                                              "Empty list"),
    "FR-PD-06": ("pick_diverse",       "Same input always produces the same output",                                  "Output is deterministic"),
    # classify_with_faiss — FR-SM
    "FR-SM-01": ("classify_with_faiss","Face similarity at or above 0.80 is classified as staff",                     "Classified as staff"),
    "FR-SM-02": ("classify_with_faiss","Different person with similarity below 0.80 is classified as customer",       "Classified as customer"),
    "FR-SM-03": ("classify_with_faiss","Boundary case: similarity exactly at the threshold",                          "Boundary handled correctly"),
    "FR-SM-04": ("classify_with_faiss","Person with zero face images is marked as no_face",                           "Classified as no_face"),
    "FR-SM-05": ("classify_with_faiss","Person with several embeddings: highest similarity wins",                     "Highest similarity is used"),
    "FR-SM-06": ("classify_with_faiss","Empty staff index -> person classified as customer",                          "Classified as customer"),
    "FR-SM-07": ("classify_with_faiss","Out-of-range registry index does not crash",                                  "No exception raised"),
    # greedy_cluster — FR-CC
    "FR-CC-01": ("greedy_cluster",     "Same person split across two tracks (similar enough) merges into 1 cluster",  "1 cluster"),
    "FR-CC-02": ("greedy_cluster",     "Two clearly different people stay as 2 separate clusters",                    "2 clusters"),
    "FR-CC-03": ("greedy_cluster",     "Single customer produces 1 cluster",                                          "1 cluster"),
    "FR-CC-04": ("greedy_cluster",     "No customers at all -> empty cluster list",                                   "Empty list"),
    "FR-CC-06": ("greedy_cluster",     "Boundary similarity exactly at the cluster threshold merges",                 "Merged into the same cluster"),
    "FR-CC-07": ("greedy_cluster",     "no_face person is not included in any cluster",                               "no_face person excluded"),
    "FR-CC-08": ("greedy_cluster",     "Five identical embeddings collapse into 1 cluster",                           "1 cluster"),
}

FT_META = {
    # OSNetExtractor.extract_batch — FT-01..05
    "FT-01": ("OSNetExtractor.extract_batch", "Output list length matches box count; each embedding is 512-dimensional",      "Length matches; each shape is (512,)"),
    "FT-02": ("OSNetExtractor.extract_batch", "Crop too small (25x55) returns None at that position; valid crop returns embedding", "Index 0 is None; index 1 has shape (512,)"),
    "FT-03": ("OSNetExtractor.extract_batch", "All crops too small -> every result is None; OSNet model is not even called", "Every result is None"),
    "FT-04": ("OSNetExtractor.extract_batch", "All real embeddings have unit length",                                          "Embedding length about 1.0 for all"),
    "FT-05": ("OSNetExtractor.extract_batch", "If batch call crashes, falls back to processing crops one at a time",           "Fallback path produces valid results"),
    # OSNetExtractor.extract — FT-06..07
    "FT-06": ("OSNetExtractor.extract",       "Single crop below the 32x64 size threshold returns None",                      "Returns None"),
    "FT-07": ("OSNetExtractor.extract",       "Out-of-bounds box is clipped or handled without error",                         "Returns None or a valid 512-d embedding"),
    # ActiveTrack.ref — FT-08..10
    "FT-08": ("ActiveTrack.ref",              "Bank with a single embedding: reference equals that embedding",                "Reference equals the embedding"),
    "FT-09": ("ActiveTrack.ref",              "Recency weighting: more recent embeddings dominate the reference",             "Recent embeddings dominate"),
    "FT-10": ("ActiveTrack.ref",              "Full bank: reference is unit-normalized",                                      "Reference length about 1.0"),
    # ActiveTrack.update_embedding — FT-11..13
    "FT-11": ("ActiveTrack.update_embedding", "Adding a new embedding grows the bank",                                        "Bank size increases"),
    "FT-12": ("ActiveTrack.update_embedding", "Bank is capped at max size; oldest embedding is evicted",                      "Bank size stays at the cap"),
    "FT-13": ("ActiveTrack.update_embedding", "A dissimilar embedding (above the gate distance) is not added",                "Embedding rejected; bank unchanged"),
    # ActiveTrack.serialize / deserialize — FT-14..17
    "FT-14": ("ActiveTrack.serialize",        "Bank is saved to a .npy file; state dict contains a 'bank_file' key",          "'bank_file' key present in state"),
    "FT-15": ("ActiveTrack.deserialize",      "Bank is restored from .npy; reference is correct after load",                  "Reference correct after restore"),
    "FT-16": ("ActiveTrack.deserialize",      "Missing .npy during restore is handled gracefully",                            "No crash"),
    "FT-17": ("ActiveTrack.serialize",        "Round-trip serialize -> deserialize preserves all values",                     "All values identical"),
    # PendingTrack — FT-18..22
    "FT-18": ("PendingTrack.__init__",        "Initialized with first box and embedding; frame count is 1",                   "Frame count equals 1"),
    "FT-19": ("PendingTrack.ref",             "Reference returns the simple (unweighted) mean of accumulated embeddings",     "Reference equals the mean"),
    "FT-20": ("PendingTrack.add",             "add() accumulates embeddings and increments the frame count",                  "Bank size grows"),
    "FT-21": ("PendingTrack -> ActiveTrack",  "Pending is promoted to active once frame count reaches the warmup threshold",  "Promotion happens"),
    "FT-22": ("PendingTrack",                 "Not promoted before the warmup threshold is reached",                          "No promotion"),
    # PersonTracker._box_distance — FT-23..26
    "FT-23": ("PersonTracker._box_distance",  "Identical boxes have distance 0",                                              "Distance equals 0.0"),
    "FT-24": ("PersonTracker._box_distance",  "Far-apart boxes (cross-frame teleport) have distance above 0.3",               "Distance greater than 0.3"),
    "FT-25": ("PersonTracker._box_distance",  "Distance is normalized by frame width/height so it is scale-invariant",        "Scale-invariant"),
    "FT-26": ("PersonTracker._box_distance",  "Centroid distance formula matches a manual calculation",                       "Distance correct"),
    # PersonTracker.match_or_create — FT-27..40
    "FT-27": ("PersonTracker.match_or_create","Same person across consecutive frames keeps the same track ID",                "Same track ID"),
    "FT-28": ("PersonTracker.match_or_create","Embedding too far from any track -> new pending track created",                "New pending track"),
    "FT-29": ("PersonTracker.match_or_create","Spatial gate blocks cross-frame teleport even when ReID is similar",           "Match blocked by spatial gate"),
    "FT-30": ("PersonTracker.match_or_create","Larger gap between frames widens the spatial gate (allows wider matches)",     "Wider spatial gate"),
    "FT-31": ("PersonTracker.match_or_create","Close match (well below the gate) updates the embedding bank",                 "Bank updated"),
    "FT-32": ("PersonTracker.match_or_create","Borderline match (in the gate band) does not update the bank",                 "Bank unchanged"),
    "FT-33": ("PersonTracker.match_or_create","Greedy cost-matrix picks the globally optimal pairing",                        "Optimal assignment"),
    "FT-34": ("PersonTracker.match_or_create","Pending track frame count increments each frame before promotion",             "Frame count increases"),
    "FT-35": ("PersonTracker.match_or_create","Pending is promoted to active when frame count reaches the warmup threshold",  "Promoted"),
    "FT-36": ("PersonTracker.match_or_create","After promotion, active track's bank is seeded from pending history",          "Bank seeded from pending"),
    "FT-37": ("PersonTracker.match_or_create","Unmatched detection creates a brand-new pending track",                        "New pending track"),
    "FT-38": ("PersonTracker.match_or_create","None embedding (undersized crop) is silently skipped without crashing",        "Skipped without error"),
    "FT-39": ("PersonTracker.match_or_create","Empty boxes list -> empty result and active tracks unchanged",                 "Empty result"),
    "FT-40": ("PersonTracker.match_or_create","More active tracks than detections -> only some are matched",                  "Partial match"),
    # PersonTracker.get_lost — FT-41..45
    "FT-41": ("PersonTracker.get_lost",       "Stale track (100s, past 90s max) returned; fresh (30s) is not",                "Only the stale track returned"),
    "FT-42": ("PersonTracker.get_lost",       "All tracks fresh -> empty list",                                               "Empty list"),
    "FT-43": ("PersonTracker.get_lost",       "No active tracks -> empty list, no exception",                                 "Empty list"),
    "FT-44": ("PersonTracker.get_lost",       "ISO timestamps (from state.json) are parsed correctly",                        "Time delta correct"),
    "FT-45": ("PersonTracker.get_lost",       "Track exactly at the threshold is NOT considered lost (strict greater-than)",  "Not lost at the boundary"),
    # PersonTracker.serialize/restore_state — FT-46..48
    "FT-46": ("PersonTracker.serialize_state","Active tracks serialized to state dict with their bank_file keys",             "Keys present in state"),
    "FT-47": ("PersonTracker.restore_state",  "Active tracks are re-created correctly from the state dict",                   "Tracks restored"),
    "FT-48": ("PersonTracker.restore_state",  "Missing .npy file during restore is handled gracefully",                       "No crash"),
    # PersonTracker.remove — FT-49..50
    "FT-49": ("PersonTracker.remove",         "Track is removed from the active dict by ID",                                  "ID removed"),
    "FT-50": ("PersonTracker.remove",         "Removing a non-existent ID raises no error",                                   "No exception raised"),
    # LineCrossDetector.get_side — FT-51..54
    "FT-51": ("LineCrossDetector.get_side",   "Point above the line -> side is +1",                                           "Side is +1"),
    "FT-52": ("LineCrossDetector.get_side",   "Point below the line -> side is -1",                                           "Side is -1"),
    "FT-53": ("LineCrossDetector.get_side",   "Point exactly on the line -> side is 0",                                       "Side is 0"),
    "FT-54": ("LineCrossDetector.get_side",   "Vertical line: side is determined correctly",                                  "Sign is correct"),
    # LineCrossDetector.update — FT-55..58
    "FT-55": ("LineCrossDetector.update",     "Crossing is detected when a point switches sides",                             "Crossing is True"),
    "FT-56": ("LineCrossDetector.update",     "No crossing when a point stays on the same side",                              "Crossing is False"),
    "FT-57": ("LineCrossDetector.update",     "First observation just stores the side; no crossing is triggered",             "Crossing is False"),
    "FT-58": ("LineCrossDetector.update",     "Direction (in/out) is reported correctly on a crossing event",                 "Direction correct"),
    # LineCrossDetector cleanup/state — FT-59..61
    "FT-59": ("LineCrossDetector.cleanup",    "Stale track entries are pruned from the crossing state",                       "Entries removed"),
    "FT-60": ("LineCrossDetector.get_state",  "Returns the current entry/exit crossing counts",                               "Counts correct"),
    "FT-61": ("LineCrossDetector.get_state",  "Returns an empty dict after reset",                                            "Empty dict"),
}


# ─── Helpers ─────────────────────────────────────────────────────────────────
def _count(results: dict) -> dict:
    c = {"PASS": 0, "FAIL": 0, "SKIP": 0, "ERROR": 0, "NOT RUN": 0}
    for v, *_ in results.values():
        c[v] = c.get(v, 0) + 1
    return c


# ─── State JSON (pipeline.save_state pattern) ────────────────────────────────
# Build the same kind of intermediate dict pipeline.py writes to state.json,
# then render Excel from it.  build_report_from_state(state) is the equivalent
# of pipeline.generate_report(date_str, state).

def _humanize_actual(verdict: str, actual: str, expected: str) -> str:
    """Keep the Actual Result column readable.
    PASS  -> mirror the (plain-English) expected text, so the cell is human-readable.
    FAIL  -> keep the captured raw value; debugging needs the original detail.
    SKIP / ERROR / NOT RUN -> blank; nothing useful to show.
    """
    if verdict == "PASS":
        return expected
    if verdict == "FAIL":
        return actual or "(no value captured)"
    return ""


def _feature_block(name: str, meta: dict, results: dict, color: str) -> dict:
    tests = {}
    for tid, (func, desc, expected) in meta.items():
        verdict, actual, notes = results.get(tid, ("NOT RUN", "", ""))
        tests[tid] = {
            "target_function": func,
            "description":     desc,
            "expected":        expected,
            "verdict":         verdict,
            "actual":          _humanize_actual(verdict, actual, expected),
            "notes":           notes,
        }
    return {"name": name, "color": color, "counts": _count(results), "tests": tests}


def assemble_state(fd_r: dict, fr_r: dict, ft_r: dict) -> dict:
    """Assemble the intermediate state dict (the .json that gets written first)."""
    return {
        "version":      STATE_VERSION,
        "run_date":     RUN_DATE,
        "run_ts":       RUN_TS,
        "run_tag":      RUN_TAG,
        "data_source":  "Real CCTV footage  |  data/2026-04-29/videos/videos/",
        "test_approach":"Integration — actual YOLO / InsightFace / OSNet; no mocking",
        "report_file":  REPORT_OUT.name,
        "features": {
            "face_detection":   _feature_block("Face Detection",   FD_META, fd_r, FD_COLOR),
            "face_recognition": _feature_block("Face Recognition", FR_META, fr_r, FR_COLOR),
            "face_tracking":    _feature_block("Face Tracking",    FT_META, ft_r, FT_COLOR),
        },
    }


def save_test_state(state: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, default=str)


def load_test_state(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _state_to_results(feature_block: dict) -> dict:
    """Inverse of _feature_block — turns state['features'][x]['tests'] back
    into the {tid: (verdict, actual, notes)} tuple form Excel builders expect."""
    return {
        tid: (t.get("verdict", "NOT RUN"),
              t.get("actual", ""),
              t.get("notes", ""))
        for tid, t in feature_block.get("tests", {}).items()
    }


def _run_script(script: Path, timeout: int, cwd: Path) -> tuple[str, str]:
    """Run a script as subprocess; return (stdout+stderr, error_msg)."""
    env = {**os.environ, "PYTHONUTF8": "1"}
    try:
        proc = subprocess.run(
            [PYTHON, str(script)],
            capture_output=True, text=True, timeout=timeout,
            cwd=str(cwd), encoding="utf-8", errors="replace",
            env=env,
        )
        return (proc.stdout or "") + "\n" + (proc.stderr or ""), ""
    except subprocess.TimeoutExpired:
        return "", f"timeout >{timeout}s"
    except Exception as exc:
        return "", str(exc)[:120]


# ─── FD runner ────────────────────────────────────────────────────────────────
def run_fd_tests(timeout: int) -> dict:
    """Run test_fd01.py .. test_fd44.py. Returns {test_id: (verdict, actual, notes)}"""
    results = {}
    scripts = sorted(FD_SCRIPTS.glob("test_fd*.py"))
    print(f"\n{'='*70}")
    print(f"  FACE DETECTION — {len(scripts)} scripts  (FD-01 .. FD-44)")
    print(f"{'='*70}")

    for script in scripts:
        m = re.match(r"test_fd(\d+)\.py", script.name)
        if not m:
            continue
        tid = f"FD-{int(m.group(1)):02d}"
        print(f"  [{tid}] {script.name:<35}", end=" ", flush=True)

        out, err = _run_script(script, timeout, REPO_ROOT)
        if err:
            results[tid] = ("ERROR", "", err)
            print(f"ERROR: {err}")
            continue

        # utils.report() prints:  "  PASS  FD-XX: title"  or  "  FAIL  FD-XX: title"
        vm = re.search(rf"^\s+(PASS|FAIL)\s+{re.escape(tid)}:", out, re.MULTILINE)
        if vm:
            verdict = vm.group(1)
        else:
            verdict = "SKIP"

        am = re.search(r"actual\s*:\s*(.+)", out)
        actual = am.group(1).strip()[:120] if am else ""

        results[tid] = (verdict, actual, "")
        print(verdict)

    for tid in FD_META:
        if tid not in results:
            results[tid] = ("NOT RUN", "", "no script file found")
    return results


# ─── FR runner ────────────────────────────────────────────────────────────────
def run_fr_tests(timeout: int) -> dict:
    """Run run_all_tests.py for FR. Returns {test_id: (verdict, '', '')}"""
    results = {}
    runner = FR_SCRIPTS / "run_all_tests.py"
    print(f"\n{'='*70}")
    print(f"  FACE RECOGNITION — 25 tests via run_all_tests.py")
    print(f"{'='*70}")

    if not runner.exists():
        print(f"  [WARN] {runner} not found")
        return {tid: ("NOT RUN", "", "runner missing") for tid in FR_META}

    print("  Running run_all_tests.py ...", flush=True)
    out, err = _run_script(runner, timeout, FR_SCRIPTS)
    if err:
        print(f"  ERROR: {err}")
        return {tid: ("ERROR", "", err) for tid in FR_META}

    # SUMMARY block:  "  [PASS]   FR-EX-01"
    for m in re.finditer(r"\[(\w+)\]\s+(FR-[A-Z]+-\d+)", out):
        v, tid = m.group(1).upper(), m.group(2)
        if v not in ("PASS", "FAIL", "SKIP", "ERROR"):
            v = "ERROR"
        results[tid] = (v, "", "")
        tag = "[P]" if v == "PASS" else ("[F]" if v == "FAIL" else "[-]")
        print(f"  {tag} [{tid}] {v}")

    for tid in FR_META:
        if tid not in results:
            results[tid] = ("NOT RUN", "", "not in runner output")
    return results


# ─── FT runner ────────────────────────────────────────────────────────────────
def run_ft_tests(timeout: int) -> dict:
    """Run all ft*.py scripts then aggregate results.json files."""
    results = {}
    scripts = sorted(FT_SCRIPTS.glob("test_ft*.py"))
    print(f"\n{'='*70}")
    print(f"  FACE TRACKING — {len(scripts)} per-test scripts  (FT-01 .. FT-61)")
    print(f"{'='*70}")
    print("  (Each script loads YOLO+OSNet on real video — may take several minutes)\n")

    for script in scripts:
        print(f"  {script.name:<45}", end=" ", flush=True)
        out, err = _run_script(script, timeout, REPO_ROOT)
        if err:
            print(f"ERROR: {err}")
            continue
        # Summary line: "RESULTS  PASS=X  FAIL=X  SKIP=X"
        ms = re.search(r"RESULTS\s+PASS=(\d+)\s+FAIL=(\d+)\s+SKIP=(\d+)", out)
        if ms:
            print(f"PASS={ms.group(1)} FAIL={ms.group(2)} SKIP={ms.group(3)}")
        else:
            print("done")

    # Read per-test results.json files (data/test_ftNN/results.json).
    # We deliberately skip any leftover bundled folders (data/ftNN_MM_*/...)
    # so stale results from the pre-split era don't shadow fresh runs.
    print(f"\n  Reading results.json files from {FT_DATA.name}/test_ft*/...")
    for sub in sorted(FT_DATA.glob("test_ft*")):
        jpath = sub / "results.json"
        if not jpath.exists():
            continue
        try:
            data = json.loads(jpath.read_text(encoding="utf-8"))
            for tid, info in data.get("results", {}).items():
                status = info.get("status", "ERROR") if isinstance(info, dict) else str(info)
                if status not in ("PASS", "FAIL", "SKIP", "ERROR"):
                    status = "ERROR"
                # Build brief note from extra fields
                note_parts = []
                if isinstance(info, dict):
                    for k, v in info.items():
                        if k == "status":
                            continue
                        if not isinstance(v, (list, dict)):
                            note_parts.append(f"{k}={v}")
                        if len(note_parts) >= 3:
                            break
                note = " | ".join(note_parts)[:120]
                results[tid] = (status, "", note)
        except Exception as exc:
            print(f"  [WARN] {jpath.parent.name}/results.json: {exc}")

    for tid in FT_META:
        if tid not in results:
            results[tid] = ("NOT RUN", "", "no results.json found")

    # Print per-test summary
    for tid in FT_META:
        v = results[tid][0]
        tag = "[P]" if v == "PASS" else ("[F]" if v == "FAIL" else "[-]")
        print(f"  {tag} [{tid}] {v}")

    return results


# ─── Read-only FT results (--skip-run) ───────────────────────────────────────
def read_existing_ft_results() -> dict:
    """Read only per-test (data/test_ft*/results.json) folders — see note in
    run_ft_tests() about ignoring the legacy bundled folders."""
    results = {}
    for sub in sorted(FT_DATA.glob("test_ft*")):
        jpath = sub / "results.json"
        if not jpath.exists():
            continue
        try:
            data = json.loads(jpath.read_text(encoding="utf-8"))
            for tid, info in data.get("results", {}).items():
                status = info.get("status", "NOT RUN") if isinstance(info, dict) else str(info)
                note_parts = []
                if isinstance(info, dict):
                    for k, v in info.items():
                        if k == "status" or isinstance(v, (list, dict)):
                            continue
                        note_parts.append(f"{k}={v}")
                        if len(note_parts) >= 3:
                            break
                results[tid] = (status, "", " | ".join(note_parts)[:120])
        except Exception:
            pass
    for tid in FT_META:
        if tid not in results:
            results[tid] = ("NOT RUN", "", "no results.json found")
    return results


# ─── Excel generation ─────────────────────────────────────────────────────────
def _col_widths(ws, widths: list):
    for i, w in enumerate(widths, 1):
        ws.column_dimensions[get_column_letter(i)].width = w


def build_summary_sheet(wb, fd_r, fr_r, ft_r, *, run_ts: str = RUN_TS,
                        report_name: str = None):
    ws = wb.create_sheet("Summary")
    _col_widths(ws, [22, 11, 11, 11, 11, 11, 11])

    _title_row(ws, "FR_Thor  —  Video-Based Integration Test Suite  —  Run Summary",
               7, 1, bg="FF1F3864", sz=14)

    meta = [
        ("Run Date",      run_ts),
        ("Data Source",   "Real CCTV footage  |  data/2026-04-29/videos/videos/"),
        ("Test Approach", "Integration — actual YOLO / InsightFace / OSNet; no mocking"),
        ("Report File",   report_name or str(REPORT_OUT.name)),
    ]
    for i, (k, v) in enumerate(meta, 2):
        _meta_row(ws, k, v, i, 7)

    # Header
    HDR = len(meta) + 3
    ws.row_dimensions[HDR - 1].height = 8
    for col, h in enumerate(["Feature", "Total", "PASS", "FAIL",
                              "SKIP", "ERROR", "NOT RUN"], 1):
        _hdr(ws.cell(row=HDR, column=col), h)
    ws.row_dimensions[HDR].height = 20

    row_data = [
        ("Face Detection",   fd_r, FD_COLOR),
        ("Face Recognition", fr_r, FR_COLOR),
        ("Face Tracking",    ft_r, FT_COLOR),
    ]
    totals = [0] * 6
    for i, (name, res, color) in enumerate(row_data):
        r = HDR + 1 + i
        ws.row_dimensions[r].height = 22
        c = _count(res)
        vals = [len(res), c["PASS"], c["FAIL"], c["SKIP"], c["ERROR"], c["NOT RUN"]]
        for j, v in enumerate(totals):
            totals[j] += vals[j]

        nc = ws.cell(row=r, column=1, value=name)
        _style(nc, bold=True, bg=color, fc="FFFFFFFF", va="center", wrap=False)

        for col, val in enumerate(vals, 2):
            cell = ws.cell(row=r, column=col, value=val)
            if col == 4 and val > 0:   # FAIL
                _style(cell, bold=True, bg=VERDICT_BG["FAIL"], fc="FFFFFFFF",
                       ha="center", va="center", wrap=False)
            elif col == 3 and val > 0: # PASS
                _style(cell, bg=VERDICT_BG["PASS"],
                       ha="center", va="center", wrap=False)
            else:
                alt = ALT_A if i % 2 == 0 else ALT_B
                _style(cell, bg=alt, ha="center", va="center", wrap=False)

    # Grand total
    tr = HDR + len(row_data) + 1
    ws.row_dimensions[tr].height = 22
    for col, val in enumerate(["TOTAL"] + totals, 1):
        c = ws.cell(row=tr, column=col, value=val)
        _style(c, bold=True, bg=ALT_A,
               ha="center" if col > 1 else "left", va="center", wrap=False)

    # Legend
    lr = tr + 2
    ws.merge_cells(start_row=lr, start_column=1, end_row=lr, end_column=7)
    lc = ws.cell(row=lr, column=1,
                 value="Legend:  PASS = assertion satisfied  |  FAIL = assertion failed  |  "
                       "SKIP = models/video not available  |  "
                       "ERROR = script crashed or timed out  |  "
                       "NOT RUN = script not executed")
    _style(lc, sz=9, bg=ALT_A, va="center", wrap=True)
    ws.row_dimensions[lr].height = 22

    return ws


def build_feature_sheet(wb, name: str, meta: dict, results: dict,
                        color: str, *, run_ts: str = RUN_TS) -> None:
    ws = wb.create_sheet(name)
    _col_widths(ws, [10, 30, 50, 32, 32, 10, 35])
    ws.freeze_panes = "A3"

    _title_row(ws, f"FR_Thor  —  {name}  —  Video-Based Test Results", 7, 1,
               bg=color, sz=13)

    c = _count(results)
    meta_rows = [
        ("Run Date",  run_ts),
        ("Data Source", "Real CCTV footage + actual ML models (no mocking)"),
        ("Results",
         f"Total: {len(meta)}  |  PASS: {c['PASS']}  |  FAIL: {c['FAIL']}  |  "
         f"SKIP: {c['SKIP']}  |  ERROR: {c['ERROR']}  |  NOT RUN: {c['NOT RUN']}"),
    ]
    for i, (k, v) in enumerate(meta_rows, 2):
        _meta_row(ws, k, v, i, 7)

    HDR = len(meta_rows) + 3
    ws.row_dimensions[HDR - 1].height = 6
    col_names = ["Test ID", "Target Function", "Description / Title",
                 "Expected Result", "Actual Result", "Verdict", "Notes"]
    for col, h in enumerate(col_names, 1):
        _hdr(ws.cell(row=HDR, column=col), h, bg=color)
    ws.row_dimensions[HDR].height = 20

    for idx, (tid, (func, title, expected)) in enumerate(meta.items()):
        r = HDR + 1 + idx
        ws.row_dimensions[r].height = 40
        bg = ALT_A if idx % 2 == 0 else ALT_B
        verdict, actual, notes = results.get(tid, ("NOT RUN", "", ""))

        for col, val in enumerate([tid, func, title, expected, actual], 1):
            c = ws.cell(row=r, column=col, value=val)
            _style(c, bg=bg, va="top")

        _verdict_cell(ws.cell(row=r, column=6), verdict)

        nc = ws.cell(row=r, column=7, value=notes)
        _style(nc, bg=bg, va="top")


def build_report_from_state(state: dict, out_path: Path = None) -> Path:
    """Render Excel from a state dict (pipeline.generate_report equivalent).
    Returns the path of the file actually written."""
    out_path = out_path or REPORT_OUT
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fd_r = _state_to_results(state["features"]["face_detection"])
    fr_r = _state_to_results(state["features"]["face_recognition"])
    ft_r = _state_to_results(state["features"]["face_tracking"])
    run_ts = state.get("run_ts", RUN_TS)

    print(f"\n{'='*70}")
    print("  Generating Excel report ...")
    print(f"{'='*70}")

    wb = openpyxl.Workbook()
    wb.remove(wb.active)

    build_summary_sheet(wb, fd_r, fr_r, ft_r, run_ts=run_ts,
                        report_name=out_path.name)
    build_feature_sheet(wb, "Face Detection",   FD_META, fd_r, FD_COLOR, run_ts=run_ts)
    build_feature_sheet(wb, "Face Recognition", FR_META, fr_r, FR_COLOR, run_ts=run_ts)
    build_feature_sheet(wb, "Face Tracking",    FT_META, ft_r, FT_COLOR, run_ts=run_ts)

    try:
        wb.save(str(out_path))
        print(f"  Saved -> {out_path}")
        return out_path
    except PermissionError:
        alt = out_path.with_name(out_path.stem + "_v2.xlsx")
        wb.save(str(alt))
        print(f"  NOTE: File is open in Excel. Saved to -> {alt}")
        return alt


def build_report(fd_r: dict, fr_r: dict, ft_r: dict) -> Path:
    """Back-compat shim: assemble state, save JSON, render Excel."""
    state = assemble_state(fd_r, fr_r, ft_r)
    save_test_state(state, STATE_JSON)
    print(f"  State JSON -> {STATE_JSON}")
    return build_report_from_state(state, REPORT_OUT)


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="FR_Thor Video-Based Test Runner & Excel Report Generator"
    )
    parser.add_argument("--skip-run", action="store_true",
                        help="Skip test execution; rebuild Excel from existing results.json files")
    parser.add_argument("--from-json", default=None,
                        help="Skip everything; render Excel from a previously saved test_state*.json")
    parser.add_argument("--fd-only",  action="store_true",
                        help="Run only Face Detection tests")
    parser.add_argument("--fr-only",  action="store_true",
                        help="Run only Face Recognition tests")
    parser.add_argument("--ft-only",  action="store_true",
                        help="Run only Face Tracking tests")
    parser.add_argument("--timeout",  type=int, default=300,
                        help="Per-script timeout in seconds (default: 300)")
    args = parser.parse_args()

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # ── --from-json replay path ─────────────────────────────────────────────
    if args.from_json:
        src = Path(args.from_json)
        if not src.is_absolute():
            src = (REPORTS_DIR / src).resolve() if not src.exists() else src.resolve()
        print(f"\n{'='*70}")
        print(f"  FR_Thor — Excel Rebuild From State JSON")
        print(f"  Source  : {src}")
        print(f"{'='*70}")
        if not src.exists():
            print(f"  [ERROR] state JSON not found: {src}")
            sys.exit(1)
        state = load_test_state(src)
        out = REPORTS_DIR / f"FR_Thor_VideoTest_Report_{state.get('run_tag', RUN_TAG)}.xlsx"
        build_report_from_state(state, out)
        print(f"\n  Done.  Excel rebuilt from {src.name}\n")
        return

    print(f"\n{'='*70}")
    print(f"  FR_Thor — Video-Based Test Runner & Report Generator")
    print(f"  Started : {RUN_TS}")
    print(f"  Repo    : {REPO_ROOT}")
    print(f"  Reports : {REPORTS_DIR}")
    print(f"  State   : {STATE_JSON.name}")
    print(f"  Excel   : {REPORT_OUT.name}")
    print(f"  Timeout : {args.timeout}s per script")
    if args.skip_run:
        print("  Mode    : --skip-run (Excel rebuild only)")
    print(f"{'='*70}")

    run_all = not (args.fd_only or args.fr_only or args.ft_only)

    # FD
    if not args.skip_run and (run_all or args.fd_only):
        fd_r = run_fd_tests(args.timeout)
    else:
        fd_r = {tid: ("NOT RUN", "", "--skip-run") for tid in FD_META}

    # FR  (whole suite runs as one subprocess; give it 4× timeout)
    if not args.skip_run and (run_all or args.fr_only):
        fr_r = run_fr_tests(args.timeout * 6)
    else:
        fr_r = {tid: ("NOT RUN", "", "--skip-run") for tid in FR_META}

    # FT
    if not args.skip_run and (run_all or args.ft_only):
        ft_r = run_ft_tests(args.timeout)
    elif args.skip_run or args.ft_only:
        print(f"\n{'='*70}")
        print("  FACE TRACKING — reading existing results.json files")
        print(f"{'='*70}")
        ft_r = read_existing_ft_results()
        for tid in FT_META:
            v = ft_r[tid][0]
            tag = "[P]" if v == "PASS" else ("[F]" if v == "FAIL" else "[-]")
            print(f"  {tag} [{tid}] {v}")
    else:
        ft_r = {tid: ("NOT RUN", "", "--skip-run") for tid in FT_META}

    # ── Pipeline-style: JSON first, then render Excel from it ──────────────
    state = assemble_state(fd_r, fr_r, ft_r)
    save_test_state(state, STATE_JSON)
    print(f"\n{'='*70}")
    print(f"  State JSON saved -> {STATE_JSON}")
    print(f"{'='*70}")
    excel_path = build_report_from_state(state, REPORT_OUT)

    # Final summary
    print(f"\n{'='*70}")
    print(f"  FINAL SUMMARY")
    print(f"  {'Feature':<22}  {'Total':>5}  {'PASS':>5}  {'FAIL':>5}  "
          f"{'SKIP':>5}  {'ERR':>5}  {'NOT RUN':>7}")
    print(f"  {'─'*65}")
    for label, res in [("Face Detection", fd_r),
                        ("Face Recognition", fr_r),
                        ("Face Tracking", ft_r)]:
        c = _count(res)
        print(f"  {label:<22}  {len(res):>5}  {c['PASS']:>5}  {c['FAIL']:>5}  "
              f"{c['SKIP']:>5}  {c['ERROR']:>5}  {c['NOT RUN']:>7}")
    print(f"{'='*70}")
    ended = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"  Finished: {ended}")
    print(f"  State   : {STATE_JSON}")
    print(f"  Report  : {excel_path}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
