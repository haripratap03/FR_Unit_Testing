"""
Annotated tracking visualisation
================================

Runs the EXACT same tracking stack as pipeline.py on a real CCTV clip and
writes an annotated MP4. Used as a visual sanity-check that the production
two-tier tracker (Pending -> Active, OSNet ReID, REID_DISTANCE, MAX_BOX_JUMP,
WARMUP_FRAMES, MAX_LOST_SECONDS) is doing what the unit-test verdicts say
it does.

Pipeline parity
---------------
- Person detector : YOLO yolo11l.pt + model.track(classes=[0], conf=CONF_THRESHOLD,
                    persist=True, half=True)        # pipeline.py:846
- ReID extractor  : pipeline.OSNetExtractor (boxmot ReidAutoBackend, OSNet-x1.0)
- Tracker         : pipeline.PersonTracker.match_or_create()
                    -> Phase 1: greedy match to active tracks (cos dist < 0.65 + box gate)
                    -> Phase 2: match to pending; promote after WARMUP_FRAMES
                    -> Phase 3: new pending entries
- Lost reaper     : tracker.get_lost(ts) using MAX_LOST_SECONDS
- Cross-clip      : --all mode serializes/restores active-track banks (.npy)
                    via PersonTracker.serialize_state / restore_state

The only thing missing vs pipeline.py is face capture, body snapshot and FR
classification - those are unrelated to the question "is tracking working".

Output
------
Unit_Test_Data/Face_Tracking/annotated_data/<video_stem>_annotated.mp4
    Down-scaled (1280x720) MP4 with per-track-id colored boxes (active),
    yellow dashed boxes (pending), and a HUD line showing frame index +
    counts of active/pending/finalized tracks.

Run
---
    python Unit_Test_Data/Face_Tracking/scripts/annotate_tracking.py
    python Unit_Test_Data/Face_Tracking/scripts/annotate_tracking.py --video 20260421_130052.mp4 --max-frames 600
    python Unit_Test_Data/Face_Tracking/scripts/annotate_tracking.py --all   # process every clip in videos/videos/
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# sklearn shim — boxmot/__init__.py imports boxmot.postprocessing.gsi which
# imports sklearn for Gaussian-process smoothing. pipeline.py never uses gsi,
# so we stub the two symbols it needs so `from boxmot.reid.core.auto_backend
# import ReidAutoBackend` (used by pipeline.OSNetExtractor) succeeds.
# ---------------------------------------------------------------------------
import sys as _sys
import types as _types
import importlib.machinery as _machinery
if "sklearn" not in _sys.modules:
    def _stub(name: str) -> _types.ModuleType:
        m = _types.ModuleType(name)
        m.__spec__ = _machinery.ModuleSpec(name, loader=None)
        m.__path__ = []  # mark as a package so submodule imports work
        return m
    _sk = _stub("sklearn")
    _gp = _stub("sklearn.gaussian_process")
    _gpk = _stub("sklearn.gaussian_process.kernels")
    _gp.GaussianProcessRegressor = type("GaussianProcessRegressor", (), {})
    _gpk.RBF = type("RBF", (), {})
    _sk.gaussian_process = _gp
    _gp.kernels = _gpk
    _sys.modules["sklearn"] = _sk
    _sys.modules["sklearn.gaussian_process"] = _gp
    _sys.modules["sklearn.gaussian_process.kernels"] = _gpk

# ---------------------------------------------------------------------------
# boxmot path shim — boxmot 12.x renamed boxmot.reid.core.auto_backend ->
# boxmot.appearance.reid_auto_backend. pipeline.py imports the old path, and
# we cannot edit pipeline.py, so alias the old import path to the new module.
# ---------------------------------------------------------------------------
try:
    from pathlib import Path as _Path
    from boxmot.appearance import reid_auto_backend as _new_auto_backend

    # boxmot 12.x signature expects a Path; pipeline.py passes a string.
    _orig_init = _new_auto_backend.ReidAutoBackend.__init__
    def _patched_reid_init(self, weights, device=None, half=False):
        if isinstance(weights, str):
            weights = _Path(weights)
        if device is None:
            import torch as _t
            device = _t.device("cuda:0" if _t.cuda.is_available() else "cpu")
        _orig_init(self, weights=weights, device=device, half=half)
    _new_auto_backend.ReidAutoBackend.__init__ = _patched_reid_init

    _reid_pkg     = _types.ModuleType("boxmot.reid")
    _reid_core    = _types.ModuleType("boxmot.reid.core")
    _reid_pkg.__path__  = []
    _reid_core.__path__ = []
    _reid_pkg.__spec__  = _machinery.ModuleSpec("boxmot.reid", loader=None)
    _reid_core.__spec__ = _machinery.ModuleSpec("boxmot.reid.core", loader=None)
    _sys.modules.setdefault("boxmot.reid", _reid_pkg)
    _sys.modules.setdefault("boxmot.reid.core", _reid_core)
    _sys.modules.setdefault("boxmot.reid.core.auto_backend", _new_auto_backend)
except Exception as _e:
    print(f"[WARN] boxmot reid shim failed: {_e}")

import argparse
import json
import time
from datetime import datetime, timedelta
from pathlib import Path

import cv2
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT  = SCRIPT_DIR.parents[2]
VIDEO_DIR  = REPO_ROOT / "data" / "2026-04-29" / "videos" / "videos"
OUT_DIR    = SCRIPT_DIR.parent / "annotated_data"

_sys.path.insert(0, str(REPO_ROOT))

# Pipeline imports — reuse production classes & thresholds verbatim
import pipeline  # noqa: E402
from pipeline import (  # noqa: E402
    OSNetExtractor,
    PersonTracker,
    CONF_THRESHOLD,
    REID_DISTANCE,
    WARMUP_FRAMES,
    MAX_BOX_JUMP,
    MAX_LOST_SECONDS,
    BANK_SIZE,
)
from ultralytics import YOLO  # noqa: E402

MODEL_PATH = REPO_ROOT / "models" / "yolo11l.pt"
OUT_W, OUT_H = 1280, 720


def _palette(n_colors: int = 32) -> list[tuple[int, int, int]]:
    rng = np.random.default_rng(seed=123)
    cols = rng.integers(60, 255, size=(n_colors, 3), dtype=np.int32)
    return [tuple(int(c) for c in cols[i]) for i in range(n_colors)]


_PALETTE = _palette()


def _color_for(track_id: int) -> tuple[int, int, int]:
    return _PALETTE[track_id % len(_PALETTE)]


def _draw_dashed_box(img, p1, p2, color, thickness=2, dash=8):
    x1, y1 = p1
    x2, y2 = p2
    for x in range(x1, x2, dash * 2):
        cv2.line(img, (x, y1), (min(x + dash, x2), y1), color, thickness)
        cv2.line(img, (x, y2), (min(x + dash, x2), y2), color, thickness)
    for y in range(y1, y2, dash * 2):
        cv2.line(img, (x1, y), (x1, min(y + dash, y2)), color, thickness)
        cv2.line(img, (x2, y), (x2, min(y + dash, y2)), color, thickness)


def _label(img, text, x, y, color, font_scale=0.6):
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)
    cv2.rectangle(img, (x, max(0, y - th - 8)), (x + tw + 6, y), color, -1)
    cv2.putText(img, text, (x + 3, y - 4),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 2, cv2.LINE_AA)


def _frame_timestamp(clip_start: datetime, frame_idx: int, fps: float) -> datetime:
    return clip_start + timedelta(seconds=frame_idx / max(fps, 1.0))


def _parse_clip_start(stem: str) -> datetime:
    # filenames look like 20260421_130052.mp4
    try:
        return datetime.strptime(stem, "%Y%m%d_%H%M%S")
    except ValueError:
        return datetime(2026, 1, 1)


def annotate_video(
    video_path: Path,
    out_path: Path,
    person_model,
    osnet: OSNetExtractor,
    tracker: PersonTracker,
    max_frames: int | None = None,
    prev_clip_end: datetime | None = None,
) -> dict:
    if not video_path.exists():
        raise FileNotFoundError(video_path)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {video_path}")

    src_fps    = cap.get(cv2.CAP_PROP_FPS) or 20.0
    src_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    src_w      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, src_fps, (OUT_W, OUT_H))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"cannot open writer for {out_path}")

    clip_start = _parse_clip_start(video_path.stem)

    # Mirror pipeline.py finalize-on-large-gap logic: if the previous clip ended
    # more than MAX_LOST_SECONDS ago, drop everything still active so stale IDs
    # aren't reused (pipeline.py finalize_all() at clip boundaries).
    if prev_clip_end is not None:
        gap_s = (clip_start - prev_clip_end).total_seconds()
        if gap_s > MAX_LOST_SECONDS:
            tracker.active.clear()
            tracker.pending.clear()

    seen_active_ids: set[int] = set()
    finalized_ids: set[int] = set()
    n_written = 0
    n_dets_total = 0
    last_ts = clip_start
    t0 = time.time()

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if max_frames is not None and n_written >= max_frames:
            break

        ts = _frame_timestamp(clip_start, n_written, src_fps)

        # -- Person detection (verbatim from pipeline.py:846) --------------
        results = person_model.track(
            frame, classes=[0], conf=CONF_THRESHOLD,
            persist=True, verbose=False, half=True,
        )

        if results and results[0].boxes is not None and len(results[0].boxes) > 0:
            boxes = results[0].boxes.xyxy.cpu().numpy()
        else:
            boxes = np.zeros((0, 4), dtype=np.float32)

        # -- PersonTracker (production logic) ------------------------------
        tracked = tracker.match_or_create(frame, boxes, ts)  # [(uid, box), ...]
        active_box_by_uid = {uid: box for uid, box in tracked}

        # Reap lost tracks before drawing so HUD reflects current state
        for uid in tracker.get_lost(ts):
            tracker.remove(uid)
            finalized_ids.add(uid)

        # -- Render --------------------------------------------------------
        canvas = cv2.resize(frame, (OUT_W, OUT_H), interpolation=cv2.INTER_AREA)
        sx = OUT_W / src_w
        sy = OUT_H / src_h

        # Pending tracks (yellow dashed) — peek into tracker state for visibility
        for pk, pend in tracker.pending.items():
            x1, y1, x2, y2 = pend.last_box
            tx1, ty1 = int(x1 * sx), int(y1 * sy)
            tx2, ty2 = int(x2 * sx), int(y2 * sy)
            _draw_dashed_box(canvas, (tx1, ty1), (tx2, ty2), (0, 215, 255), 2, dash=8)
            _label(canvas, f"PEND f={pend.frames_seen}/{WARMUP_FRAMES}",
                   tx1, ty1, (0, 215, 255), font_scale=0.5)

        # Active tracks (per-id colored)
        for uid, box in tracked:
            seen_active_ids.add(uid)
            x1, y1, x2, y2 = box
            tx1, ty1 = int(x1 * sx), int(y1 * sy)
            tx2, ty2 = int(x2 * sx), int(y2 * sy)
            color = _color_for(uid)
            cv2.rectangle(canvas, (tx1, ty1), (tx2, ty2), color, 2)
            _label(canvas, f"ID {uid}", tx1, ty1, color, font_scale=0.7)
            n_dets_total += 1

        # HUD
        hud_lines = [
            f"frame {n_written + 1}/{min(src_frames, max_frames or src_frames)}    "
            f"clip {video_path.name}    t={ts.strftime('%H:%M:%S')}",
            f"active={len(tracker.active)}  pending={len(tracker.pending)}  "
            f"finalized(this clip)={len(finalized_ids)}  unique-IDs-seen={len(seen_active_ids)}",
            f"REID_DIST={REID_DISTANCE}  WARMUP={WARMUP_FRAMES}  "
            f"MAX_BOX_JUMP={MAX_BOX_JUMP}  MAX_LOST_S={MAX_LOST_SECONDS}",
        ]
        cv2.rectangle(canvas, (0, 0), (OUT_W, 76), (0, 0, 0), -1)
        for i, line in enumerate(hud_lines):
            cv2.putText(canvas, line, (10, 22 + i * 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1, cv2.LINE_AA)

        writer.write(canvas)
        n_written += 1
        last_ts = ts

        if n_written % 50 == 0:
            elapsed = time.time() - t0
            fps = n_written / elapsed if elapsed > 0 else 0.0
            print(f"  ... {n_written} frames written  "
                  f"({fps:.1f} fps,  active={len(tracker.active)} "
                  f"pending={len(tracker.pending)} "
                  f"unique-IDs={len(seen_active_ids)})")

    cap.release()
    writer.release()

    return {
        "video": video_path.name,
        "out": str(out_path),
        "frames_written": n_written,
        "detections_total": n_dets_total,
        "unique_active_track_ids": len(seen_active_ids),
        "finalized_in_clip": len(finalized_ids),
        "active_at_end": len(tracker.active),
        "pending_at_end": len(tracker.pending),
        "elapsed_s": round(time.time() - t0, 1),
        "clip_end_ts": last_ts.isoformat(),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="20260421_130052.mp4",
                    help="video filename inside data/2026-04-29/videos/videos/")
    ap.add_argument("--max-frames", type=int, default=400,
                    help="cap frames per clip (default 400 ~ 20s @ 20fps); -1 = whole clip")
    ap.add_argument("--all", action="store_true",
                    help="process every .mp4 in the videos directory; tracker state is "
                         "carried across clips (pipeline.py cross-clip continuity)")
    args = ap.parse_args()

    if not MODEL_PATH.exists():
        print(f"[ERROR] yolo11l.pt missing at {MODEL_PATH}")
        _sys.exit(1)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    max_frames = None if args.max_frames == -1 else args.max_frames

    if args.all:
        clips = sorted(VIDEO_DIR.glob("*.mp4"))
    else:
        clips = [VIDEO_DIR / args.video]

    if not clips:
        print(f"[ERROR] no clips found at {VIDEO_DIR}")
        _sys.exit(1)

    print(f"Annotating {len(clips)} clip(s)  ->  {OUT_DIR}")
    print(f"Tracker  : pipeline.PersonTracker (production)")
    print(f"Detector : YOLO {MODEL_PATH.name}    conf>={CONF_THRESHOLD}    classes=[person]")
    print(f"OSNet    : models/osnet_x1_0_msmt17.pt    bank_size={BANK_SIZE}")
    print(f"Thresholds: REID_DIST={REID_DISTANCE}  WARMUP={WARMUP_FRAMES}  "
          f"BOX_JUMP={MAX_BOX_JUMP}  MAX_LOST_S={MAX_LOST_SECONDS}")
    print()

    person_model = YOLO(str(MODEL_PATH))
    osnet = OSNetExtractor()
    if osnet._model is None:
        print("[WARN] OSNet failed to load — production code falls back to RANDOM "
              "embeddings, which makes track IDs unstable. Verify boxmot install.")
    tracker = PersonTracker(osnet)

    summaries = []
    prev_clip_end: datetime | None = None
    for clip in clips:
        out_path = OUT_DIR / f"{clip.stem}_annotated.mp4"
        print(f"[{clip.name}] -> {out_path.name}")
        try:
            summary = annotate_video(
                clip, out_path, person_model, osnet, tracker,
                max_frames=max_frames,
                prev_clip_end=prev_clip_end,
            )
            summaries.append(summary)
            prev_clip_end = datetime.fromisoformat(summary["clip_end_ts"])
            print(f"  done: {summary['frames_written']} frames, "
                  f"{summary['detections_total']} dets, "
                  f"{summary['unique_active_track_ids']} unique active IDs, "
                  f"active@end={summary['active_at_end']} pending@end={summary['pending_at_end']}, "
                  f"{summary['elapsed_s']}s")
        except Exception as exc:
            print(f"  ERROR: {type(exc).__name__}: {exc}")
        print()

    if summaries:
        print("=" * 72)
        print("Summary")
        print("=" * 72)
        for s in summaries:
            print(f"  {s['video']:<32} frames={s['frames_written']:<5} "
                  f"dets={s['detections_total']:<6} "
                  f"unique-active={s['unique_active_track_ids']:<4} "
                  f"final={s['finalized_in_clip']:<4} time={s['elapsed_s']}s")


if __name__ == "__main__":
    main()
