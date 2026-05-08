"""
_ft_helpers.py  —  Shared utilities for the per-test Face Tracking scripts.

Each test_ftXX.py focuses on one FT-XX assertion. To keep those scripts small
and free of duplicated YOLO/OSNet boilerplate, the common setup lives here.

What this module provides:
  paths      REPO_ROOT, VIDEO_DIR, VIDEO_PATH, VIDEO_PATH2, YOLO_PATH, DATA_ROOT
  models     load_yolo()
  detection  detect_persons(), sample_frame(), read_frames()
  embeddings collect_embeddings(), collect_person_detections(),
             extract_person_centroids(), extract_frame_detections()
  factories  make_active_track(), make_pending_tracker_struct(),
             make_tracker(), make_emb_at_dist()
  output     prepare_data_dir(), save_single_test_result()

Usage from a test_ftXX.py:

    from _ft_helpers import (
        VIDEO_PATH, prepare_data_dir, save_single_test_result, ...
    )

The helpers never run themselves; importing has only the side-effect of
adding REPO_ROOT to sys.path so `import pipeline` works.
"""

from __future__ import annotations

import json
import sys
from collections import deque
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock

import cv2
import numpy as np

# ── paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent              # .../Face_Tracking/scripts
DATA_ROOT  = SCRIPT_DIR.parent / "data"                   # .../Face_Tracking/data
REPO_ROOT  = SCRIPT_DIR.parent.parent.parent              # .../FR_Thor

VIDEO_DIR  = REPO_ROOT / "data" / "2026-04-29" / "videos" / "videos"
_clips     = sorted(VIDEO_DIR.glob("*.mp4"))
VIDEO_PATH  = _clips[0] if _clips         else VIDEO_DIR / "20260421_130052.mp4"
VIDEO_PATH2 = _clips[1] if len(_clips) > 1 else VIDEO_DIR / "20260421_130152.mp4"

YOLO_PATH  = str(REPO_ROOT / "models" / "yolo11l.pt")

# Make `import pipeline` work for callers
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# ── data dir / output helpers ────────────────────────────────────────────────
def prepare_data_dir(test_id: str) -> Path:
    """Make and return Unit_Test_Data/Face_Tracking/data/test_ftNN/."""
    folder = DATA_ROOT / test_id.lower().replace("-", "_")   # FT-01 -> test_ft01
    if folder.name.startswith("ft"):                         # ensure test_ prefix
        folder = DATA_ROOT / f"test_{folder.name}"
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def save_single_test_result(test_id: str, data_out: Path, result: dict,
                            extras: dict | None = None) -> None:
    """Write a results.json containing only this one test's outcome.

    Schema matches what run_all_and_report.py expects:
      {"results": {"FT-XX": {"status": "PASS"|"FAIL"|"SKIP", ...}}, ...}
    """
    payload = {
        "test_id": test_id,
        "timestamp": datetime.now().isoformat(),
        "results": {test_id: result},
    }
    if extras:
        payload.update(extras)
    with open(data_out / "results.json", "w") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"\nSaved -> {data_out / 'results.json'}")


def print_verdict(test_id: str, status: str) -> None:
    bar = "=" * 60
    print(f"\n{bar}\nRESULT  {status}  {test_id}\n{bar}")


# ── model loading ────────────────────────────────────────────────────────────
def load_yolo():
    from ultralytics import YOLO
    return YOLO(YOLO_PATH)


def load_osnet():
    import pipeline
    return pipeline.OSNetExtractor()


# ── detection / video helpers ────────────────────────────────────────────────
def detect_persons(model, frame, conf: float = 0.45):
    """Return [[x1,y1,x2,y2], ...] for class=0 (person) detections."""
    res = model(frame, classes=[0], conf=conf, verbose=False)[0]
    return [[int(v) for v in b.xyxy[0].tolist()] for b in res.boxes]


def sample_frame(video_path: Path = VIDEO_PATH, frame_idx: int = 30):
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    cap.release()
    assert ok, f"Could not read frame {frame_idx} from {video_path}"
    return frame


def read_frames(video_path: Path, start: int = 0, count: int = 60, step: int = 2):
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    frames = []
    for i in range(count * step):
        ok, f = cap.read()
        if not ok:
            break
        if i % step == 0:
            frames.append(f)
    cap.release()
    return frames


# ── embedding collectors ─────────────────────────────────────────────────────
def collect_embeddings(n: int = 6, frame_step: int = 5,
                       video_path: Path = VIDEO_PATH):
    """Track the largest person across sampled frames; return real 512-d
    embeddings + their bounding boxes (no normalisation tricks)."""
    import pipeline
    yolo  = load_yolo()
    osnet = pipeline.OSNetExtractor()
    cap   = cv2.VideoCapture(str(video_path))
    embs, boxes = [], []
    idx = 0
    while cap.isOpened() and len(embs) < n:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % frame_step == 0:
            dets = detect_persons(yolo, frame)
            if dets:
                box = max(dets, key=lambda b: (b[2]-b[0])*(b[3]-b[1]))
                e = osnet.extract(frame, box)
                if e is not None:
                    embs.append(e)
                    boxes.append(box)
        idx += 1
    cap.release()
    return embs, boxes


def collect_person_detections(n: int = 16, frame_step: int = 3,
                              video_path: Path = VIDEO_PATH):
    """Alias of collect_embeddings with a different default — kept for
    readability in the FT-18..22 family which historically used n=20/step=3."""
    return collect_embeddings(n=n, frame_step=frame_step, video_path=video_path)


def extract_person_centroids(n_frames: int = 80, conf: float = 0.45,
                             video_path: Path = VIDEO_PATH):
    """Run YOLO and return [(nx, ny), ...] normalised centroids for all person
    detections across the first n_frames, plus (frame_w, frame_h)."""
    yolo = load_yolo()
    cap  = cv2.VideoCapture(str(video_path))
    ok, first = cap.read()
    h, w = first.shape[:2] if ok else (1520, 2688)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    centroids = []
    for _ in range(n_frames):
        ok, frame = cap.read()
        if not ok:
            break
        res = yolo(frame, classes=[0], conf=conf, verbose=False)[0]
        for b in res.boxes:
            x1, y1, x2, y2 = [int(v) for v in b.xyxy[0].tolist()]
            centroids.append((((x1+x2)/2)/w, ((y1+y2)/2)/h))
    cap.release()
    return centroids, (w, h)


def extract_frame_detections(n_frames: int = 80, conf: float = 0.45,
                             video_path: Path = VIDEO_PATH):
    """Like extract_person_centroids, but grouped per frame:
       [(frame_idx, [(nx,ny), ...]), ...] + (w, h)."""
    yolo = load_yolo()
    cap  = cv2.VideoCapture(str(video_path))
    ok, first = cap.read()
    h, w = first.shape[:2] if ok else (1520, 2688)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    per_frame = []
    for fi in range(n_frames):
        ok, frame = cap.read()
        if not ok:
            break
        res = yolo(frame, classes=[0], conf=conf, verbose=False)[0]
        pts = [(((b.xyxy[0][0].item()+b.xyxy[0][2].item())/2)/w,
                ((b.xyxy[0][1].item()+b.xyxy[0][3].item())/2)/h)
               for b in res.boxes]
        per_frame.append((fi, pts))
    cap.release()
    return per_frame, (w, h)


# ── factories ────────────────────────────────────────────────────────────────
def make_active_track(emb, box=None, timestamp=None, lost_seconds: float = 0.0,
                      bank_maxlen: int | None = None):
    import pipeline
    t = pipeline.ActiveTrack.__new__(pipeline.ActiveTrack)
    t.bank = deque(maxlen=bank_maxlen if bank_maxlen else pipeline.BANK_SIZE)
    t.bank.append(emb.copy())
    t.last_box = list(box) if box else [100, 100, 200, 300]
    t.last_timestamp = timestamp or datetime.now()
    t.lost_seconds = lost_seconds
    return t


def make_tracker(osnet=None, frame_wh=(2688, 1520)):
    """A bare PersonTracker shell — no YOLO, no real detection. The osnet
    backend is a MagicMock unless the caller passes a real OSNetExtractor."""
    import pipeline
    t = pipeline.PersonTracker.__new__(pipeline.PersonTracker)
    t.osnet = osnet if osnet is not None else MagicMock()
    t.active = {}
    t.pending = {}
    t.pending_next = 0
    t.next_id = 1
    t.frame_wh = frame_wh
    return t


def make_emb_at_dist(ref: np.ndarray, dist: float, seed: int = 77) -> np.ndarray:
    """Return a unit vector at cosine distance ~dist from `ref`."""
    dot = float(np.clip(1.0 - dist, -1.0, 1.0))
    rng = np.random.default_rng(seed)
    perp = rng.standard_normal(len(ref)).astype(np.float32)
    perp -= perp.dot(ref) * ref
    n = np.linalg.norm(perp)
    perp = perp / n if n > 1e-8 else rng.standard_normal(len(ref)).astype(np.float32)
    sin = float(np.sqrt(max(0.0, 1.0 - dot ** 2)))
    v = dot * ref + sin * perp
    return v / np.linalg.norm(v)


def random_unit_vec(seed: int, dim: int = 512) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim).astype(np.float32)
    return v / np.linalg.norm(v)
