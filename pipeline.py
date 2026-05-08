#!/usr/bin/env python3
"""
Offline FR Pipeline — Fetch videos from recording Jetson, track, capture faces, FR, report.

Usage:
    python pipeline.py                    # Continuous service mode
    python pipeline.py --once             # Process once and exit
    python pipeline.py --date 2026-04-06  # Process specific date
    python pipeline.py --local-dir ./videos/2026-04-06  # Process local videos (skip fetch)

Folder structure:
    offline_fr/
    ├── config.json
    ├── pipeline.py
    ├── models/              (yolo11l.pt, yolov11s-face.pt, osnet_x1_0_msmt17.pt)
    ├── insightface_models/  (antelopev2 ONNX files)
    ├── vector_db/           (faiss.index, staff_registry.json)
    ├── data/
    │   └── YYYY-MM-DD/
    │       ├── videos/      (fetched clips)
    │       ├── faces/       (captured face images)
    │       ├── tracks/      (embedding bank .npy files)
    │       ├── state.json   (processing state)
    │       └── report/      (Excel report)
    └── logs/
"""

import os
import re
import sys
import json
import time
import shutil
import signal
import logging
import argparse
import subprocess
import threading
import queue
import numpy as np
import cv2
from datetime import datetime, timedelta
from collections import deque, defaultdict
from concurrent.futures import ThreadPoolExecutor

VERSION = "1.7.0"  # bump this on every deploy

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("offline_fr")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# =============================================================================
# CONFIG
# =============================================================================

def load_config():
    config_path = os.path.join(SCRIPT_DIR, "config.json")
    with open(config_path, 'r') as f:
        return json.load(f)

CONFIG = load_config()
STORE_NAME = CONFIG.get("store_name", "unknown")
REC_CFG = CONFIG.get("recording_jetson", {})
VID_CFG = CONFIG.get("video", {})
TRK_CFG = CONFIG.get("tracking", {})
FC_CFG = CONFIG.get("face_capture", {})
FR_CFG = CONFIG.get("fr", {})
LINE_CFG = CONFIG.get("line", {})
IN_SIDE = CONFIG.get("in_side", "below")

PROCESSING_FPS = VID_CFG.get("processing_fps", 4)
EXPECTED_FPS = VID_CFG.get("expected_fps", 30)
MAX_FACES = FC_CFG.get("max_faces_per_person", 10)
REID_DISTANCE = TRK_CFG.get("reid_distance", 0.25)
WARMUP_FRAMES = TRK_CFG.get("warmup_frames", 3)
BANK_SIZE = TRK_CFG.get("bank_size", 10)
UPDATE_GATE = TRK_CFG.get("update_gate", 0.8)
MAX_BOX_JUMP = TRK_CFG.get("max_box_jump", 0.3)
MAX_LOST_SECONDS = TRK_CFG.get("max_lost_seconds", 90)
CONF_THRESHOLD = TRK_CFG.get("conf_threshold", 0.45)
MIN_CROP_W = TRK_CFG.get("min_crop_w", 32)
MIN_CROP_H = TRK_CFG.get("min_crop_h", 64)

# Post-capture face filters (run once after all clips are processed)
FILTER_MIN_KEYPOINTS = FC_CFG.get("min_keypoints", 3)
FILTER_MIN_KP_CONF = FC_CFG.get("min_kp_conf", 0.5)
FILTER_MIN_SKIN = FC_CFG.get("min_skin_ratio", 0.20)
FILTER_MIN_QUALITY = FC_CFG.get("min_quality", 25)
POST_FILTER_ENABLED = FC_CFG.get("post_filter_enabled", True)
POST_FILTER_KEEP_TOP_N = FC_CFG.get("post_filter_keep_top_n", 20)


def data_dir(date_str):
    d = os.path.join(SCRIPT_DIR, "data", date_str)
    for sub in ["videos", "faces", "tracks", "report"]:
        os.makedirs(os.path.join(d, sub), exist_ok=True)
    return d


# =============================================================================
# TIMESTAMP FROM FILENAME
# =============================================================================

def parse_video_timestamp(filename):
    """Parse timestamp from filename like 20260406_153315.mp4"""
    base = os.path.splitext(os.path.basename(filename))[0]
    try:
        return datetime.strptime(base, "%Y%m%d_%H%M%S")
    except ValueError:
        return None


def frame_timestamp(clip_start, frame_idx, fps):
    """Real wall-clock time for a frame."""
    return clip_start + timedelta(seconds=frame_idx / fps)


# =============================================================================
# VIDEO FETCHER
# =============================================================================

def fetch_videos(date_str, state):
    """Fetch new video clips from recording Jetson via SCP."""
    host = REC_CFG.get("host")
    user = REC_CFG.get("user")
    password = REC_CFG.get("password", "")
    ssh_key = REC_CFG.get("ssh_key", "")
    remote_base = REC_CFG.get("video_base_path", "")
    min_age = REC_CFG.get("min_file_age_seconds", 10)

    if not host or not user or not remote_base:
        return []

    remote_dir = f"{remote_base}/{date_str}"
    local_dir = os.path.join(data_dir(date_str), "videos")
    processed = set(state.get("processed_clips", []))

    # List remote files via SSH
    ssh_prefix = f"ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10"
    if ssh_key:
        ssh_prefix += f" -i {ssh_key}"
    if password:
        ssh_prefix = f"sshpass -p '{password}' " + ssh_prefix

    try:
        cmd = f"{ssh_prefix} {user}@{host} 'ls -1 {remote_dir}/*.mp4 2>/dev/null'"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            logger.debug(f"No remote videos found for {date_str}")
            return []

        remote_files = [os.path.basename(f.strip()) for f in result.stdout.strip().split('\n') if f.strip()]
    except Exception as e:
        logger.warning(f"SSH list failed: {e}")
        return []

    # Filter out already processed
    new_files = sorted([f for f in remote_files if f not in processed and f.endswith('.mp4')])
    if not new_files:
        return []

    # SCP new files
    fetched = []
    for filename in new_files:
        remote_path = f"{remote_dir}/{filename}"
        local_path = os.path.join(local_dir, filename)

        if os.path.exists(local_path):
            fetched.append(filename)
            continue

        scp_prefix = f"scp -o StrictHostKeyChecking=no -o ConnectTimeout=10"
        if ssh_key:
            scp_prefix += f" -i {ssh_key}"
        if password:
            scp_prefix = f"sshpass -p '{password}' " + scp_prefix

        try:
            cmd = f"{scp_prefix} {user}@{host}:{remote_path} {local_path}"
            result = subprocess.run(cmd, shell=True, capture_output=True, timeout=120)
            if result.returncode == 0:
                fetched.append(filename)
                logger.info(f"Fetched: {filename}")
            else:
                logger.warning(f"SCP failed for {filename}")
        except Exception as e:
            logger.warning(f"SCP error: {e}")

    return sorted(fetched)


# =============================================================================
# STATE MANAGEMENT (replaces DB)
# =============================================================================

def load_state(date_str):
    state_path = os.path.join(data_dir(date_str), "state.json")
    if os.path.exists(state_path):
        with open(state_path, 'r') as f:
            return json.load(f)
    return {
        "date": date_str,
        "processed_clips": [],
        "next_track_id": 1,
        "active_tracks": {},
        "persons": [],
        "fr_results": {"staff": [], "customer_clusters": [], "no_face_count": 0}
    }


def save_state(date_str, state):
    state_path = os.path.join(data_dir(date_str), "state.json")
    with open(state_path, 'w') as f:
        json.dump(state, f, indent=2, default=str)


# =============================================================================
# OSNET EXTRACTOR
# =============================================================================

def _l2(v):
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


class OSNetExtractor:
    def __init__(self):
        self._model = None
        weights = os.path.join(SCRIPT_DIR, "models", "osnet_x1_0_msmt17.pt")
        try:
            from boxmot.reid.core.auto_backend import ReidAutoBackend
            import torch
            dev = 0 if torch.cuda.is_available() else "cpu"
            self._model = ReidAutoBackend(weights=weights, device=dev, half=True)
            logger.info(f"OSNet loaded: {weights}")
        except Exception as e:
            logger.warning(f"OSNet failed: {e}. Using random embeddings.")

    def extract(self, frame, bbox):
        x1, y1, x2, y2 = (int(v) for v in bbox)
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
        if (x2 - x1) < MIN_CROP_W or (y2 - y1) < MIN_CROP_H:
            return None
        if self._model is None:
            return _l2(np.random.randn(512).astype(np.float32))
        try:
            return _l2(self._model.model.get_features(
                np.array([[x1, y1, x2, y2]], np.float32), frame)[0].astype(np.float32))
        except:
            return None

    def extract_batch(self, frame, boxes):
        """Extract embeddings for all boxes at once — much faster on GPU."""
        if len(boxes) == 0:
            return []

        results = []
        valid_boxes = []
        valid_indices = []
        h, w = frame.shape[:2]

        for i, bbox in enumerate(boxes):
            x1, y1, x2, y2 = (int(v) for v in bbox)
            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
            if (x2 - x1) >= MIN_CROP_W and (y2 - y1) >= MIN_CROP_H:
                valid_boxes.append([x1, y1, x2, y2])
                valid_indices.append(i)

        # Initialize results with None for all boxes
        results = [None] * len(boxes)

        if not valid_boxes:
            return results

        if self._model is None:
            for idx in valid_indices:
                results[idx] = _l2(np.random.randn(512).astype(np.float32))
            return results

        try:
            # Batch extraction — single GPU call for all persons
            feats = self._model.model.get_features(
                np.array(valid_boxes, np.float32), frame)
            for i, idx in enumerate(valid_indices):
                results[idx] = _l2(feats[i].astype(np.float32))
        except Exception as e:
            # Fallback to sequential
            for i, idx in enumerate(valid_indices):
                try:
                    feat = self._model.model.get_features(
                        np.array([valid_boxes[i]], np.float32), frame)
                    results[idx] = _l2(feat[0].astype(np.float32))
                except:
                    pass

        return results


# =============================================================================
# LINE CROSS DETECTOR
# =============================================================================

class LineCrossDetector:
    def __init__(self, line_a, line_b, in_side):
        self.line_a = line_a
        self.line_b = line_b
        self.in_side = in_side
        self.out_side = 'below' if in_side == 'above' else 'above'
        self.person_sides = {}

    def get_side(self, nx, ny):
        x1, y1 = self.line_a['x'], self.line_a['y']
        x2, y2 = self.line_b['x'], self.line_b['y']
        if x2 == x1:
            line_y = y1
        else:
            line_y = y1 + (y2 - y1) / (x2 - x1) * (nx - x1)
        return 'above' if ny < line_y else 'below'

    def update(self, track_id, nx, ny):
        current = self.get_side(nx, ny)
        if track_id not in self.person_sides:
            self.person_sides[track_id] = current
            return None
        prev = self.person_sides[track_id]
        self.person_sides[track_id] = current
        if prev == current:
            return None
        if prev == self.out_side and current == self.in_side:
            return 'enter'
        if prev == self.in_side and current == self.out_side:
            return 'exit'
        return None

    def cleanup(self, track_id):
        self.person_sides.pop(track_id, None)

    def get_state(self):
        return dict(self.person_sides)

    def restore_state(self, sides):
        self.person_sides = dict(sides)


# =============================================================================
# TRACKER (with serialization for cross-clip continuity)
# =============================================================================

class ActiveTrack:
    def __init__(self, emb, box, timestamp):
        self.bank = deque(maxlen=BANK_SIZE)
        self.bank.append(emb.copy())
        self.last_box = list(box)
        self.last_timestamp = timestamp
        self.lost_seconds = 0.0

    @property
    def ref(self):
        n = len(self.bank)
        w = np.linspace(0.5, 1.0, n)
        return _l2(np.average(list(self.bank), axis=0, weights=w))

    def update_embedding(self, emb):
        self.bank.append(emb.copy())

    def serialize(self, tracks_dir, track_id):
        bank_path = os.path.join(tracks_dir, f"track_{track_id:04d}_bank.npy")
        np.save(bank_path, np.array(list(self.bank)))
        return {
            "bank_file": os.path.basename(bank_path),
            "last_box": self.last_box,
            "last_timestamp": str(self.last_timestamp),
            "lost_seconds": self.lost_seconds
        }

    @staticmethod
    def deserialize(data, tracks_dir):
        bank_path = os.path.join(tracks_dir, data["bank_file"])
        bank_arr = np.load(bank_path)
        track = ActiveTrack.__new__(ActiveTrack)
        track.bank = deque(maxlen=BANK_SIZE)
        for row in bank_arr:
            track.bank.append(row.astype(np.float32))
        track.last_box = [float(v) for v in data["last_box"]]
        track.last_timestamp = data["last_timestamp"]
        track.lost_seconds = float(data.get("lost_seconds", 0.0))
        return track


class PendingTrack:
    def __init__(self, emb, box):
        self.bank = [emb.copy()]
        self.last_box = list(box)
        self.frames_seen = 1

    @property
    def ref(self):
        return _l2(np.mean(self.bank, axis=0))

    def add(self, emb, box):
        self.bank.append(emb.copy())
        self.last_box = list(box)
        self.frames_seen += 1


class PersonTracker:
    def __init__(self, osnet):
        self.osnet = osnet
        self.active = {}       # track_id -> ActiveTrack
        self.pending = {}      # pending_key -> PendingTrack
        self.pending_next = 0
        self.next_id = 1
        self.frame_wh = (1280, 720)

    def _box_distance(self, box_a, box_b):
        w, h = self.frame_wh
        cx_a = (box_a[0] + box_a[2]) / 2 / w
        cy_a = (box_a[1] + box_a[3]) / 2 / h
        cx_b = (box_b[0] + box_b[2]) / 2 / w
        cy_b = (box_b[1] + box_b[3]) / 2 / h
        return ((cx_a - cx_b)**2 + (cy_a - cy_b)**2) ** 0.5

    def match_or_create(self, frame, boxes, timestamp, gap_seconds=0):
        self.frame_wh = (frame.shape[1], frame.shape[0])
        box_jump = MAX_BOX_JUMP * max(1, gap_seconds / 0.5) if gap_seconds > 0 else MAX_BOX_JUMP
        box_jump = min(box_jump, 0.8)

        if len(boxes) == 0:
            return []

        # Batch OSNet extraction — single GPU call for all persons
        embeddings = self.osnet.extract_batch(frame, boxes)
        results = []
        used_det = set()
        used_active = set()

        # Phase 1: Match to active tracks
        if self.active:
            active_ids = list(self.active.keys())
            active_refs = [self.active[uid].ref for uid in active_ids]
            valid_dets = [(di, emb) for di, emb in enumerate(embeddings) if emb is not None]

            if valid_dets and active_refs:
                det_mat = np.stack([e for _, e in valid_dets])
                ref_mat = np.stack(active_refs)
                dist_matrix = 1.0 - det_mat @ ref_mat.T

                used_d = [False] * len(valid_dets)
                used_a = [False] * len(active_ids)

                for _ in range(min(len(valid_dets), len(active_ids))):
                    tmp = dist_matrix.copy()
                    for r in range(len(valid_dets)):
                        if used_d[r]: tmp[r, :] = 1.0
                    for c in range(len(active_ids)):
                        if used_a[c]: tmp[:, c] = 1.0

                    for r, (di, _) in enumerate(valid_dets):
                        if used_d[r]: continue
                        for c, uid in enumerate(active_ids):
                            if used_a[c]: continue
                            if self._box_distance(boxes[di], self.active[uid].last_box) > box_jump:
                                tmp[r, c] = 1.0

                    best_idx = int(np.argmin(tmp))
                    r, c = divmod(best_idx, len(active_ids))
                    if tmp[r, c] >= REID_DISTANCE:
                        break

                    di = valid_dets[r][0]
                    uid = active_ids[c]
                    det_emb = valid_dets[r][1]

                    used_d[r] = True
                    used_a[c] = True
                    used_det.add(di)
                    used_active.add(uid)

                    if tmp[r, c] < REID_DISTANCE * UPDATE_GATE:
                        self.active[uid].update_embedding(det_emb)

                    self.active[uid].last_box = list(boxes[di])
                    self.active[uid].last_timestamp = timestamp
                    self.active[uid].lost_seconds = 0
                    results.append((uid, boxes[di]))

        # Phase 2: Match to pending
        if self.pending:
            pend_keys = list(self.pending.keys())
            pend_refs = [self.pending[pk].ref for pk in pend_keys]
            unmatched = [(di, embeddings[di]) for di in range(len(boxes))
                         if di not in used_det and embeddings[di] is not None]

            if unmatched and pend_refs:
                det_mat = np.stack([e for _, e in unmatched])
                ref_mat = np.stack(pend_refs)
                pd = 1.0 - det_mat @ ref_mat.T
                used_um = [False] * len(unmatched)
                used_pk = [False] * len(pend_keys)

                for _ in range(min(len(unmatched), len(pend_keys))):
                    tmp = pd.copy()
                    for r in range(len(unmatched)):
                        if used_um[r]: tmp[r, :] = 1.0
                    for c in range(len(pend_keys)):
                        if used_pk[c]: tmp[:, c] = 1.0

                    best_idx = int(np.argmin(tmp))
                    r, c = divmod(best_idx, len(pend_keys))
                    if tmp[r, c] >= REID_DISTANCE:
                        break

                    di = unmatched[r][0]
                    pk = pend_keys[c]
                    used_um[r] = True
                    used_pk[c] = True
                    used_det.add(di)

                    pend = self.pending[pk]
                    pend.add(unmatched[r][1], boxes[di])

                    if pend.frames_seen >= WARMUP_FRAMES:
                        uid = self.next_id
                        self.next_id += 1
                        track = ActiveTrack(pend.ref, boxes[di], timestamp)
                        for e in pend.bank:
                            track.bank.append(e.copy())
                        self.active[uid] = track
                        del self.pending[pk]
                        results.append((uid, boxes[di]))

        # Phase 3: New pending
        for di in range(len(boxes)):
            if di not in used_det and embeddings[di] is not None:
                pk = self.pending_next
                self.pending_next += 1
                self.pending[pk] = PendingTrack(embeddings[di], boxes[di])

        return results

    def get_lost(self, current_time, max_lost_seconds=MAX_LOST_SECONDS):
        lost = []
        for uid, track in self.active.items():
            if uid not in []:
                try:
                    last_ts = track.last_timestamp
                    if isinstance(last_ts, str):
                        last_ts = datetime.fromisoformat(last_ts)
                    elapsed = (current_time - last_ts).total_seconds()
                    if elapsed > max_lost_seconds:
                        lost.append(uid)
                except:
                    if track.lost_seconds > max_lost_seconds:
                        lost.append(uid)
        return lost

    def increment_lost(self, used_active, elapsed_seconds):
        for uid in self.active:
            if uid not in used_active:
                self.active[uid].lost_seconds += elapsed_seconds

    def remove(self, uid):
        self.active.pop(uid, None)

    def serialize_state(self, tracks_dir):
        active_data = {}
        for uid, track in self.active.items():
            active_data[str(uid)] = track.serialize(tracks_dir, uid)
        return {
            "next_track_id": self.next_id,
            "active_tracks": active_data,
            "pending_next": self.pending_next
        }

    def restore_state(self, state_data, tracks_dir):
        self.next_id = state_data.get("next_track_id", 1)
        self.pending_next = state_data.get("pending_next", 0)
        self.pending = {}
        self.active = {}
        for uid_str, data in state_data.get("active_tracks", {}).items():
            try:
                self.active[int(uid_str)] = ActiveTrack.deserialize(data, tracks_dir)
            except Exception as e:
                logger.warning(f"Failed to restore track {uid_str}: {e}")


# =============================================================================
# FACE QUALITY SCORING
# =============================================================================

def score_face(face_crop, yolo_conf, keypoints=None):
    # Hard gates — only reject truly unusable faces
    if yolo_conf < 0.4:
        return 0, "low_yolo"
    face_h, face_w = face_crop.shape[:2]
    if face_w < 40 or face_h < 40:
        return 0, "too_small"

    gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
    blur = cv2.Laplacian(gray, cv2.CV_64F).var()
    brightness = gray.mean()

    # Soft scoring — let quality compete, don't reject outright
    scores = {}
    scores['yolo'] = yolo_conf * 20
    scores['blur'] = min(blur / 300, 1.0) * 25
    scores['size'] = min(min(face_w, face_h) / 120, 1.0) * 20
    scores['brightness'] = max(1 - abs(brightness - 130) / 130, 0) * 15

    if keypoints is not None:
        try:
            nose, leye, reye = keypoints[0], keypoints[1], keypoints[2]
            lear, rear = keypoints[3], keypoints[4]
            if leye[2] < 0.3 or reye[2] < 0.3 or nose[2] < 0.3:
                scores['pose'] = 0.3 * 20
            else:
                ld = abs(nose[0] - leye[0])
                rd = abs(nose[0] - reye[0])
                sym = min(ld, rd) / max(ld, rd) if max(ld, rd) > 0 else 0
                ear_pen = 0.8 if (lear[2] > 0.5 and rear[2] > 0.5) else 1.0
                scores['pose'] = sym * ear_pen * 20
        except:
            scores['pose'] = 10
    else:
        scores['pose'] = 10

    return round(sum(scores.values()), 2), scores


# =============================================================================
# CAPTURE-TIME FACE VALIDATION (ported from filter_faces.py)
# =============================================================================

def _check_skin_tone(face_crop, min_ratio=FILTER_MIN_SKIN):
    h, w = face_crop.shape[:2]
    cx1, cx2 = int(w * 0.2), int(w * 0.8)
    cy1, cy2 = int(h * 0.2), int(h * 0.8)
    center = face_crop[cy1:cy2, cx1:cx2]
    if center.size == 0:
        return False, 0.0
    hsv = cv2.cvtColor(center, cv2.COLOR_BGR2HSV)
    lower1 = np.array([0, 30, 60], dtype=np.uint8)
    upper1 = np.array([25, 180, 255], dtype=np.uint8)
    lower2 = np.array([160, 30, 60], dtype=np.uint8)
    upper2 = np.array([180, 180, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower1, upper1) | cv2.inRange(hsv, lower2, upper2)
    total = center.shape[0] * center.shape[1]
    ratio = cv2.countNonZero(mask) / total if total > 0 else 0.0
    return ratio >= min_ratio, round(ratio, 4)


def _parse_quality_from_filename(fname):
    """Extract quality score from face_XXXXXX_qNN.jpg. Returns int or -1."""
    m = re.search(r'_q(\d+)', fname)
    return int(m.group(1)) if m else -1


def _count_kps_in_bbox(kps, bbox, margin_frac=0.2):
    """InsightFace 5 landmarks have no per-point conf — validate by bbox containment."""
    if kps is None or len(kps) < 5:
        return 0
    x1, y1, x2, y2 = bbox
    fw, fh = x2 - x1, y2 - y1
    mx, my = fw * margin_frac, fh * margin_frac
    return int(sum(1 for kp in kps
                   if (x1 - mx <= kp[0] <= x2 + mx and y1 - my <= kp[1] <= y2 + my)))


def _move_to_rejected(folder, fname, reason):
    rej_dir = os.path.join(folder, "rejected")
    os.makedirs(rej_dir, exist_ok=True)
    src = os.path.join(folder, fname)
    dst = os.path.join(rej_dir, f"REJ_{reason[:60]}__{fname}")
    try:
        shutil.move(src, dst)
    except Exception:
        pass


# =============================================================================
# CLIP PROCESSOR
# =============================================================================

class FrameReader(threading.Thread):
    """Threaded frame reader — decodes frames in background while GPU processes."""

    def __init__(self, video_path, skip=1, max_queue=32):
        super().__init__(daemon=True)
        self.video_path = video_path
        self.skip = skip
        self.max_queue = max_queue
        self.queue = queue.Queue(maxsize=max_queue)
        self.stopped = False

    def run(self):
        cap = cv2.VideoCapture(self.video_path)
        idx = 0
        while cap.isOpened() and not self.stopped:
            ret, frame = cap.read()
            if not ret:
                break
            if idx % self.skip == 0:
                self.queue.put((idx, frame))
            idx += 1
        cap.release()
        self.queue.put(None)  # sentinel

    def stop(self):
        self.stopped = True


class ClipProcessor:
    def __init__(self):
        self.person_model = None
        self.face_model = None
        self.osnet = None
        self.tracker = None
        self.line_detector = None
        self.tracked_persons = {}
        self.frame_count = 0
        self.stats = {'entered': 0, 'faces_captured': 0, 'faces_rejected': 0}
        self._face_save_pool = ThreadPoolExecutor(max_workers=4)  # async face saves

    def initialize(self):
        import torch
        from ultralytics import YOLO
        yolo_path = os.path.join(SCRIPT_DIR, "models", "yolo11l.pt")
        face_path = os.path.join(SCRIPT_DIR, "models", "yolov11s-face.pt")

        logger.info(f"Loading YOLO person: {yolo_path}")
        self.person_model = YOLO(yolo_path)
        logger.info(f"Loading YOLO face: {face_path}")
        self.face_model = YOLO(face_path)

        # Force GPU + half precision for speed
        if torch.cuda.is_available():
            self.person_model.to("cuda")
            self.face_model.to("cuda")
            gpu_name = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
            logger.info(f"GPU: {gpu_name}")
            logger.info(f"VRAM: {vram:.1f} GB")

            # Try TensorRT export for max speed (first run takes a few minutes)
            if CONFIG.get("use_tensorrt", False):
                try:
                    trt_person = yolo_path.replace('.pt', '.engine')
                    trt_face = face_path.replace('.pt', '.engine')
                    if not os.path.exists(trt_person):
                        logger.info("Exporting person model to TensorRT (one-time)...")
                        self.person_model.export(format="engine", half=True)
                    if not os.path.exists(trt_face):
                        logger.info("Exporting face model to TensorRT (one-time)...")
                        self.face_model.export(format="engine", half=True)
                    from ultralytics import YOLO as _YOLO
                    if os.path.exists(trt_person):
                        self.person_model = _YOLO(trt_person)
                        logger.info(f"TensorRT person model loaded")
                    if os.path.exists(trt_face):
                        self.face_model = _YOLO(trt_face)
                        logger.info(f"TensorRT face model loaded")
                except Exception as e:
                    logger.warning(f"TensorRT export failed: {e}, using PyTorch")

        self.osnet = OSNetExtractor()
        self.tracker = PersonTracker(self.osnet)

        self.line_detector = None

    def restore_from_state(self, state, date_str):
        tracks_dir = os.path.join(data_dir(date_str), "tracks")
        self.tracker.restore_state(state, tracks_dir)
        self.tracked_persons = {}
        # Restore tracked persons from state
        for person in state.get("persons_in_progress", []):
            tid = person.get("track_id")
            if tid is not None:
                self.tracked_persons[tid] = person
        # Restore line detector state
        if self.line_detector and "line_sides" in state:
            self.line_detector.restore_state(state["line_sides"])
        logger.info(f"Restored: {len(self.tracker.active)} active tracks, next_id={self.tracker.next_id}")

    def process_clip(self, video_path, date_str, state):
        """Process a single video clip with threaded IO. Returns list of finalized persons."""
        clip_start = parse_video_timestamp(os.path.basename(video_path))
        if clip_start is None:
            logger.warning(f"Cannot parse timestamp from {video_path}, skipping")
            return []

        # Get video metadata
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.warning(f"Cannot open {video_path}")
            return []
        native_fps = cap.get(cv2.CAP_PROP_FPS) or EXPECTED_FPS
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        skip = max(1, int(native_fps / PROCESSING_FPS))
        faces_dir = os.path.join(data_dir(date_str), "faces")
        expected_process_frames = total_frames // skip

        logger.info(f"Processing: {os.path.basename(video_path)} | "
                    f"fps={native_fps:.0f}, frames={total_frames}, process={expected_process_frames}, "
                    f"skip={skip}, start={clip_start.strftime('%H:%M:%S')}")

        # Start threaded frame reader (decode in background)
        reader = FrameReader(video_path, skip=skip, max_queue=64)
        reader.start()

        finalized = []
        processed = 0
        t_start = time.time()

        # Profiling accumulators
        _t_read = 0.0
        _t_yolo = 0.0
        _t_osnet = 0.0
        _t_track = 0.0
        _t_face = 0.0
        _t_misc = 0.0
        _read_waits = 0

        while True:
            _p0 = time.time()
            qsize = reader.queue.qsize()
            item = reader.queue.get()
            _p1 = time.time()
            if qsize == 0:
                _read_waits += 1

            if item is None:
                break

            frame_idx, frame = item
            self.frame_count += 1
            processed += 1
            ts = frame_timestamp(clip_start, frame_idx, native_fps)
            h, w = frame.shape[:2]

            # YOLO person detection (GPU, half precision)
            _p2 = time.time()
            results = self.person_model.track(
                frame, classes=[0], conf=CONF_THRESHOLD,
                persist=True, verbose=False, half=True
            )
            _p3 = time.time()

            if not results or results[0].boxes is None or len(results[0].boxes) == 0:
                for uid in self.tracker.get_lost(ts):
                    p = self._finalize_person(uid, ts)
                    if p:
                        finalized.append(p)
                _t_read += (_p1 - _p0)
                _t_yolo += (_p3 - _p2)
                continue

            boxes = results[0].boxes.xyxy.cpu().numpy()

            _p4 = time.time()
            tracked = self.tracker.match_or_create(frame, boxes, ts)
            _p5 = time.time()

            face_targets = []
            for uid, box in tracked:
                cx = (box[0] + box[2]) / 2
                cy = box[3]
                nx, ny = cx / w, cy / h

                # Track every detected person (no line-crossing gate)
                if uid not in self.tracked_persons:
                    self._start_person(uid, box, ts, faces_dir)

                if uid in self.tracked_persons:
                    person = self.tracked_persons[uid]
                    person['last_seen'] = ts.isoformat()

                    x1, y1, x2, y2 = [int(v) for v in box]
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w, x2), min(h, y2)
                    crop = frame[y1:y2, x1:x2]

                    # Save body snapshot — first good one immediately, update at 2s for better quality
                    if crop.size > 0:
                        if not person['body_snapshot_saved']:
                            snap_path = os.path.join(person['images_folder'], "body_snapshot.jpg")
                            self._face_save_pool.submit(cv2.imwrite, snap_path, crop.copy())
                            person['body_snapshot'] = snap_path
                            person['body_snapshot_saved'] = True
                        elif not person.get('body_snapshot_updated'):
                            first_seen = datetime.fromisoformat(person['first_seen'])
                            elapsed_s = (ts - first_seen).total_seconds()
                            if elapsed_s >= 2.0:
                                self._face_save_pool.submit(cv2.imwrite, person['body_snapshot'], crop.copy())
                                person['body_snapshot_updated'] = True

                    if crop.size > 0:
                        face_targets.append((uid, crop, (x1, y1), box))

            _p6 = time.time()
            if face_targets:
                self._batch_capture_faces(face_targets, ts)
            _p7 = time.time()

            # Check lost
            for uid in self.tracker.get_lost(ts):
                p = self._finalize_person(uid, ts)
                if p:
                    finalized.append(p)
            _p8 = time.time()

            _t_read += (_p1 - _p0)
            _t_yolo += (_p3 - _p2)
            _t_osnet += (_p5 - _p4)
            _t_track += (_p6 - _p5)
            _t_face += (_p7 - _p6)
            _t_misc += (_p8 - _p7)

        reader.stop()
        elapsed = time.time() - t_start
        fps_actual = processed / elapsed if elapsed > 0 else 0

        _total = _t_read + _t_yolo + _t_osnet + _t_track + _t_face + _t_misc
        if _total > 0:
            logger.info(f"  PERF [{processed} frames, {elapsed:.1f}s, {fps_actual:.1f} fps]")
            logger.info(f"    Frame read/wait : {_t_read:6.2f}s  {100*_t_read/_total:4.1f}%  (queue empty {_read_waits}x)")
            logger.info(f"    YOLO person     : {_t_yolo:6.2f}s  {100*_t_yolo/_total:4.1f}%  [GPU]")
            logger.info(f"    OSNet re-ID     : {_t_osnet:6.2f}s  {100*_t_osnet/_total:4.1f}%  [GPU+CPU]")
            logger.info(f"    Track/crop      : {_t_track:6.2f}s  {100*_t_track/_total:4.1f}%  [CPU]")
            logger.info(f"    Face detect     : {_t_face:6.2f}s  {100*_t_face/_total:4.1f}%  [GPU]")
            logger.info(f"    Misc/finalize   : {_t_misc:6.2f}s  {100*_t_misc/_total:4.1f}%  [CPU]")
        logger.info(f"  Finalized {len(finalized)} persons | "
                    f"faces captured={self.stats['faces_captured']} "
                    f"rejected={self.stats['faces_rejected']}")
        return finalized

    def _batch_capture_faces(self, face_targets, ts):
        """Run face detection on person crops sequentially (TRT engine is batch=1)."""
        for uid, crop, offset, box in face_targets:
            if uid not in self.tracked_persons:
                continue
            person = self.tracked_persons[uid]

            # Cooldown: wait at least 1.5s between face captures for same person
            last_capture = person.get('_last_face_capture')
            if last_capture:
                elapsed = (ts - datetime.fromisoformat(last_capture)).total_seconds()
                if elapsed < 1.5:
                    continue

            # Skip face detection entirely if already have enough good faces
            if len(person['face_images']) >= MAX_FACES:
                min_score = min(person['face_images'], key=lambda x: x[0])[0]
                # Only try if we've waited long enough for a potentially better angle
                if last_capture:
                    elapsed = (ts - datetime.fromisoformat(last_capture)).total_seconds()
                    if elapsed < 3.0:
                        continue

            face_results = self.face_model(crop, conf=0.3, verbose=False, half=True)
            if not face_results or face_results[0].boxes is None or len(face_results[0].boxes) == 0:
                continue

            face_boxes = face_results[0].boxes
            best_idx = face_boxes.conf.argmax()
            yolo_conf = float(face_boxes.conf[best_idx])
            fx1, fy1, fx2, fy2 = face_boxes.xyxy[best_idx].cpu().numpy().astype(int)

            keypoints = None
            if face_results[0].keypoints is not None:
                try:
                    kps = face_results[0].keypoints[best_idx].data.cpu().numpy()[0]
                    if len(kps) >= 5:
                        keypoints = kps
                except:
                    pass

            pad = 10
            fx1 = max(0, fx1 - pad)
            fy1 = max(0, fy1 - pad)
            fx2 = min(crop.shape[1], fx2 + pad)
            fy2 = min(crop.shape[0], fy2 + pad)

            face_crop = crop[fy1:fy2, fx1:fx2]
            if face_crop.size == 0 or face_crop.shape[0] < 20 or face_crop.shape[1] < 20:
                continue

            quality, reason = score_face(face_crop, yolo_conf, keypoints)
            if quality == 0:
                continue

            if len(person['face_images']) >= MAX_FACES:
                min_score = min(person['face_images'], key=lambda x: x[0])[0]
                if quality <= min_score:
                    continue

            filename = f"face_{self.frame_count:06d}_q{quality:.0f}.jpg"
            filepath = os.path.join(person['images_folder'], filename)

            # Async file write — don't block GPU
            face_copy = face_crop.copy()
            self._face_save_pool.submit(cv2.imwrite, filepath, face_copy)

            person['face_images'].append((quality, filepath))
            person['_last_face_capture'] = ts.isoformat()
            self.stats['faces_captured'] += 1

            if len(person['face_images']) > MAX_FACES:
                person['face_images'].sort(key=lambda x: x[0], reverse=True)
                removed = person['face_images'].pop()
                self._face_save_pool.submit(lambda p: os.path.exists(p) and os.remove(p), removed[1])

            person['best_face_image'] = max(person['face_images'], key=lambda x: x[0])[1]

    def _start_person(self, track_id, box, ts, faces_dir):
        folder = os.path.join(faces_dir, f"person_{track_id:04d}")
        os.makedirs(folder, exist_ok=True)
        self.tracked_persons[track_id] = {
            'track_id': track_id,
            'first_seen': ts.isoformat(),
            'last_seen': ts.isoformat(),
            'crossing_timestamp': ts.isoformat(),
            'images_folder': folder,
            'face_images': [],
            'best_face_image': None,
            'body_snapshot': None,
            'body_snapshot_saved': False,
        }
        self.stats['entered'] += 1

    def _finalize_person(self, track_id, ts):
        if track_id not in self.tracked_persons:
            self.tracker.remove(track_id)
            if self.line_detector:
                self.line_detector.cleanup(track_id)
            return None

        person = self.tracked_persons.pop(track_id)
        person['last_seen'] = ts.isoformat()
        person['face_images_count'] = len(person['face_images'])
        self.tracker.remove(track_id)
        if self.line_detector:
            self.line_detector.cleanup(track_id)

        # Convert face_images to serializable format
        person['face_images'] = [(s, p) for s, p in person['face_images']]
        if person['face_images_count'] > 0:
            logger.info(f"  Finalized person {track_id}: {person['face_images_count']} faces")
        elif person.get('body_snapshot'):
            logger.info(f"  Finalized person {track_id}: no face, body snapshot saved")
        else:
            logger.info(f"  Finalized person {track_id}: no face, no body")
        return person

    def finalize_all(self, ts):
        """Finalize all remaining tracked persons."""
        finalized = []
        for uid in list(self.tracked_persons.keys()):
            p = self._finalize_person(uid, ts)
            if p:
                finalized.append(p)
        return finalized

    def get_save_state(self, date_str):
        """Get state for serialization."""
        tracks_dir = os.path.join(data_dir(date_str), "tracks")
        tracker_state = self.tracker.serialize_state(tracks_dir)
        # Save in-progress persons
        persons_in_progress = []
        for tid, p in self.tracked_persons.items():
            pp = dict(p)
            pp['face_images'] = [(s, path) for s, path in pp['face_images']]
            persons_in_progress.append(pp)
        tracker_state["persons_in_progress"] = persons_in_progress
        if self.line_detector:
            tracker_state["line_sides"] = self.line_detector.get_state()
        return tracker_state


# =============================================================================
# FR PROCESSOR (local, no API)
# =============================================================================

def run_fr_processing(date_str, state):
    """Run FR on all unprocessed persons in state."""
    persons = [p for p in state.get("persons", []) if not p.get("fr_processed")]
    if not persons:
        logger.info("No unprocessed persons for FR")
        return

    logger.info(f"FR processing {len(persons)} persons")

    # Load InsightFace
    from insightface.app import FaceAnalysis
    models_dir = os.path.join(SCRIPT_DIR, "insightface_models")
    providers = [("CUDAExecutionProvider", {}), ("CPUExecutionProvider", {})]
    face_app = FaceAnalysis(name="antelopev2", root=models_dir, providers=providers)
    face_app.prepare(ctx_id=0, det_size=(640, 640), det_thresh=0.3)

    # Load FAISS
    import faiss
    faiss_path = os.path.join(SCRIPT_DIR, "vector_db", "faiss.index")
    registry_path = os.path.join(SCRIPT_DIR, "vector_db", "staff_registry.json")

    faiss_index = None
    staff_registry = []
    if os.path.exists(faiss_path):
        faiss_index = faiss.read_index(faiss_path)
        if os.path.exists(registry_path):
            with open(registry_path, 'r') as f:
                staff_registry = json.load(f)
        logger.info(f"FAISS loaded: {faiss_index.ntotal} staff")

    threshold = FR_CFG.get("similarity_threshold", 0.95)
    cluster_threshold = FR_CFG.get("customer_cluster_threshold", 0.60)

    staff_by_name = defaultdict(list)
    customer_data = []
    no_face_list = []

    filter_stats = {'kept': 0, 'rej_quality': 0, 'rej_skin': 0,
                    'rej_no_face': 0, 'rej_kps': 0, 'rej_excess': 0}

    for i, person in enumerate(persons):
        pid = person.get('track_id', i)
        folder = person.get('images_folder', '')

        # Walk faces, validate (3 filters), embed, then keep top N by quality
        candidates = []  # list of (quality, emb, fname)
        if os.path.exists(folder):
            jpgs = sorted([f for f in os.listdir(folder)
                           if f.endswith('.jpg') and f != 'body_snapshot.jpg'
                           and not os.path.isdir(os.path.join(folder, f))])
            for jpg in jpgs:
                fpath = os.path.join(folder, jpg)
                q = _parse_quality_from_filename(jpg)

                # Filter 1: quality floor (cheapest)
                if POST_FILTER_ENABLED and 0 <= q < FILTER_MIN_QUALITY:
                    _move_to_rejected(folder, jpg, f"quality_{q}")
                    filter_stats['rej_quality'] += 1
                    continue

                img = cv2.imread(fpath)
                if img is None:
                    continue

                # Filter 2: skin-tone (cheap, no model)
                if POST_FILTER_ENABLED:
                    skin_ok, skin_ratio = _check_skin_tone(img, FILTER_MIN_SKIN)
                    if not skin_ok:
                        _move_to_rejected(folder, jpg, f"skin_{skin_ratio:.2f}")
                        filter_stats['rej_skin'] += 1
                        continue

                # InsightFace detect + embed (largest face)
                faces = face_app.get(img)
                if not faces:
                    if POST_FILTER_ENABLED:
                        _move_to_rejected(folder, jpg, "no_face")
                        filter_stats['rej_no_face'] += 1
                    continue
                f = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))

                # Filter 3: keypoints inside bbox
                if POST_FILTER_ENABLED:
                    kp_valid = _count_kps_in_bbox(f.kps, f.bbox)
                    if kp_valid < FILTER_MIN_KEYPOINTS:
                        _move_to_rejected(folder, jpg, f"kps_{kp_valid}")
                        filter_stats['rej_kps'] += 1
                        continue

                emb = f.embedding
                if emb is None:
                    continue
                norm = np.linalg.norm(emb)
                if norm > 0:
                    emb = emb / norm
                candidates.append((q if q >= 0 else 0, emb.astype(np.float32), jpg))

        # Keep top N by quality, move excess to rejected/
        candidates.sort(key=lambda x: -x[0])
        if POST_FILTER_ENABLED and len(candidates) > POST_FILTER_KEEP_TOP_N:
            for q, _, jpg in candidates[POST_FILTER_KEEP_TOP_N:]:
                _move_to_rejected(folder, jpg, f"excess_q{q}")
                filter_stats['rej_excess'] += 1
            candidates = candidates[:POST_FILTER_KEEP_TOP_N]

        all_embs = [e for _, e, _ in candidates]
        person['face_images_count'] = len(candidates)
        filter_stats['kept'] += len(candidates)

        # Refresh best_face_image to point at the highest-quality kept face
        if candidates:
            person['best_face_image'] = os.path.join(folder, candidates[0][2])

        if not all_embs:
            person['fr_processed'] = True
            person['update_type'] = 'no_face'
            no_face_list.append({
                'track_id': pid,
                'first_seen': person.get('first_seen', ''),
                'last_seen': person.get('last_seen', ''),
                'body_snapshot': person.get('body_snapshot'),
                'images_folder': person.get('images_folder', ''),
                'face_images_count': person.get('face_images_count', 0),
            })
            continue

        # Match against staff
        best_sim = -1.0
        best_name = None
        best_emb = all_embs[0]

        if faiss_index and faiss_index.ntotal > 0:
            for emb in all_embs:
                D, I = faiss_index.search(np.expand_dims(emb, 0).astype(np.float32), 1)
                sim = float(D[0][0])
                idx = int(I[0][0])
                if sim > best_sim:
                    best_sim = sim
                    if 0 <= idx < len(staff_registry):
                        best_name = staff_registry[idx]['name']
                    best_emb = emb

        if best_sim >= threshold and best_name:
            person['fr_processed'] = True
            person['recognized_name'] = best_name
            person['similarity_score'] = round(best_sim, 4)
            person['update_type'] = 'staff'
            staff_by_name[best_name].append(person)
            logger.info(f"  Person {pid}: STAFF — {best_name} ({best_sim:.4f})")
        else:
            person['fr_processed'] = True
            person['similarity_score'] = round(best_sim, 4)
            person['update_type'] = 'customer'
            customer_data.append((pid, best_emb, person))
            logger.info(f"  Person {pid}: CUSTOMER ({best_sim:.4f})")

    # Customer clustering
    clusters = []
    assigned = set()
    for i, (pid_i, emb_i, p_i) in enumerate(customer_data):
        if i in assigned:
            continue
        cluster = {
            'cluster_id': len(clusters) + 1,
            'track_ids': [p_i.get('track_id', 0)],
            'persons': [p_i],
            'embeddings': [emb_i],
        }
        assigned.add(i)
        for j, (pid_j, emb_j, p_j) in enumerate(customer_data):
            if j in assigned:
                continue
            max_sim = max(float(np.dot(emb_j, ce)) for ce in cluster['embeddings'])
            if max_sim >= cluster_threshold:
                cluster['track_ids'].append(p_j.get('track_id', 0))
                cluster['persons'].append(p_j)
                cluster['embeddings'].append(emb_j)
                assigned.add(j)

        fs = min((p.get('first_seen', '') for p in cluster['persons']), default='')
        ls = max((p.get('last_seen', '') for p in cluster['persons']), default='')
        cluster['first_seen'] = fs
        cluster['last_seen'] = ls
        cluster['visit_count'] = len(cluster['persons'])
        cluster['best_face'] = None
        for p in cluster['persons']:
            bf = p.get('best_face_image')
            if bf and os.path.exists(str(bf)):
                cluster['best_face'] = bf
                break
        # Per-track detections for report
        cluster['detections'] = [{
            'track_id': p.get('track_id'),
            'first_seen': p.get('first_seen'),
            'last_seen': p.get('last_seen'),
            'face_images_count': p.get('face_images_count', 0),
        } for p in cluster['persons']]
        del cluster['embeddings']
        del cluster['persons']
        clusters.append(cluster)

    clusters.sort(key=lambda c: -c['visit_count'])

    # Build staff results
    staff_results = []
    for name, plist in staff_by_name.items():
        fs = min((p.get('first_seen', '') for p in plist), default='')
        ls = max((p.get('last_seen', '') for p in plist), default='')
        staff_results.append({
            'name': name,
            'first_seen': fs,
            'last_seen': ls,
            'track_ids': [p.get('track_id', 0) for p in plist],
            'visit_count': len(plist),
            'best_face': next((p.get('best_face_image') for p in plist if p.get('best_face_image')), None),
            'detections': [{'track_id': p.get('track_id'), 'first_seen': p.get('first_seen'),
                           'last_seen': p.get('last_seen'),
                           'face_images_count': p.get('face_images_count', 0)} for p in plist],
        })

    state["fr_results"] = {
        "staff": staff_results,
        "customer_clusters": clusters,
        "no_face": no_face_list,
        "no_face_count": len(no_face_list)
    }

    logger.info(f"FR done: {len(staff_results)} staff, {len(clusters)} unique customers, {len(no_face_list)} no-face")
    if POST_FILTER_ENABLED:
        logger.info(f"  Face filter: kept={filter_stats['kept']} "
                    f"rej_quality={filter_stats['rej_quality']} "
                    f"rej_skin={filter_stats['rej_skin']} "
                    f"rej_no_face={filter_stats['rej_no_face']} "
                    f"rej_kps={filter_stats['rej_kps']} "
                    f"rej_excess={filter_stats['rej_excess']} "
                    f"(top_n={POST_FILTER_KEEP_TOP_N})")


# =============================================================================
# REPORT GENERATOR
# =============================================================================

def _parse_ts(s):
    if not s:
        return None
    try:
        return datetime.fromisoformat(s) if isinstance(s, str) else s
    except Exception:
        return None


def _hms(ts):
    dt = _parse_ts(ts)
    return dt.strftime("%H:%M:%S") if dt else "N/A"


def _duration_sec(fs, ls):
    a, b = _parse_ts(fs), _parse_ts(ls)
    if not a or not b:
        return 0
    return max(0, int((b - a).total_seconds()))


def _fmt_duration(sec):
    if sec <= 0:
        return "0s"
    m, s = divmod(int(sec), 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h}h {m}m {s}s"
    if m:
        return f"{m}m {s}s"
    return f"{s}s"


def _hour_bucket(ts):
    dt = _parse_ts(ts)
    return dt.hour if dt else None


def generate_report(date_str, state):
    try:
        from openpyxl import Workbook
        from openpyxl.styles import Font, Alignment, PatternFill, Border, Side
    except ImportError:
        logger.warning("openpyxl not installed, skipping report")
        return None

    fr = state.get("fr_results", {})
    staff_results = fr.get("staff", [])
    clusters = fr.get("customer_clusters", [])
    no_face_list = fr.get("no_face", [])
    no_face = fr.get("no_face_count", len(no_face_list))
    persons = state.get("persons", [])

    # Build track_id -> (type, label, cluster_id_or_name, best_face) lookup
    track_lookup = {}
    for s in staff_results:
        for d in s.get('detections', []):
            track_lookup[d['track_id']] = {
                'type': 'Staff',
                'label': s['name'],
                'best_face': s.get('best_face'),
            }
    for cl in clusters:
        label = f"Visitor #{cl['cluster_id']}" + (" (Repeated)" if cl['visit_count'] > 1 else "")
        for d in cl.get('detections', []):
            track_lookup[d['track_id']] = {
                'type': 'Customer',
                'label': label,
                'best_face': cl.get('best_face'),
            }
    for nf in no_face_list:
        tid = nf.get('track_id')
        if tid is not None and tid not in track_lookup:
            track_lookup[tid] = {
                'type': 'No-Face',
                'label': f"Unknown (ID {tid})",
                'best_face': nf.get('body_snapshot'),
            }

    wb = Workbook()
    hfont = Font(bold=True, size=12, color="FFFFFF")
    hfill = PatternFill(start_color="4472C4", end_color="4472C4", fill_type="solid")
    tfont = Font(bold=True, size=14)
    green = PatternFill(start_color="E2EFDA", end_color="E2EFDA", fill_type="solid")
    orange = PatternFill(start_color="FCE4D6", end_color="FCE4D6", fill_type="solid")
    blue = PatternFill(start_color="DDEBF7", end_color="DDEBF7", fill_type="solid")
    gray = PatternFill(start_color="EDEDED", end_color="EDEDED", fill_type="solid")
    border = Border(left=Side(style='thin'), right=Side(style='thin'),
                    top=Side(style='thin'), bottom=Side(style='thin'))
    center = Alignment(horizontal='center', vertical='center', wrap_text=True)

    def write_header(ws, row, headers):
        for col, h in enumerate(headers, 1):
            c = ws.cell(row=row, column=col, value=h)
            c.font, c.fill, c.border, c.alignment = hfont, hfill, border, center

    def put(ws, r, c, v, fill=None):
        cell = ws.cell(row=r, column=c, value=v)
        cell.alignment = center
        cell.border = border
        if fill:
            cell.fill = fill
        return cell

    def add_face(ws, path, cell):
        if not path or not os.path.exists(str(path)):
            return
        try:
            from openpyxl.drawing.image import Image as XLImage
            from PIL import Image as PILImage
            img = PILImage.open(path)
            img.thumbnail((80, 80), PILImage.Resampling.LANCZOS)
            thumb = path + ".thumb.jpg"
            img.save(thumb, "JPEG", quality=85)
            xl = XLImage(thumb)
            xl.width, xl.height = 80, 80
            ws.add_image(xl, cell)
        except:
            pass

    thumbs_to_clean = set()

    # ─────────────────────────────────────────────────────────
    # SHEET 1: Track IDs (flat list — every tracked person)
    # ─────────────────────────────────────────────────────────
    ws = wb.active
    ws.title = "Track IDs"
    ws.merge_cells('A1:I1')
    ws['A1'].value = f"All Track IDs - {date_str}"
    ws['A1'].font, ws['A1'].alignment = tfont, center
    write_header(ws, 3, ['S.No', 'Face', 'Track ID', 'Type', 'Label',
                         'First Seen', 'Last Seen', 'Duration', 'Faces'])
    widths = {'A': 6, 'B': 14, 'C': 10, 'D': 12, 'E': 22,
              'F': 14, 'G': 14, 'H': 14, 'I': 8}
    for col, w in widths.items():
        ws.column_dimensions[col].width = w

    sorted_persons = sorted(persons, key=lambda p: _parse_ts(p.get('first_seen')) or datetime.min)
    row = 4
    for idx, p in enumerate(sorted_persons, 1):
        tid = p.get('track_id')
        lookup = track_lookup.get(tid, {'type': 'Unknown', 'label': f'ID {tid}', 'best_face': None})
        fs, ls = p.get('first_seen'), p.get('last_seen')
        dur = _duration_sec(fs, ls)
        ws.row_dimensions[row].height = 70

        fill = green if lookup['type'] == 'Staff' else (
            orange if lookup['type'] == 'Customer' else gray)

        put(ws, row, 1, idx, fill)
        add_face(ws, lookup.get('best_face'), f"B{row}")
        put(ws, row, 2, None, fill)  # face cell background
        put(ws, row, 3, tid, fill)
        put(ws, row, 4, lookup['type'], fill)
        put(ws, row, 5, lookup['label'], fill)
        put(ws, row, 6, _hms(fs), fill)
        put(ws, row, 7, _hms(ls), fill)
        put(ws, row, 8, _fmt_duration(dur), fill)
        put(ws, row, 9, p.get('face_images_count', 0), fill)
        if lookup.get('best_face'):
            thumbs_to_clean.add(str(lookup['best_face']) + ".thumb.jpg")
        row += 1

    # ─────────────────────────────────────────────────────────
    # SHEET 2: Staff Log
    # ─────────────────────────────────────────────────────────
    ws = wb.create_sheet("Staff Log")
    ws.merge_cells('A1:H1')
    ws['A1'].value = f"Staff Log - {date_str}"
    ws['A1'].font, ws['A1'].alignment = tfont, center
    write_header(ws, 3, ['S.No', 'Face', 'Name', 'First Detected', 'Last Detected',
                         'Times', 'Total Duration', 'Detection Log'])
    widths = {'A': 6, 'B': 14, 'C': 18, 'D': 16, 'E': 16, 'F': 8, 'G': 16, 'H': 50}
    for col, w in widths.items():
        ws.column_dimensions[col].width = w

    row = 4
    for idx, s in enumerate(staff_results, 1):
        ws.row_dimensions[row].height = 80
        total_dur = sum(_duration_sec(d.get('first_seen'), d.get('last_seen'))
                        for d in s.get('detections', []))
        put(ws, row, 1, idx, green)
        add_face(ws, s.get('best_face'), f"B{row}")
        put(ws, row, 2, None, green)
        put(ws, row, 3, s['name'], green)
        put(ws, row, 4, _hms(s.get('first_seen')), green)
        put(ws, row, 5, _hms(s.get('last_seen')), green)
        put(ws, row, 6, s.get('visit_count', 0), green)
        put(ws, row, 7, _fmt_duration(total_dur), green)
        log_lines = []
        for d in s.get('detections', []):
            dur = _fmt_duration(_duration_sec(d.get('first_seen'), d.get('last_seen')))
            log_lines.append(f"ID {d.get('track_id')}: {_hms(d.get('first_seen'))} → {_hms(d.get('last_seen'))} ({dur}, {d.get('face_images_count', 0)} faces)")
        put(ws, row, 8, "\n".join(log_lines), green)
        if s.get('best_face'):
            thumbs_to_clean.add(str(s['best_face']) + ".thumb.jpg")
        row += 1

    # ─────────────────────────────────────────────────────────
    # SHEET 3: Customer Analysis
    # ─────────────────────────────────────────────────────────
    ws = wb.create_sheet("Customer Analysis")
    ws.merge_cells('A1:I1')
    ws['A1'].value = f"Customer Analysis - {date_str}"
    ws['A1'].font, ws['A1'].alignment = tfont, center
    write_header(ws, 3, ['S.No', 'Face', 'Visitor', 'Times Seen', 'Track IDs',
                         'First', 'Last', 'Total Duration', 'Detection Log'])
    widths = {'A': 6, 'B': 14, 'C': 20, 'D': 10, 'E': 22,
              'F': 14, 'G': 14, 'H': 14, 'I': 50}
    for col, w in widths.items():
        ws.column_dimensions[col].width = w

    row = 4
    for idx, cl in enumerate(clusters, 1):
        ws.row_dimensions[row].height = 80
        label = f"Visitor #{cl['cluster_id']}"
        if cl['visit_count'] > 1:
            label += " (Repeated)"
        fill = orange if cl['visit_count'] > 1 else blue
        total_dur = sum(_duration_sec(d.get('first_seen'), d.get('last_seen'))
                        for d in cl.get('detections', []))

        put(ws, row, 1, idx, fill)
        add_face(ws, cl.get('best_face'), f"B{row}")
        put(ws, row, 2, None, fill)
        put(ws, row, 3, label, fill)
        put(ws, row, 4, cl['visit_count'], fill)
        put(ws, row, 5, ", ".join(str(t) for t in cl.get('track_ids', [])), fill)
        put(ws, row, 6, _hms(cl.get('first_seen')), fill)
        put(ws, row, 7, _hms(cl.get('last_seen')), fill)
        put(ws, row, 8, _fmt_duration(total_dur), fill)
        log_lines = []
        for d in cl.get('detections', []):
            dur = _fmt_duration(_duration_sec(d.get('first_seen'), d.get('last_seen')))
            log_lines.append(f"ID {d.get('track_id')}: {_hms(d.get('first_seen'))} → {_hms(d.get('last_seen'))} ({dur}, {d.get('face_images_count', 0)} faces)")
        put(ws, row, 9, "\n".join(log_lines), fill)
        if cl.get('best_face'):
            thumbs_to_clean.add(str(cl['best_face']) + ".thumb.jpg")
        row += 1

    # ─────────────────────────────────────────────────────────
    # SHEET 4: No-Face Persons
    # ─────────────────────────────────────────────────────────
    ws = wb.create_sheet("No-Face Persons")
    ws.merge_cells('A1:G1')
    ws['A1'].value = f"No-Face Persons - {date_str}"
    ws['A1'].font, ws['A1'].alignment = tfont, center
    write_header(ws, 3, ['S.No', 'Body', 'Track ID', 'First Seen', 'Last Seen',
                         'Duration', 'Folder'])
    widths = {'A': 6, 'B': 14, 'C': 10, 'D': 14, 'E': 14, 'F': 14, 'G': 40}
    for col, w in widths.items():
        ws.column_dimensions[col].width = w

    row = 4
    sorted_nf = sorted(no_face_list, key=lambda n: _parse_ts(n.get('first_seen')) or datetime.min)
    for idx, nf in enumerate(sorted_nf, 1):
        ws.row_dimensions[row].height = 70
        dur = _duration_sec(nf.get('first_seen'), nf.get('last_seen'))
        put(ws, row, 1, idx, gray)
        add_face(ws, nf.get('body_snapshot'), f"B{row}")
        put(ws, row, 2, None, gray)
        put(ws, row, 3, nf.get('track_id'), gray)
        put(ws, row, 4, _hms(nf.get('first_seen')), gray)
        put(ws, row, 5, _hms(nf.get('last_seen')), gray)
        put(ws, row, 6, _fmt_duration(dur), gray)
        put(ws, row, 7, nf.get('images_folder', ''), gray)
        if nf.get('body_snapshot'):
            thumbs_to_clean.add(str(nf['body_snapshot']) + ".thumb.jpg")
        row += 1

    # ─────────────────────────────────────────────────────────
    # SHEET 5: Hourly Footfall
    # ─────────────────────────────────────────────────────────
    ws = wb.create_sheet("Hourly Footfall")
    ws.merge_cells('A1:E1')
    ws['A1'].value = f"Hourly Footfall - {date_str}"
    ws['A1'].font, ws['A1'].alignment = tfont, center
    write_header(ws, 3, ['Hour', 'Staff Detections', 'Customer Detections',
                         'No-Face', 'Total'])
    for col, w in {'A': 10, 'B': 18, 'C': 20, 'D': 12, 'E': 10}.items():
        ws.column_dimensions[col].width = w

    buckets = {h: {'staff': 0, 'customer': 0, 'no_face': 0} for h in range(24)}
    for s in staff_results:
        for d in s.get('detections', []):
            h = _hour_bucket(d.get('first_seen'))
            if h is not None:
                buckets[h]['staff'] += 1
    for cl in clusters:
        for d in cl.get('detections', []):
            h = _hour_bucket(d.get('first_seen'))
            if h is not None:
                buckets[h]['customer'] += 1
    for nf in no_face_list:
        h = _hour_bucket(nf.get('first_seen'))
        if h is not None:
            buckets[h]['no_face'] += 1

    row = 4
    peak_hour, peak_total = None, 0
    for h in range(24):
        b = buckets[h]
        total = b['staff'] + b['customer'] + b['no_face']
        if total == 0:
            continue
        if total > peak_total:
            peak_total, peak_hour = total, h
        put(ws, row, 1, f"{h:02d}:00")
        put(ws, row, 2, b['staff'])
        put(ws, row, 3, b['customer'])
        put(ws, row, 4, b['no_face'])
        put(ws, row, 5, total)
        row += 1

    # ─────────────────────────────────────────────────────────
    # SHEET 6: Summary
    # ─────────────────────────────────────────────────────────
    ws = wb.create_sheet("Summary")
    ws.merge_cells('A1:B1')
    ws['A1'].value = f"Summary - {date_str}"
    ws['A1'].font, ws['A1'].alignment = tfont, center
    ws.column_dimensions['A'].width = 32
    ws.column_dimensions['B'].width = 40

    durations = [_duration_sec(p.get('first_seen'), p.get('last_seen')) for p in persons]
    durations = [d for d in durations if d > 0]
    avg_dur = sum(durations) / len(durations) if durations else 0
    total_dur = sum(durations)
    repeated = [c for c in clusters if c['visit_count'] > 1]
    first_ts = min((_parse_ts(p.get('first_seen')) for p in persons if _parse_ts(p.get('first_seen'))),
                   default=None)
    last_ts = max((_parse_ts(p.get('last_seen')) for p in persons if _parse_ts(p.get('last_seen'))),
                  default=None)

    data = [
        ("Metric", "Value"),
        ("Date", date_str),
        ("Store", STORE_NAME),
        ("Clips Processed", len(state.get("processed_clips", []))),
        ("Total Track IDs", len(persons)),
        ("Staff (unique)", len(staff_results)),
        ("Unique Customers", len(clusters)),
        ("Repeated Visitors", len(repeated)),
        ("No-Face Persons", no_face),
        ("Avg Dwell Time", _fmt_duration(avg_dur)),
        ("Total Tracked Time", _fmt_duration(total_dur)),
        ("First Detection", _hms(first_ts) if first_ts else "N/A"),
        ("Last Detection", _hms(last_ts) if last_ts else "N/A"),
        ("Peak Hour", f"{peak_hour:02d}:00 ({peak_total} detections)" if peak_hour is not None else "N/A"),
    ]
    row = 3
    for m, v in data:
        c1 = ws.cell(row=row, column=1, value=m)
        c2 = ws.cell(row=row, column=2, value=v)
        c1.border = c2.border = border
        if row == 3:
            c1.font = c2.font = hfont
            c1.fill = c2.fill = hfill
        row += 1

    # Save
    report_dir = os.path.join(data_dir(date_str), "report")
    path = os.path.join(report_dir, f"fr_report_{date_str}.xlsx")
    wb.save(path)
    logger.info(f"Report saved: {path}")

    # Cleanup thumbnails
    for t in thumbs_to_clean:
        try:
            if os.path.exists(t):
                os.remove(t)
        except Exception:
            pass

    return path


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def process_day(date_str, local_dir=None, run_fr=True):
    """Process all clips for a single day."""
    state = load_state(date_str)
    dd = data_dir(date_str)

    # Fetch or use local videos
    if local_dir:
        videos_dir = local_dir
    else:
        new_clips = fetch_videos(date_str, state)
        if new_clips:
            logger.info(f"Fetched {len(new_clips)} new clips")
        videos_dir = os.path.join(dd, "videos")

    # Get all clips sorted chronologically
    all_clips = sorted([f for f in os.listdir(videos_dir) if f.endswith('.mp4')])
    processed = set(state.get("processed_clips", []))
    new_clips = [f for f in all_clips if f not in processed]

    if not new_clips:
        logger.info(f"No new clips to process for {date_str}")
        return state

    logger.info(f"Processing {len(new_clips)} new clips for {date_str}")

    # Initialize processor
    processor = ClipProcessor()
    processor.initialize()

    # Restore state from previous processing
    if state.get("active_tracks"):
        processor.restore_from_state(state, date_str)

    # Process each clip in order
    for clip_name in new_clips:
        clip_path = os.path.join(videos_dir, clip_name)

        # Check gap from previous clip
        if state.get("processed_clips"):
            prev_clip = state["processed_clips"][-1]
            prev_ts = parse_video_timestamp(prev_clip)
            curr_ts = parse_video_timestamp(clip_name)
            if prev_ts and curr_ts:
                # Assume previous clip was ~60s
                prev_end = prev_ts + timedelta(seconds=60)
                gap = (curr_ts - prev_end).total_seconds()
                if gap > MAX_LOST_SECONDS:
                    logger.info(f"  Gap of {gap:.0f}s > {MAX_LOST_SECONDS}s, finalizing all tracks")
                    finalized = processor.finalize_all(prev_end)
                    for p in finalized:
                        state["persons"].append(p)

        # Process the clip
        finalized = processor.process_clip(clip_path, date_str, state)
        for p in finalized:
            state["persons"].append(p)

        state["processed_clips"].append(clip_name)

        # Save tracker state
        tracker_state = processor.get_save_state(date_str)
        state["next_track_id"] = tracker_state["next_track_id"]
        state["active_tracks"] = tracker_state["active_tracks"]

        save_state(date_str, state)

    # Finalize remaining if this is a one-shot or end of day
    last_clip = new_clips[-1]
    last_ts = parse_video_timestamp(last_clip)
    if last_ts:
        end_ts = last_ts + timedelta(seconds=60)
        finalized = processor.finalize_all(end_ts)
        for p in finalized:
            state["persons"].append(p)
        save_state(date_str, state)

    # Run FR
    if run_fr:
        try:
            run_fr_processing(date_str, state)
            save_state(date_str, state)
        except Exception as e:
            logger.warning(f"FR processing failed: {e}")

    # Generate report
    try:
        generate_report(date_str, state)
    except Exception as e:
        logger.warning(f"Report generation failed: {e}")

    save_state(date_str, state)

    # Print summary
    fr = state.get("fr_results", {})
    print(f"\n{'='*60}")
    print(f"PROCESSING SUMMARY — {date_str}")
    print(f"{'='*60}")
    print(f"Clips processed: {len(state.get('processed_clips', []))}")
    print(f"Total persons: {len(state.get('persons', []))}")
    print(f"Staff: {len(fr.get('staff', []))}")
    print(f"Unique customers: {len(fr.get('customer_clusters', []))}")
    print(f"No face: {fr.get('no_face_count', 0)}")
    print(f"Report: data/{date_str}/report/fr_report_{date_str}.xlsx")
    print(f"{'='*60}")

    return state


def main():
    parser = argparse.ArgumentParser(description="Offline FR Pipeline")
    parser.add_argument("--once", action="store_true", help="Process once and exit")
    parser.add_argument("--date", "-d", default=None, help="Date YYYY-MM-DD (default: today)")
    parser.add_argument("--local-dir", "-l", default=None, help="Local video directory (skip fetch)")
    parser.add_argument("--no-fr", action="store_true", help="Skip FR processing")

    args = parser.parse_args()
    date_str = args.date or datetime.now().strftime("%Y-%m-%d")

    logger.info("=" * 50)
    logger.info(f"Offline FR Pipeline v{VERSION}")
    logger.info(f"Date: {date_str}")
    if args.local_dir:
        logger.info(f"Local dir: {args.local_dir}")
    else:
        logger.info(f"Remote: {REC_CFG.get('user')}@{REC_CFG.get('host')}:{REC_CFG.get('video_base_path')}")
    logger.info("=" * 50)

    if args.once or args.local_dir:
        process_day(date_str, local_dir=args.local_dir, run_fr=not args.no_fr)
    else:
        # Continuous service mode
        running = True
        def stop(sig, frame):
            nonlocal running
            running = False
            logger.info("Stopping...")
        signal.signal(signal.SIGINT, stop)
        signal.signal(signal.SIGTERM, stop)

        poll_interval = REC_CFG.get("poll_interval_seconds", 30)
        logger.info(f"Service mode: polling every {poll_interval}s")

        while running:
            today = datetime.now().strftime("%Y-%m-%d")
            try:
                process_day(today, run_fr=not args.no_fr)
            except Exception as e:
                logger.warning(f"Pipeline error: {e}")
            time.sleep(poll_interval)

        logger.info("Pipeline stopped")


if __name__ == "__main__":
    main()
