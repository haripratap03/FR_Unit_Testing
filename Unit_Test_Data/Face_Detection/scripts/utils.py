"""
Shared utilities for FD real-footage test scripts.
All extraction uses YOLO face model on real CCTV footage.
"""
import os, sys, json, glob
import numpy as np
import cv2

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_BASE   = os.path.abspath(os.path.join(SCRIPTS_DIR, '..', 'data'))
REPO_ROOT   = os.path.abspath(os.path.join(SCRIPTS_DIR, '..', '..', '..'))
VIDEOS_DIR  = os.path.join(REPO_ROOT, 'data', '2026-04-29', 'videos', 'videos')
FACES_DIR   = os.path.join(REPO_ROOT, 'data', '2026-04-29', 'faces')
MODELS_DIR  = os.path.join(REPO_ROOT, 'models')
INSIGHT_DIR = os.path.join(REPO_ROOT, 'insightface_models')
YOLO_FACE   = os.path.join(MODELS_DIR, 'yolov11s-face.pt')

sys.path.insert(0, REPO_ROOT)


# ── paths ──────────────────────────────────────────────────────────────────────

def get_videos():
    return sorted(glob.glob(os.path.join(VIDEOS_DIR, '*.mp4')))


def data_path(test_id, filename=''):
    d = os.path.join(DATA_BASE, test_id)
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, filename) if filename else d


# ── model loaders ──────────────────────────────────────────────────────────────

def load_yolo_face():
    """Load YOLO face model. Returns model or None."""
    try:
        from ultralytics import YOLO
        if not os.path.exists(YOLO_FACE):
            print(f"[WARN] YOLO face model not found: {YOLO_FACE}")
            return None
        print("[INFO] Loading YOLO face model...")
        return YOLO(YOLO_FACE)
    except Exception as e:
        print(f"[WARN] YOLO load failed: {e}")
        return None


def load_insightface(det_thresh=0.15):
    """Load InsightFace FaceAnalysis app. Returns app or None."""
    try:
        from insightface.app import FaceAnalysis
        providers = [("CUDAExecutionProvider", {}), ("CPUExecutionProvider", {})]
        app = FaceAnalysis(name="antelopev2", root=INSIGHT_DIR, providers=providers)
        app.prepare(ctx_id=0, det_thresh=det_thresh, det_size=(640, 640))
        print("[INFO] InsightFace loaded")
        return app
    except Exception as e:
        print(f"[WARN] InsightFace load failed: {e}")
        return None


# ── video helpers ──────────────────────────────────────────────────────────────

def iter_frames(video_path, step=5, limit=None):
    """Yield (frame_idx, frame) every `step` frames."""
    cap = cv2.VideoCapture(video_path)
    idx = n = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if idx % step == 0:
            yield idx, frame
            n += 1
            if limit and n >= limit:
                break
        idx += 1
    cap.release()


def yolo_detections(model, frame, conf=0.20):
    """Run YOLO face detection. Returns list of {crop, conf, kps, bbox, frame}."""
    r = model(frame, conf=conf, verbose=False, half=False)
    out = []
    if not r or r[0].boxes is None:
        return out
    boxes, kp_data = r[0].boxes, r[0].keypoints
    h, w = frame.shape[:2]
    for i in range(len(boxes)):
        x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy().astype(int)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        crop = frame[y1:y2, x1:x2].copy()
        if crop.size == 0 or min(crop.shape[:2]) < 8:
            continue
        kps = None
        if kp_data is not None:
            try:
                kps = kp_data[i].data.cpu().numpy()[0]
            except Exception:
                pass
        out.append({
            'crop': crop, 'conf': float(boxes.conf[i]),
            'kps': kps, 'bbox': (x1, y1, x2, y2), 'frame': frame
        })
    return out


def scan_faces(model, max_faces=400, step=15, n_vids=5, *, validate=True):
    """Scan first n_vids videos for face detections.

    validate=True (default): rejects non-human-face YOLO false positives
    (bikes, vehicle parts, walls) using skin-tone + InsightFace verification.
    Adds ~50–100 ms per candidate but prevents bike crops landing as test data.
    Pass validate=False for raw YOLO output (for tests that need any detection).
    """
    app = _get_validator_app() if validate else None
    all_d = []
    for vp in get_videos()[:n_vids]:
        for _, frame in iter_frames(vp, step=step):
            new_dets = yolo_detections(model, frame)
            if validate:
                new_dets = [d for d in new_dets if is_human_face(d['crop'], app=app)]
            all_d.extend(new_dets)
            if len(all_d) >= max_faces:
                return all_d
    return all_d


# ── human-face validation (anti-bike-false-positive) ──────────────────────────

_INSIGHTFACE_APP = None  # cached lazy-loaded InsightFace app


def _get_validator_app():
    """Lazy-load InsightFace once for face validation. Returns None if unavailable."""
    global _INSIGHTFACE_APP
    if _INSIGHTFACE_APP is None:
        _INSIGHTFACE_APP = load_insightface(det_thresh=0.30)
    return _INSIGHTFACE_APP


def _skin_ratio(crop):
    """HSV skin pixel fraction over center 60% — mirrors pipeline._check_skin_tone."""
    h, w = crop.shape[:2]
    if h < 4 or w < 4:
        return 0.0
    cy, cx = h // 2, w // 2
    dy, dx = max(int(h * 0.30), 1), max(int(w * 0.30), 1)
    center = crop[cy - dy:cy + dy, cx - dx:cx + dx]
    if center.size == 0:
        return 0.0
    hsv = cv2.cvtColor(center, cv2.COLOR_BGR2HSV)
    m1  = cv2.inRange(hsv, (0,   20, 70), (20,  255, 255))
    m2  = cv2.inRange(hsv, (160, 20, 70), (180, 255, 255))
    skin = cv2.bitwise_or(m1, m2)
    return float((skin > 0).sum()) / float(skin.size)


def is_human_face(crop, app=None, min_skin=0.08):
    """Validate a YOLO face detection is actually a human face (not a vehicle/wall).

    Two-stage check:
      1. Skin-tone HSV ratio in center 60% — fast, cheap reject for bikes/walls
      2. InsightFace.get() — confirms a real face landmark set is present

    Returns True only if both checks pass. If InsightFace is unavailable, the
    skin-tone result is used alone (skin-tone alone catches the vehicle case).
    """
    if crop is None or crop.size == 0 or min(crop.shape[:2]) < 16:
        return False
    if _skin_ratio(crop) < min_skin:
        return False
    if app is None:
        app = _get_validator_app()
    if app is None:
        return True
    try:
        return len(app.get(crop)) > 0
    except Exception:
        return True


def validate_face_dets(dets, app=None):
    """Filter a list of YOLO detections to only those validated as human faces."""
    if app is None:
        app = _get_validator_app()
    return [d for d in dets if is_human_face(d['crop'], app=app)]


# ── face selectors ─────────────────────────────────────────────────────────────

def lap_var(img):
    return cv2.Laplacian(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()


def brightness(img):
    return float(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).mean())


def best_face(dets):
    """Sharpest, largest, best-lit detection."""
    def sc(d):
        g = cv2.cvtColor(d['crop'], cv2.COLOR_BGR2GRAY)
        blur = cv2.Laplacian(g, cv2.CV_64F).var()
        size = min(d['crop'].shape[:2])
        bright = max(1 - abs(g.mean() - 130) / 130, 0)
        return blur * (size / 100) * bright
    return max(dets, key=sc)


def find_profile_face(dets):
    """Most asymmetric keypoints — lowest symmetry score (profile view)."""
    best, best_asym = None, -1.0
    for d in dets:
        kps = d.get('kps')
        if kps is None or len(kps) < 3:
            continue
        nose, leye, reye = kps[0], kps[1], kps[2]
        if nose[2] < 0.3 or leye[2] < 0.3 or reye[2] < 0.3:
            continue
        ld = abs(nose[0] - leye[0])
        rd = abs(nose[0] - reye[0])
        if max(ld, rd) == 0:
            continue
        sym = min(ld, rd) / max(ld, rd)
        if (1 - sym) > best_asym:
            best_asym = 1 - sym
            best = d
    return best


def find_both_ears_face(dets):
    """First detection where both ear kp confidences > 0.5."""
    for d in dets:
        kps = d.get('kps')
        if kps is not None and len(kps) >= 5:
            if kps[3][2] > 0.5 and kps[4][2] > 0.5:
                return d
    return None


def find_dark_face(dets, max_bright=80):
    """Darkest face below max_bright threshold."""
    cands = [d for d in dets if brightness(d['crop']) < max_bright]
    return min(cands, key=lambda d: brightness(d['crop'])) if cands else None


def _is_frontal(d, sym_tol=0.55):
    """True if detection has a roughly frontal pose: nose between eyes, eyes balanced."""
    kps = d.get('kps')
    if kps is None or len(kps) < 3:
        return False
    nose, leye, reye = kps[0], kps[1], kps[2]
    if nose[2] < 0.3 or leye[2] < 0.3 or reye[2] < 0.3:
        return False
    ld = abs(nose[0] - leye[0])
    rd = abs(nose[0] - reye[0])
    if max(ld, rd) == 0:
        return False
    sym = min(ld, rd) / max(ld, rd)
    return sym >= sym_tol


def find_frontal_dark_face(dets, max_bright=110, sym_tol=0.55):
    """Darkest face among frontal candidates. Falls back to overall darkest if none."""
    frontal = [d for d in dets if _is_frontal(d, sym_tol=sym_tol)]
    cands = [d for d in frontal if brightness(d['crop']) < max_bright]
    if cands:
        return min(cands, key=lambda d: brightness(d['crop']))
    if frontal:
        return min(frontal, key=lambda d: brightness(d['crop']))
    return None


def find_bright_face(dets, min_bright=200):
    """Brightest face above min_bright threshold."""
    cands = [d for d in dets if brightness(d['crop']) > min_bright]
    return max(cands, key=lambda d: brightness(d['crop'])) if cands else None


def find_small_face(dets, max_px=50):
    """Smallest face detection."""
    cands = [d for d in dets if max(d['crop'].shape[:2]) <= max_px]
    if not cands:
        return min(dets, key=lambda d: max(d['crop'].shape[:2]))
    return min(cands, key=lambda d: max(d['crop'].shape[:2]))


def find_empty_frame(model, n_vids=3, step=15, *, validate=True):
    """Find a frame with zero face detections (after optional human-face validation)."""
    app = _get_validator_app() if validate else None
    for vp in get_videos()[:n_vids]:
        for _, frame in iter_frames(vp, step=step, limit=200):
            dets = yolo_detections(model, frame)
            if validate:
                dets = [d for d in dets if is_human_face(d['crop'], app=app)]
            if len(dets) == 0:
                return frame
    return None


def find_multi_face_frame(model, n_vids=5, step=15, *, validate=True):
    """Find a frame with 2+ simultaneous validated human-face detections."""
    app = _get_validator_app() if validate else None
    for vp in get_videos()[:n_vids]:
        for _, frame in iter_frames(vp, step=step, limit=500):
            dets = yolo_detections(model, frame)
            if validate:
                dets = [d for d in dets if is_human_face(d['crop'], app=app)]
            if len(dets) >= 2:
                return frame, dets
    return None, None


def find_blurry_face(dets, max_lap_var=80):
    """Blurriest face detection (lowest Laplacian variance)."""
    if not dets:
        return None
    cands = [d for d in dets if lap_var(d['crop']) <= max_lap_var]
    if cands:
        return min(cands, key=lambda d: lap_var(d['crop']))
    return min(dets, key=lambda d: lap_var(d['crop']))


def is_clean_nonface_crop(crop, app=None, max_skin=0.18):
    """Validate a crop is non-face AND non-skin: skin_ratio low and InsightFace finds nothing."""
    if crop is None or crop.size == 0:
        return False
    if _skin_ratio(crop) >= max_skin:
        return False
    if app is None:
        app = _get_validator_app()
    if app is None:
        return True
    try:
        return len(app.get(crop)) == 0
    except Exception:
        return True


def synth_nonface_crop(size=160):
    """Pure-blue uniform crop: 0% skin (HSV hue ~120) and no facial structure."""
    return np.full((size, size, 3), [255, 0, 0], dtype=np.uint8)


def find_low_conf_kp_face(dets):
    """Face where at least one of nose/leye/reye kp confs is < 0.3."""
    for d in dets:
        kps = d.get('kps')
        if kps is not None and len(kps) >= 3:
            if any(kps[i][2] < 0.3 for i in range(3)):
                return d
    return None


# ── I/O helpers ────────────────────────────────────────────────────────────────

def save_img(test_id, filename, img):
    p = data_path(test_id, filename)
    cv2.imwrite(p, img)
    return p


def save_json(test_id, filename, data):
    def fix(o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.float32, np.float64)):
            return float(o)
        if isinstance(o, (np.int32, np.int64)):
            return int(o)
        return o
    p = data_path(test_id, filename)
    with open(p, 'w') as f:
        json.dump(data, f, default=fix, indent=2)
    return p


def load_img(test_id, filename):
    p = data_path(test_id, filename)
    return cv2.imread(p) if os.path.exists(p) else None


def load_json_file(test_id, filename):
    p = data_path(test_id, filename)
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)


# ── pipeline face file finder ──────────────────────────────────────────────────

def find_pipeline_face(min_q=None, max_q=None, exact_q=None, no_q=False):
    """
    Find a real face filename from pipeline output faces directory.
    Returns full path or None.
    """
    import re
    if not os.path.exists(FACES_DIR):
        return None
    for pdir in sorted(os.listdir(FACES_DIR)):
        pd = os.path.join(FACES_DIR, pdir)
        if not os.path.isdir(pd):
            continue
        for fname in os.listdir(pd):
            if not fname.endswith('.jpg'):
                continue
            m = re.search(r'_q(\d+)', fname)
            if no_q and m is None:
                return os.path.join(pd, fname)
            if m:
                q = int(m.group(1))
                if exact_q is not None and q == exact_q:
                    return os.path.join(pd, fname)
                if min_q is not None and max_q is not None and min_q <= q <= max_q:
                    return os.path.join(pd, fname)
                if min_q is not None and max_q is None and q >= min_q:
                    return os.path.join(pd, fname)
                if max_q is not None and min_q is None and q <= max_q:
                    return os.path.join(pd, fname)
    return None


def find_body_snapshot():
    """Find any body_snapshot.jpg from pipeline faces output."""
    if not os.path.exists(FACES_DIR):
        return None
    for pdir in os.listdir(FACES_DIR):
        pd = os.path.join(FACES_DIR, pdir)
        p = os.path.join(pd, 'body_snapshot.jpg')
        if os.path.exists(p):
            return p
    return None


# ── result reporter ────────────────────────────────────────────────────────────

def report(test_id, title, passed, actual=None, expected=None):
    bar = '─' * 56
    print(f"\n{bar}")
    print(f"  {'PASS' if passed else 'FAIL'}  {test_id}: {title}")
    if actual   is not None:
        print(f"  actual  : {actual}")
    if expected is not None:
        print(f"  expected: {expected}")
    print(bar)
    return passed
