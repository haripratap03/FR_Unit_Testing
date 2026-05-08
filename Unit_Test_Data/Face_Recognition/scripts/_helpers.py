"""
Shared utilities for Face Recognition real-image test scripts.
All scripts in this directory import from here.
"""

import sys
import os
import shutil
import numpy as np
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT   = SCRIPT_DIR.parents[2]          # scripts/ → Face_Recognition/ → Unit_Test_Data/ → FR_Thor/
FACES_DIR   = REPO_ROOT / "data" / "2026-04-29" / "faces"
VIDEO_DIR   = REPO_ROOT / "data" / "2026-04-29" / "videos" / "videos"
INSIGHT_ROOT = REPO_ROOT / "insightface_models"
DATA_DIR    = SCRIPT_DIR.parent / "data"     # Unit_Test_Data/Face_Recognition/data/

sys.path.insert(0, str(REPO_ROOT))


# ── InsightFace loader ────────────────────────────────────────────────────────

def load_face_app(det_thresh: float = 0.15):
    """
    Load InsightFace FaceAnalysis (antelopev2).
    Returns the app object, or None if models are missing.
    """
    try:
        from insightface.app import FaceAnalysis
        app = FaceAnalysis(name="antelopev2", root=str(INSIGHT_ROOT))
        app.prepare(ctx_id=0, det_thresh=det_thresh, det_size=(640, 640))
        return app
    except Exception as exc:
        print(f"  [SKIP] InsightFace unavailable: {exc}")
        return None


# ── Embedding extraction ──────────────────────────────────────────────────────

def extract_real_embeddings(face_app, folder: Path, max_images: int = 10):
    """
    Run extract_embeddings (rerun_fr.py) on a real person folder.
    Returns (embeddings_list, stats_dict).
    """
    from rerun_fr import extract_embeddings
    body = str(folder / "body_snapshot.jpg") if (folder / "body_snapshot.jpg").exists() else None
    return extract_embeddings(face_app, str(folder), body_snapshot=body)


def get_all_jpg_embeddings(face_app, folder: Path):
    """
    Walk a folder, run InsightFace on each .jpg, return list of unit-norm embeddings.
    Skips body_snapshot.jpg (mirrors rerun_fr.py scan logic).
    """
    from rerun_fr import _extract_with_strategies
    import cv2
    embeddings = []
    jpgs = sorted(
        [f for f in os.listdir(str(folder))
         if f.endswith(".jpg") and f != "body_snapshot.jpg"],
        reverse=True,
    )
    for fname in jpgs:
        img = cv2.imread(str(folder / fname))
        if img is None:
            continue
        emb, _ = _extract_with_strategies(face_app, img)
        if emb is not None:
            embeddings.append(emb)
    return embeddings


# ── FAISS helpers ─────────────────────────────────────────────────────────────

def build_faiss_index(embeddings):
    """Build an in-memory IndexFlatIP (cosine-sim via L2-norm) from a list of embeddings."""
    import faiss
    index = faiss.IndexFlatIP(512)
    for emb in embeddings:
        n = np.linalg.norm(emb)
        normed = (emb / n).astype(np.float32) if n > 0 else emb.astype(np.float32)
        index.add(normed.reshape(1, -1))
    return index


def classify_with_faiss(embeddings, faiss_index, registry, threshold: float = 0.80):
    """
    Exact replica of pipeline.py:1213 classification block.
    Returns ('staff', name, best_sim) or ('customer', None, best_sim) or ('no_face', None, None).

    Matches production code exactly:
    - Uses np.expand_dims (no re-normalization — caller must pass L2-normalized embeddings)
    - Staff condition: best_sim >= threshold AND best_name is not None (pipeline.py:1229)
    - Out-of-range index → best_name stays None → falls to customer
    - Uses registry[idx]['name'] via .get() for safety (avoids KeyError on malformed entries)
    """
    if not embeddings:
        return ("no_face", None, None)
    if faiss_index is None or faiss_index.ntotal == 0:
        return ("customer", None, -1.0)

    best_sim = -1.0
    best_name = None
    for emb in embeddings:
        D, I = faiss_index.search(np.expand_dims(emb, 0).astype(np.float32), 1)
        sim = float(D[0][0])
        idx = int(I[0][0])
        if sim > best_sim:
            best_sim = sim
            if 0 <= idx < len(registry):
                best_name = registry[idx].get("name")

    # Mirrors pipeline.py:1229: requires BOTH threshold AND a valid name
    if best_sim >= threshold and best_name:
        return ("staff", best_name, best_sim)
    return ("customer", None, best_sim)


# ── run_fr_processing with real FAISS index (for FR-SM tests) ────────────────

def run_fr_with_real_index(state, faiss_index, registry, *, post_filter=False):
    """
    Call the ACTUAL pipeline.run_fr_processing with a FAISS index built from
    real embeddings.  Writes the index and registry to temp files in vector_db/,
    restoring whatever was there before.

    InsightFace, cv2.imread, os.listdir all run on real data — only the FAISS
    index and registry JSON are substituted so we can inject real-image-derived
    gallery embeddings.

    Args:
        state       : dict with 'persons' list (each 'images_folder' must point
                      to a real directory containing .jpg face crops)
        faiss_index : faiss.IndexFlatIP built via build_faiss_index()
        registry    : list of {'name': str} dicts, one per FAISS index slot
        post_filter : enable quality/skin/kps pre-filter (default False)

    Returns the mutated state dict.
    """
    import faiss as _faiss
    import json as _json
    import tempfile
    from unittest.mock import patch
    from pipeline import run_fr_processing

    vdb_dir = REPO_ROOT / "vector_db"
    vdb_dir.mkdir(exist_ok=True)
    idx_path = vdb_dir / "faiss.index"
    reg_path = vdb_dir / "staff_registry.json"

    # Back up any existing files
    idx_backup = idx_path.read_bytes() if idx_path.exists() else None
    reg_backup = reg_path.read_bytes() if reg_path.exists() else None

    try:
        _faiss.write_index(faiss_index, str(idx_path))
        reg_path.write_text(_json.dumps(registry))
        with patch('pipeline.POST_FILTER_ENABLED', post_filter):
            run_fr_processing("2026-04-29", state)
    finally:
        # Restore originals (or delete if they didn't exist before)
        if idx_backup is None:
            idx_path.unlink(missing_ok=True)
        else:
            idx_path.write_bytes(idx_backup)
        if reg_backup is None:
            reg_path.unlink(missing_ok=True)
        else:
            reg_path.write_bytes(reg_backup)

    return state


# ── Greedy clustering (mirrors pipeline.py:1245 / rerun_fr.py) ────────────────

def greedy_cluster(customer_data, cluster_threshold: float = 0.60):
    """
    Verbatim replica of the greedy single-linkage loop from pipeline.py / rerun_fr.py.
    customer_data: list of (track_id, embedding, person_dict)
    """
    clusters = []
    assigned = set()

    for i, (pid_i, emb_i, _p_i) in enumerate(customer_data):
        if i in assigned:
            continue
        cluster = {
            "cluster_id": len(clusters) + 1,
            "track_ids": [pid_i],
            "embeddings": [emb_i],
        }
        assigned.add(i)
        for j, (pid_j, emb_j, _p_j) in enumerate(customer_data):
            if j in assigned:
                continue
            max_sim = max(float(np.dot(emb_j, ce)) for ce in cluster["embeddings"])
            if max_sim >= cluster_threshold:
                cluster["track_ids"].append(pid_j)
                cluster["embeddings"].append(emb_j)
                assigned.add(j)
        cluster["visit_count"] = len(cluster["track_ids"])
        clusters.append(cluster)

    clusters.sort(key=lambda c: -c["visit_count"])
    return clusters


# ── File/folder helpers ───────────────────────────────────────────────────────

def get_person_folder(person_id: int) -> Path:
    return FACES_DIR / f"person_{person_id:04d}"


def copy_face_folder(src: Path, dst: Path, include_body_snapshot: bool = True):
    """Copy face images (and optionally body_snapshot.jpg) from src to dst."""
    dst.mkdir(parents=True, exist_ok=True)
    for f in src.iterdir():
        if f.suffix == ".jpg":
            if not include_body_snapshot and f.name == "body_snapshot.jpg":
                continue
            shutil.copy2(str(f), str(dst / f.name))


def count_face_jpgs(folder: Path) -> int:
    """Count .jpg files in folder excluding body_snapshot.jpg."""
    return sum(
        1 for f in folder.iterdir()
        if f.suffix == ".jpg" and f.name != "body_snapshot.jpg"
    )


# ── Result reporting ──────────────────────────────────────────────────────────

def report(test_id: str, description: str, passed: bool, detail: str = ""):
    tag = "PASS" if passed else "FAIL"
    line = f"[{tag}] {test_id} — {description}"
    if detail:
        line += f"\n       {detail}"
    print(line)
    return passed
