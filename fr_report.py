#!/usr/bin/env python3
"""
Standalone FR Report — scan all face folders for a date, run FR, generate one report.

This script does NOT need state.json. It directly scans the faces/ directory,
picks up every person_XXXX folder, extracts embeddings, matches against staff,
clusters customers, and generates a single Excel report.

Usage:
    python fr_report.py --date 2026-04-09
    python fr_report.py --date 2026-04-09 --staff-threshold 0.55
    python fr_report.py --date 2026-04-09 --faces-dir /custom/path/to/faces
"""

import os
import sys
import json
import argparse
import numpy as np
import cv2
import logging
import time
import re
from collections import defaultdict

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("fr_report")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(SCRIPT_DIR, "config.json"), 'r') as f:
    CONFIG = json.load(f)

STORE_NAME = CONFIG.get("store_name", "Store")
VECTOR_DB_DIR = os.path.join(SCRIPT_DIR, "vector_db")
FAISS_PATH = os.path.join(VECTOR_DB_DIR, "faiss.index")
REGISTRY_PATH = os.path.join(VECTOR_DB_DIR, "staff_registry.json")


def data_dir(date_str):
    return os.path.join(SCRIPT_DIR, "data", date_str)


def load_registry():
    if os.path.exists(REGISTRY_PATH):
        with open(REGISTRY_PATH, 'r') as f:
            return json.load(f)
    return []


# ─────────────────────────────────────────────────────────────────────────────
# SCAN FACES DIRECTORY
# ─────────────────────────────────────────────────────────────────────────────

def load_state_timestamps(date_str):
    """Load first_seen/last_seen from state.json, keyed by track_id."""
    state_path = os.path.join(data_dir(date_str), "state.json")
    if not os.path.exists(state_path):
        return {}
    with open(state_path, 'r') as f:
        state = json.load(f)
    ts_map = {}
    for p in state.get("persons", []):
        tid = p.get("track_id")
        if tid is not None:
            ts_map[tid] = {
                'first_seen': p.get('first_seen', ''),
                'last_seen': p.get('last_seen', ''),
            }
    return ts_map


def scan_faces(faces_dir, date_str=None):
    """Scan faces/ directory and build person list from folders.
    Skips folders that have no face images (body-only).
    Merges first_seen/last_seen from state.json if date_str is provided.
    """
    persons = []

    if not os.path.exists(faces_dir):
        logger.error(f"Faces directory not found: {faces_dir}")
        return persons

    # Load timestamps from state.json
    ts_map = load_state_timestamps(date_str) if date_str else {}

    # Find all person_XXXX folders
    folders = sorted([f for f in os.listdir(faces_dir)
                      if os.path.isdir(os.path.join(faces_dir, f))
                      and f.startswith("person_")])

    skipped_body_only = 0
    for folder_name in folders:
        folder_path = os.path.join(faces_dir, folder_name)

        # Extract track ID from folder name
        match = re.search(r'person_(\d+)', folder_name)
        track_id = int(match.group(1)) if match else 0

        # List face images (exclude body_snapshot and rejected/ subfolder contents)
        all_files = sorted(os.listdir(folder_path))
        face_jpgs = [f for f in all_files
                     if f.endswith('.jpg') and f != 'body_snapshot.jpg'
                     and not os.path.isdir(os.path.join(folder_path, f))]

        # Skip folders with no face images
        if not face_jpgs:
            skipped_body_only += 1
            continue

        best_face = None
        # Pick highest quality from filename (face_XXXXXX_qNN.jpg)
        def get_quality(fname):
            m = re.search(r'_q(\d+)', fname)
            return int(m.group(1)) if m else 0
        best_jpg = max(face_jpgs, key=get_quality)
        best_face = os.path.join(folder_path, best_jpg)

        # Get timestamps from state.json
        ts = ts_map.get(track_id, {})

        persons.append({
            'track_id': track_id,
            'images_folder': folder_path,
            'face_images': [(0, os.path.join(folder_path, f)) for f in face_jpgs],
            'face_images_count': len(face_jpgs),
            'best_face_image': best_face,
            'first_seen': ts.get('first_seen', ''),
            'last_seen': ts.get('last_seen', ''),
        })

    logger.info(f"Found {len(persons)} person folders with faces in {faces_dir}")
    if skipped_body_only:
        logger.info(f"  Skipped {skipped_body_only} body-only folders (no face images)")

    return persons


# ─────────────────────────────────────────────────────────────────────────────
# EXTRACT EMBEDDINGS
# ─────────────────────────────────────────────────────────────────────────────

def extract_embeddings(face_app, person):
    """Extract face embeddings from a person's folder with multiple strategies."""
    embeddings = []
    folder = person['images_folder']

    # Face images first
    for _, img_path in person['face_images']:
        if not os.path.exists(img_path):
            continue
        emb = _extract_from_image(face_app, img_path)
        if emb is not None:
            embeddings.append(emb)

    # Body snapshot as fallback
    if not embeddings and person.get('body_snapshot'):
        emb = _extract_from_image(face_app, person['body_snapshot'])
        if emb is not None:
            embeddings.append(emb)

    return embeddings


def _extract_from_image(face_app, img_path):
    """Try multiple strategies to extract face embedding from an image."""
    img = cv2.imread(img_path)
    if img is None:
        return None

    # Strategy 1: direct
    emb = _detect_face(face_app, img)
    if emb is not None:
        return emb

    # Strategy 2: upscale small images
    h, w = img.shape[:2]
    if w < 200 or h < 200:
        scale = max(200 / w, 200 / h)
        upscaled = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        emb = _detect_face(face_app, upscaled)
        if emb is not None:
            return emb

    # Strategy 3: contrast enhancement
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    enhanced = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
    emb = _detect_face(face_app, enhanced)
    if emb is not None:
        return emb

    # Strategy 4: pad edges
    pad = 40
    padded = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_REFLECT_101)
    emb = _detect_face(face_app, padded)
    return emb


def _detect_face(face_app, img):
    """Detect largest face, return normalized embedding."""
    try:
        faces = face_app.get(img)
        if not faces:
            return None
        best = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        emb = best.embedding
        if emb is None:
            return None
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb = emb / norm
        return emb.astype(np.float32)
    except:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# FR + CLUSTERING
# ─────────────────────────────────────────────────────────────────────────────

def run_fr(persons, staff_threshold=0.60, cluster_threshold=0.55):
    """Run FR on all persons — match staff, cluster customers, list no-face."""

    logger.info("Loading InsightFace (det_thresh=0.15)...")
    from insightface.app import FaceAnalysis
    models_dir = os.path.join(SCRIPT_DIR, "insightface_models")
    providers = [("CUDAExecutionProvider", {}), ("CPUExecutionProvider", {})]
    face_app = FaceAnalysis(name="antelopev2", root=models_dir, providers=providers)
    face_app.prepare(ctx_id=0, det_size=(640, 640), det_thresh=0.15)

    # Load FAISS
    import faiss
    faiss_index = None
    staff_registry = []
    if os.path.exists(FAISS_PATH):
        faiss_index = faiss.read_index(FAISS_PATH)
        if os.path.exists(REGISTRY_PATH):
            with open(REGISTRY_PATH, 'r') as f:
                staff_registry = json.load(f)
        logger.info(f"FAISS loaded: {faiss_index.ntotal} staff vectors")
    else:
        logger.info("No staff registry found")

    t_start = time.time()

    staff_by_name = defaultdict(list)
    customer_data = []
    no_face_list = []

    for person in persons:
        pid = person['track_id']

        all_embs = extract_embeddings(face_app, person)

        if not all_embs:
            person['update_type'] = 'no_face'
            no_face_list.append(person)
            logger.info(f"  Person {pid}: NO FACE ({person['face_images_count']} images)")
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

        if best_sim >= staff_threshold and best_name:
            person['recognized_name'] = best_name
            person['similarity_score'] = round(best_sim, 4)
            person['update_type'] = 'staff'
            staff_by_name[best_name].append(person)
            logger.info(f"  Person {pid}: STAFF — {best_name} ({best_sim:.4f})")
        else:
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
            'track_ids': [p_i['track_id']],
            'persons': [p_i],
            'embeddings': [emb_i],
        }
        assigned.add(i)
        for j, (pid_j, emb_j, p_j) in enumerate(customer_data):
            if j in assigned:
                continue
            max_sim = max(float(np.dot(emb_j, ce)) for ce in cluster['embeddings'])
            if max_sim >= cluster_threshold:
                cluster['track_ids'].append(p_j['track_id'])
                cluster['persons'].append(p_j)
                cluster['embeddings'].append(emb_j)
                assigned.add(j)

        fs = min((p.get('first_seen', '') for p in cluster['persons'] if p.get('first_seen')), default='')
        ls = max((p.get('last_seen', '') for p in cluster['persons'] if p.get('last_seen')), default='')
        cluster['first_seen'] = fs
        cluster['last_seen'] = ls
        cluster['visit_count'] = len(cluster['persons'])
        cluster['best_face'] = None
        for p in cluster['persons']:
            bf = p.get('best_face_image')
            if bf and os.path.exists(str(bf)):
                cluster['best_face'] = bf
                break
        clusters.append(cluster)

    clusters.sort(key=lambda c: -c['visit_count'])

    # Build staff results
    staff_results = []
    for name, plist in staff_by_name.items():
        fs = min((p.get('first_seen', '') for p in plist if p.get('first_seen')), default='')
        ls = max((p.get('last_seen', '') for p in plist if p.get('last_seen')), default='')
        staff_results.append({
            'name': name,
            'track_ids': [p['track_id'] for p in plist],
            'visit_count': len(plist),
            'first_seen': fs,
            'last_seen': ls,
            'best_face': next((p.get('best_face_image') for p in plist
                              if p.get('best_face_image')), None),
        })

    elapsed = time.time() - t_start
    logger.info(f"FR done in {elapsed:.1f}s: {len(staff_results)} staff, "
                f"{len(clusters)} unique customers, {len(no_face_list)} no-face")

    return staff_results, clusters, no_face_list


# ─────────────────────────────────────────────────────────────────────────────
# REPORT
# ─────────────────────────────────────────────────────────────────────────────

def generate_report(date_str, staff_results, clusters, no_face_list, total):
    try:
        from openpyxl import Workbook
        from openpyxl.styles import Font, Alignment, PatternFill, Border, Side
    except ImportError:
        logger.error("openpyxl not installed: pip install openpyxl")
        return

    wb = Workbook()
    hfont = Font(bold=True, size=12, color="FFFFFF")
    hfill = PatternFill(start_color="4472C4", end_color="4472C4", fill_type="solid")
    tfont = Font(bold=True, size=14)
    green = PatternFill(start_color="E2EFDA", end_color="E2EFDA", fill_type="solid")
    orange = PatternFill(start_color="FCE4D6", end_color="FCE4D6", fill_type="solid")
    red_fill = PatternFill(start_color="FFC7CE", end_color="FFC7CE", fill_type="solid")
    red_hdr = PatternFill(start_color="C00000", end_color="C00000", fill_type="solid")
    border = Border(left=Side(style='thin'), right=Side(style='thin'),
                    top=Side(style='thin'), bottom=Side(style='thin'))
    center = Alignment(horizontal='center', vertical='center', wrap_text=True)
    thumbs = []

    def write_header(ws, row, headers, fill=hfill):
        for col, h in enumerate(headers, 1):
            c = ws.cell(row=row, column=col, value=h)
            c.font, c.fill, c.border, c.alignment = hfont, fill, border, center

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
            thumbs.append(thumb)
            xl = XLImage(thumb)
            xl.width, xl.height = 80, 80
            ws.add_image(xl, cell)
        except:
            pass

    # ── Sheet 1: Staff ──
    ws = wb.active
    ws.title = "Staff"
    ws.merge_cells('A1:E1')
    ws['A1'].value = f"Staff — {date_str} ({STORE_NAME})"
    ws['A1'].font, ws['A1'].alignment = tfont, center
    write_header(ws, 3, ['S.No', 'Face', 'Name', 'First Seen', 'Last Seen', 'Times Seen', 'Track IDs'])
    ws.column_dimensions['A'].width = 6
    ws.column_dimensions['B'].width = 14
    ws.column_dimensions['C'].width = 20
    ws.column_dimensions['D'].width = 16
    ws.column_dimensions['E'].width = 16
    ws.column_dimensions['F'].width = 12
    ws.column_dimensions['G'].width = 30

    row = 4
    for idx, s in enumerate(staff_results, 1):
        ws.row_dimensions[row].height = 70
        ws.cell(row=row, column=1, value=idx).alignment = center
        add_face(ws, s.get('best_face'), f"B{row}")
        ws.cell(row=row, column=3, value=s['name']).alignment = center
        fs = s.get('first_seen', '')
        ls = s.get('last_seen', '')
        if len(fs) > 10: fs = fs[11:19]
        if len(ls) > 10: ls = ls[11:19]
        ws.cell(row=row, column=4, value=fs).alignment = center
        ws.cell(row=row, column=5, value=ls).alignment = center
        ws.cell(row=row, column=6, value=s['visit_count']).alignment = center
        ws.cell(row=row, column=7, value=str(s['track_ids'])).alignment = center
        for c in range(1, 8):
            ws.cell(row=row, column=c).border = border
            ws.cell(row=row, column=c).fill = green
        row += 1

    # ── Sheet 2: Customers ──
    ws2 = wb.create_sheet("Customers")
    ws2.merge_cells('A1:F1')
    ws2['A1'].value = f"Customers — {date_str}"
    ws2['A1'].font, ws2['A1'].alignment = tfont, center
    write_header(ws2, 3, ['S.No', 'Face', 'Visitor', 'First Seen', 'Last Seen', 'Times Seen', 'Track IDs'])
    ws2.column_dimensions['A'].width = 6
    ws2.column_dimensions['B'].width = 14
    ws2.column_dimensions['C'].width = 20
    ws2.column_dimensions['D'].width = 16
    ws2.column_dimensions['E'].width = 16
    ws2.column_dimensions['F'].width = 12
    ws2.column_dimensions['G'].width = 30

    row = 4
    for idx, cl in enumerate(clusters, 1):
        ws2.row_dimensions[row].height = 70
        label = f"Visitor #{cl['cluster_id']}"
        if cl['visit_count'] > 1:
            label += " (Repeated)"
        ws2.cell(row=row, column=1, value=idx).alignment = center
        add_face(ws2, cl.get('best_face'), f"B{row}")
        ws2.cell(row=row, column=3, value=label).alignment = center
        fs = cl.get('first_seen', '')
        ls = cl.get('last_seen', '')
        if len(str(fs)) > 10: fs = str(fs)[11:19]
        if len(str(ls)) > 10: ls = str(ls)[11:19]
        ws2.cell(row=row, column=4, value=fs).alignment = center
        ws2.cell(row=row, column=5, value=ls).alignment = center
        ws2.cell(row=row, column=6, value=cl['visit_count']).alignment = center
        ws2.cell(row=row, column=7, value=str(cl.get('track_ids', []))).alignment = center
        for c in range(1, 8):
            ws2.cell(row=row, column=c).border = border
            if cl['visit_count'] > 1:
                ws2.cell(row=row, column=c).fill = orange
        row += 1

    # ── Sheet 3: No Face ──
    ws_nf = wb.create_sheet("No Face")
    ws_nf.merge_cells('A1:E1')
    ws_nf['A1'].value = f"No Face — {date_str} ({len(no_face_list)} persons)"
    ws_nf['A1'].font, ws_nf['A1'].alignment = tfont, center
    write_header(ws_nf, 3, ['S.No', 'Track ID', 'Body Snapshot', 'Face Count', 'Folder'],
                 fill=red_hdr)
    ws_nf.column_dimensions['A'].width = 6
    ws_nf.column_dimensions['B'].width = 10
    ws_nf.column_dimensions['C'].width = 14
    ws_nf.column_dimensions['D'].width = 12
    ws_nf.column_dimensions['E'].width = 50

    row = 4
    for idx, nf in enumerate(no_face_list, 1):
        ws_nf.row_dimensions[row].height = 70
        ws_nf.cell(row=row, column=1, value=idx).alignment = center
        ws_nf.cell(row=row, column=2, value=nf['track_id']).alignment = center
        add_face(ws_nf, nf.get('body_snapshot'), f"C{row}")
        ws_nf.cell(row=row, column=4, value=nf['face_images_count']).alignment = center
        ws_nf.cell(row=row, column=5, value=nf['images_folder']).alignment = center
        for c in range(1, 6):
            ws_nf.cell(row=row, column=c).border = border
            ws_nf.cell(row=row, column=c).fill = red_fill
        row += 1

    # ── Sheet 4: Summary ──
    ws3 = wb.create_sheet("Summary")
    ws3.merge_cells('A1:B1')
    ws3['A1'].value = f"Summary — {date_str}"
    ws3['A1'].font, ws3['A1'].alignment = tfont, center
    ws3.column_dimensions['A'].width = 25
    ws3.column_dimensions['B'].width = 30

    repeated = [c for c in clusters if c['visit_count'] > 1]
    data = [
        ("Metric", "Value"),
        ("Date", date_str),
        ("Store", STORE_NAME),
        ("Total Persons", total),
        ("With Face", total - len(no_face_list)),
        ("Staff", len(staff_results)),
        ("Unique Customers", len(clusters)),
        ("Repeated Visitors", len(repeated)),
        ("No Face", len(no_face_list)),
    ]
    row = 3
    for m, v in data:
        ws3.cell(row=row, column=1, value=m).border = border
        ws3.cell(row=row, column=2, value=v).border = border
        if row == 3:
            ws3.cell(row=row, column=1).font = hfont
            ws3.cell(row=row, column=1).fill = hfill
            ws3.cell(row=row, column=2).font = hfont
            ws3.cell(row=row, column=2).fill = hfill
        row += 1

    # Save
    report_dir = os.path.join(data_dir(date_str), "report")
    os.makedirs(report_dir, exist_ok=True)
    path = os.path.join(report_dir, f"fr_faces_report_{date_str}.xlsx")
    wb.save(path)

    for t in thumbs:
        if os.path.exists(t):
            os.remove(t)

    print(f"\nReport saved: {path}")
    return path


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Standalone FR Report from faces/ folder")
    parser.add_argument("--date", "-d", required=True, help="Date YYYY-MM-DD")
    parser.add_argument("--faces-dir", default=None,
                        help="Custom faces directory (default: data/<date>/faces)")
    parser.add_argument("--staff-threshold", type=float, default=0.60)
    parser.add_argument("--cluster-threshold", type=float, default=0.55)

    args = parser.parse_args()

    faces_dir = args.faces_dir or os.path.join(data_dir(args.date), "faces")

    # Scan
    persons = scan_faces(faces_dir, date_str=args.date)
    if not persons:
        print(f"No person folders found in {faces_dir}")
        return

    # FR
    staff_results, clusters, no_face_list = run_fr(
        persons,
        staff_threshold=args.staff_threshold,
        cluster_threshold=args.cluster_threshold,
    )

    # Print summary
    total = len(persons)
    repeated = [c for c in clusters if c['visit_count'] > 1]
    print(f"\n{'='*50}")
    print(f"FR REPORT — {args.date}")
    print(f"{'='*50}")
    print(f"Total persons  : {total}")
    print(f"Staff          : {len(staff_results)}")
    print(f"Customers      : {len(clusters)}")
    print(f"Repeated       : {len(repeated)}")
    print(f"No face        : {len(no_face_list)}")
    print(f"{'='*50}")

    # Report
    generate_report(args.date, staff_results, clusters, no_face_list, total)


if __name__ == "__main__":
    main()
