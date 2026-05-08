#!/usr/bin/env python3
"""
Re-run FR + Generate Report — improved face recognition with separate no-face list.

Improvements over pipeline FR:
  - Lower InsightFace det_thresh (0.3 → 0.15) to catch angled/partial faces
  - Upscale small face images before embedding extraction
  - Contrast enhancement (CLAHE) on dark/washed-out images
  - Padded detection for faces at crop edges
  - Body snapshot used as fallback
  - Lower staff similarity threshold (0.95 → 0.60)
  - No-face persons listed separately (not mixed with customers)

Usage:
    # Re-run FR on all persons + generate report
    python rerun_fr.py --date 2026-04-09

    # Only re-run on persons that failed FR (no_face + unprocessed)
    python rerun_fr.py --date 2026-04-09 --retry-only

    # Custom thresholds
    python rerun_fr.py --date 2026-04-09 --staff-threshold 0.55 --cluster-threshold 0.50

    # Dry run — see results without saving
    python rerun_fr.py --date 2026-04-09 --dry-run

    # Just list current state
    python rerun_fr.py --date 2026-04-09 --list
"""

import os
import sys
import json
import argparse
import numpy as np
import cv2
import logging
import time
from collections import defaultdict
from datetime import datetime

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("rerun_fr")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(SCRIPT_DIR, "config.json"), 'r') as f:
    CONFIG = json.load(f)

STORE_NAME = CONFIG.get("store_name", "Store")
VECTOR_DB_DIR = os.path.join(SCRIPT_DIR, "vector_db")
FAISS_PATH = os.path.join(VECTOR_DB_DIR, "faiss.index")
REGISTRY_PATH = os.path.join(VECTOR_DB_DIR, "staff_registry.json")


def data_dir(date_str):
    return os.path.join(SCRIPT_DIR, "data", date_str)


def load_state(date_str):
    path = os.path.join(data_dir(date_str), "state.json")
    if not os.path.exists(path):
        logger.error(f"No state.json for {date_str}")
        sys.exit(1)
    with open(path, 'r') as f:
        return json.load(f)


def save_state(date_str, state):
    path = os.path.join(data_dir(date_str), "state.json")
    with open(path, 'w') as f:
        json.dump(state, f, indent=2, default=str)


def load_registry():
    if os.path.exists(REGISTRY_PATH):
        with open(REGISTRY_PATH, 'r') as f:
            return json.load(f)
    return []


# ─────────────────────────────────────────────────────────────────────────────
# IMPROVED FACE EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────

def extract_embeddings(face_app, images_folder, body_snapshot=None):
    """
    Extract face embeddings with multiple detection strategies.
    Returns (embeddings, stats).
    """
    embeddings = []
    stats = {
        'images_scanned': 0,
        'faces_extracted': 0,
        'strategies_used': [],
    }

    image_paths = []
    if images_folder and os.path.exists(images_folder):
        jpgs = sorted([f for f in os.listdir(images_folder)
                       if f.endswith('.jpg') and f != 'body_snapshot.jpg'], reverse=True)
        image_paths = [os.path.join(images_folder, f) for f in jpgs]

    # Body snapshot as last resort
    if body_snapshot and os.path.exists(body_snapshot):
        image_paths.append(body_snapshot)

    stats['images_scanned'] = len(image_paths)

    for img_path in image_paths:
        img = cv2.imread(img_path)
        if img is None:
            continue

        emb, strategy = _extract_with_strategies(face_app, img)
        if emb is not None:
            embeddings.append(emb)
            stats['faces_extracted'] += 1
            if strategy not in stats['strategies_used']:
                stats['strategies_used'].append(strategy)

    return embeddings, stats


def _extract_with_strategies(face_app, img):
    """Try multiple strategies to detect a face. Returns (embedding, strategy_name)."""

    # Strategy 1: direct
    emb = _detect_face(face_app, img)
    if emb is not None:
        return emb, 'direct'

    # Strategy 2: upscale small images
    h, w = img.shape[:2]
    if w < 200 or h < 200:
        scale = max(200 / w, 200 / h)
        upscaled = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        emb = _detect_face(face_app, upscaled)
        if emb is not None:
            return emb, 'upscaled'

    # Strategy 3: contrast enhancement (CLAHE)
    enhanced = _enhance_contrast(img)
    emb = _detect_face(face_app, enhanced)
    if emb is not None:
        return emb, 'enhanced'

    # Strategy 4: pad edges (face cut off at border)
    pad = 40
    padded = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_REFLECT_101)
    emb = _detect_face(face_app, padded)
    if emb is not None:
        return emb, 'padded'

    return None, None


def _detect_face(face_app, img):
    """Detect face, return normalized embedding of largest face."""
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


def _enhance_contrast(img):
    """CLAHE contrast enhancement."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)


# ─────────────────────────────────────────────────────────────────────────────
# LIST
# ─────────────────────────────────────────────────────────────────────────────

def list_persons(date_str):
    state = load_state(date_str)
    persons = state.get("persons", [])

    n_staff = sum(1 for p in persons if p.get('update_type') == 'staff')
    n_cust = sum(1 for p in persons if p.get('update_type') == 'customer')
    n_noface = sum(1 for p in persons if p.get('update_type') == 'no_face'
                   or p.get('face_images_count', 0) == 0)
    n_unproc = sum(1 for p in persons if not p.get('fr_processed'))

    print(f"\n{'='*70}")
    print(f"{date_str} — {len(persons)} persons")
    print(f"  Staff: {n_staff}  |  Customers: {n_cust}  |  No Face: {n_noface}  |  Unprocessed: {n_unproc}")
    print(f"{'='*70}")
    print(f"{'ID':>5}  {'Faces':>5}  {'Type':<10}  {'Name':<15}  {'Score':>6}  "
          f"{'First':>8}  {'Last':>8}  {'Body?':>5}")
    print(f"{'-'*5}  {'-'*5}  {'-'*10}  {'-'*15}  {'-'*6}  "
          f"{'-'*8}  {'-'*8}  {'-'*5}")

    for p in persons:
        tid = p.get('track_id', '?')
        fc = p.get('face_images_count', 0)
        utype = p.get('update_type', '-')
        name = p.get('recognized_name', '')
        score = f"{p['similarity_score']:.3f}" if p.get('similarity_score') else ''
        fs = p.get('first_seen', '')[:19].split('T')[-1] if p.get('first_seen') else ''
        ls = p.get('last_seen', '')[:19].split('T')[-1] if p.get('last_seen') else ''
        body = 'yes' if p.get('body_snapshot') else 'no'
        print(f"{tid:>5}  {fc:>5}  {utype:<10}  {name:<15}  {score:>6}  "
              f"{fs:>8}  {ls:>8}  {body:>5}")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# FR RE-RUN
# ─────────────────────────────────────────────────────────────────────────────

def rerun_fr(date_str, retry_only=False, staff_threshold=0.60, cluster_threshold=0.55,
             dry_run=False):
    state = load_state(date_str)
    persons = state.get("persons", [])

    if retry_only:
        targets = [p for p in persons if
                   p.get('update_type') == 'no_face' or not p.get('fr_processed')]
        logger.info(f"Retry-only mode: {len(targets)} of {len(persons)} persons")
    else:
        targets = persons
        # Reset all FR flags for full re-run
        for p in targets:
            p.pop('fr_processed', None)
            p.pop('update_type', None)
            p.pop('recognized_name', None)
            p.pop('similarity_score', None)
            p.pop('fr_rerun', None)

    if not targets:
        print("No persons to process")
        return

    print(f"\n{'='*60}")
    print(f"RE-RUN FR — {date_str}")
    print(f"{'='*60}")
    print(f"Persons to process : {len(targets)}")
    print(f"Staff threshold    : {staff_threshold}")
    print(f"Cluster threshold  : {cluster_threshold}")
    print(f"Mode               : {'retry failed only' if retry_only else 'full re-run'}")
    if dry_run:
        print(f"** DRY RUN **")
    print(f"{'='*60}\n")

    # Load InsightFace — lower det_thresh for better recall
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
        logger.info("No staff registry found — all persons will be customers or no-face")

    t_start = time.time()

    staff_by_name = defaultdict(list)
    customer_data = []
    no_face_list = []

    for i, person in enumerate(targets):
        pid = person.get('track_id', i)
        folder = person.get('images_folder', '')
        body_snap = person.get('body_snapshot')

        # Extract embeddings with improved strategies
        all_embs, stats = extract_embeddings(face_app, folder, body_snap)

        if not all_embs:
            if not dry_run:
                person['fr_processed'] = True
                person['update_type'] = 'no_face'
                person['fr_rerun'] = True
            no_face_list.append({
                'track_id': pid,
                'first_seen': person.get('first_seen', ''),
                'last_seen': person.get('last_seen', ''),
                'body_snapshot': person.get('body_snapshot'),
                'images_folder': folder,
                'face_images_count': person.get('face_images_count', 0),
                'images_scanned': stats['images_scanned'],
            })
            logger.info(f"  Person {pid}: NO FACE ({stats['images_scanned']} images scanned)")
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
            if not dry_run:
                person['fr_processed'] = True
                person['recognized_name'] = best_name
                person['similarity_score'] = round(best_sim, 4)
                person['update_type'] = 'staff'
                person['fr_rerun'] = True
            staff_by_name[best_name].append(person)
            logger.info(f"  Person {pid}: STAFF — {best_name} ({best_sim:.4f}) "
                        f"[{stats['faces_extracted']} embs, {stats['strategies_used']}]")
        else:
            if not dry_run:
                person['fr_processed'] = True
                person['similarity_score'] = round(best_sim, 4)
                person['update_type'] = 'customer'
                person['fr_rerun'] = True
            customer_data.append((pid, best_emb, person))
            logger.info(f"  Person {pid}: CUSTOMER ({best_sim:.4f}) "
                        f"[{stats['faces_extracted']} embs, {stats['strategies_used']}]")

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
            'best_face': next((p.get('best_face_image') for p in plist
                              if p.get('best_face_image')), None),
            'detections': [{'track_id': p.get('track_id'),
                           'first_seen': p.get('first_seen'),
                           'last_seen': p.get('last_seen')} for p in plist],
        })

    elapsed = time.time() - t_start

    # Print summary
    print(f"\n{'='*60}")
    print(f"RESULTS ({elapsed:.1f}s)")
    print(f"{'='*60}")
    print(f"Staff              : {len(staff_results)}")
    for s in staff_results:
        print(f"  {s['name']}: {s['visit_count']} detections, IDs {s['track_ids']}")
    print(f"Unique customers   : {len(clusters)}")
    repeated = [c for c in clusters if c['visit_count'] > 1]
    print(f"Repeated visitors  : {len(repeated)}")
    print(f"No face            : {len(no_face_list)}")
    print(f"{'='*60}")

    # Save state
    if not dry_run:
        # Clean cluster data for serialization
        clusters_clean = []
        for cl in clusters:
            c = dict(cl)
            c.pop('embeddings', None)
            c.pop('persons', None)
            clusters_clean.append(c)

        state["fr_results"] = {
            "staff": staff_results,
            "customer_clusters": clusters_clean,
            "no_face": no_face_list,
            "no_face_count": len(no_face_list),
        }
        save_state(date_str, state)
        logger.info("State saved")

    # Generate report
    if not dry_run:
        # Clean clusters for report
        clusters_for_report = []
        for cl in clusters:
            c = dict(cl)
            c.pop('embeddings', None)
            c.pop('persons', None)
            clusters_for_report.append(c)
        _generate_report(date_str, staff_results, clusters_for_report, no_face_list, state)


# ─────────────────────────────────────────────────────────────────────────────
# REPORT
# ─────────────────────────────────────────────────────────────────────────────

def _generate_report(date_str, staff_results, clusters, no_face_list, state):
    try:
        from openpyxl import Workbook
        from openpyxl.styles import Font, Alignment, PatternFill, Border, Side
    except ImportError:
        logger.warning("openpyxl not installed, skipping report. pip install openpyxl")
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

    thumbs = []  # track for cleanup

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

    # ── Sheet 1: Staff Log ──
    ws = wb.active
    ws.title = "Staff Log"
    ws.merge_cells('A1:G1')
    ws['A1'].value = f"Staff Log — {date_str} ({STORE_NAME})"
    ws['A1'].font, ws['A1'].alignment = tfont, center
    write_header(ws, 3, ['S.No', 'Face', 'Name', 'First Detected',
                         'Last Detected', 'Times', 'Detection Log'])
    ws.column_dimensions['A'].width = 6
    ws.column_dimensions['B'].width = 14
    ws.column_dimensions['C'].width = 18
    ws.column_dimensions['D'].width = 18
    ws.column_dimensions['E'].width = 18
    ws.column_dimensions['F'].width = 10
    ws.column_dimensions['G'].width = 45

    row = 4
    for idx, s in enumerate(staff_results, 1):
        ws.row_dimensions[row].height = 70
        ws.cell(row=row, column=1, value=idx).alignment = center
        add_face(ws, s.get('best_face'), f"B{row}")
        ws.cell(row=row, column=3, value=s['name']).alignment = center
        fs = s.get('first_seen', 'N/A')
        ls = s.get('last_seen', 'N/A')
        if len(fs) > 10: fs = fs[11:19]
        if len(ls) > 10: ls = ls[11:19]
        ws.cell(row=row, column=4, value=fs).alignment = center
        ws.cell(row=row, column=5, value=ls).alignment = center
        ws.cell(row=row, column=6, value=s.get('visit_count', 0)).alignment = center
        log_lines = []
        for d in s.get('detections', []):
            dfs = d.get('first_seen', '?')
            dls = d.get('last_seen', '?')
            if len(dfs) > 10: dfs = dfs[11:19]
            if len(dls) > 10: dls = dls[11:19]
            log_lines.append(f"ID {d.get('track_id')}: {dfs}-{dls}")
        ws.cell(row=row, column=7, value="\n".join(log_lines)).alignment = center
        for c in range(1, 8):
            ws.cell(row=row, column=c).border = border
            ws.cell(row=row, column=c).fill = green
        row += 1

    # ── Sheet 2: Customer Analysis ──
    ws2 = wb.create_sheet("Customer Analysis")
    ws2.merge_cells('A1:G1')
    ws2['A1'].value = f"Customer Analysis — {date_str}"
    ws2['A1'].font, ws2['A1'].alignment = tfont, center
    write_header(ws2, 3, ['S.No', 'Face', 'Visitor', 'Times Seen',
                          'Track IDs', 'First', 'Last'])
    ws2.column_dimensions['A'].width = 6
    ws2.column_dimensions['B'].width = 14
    ws2.column_dimensions['C'].width = 18
    ws2.column_dimensions['D'].width = 12
    ws2.column_dimensions['E'].width = 30
    ws2.column_dimensions['F'].width = 16
    ws2.column_dimensions['G'].width = 16

    row = 4
    for idx, cl in enumerate(clusters, 1):
        ws2.row_dimensions[row].height = 70
        label = f"Visitor #{cl['cluster_id']}"
        if cl['visit_count'] > 1:
            label += " (Repeated)"
        ws2.cell(row=row, column=1, value=idx).alignment = center
        add_face(ws2, cl.get('best_face'), f"B{row}")
        ws2.cell(row=row, column=3, value=label).alignment = center
        ws2.cell(row=row, column=4, value=cl['visit_count']).alignment = center
        ws2.cell(row=row, column=5, value=str(cl.get('track_ids', []))).alignment = center
        fs = cl.get('first_seen', 'N/A')
        ls = cl.get('last_seen', 'N/A')
        if len(str(fs)) > 10: fs = str(fs)[11:19]
        if len(str(ls)) > 10: ls = str(ls)[11:19]
        ws2.cell(row=row, column=6, value=fs).alignment = center
        ws2.cell(row=row, column=7, value=ls).alignment = center
        for c in range(1, 8):
            ws2.cell(row=row, column=c).border = border
            if cl['visit_count'] > 1:
                ws2.cell(row=row, column=c).fill = orange
        row += 1

    # ── Sheet 3: No Face ──
    ws_nf = wb.create_sheet("No Face")
    ws_nf.merge_cells('A1:G1')
    ws_nf['A1'].value = f"No Face Persons — {date_str} ({len(no_face_list)} persons)"
    ws_nf['A1'].font, ws_nf['A1'].alignment = tfont, center
    write_header(ws_nf, 3, ['S.No', 'Track ID', 'Body Snapshot', 'First Detected',
                            'Last Detected', 'Duration', 'Images Folder'], fill=red_hdr)
    ws_nf.column_dimensions['A'].width = 6
    ws_nf.column_dimensions['B'].width = 10
    ws_nf.column_dimensions['C'].width = 14
    ws_nf.column_dimensions['D'].width = 18
    ws_nf.column_dimensions['E'].width = 18
    ws_nf.column_dimensions['F'].width = 10
    ws_nf.column_dimensions['G'].width = 50

    row = 4
    for idx, nf in enumerate(no_face_list, 1):
        ws_nf.row_dimensions[row].height = 70
        ws_nf.cell(row=row, column=1, value=idx).alignment = center
        ws_nf.cell(row=row, column=2, value=nf.get('track_id', '?')).alignment = center
        add_face(ws_nf, nf.get('body_snapshot'), f"C{row}")
        fs = nf.get('first_seen', 'N/A')
        ls = nf.get('last_seen', 'N/A')
        if len(str(fs)) > 10: fs = str(fs)[11:19]
        if len(str(ls)) > 10: ls = str(ls)[11:19]
        duration = ''
        try:
            t1 = datetime.fromisoformat(nf['first_seen'])
            t2 = datetime.fromisoformat(nf['last_seen'])
            duration = f"{(t2 - t1).total_seconds():.0f}s"
        except:
            pass
        ws_nf.cell(row=row, column=4, value=fs).alignment = center
        ws_nf.cell(row=row, column=5, value=ls).alignment = center
        ws_nf.cell(row=row, column=6, value=duration).alignment = center
        ws_nf.cell(row=row, column=7, value=nf.get('images_folder', '')).alignment = center
        for c in range(1, 8):
            ws_nf.cell(row=row, column=c).border = border
            ws_nf.cell(row=row, column=c).fill = red_fill
        row += 1

    # ── Sheet 4: Summary ──
    ws3 = wb.create_sheet("Summary")
    ws3.merge_cells('A1:B1')
    ws3['A1'].value = f"Summary — {date_str}"
    ws3['A1'].font, ws3['A1'].alignment = tfont, center
    ws3.column_dimensions['A'].width = 30
    ws3.column_dimensions['B'].width = 40

    total_persons = len(state.get("persons", []))
    repeated = [c for c in clusters if c['visit_count'] > 1]
    data = [
        ("Metric", "Value"),
        ("Date", date_str),
        ("Store", STORE_NAME),
        ("Total Persons", total_persons),
        ("Staff", len(staff_results)),
        ("Unique Customers", len(clusters)),
        ("Repeated Visitors", len(repeated)),
        ("No Face", len(no_face_list)),
        ("Clips Processed", len(state.get("processed_clips", []))),
        ("Staff Threshold", staff_results and "0.60" or "N/A"),
        ("Cluster Threshold", "0.55"),
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
    path = os.path.join(report_dir, f"fr_rerun_report_{date_str}.xlsx")
    wb.save(path)

    # Cleanup thumbnails
    for t in thumbs:
        if os.path.exists(t):
            os.remove(t)

    print(f"\nReport saved: {path}")


def main():
    parser = argparse.ArgumentParser(description="Re-run FR + Generate Report")
    parser.add_argument("--date", "-d", required=True, help="Date YYYY-MM-DD")
    parser.add_argument("--list", action="store_true", help="Just list persons")
    parser.add_argument("--retry-only", action="store_true",
                        help="Only retry no-face and unprocessed persons")
    parser.add_argument("--staff-threshold", type=float, default=0.60,
                        help="Staff similarity threshold (default: 0.60)")
    parser.add_argument("--cluster-threshold", type=float, default=0.55,
                        help="Customer cluster threshold (default: 0.55)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Preview without saving")

    args = parser.parse_args()

    if args.list:
        list_persons(args.date)
        return

    rerun_fr(
        args.date,
        retry_only=args.retry_only,
        staff_threshold=args.staff_threshold,
        cluster_threshold=args.cluster_threshold,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
