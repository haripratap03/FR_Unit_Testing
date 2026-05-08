"""
Staff Registration Tool — register known staff from pipeline results.

Usage:
    # Interactive: shows each person's face, you label them
    python register_staff.py --date 2026-04-09

    # Direct: assign name to specific person IDs (supports multiple IDs for same person)
    python register_staff.py --date 2026-04-09 --name "Ravi" --ids 3 7 12

    # List all persons from a date to see IDs and face paths
    python register_staff.py --date 2026-04-09 --list

    # Show current staff registry
    python register_staff.py --show-registry

    # Remove a staff member
    python register_staff.py --remove "Ravi"
"""

import os
import sys
import json
import argparse
import numpy as np
import cv2
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("register_staff")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Load config
with open(os.path.join(SCRIPT_DIR, "config.json"), 'r') as f:
    CONFIG = json.load(f)

VECTOR_DB_DIR = os.path.join(SCRIPT_DIR, "vector_db")
FAISS_PATH = os.path.join(VECTOR_DB_DIR, "faiss.index")
REGISTRY_PATH = os.path.join(VECTOR_DB_DIR, "staff_registry.json")


def data_dir(date_str):
    return os.path.join(SCRIPT_DIR, "data", date_str)


def load_state(date_str):
    state_path = os.path.join(data_dir(date_str), "state.json")
    if not os.path.exists(state_path):
        logger.error(f"No state.json found for {date_str}")
        sys.exit(1)
    with open(state_path, 'r') as f:
        return json.load(f)


def load_registry():
    if os.path.exists(REGISTRY_PATH):
        with open(REGISTRY_PATH, 'r') as f:
            return json.load(f)
    return []


def save_registry(registry):
    os.makedirs(VECTOR_DB_DIR, exist_ok=True)
    with open(REGISTRY_PATH, 'w') as f:
        json.dump(registry, f, indent=2)


def load_faiss_index():
    import faiss
    if os.path.exists(FAISS_PATH):
        return faiss.read_index(FAISS_PATH)
    # Inner product index for cosine similarity (embeddings are L2-normalized)
    return faiss.IndexFlatIP(512)


def save_faiss_index(index):
    import faiss
    os.makedirs(VECTOR_DB_DIR, exist_ok=True)
    faiss.write_index(index, FAISS_PATH)


def get_face_app():
    from insightface.app import FaceAnalysis
    models_dir = os.path.join(SCRIPT_DIR, "insightface_models")
    providers = [("CUDAExecutionProvider", {}), ("CPUExecutionProvider", {})]
    face_app = FaceAnalysis(name="antelopev2", root=models_dir, providers=providers)
    face_app.prepare(ctx_id=0, det_size=(640, 640), det_thresh=0.3)
    return face_app


def extract_embeddings(face_app, images_folder):
    """Extract face embeddings from all images in a person's folder."""
    embeddings = []
    if not os.path.exists(images_folder):
        return embeddings

    jpgs = sorted([f for f in os.listdir(images_folder) if f.endswith('.jpg')], reverse=True)
    for jpg in jpgs:
        img = cv2.imread(os.path.join(images_folder, jpg))
        if img is None:
            continue
        faces = face_app.get(img)
        if not faces:
            continue
        emb = faces[0].embedding
        if emb is None:
            continue
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb = emb / norm
        embeddings.append(emb.astype(np.float32))

    return embeddings


def register_staff_from_ids(date_str, name, person_ids):
    """Register a staff member using face images from one or more person IDs."""
    state = load_state(date_str)
    persons = state.get("persons", [])

    # Find matching persons
    matched = []
    for p in persons:
        if p.get("track_id") in person_ids:
            matched.append(p)

    if not matched:
        logger.error(f"No persons found with IDs {person_ids} in {date_str}")
        logger.info(f"Available IDs: {[p.get('track_id') for p in persons]}")
        return False

    logger.info(f"Found {len(matched)} person(s) for '{name}': IDs {[p['track_id'] for p in matched]}")

    # Extract embeddings from all matched persons
    face_app = get_face_app()
    all_embeddings = []

    for person in matched:
        folder = person.get("images_folder", "")
        embs = extract_embeddings(face_app, folder)
        logger.info(f"  Person {person['track_id']}: {len(embs)} face embeddings from {folder}")
        all_embeddings.extend(embs)

    if not all_embeddings:
        logger.error(f"No face embeddings extracted for '{name}'. Check face images exist.")
        return False

    # Compute average embedding for this person
    avg_embedding = np.mean(all_embeddings, axis=0).astype(np.float32)
    norm = np.linalg.norm(avg_embedding)
    if norm > 0:
        avg_embedding = avg_embedding / norm

    # Load existing registry and index
    registry = load_registry()
    index = load_faiss_index()

    # Check if staff already exists — update instead of duplicating
    existing_idx = None
    for i, entry in enumerate(registry):
        if entry["name"].lower() == name.lower():
            existing_idx = i
            break

    if existing_idx is not None:
        # Merge: average old embedding with new embeddings
        old_emb_count = registry[existing_idx].get("embedding_count", 1)
        old_total = old_emb_count

        # Rebuild index without old entry, then add merged
        logger.info(f"  Updating existing staff '{name}' (had {old_emb_count} embeddings, adding {len(all_embeddings)} new)")

        # We need to rebuild FAISS since it doesn't support in-place update
        import faiss
        new_index = faiss.IndexFlatIP(512)
        new_registry = []

        for i, entry in enumerate(registry):
            if i == existing_idx:
                continue
            # Re-extract the embedding from the index
            emb = index.reconstruct(i).reshape(1, -1)
            new_index.add(emb)
            new_registry.append(entry)

        # Add merged entry
        new_count = old_total + len(all_embeddings)
        new_index.add(avg_embedding.reshape(1, -1))
        new_registry.append({
            "name": name,
            "person_ids": list(set(registry[existing_idx].get("person_ids", []) + person_ids)),
            "date_registered": registry[existing_idx].get("date_registered", date_str),
            "date_updated": date_str,
            "embedding_count": new_count,
        })

        index = new_index
        registry = new_registry
    else:
        # Add new staff entry
        index.add(avg_embedding.reshape(1, -1))
        registry.append({
            "name": name,
            "person_ids": person_ids,
            "date_registered": date_str,
            "embedding_count": len(all_embeddings),
        })

    # Also add individual embeddings for better matching diversity
    # Store top-N diverse embeddings (up to 5) per person for robustness
    if len(all_embeddings) > 1:
        # Pick up to 4 most diverse additional embeddings
        extra = _pick_diverse(all_embeddings, avg_embedding, max_extra=4)
        for emb in extra:
            index.add(emb.reshape(1, -1))
            registry.append({
                "name": name,
                "person_ids": person_ids,
                "date_registered": date_str,
                "is_extra_embedding": True,
            })

    save_faiss_index(index)
    save_registry(registry)

    logger.info(f"Registered '{name}' with {len(all_embeddings)} face embeddings "
                f"(index now has {index.ntotal} vectors for {len(set(e['name'] for e in registry))} staff)")
    return True


def _pick_diverse(embeddings, avg, max_extra=4):
    """Pick the most diverse embeddings relative to the average."""
    dists = []
    for emb in embeddings:
        sim = float(np.dot(emb, avg))
        dists.append((1.0 - sim, emb))
    # Sort by distance from average (most different first)
    dists.sort(key=lambda x: -x[0])
    return [emb for _, emb in dists[:max_extra]]


def list_persons(date_str):
    """List all persons from a processed date."""
    state = load_state(date_str)
    persons = state.get("persons", [])

    if not persons:
        print(f"No persons found for {date_str}")
        return

    print(f"\n{'='*70}")
    print(f"Persons from {date_str} — {len(persons)} total")
    print(f"{'='*70}")
    print(f"{'ID':>5}  {'Faces':>5}  {'First Seen':>12}  {'Last Seen':>12}  {'Images Folder'}")
    print(f"{'-'*5}  {'-'*5}  {'-'*12}  {'-'*12}  {'-'*40}")

    for p in persons:
        tid = p.get("track_id", "?")
        fc = p.get("face_images_count", 0)
        fs = p.get("first_seen", "")[:19].split("T")[-1] if p.get("first_seen") else ""
        ls = p.get("last_seen", "")[:19].split("T")[-1] if p.get("last_seen") else ""
        folder = p.get("images_folder", "")
        fr_label = ""
        if p.get("recognized_name"):
            fr_label = f"  [STAFF: {p['recognized_name']}]"
        elif p.get("update_type") == "no_face":
            fr_label = "  [NO FACE]"
        print(f"{tid:>5}  {fc:>5}  {fs:>12}  {ls:>12}  {folder}{fr_label}")

    print()


def show_registry():
    """Show current staff registry."""
    registry = load_registry()
    if not registry:
        print("Staff registry is empty.")
        return

    # Group by name
    by_name = {}
    for entry in registry:
        name = entry["name"]
        if name not in by_name:
            by_name[name] = {
                "person_ids": entry.get("person_ids", []),
                "date_registered": entry.get("date_registered", ""),
                "embedding_count": 0,
                "index_entries": 0,
            }
        by_name[name]["index_entries"] += 1
        if not entry.get("is_extra_embedding"):
            by_name[name]["embedding_count"] = entry.get("embedding_count", 0)
            by_name[name]["person_ids"] = list(set(
                by_name[name]["person_ids"] + entry.get("person_ids", [])
            ))

    print(f"\n{'='*60}")
    print(f"Staff Registry — {len(by_name)} staff members")
    print(f"{'='*60}")
    print(f"{'Name':<20} {'IDs':<15} {'Embeddings':>10}  {'Registered'}")
    print(f"{'-'*20} {'-'*15} {'-'*10}  {'-'*12}")

    for name, info in sorted(by_name.items()):
        ids_str = ",".join(str(i) for i in info["person_ids"])
        print(f"{name:<20} {ids_str:<15} {info['index_entries']:>10}  {info['date_registered']}")

    if os.path.exists(FAISS_PATH):
        import faiss
        idx = faiss.read_index(FAISS_PATH)
        print(f"\nFAISS index: {idx.ntotal} total vectors")
    print()


def remove_staff(name):
    """Remove a staff member from registry and FAISS index."""
    registry = load_registry()

    # Find indices to remove
    remove_indices = [i for i, e in enumerate(registry) if e["name"].lower() == name.lower()]

    if not remove_indices:
        logger.error(f"Staff '{name}' not found in registry")
        return False

    import faiss
    index = load_faiss_index()

    # Rebuild without removed entries
    new_index = faiss.IndexFlatIP(512)
    new_registry = []

    for i, entry in enumerate(registry):
        if i in remove_indices:
            continue
        emb = index.reconstruct(i).reshape(1, -1)
        new_index.add(emb)
        new_registry.append(entry)

    save_faiss_index(new_index)
    save_registry(new_registry)

    logger.info(f"Removed '{name}' ({len(remove_indices)} embeddings). "
                f"Registry now has {len(new_registry)} entries.")
    return True


def interactive_mode(date_str):
    """Interactive mode: show each person's best face, ask for name."""
    state = load_state(date_str)
    persons = state.get("persons", [])

    if not persons:
        print(f"No persons found for {date_str}")
        return

    persons_with_faces = [p for p in persons if p.get("face_images_count", 0) > 0]
    print(f"\n{len(persons_with_faces)} persons with faces (out of {len(persons)} total)")
    print("For each person, enter:\n"
          "  - Staff name to register\n"
          "  - 's' to skip\n"
          "  - 'q' to quit\n"
          "  - 'merge <id>' to merge with another person ID under same name\n")

    registered = {}  # name -> list of IDs

    for p in persons_with_faces:
        tid = p.get("track_id", "?")
        folder = p.get("images_folder", "")
        fc = p.get("face_images_count", 0)
        fs = p.get("first_seen", "")[:19]
        ls = p.get("last_seen", "")[:19]

        # Show best face image
        best_face = p.get("best_face_image", "")
        if best_face and os.path.exists(best_face):
            img = cv2.imread(best_face)
            if img is not None:
                # Resize for display
                h, w = img.shape[:2]
                scale = min(300 / w, 300 / h, 1.0)
                display = cv2.resize(img, (int(w * scale), int(h * scale)))
                cv2.imshow(f"Person {tid}", display)
                cv2.waitKey(500)

        print(f"\n--- Person ID: {tid} | Faces: {fc} | {fs} - {ls} ---")
        print(f"    Folder: {folder}")

        resp = input("    Name (or s/q/merge <id>): ").strip()

        cv2.destroyAllWindows()

        if resp.lower() == 'q':
            break
        if resp.lower() == 's' or resp == '':
            continue
        if resp.lower().startswith('merge '):
            # merge this person with an already-named person
            try:
                merge_id = int(resp.split()[1])
                # Find which name that ID belongs to
                merge_name = None
                for name, ids in registered.items():
                    if merge_id in ids:
                        merge_name = name
                        break
                if merge_name:
                    registered[merge_name].append(tid)
                    print(f"    Merged {tid} with '{merge_name}' (IDs: {registered[merge_name]})")
                else:
                    print(f"    ID {merge_id} not registered yet. Register it first.")
            except (ValueError, IndexError):
                print("    Invalid merge command. Use: merge <id>")
            continue

        name = resp
        if name in registered:
            registered[name].append(tid)
        else:
            registered[name] = [tid]
        print(f"    Queued '{name}' with IDs: {registered[name]}")

    # Now register all
    if registered:
        print(f"\nRegistering {len(registered)} staff members...")
        for name, ids in registered.items():
            register_staff_from_ids(date_str, name, ids)
        print("Done!")
    else:
        print("No staff registered.")


def main():
    parser = argparse.ArgumentParser(description="Staff Registration Tool")
    parser.add_argument("--date", "-d", default=None, help="Date YYYY-MM-DD of processed data")
    parser.add_argument("--name", "-n", default=None, help="Staff name to register")
    parser.add_argument("--ids", nargs="+", type=int, default=None, help="Person IDs to register as this staff")
    parser.add_argument("--list", action="store_true", help="List all persons from a date")
    parser.add_argument("--show-registry", action="store_true", help="Show current staff registry")
    parser.add_argument("--remove", default=None, help="Remove a staff member by name")
    parser.add_argument("--interactive", "-i", action="store_true", help="Interactive mode with face preview")

    args = parser.parse_args()

    if args.show_registry:
        show_registry()
        return

    if args.remove:
        remove_staff(args.remove)
        return

    if not args.date and not args.show_registry:
        parser.error("--date is required (except for --show-registry or --remove)")

    if args.list:
        list_persons(args.date)
        return

    if args.name and args.ids:
        register_staff_from_ids(args.date, args.name, args.ids)
        return

    if args.interactive:
        interactive_mode(args.date)
        return

    # Default: if name+ids not given, go interactive
    if not args.name:
        interactive_mode(args.date)
        return

    parser.error("Provide both --name and --ids, or use --interactive")


if __name__ == "__main__":
    main()
