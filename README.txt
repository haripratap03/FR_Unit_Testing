Offline FR Pipeline — Process recorded videos for face recognition
==================================================================

FOLDER STRUCTURE (after setup):
    offline_fr/
    ├── config.json                # Edit this with Jetson IP, paths, line config
    ├── pipeline.py                # Main script
    ├── README.txt
    ├── models/                    # Copy/download these
    │   ├── yolo11l.pt             (auto-downloads)
    │   ├── yolov11s-face.pt       (wget from GitHub)
    │   └── osnet_x1_0_msmt17.pt  (auto-downloads)
    ├── insightface_models/        # Copy from existing setup
    │   └── models/antelopev2/
    │       ├── 1k3d68.onnx
    │       ├── 2d106det.onnx
    │       ├── genderage.onnx
    │       ├── scrfd_10g_bnkps.onnx
    │       └── w600k_r50.onnx
    ├── vector_db/                 # Copy or run enrollment
    │   ├── faiss.index
    │   └── staff_registry.json
    └── data/                      # Auto-created
        └── YYYY-MM-DD/
            ├── videos/            # Fetched clips
            ├── faces/             # Face images
            ├── tracks/            # Embedding banks
            ├── state.json         # Processing state
            └── report/            # Excel report


SETUP:
    pip install "numpy<2" ultralytics boxmot opencv-python insightface "faiss-cpu<1.8" onnxruntime httpx openpyxl Pillow "matplotlib<3.9"

    # Face model (manual download)
    mkdir -p models
    wget -O models/yolov11s-face.pt https://github.com/akanametov/yolo-face/releases/download/1.0.0/yolov11s-face.pt

    # If using password for SSH (install sshpass)
    sudo apt install sshpass


USAGE:

    # 1. Edit config.json with recording Jetson details and line config

    # 2. Process today's videos (continuous — polls for new clips)
    python pipeline.py

    # 3. Process once and exit
    python pipeline.py --once

    # 4. Process specific date
    python pipeline.py --date 2026-04-06 --once

    # 5. Process local videos (skip SSH fetch)
    python pipeline.py --local-dir /path/to/videos/2026-04-06 --date 2026-04-06

    # 6. Skip FR (tracking + face capture only)
    python pipeline.py --once --no-fr


VIDEO FORMAT:
    Filename: YYYYMMDD_HHMMSS.mp4  (e.g. 20260406_153315.mp4)
    Duration: ~1 minute per clip
    One camera per folder


CROSS-CLIP TRACKING:
    - Tracker state (embedding banks, active IDs) persists between clips
    - If gap between clips > max_lost_seconds (90s), tracks are finalized
    - Consecutive clips (0-1s gap) maintain tracking seamlessly


OUTPUT:
    data/YYYY-MM-DD/
    ├── faces/person_XXXX/face_000001_q87.jpg  # Quality-scored face images
    ├── state.json                              # Full processing state (replaces DB)
    └── report/fr_report_YYYY-MM-DD.xlsx        # Excel report (Staff + Customer + Summary)
