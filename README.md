# ClassSight AI – AI-Powered Classroom Analytics Platform

## Phase 1: Person Detection & Tracking Foundation

ClassSight AI is an AI-powered classroom analytics platform that uses computer vision to detect and track students in classroom video footage. This is **Phase 1**, which establishes the foundational pipeline for person detection and tracking.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Phase 1 Objective](#phase-1-objective)
3. [Architecture](#architecture)
4. [Installation](#installation)
5. [Virtual Environment Setup](#virtual-environment-setup)
6. [Dependency Installation](#dependency-installation)
7. [Configuring Dataset Paths](#configuring-dataset-paths)
8. [Testing Image Detection](#testing-image-detection)
9. [Processing Video](#processing-video)
10. [Understanding Tracking IDs](#understanding-tracking-ids)
11. [Performance Metrics](#performance-metrics)
12. [Known Limitations](#known-limitations)
13. [Git Workflow](#git-workflow)

---

## Project Overview

ClassSight AI uses pretrained YOLO models with ByteTrack to detect and track persons in classroom environments. The system processes video frame-by-frame, assigns persistent tracking IDs to each detected person, and generates performance reports.

**Technology Stack:**
- **Language:** Python 3.11+
- **Computer Vision:** OpenCV, NumPy
- **Object Detection:** Ultralytics YOLO (pretrained, COCO person class)
- **Tracking:** ByteTrack (via Ultralytics built-in support)
- **Reporting:** JSON, CSV (optional Pandas)

---

## Phase 1 Objective

Build and validate the core detection and tracking pipeline:

1. ✅ Read and process classroom images/video with OpenCV
2. ✅ Detect persons using pretrained YOLO (nano model)
3. ✅ Filter detections to COCO "person" class only
4. ✅ Track persons across frames with ByteTrack (persistent IDs)
5. ✅ Display bounding boxes and tracking IDs
6. ✅ Measure and report performance metrics
7. ✅ Save processed output video and JSON/CSV reports

**Not implemented in Phase 1:** Face recognition, attendance, phone/sleep/hand-raise detection, engagement scoring, database, backend APIs, dashboard.

---

## Architecture

```
Classroom Video / Image
        ↓
    OpenCV Video Input
        ↓
  YOLO Person Detection (pretrained nano model)
        ↓
  ByteTrack Tracking (persist=True)
        ↓
  Persistent Track IDs
        ↓
  Annotated Output Video + Performance Metrics
```

### Project Structure

```
classsight-ai/
│
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── .gitignore                   # Git exclusions
├── config.py                    # Central configuration
│
├── src/                         # Source package
│   ├── __init__.py
│   ├── detection/
│   │   ├── __init__.py
│   │   └── yolo_detector.py     # YOLO person detection
│   ├── tracking/
│   │   ├── __init__.py
│   │   └── student_tracker.py   # ByteTrack integration
│   ├── video/
│   │   ├── __init__.py
│   │   └── video_processor.py   # Main video processing pipeline
│   ├── evaluation/
│   │   ├── __init__.py
│   │   └── metrics.py           # Performance metrics & reporting
│   └── utils/
│       ├── __init__.py
│       └── helpers.py           # Utility functions
│
├── scripts/                     # CLI entry points
│   ├── test_image.py            # Single image detection
│   └── process_video.py         # Video processing with tracking
│
├── data/                        # Input data (not committed)
│   ├── README.md
│   ├── images/
│   └── videos/
│
├── models/                      # Model weights (auto-downloaded)
│   └── README.md
│
├── outputs/                     # Generated outputs (not committed)
│   ├── images/
│   ├── videos/
│   └── reports/
│
└── tests/                       # Unit tests
    ├── test_detector.py
    └── test_video.py
```

---

## Installation

### Prerequisites

- Python 3.11 or later
- pip
- Git

### Virtual Environment Setup

**macOS / Linux:**
```bash
cd classsight-ai
python3 -m venv .venv
source .venv/bin/activate
```

**Windows:**
```bash
cd classsight-ai
python -m venv .venv
.venv\Scripts\activate
```

### Dependency Installation

```bash
pip install -r requirements.txt
```

### Verify Installation

```bash
python --version
python -c "import cv2; print(f'OpenCV: {cv2.__version__}')"
python -c "from ultralytics import YOLO; print('Ultralytics: OK')"
python -c "import numpy; print(f'NumPy: {numpy.__version__}')"
```

---

## Configuring Dataset Paths

Paths can be configured in three ways (in priority order):

### 1. Environment Variables

```bash
export CLASSSIGHT_TEST_IMAGE="path/to/test/image.jpg"
export CLASSSIGHT_TEST_VIDEO="path/to/test/video.mp4"
export CLASSSIGHT_OUC_CGE_ROOT="path/to/OUC-CGE/dataset"
export CLASSSIGHT_MODEL="yolo11n.pt"
export CLASSSIGHT_CONF="0.4"
export CLASSSIGHT_IOU="0.5"
```

### 2. Edit `config.py`

Modify the default values directly in `config.py`.

### 3. Command-Line Arguments

Pass paths as arguments to the scripts (see usage sections below).

### Using OUC-CGE Dataset

The OUC-CGE dataset should **not** be placed inside the project directory. Instead, set the path:

```bash
export CLASSSIGHT_OUC_CGE_ROOT="/path/to/your/OUC-CGE"
```

Or specify video paths directly via `--source` when running scripts.

---

## Testing Image Detection

Run person detection on a single image:

```bash
python scripts/test_image.py --source data/images/classroom.jpg
```

### Options

| Argument   | Default      | Description                      |
|------------|--------------|----------------------------------|
| `--source` | *(required)* | Path to input image              |
| `--model`  | `yolo11n.pt` | YOLO model name                  |
| `--conf`   | `0.4`        | Confidence threshold             |
| `--iou`    | `0.5`        | IoU threshold for NMS            |
| `--output` | auto         | Output image path                |
| `--show`   | `False`      | Display result in a window       |

### Example Output

```
[INFO] Processing image: classroom.jpg
[INFO] Image size: 1920x1080

==================================================
Detection Results
==================================================
Persons detected: 28
Inference time: 45.3ms

  Person 1: confidence=0.9234, bbox=[120, 85, 210, 340]
  Person 2: confidence=0.8891, bbox=[350, 92, 430, 355]
  ...
==================================================

[INFO] Output saved: outputs/images/classroom_detected.jpg
```

---

## Processing Video

Process a classroom video with detection and ByteTrack tracking:

```bash
python scripts/process_video.py --source "data/videos/classroom.mp4"
```

### Full Example

```bash
python scripts/process_video.py \
    --source "data/videos/classroom.mp4" \
    --output "outputs/videos/classroom_tracked.mp4" \
    --conf 0.4 \
    --iou 0.5 \
    --frame-skip 1 \
    --show
```

### Options

| Argument       | Default      | Description                           |
|----------------|--------------|---------------------------------------|
| `--source`     | *(required)* | Path to input video                   |
| `--output`     | auto         | Output video path                     |
| `--model`      | `yolo11n.pt` | YOLO model name                       |
| `--conf`       | `0.4`        | Confidence threshold                  |
| `--iou`        | `0.5`        | IoU threshold for NMS                 |
| `--frame-skip` | `1`          | Process every Nth frame               |
| `--show`       | `False`      | Display video window (press Q to quit)|

### Frame Skipping

- `--frame-skip 1` — Process every frame (highest quality, slower)
- `--frame-skip 2` — Process every 2nd frame (faster, less smooth)
- `--frame-skip 5` — Process every 5th frame (fastest, may miss tracks)

---

## Understanding Tracking IDs

### What Track IDs Mean

- Each detected person is assigned a **Track ID** (e.g., `ID: 1`, `ID: 2`)
- IDs are assigned by the ByteTrack algorithm based on motion/position
- IDs **persist across frames** — the same person keeps the same ID

### What Track IDs Do NOT Mean

> ⚠️ **Track IDs are NOT student identities.**

- `ID: 1` does **not** mean "Student named Rahul"
- `ID: 1` means "the first person the tracker started tracking"
- If a person leaves and returns, they may get a **new ID**
- **Total unique IDs ≠ attendance count**

Identity recognition (face recognition, name mapping) will be added in a future phase.

---

## Performance Metrics

After processing a video, a JSON report is automatically generated in `outputs/reports/`.

### Report Contents

```json
{
    "video": "classroom_sample.mp4",
    "model": "yolo11n.pt",
    "resolution": "1920x1080",
    "total_frames": 5000,
    "processed_frames": 5000,
    "average_fps": 18.4,
    "average_confidence": 0.82,
    "average_visible_students": 27.3,
    "max_visible_students": 32,
    "unique_track_ids": 41,
    "total_processing_time_seconds": 271.74,
    "processing_date": "2024-01-15T14:30:00",
    "low_confidence_frames": 12
}
```

### Metrics Explained

| Metric                    | Description                                           |
|---------------------------|-------------------------------------------------------|
| `average_fps`             | Average processing speed                              |
| `average_confidence`      | Mean detection confidence across all detections       |
| `average_visible_students`| Mean number of detected persons per frame             |
| `max_visible_students`    | Peak detection count in any single frame              |
| `unique_track_ids`        | Total unique tracker IDs assigned (NOT attendance)    |
| `low_confidence_frames`   | Frames with at least one detection below threshold    |

---

## Known Limitations

1. **Track IDs ≠ Student Identities.** ByteTrack assigns numeric IDs based on motion tracking, not face/identity recognition.

2. **Unique Track IDs ≠ Attendance Count.** A person who leaves and re-enters may be assigned a new ID. The total unique ID count can exceed the actual number of people.

3. **YOLO Does Not Understand Behavior.** Person detection identifies *where* people are, not *what* they are doing. Engagement, attention, and activity analysis are not part of Phase 1.

4. **Detection Confidence ≠ Accuracy.** A high confidence score means the model is confident in its prediction, not that the detection is necessarily correct. Always validate on real classroom footage.

5. **Engagement is NOT Implemented.** No behavior analysis, hand-raise detection, phone detection, or sleeping detection exists in Phase 1.

6. **OUC-CGE Compatibility.** The OUC-CGE dataset may not contain annotations compatible with person detection evaluation. Only inference performance and confidence statistics are measured unless compatible ground-truth annotations are provided.

7. **Real Classroom Variability.** Performance depends on camera angle, lighting, occlusion, student density, and image resolution. Results on sample videos may not generalize to all classrooms.

8. **Occlusion and Dense Scenes.** Students sitting behind others or in tightly packed rows may be missed or merged. ByteTrack tracking may lose IDs during heavy occlusion.

---

## Git Workflow

### Initial Setup

```bash
git init
git add .
git commit -m "Phase 1: Person detection and tracking foundation"
```

### What's Tracked

- ✅ Source code (`src/`, `scripts/`, `tests/`)
- ✅ Configuration (`config.py`, `requirements.txt`)
- ✅ Documentation (`README.md`, data/README.md, models/README.md)
- ✅ `.gitignore`

### What's NOT Tracked

- ❌ Virtual environment (`.venv/`)
- ❌ Model weights (`models/*.pt`)
- ❌ Datasets (`data/ouc-cge/`, `datasets/`)
- ❌ Output files (`outputs/videos/`, `outputs/reports/`)
- ❌ Python cache (`__pycache__/`, `*.pyc`)

### Branch Strategy (Recommended)

```bash
git checkout -b phase-1/detection-tracking
# ... make changes ...
git add .
git commit -m "feat: add YOLO detection module"
git push origin phase-1/detection-tracking
```

---

## Running Tests

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test file
python -m pytest tests/test_detector.py -v
python -m pytest tests/test_video.py -v
```

---

## Next Steps (Future Phases)

Phase 1 provides the stable foundation for:

- **Phase 2:** Face recognition and student identity mapping
- **Phase 3:** Behavior analysis (hand raises, phone use, sleeping)
- **Phase 4:** Engagement scoring and analytics dashboard
- **Phase 5:** Database integration and API backend

---

## License

This project is for educational and research purposes.

---

*ClassSight AI – Phase 1 | Person Detection & Tracking Foundation*
