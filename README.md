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
14. [Phase 2: Face Recognition & Student Identity Mapping](#phase-2-face-recognition--student-identity-mapping)
15. [Phase 3: Classroom Behavior Analysis](#phase-3-classroom-behavior-analysis)
16. [Next Steps](#next-steps-future-phases)
17. [License](#license)

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
│   │   └── video_processor.py   # Main video processing pipelne
│   ├── evaluation/
│   │   ├── __init__.py
│   │   └── metrics.py           # Performance metrics & reporting
│   └── utils/i
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

## Phase 2: Face Recognition & Student Identity Mapping

Phase 2 extends the YOLO + ByteTrack tracking pipeline with face recognition to map Track IDs to enrolled student identities.

### Phase 2 Architecture

```
Classroom Video
      ↓
OpenCV Video Processing
      ↓
YOLO Person Detection (Phase 1)
      ↓
ByteTrack Tracking (Phase 1)
      ↓
Track IDs
      ↓
Face Detection (InsightFace SCRFD)
      ↓
Face Embedding (ArcFace, 512-dim)
      ↓
Identity Matching (cosine similarity)
      ↓
Temporal Confirmation
      ↓
Track ID → Student ID Mapping
      ↓
Identity Report + Annotated Video
```

### Phase 2 Dependencies

Phase 2 adds two new packages:

```
insightface>=0.7.3    # Face detection + ArcFace embedding extraction
onnxruntime>=1.16.0   # ONNX inference backend (CPU, CoreML on macOS)
```

Install:

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

The InsightFace `buffalo_l` model pack (~326MB) auto-downloads on first use.

### Student Enrollment

Enroll students with face images before running identity recognition:

```bash
# Enroll from individual images
python scripts/enroll_student.py \
    --student-id STU001 \
    --name "Student 001" \
    --images data/students/student_001/img1.jpg data/students/student_001/img2.jpg

# Enroll from a directory of images
python scripts/enroll_student.py \
    --student-id STU001 \
    --name "Student 001" \
    --student-dir data/students/student_001/

# List enrolled students
python scripts/enroll_student.py --student-id _ --name _ --list

# Remove a student
python scripts/enroll_student.py --student-id STU001 --name _ --remove
```

**Enrollment pipeline:**
1. Face detection in each image
2. Quality validation (single face, sufficient size, not blurry)
3. Face embedding extraction (512-dim ArcFace)
4. Multi-image embedding averaging and normalization
5. Storage as `.npy` file + registry update

**Enrollment rejects images when:** no face detected, multiple faces, face too small, face too blurry.

### Running Phase 2 Video Processing

```bash
# Basic usage
python scripts/process_video_identity.py --source data/videos/classroom.mp4

# With custom thresholds
python scripts/process_video_identity.py \
    --source data/videos/classroom.mp4 \
    --recognition-threshold 0.5 \
    --confirmation-frames 5 \
    --recognition-interval 3

# Show real-time display
python scripts/process_video_identity.py \
    --source data/videos/classroom.mp4 \
    --show
```

### Phase 2 Configuration

| Setting | Default | Env Variable | Description |
|---------|---------|-------------|-------------|
| `RECOGNITION_THRESHOLD` | 0.4 | `CLASSSIGHT_RECOGNITION_THRESHOLD` | Cosine similarity threshold for match |
| `CONFIRMATION_FRAMES` | 3 | `CLASSSIGHT_CONFIRMATION_FRAMES` | Consecutive matches to confirm identity |
| `RECOGNITION_INTERVAL` | 5 | `CLASSSIGHT_RECOGNITION_INTERVAL` | Run recognition every N frames |
| `FACE_MIN_SIZE` | 40 | — | Minimum face size in pixels |
| `IDENTITY_RETENTION_FRAMES` | 30 | — | Frames to retain identity after face loss |
| `INSIGHTFACE_MODEL` | `buffalo_l` | `CLASSSIGHT_INSIGHTFACE_MODEL` | InsightFace model pack |

### Identity Report

Phase 2 generates `outputs/reports/identity_report_<timestamp>.json`:

```json
{
    "video": "classroom.mp4",
    "detection_model": "yolo11n.pt",
    "recognition_model": "InsightFace (buffalo_l)",
    "unique_tracks": 25,
    "confirmed_students": 18,
    "unknown_tracks": 4,
    "recognition_attempts": 150,
    "average_fps": 12.4,
    "average_recognition_score": 0.72,
    "identity_mappings": {
        "1": {"student_id": "STU001", "name": "Student 001", "status": "CONFIRMED"},
        "15": {"student_id": null, "name": null, "status": "UNKNOWN"}
    }
}
```

### Track ID → Student ID Mapping

- **Track IDs are NOT student identities.** They are temporary ByteTrack assignments.
- Face recognition provides **identity evidence** that is accumulated over time.
- Identity is **CONFIRMED** only after multiple consecutive face matches (configurable).
- During temporary face loss (occlusion, angle), confirmed identity is **retained** for a configurable number of frames.
- **UNKNOWN** students are honestly labeled — the system never forces a match.

Identity states: `CONFIRMED`, `PENDING`, `UNKNOWN`, `NO_FACE`

### Privacy Considerations

See [PRIVACY.md](PRIVACY.md) for detailed data protection guidance.

Key points:
- All biometric data (face images, embeddings) is stored locally only
- `.gitignore` excludes all student data from version control
- Only numerical embeddings are stored, not raw face images
- No automatic disciplinary decisions are made
- Real deployment requires institutional consent and data protection compliance

### Phase 2 Known Limitations

1. **Face angle**: Side-facing students may not be recognized
2. **Lighting**: Extreme lighting conditions affect face detection quality
3. **Distance**: Very small faces (far from camera) may not meet minimum size threshold
4. **Occlusion**: Partially occluded faces have lower recognition accuracy
5. **Recognition speed**: Face recognition adds processing overhead (~30-50ms per face on CPU)
6. **Model license**: InsightFace models are for non-commercial use
7. **This is NOT attendance**: Confirmed identities should not be automatically interpreted as attendance records

### Running Tests

```bash
# Run all tests (Phase 1 + Phase 2)
python -m pytest tests/ -v

# Phase 1 tests only
python -m pytest tests/test_detector.py tests/test_video.py -v

# Phase 2 tests only (logic tests, no model download required)
python -m pytest tests/test_recognition.py -v -k "not Integration"

# Phase 2 integration tests (requires InsightFace model)
python -m pytest tests/test_recognition.py -v -k "Integration"
```

---

## Phase 3: Classroom Behavior Analysis
 
Phase 3 introduces a modular behavior analysis engine that detects observable physical classroom behaviors for each tracked student, integrating with Phase 1 tracking and Phase 2 student identity mapping.
 
### Phase 3 Architecture
 
```
Classroom Video
      ↓
OpenCV Video Processing
      ↓
YOLO Person Detection & ByteTrack Tracking (Phase 1)
      ↓
Persistent Track IDs
      ↓
Face Recognition & Identity Mapping (Phase 2)
      ↓
Track ID → Student ID
      ↓
┌─────────────────────────────────────────────────────────┐
│               BehaviorEngine (Phase 3)                  │
│                                                         │
│  PoseAnalyzer (YOLO11n-pose: 17 COCO Keypoints)         │
│  PhoneDetector (YOLO11n: COCO Class 67 Cell Phone)      │
│  HandDetector (Wrist-above-shoulder geometry)           │
│  HeadPoseAnalyzer (FORWARD / DOWN / LEFT / RIGHT)       │
│  SleepDetector (Sustained head-down indicator)          │
│  EventManager (Temporal confirmation & duration filter) │
└─────────────────────────────────────────────────────────┘
      ↓
Temporally Confirmed Behavior Events
      ↓
Annotated Video + Behavior Analytics Report (JSON & CSV)
```
 
### Behaviors Analyzed
 
1. **Phone Usage (`phone_usage`)**:
   - Detects cell phones via YOLO pretrained on COCO (class 67).
   - Associates phone detections with the nearest student via wrist keypoint proximity or bounding box overlap.
   - Requires temporal confirmation (default 3 frames) to filter out transient false alarms.
 
2. **Raised Hand (`raised_hand`)**:
   - Evaluates upper-body pose geometry: wrist $Y <$ shoulder $Y$ and elbow $Y <$ shoulder $Y$.
   - Resolution-independent normalization using torso height.
   - Reports `left`, `right`, or `both` hands raised.
 
3. **Head Orientation (`head_forward`, `head_down`, `head_left`, `head_right`)**:
   - Classifies observable head direction using facial keypoints (nose, eyes, ears) relative to shoulder midpoint and width.
   - "Head facing down" indicates physical posture; it is **not** judged as inattention.
 
4. **Possible Sleeping (`possible_sleeping`)**:
   - Sustained `DOWN` head orientation for a configurable duration (default 5.0 seconds).
   - Head position near or below shoulder level.
   - Labeled neutrally as `POSSIBLE_SLEEPING`.
 
### Temporal Confirmation & Event Schema
 
No event is reported from a single frame. The `EventManager` enforces:
- **Confirmation Threshold**: Requires consecutive positive detections before transitioning to active.
- **Minimum Duration**: Discards events shorter than `min_event_duration` (default 0.5s).
- **Grace Period**: Tolerates brief occlusion/misses (gap tolerance) before concluding an event.
 
Standard Event Schema:
```json
{
    "track_id": 2,
    "student_id": "STU001",
    "event_type": "head_right",
    "confidence": 0.9804,
    "start_time": 0.0,
    "end_time": 1.27,
    "duration": 1.27,
    "start_frame": 1,
    "end_frame": 39
}
```
 
### Running Phase 3 Video Processing
 
```bash
# Process video with all three phases active
python scripts/process_video_behavior.py --source data/videos/classroom.mp4
 
# Custom behavior parameters
python scripts/process_video_behavior.py \
    --source data/videos/classroom.mp4 \
    --pose-model yolo11n-pose.pt \
    --phone-threshold 0.3 \
    --phone-confirmation-frames 3 \
    --hand-confirmation-frames 2 \
    --sleep-seconds 5.0 \
    --min-event-duration 0.5
 
# Display visualization window in real time
python scripts/process_video_behavior.py --source data/videos/classroom.mp4 --show
```
 
### Phase 3 Configuration Settings
 
| Setting | Default | Env Variable | Description |
|---------|---------|-------------|-------------|
| `POSE_MODEL_NAME` | `yolo11n-pose.pt` | `CLASSSIGHT_POSE_MODEL` | YOLO pose estimation model weights |
| `PHONE_CLASS_ID` | 67 | — | COCO class ID for cell phone |
| `PHONE_CONFIDENCE` | 0.3 | `CLASSSIGHT_PHONE_CONFIDENCE` | Minimum confidence for phone detection |
| `PHONE_PROXIMITY_PIXELS` | 100 | — | Pixel radius for phone-student association |
| `PHONE_CONFIRMATION_FRAMES` | 3 | `CLASSSIGHT_PHONE_CONFIRMATION_FRAMES` | Consecutive frames to confirm phone event |
| `HAND_RAISE_CONFIRMATION_FRAMES` | 2 | — | Consecutive frames to confirm raised hand |
| `HEAD_DOWN_CONFIRMATION_FRAMES` | 5 | — | Consecutive frames to confirm head-down |
| `SLEEP_CONFIRMATION_SECONDS` | 5.0 | `CLASSSIGHT_SLEEP_CONFIRMATION_SECONDS` | Sustained head-down seconds for sleep indicator |
| `MIN_EVENT_DURATION` | 0.5 | `CLASSSIGHT_MIN_EVENT_DURATION` | Minimum event duration in seconds |
| `BEHAVIOR_INTERVAL` | 1 | — | Run behavior analysis every N frames |
 
### Behavior Analysis Report
 
Phase 3 generates `outputs/reports/behavior_report_<timestamp>.json` and `.csv`:
 
```json
{
    "video": "classroom.mp4",
    "detection_model": "yolo11n.pt",
    "pose_model": "yolo11n-pose.pt",
    "resolution": "1920x1080",
    "total_frames": 500,
    "processed_frames": 500,
    "unique_tracks": 25,
    "identified_students": 18,
    "unknown_tracks": 7,
    "total_behavior_events": 14,
    "event_counts": {
        "raised_hand": 4,
        "phone_usage": 2,
        "head_left": 3,
        "head_right": 5
    },
    "average_fps": 10.5,
    "average_behavior_time_ms": 40.9,
    "per_student_events": {
        "STU001": [ ... ],
        "track_5": [ ... ]
    }
}
```
 
### Phase 3 Known Limitations & Ethical Boundary
 
 1. **No Engagement Scoring**: Engagement scoring and student ranking are intentionally deferred to Phase 4.
 2. **Observable Physical Indicators Only**: The system observes physical posture and items; it makes no claims about attention, interest, or mental state.
 3. **Small Object Limits**: Detecting small smartphones at a distance or behind desks has lower recall with base COCO models.
 4. **Desk Occlusion**: Lower body and arm positions below desk level are occluded and cannot be analyzed.
 5. **Looking Down ≠ Disengaged**: Students reading textbooks or writing notes will trigger `head_down` orientation.
 6. **Processing Speed**: Adding full-frame pose estimation and phone detection lowers CPU processing rate to ~10-12 FPS.
 
 ### Running All Tests
 
 ```bash
 # Run all tests (Phase 1 to Phase 4)
 python -m pytest tests/ -v
 ```
 
 ---
 
 ## Phase 4: Engagement & Attention Analytics
 
 Phase 4 builds an engagement and attention analytics layer on top of Phase 3 behavioral events. It converts observable classroom behaviors into transparent, time-based engagement indicators and classroom analytics using a configurable heuristic model.
 
 ### Phase 4 Architecture
 
 ```
 Temporally Confirmed Behavior Events (Phase 3)
       ↓
 ┌─────────────────────────────────────────────────────────┐
 │             EngagementEngine (Phase 4)                  │
 │                                                         │
 │  FeatureExtractor: Aggregates ratios & rates per track  │
 │  SessionAnalyzer: Classroom distribution & timelines    │
 │  DataQuality: Evaluates min-observation sufficiency     │
 │  HeuristicModel: Weighted linear score [0-100]          │
 └─────────────────────────────────────────────────────────┘
       ↓
 Student Engagement Indicators + Timeline Analytics
       ↓
 JSON / CSV Reports + Matplotlib Visualizations
 ```
 
 ### The Heuristic Engagement Model
 
 Phase 4 uses a **configurable weighted linear model** rather than an opaque deep-learning model. This ensures all scores are fully transparent and explainable.
 
 **Default Configuration:**
 - Base Score: `50.0`
 - Forward-facing ratio: `+0.40`
 - Raised hand rate (per min): `+0.20`
 - Phone usage ratio: `-0.20`
 - Head-down ratio: `-0.10`
 - Sleep indicator ratio: `-0.10`
 
 The final score is clamped between 0 and 100, and classified into:
 - **HIGH** (70–100)
 - **MODERATE** (40–69)
 - **LOW** (0–39)
 
 ### Running Phase 4 Engagement Analysis
 
 You can run Phase 4 on an existing Phase 3 JSON report, or run the full Phase 1-4 video pipeline:
 
 ```bash
 # Run on an existing Phase 3 behavior report (fast)
 python scripts/analyze_engagement.py --report outputs/reports/behavior_report_xxx.json
 
 # Run full video pipeline (Phase 1 → Phase 4)
 python scripts/analyze_engagement.py --source data/videos/classroom.mp4
 
 # Override weights for specific metrics
 python scripts/analyze_engagement.py \
     --report outputs/reports/behavior_report_xxx.json \
     --forward-weight 0.35 \
     --phone-weight -0.30
 ```
 
 ### Data Quality & Privacy Boundaries
 
 - **Insufficient Data Handling:** If a student is tracked for less than `MIN_OBSERVATION_SECONDS` (default: 30s), they receive an `INSUFFICIENT_DATA` label instead of an inaccurate score.
 - **No Individual Ranking:** Reports aggregate student behaviors, but do not rank individual students against each other.
 - **Unknown Students Supported:** Tracks without a recognized face are still analyzed and grouped under their `track_id` (e.g., `track_5`).
 - **Explainable Factors:** Every score output includes a list of `contributing_factors` explaining exactly why the score was given.
 
 ### Output Visualizations
 
 If `matplotlib` is installed, the system automatically generates:
 1. **Engagement Timeline (`.png`)**: Classroom engagement indicator charted over time windows (default 60s windows).
 2. **Engagement Distribution (`.png`)**: Histogram of student scores across LOW/MODERATE/HIGH bands.
 3. **Behavior Breakdown (`.png`)**: Horizontal bar chart of total behavior durations across the classroom.
 
 ---
 
 ---

## Phase 5: Backend & Data Platform

Phase 5 transforms the standalone AI analytical pipeline into a persistent, multi-tenant backend platform. It introduces a normalized relational database schema, Alembic migrations, secure FastAPI REST APIs with JWT authentication, role-based access control (RBAC), and an automated ingestion service that maps Phase 3 and Phase 4 structured analytical reports into relational records.

```
                 CLASSROOM VIDEO
                       ↓
                 ┌─────────────┐
                 │ AI PIPELINE │
                 └─────────────┘
                       ↓
               Phase 1 Detection
                       ↓
                   Tracking
                       ↓
               Phase 2 Identity
                       ↓
               Phase 3 Behavior
                       ↓
              Phase 4 Engagement
                       ↓
                JSON/CSV Results
                       ↓
              ┌─────────────────┐
              │  PHASE 5        │
              │  INGESTION      │
              └─────────────────┘
                       ↓
              PostgreSQL / SQLite
                       ↓
                    FastAPI
                       ↓
              Secure REST APIs
                       ↓
                 PHASE 6
              Web Dashboard
```

### Technology Stack
- **API Framework:** FastAPI 0.110+ (Asynchronous ASGI)
- **ASGI Server:** Uvicorn
- **ORM:** SQLAlchemy 2.0+ (Declarative 2.x models with relationships)
- **Migrations:** Alembic 1.13+
- **Validation:** Pydantic v2 with `pydantic-settings`
- **Security:** Cryptographic `bcrypt` password hashing, PyJWT for signed HMAC-SHA256 tokens
- **Database Engine:** PostgreSQL 16 (production) with SQLite support (testing/development)
- **Testing:** pytest with `TestClient` (HTTPX)

### Database Schema (12 Relational Tables)
The database schema is fully normalized and handles persistence across academic and analytical domains:

| Table | Purpose | Primary Fields |
|---|---|---|
| `users` | User & Instructor Accounts | `id`, `email`, `password_hash`, `full_name`, `role` (ADMIN/TEACHER), `is_active`, `timestamps` |
| `students` | Student Profiles | `id`, `student_id` (e.g. STU001), `name`, `email`, `roll_number`, `is_active`, `timestamps` |
| `classrooms` | Academic Classrooms | `id`, `name`, `room_number`, `description`, `teacher_id` (FK), `timestamps` |
| `classroom_students` | Many-to-Many Enrollment | `id`, `classroom_id` (FK), `student_id` (FK), `enrolled_at`, `is_active`, unique constraint |
| `lectures` | Scheduled / Completed Lectures | `id`, `classroom_id` (FK), `teacher_id` (FK), `title`, `subject`, `start_time`, `end_time`, `status` |
| `processing_sessions` | AI Processing Runs | `id`, `lecture_id` (FK), `idempotency_key` (unique/indexed), `video_source`, `status`, `processing_fps`, `total_frames`, `processed_frames`, `error_message` |
| `tracks` | ByteTrack Detections in Session | `id`, `processing_session_id` (FK), `track_id`, `student_id` (FK nullable), `visible_duration`, `identity_status` |
| `attendance` | Derived Attendance Records | `id`, `lecture_id` (FK), `student_id` (FK), `first_seen_at`, `last_seen_at`, `visible_duration`, `status` (PRESENT/ABSENT/UNCERTAIN), `confidence`, `UNIQUE(lecture_id, student_id)` |
| `behavior_events` | Observable Behavioral Events | `id`, `processing_session_id` (FK), `track_id`, `student_id` (FK nullable), `event_type`, `confidence`, `start_time`, `end_time`, `duration`, `frames` |
| `engagement_metrics` | Student Engagement Metrics | `id`, `processing_session_id` (FK), `student_id` (FK), `track_id`, `visible_duration`, `forward_ratio`, `head_down_ratio`, `phone_ratio`, `raised_hand_count`, `raised_hand_rate`, `sleep_ratio`, `engagement_score`, `engagement_band`, `data_quality`, `contributing_factors` |
| `engagement_timeline`| Classroom Time Windows | `id`, `processing_session_id` (FK), `window_start`, `window_end`, `engagement_indicator`, `student_count`, `participation_count`, `phone_event_count`, `head_down_duration`, `sleep_event_count` |
| `reports` | Report Artifact Metadata | `id`, `processing_session_id` (FK), `report_type` (BEHAVIOR/ENGAGEMENT/ATTENDANCE/SUMMARY), `file_path` |

### Environment Configuration
Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```
Replace the placeholder database password and JWT secret in `.env` before starting PostgreSQL or the backend. Do not commit `.env`.
Key configuration parameters:
- `DATABASE_URL`: PostgreSQL connection string (or `sqlite:///./classsight.db`)
- `JWT_SECRET_KEY`: Minimum 32-byte secret key for signing authentication tokens
- `JWT_ACCESS_TOKEN_EXPIRE_MINUTES`: Token validity period (default 1440 mins / 24 hours)
- `MIN_ATTENDANCE_DURATION`: Configurable duration threshold (seconds) to confirm student presence (default 30.0s)
- `MIN_ATTENDANCE_OBSERVATIONS`: Minimum positive recognition events to verify attendance (default 2)

### Running PostgreSQL with Docker
To launch a dedicated local PostgreSQL database:
```bash
docker compose up -d
```
To check container status:
```bash
docker compose ps
```

### Database Migrations with Alembic
Apply all pending schema migrations:
```bash
alembic upgrade head
```
To roll back the last migration:
```bash
alembic downgrade -1
```

### Running the FastAPI Backend
Start the backend server with live reload:
```bash
uvicorn src.backend.main:app --host 0.0.0.0 --port 8000 --reload
```
Interactive API documentation will be available at:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

### REST API Endpoints Overview (46 Routes / 55 HTTP Operations)

#### Authentication & Users
- `POST /api/auth/register` — Register a new teacher or admin account
- `POST /api/auth/login` — Authenticate and receive a JWT access token
- `GET /api/auth/me` — Inspect profile of the authenticated user
- `GET /api/users` — List user accounts (Admin only)
- `GET /api/users/{id}` — Get user details (Admin only)

#### Student Management
- `GET /api/students` — List enrolled students
- `POST /api/students` — Register a new student
- `GET /api/students/{id}` — Retrieve student details
- `PATCH /api/students/{id}` — Update student record
- `DELETE /api/students/{id}` — Remove student record
- `GET /api/students/{id}/attendance` — Student attendance history
- `GET /api/students/{id}/behavior` — Student behavior event history
- `GET /api/students/{id}/engagement` — Student engagement scores & features

#### Classroom Management (Teacher-Isolated)
- `GET /api/classrooms` — List classrooms owned by teacher (or all for Admin)
- `POST /api/classrooms` — Create a new classroom
- `GET /api/classrooms/{id}` — Retrieve classroom details
- `PATCH /api/classrooms/{id}` — Update classroom information
- `DELETE /api/classrooms/{id}` — Remove classroom
- `GET /api/classrooms/{id}/students` — List students enrolled in classroom
- `POST /api/classrooms/{id}/students` — Enroll student in classroom
- `DELETE /api/classrooms/{id}/students/{student_id}` — Remove student from classroom
- `GET /api/classrooms/{id}/attendance/summary` — Aggregate attendance rate
- `GET /api/classrooms/{id}/behavior/summary` — Aggregate behavior counts & durations
- `GET /api/classrooms/{id}/engagement/summary` — Aggregate engagement distribution

#### Lectures & Analytics
- `GET /api/lectures` — List lectures for teacher
- `POST /api/lectures` — Schedule or register a lecture
- `GET /api/lectures/{id}` — Retrieve lecture details
- `PATCH /api/lectures/{id}` — Update lecture status
- `GET /api/lectures/{id}/attendance` — Attendance records for lecture
- `GET /api/lectures/{id}/attendance/{student_id}` — Specific student attendance
- `GET /api/lectures/{id}/behavior` — Filterable behavior events (`event_type`, `student_id`, time)
- `GET /api/lectures/{id}/engagement` — Student engagement records
- `GET /api/lectures/{id}/engagement/timeline` — Time-window engagement timeline
- `GET /api/lectures/{id}/summary` — Comprehensive lecture summary

#### Processing & AI Report Ingestion
- `POST /api/processing/sessions` — Create processing session record
- `GET /api/processing/sessions/{id}` — Inspect processing session
- `GET /api/processing/sessions/{id}/status` — Check processing progress & status
- `POST /api/processing/sessions/ingest` — Ingest Phase 3 / Phase 4 JSON reports (Idempotent)
- `POST /api/processing/sessions/{id}/ingest` — Ingest report into specific session

#### Report Exporting
- `GET /api/lectures/{id}/reports/behavior?format=json|csv` — Export behavior events
- `GET /api/lectures/{id}/reports/engagement?format=json|csv` — Export engagement metrics
- `GET /api/lectures/{id}/reports/attendance?format=json|csv` — Export attendance sheet
- `GET /api/lectures/{id}/reports/summary?format=json|csv` — Export executive summary

#### Health & Status
- `GET /health` — Service and database health check
- `GET /api/health` — API health check

### Automated Ingestion, Idempotency & Attendance Derivation
The hardened ingestion service (`src/backend/services/ingestion_service.py`) automatically maps JSON outputs from Phase 3 (`behavior_report_*.json`) and Phase 4 (`engagement_report_*.json`):
1. **Idempotency Protection:** Generates a deterministic SHA-256 fingerprint from `lecture_id` and report content. Repeated ingestion returns `is_duplicate=True` without duplicating child records.
2. **Tracks & Students:** Resolves student IDs against the enrolled student registry.
3. **Behavior Events:** Ingests individual events with start/end timestamps and confidence. Supports attendance derivation from behavior-only events when student summaries are absent.
4. **Engagement Metrics:** Stores normalized ratios, engagement scores, quality bands, and explainable contributing factors.
5. **Attendance Logic & Upsert:** Evaluates confirmed visible duration against configurable thresholds using upsert on `(lecture_id, student_id)`:
   - Visible $\ge$ `MIN_ATTENDANCE_DURATION` or observations $\ge$ `MIN_ATTENDANCE_OBSERVATIONS`: Marked **PRESENT**.
   - Visible $> 0$ but below threshold: Marked **UNCERTAIN**.
   - Enrolled in classroom but not detected: Marked **ABSENT**.
   - `present_count` in lecture summary strictly checks enrolled students to prevent attendance rates exceeding 100%.

### Verification & Testing
Run the complete test suite across all phases:
```bash
python -m pytest tests/ -v
```
All **171 tests** (136 AI Pipeline + 35 Backend & Integration) pass with 100% success.

To run the automated end-to-end semantic integration verification:
```bash
python scripts/verify_e2e_backend.py
```

---

## Future Roadmap (Phase 6)

With Phase 5 complete, the backend is ready for **Phase 6: Web Dashboard**:
- React / Next.js / Tailwind CSS web interface
- Real-time classroom lecture monitoring
- Interactive student engagement timeline charts
- Behavior breakdown heatmaps and attendance rosters
- One-click PDF/CSV lecture analytical reports

---

## License

This project is for educational and research purposes.

---

*ClassSight AI – Phase 1-5 | Detection, Identity, Behavior, Engagement & Persistent Backend Platform*
# ClassSight
# ClassSighttt
# ClassSighttt
# ClassSighttt
# ClassSight
# ClassSight
# ClassSight
# ClassSight
# ClassSight
# ClassSight
