"""Configuration module for ClassSight AI.

Provides centralized configuration management with environment variable overrides
and default path resolution.
"""

from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import Optional


@dataclass
class Config:
    """Configuration settings for the ClassSight AI platform.

    Attributes:
        ROOT_DIR: Base project directory root.
        DATA_DIR: Base directory for dataset and input media files.
        MODELS_DIR: Directory where model weights are stored.
        OUTPUT_DIR: Base directory for all generated outputs.
        OUTPUT_IMAGES: Subdirectory for output images and visualizations.
        OUTPUT_VIDEOS: Subdirectory for output processed videos.
        OUTPUT_REPORTS: Subdirectory for analytics and evaluation reports.
        TEST_IMAGE_PATH: Path to default test image file.
        TEST_VIDEO_PATH: Path to default test video file.
        OUC_CGE_ROOT: Path to OUC-CGE dataset root directory.
        MODEL_NAME: YOLO model identifier or local weights filename.
        CONFIDENCE_THRESHOLD: Minimum confidence score for person detections.
        IOU_THRESHOLD: Intersection-over-Union threshold for Non-Maximum Suppression.
        PERSON_CLASS_ID: COCO dataset class index corresponding to person.
        FRAME_SKIP: Number of frames to step between processed video frames.
        SHOW_DISPLAY: Whether to display a real-time OpenCV rendering window.

        # Phase 2: Face Recognition settings
        STUDENTS_DIR: Path to enrolled student image folders.
        IDENTITIES_FILE: Path to student identity registry JSON.
        EMBEDDINGS_DIR: Path to stored face embeddings.
        RECOGNITION_THRESHOLD: Cosine similarity threshold for identity match.
        CONFIRMATION_FRAMES: Consecutive recognitions needed to confirm identity.
        RECOGNITION_INTERVAL: Run face recognition every N processed frames.
        FACE_MIN_SIZE: Minimum face size in pixels to attempt recognition.
        IDENTITY_RETENTION_FRAMES: Frames to retain identity after face loss.
        INSIGHTFACE_MODEL: InsightFace model pack name.
    """

    ROOT_DIR: Path = field(default_factory=lambda: Path(__file__).resolve().parent)

    # Path settings
    DATA_DIR: Path = field(init=False)
    MODELS_DIR: Path = field(init=False)
    OUTPUT_DIR: Path = field(init=False)
    OUTPUT_IMAGES: Path = field(init=False)
    OUTPUT_VIDEOS: Path = field(init=False)
    OUTPUT_REPORTS: Path = field(init=False)

    TEST_IMAGE_PATH: Path = field(init=False)
    TEST_VIDEO_PATH: Path = field(init=False)
    OUC_CGE_ROOT: Optional[Path] = field(init=False)

    # Model settings
    MODEL_NAME: str = "yolo11n.pt"
    CONFIDENCE_THRESHOLD: float = 0.4
    IOU_THRESHOLD: float = 0.5
    PERSON_CLASS_ID: int = 0

    # Processing settings
    FRAME_SKIP: int = 1
    SHOW_DISPLAY: bool = False

    # Phase 2: Face Recognition settings
    STUDENTS_DIR: Path = field(init=False)
    IDENTITIES_FILE: Path = field(init=False)
    EMBEDDINGS_DIR: Path = field(init=False)
    RECOGNITION_THRESHOLD: float = 0.4
    CONFIRMATION_FRAMES: int = 3
    RECOGNITION_INTERVAL: int = 5
    FACE_MIN_SIZE: int = 40
    IDENTITY_RETENTION_FRAMES: int = 30
    INSIGHTFACE_MODEL: str = "buffalo_l"

    # Phase 3: Behavior Analysis settings
    POSE_MODEL_NAME: str = "yolo11n-pose.pt"
    PHONE_CLASS_ID: int = 67
    PHONE_CONFIDENCE: float = 0.3
    PHONE_PROXIMITY_PIXELS: int = 100
    PHONE_CONFIRMATION_FRAMES: int = 3
    HAND_RAISE_CONFIRMATION_FRAMES: int = 2
    HEAD_DOWN_CONFIRMATION_FRAMES: int = 5
    SLEEP_CONFIRMATION_SECONDS: float = 5.0
    MIN_EVENT_DURATION: float = 0.5
    BEHAVIOR_INTERVAL: int = 1

    # Phase 4: Engagement Analytics settings
    ENGAGEMENT_WEIGHTS: dict = field(default_factory=lambda: {
        "forward_ratio": 0.40,
        "raised_hand_rate": 0.20,
        "phone_ratio": -0.20,
        "head_down_ratio": -0.10,
        "sleep_ratio": -0.10,
    })
    ENGAGEMENT_BASE_SCORE: float = 50.0
    ENGAGEMENT_TIME_WINDOW: float = 60.0
    MIN_OBSERVATION_SECONDS: float = 30.0
    MIN_DATA_COVERAGE: float = 0.30
    RAISED_HAND_RATE_CAP: float = 1.0
    ENGAGEMENT_LEVEL_THRESHOLDS: dict = field(default_factory=lambda: {
        "LOW": 0,
        "MODERATE": 40,
        "HIGH": 70,
    })

    def __post_init__(self) -> None:
        """Initialize dependent paths and apply environment variable overrides."""
        self.DATA_DIR = self.ROOT_DIR / "data"
        self.MODELS_DIR = self.ROOT_DIR / "models"
        self.OUTPUT_DIR = self.ROOT_DIR / "outputs"
        self.OUTPUT_IMAGES = self.OUTPUT_DIR / "images"
        self.OUTPUT_VIDEOS = self.OUTPUT_DIR / "videos"
        self.OUTPUT_REPORTS = self.OUTPUT_DIR / "reports"

        # Phase 2 paths
        self.STUDENTS_DIR = self.DATA_DIR / "students"
        self.IDENTITIES_FILE = self.DATA_DIR / "identities" / "students.json"
        self.EMBEDDINGS_DIR = self.DATA_DIR / "identities" / "embeddings"

        # Apply environment variable overrides if provided
        test_image_env = os.getenv("CLASSSIGHT_TEST_IMAGE")
        self.TEST_IMAGE_PATH = (
            Path(test_image_env)
            if test_image_env
            else self.DATA_DIR / "images" / "test.jpg"
        )

        test_video_env = os.getenv("CLASSSIGHT_TEST_VIDEO")
        self.TEST_VIDEO_PATH = (
            Path(test_video_env)
            if test_video_env
            else self.DATA_DIR / "videos" / "test.mp4"
        )

        ouc_env = os.getenv("CLASSSIGHT_OUC_CGE_ROOT")
        self.OUC_CGE_ROOT = (
            Path(ouc_env) if ouc_env else self.DATA_DIR / "ouc-cge"
        )

        model_env = os.getenv("CLASSSIGHT_MODEL")
        if model_env:
            self.MODEL_NAME = model_env

        conf_env = os.getenv("CLASSSIGHT_CONF")
        if conf_env:
            try:
                self.CONFIDENCE_THRESHOLD = float(conf_env)
            except ValueError:
                pass

        iou_env = os.getenv("CLASSSIGHT_IOU")
        if iou_env:
            try:
                self.IOU_THRESHOLD = float(iou_env)
            except ValueError:
                pass

        # Phase 2 environment variable overrides
        recog_thresh_env = os.getenv("CLASSSIGHT_RECOGNITION_THRESHOLD")
        if recog_thresh_env:
            try:
                self.RECOGNITION_THRESHOLD = float(recog_thresh_env)
            except ValueError:
                pass

        confirm_env = os.getenv("CLASSSIGHT_CONFIRMATION_FRAMES")
        if confirm_env:
            try:
                self.CONFIRMATION_FRAMES = int(confirm_env)
            except ValueError:
                pass

        recog_interval_env = os.getenv("CLASSSIGHT_RECOGNITION_INTERVAL")
        if recog_interval_env:
            try:
                self.RECOGNITION_INTERVAL = int(recog_interval_env)
            except ValueError:
                pass

        insightface_env = os.getenv("CLASSSIGHT_INSIGHTFACE_MODEL")
        if insightface_env:
            self.INSIGHTFACE_MODEL = insightface_env

        # Phase 3 environment variable overrides
        pose_model_env = os.getenv("CLASSSIGHT_POSE_MODEL")
        if pose_model_env:
            self.POSE_MODEL_NAME = pose_model_env

        phone_conf_env = os.getenv("CLASSSIGHT_PHONE_CONFIDENCE")
        if phone_conf_env:
            try:
                self.PHONE_CONFIDENCE = float(phone_conf_env)
            except ValueError:
                pass

        phone_confirm_env = os.getenv("CLASSSIGHT_PHONE_CONFIRMATION_FRAMES")
        if phone_confirm_env:
            try:
                self.PHONE_CONFIRMATION_FRAMES = int(phone_confirm_env)
            except ValueError:
                pass

        sleep_confirm_env = os.getenv("CLASSSIGHT_SLEEP_CONFIRMATION_SECONDS")
        if sleep_confirm_env:
            try:
                self.SLEEP_CONFIRMATION_SECONDS = float(sleep_confirm_env)
            except ValueError:
                pass

        min_event_env = os.getenv("CLASSSIGHT_MIN_EVENT_DURATION")
        if min_event_env:
            try:
                self.MIN_EVENT_DURATION = float(min_event_env)
            except ValueError:
                pass

        # Phase 4 environment variable overrides
        time_window_env = os.getenv("CLASSSIGHT_ENGAGEMENT_TIME_WINDOW")
        if time_window_env:
            try:
                self.ENGAGEMENT_TIME_WINDOW = float(time_window_env)
            except ValueError:
                pass

        min_obs_env = os.getenv("CLASSSIGHT_MIN_OBSERVATION_SECONDS")
        if min_obs_env:
            try:
                self.MIN_OBSERVATION_SECONDS = float(min_obs_env)
            except ValueError:
                pass


def get_config() -> Config:
    """Instantiate and return the configuration instance.

    Returns:
        Config: Initialized Config object containing all runtime settings.
    """
    return Config()

