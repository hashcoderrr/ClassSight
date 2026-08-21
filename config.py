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

    def __post_init__(self) -> None:
        """Initialize dependent paths and apply environment variable overrides."""
        self.DATA_DIR = self.ROOT_DIR / "data"
        self.MODELS_DIR = self.ROOT_DIR / "models"
        self.OUTPUT_DIR = self.ROOT_DIR / "outputs"
        self.OUTPUT_IMAGES = self.OUTPUT_DIR / "images"
        self.OUTPUT_VIDEOS = self.OUTPUT_DIR / "videos"
        self.OUTPUT_REPORTS = self.OUTPUT_DIR / "reports"

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


def get_config() -> Config:
    """Instantiate and return the configuration instance.

    Returns:
        Config: Initialized Config object containing all runtime settings.
    """
    return Config()
