import os
import cv2
from pathlib import Path
from typing import Tuple, Optional
import time

# Color palette for bounding boxes (BGR format)
COLOR_PALETTE = [
    (255, 0, 0),     # Blue
    (0, 255, 0),     # Green  
    (0, 0, 255),     # Red
    (255, 255, 0),   # Cyan
    (255, 0, 255),   # Magenta
    (0, 255, 255),   # Yellow
    (128, 0, 255),   # Purple
    (255, 128, 0),   # Light Blue
    (0, 128, 255),   # Orange
    (128, 255, 0),   # Light Green
]

def ensure_dir(path: str | Path) -> Path:
    """Create directory if it doesn't exist."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path

def validate_video_path(path: str | Path) -> Path:
    """Validate that a video file exists and is readable."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Video file not found: {path}")
    if not path.is_file():
        raise ValueError(f"Path is not a file: {path}")
    
    valid_extensions = {'.mp4', '.avi', '.mkv', '.mov', '.wmv', '.flv', '.webm'}
    if path.suffix.lower() not in valid_extensions:
        raise ValueError(f"Unsupported video format '{path.suffix}'. Supported: {valid_extensions}")
    
    # Try opening with OpenCV
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise IOError(f"Cannot open video file: {path}")
    
    ret, frame = cap.read()
    cap.release()
    if not ret or frame is None:
        raise IOError(f"Video file is empty or corrupt: {path}")
    
    return path

def validate_image_path(path: str | Path) -> Path:
    """Validate that an image file exists and is readable."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Image file not found: {path}")
    if not path.is_file():
        raise ValueError(f"Path is not a file: {path}")
    
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}
    if path.suffix.lower() not in valid_extensions:
        raise ValueError(f"Unsupported image format '{path.suffix}'. Supported: {valid_extensions}")
    
    image = cv2.imread(str(path))
    if image is None:
        raise IOError(f"Cannot read image: {path}")
    
    return path

def format_time(seconds: float) -> str:
    """Format seconds into human-readable time string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = int(seconds // 60)
    secs = seconds % 60
    if minutes < 60:
        return f"{minutes}m {secs:.1f}s"
    hours = int(minutes // 60)
    mins = minutes % 60
    return f"{hours}h {mins}m {secs:.1f}s"

def get_color(track_id: int) -> Tuple[int, int, int]:
    """Get a consistent color for a given track ID."""
    return COLOR_PALETTE[track_id % len(COLOR_PALETTE)]

def draw_detection(frame, box, label: str, color: Tuple[int, int, int], thickness: int = 2):
    """Draw a bounding box with label on a frame."""
    x1, y1, x2, y2 = [int(v) for v in box]
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    
    # Label background
    (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(frame, (x1, y1 - text_h - baseline - 5), (x1 + text_w, y1), color, -1)
    cv2.putText(frame, label, (x1, y1 - baseline - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    return frame

def draw_info_overlay(frame, info: dict):
    """Draw an information overlay on the top-left of the frame.
    
    Args:
        frame: the image/frame to draw on
        info: dict of key-value pairs to display
    """
    y_offset = 25
    for key, value in info.items():
        text = f"{key}: {value}"
        cv2.putText(frame, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        y_offset += 25
    return frame
