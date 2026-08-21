"""Tests for the YOLO person detector."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import numpy as np
import cv2

from src.detection.yolo_detector import YOLODetector


class TestYOLODetector:
    """Tests for YOLODetector class."""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """Initialize detector for tests."""
        self.detector = YOLODetector(
            model_name='yolo11n.pt',
            confidence=0.3,
            iou=0.5
        )
    
    def test_initialization(self):
        """Test that detector initializes correctly."""
        assert self.detector.model is not None
        assert self.detector.confidence == 0.3
        assert self.detector.iou == 0.5
        assert self.detector.PERSON_CLASS_ID == 0
    
    def test_detect_returns_correct_keys(self):
        """Test that detect() returns dict with expected keys."""
        # Create a blank test image
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = self.detector.detect(frame)
        
        assert 'boxes' in result
        assert 'confidences' in result
        assert 'count' in result
        assert 'inference_time' in result
        assert 'results' in result
    
    def test_detect_on_blank_image(self):
        """Test detection on a blank image returns no detections."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = self.detector.detect(frame)
        
        assert result['count'] == 0
        assert len(result['boxes']) == 0
        assert len(result['confidences']) == 0
        assert result['inference_time'] > 0
    
    def test_detect_filters_person_class(self):
        """Test that only person class is detected."""
        # The detector should only detect persons (class 0)
        # On a blank image, no objects should be detected
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = self.detector.detect(frame)
        assert result['count'] == 0
    
    def test_confidence_threshold(self):
        """Test that confidence threshold filters low-confidence detections."""
        high_conf_detector = YOLODetector(confidence=0.9)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = high_conf_detector.detect(frame)
        
        # All returned confidences should be above threshold
        for conf in result['confidences']:
            assert conf >= 0.9
    
    def test_detect_returns_valid_boxes(self):
        """Test that returned boxes have valid format."""
        # Create an image with some content to potentially trigger detections
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        result = self.detector.detect(frame)
        
        for box in result['boxes']:
            assert len(box) == 4
            x1, y1, x2, y2 = box
            assert x2 > x1  # width > 0
            assert y2 > y1  # height > 0
    
    def test_invalid_model_raises_error(self):
        """Test that invalid model name raises an error."""
        with pytest.raises(RuntimeError):
            YOLODetector(model_name='nonexistent_model.pt')
