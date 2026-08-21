"""Tests for video processing components."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import numpy as np
import cv2
import json
import tempfile
import os

from src.video.video_processor import VideoProcessor
from src.evaluation.metrics import PerformanceMetrics
from src.tracking.student_tracker import StudentTracker
from src.utils.helpers import ensure_dir, validate_video_path, format_time


def create_synthetic_video(path: str, num_frames: int = 30, 
                            width: int = 640, height: int = 480, 
                            fps: float = 30.0):
    """Create a synthetic test video."""
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(path, fourcc, fps, (width, height))
    
    for i in range(num_frames):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        # Draw some rectangles to simulate content
        cv2.rectangle(frame, (100, 100), (200, 350), (0, 255, 0), -1)
        cv2.rectangle(frame, (300, 80), (400, 340), (0, 0, 255), -1)
        # Add frame number
        cv2.putText(frame, f"Frame {i+1}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        out.write(frame)
    
    out.release()


class TestPerformanceMetrics:
    """Tests for PerformanceMetrics class."""
    
    def test_initialization(self):
        metrics = PerformanceMetrics()
        assert metrics.total_frames_processed == 0
        assert metrics.average_fps == 0.0
        assert metrics.average_confidence == 0.0
    
    def test_update(self):
        metrics = PerformanceMetrics()
        metrics.update(
            frame_time=0.05,
            confidences=[0.8, 0.9],
            student_count=2,
            track_ids=[1, 2],
            frame_number=1
        )
        
        assert metrics.total_frames_processed == 1
        assert metrics.unique_track_ids == 2
        assert metrics.current_fps == pytest.approx(20.0, rel=0.1)
    
    def test_average_metrics(self):
        metrics = PerformanceMetrics()
        metrics.update(0.05, [0.8], 1, [1], 1)
        metrics.update(0.1, [0.6], 1, [2], 2)
        
        assert metrics.average_confidence == pytest.approx(0.7, rel=0.01)
        assert metrics.unique_track_ids == 2
    
    def test_generate_report(self):
        metrics = PerformanceMetrics()
        metrics.start()
        metrics.update(0.05, [0.8, 0.9], 2, [1, 2], 1)
        metrics.update(0.04, [0.85], 1, [1], 2)
        metrics.stop()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            report = metrics.generate_report(
                video_name='test.mp4',
                resolution='640x480',
                total_frames=2,
                model_name='yolo11n.pt',
                output_dir=tmpdir
            )
            
            assert report['video'] == 'test.mp4'
            assert report['processed_frames'] == 2
            assert report['unique_track_ids'] == 2
            
            # Check JSON file was created
            json_files = list(Path(tmpdir).glob('*.json'))
            assert len(json_files) == 1
            
            with open(json_files[0]) as f:
                saved_report = json.load(f)
            assert saved_report['video'] == 'test.mp4'


class TestStudentTracker:
    """Tests for StudentTracker class."""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        self.tracker = StudentTracker(model_name='yolo11n.pt')
    
    def test_initialization(self):
        assert self.tracker.model is not None
        assert len(self.tracker.unique_track_ids) == 0
    
    def test_track_returns_correct_keys(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = self.tracker.track(frame)
        
        assert 'boxes' in result
        assert 'confidences' in result
        assert 'track_ids' in result
        assert 'count' in result
        assert 'unique_ids_total' in result
    
    def test_track_blank_frame(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = self.tracker.track(frame)
        
        assert result['count'] == 0


class TestVideoProcessor:
    """Tests for VideoProcessor class."""
    
    def test_initialization(self):
        processor = VideoProcessor(frame_skip=2)
        assert processor.frame_skip == 2
        assert processor.confidence == 0.4
    
    def test_process_synthetic_video(self):
        """Test processing a synthetic video."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create synthetic video
            video_path = os.path.join(tmpdir, 'test_input.mp4')
            create_synthetic_video(video_path, num_frames=10)
            
            output_path = os.path.join(tmpdir, 'test_output.mp4')
            
            processor = VideoProcessor(
                confidence=0.3,
                frame_skip=1,
                show=False
            )
            
            report = processor.process(
                source=video_path,
                output_path=output_path
            )
            
            # Verify output video exists
            assert os.path.exists(output_path)
            assert os.path.getsize(output_path) > 0
            
            # Verify report
            assert report['video'] == 'test_input.mp4'
            assert report['processed_frames'] > 0
    
    def test_invalid_source(self):
        """Test that invalid source raises FileNotFoundError."""
        processor = VideoProcessor()
        with pytest.raises(FileNotFoundError):
            processor.process(source='nonexistent_video.mp4')


class TestHelpers:
    """Tests for utility functions."""
    
    def test_format_time_seconds(self):
        assert format_time(30.5) == '30.5s'
    
    def test_format_time_minutes(self):
        result = format_time(125.0)
        assert '2m' in result
    
    def test_format_time_hours(self):
        result = format_time(3725.0)
        assert '1h' in result
    
    def test_ensure_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            new_dir = Path(tmpdir) / 'new' / 'nested' / 'dir'
            result = ensure_dir(new_dir)
            assert result.exists()
    
    def test_validate_video_nonexistent(self):
        with pytest.raises(FileNotFoundError):
            validate_video_path('nonexistent.mp4')
    
    def test_validate_image_nonexistent(self):
        from src.utils.helpers import validate_image_path
        with pytest.raises(FileNotFoundError):
            validate_image_path('nonexistent.jpg')
