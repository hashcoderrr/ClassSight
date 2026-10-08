"""Tests for Phase 3 behavior analysis components."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import numpy as np


# ============================================================
# Tests for EventManager
# ============================================================

class TestEventManager:
    """Tests for temporal event confirmation."""
    
    def setup_method(self):
        from src.behavior.event_manager import EventManager
        self.manager = EventManager(
            confirmation_frames=3,
            min_duration=0.1,
            fps=10.0,
            gap_tolerance=2
        )
    
    def test_initialization(self):
        assert self.manager.confirmation_frames == 3
        assert self.manager.min_duration == 0.1
        assert self.manager.fps == 10.0
    
    def test_single_frame_not_confirmed(self):
        """Single detection should NOT generate an event."""
        self.manager.update(1, 'phone_usage', True, 0.9, 1)
        self.manager.update(1, 'phone_usage', False, 0.0, 2)
        self.manager.update(1, 'phone_usage', False, 0.0, 3)
        self.manager.update(1, 'phone_usage', False, 0.0, 4)
        self.manager.update(1, 'phone_usage', False, 0.0, 5)
        events = self.manager.get_confirmed_events()
        assert len(events) == 0
    
    def test_confirmation_after_n_frames(self):
        """Event should be confirmed after N consecutive detections."""
        # 5 consecutive detections (> confirmation_frames=3)
        for i in range(1, 6):
            self.manager.update(1, 'phone_usage', True, 0.8, i)
        # Stop detecting to end the event
        for i in range(6, 12):
            self.manager.update(1, 'phone_usage', False, 0.0, i)
        
        events = self.manager.get_confirmed_events()
        assert len(events) == 1
        assert events[0].event_type == 'phone_usage'
        assert events[0].track_id == 1
    
    def test_min_duration_filter(self):
        """Events shorter than min_duration should be discarded."""
        from src.behavior.event_manager import EventManager
        manager = EventManager(
            confirmation_frames=1,
            min_duration=5.0,  # Very long minimum
            fps=10.0
        )
        # Only 3 frames at 10fps = 0.2 seconds (< 5.0)
        for i in range(1, 4):
            manager.update(1, 'phone_usage', True, 0.8, i)
        for i in range(4, 10):
            manager.update(1, 'phone_usage', False, 0.0, i)
        
        events = manager.get_confirmed_events()
        assert len(events) == 0  # Too short
    
    def test_active_events(self):
        """Active events should be trackable during detection."""
        for i in range(1, 5):
            self.manager.update(1, 'raised_hand', True, 0.9, i)
        
        active = self.manager.get_active_events(1)
        assert 'raised_hand' in active
    
    def test_multiple_tracks(self):
        """Different tracks should have independent events."""
        for i in range(1, 6):
            self.manager.update(1, 'phone_usage', True, 0.8, i)
            self.manager.update(2, 'raised_hand', True, 0.9, i)
        
        active_1 = self.manager.get_active_events(1)
        active_2 = self.manager.get_active_events(2)
        assert 'phone_usage' in active_1
        assert 'raised_hand' in active_2
    
    def test_finalize(self):
        """Finalize should close all pending active events."""
        for i in range(1, 10):
            self.manager.update(1, 'phone_usage', True, 0.85, i)
        
        events = self.manager.finalize()
        assert len(events) >= 1
        assert events[0].event_type == 'phone_usage'
    
    def test_finalize_with_identity_mappings(self):
        """Finalize should attach student IDs from mappings."""
        for i in range(1, 10):
            self.manager.update(1, 'phone_usage', True, 0.85, i)
        
        mappings = {1: {'student_id': 'STU001', 'name': 'Alice'}}
        events = self.manager.finalize(mappings)
        assert len(events) >= 1
        assert events[0].student_id == 'STU001'
    
    def test_event_to_dict(self):
        """BehaviorEvent.to_dict() should return serializable dict."""
        from src.behavior.event_manager import BehaviorEvent
        event = BehaviorEvent(
            track_id=1, student_id='STU001', event_type='raised_hand',
            confidence=0.9, start_time=1.0, end_time=3.5,
            duration=2.5, start_frame=10, end_frame=35
        )
        d = event.to_dict()
        assert d['track_id'] == 1
        assert d['student_id'] == 'STU001'
        assert d['event_type'] == 'raised_hand'
        assert d['duration'] == 2.5
    
    def test_event_counts(self):
        """Event counts should be accurate."""
        for i in range(1, 10):
            self.manager.update(1, 'phone_usage', True, 0.8, i)
        self.manager.finalize()
        counts = self.manager.get_event_counts()
        assert 'phone_usage' in counts
    
    def test_reset(self):
        self.manager.update(1, 'phone_usage', True, 0.8, 1)
        self.manager.reset()
        assert len(self.manager.get_confirmed_events()) == 0


# ============================================================
# Tests for HandDetector
# ============================================================

class TestHandDetector:
    """Tests for raised hand geometry detection."""
    
    def setup_method(self):
        from src.behavior.hand_detector import HandDetector
        self.detector = HandDetector(min_keypoint_confidence=0.3)
    
    def test_initialization(self):
        assert self.detector.min_kp_conf == 0.3
    
    def test_hand_raised_above_shoulder(self):
        """Wrist above shoulder should detect raised hand."""
        # Image coords: y increases downward. Wrist at y=50, shoulder at y=200
        pose_results = {
            1: {
                'keypoints': np.zeros((17, 3)),
                'valid': True,
                'named_keypoints': {
                    'right_wrist': (300, 50, 0.9),
                    'right_elbow': (290, 120, 0.8),
                    'right_shoulder': (280, 200, 0.9),
                    'right_hip': (270, 350, 0.8),
                }
            }
        }
        
        results = self.detector.detect(pose_results)
        assert results[1]['detected']
        assert results[1]['hand'] == 'right'
        assert results[1]['confidence'] > 0
    
    def test_hand_below_shoulder_not_raised(self):
        """Wrist below shoulder should NOT detect raised hand."""
        pose_results = {
            1: {
                'keypoints': np.zeros((17, 3)),
                'valid': True,
                'named_keypoints': {
                    'right_wrist': (300, 300, 0.9),  # Below shoulder
                    'right_elbow': (290, 250, 0.8),
                    'right_shoulder': (280, 200, 0.9),
                    'right_hip': (270, 350, 0.8),
                }
            }
        }
        
        results = self.detector.detect(pose_results)
        assert not results[1]['detected']
    
    def test_both_hands_raised(self):
        pose_results = {
            1: {
                'keypoints': np.zeros((17, 3)),
                'valid': True,
                'named_keypoints': {
                    'left_wrist': (200, 50, 0.9),
                    'left_elbow': (210, 120, 0.8),
                    'left_shoulder': (220, 200, 0.9),
                    'left_hip': (230, 350, 0.8),
                    'right_wrist': (300, 50, 0.9),
                    'right_elbow': (290, 120, 0.8),
                    'right_shoulder': (280, 200, 0.9),
                    'right_hip': (270, 350, 0.8),
                }
            }
        }
        results = self.detector.detect(pose_results)
        assert results[1]['detected']
        assert results[1]['hand'] == 'both'
    
    def test_invalid_pose(self):
        pose_results = {
            1: {'keypoints': np.zeros((17, 3)), 'valid': False, 'named_keypoints': {}}
        }
        results = self.detector.detect(pose_results)
        assert not results[1]['detected']
    
    def test_low_confidence_rejected(self):
        pose_results = {
            1: {
                'keypoints': np.zeros((17, 3)),
                'valid': True,
                'named_keypoints': {
                    'right_wrist': (300, 50, 0.1),  # Low confidence
                    'right_shoulder': (280, 200, 0.1),
                }
            }
        }
        results = self.detector.detect(pose_results)
        assert not results[1]['detected']


# ============================================================
# Tests for HeadPoseAnalyzer
# ============================================================

class TestHeadPoseAnalyzer:
    """Tests for head orientation classification."""
    
    def setup_method(self):
        from src.behavior.head_pose import HeadPoseAnalyzer
        self.analyzer = HeadPoseAnalyzer(min_keypoint_confidence=0.3)
    
    def test_initialization(self):
        assert self.analyzer.min_kp_conf == 0.3
    
    def test_forward_orientation(self):
        """Nose centered between shoulders, above shoulder level."""
        pose_results = {
            1: {
                'keypoints': np.zeros((17, 3)),
                'valid': True,
                'named_keypoints': {
                    'nose': (250, 100, 0.9),
                    'left_shoulder': (200, 200, 0.9),
                    'right_shoulder': (300, 200, 0.9),
                    'left_eye': (240, 90, 0.9),
                    'right_eye': (260, 90, 0.9),
                }
            }
        }
        results = self.analyzer.analyze(pose_results)
        assert results[1]['orientation'] == 'FORWARD'
    
    def test_down_orientation(self):
        """Nose near shoulder level should classify as DOWN."""
        pose_results = {
            1: {
                'keypoints': np.zeros((17, 3)),
                'valid': True,
                'named_keypoints': {
                    'nose': (250, 195, 0.9),  # Near shoulder Y=200
                    'left_shoulder': (200, 200, 0.9),
                    'right_shoulder': (300, 200, 0.9),
                }
            }
        }
        results = self.analyzer.analyze(pose_results)
        assert results[1]['orientation'] == 'DOWN'
    
    def test_left_orientation(self):
        """Nose shifted far left should classify as LEFT."""
        pose_results = {
            1: {
                'keypoints': np.zeros((17, 3)),
                'valid': True,
                'named_keypoints': {
                    'nose': (170, 100, 0.9),  # Far left of center (250)
                    'left_shoulder': (200, 200, 0.9),
                    'right_shoulder': (300, 200, 0.9),
                    'right_ear': (310, 110, 0.9),  # Only right ear visible
                }
            }
        }
        results = self.analyzer.analyze(pose_results)
        assert results[1]['orientation'] == 'LEFT'
    
    def test_right_orientation(self):
        pose_results = {
            1: {
                'keypoints': np.zeros((17, 3)),
                'valid': True,
                'named_keypoints': {
                    'nose': (330, 100, 0.9),  # Far right of center (250)
                    'left_shoulder': (200, 200, 0.9),
                    'right_shoulder': (300, 200, 0.9),
                    'left_ear': (190, 110, 0.9),
                }
            }
        }
        results = self.analyzer.analyze(pose_results)
        assert results[1]['orientation'] == 'RIGHT'
    
    def test_unknown_no_nose(self):
        pose_results = {
            1: {
                'keypoints': np.zeros((17, 3)),
                'valid': True,
                'named_keypoints': {
                    'left_shoulder': (200, 200, 0.9),
                    'right_shoulder': (300, 200, 0.9),
                }
            }
        }
        results = self.analyzer.analyze(pose_results)
        assert results[1]['orientation'] == 'UNKNOWN'
    
    def test_unknown_invalid_pose(self):
        pose_results = {
            1: {'keypoints': np.zeros((17, 3)), 'valid': False, 'named_keypoints': {}}
        }
        results = self.analyzer.analyze(pose_results)
        assert results[1]['orientation'] == 'UNKNOWN'


# ============================================================
# Tests for SleepDetector
# ============================================================

class TestSleepDetector:
    """Tests for sleeping/inattentive posture detection."""
    
    def setup_method(self):
        from src.behavior.sleep_detector import SleepDetector
        self.detector = SleepDetector(
            confirmation_seconds=2.0,
            fps=10.0
        )
    
    def test_initialization(self):
        assert self.detector.confirmation_seconds == 2.0
        assert self.detector.confirmation_frames == 20
    
    def test_short_head_down_not_sleeping(self):
        """Brief head-down should NOT trigger sleeping indicator."""
        pose_results = {1: {'valid': True, 'named_keypoints': {'nose': (250, 195, 0.9), 'left_shoulder': (200, 200, 0.9)}}}
        head_states = {1: {'orientation': 'DOWN', 'confidence': 0.8}}
        
        # Only 5 frames at 10fps = 0.5s (< 2.0s)
        for _ in range(5):
            results = self.detector.detect(pose_results, head_states, [1])
        
        assert not results[1]['detected']
    
    def test_sustained_head_down_sleeping(self):
        """Sustained head-down for confirmation_seconds should trigger."""
        pose_results = {1: {'valid': True, 'named_keypoints': {'nose': (250, 195, 0.9), 'left_shoulder': (200, 200, 0.9)}}}
        head_states = {1: {'orientation': 'DOWN', 'confidence': 0.8}}
        
        # 25 frames at 10fps = 2.5s (> 2.0s)
        results = None
        for _ in range(25):
            results = self.detector.detect(pose_results, head_states, [1])
        
        assert results[1]['detected']
        assert results[1]['indicator'] == 'POSSIBLE_SLEEPING'
    
    def test_head_up_resets(self):
        """Looking up should reset the head-down counter."""
        pose_results = {1: {'valid': True, 'named_keypoints': {}}}
        head_down = {1: {'orientation': 'DOWN', 'confidence': 0.8}}
        head_up = {1: {'orientation': 'FORWARD', 'confidence': 0.9}}
        
        # 10 frames down, then up
        for _ in range(10):
            self.detector.detect(pose_results, head_down, [1])
        for _ in range(3):
            self.detector.detect(pose_results, head_up, [1])
        
        # Counter should have reset
        assert self.detector._head_down_counts.get(1, 0) == 0
    
    def test_reset(self):
        self.detector._head_down_counts[1] = 100
        self.detector.reset()
        assert len(self.detector._head_down_counts) == 0


# ============================================================
# Tests for PhoneDetector (logic only, no model)
# ============================================================

class TestPhoneDetectorLogic:
    """Tests for phone detection association logic (without YOLO model)."""
    
    def test_box_center(self):
        from src.behavior.phone_detector import _box_center
        cx, cy = _box_center([100, 200, 300, 400])
        assert cx == 200.0
        assert cy == 300.0
    
    def test_distance(self):
        from src.behavior.phone_detector import _distance
        d = _distance((0, 0), (3, 4))
        assert abs(d - 5.0) < 1e-6
    
    def test_boxes_overlap(self):
        from src.behavior.phone_detector import _boxes_overlap
        assert _boxes_overlap([100, 100, 200, 200], [150, 150, 250, 250])
        assert not _boxes_overlap([100, 100, 200, 200], [300, 300, 400, 400])
    
    def test_boxes_overlap_with_margin(self):
        from src.behavior.phone_detector import _boxes_overlap
        # These don't overlap normally
        assert not _boxes_overlap([100, 100, 200, 200], [220, 220, 300, 300])
        # But with margin=30 they should
        assert _boxes_overlap([100, 100, 200, 200], [220, 220, 300, 300], margin=30)


# ============================================================
# Tests for BehaviorMetrics
# ============================================================

class TestBehaviorMetrics:
    """Tests for behavior metrics and reporting."""
    
    def test_initialization(self):
        from src.evaluation.behavior_metrics import BehaviorMetrics
        m = BehaviorMetrics()
        assert m.total_frames_processed == 0
    
    def test_update_frame(self):
        from src.evaluation.behavior_metrics import BehaviorMetrics
        m = BehaviorMetrics()
        m.update_frame(0.05, 3, [1, 2, 3])
        assert m.total_frames_processed == 1
        assert len(m.all_track_ids) == 3
    
    def test_generate_report(self):
        from src.evaluation.behavior_metrics import BehaviorMetrics
        from src.behavior.event_manager import BehaviorEvent
        import tempfile
        
        m = BehaviorMetrics()
        m.start()
        m.update_frame(0.05, 2, [1, 2])
        m.stop()
        
        event = BehaviorEvent(
            track_id=1, student_id='STU001', event_type='raised_hand',
            confidence=0.9, start_time=1.0, end_time=3.0,
            duration=2.0, start_frame=10, end_frame=30
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            report = m.generate_behavior_report(
                video_name='test.mp4',
                resolution='640x480',
                total_frames=30,
                detection_model='yolo11n.pt',
                pose_model='yolo11n-pose.pt',
                identity_summary={'confirmed': 1, 'unknown': 0},
                behavior_events=[event],
                event_counts={'raised_hand': 1},
                identity_mappings={},
                output_dir=tmpdir
            )
            
            assert report['video'] == 'test.mp4'
            assert report['total_behavior_events'] == 1
            assert 'raised_hand' in report['event_counts']
            
            from pathlib import Path
            json_files = list(Path(tmpdir).glob('*.json'))
            assert len(json_files) == 1


# ============================================================
# Tests for PoseAnalyzer (integration, needs YOLO pose model)
# ============================================================

class TestPoseAnalyzerIntegration:
    """Integration tests requiring YOLO pose model."""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        try:
            from src.behavior.pose_analyzer import PoseAnalyzer
            self.analyzer = PoseAnalyzer(model_name='yolo11n-pose.pt')
            self._available = True
        except Exception:
            self._available = False
            pytest.skip("YOLO pose model not available")
    
    def test_analyze_blank_frame(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        results = self.analyzer.analyze(frame, [], [])
        assert isinstance(results, dict)
        assert len(results) == 0
    
    def test_analyze_returns_correct_structure(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        results = self.analyzer.analyze(frame, [[100, 100, 200, 400]], [1])
        assert isinstance(results, dict)
        if 1 in results:
            assert 'keypoints' in results[1]
            assert 'valid' in results[1]
            assert 'named_keypoints' in results[1]
    
    def test_iou_computation(self):
        from src.behavior.pose_analyzer import compute_iou
        # Perfect overlap
        assert compute_iou([0, 0, 100, 100], [0, 0, 100, 100]) == 1.0
        # No overlap
        assert compute_iou([0, 0, 50, 50], [100, 100, 200, 200]) == 0.0
        # Partial overlap
        iou = compute_iou([0, 0, 100, 100], [50, 50, 150, 150])
        assert 0 < iou < 1
