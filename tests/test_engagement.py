"""Phase 4 unit tests for engagement analytics.

Tests feature extraction, engagement scoring, data quality,
classroom aggregation, timeline analysis, participation metrics,
sanity scenarios, and report generation.
"""

import json
import tempfile
import unittest
from pathlib import Path

import sys
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.behavior.event_manager import BehaviorEvent
from src.engagement.feature_extractor import FeatureExtractor
from src.engagement.engagement_model import (
    HeuristicEngagementModel, BaseEngagementModel
)
from src.engagement.engagement_engine import (
    EngagementEngine, classify_data_quality
)
from src.engagement.session_analyzer import SessionAnalyzer
from src.engagement.engagement_metrics import EngagementReporter


def _make_event(track_id=1, student_id=None, event_type='head_down',
                confidence=0.9, start_time=0.0, end_time=5.0,
                duration=None, start_frame=1, end_frame=150):
    """Helper to create BehaviorEvent objects for tests."""
    if duration is None:
        duration = end_time - start_time
    return BehaviorEvent(
        track_id=track_id,
        student_id=student_id,
        event_type=event_type,
        confidence=confidence,
        start_time=start_time,
        end_time=end_time,
        duration=duration,
        start_frame=start_frame,
        end_frame=end_frame,
    )


class TestFeatureExtractor(unittest.TestCase):
    """Test behavioral feature extraction."""
    
    def test_basic_extraction(self):
        """Test basic feature extraction from events."""
        events = [
            _make_event(track_id=1, student_id='STU001',
                       event_type='head_down', duration=10.0,
                       start_time=0, end_time=10),
            _make_event(track_id=1, student_id='STU001',
                       event_type='raised_hand', duration=3.0,
                       start_time=15, end_time=18),
        ]
        extractor = FeatureExtractor(session_duration=60.0)
        features = extractor.extract(events)
        
        self.assertIn('STU001', features)
        f = features['STU001']
        self.assertEqual(f['student_id'], 'STU001')
        self.assertAlmostEqual(f['head_down_duration'], 10.0, places=1)
        self.assertAlmostEqual(f['raised_hand_duration'], 3.0, places=1)
        self.assertEqual(f['raised_hand_count'], 1)
        self.assertAlmostEqual(f['visible_duration'], 60.0)
    
    def test_forward_duration_calculation(self):
        """Forward = visible - (head_down + head_left + head_right)."""
        events = [
            _make_event(event_type='head_down', duration=10.0,
                       start_time=0, end_time=10),
            _make_event(event_type='head_left', duration=5.0,
                       start_time=10, end_time=15),
            _make_event(event_type='head_right', duration=5.0,
                       start_time=15, end_time=20),
        ]
        extractor = FeatureExtractor(session_duration=100.0)
        features = extractor.extract(events)
        f = list(features.values())[0]
        # forward = 100 - 10 - 5 - 5 = 80
        self.assertAlmostEqual(f['forward_duration'], 80.0, places=1)
        self.assertAlmostEqual(f['forward_ratio'], 0.80, places=2)
    
    def test_ratio_normalization(self):
        """Ratios should be between 0 and 1."""
        events = [
            _make_event(event_type='phone_usage', duration=30.0,
                       start_time=0, end_time=30),
        ]
        extractor = FeatureExtractor(session_duration=60.0)
        features = extractor.extract(events)
        f = list(features.values())[0]
        self.assertAlmostEqual(f['phone_ratio'], 0.5, places=2)
        self.assertGreaterEqual(f['forward_ratio'], 0.0)
        self.assertLessEqual(f['forward_ratio'], 1.0)
    
    def test_zero_duration_handling(self):
        """No events + zero session → all ratios 0."""
        extractor = FeatureExtractor(session_duration=0.0)
        features = extractor.extract([])
        self.assertEqual(len(features), 0)
    
    def test_rate_calculation(self):
        """Rates should be per minute."""
        events = [
            _make_event(event_type='raised_hand', duration=2.0,
                       start_time=0, end_time=2),
            _make_event(event_type='raised_hand', duration=2.0,
                       start_time=30, end_time=32),
            _make_event(event_type='raised_hand', duration=2.0,
                       start_time=58, end_time=60),
        ]
        extractor = FeatureExtractor(session_duration=120.0)  # 2 minutes
        features = extractor.extract(events)
        f = list(features.values())[0]
        self.assertEqual(f['raised_hand_count'], 3)
        # 3 events / 2 minutes = 1.5 per minute
        self.assertAlmostEqual(f['raised_hand_rate'], 1.5, places=1)
    
    def test_multiple_students(self):
        """Multiple students produce separate feature sets."""
        events = [
            _make_event(track_id=1, student_id='STU001',
                       event_type='head_down', duration=10.0,
                       start_time=0, end_time=10),
            _make_event(track_id=2, student_id='STU002',
                       event_type='raised_hand', duration=5.0,
                       start_time=0, end_time=5),
        ]
        extractor = FeatureExtractor(session_duration=60.0)
        features = extractor.extract(events)
        self.assertIn('STU001', features)
        self.assertIn('STU002', features)
        self.assertEqual(features['STU001']['head_down_duration'], 10.0)
        self.assertEqual(features['STU002']['raised_hand_count'], 1)
    
    def test_unknown_student_keying(self):
        """Unknown students use track_N as key."""
        events = [
            _make_event(track_id=5, student_id=None,
                       event_type='phone_usage', duration=8.0,
                       start_time=0, end_time=8),
        ]
        extractor = FeatureExtractor(session_duration=60.0)
        features = extractor.extract(events)
        self.assertIn('track_5', features)
        self.assertIsNone(features['track_5']['student_id'])
    
    def test_track_visibility_override(self):
        """Track-specific visibility overrides session duration."""
        events = [
            _make_event(track_id=1, student_id='STU001',
                       event_type='head_down', duration=5.0,
                       start_time=0, end_time=5),
        ]
        extractor = FeatureExtractor(
            session_duration=300.0,
            track_visibility={1: 50.0}
        )
        features = extractor.extract(events)
        f = features['STU001']
        self.assertAlmostEqual(f['visible_duration'], 50.0)
        # head_down_ratio = 5 / 50 = 0.10
        self.assertAlmostEqual(f['head_down_ratio'], 0.10, places=2)
    
    def test_window_extraction(self):
        """extract_for_window clips events to window boundaries."""
        events = [
            _make_event(event_type='head_down', duration=20.0,
                       start_time=10.0, end_time=30.0),
        ]
        extractor = FeatureExtractor(session_duration=60.0)
        window = extractor.extract_for_window(events, 20.0, 40.0)
        f = list(window.values())[0]
        # Clipped to [20, 30] → 10 seconds
        self.assertAlmostEqual(f['head_down_duration'], 10.0, places=1)
    
    def test_average_confidence(self):
        """Average confidence should be calculated correctly."""
        events = [
            _make_event(event_type='head_down', confidence=0.8,
                       duration=5.0, start_time=0, end_time=5),
            _make_event(event_type='phone_usage', confidence=0.6,
                       duration=3.0, start_time=5, end_time=8),
        ]
        extractor = FeatureExtractor(session_duration=60.0)
        features = extractor.extract(events)
        f = list(features.values())[0]
        self.assertAlmostEqual(f['average_confidence'], 0.7, places=1)


class TestEngagementModel(unittest.TestCase):
    """Test heuristic engagement scoring model."""
    
    def test_default_weights(self):
        """Model with default weights produces sensible score."""
        model = HeuristicEngagementModel()
        features = {
            'forward_ratio': 0.80,
            'raised_hand_rate': 0.5,
            'phone_ratio': 0.0,
            'head_down_ratio': 0.10,
            'sleep_ratio': 0.0,
        }
        result = model.calculate(features)
        self.assertIn('score', result)
        self.assertIn('level', result)
        self.assertIn('factors', result)
        # High forward + some participation → should be HIGH
        self.assertGreaterEqual(result['score'], 70)
    
    def test_high_engagement(self):
        """High forward + participation → HIGH level."""
        model = HeuristicEngagementModel()
        features = {
            'forward_ratio': 0.90,
            'raised_hand_rate': 1.0,
            'phone_ratio': 0.0,
            'head_down_ratio': 0.05,
            'sleep_ratio': 0.0,
        }
        result = model.calculate(features)
        self.assertEqual(result['level'], 'HIGH')
    
    def test_low_engagement(self):
        """High phone + head-down → LOW level."""
        model = HeuristicEngagementModel()
        features = {
            'forward_ratio': 0.10,
            'raised_hand_rate': 0.0,
            'phone_ratio': 0.60,
            'head_down_ratio': 0.50,
            'sleep_ratio': 0.20,
        }
        result = model.calculate(features)
        self.assertEqual(result['level'], 'LOW')
    
    def test_all_zeros_base_score(self):
        """All zeros → base score (50)."""
        model = HeuristicEngagementModel(base_score=50.0)
        features = {
            'forward_ratio': 0.0,
            'raised_hand_rate': 0.0,
            'phone_ratio': 0.0,
            'head_down_ratio': 0.0,
            'sleep_ratio': 0.0,
        }
        result = model.calculate(features)
        self.assertAlmostEqual(result['score'], 50.0, places=1)
    
    def test_score_clamped_0_100(self):
        """Score should never exceed [0, 100]."""
        model = HeuristicEngagementModel(base_score=50.0)
        # Extreme positive
        features = {
            'forward_ratio': 5.0,  # Unrealistic but tests clamping
            'raised_hand_rate': 10.0,
            'phone_ratio': 0.0,
            'head_down_ratio': 0.0,
            'sleep_ratio': 0.0,
        }
        result = model.calculate(features)
        self.assertLessEqual(result['score'], 100.0)
        
        # Extreme negative
        features2 = {
            'forward_ratio': 0.0,
            'raised_hand_rate': 0.0,
            'phone_ratio': 5.0,
            'head_down_ratio': 5.0,
            'sleep_ratio': 5.0,
        }
        result2 = model.calculate(features2)
        self.assertGreaterEqual(result2['score'], 0.0)
    
    def test_weight_configuration(self):
        """Custom weights should change score."""
        custom = {'forward_ratio': 0.80}  # Double forward weight
        model_custom = HeuristicEngagementModel(weights=custom)
        model_default = HeuristicEngagementModel()
        
        features = {
            'forward_ratio': 0.90,
            'raised_hand_rate': 0.0,
            'phone_ratio': 0.0,
            'head_down_ratio': 0.0,
            'sleep_ratio': 0.0,
        }
        
        score_custom = model_custom.calculate(features)['score']
        score_default = model_default.calculate(features)['score']
        self.assertGreater(score_custom, score_default)
    
    def test_explainable_factors(self):
        """Factors should be generated for meaningful contributions."""
        model = HeuristicEngagementModel()
        features = {
            'forward_ratio': 0.80,
            'raised_hand_rate': 0.5,
            'phone_ratio': 0.10,
            'head_down_ratio': 0.05,
            'sleep_ratio': 0.0,
        }
        result = model.calculate(features)
        self.assertGreater(len(result['factors']), 0)
        for factor in result['factors']:
            self.assertIn('factor', factor)
            self.assertIn('direction', factor)
            self.assertIn('impact', factor)
    
    def test_level_boundaries(self):
        """Test exact boundary classification."""
        model = HeuristicEngagementModel()
        # Exactly at MODERATE boundary
        features_40 = {
            'forward_ratio': 0.0,
            'raised_hand_rate': 0.0,
            'phone_ratio': 0.50,
            'head_down_ratio': 0.0,
            'sleep_ratio': 0.0,
        }
        result_40 = model.calculate(features_40)
        # Score = 50 + 0 + 0 + (-0.20*0.50*100) + 0 + 0 = 50 - 10 = 40
        self.assertEqual(result_40['level'], 'MODERATE')
    
    def test_raised_hand_rate_cap(self):
        """Raised hand rate should be capped."""
        model = HeuristicEngagementModel(raised_hand_rate_cap=1.0)
        features = {
            'forward_ratio': 0.0,
            'raised_hand_rate': 5.0,  # Way above cap
            'phone_ratio': 0.0,
            'head_down_ratio': 0.0,
            'sleep_ratio': 0.0,
        }
        result = model.calculate(features)
        # With cap=1.0, effective contribution = 1.0 * 0.20 * 100 = 20
        # Score = 50 + 20 = 70
        self.assertAlmostEqual(result['score'], 70.0, places=1)
    
    def test_model_info(self):
        """get_model_info should return transparency metadata."""
        model = HeuristicEngagementModel()
        info = model.get_model_info()
        self.assertIn('type', info)
        self.assertIn('weights', info)
        self.assertIn('disclaimer', info)
        self.assertEqual(info['type'], 'HeuristicEngagementModel')
    
    def test_is_abstract_base(self):
        """BaseEngagementModel should be abstract."""
        with self.assertRaises(TypeError):
            BaseEngagementModel()


class TestDataQuality(unittest.TestCase):
    """Test data quality classification."""
    
    def test_insufficient_data(self):
        """Below minimum observation → INSUFFICIENT_DATA."""
        result = classify_data_quality(
            visible_duration=10.0,
            session_duration=300.0,
            min_observation_seconds=30.0,
        )
        self.assertEqual(result, 'INSUFFICIENT_DATA')
    
    def test_high_quality(self):
        """Above 70% coverage → HIGH."""
        result = classify_data_quality(
            visible_duration=250.0,
            session_duration=300.0,
            min_observation_seconds=30.0,
        )
        self.assertEqual(result, 'HIGH')
    
    def test_medium_quality(self):
        """Between 30-70% coverage → MEDIUM."""
        result = classify_data_quality(
            visible_duration=150.0,
            session_duration=300.0,
            min_observation_seconds=30.0,
        )
        self.assertEqual(result, 'MEDIUM')
    
    def test_low_quality(self):
        """Below 30% but above min observation → LOW."""
        result = classify_data_quality(
            visible_duration=60.0,
            session_duration=300.0,
            min_observation_seconds=30.0,
            min_coverage_ratio=0.30,
        )
        self.assertEqual(result, 'LOW')
    
    def test_zero_session_duration(self):
        """Zero session → INSUFFICIENT_DATA."""
        result = classify_data_quality(
            visible_duration=0.0,
            session_duration=0.0,
            min_observation_seconds=30.0,
        )
        self.assertEqual(result, 'INSUFFICIENT_DATA')


class TestSessionAnalyzer(unittest.TestCase):
    """Test classroom-level and timeline analytics."""
    
    def test_classroom_aggregate(self):
        """Test basic classroom aggregation."""
        student_results = {
            'STU001': {
                'engagement_indicator': 80.0,
                'engagement_level': 'HIGH',
                'data_quality': 'HIGH',
                'features': {
                    'student_id': 'STU001',
                    'raised_hand_count': 3,
                    'raised_hand_duration': 9.0,
                    'phone_duration': 0.0,
                    'head_down_duration': 5.0,
                    'head_left_duration': 2.0,
                    'head_right_duration': 3.0,
                    'possible_sleep_duration': 0.0,
                    'forward_duration': 50.0,
                    'visible_duration': 60.0,
                    'phone_event_count': 0,
                },
            },
            'STU002': {
                'engagement_indicator': 55.0,
                'engagement_level': 'MODERATE',
                'data_quality': 'MEDIUM',
                'features': {
                    'student_id': 'STU002',
                    'raised_hand_count': 1,
                    'raised_hand_duration': 3.0,
                    'phone_duration': 10.0,
                    'head_down_duration': 15.0,
                    'head_left_duration': 0.0,
                    'head_right_duration': 0.0,
                    'possible_sleep_duration': 0.0,
                    'forward_duration': 45.0,
                    'visible_duration': 60.0,
                    'phone_event_count': 2,
                },
            },
        }
        
        analyzer = SessionAnalyzer(time_window=60.0)
        classroom = analyzer.analyze_classroom(student_results)
        
        self.assertEqual(classroom['scored_students'], 2)
        self.assertAlmostEqual(
            classroom['average_engagement'], 67.5, places=1
        )
        self.assertEqual(classroom['engagement_distribution']['HIGH'], 1)
        self.assertEqual(classroom['engagement_distribution']['MODERATE'], 1)
    
    def test_participation_rate(self):
        """Participation rate = hand_raisers / total."""
        student_results = {
            'A': {
                'engagement_indicator': 70,
                'data_quality': 'HIGH',
                'features': {'raised_hand_count': 2, 'student_id': 'A',
                             'raised_hand_duration': 0, 'phone_duration': 0,
                             'head_down_duration': 0, 'head_left_duration': 0,
                             'head_right_duration': 0, 'possible_sleep_duration': 0,
                             'forward_duration': 60, 'visible_duration': 60,
                             'phone_event_count': 0},
            },
            'B': {
                'engagement_indicator': 50,
                'data_quality': 'HIGH',
                'features': {'raised_hand_count': 0, 'student_id': 'B',
                             'raised_hand_duration': 0, 'phone_duration': 0,
                             'head_down_duration': 0, 'head_left_duration': 0,
                             'head_right_duration': 0, 'possible_sleep_duration': 0,
                             'forward_duration': 60, 'visible_duration': 60,
                             'phone_event_count': 0},
            },
        }
        analyzer = SessionAnalyzer()
        classroom = analyzer.analyze_classroom(student_results)
        participation = classroom['participation']
        self.assertEqual(participation['unique_participating_students'], 1)
        self.assertAlmostEqual(
            participation['participation_rate'], 0.5, places=2
        )
    
    def test_empty_results(self):
        """Empty student results → graceful defaults."""
        analyzer = SessionAnalyzer()
        classroom = analyzer.analyze_classroom({})
        self.assertEqual(classroom['scored_students'], 0)
        self.assertEqual(classroom['average_engagement'], 0.0)
    
    def test_timeline_analysis(self):
        """Timeline should produce windows covering session."""
        events = [
            _make_event(event_type='head_down', duration=5.0,
                       start_time=10.0, end_time=15.0),
        ]
        extractor = FeatureExtractor(session_duration=120.0)
        model = HeuristicEngagementModel()
        analyzer = SessionAnalyzer(time_window=60.0)
        
        timeline = analyzer.analyze_timeline(
            events, 120.0, extractor, model
        )
        
        self.assertEqual(len(timeline), 2)  # 120s / 60s = 2 windows
        self.assertEqual(timeline[0]['start'], 0.0)
        self.assertEqual(timeline[0]['end'], 60.0)
        self.assertEqual(timeline[1]['start'], 60.0)
        self.assertEqual(timeline[1]['end'], 120.0)
    
    def test_insufficient_data_excluded_from_average(self):
        """Students with INSUFFICIENT_DATA excluded from average."""
        student_results = {
            'STU001': {
                'engagement_indicator': 80.0,
                'engagement_level': 'HIGH',
                'data_quality': 'HIGH',
                'features': {'student_id': 'STU001', 'raised_hand_count': 0,
                             'raised_hand_duration': 0, 'phone_duration': 0,
                             'head_down_duration': 0, 'head_left_duration': 0,
                             'head_right_duration': 0, 'possible_sleep_duration': 0,
                             'forward_duration': 60, 'visible_duration': 60,
                             'phone_event_count': 0},
            },
            'track_5': {
                'engagement_indicator': None,
                'engagement_level': 'INSUFFICIENT_DATA',
                'data_quality': 'INSUFFICIENT_DATA',
                'features': {'student_id': None, 'raised_hand_count': 0,
                             'raised_hand_duration': 0, 'phone_duration': 0,
                             'head_down_duration': 0, 'head_left_duration': 0,
                             'head_right_duration': 0, 'possible_sleep_duration': 0,
                             'forward_duration': 0, 'visible_duration': 5,
                             'phone_event_count': 0},
            },
        }
        analyzer = SessionAnalyzer()
        classroom = analyzer.analyze_classroom(student_results)
        self.assertEqual(classroom['scored_students'], 1)
        self.assertEqual(classroom['insufficient_data_count'], 1)
        self.assertAlmostEqual(classroom['average_engagement'], 80.0)


class TestEngagementEngine(unittest.TestCase):
    """Test the top-level engagement engine."""
    
    def test_full_analysis(self):
        """End-to-end analysis with events."""
        events = [
            _make_event(track_id=1, student_id='STU001',
                       event_type='head_down', duration=10.0,
                       start_time=0, end_time=10),
            _make_event(track_id=1, student_id='STU001',
                       event_type='raised_hand', duration=3.0,
                       start_time=20, end_time=23),
            _make_event(track_id=2, student_id=None,
                       event_type='phone_usage', duration=15.0,
                       start_time=5, end_time=20),
        ]
        
        engine = EngagementEngine(
            min_observation_seconds=5.0,
            time_window=30.0,
        )
        result = engine.analyze(
            events=events,
            session_duration=60.0,
        )
        
        self.assertIn('students', result)
        self.assertIn('classroom', result)
        self.assertIn('timeline', result)
        self.assertIn('model', result)
        
        # STU001 should have a score
        self.assertIn('STU001', result['students'])
        stu = result['students']['STU001']
        self.assertIsNotNone(stu['engagement_indicator'])
        self.assertIn(stu['engagement_level'], ['HIGH', 'MODERATE', 'LOW'])
        
        # Unknown track should also be scored
        self.assertIn('track_2', result['students'])
        self.assertIsNone(
            result['students']['track_2']['student_id']
        )
    
    def test_insufficient_data_handling(self):
        """Very short observation → INSUFFICIENT_DATA."""
        events = [
            _make_event(track_id=1, student_id='STU001',
                       event_type='head_down', duration=2.0,
                       start_time=0, end_time=2),
        ]
        engine = EngagementEngine(
            min_observation_seconds=30.0,
        )
        result = engine.analyze(
            events=events,
            session_duration=5.0,  # Very short
        )
        stu = result['students']['STU001']
        self.assertEqual(stu['data_quality'], 'INSUFFICIENT_DATA')
        self.assertIsNone(stu['engagement_indicator'])
    
    def test_empty_events(self):
        """No events → empty students dict."""
        engine = EngagementEngine()
        result = engine.analyze(events=[], session_duration=60.0)
        self.assertEqual(len(result['students']), 0)
        self.assertEqual(
            result['classroom']['scored_students'], 0
        )


class TestSanityScenarios(unittest.TestCase):
    """Controlled scenarios to verify scoring behaves sensibly."""
    
    def test_scenario_a_engaged_student(self):
        """High forward, raised hands, no negatives → HIGH."""
        events = [
            _make_event(track_id=1, student_id='STU001',
                       event_type='raised_hand', duration=3.0,
                       start_time=10, end_time=13),
            _make_event(track_id=1, student_id='STU001',
                       event_type='raised_hand', duration=2.0,
                       start_time=30, end_time=32),
            _make_event(track_id=1, student_id='STU001',
                       event_type='raised_hand', duration=2.0,
                       start_time=55, end_time=57),
        ]
        engine = EngagementEngine(min_observation_seconds=5.0)
        result = engine.analyze(events=events, session_duration=60.0)
        stu = result['students']['STU001']
        self.assertEqual(stu['engagement_level'], 'HIGH')
    
    def test_scenario_b_distracted_student(self):
        """High phone + head-down, no participation → LOW."""
        events = [
            _make_event(track_id=1, student_id='STU001',
                       event_type='phone_usage', duration=50.0,
                       start_time=0, end_time=50),
            _make_event(track_id=1, student_id='STU001',
                       event_type='head_down', duration=45.0,
                       start_time=5, end_time=50),
            _make_event(track_id=1, student_id='STU001',
                       event_type='possible_sleeping', duration=30.0,
                       start_time=10, end_time=40),
        ]
        engine = EngagementEngine(min_observation_seconds=5.0)
        result = engine.analyze(events=events, session_duration=60.0)
        stu = result['students']['STU001']
        self.assertEqual(stu['engagement_level'], 'LOW')
    
    def test_scenario_c_insufficient_observation(self):
        """Very brief observation → INSUFFICIENT_DATA."""
        events = [
            _make_event(track_id=1, student_id='STU001',
                       event_type='head_down', duration=3.0,
                       start_time=0, end_time=3),
        ]
        engine = EngagementEngine(min_observation_seconds=30.0)
        result = engine.analyze(events=events, session_duration=10.0)
        stu = result['students']['STU001']
        self.assertEqual(stu['data_quality'], 'INSUFFICIENT_DATA')
    
    def test_scenario_d_unknown_student(self):
        """Unknown student → track-level result, no invented ID."""
        events = [
            _make_event(track_id=15, student_id=None,
                       event_type='head_left', duration=10.0,
                       start_time=0, end_time=10),
        ]
        engine = EngagementEngine(min_observation_seconds=5.0)
        result = engine.analyze(events=events, session_duration=60.0)
        self.assertIn('track_15', result['students'])
        track_result = result['students']['track_15']
        self.assertIsNone(track_result['student_id'])
        self.assertIsNotNone(track_result['engagement_indicator'])
    
    def test_scenario_e_writing_notes(self):
        """Head-down (notes) alone doesn't cause LOW."""
        events = [
            _make_event(track_id=1, student_id='STU001',
                       event_type='head_down', duration=20.0,
                       start_time=0, end_time=20),
        ]
        engine = EngagementEngine(min_observation_seconds=5.0)
        result = engine.analyze(events=events, session_duration=60.0)
        stu = result['students']['STU001']
        # head_down_ratio = 20/60 = 0.33
        # forward_ratio = 40/60 = 0.67
        # Score ≈ 50 + 0.67*40 + 0 - 0 - 0.33*10 - 0 = 50 + 26.8 - 3.3 = 73.5
        # Should be HIGH or at least MODERATE, NOT LOW
        self.assertNotEqual(stu['engagement_level'], 'LOW')


class TestEngagementReporter(unittest.TestCase):
    """Test report generation."""
    
    def test_json_report_structure(self):
        """Report should contain all required sections."""
        analysis = {
            'students': {
                'STU001': {
                    'track_id': 1,
                    'student_id': 'STU001',
                    'engagement_indicator': 75.0,
                    'engagement_level': 'HIGH',
                    'data_quality': 'HIGH',
                    'contributing_factors': [],
                    'features': {
                        'visible_duration': 60.0,
                        'forward_ratio': 0.8,
                        'head_down_ratio': 0.1,
                        'phone_ratio': 0.0,
                        'sleep_ratio': 0.0,
                        'raised_hand_count': 2,
                        'raised_hand_rate': 1.0,
                        'total_event_count': 3,
                    },
                },
            },
            'classroom': {
                'scored_students': 1,
                'average_engagement': 75.0,
            },
            'timeline': [
                {'start': 0, 'end': 60, 'engagement_indicator': 75.0,
                 'student_count': 1, 'events': 3},
            ],
            'model': {'type': 'HeuristicEngagementModel',
                     'disclaimer': 'test'},
        }
        
        with tempfile.TemporaryDirectory() as tmpdir:
            reporter = EngagementReporter()
            report = reporter.generate_report(
                analysis_result=analysis,
                video_name='test.mp4',
                session_duration=60.0,
                output_dir=tmpdir,
                save_plots=False,
            )
            
            self.assertIn('session', report)
            self.assertIn('model', report)
            self.assertIn('classroom', report)
            self.assertIn('students', report)
            self.assertIn('timeline', report)
            
            # Verify JSON file was created
            json_files = list(Path(tmpdir).glob('engagement_report_*.json'))
            self.assertEqual(len(json_files), 1)
            
            # Verify CSV file was created
            csv_files = list(Path(tmpdir).glob('engagement_report_*.csv'))
            self.assertEqual(len(csv_files), 1)
    
    def test_report_contains_model_disclaimer(self):
        """Report must contain model weights and disclaimer."""
        analysis = {
            'students': {},
            'classroom': {},
            'timeline': [],
            'model': {
                'type': 'HeuristicEngagementModel',
                'weights': {'forward_ratio': 0.40},
                'disclaimer': 'Engagement indicators are inferred...',
            },
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            reporter = EngagementReporter()
            report = reporter.generate_report(
                analysis_result=analysis,
                output_dir=tmpdir,
                save_plots=False,
            )
            self.assertIn('disclaimer', report['model'])
            self.assertIn('weights', report['model'])
    
    def test_csv_student_rows(self):
        """CSV should contain student engagement rows."""
        analysis = {
            'students': {
                'STU001': {
                    'track_id': 1,
                    'student_id': 'STU001',
                    'engagement_indicator': 75.0,
                    'engagement_level': 'HIGH',
                    'data_quality': 'HIGH',
                    'contributing_factors': [],
                    'features': {
                        'visible_duration': 60.0,
                        'forward_ratio': 0.8,
                        'head_down_ratio': 0.1,
                        'phone_ratio': 0.0,
                        'sleep_ratio': 0.0,
                        'raised_hand_count': 2,
                        'raised_hand_rate': 1.0,
                        'total_event_count': 3,
                    },
                },
            },
            'classroom': {},
            'timeline': [],
            'model': {},
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            reporter = EngagementReporter()
            reporter.generate_report(
                analysis_result=analysis,
                output_dir=tmpdir,
                save_plots=False,
            )
            csv_files = list(Path(tmpdir).glob('*.csv'))
            self.assertTrue(len(csv_files) > 0)
            content = csv_files[0].read_text()
            self.assertIn('STU001', content)
            self.assertIn('HIGH', content)


if __name__ == '__main__':
    unittest.main()
