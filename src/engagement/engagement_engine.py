"""Top-level engagement analysis engine.

Orchestrates feature extraction, engagement scoring, data quality
classification, and session analysis from Phase 3 behavior events.

This is the primary entry point for Phase 4 analytics.
"""

from typing import Optional

from .feature_extractor import FeatureExtractor
from .engagement_model import HeuristicEngagementModel, BaseEngagementModel
from .session_analyzer import SessionAnalyzer


def classify_data_quality(visible_duration: float,
                           session_duration: float,
                           min_observation_seconds: float = 30.0,
                           min_coverage_ratio: float = 0.30) -> str:
    """Classify data quality based on observation coverage.
    
    Args:
        visible_duration: How long the student was observed (seconds).
        session_duration: Total session duration (seconds).
        min_observation_seconds: Minimum observation time for any score.
        min_coverage_ratio: Minimum coverage for MEDIUM quality.
    
    Returns:
        One of: 'HIGH', 'MEDIUM', 'LOW', 'INSUFFICIENT_DATA'
    """
    if visible_duration < min_observation_seconds:
        return 'INSUFFICIENT_DATA'
    
    if session_duration <= 0:
        return 'INSUFFICIENT_DATA'
    
    coverage = visible_duration / session_duration
    
    if coverage >= 0.7:
        return 'HIGH'
    elif coverage >= min_coverage_ratio:
        return 'MEDIUM'
    else:
        return 'LOW'


class EngagementEngine:
    """Top-level engagement analysis orchestrator.
    
    Composes FeatureExtractor, EngagementModel, and SessionAnalyzer
    to produce complete engagement analytics from Phase 3 events.
    """
    
    def __init__(self,
                 weights: Optional[dict] = None,
                 base_score: float = 50.0,
                 level_thresholds: Optional[dict] = None,
                 raised_hand_rate_cap: float = 1.0,
                 time_window: float = 60.0,
                 min_observation_seconds: float = 30.0,
                 min_data_coverage: float = 0.30,
                 model: Optional[BaseEngagementModel] = None):
        """Initialize the engagement engine.
        
        Args:
            weights: Engagement scoring weights.
            base_score: Base engagement score.
            level_thresholds: Score level classification thresholds.
            raised_hand_rate_cap: Max effective raised hand rate.
            time_window: Time window for timeline analysis (seconds).
            min_observation_seconds: Minimum seconds for data sufficiency.
            min_data_coverage: Minimum coverage ratio for MEDIUM quality.
            model: Optional custom engagement model. If None, uses
                HeuristicEngagementModel with the provided weights.
        """
        self.model = model or HeuristicEngagementModel(
            weights=weights,
            base_score=base_score,
            level_thresholds=level_thresholds,
            raised_hand_rate_cap=raised_hand_rate_cap,
        )
        self.time_window = time_window
        self.min_observation_seconds = min_observation_seconds
        self.min_data_coverage = min_data_coverage
        self.analyzer = SessionAnalyzer(time_window=time_window)
    
    def analyze(self, events: list,
                session_duration: float,
                identity_mappings: Optional[dict] = None,
                track_visibility: Optional[dict[int, float]] = None,
                identity_summary: Optional[dict] = None) -> dict:
        """Run complete engagement analysis.
        
        Args:
            events: List of BehaviorEvent objects (or dicts) from Phase 3.
            session_duration: Total video/session duration in seconds.
            identity_mappings: Phase 2 identity mappings
                {track_id: {'student_id': str, 'name': str, ...}}
            track_visibility: Optional per-track visible seconds.
            identity_summary: Phase 2 identity summary dict.
        
        Returns:
            Complete engagement analysis dict with:
                'students': per-student analytics
                'classroom': aggregate classroom metrics
                'timeline': time-window engagement trend
                'model': model configuration and disclaimer
        """
        # Feature extraction
        extractor = FeatureExtractor(
            session_duration=session_duration,
            track_visibility=track_visibility,
        )
        all_features = extractor.extract(events)
        
        # Per-student scoring
        student_results = {}
        for key, features in all_features.items():
            visible = features.get('visible_duration', 0.0)
            
            # Data quality classification
            data_quality = classify_data_quality(
                visible_duration=visible,
                session_duration=session_duration,
                min_observation_seconds=self.min_observation_seconds,
                min_coverage_ratio=self.min_data_coverage,
            )
            
            # Engagement scoring
            if data_quality == 'INSUFFICIENT_DATA':
                score_result = {
                    'score': None,
                    'level': 'INSUFFICIENT_DATA',
                    'factors': [],
                }
            else:
                score_result = self.model.calculate(features)
            
            student_results[key] = {
                'track_id': features.get('track_id'),
                'student_id': features.get('student_id'),
                'features': features,
                'engagement_indicator': score_result['score'],
                'engagement_level': score_result['level'],
                'data_quality': data_quality,
                'contributing_factors': score_result['factors'],
            }
        
        # Classroom-level analytics
        classroom = self.analyzer.analyze_classroom(
            student_results, identity_summary
        )
        
        # Timeline analysis
        timeline = self.analyzer.analyze_timeline(
            events, session_duration, extractor, self.model
        )
        
        return {
            'students': student_results,
            'classroom': classroom,
            'timeline': timeline,
            'model': self.model.get_model_info(),
        }
