"""Classroom-level and time-window engagement analytics.

Provides aggregate classroom statistics, time-window engagement
timeline, participation analytics, and behavior breakdown.

All metrics are aggregate observations — no individual student ranking.
"""

from typing import Optional
import statistics


class SessionAnalyzer:
    """Analyzes classroom-level engagement and temporal trends."""
    
    def __init__(self, time_window: float = 60.0):
        """Initialize session analyzer.
        
        Args:
            time_window: Time window size in seconds for timeline analysis.
        """
        self.time_window = max(1.0, time_window)
    
    def analyze_classroom(self, student_results: dict[str, dict],
                           identity_summary: Optional[dict] = None) -> dict:
        """Generate classroom-level aggregate analytics.
        
        Args:
            student_results: Dict mapping student/track key to result dict.
                Each result must have: 'engagement_indicator', 'data_quality',
                'features' (with event counts and durations).
            identity_summary: Optional Phase 2 identity summary dict.
        
        Returns:
            Classroom aggregate analytics dict.
        """
        if not student_results:
            return self._empty_classroom(identity_summary)
        
        # Separate scored vs insufficient-data students
        scored = {}
        insufficient = {}
        for key, result in student_results.items():
            if result.get('data_quality') == 'INSUFFICIENT_DATA':
                insufficient[key] = result
            else:
                scored[key] = result
        
        scores = [r['engagement_indicator'] for r in scored.values()]
        
        # Engagement distribution
        distribution = {'HIGH': 0, 'MODERATE': 0, 'LOW': 0}
        for r in scored.values():
            level = r.get('engagement_level', 'LOW')
            if level in distribution:
                distribution[level] += 1
        
        # Count identified vs unknown
        identified = sum(
            1 for r in student_results.values()
            if r.get('features', {}).get('student_id') is not None
        )
        unknown = len(student_results) - identified
        
        # Aggregate behavior metrics from features
        behavior_breakdown = self._aggregate_behavior(
            [r.get('features', {}) for r in student_results.values()]
        )
        
        # Participation analytics
        participation = self._analyze_participation(
            student_results
        )
        
        total_observed = len(student_results)
        if identity_summary:
            total_observed = max(
                total_observed,
                identity_summary.get('total_tracks', total_observed)
            )
        
        return {
            'students_observed': total_observed,
            'identified_students': identity_summary.get('confirmed', identified) if identity_summary else identified,
            'unknown_tracks': identity_summary.get('unknown', unknown) if identity_summary else unknown,
            'scored_students': len(scored),
            'insufficient_data_count': len(insufficient),
            'average_engagement': round(
                statistics.mean(scores), 1
            ) if scores else 0.0,
            'median_engagement': round(
                statistics.median(scores), 1
            ) if scores else 0.0,
            'min_engagement': round(min(scores), 1) if scores else 0.0,
            'max_engagement': round(max(scores), 1) if scores else 0.0,
            'engagement_distribution': distribution,
            'participation': participation,
            'behavior_breakdown': behavior_breakdown,
        }
    
    def analyze_timeline(self, events: list,
                          session_duration: float,
                          feature_extractor,
                          engagement_model) -> list[dict]:
        """Generate time-window engagement timeline.
        
        Args:
            events: All BehaviorEvent objects (or dicts)
            session_duration: Total session duration in seconds
            feature_extractor: FeatureExtractor instance
            engagement_model: BaseEngagementModel instance
        
        Returns:
            List of time-window dicts with engagement indicators.
        """
        if session_duration <= 0:
            return []
        
        timeline = []
        window_start = 0.0
        
        while window_start < session_duration:
            window_end = min(
                window_start + self.time_window,
                session_duration
            )
            
            # Extract features for this window
            window_features = feature_extractor.extract_for_window(
                events, window_start, window_end
            )
            
            # Calculate aggregate engagement for the window
            window_scores = []
            window_events = 0
            student_count = len(window_features)
            
            for features in window_features.values():
                result = engagement_model.calculate(features)
                window_scores.append(result['score'])
                window_events += features.get('total_event_count', 0)
            
            engagement = (
                round(statistics.mean(window_scores), 1)
                if window_scores else 0.0
            )
            
            timeline.append({
                'start': round(window_start, 2),
                'end': round(window_end, 2),
                'engagement_indicator': engagement,
                'student_count': student_count,
                'events': window_events,
            })
            
            window_start = window_end
        
        return timeline
    
    def _analyze_participation(self, student_results: dict) -> dict:
        """Analyze raised-hand participation metrics."""
        total_hands = 0
        participating = set()
        
        for key, result in student_results.items():
            features = result.get('features', {})
            hand_count = features.get('raised_hand_count', 0)
            total_hands += hand_count
            if hand_count > 0:
                participating.add(key)
        
        total_observed = len(student_results)
        rate = (
            len(participating) / total_observed
            if total_observed > 0 else 0.0
        )
        avg_per_student = (
            total_hands / total_observed
            if total_observed > 0 else 0.0
        )
        
        return {
            'total_raised_hand_events': total_hands,
            'unique_participating_students': len(participating),
            'participation_rate': round(rate, 4),
            'average_per_student': round(avg_per_student, 2),
        }
    
    def _aggregate_behavior(self, features_list: list[dict]) -> dict:
        """Aggregate behavior durations and counts across all students."""
        behavior_types = [
            'phone_usage', 'raised_hand', 'head_down',
            'head_left', 'head_right', 'possible_sleeping'
        ]
        
        breakdown = {}
        for btype in behavior_types:
            dur_key = {
                'phone_usage': 'phone_duration',
                'raised_hand': 'raised_hand_duration',
                'head_down': 'head_down_duration',
                'head_left': 'head_left_duration',
                'head_right': 'head_right_duration',
                'possible_sleeping': 'possible_sleep_duration',
            }[btype]
            count_key = {
                'phone_usage': 'phone_event_count',
                'raised_hand': 'raised_hand_count',
            }.get(btype)
            
            total_dur = sum(
                f.get(dur_key, 0.0) for f in features_list
            )
            
            entry = {
                'total_duration': round(total_dur, 2),
            }
            if count_key:
                total_count = sum(
                    f.get(count_key, 0) for f in features_list
                )
                entry['event_count'] = total_count
            
            breakdown[btype] = entry
        
        # Head orientation summary
        total_forward = sum(
            f.get('forward_duration', 0.0) for f in features_list
        )
        total_visible = sum(
            f.get('visible_duration', 0.0) for f in features_list
        )
        total_down = sum(
            f.get('head_down_duration', 0.0) for f in features_list
        )
        total_left = sum(
            f.get('head_left_duration', 0.0) for f in features_list
        )
        total_right = sum(
            f.get('head_right_duration', 0.0) for f in features_list
        )
        
        def safe_pct(num, den):
            return round(num / den * 100, 1) if den > 0 else 0.0
        
        breakdown['head_orientation_summary'] = {
            'forward_pct': safe_pct(total_forward, total_visible),
            'down_pct': safe_pct(total_down, total_visible),
            'left_pct': safe_pct(total_left, total_visible),
            'right_pct': safe_pct(total_right, total_visible),
        }
        
        return breakdown
    
    def _empty_classroom(self, identity_summary: Optional[dict] = None) -> dict:
        """Return empty classroom analytics."""
        return {
            'students_observed': identity_summary.get('total_tracks', 0) if identity_summary else 0,
            'identified_students': identity_summary.get('confirmed', 0) if identity_summary else 0,
            'unknown_tracks': identity_summary.get('unknown', 0) if identity_summary else 0,
            'scored_students': 0,
            'insufficient_data_count': 0,
            'average_engagement': 0.0,
            'median_engagement': 0.0,
            'min_engagement': 0.0,
            'max_engagement': 0.0,
            'engagement_distribution': {'HIGH': 0, 'MODERATE': 0, 'LOW': 0},
            'participation': {
                'total_raised_hand_events': 0,
                'unique_participating_students': 0,
                'participation_rate': 0.0,
                'average_per_student': 0.0,
            },
            'behavior_breakdown': {},
        }
