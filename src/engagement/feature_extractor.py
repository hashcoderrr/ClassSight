"""Behavioral feature extraction from Phase 3 behavior events.

Converts raw BehaviorEvent objects into structured per-student/track
behavioral features suitable for engagement scoring.

Features are purely descriptive — no inference or judgment is applied here.
"""

from typing import Optional


# Event types that contribute duration
_DURATION_EVENT_TYPES = {
    'head_down', 'head_left', 'head_right',
    'phone_usage', 'raised_hand', 'possible_sleeping',
}


class FeatureExtractor:
    """Extract behavioral features from Phase 3 BehaviorEvent list.
    
    Groups events by student/track, aggregates durations and counts,
    and computes normalized ratios relative to visible duration.
    """
    
    def __init__(self, session_duration: float = 0.0,
                 track_visibility: Optional[dict[int, float]] = None):
        """Initialize the feature extractor.
        
        Args:
            session_duration: Total video/session duration in seconds.
            track_visibility: Optional dict mapping track_id -> visible seconds.
                If not provided, session_duration is used as visible_duration
                for all tracks.
        """
        self.session_duration = max(0.0, session_duration)
        self.track_visibility = track_visibility or {}
    
    def extract(self, events: list) -> dict[str, dict]:
        """Extract features for all students/tracks from behavior events.
        
        Args:
            events: List of BehaviorEvent objects (or dicts with same keys)
                from BehaviorEngine.finalize().
        
        Returns:
            Dict mapping student/track key to feature dict.
            Key is student_id if available, else f"track_{track_id}".
        """
        # Group events by effective key
        grouped: dict[str, list] = {}
        key_to_track: dict[str, int] = {}
        key_to_student: dict[str, Optional[str]] = {}
        
        for event in events:
            # Support both BehaviorEvent objects and dicts
            if isinstance(event, dict):
                sid = event.get('student_id')
                tid = event.get('track_id', -1)
                etype = event.get('event_type', '')
                duration = event.get('duration', 0.0)
                confidence = event.get('confidence', 0.0)
            else:
                sid = event.student_id
                tid = event.track_id
                etype = event.event_type
                duration = event.duration
                confidence = event.confidence
            
            key = sid if sid else f"track_{tid}"
            if key not in grouped:
                grouped[key] = []
                key_to_track[key] = tid
                key_to_student[key] = sid
            grouped[key].append({
                'event_type': etype,
                'duration': duration,
                'confidence': confidence,
                'track_id': tid,
            })
        
        # Extract features per key
        result = {}
        for key, event_list in grouped.items():
            tid = key_to_track[key]
            sid = key_to_student[key]
            result[key] = self._extract_single(
                key, tid, sid, event_list
            )
        
        return result
    
    def _extract_single(self, key: str, track_id: int,
                         student_id: Optional[str],
                         events: list[dict]) -> dict:
        """Extract features for a single student/track."""
        visible = self.track_visibility.get(
            track_id, self.session_duration
        )
        visible = max(0.0, visible)
        
        # Sum durations per event type
        durations: dict[str, float] = {}
        counts: dict[str, int] = {}
        total_confidence = 0.0
        
        for ev in events:
            etype = ev['event_type']
            dur = max(0.0, ev['duration'])
            durations[etype] = durations.get(etype, 0.0) + dur
            counts[etype] = counts.get(etype, 0) + 1
            total_confidence += ev['confidence']
        
        # Absolute durations
        head_down = durations.get('head_down', 0.0)
        head_left = durations.get('head_left', 0.0)
        head_right = durations.get('head_right', 0.0)
        phone = durations.get('phone_usage', 0.0)
        raised_hand = durations.get('raised_hand', 0.0)
        sleep = durations.get('possible_sleeping', 0.0)
        
        # Forward duration = visible - non-forward head time (clamped >= 0)
        non_forward = head_down + head_left + head_right
        forward = max(0.0, visible - non_forward)
        
        # Visible minutes for rate calculations
        visible_minutes = visible / 60.0 if visible > 0 else 0.0
        
        # Normalized ratios (safe division)
        def safe_ratio(numerator: float, denominator: float) -> float:
            return numerator / denominator if denominator > 0 else 0.0
        
        total_events = sum(counts.values())
        avg_confidence = (
            total_confidence / total_events if total_events > 0 else 0.0
        )
        
        return {
            'student_id': student_id,
            'track_id': track_id,
            'session_duration': round(self.session_duration, 2),
            'visible_duration': round(visible, 2),
            # Absolute durations
            'forward_duration': round(forward, 2),
            'head_down_duration': round(head_down, 2),
            'head_left_duration': round(head_left, 2),
            'head_right_duration': round(head_right, 2),
            'phone_duration': round(phone, 2),
            'raised_hand_duration': round(raised_hand, 2),
            'possible_sleep_duration': round(sleep, 2),
            # Normalized ratios
            'forward_ratio': round(safe_ratio(forward, visible), 4),
            'head_down_ratio': round(safe_ratio(head_down, visible), 4),
            'head_left_ratio': round(safe_ratio(head_left, visible), 4),
            'head_right_ratio': round(safe_ratio(head_right, visible), 4),
            'phone_ratio': round(safe_ratio(phone, visible), 4),
            'sleep_ratio': round(safe_ratio(sleep, visible), 4),
            # Count features
            'raised_hand_count': counts.get('raised_hand', 0),
            'phone_event_count': counts.get('phone_usage', 0),
            'total_event_count': total_events,
            # Rate features (per minute)
            'raised_hand_rate': round(
                safe_ratio(counts.get('raised_hand', 0), visible_minutes), 4
            ),
            'event_rate': round(
                safe_ratio(total_events, visible_minutes), 4
            ),
            'average_confidence': round(avg_confidence, 4),
        }

    def extract_for_window(self, events: list,
                            window_start: float,
                            window_end: float) -> dict[str, dict]:
        """Extract features for events overlapping a time window.
        
        Args:
            events: Full event list
            window_start: Window start time in seconds
            window_end: Window end time in seconds
        
        Returns:
            Feature dict for the window period
        """
        window_duration = max(0.0, window_end - window_start)
        
        # Filter events overlapping with window
        filtered = []
        for event in events:
            if isinstance(event, dict):
                st = event.get('start_time', 0.0)
                et = event.get('end_time', 0.0)
            else:
                st = event.start_time
                et = event.end_time
            
            # Check overlap
            if st < window_end and et > window_start:
                # Clip duration to window
                clipped_start = max(st, window_start)
                clipped_end = min(et, window_end)
                clipped_duration = max(0.0, clipped_end - clipped_start)
                
                if isinstance(event, dict):
                    ev_copy = dict(event)
                    ev_copy['duration'] = clipped_duration
                else:
                    ev_copy = {
                        'track_id': event.track_id,
                        'student_id': event.student_id,
                        'event_type': event.event_type,
                        'duration': clipped_duration,
                        'confidence': event.confidence,
                        'start_time': clipped_start,
                        'end_time': clipped_end,
                    }
                filtered.append(ev_copy)
        
        # Create a temporary extractor for the window
        window_extractor = FeatureExtractor(
            session_duration=window_duration
        )
        return window_extractor.extract(filtered)
