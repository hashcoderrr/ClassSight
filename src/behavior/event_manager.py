"""Temporal behavior event confirmation and aggregation.

Ensures no behavior event is generated from a single frame prediction.
Requires N consecutive positive detections before confirming an event,
and discards events shorter than a minimum duration.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class BehaviorEvent:
    """A confirmed behavior event."""
    track_id: int
    student_id: Optional[str]  # From Phase 2 identity mapping, or None
    event_type: str            # 'phone_usage', 'raised_hand', 'head_down', etc.
    confidence: float          # Average confidence over event duration
    start_time: float          # Video timestamp in seconds
    end_time: float            # Video timestamp in seconds
    duration: float            # end_time - start_time
    start_frame: int
    end_frame: int

    def to_dict(self) -> dict:
        return {
            'track_id': self.track_id,
            'student_id': self.student_id,
            'event_type': self.event_type,
            'confidence': round(self.confidence, 4),
            'start_time': round(self.start_time, 2),
            'end_time': round(self.end_time, 2),
            'duration': round(self.duration, 2),
            'start_frame': self.start_frame,
            'end_frame': self.end_frame
        }


@dataclass
class _PendingEvent:
    """Internal state for an event being accumulated."""
    track_id: int
    event_type: str
    consecutive_frames: int = 0
    total_confidence: float = 0.0
    total_frames: int = 0
    start_frame: int = 0
    end_frame: int = 0
    is_active: bool = False    # True once confirmation threshold reached
    gap_frames: int = 0         # Consecutive frames without detection


class EventManager:
    """Manages temporal confirmation of behavior events.
    
    Events go through this lifecycle:
    1. Raw detection → accumulate in _PendingEvent
    2. After confirmation_frames consecutive detections → event becomes 'active'
    3. While detections continue → event stays active, extends duration
    4. When detections stop for gap_tolerance frames → event ends
    5. If event duration >= min_duration → emit as BehaviorEvent
    6. If event duration < min_duration → discard
    """
    
    def __init__(self, confirmation_frames: int = 3,
                 min_duration: float = 0.5,
                 fps: float = 25.0,
                 gap_tolerance: int = 2):
        """Initialize the event manager.
        
        Args:
            confirmation_frames: Consecutive positive frames to confirm event
            min_duration: Minimum event duration in seconds to emit
            fps: Video FPS for timestamp calculation
            gap_tolerance: Frames without detection before ending an event
        """
        self.confirmation_frames = max(1, confirmation_frames)
        self.min_duration = max(0.0, min_duration)
        self.fps = max(1.0, fps)
        self.gap_tolerance = max(1, gap_tolerance)
        
        # {(track_id, event_type): _PendingEvent}
        self._pending: dict[tuple, _PendingEvent] = {}
        self._confirmed_events: list[BehaviorEvent] = []
    
    def update(self, track_id: int, event_type: str,
               detected: bool, confidence: float,
               frame_number: int):
        """Update with a per-frame raw prediction.
        
        Args:
            track_id: ByteTrack track ID
            event_type: Event type string (e.g., 'phone_usage')
            detected: Whether the behavior was detected this frame
            confidence: Detection confidence (0-1)
            frame_number: Current frame number (1-indexed)
        """
        key = (track_id, event_type)
        
        if detected:
            if key not in self._pending:
                self._pending[key] = _PendingEvent(
                    track_id=track_id,
                    event_type=event_type,
                    start_frame=frame_number
                )
            
            pending = self._pending[key]
            pending.consecutive_frames += 1
            pending.total_confidence += confidence
            pending.total_frames += 1
            pending.end_frame = frame_number
            pending.gap_frames = 0
            
            if not pending.is_active:
                if pending.consecutive_frames >= self.confirmation_frames:
                    pending.is_active = True
        else:
            if key in self._pending:
                pending = self._pending[key]
                pending.gap_frames += 1
                pending.consecutive_frames = 0  # Reset consecutive count
                
                if pending.gap_frames > self.gap_tolerance:
                    # Event has ended
                    if pending.is_active:
                        self._finalize_event(pending)
                    del self._pending[key]
    
    def _finalize_event(self, pending: _PendingEvent,
                         student_id: Optional[str] = None):
        """Convert a pending event to a confirmed BehaviorEvent."""
        start_time = (pending.start_frame - 1) / self.fps
        end_time = (pending.end_frame - 1) / self.fps
        duration = end_time - start_time
        
        if duration < self.min_duration:
            return  # Discard too-short events
        
        avg_conf = (pending.total_confidence / pending.total_frames
                    if pending.total_frames > 0 else 0.0)
        
        event = BehaviorEvent(
            track_id=pending.track_id,
            student_id=student_id,
            event_type=pending.event_type,
            confidence=avg_conf,
            start_time=start_time,
            end_time=end_time,
            duration=duration,
            start_frame=pending.start_frame,
            end_frame=pending.end_frame
        )
        self._confirmed_events.append(event)
    
    def get_active_events(self, track_id: int) -> dict[str, bool]:
        """Get currently active (in-progress) event types for a track.
        
        Returns: dict mapping event_type -> True for active events
        """
        active = {}
        for (tid, etype), pending in self._pending.items():
            if tid == track_id and pending.is_active:
                active[etype] = True
        return active
    
    def get_all_active_events(self) -> dict[int, dict[str, bool]]:
        """Get all active events grouped by track_id."""
        result: dict[int, dict[str, bool]] = {}
        for (tid, etype), pending in self._pending.items():
            if pending.is_active:
                if tid not in result:
                    result[tid] = {}
                result[tid][etype] = True
        return result
    
    def get_confirmed_events(self) -> list[BehaviorEvent]:
        """Return all completed, confirmed events."""
        return list(self._confirmed_events)
    
    def finalize(self, identity_mappings: Optional[dict] = None) -> list[BehaviorEvent]:
        """Close all pending events at video end.
        
        Args:
            identity_mappings: Optional dict mapping track_id -> {'student_id': str, 'name': str}
                to attach student IDs to events.
        
        Returns:
            All confirmed events (including newly finalized ones)
        """
        for key in list(self._pending.keys()):
            pending = self._pending[key]
            if pending.is_active:
                sid = None
                if identity_mappings and pending.track_id in identity_mappings:
                    sid = identity_mappings[pending.track_id].get('student_id')
                self._finalize_event(pending, student_id=sid)
        self._pending.clear()
        
        # Attach student IDs to all events if mappings provided
        if identity_mappings:
            for event in self._confirmed_events:
                if event.student_id is None and event.track_id in identity_mappings:
                    event.student_id = identity_mappings[event.track_id].get('student_id')
        
        return list(self._confirmed_events)
    
    def get_event_counts(self) -> dict[str, int]:
        """Count confirmed events by type."""
        counts: dict[str, int] = {}
        for event in self._confirmed_events:
            counts[event.event_type] = counts.get(event.event_type, 0) + 1
        return counts
    
    def reset(self):
        """Reset all state."""
        self._pending.clear()
        self._confirmed_events.clear()
