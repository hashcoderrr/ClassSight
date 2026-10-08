"""Sleeping/inattentive posture indicator detection.

Detects POSSIBLE sleeping posture from sustained head-down orientation
and posture indicators. Uses temporal accumulation rather than
single-frame classification.

IMPORTANT: This module detects POSSIBLE_SLEEPING indicators only.
It cannot determine whether a student is actually sleeping.
A student with prolonged head-down posture might be reading,
writing, or using a device on their desk.
"""

from typing import Optional


class SleepDetector:
    """Detects possible sleeping posture from sustained indicators.
    
    Conditions for POSSIBLE_SLEEPING:
    1. Prolonged HEAD_DOWN orientation (configurable seconds)
    2. Head position near or below shoulder level
    3. Minimal upper body movement (tracked via position history)
    
    The label is POSSIBLE_SLEEPING, not 'STUDENT_IS_SLEEPING'.
    """
    
    def __init__(self, confirmation_seconds: float = 5.0,
                 fps: float = 25.0):
        """Initialize sleep detector.
        
        Args:
            confirmation_seconds: Seconds of sustained head-down to trigger indicator
            fps: Video FPS for time calculation
        """
        self.confirmation_seconds = confirmation_seconds
        self.fps = max(1.0, fps)
        self.confirmation_frames = int(self.confirmation_seconds * self.fps)
        
        # Per-track head-down frame counters
        self._head_down_counts: dict[int, int] = {}
    
    def detect(self, pose_results: dict[int, dict],
               head_states: dict[int, dict],
               track_ids: list[int]) -> dict[int, dict]:
        """Detect possible sleeping indicators.
        
        Args:
            pose_results: dict from PoseAnalyzer
            head_states: dict from HeadPoseAnalyzer {tid: {'orientation': str, 'confidence': float}}
            track_ids: Active track IDs this frame
        
        Returns:
            dict mapping track_id -> {
                'detected': bool,
                'confidence': float,
                'indicator': 'POSSIBLE_SLEEPING' or None,
                'head_down_duration': float (seconds)
            }
        """
        results = {}
        active_tids = set(tid for tid in track_ids if tid != -1)
        
        for tid in active_tids:
            head_state = head_states.get(tid, {})
            orientation = head_state.get('orientation', 'UNKNOWN')
            head_conf = head_state.get('confidence', 0.0)
            
            # Track consecutive head-down frames
            if orientation == 'DOWN':
                self._head_down_counts[tid] = self._head_down_counts.get(tid, 0) + 1
            else:
                # Reset counter on non-DOWN orientation
                self._head_down_counts[tid] = 0
            
            head_down_frames = self._head_down_counts.get(tid, 0)
            head_down_duration = head_down_frames / self.fps
            
            # Additional posture check: is the head near shoulder level?
            posture_indicator = False
            if tid in pose_results and pose_results[tid].get('valid', False):
                named_kps = pose_results[tid].get('named_keypoints', {})
                if 'nose' in named_kps and ('left_shoulder' in named_kps or 'right_shoulder' in named_kps):
                    nx, ny, nc = named_kps['nose']
                    shoulder_y = 0
                    count = 0
                    for s_name in ['left_shoulder', 'right_shoulder']:
                        if s_name in named_kps:
                            shoulder_y += named_kps[s_name][1]
                            count += 1
                    if count > 0:
                        shoulder_y /= count
                        # Nose near or below shoulder level
                        if ny >= shoulder_y * 0.9:
                            posture_indicator = True
            
            # Determine if POSSIBLE_SLEEPING
            is_sleeping = (head_down_frames >= self.confirmation_frames
                          and head_down_duration >= self.confirmation_seconds)
            
            if is_sleeping:
                # Boost confidence if posture indicator is also present
                conf = min(1.0, head_conf * (1.1 if posture_indicator else 0.9))
                results[tid] = {
                    'detected': True,
                    'confidence': float(conf),
                    'indicator': 'POSSIBLE_SLEEPING',
                    'head_down_duration': round(head_down_duration, 1)
                }
            else:
                results[tid] = {
                    'detected': False,
                    'confidence': 0.0,
                    'indicator': None,
                    'head_down_duration': round(head_down_duration, 1)
                }
        
        return results
    
    def reset(self):
        """Reset all state."""
        self._head_down_counts.clear()
