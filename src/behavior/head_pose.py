"""Head orientation classification from COCO pose keypoints.

Classifies observable head orientation as FORWARD, DOWN, LEFT, RIGHT,
or UNKNOWN based on geometric relationships between facial and
upper-body keypoints.

IMPORTANT: Head orientation is an observable indicator only.
'Head facing down' does NOT automatically mean 'not attentive' —
the student could be writing, reading, or using a laptop.
"""

import numpy as np
from typing import Optional


class HeadPoseAnalyzer:
    """Classifies head orientation from COCO pose keypoints.
    
    Uses nose, eyes, ears, and shoulders to estimate head direction:
    - FORWARD: nose centered between shoulders, eyes visible
    - DOWN: nose close to shoulder level or below eye level
    - LEFT: nose shifted toward left ear direction
    - RIGHT: nose shifted toward right ear direction
    - UNKNOWN: insufficient keypoints for classification
    
    All comparisons use ratios relative to shoulder width for
    resolution independence.
    """
    
    def __init__(self, min_keypoint_confidence: float = 0.3,
                 horizontal_threshold: float = 0.3,
                 down_threshold: float = 0.15):
        """Initialize head pose analyzer.
        
        Args:
            min_keypoint_confidence: Minimum keypoint confidence to use
            horizontal_threshold: Nose horizontal offset ratio (relative to
                shoulder width) to classify as LEFT/RIGHT
            down_threshold: Nose vertical offset ratio to classify as DOWN
        """
        self.min_kp_conf = min_keypoint_confidence
        self.horizontal_threshold = horizontal_threshold
        self.down_threshold = down_threshold
    
    def analyze(self, pose_results: dict[int, dict]) -> dict[int, dict]:
        """Classify head orientation for each tracked person.
        
        Args:
            pose_results: dict from PoseAnalyzer.analyze()
        
        Returns:
            dict mapping track_id -> {
                'orientation': str,  # FORWARD, DOWN, LEFT, RIGHT, UNKNOWN
                'confidence': float
            }
        """
        results = {}
        
        for tid, pose in pose_results.items():
            if not pose.get('valid', False):
                results[tid] = {'orientation': 'UNKNOWN', 'confidence': 0.0}
                continue
            
            named = pose.get('named_keypoints', {})
            orientation, confidence = self._classify_orientation(named)
            results[tid] = {
                'orientation': orientation,
                'confidence': confidence
            }
        
        return results
    
    def _classify_orientation(self, named_kps: dict) -> tuple[str, float]:
        """Classify head orientation from named keypoints.
        
        Returns: (orientation_str, confidence_float)
        """
        # Need at least nose and one shoulder
        if 'nose' not in named_kps:
            return ('UNKNOWN', 0.0)
        
        has_left_shoulder = 'left_shoulder' in named_kps
        has_right_shoulder = 'right_shoulder' in named_kps
        
        if not has_left_shoulder and not has_right_shoulder:
            return ('UNKNOWN', 0.0)
        
        nx, ny, nc = named_kps['nose']
        
        # Compute shoulder midpoint and width
        if has_left_shoulder and has_right_shoulder:
            lsx, lsy, lsc = named_kps['left_shoulder']
            rsx, rsy, rsc = named_kps['right_shoulder']
            mid_x = (lsx + rsx) / 2
            mid_y = (lsy + rsy) / 2
            shoulder_width = abs(rsx - lsx)
        elif has_left_shoulder:
            lsx, lsy, lsc = named_kps['left_shoulder']
            mid_x = lsx
            mid_y = lsy
            shoulder_width = 100  # Fallback
        else:
            rsx, rsy, rsc = named_kps['right_shoulder']
            mid_x = rsx
            mid_y = rsy
            shoulder_width = 100
        
        if shoulder_width < 10:
            shoulder_width = 100  # Avoid division issues
        
        # --- Check DOWN first ---
        # If nose is close to or below shoulder level
        # In image coords: larger Y = lower position
        vertical_offset = (ny - mid_y) / shoulder_width
        
        # Check eye positions for DOWN verification
        has_eyes = 'left_eye' in named_kps or 'right_eye' in named_kps
        eye_y = None
        if 'left_eye' in named_kps and 'right_eye' in named_kps:
            ley, rey = named_kps['left_eye'][1], named_kps['right_eye'][1]
            eye_y = (ley + rey) / 2
        elif 'left_eye' in named_kps:
            eye_y = named_kps['left_eye'][1]
        elif 'right_eye' in named_kps:
            eye_y = named_kps['right_eye'][1]
        
        # DOWN: nose close to shoulder level or nose-eye gap is small
        if vertical_offset > -self.down_threshold:
            # Nose is near or below shoulder midpoint
            down_conf = min(1.0, 0.5 + max(0, vertical_offset) * 2)
            return ('DOWN', float(down_conf * nc))
        
        # Additional DOWN check: if eyes and nose are very close vertically
        if eye_y is not None:
            nose_eye_gap = abs(ny - eye_y) / shoulder_width
            if nose_eye_gap < 0.05:  # Eyes and nose nearly same Y = looking down
                return ('DOWN', float(0.6 * nc))
        
        # --- Check LEFT/RIGHT ---
        horizontal_offset = (nx - mid_x) / shoulder_width
        
        # Also use ear visibility as secondary signal
        has_left_ear = 'left_ear' in named_kps
        has_right_ear = 'right_ear' in named_kps
        
        if horizontal_offset < -self.horizontal_threshold:
            # Nose is to the left of center
            conf = min(1.0, 0.5 + abs(horizontal_offset) * 1.5)
            # Boost if only right ear is visible (head turned left)
            if has_right_ear and not has_left_ear:
                conf = min(1.0, conf + 0.15)
            return ('LEFT', float(conf * nc))
        
        if horizontal_offset > self.horizontal_threshold:
            conf = min(1.0, 0.5 + abs(horizontal_offset) * 1.5)
            if has_left_ear and not has_right_ear:
                conf = min(1.0, conf + 0.15)
            return ('RIGHT', float(conf * nc))
        
        # --- FORWARD ---
        # Nose is centered, above shoulder level
        forward_conf = min(1.0, 0.7 + (1.0 - abs(horizontal_offset) / self.horizontal_threshold) * 0.3)
        return ('FORWARD', float(forward_conf * nc))
