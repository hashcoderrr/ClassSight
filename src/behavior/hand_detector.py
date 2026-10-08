"""Raised hand detection using pose keypoint geometry.

Detects raised hands by analyzing the spatial relationship between
wrist, elbow, and shoulder keypoints. Uses normalized coordinates
relative to body proportions for resolution independence.
"""

import numpy as np
from typing import Optional


class HandDetector:
    """Detects raised hands from pose keypoint spatial relationships.
    
    A hand is considered "raised" when:
    - Wrist Y < Shoulder Y (in image coords, higher position = smaller Y)
    - Elbow Y < Shoulder Y (arm is extended upward, not just bent)
    - Keypoints have sufficient confidence
    
    Uses ratios relative to shoulder-to-hip distance for resolution
    independence across different camera distances and resolutions.
    """
    
    def __init__(self, min_keypoint_confidence: float = 0.3,
                 wrist_above_factor: float = 0.0):
        """Initialize the hand detector.
        
        Args:
            min_keypoint_confidence: Minimum confidence for keypoints to be used
            wrist_above_factor: How far above shoulder the wrist must be,
                as a fraction of torso height. 0.0 = at shoulder level.
                Negative = above shoulder. Default 0.0 works well.
        """
        self.min_kp_conf = min_keypoint_confidence
        self.wrist_above_factor = wrist_above_factor
    
    def detect(self, pose_results: dict[int, dict]) -> dict[int, dict]:
        """Detect raised hands from pose keypoints.
        
        Args:
            pose_results: dict from PoseAnalyzer.analyze()
                {track_id: {'keypoints': np.ndarray(17,3), 'valid': bool, 'named_keypoints': {...}}}
        
        Returns:
            dict mapping track_id -> {
                'detected': bool,
                'confidence': float,
                'hand': 'left' | 'right' | 'both' | None
            }
        """
        results = {}
        
        for tid, pose in pose_results.items():
            if not pose.get('valid', False):
                results[tid] = {'detected': False, 'confidence': 0.0, 'hand': None}
                continue
            
            kps = pose['keypoints']  # (17, 3): x, y, conf
            named = pose.get('named_keypoints', {})
            
            left_raised = self._check_hand_raised(
                named, 'left_wrist', 'left_elbow', 'left_shoulder', 'left_hip'
            )
            right_raised = self._check_hand_raised(
                named, 'right_wrist', 'right_elbow', 'right_shoulder', 'right_hip'
            )
            
            if left_raised['raised'] and right_raised['raised']:
                hand = 'both'
                confidence = max(left_raised['confidence'], right_raised['confidence'])
            elif left_raised['raised']:
                hand = 'left'
                confidence = left_raised['confidence']
            elif right_raised['raised']:
                hand = 'right'
                confidence = right_raised['confidence']
            else:
                hand = None
                confidence = 0.0
            
            results[tid] = {
                'detected': hand is not None,
                'confidence': confidence,
                'hand': hand
            }
        
        return results
    
    def _check_hand_raised(self, named_kps: dict,
                            wrist_name: str, elbow_name: str,
                            shoulder_name: str, hip_name: str) -> dict:
        """Check if a specific hand is raised.
        
        Returns: {'raised': bool, 'confidence': float}
        """
        # Need at least wrist and shoulder
        if wrist_name not in named_kps or shoulder_name not in named_kps:
            return {'raised': False, 'confidence': 0.0}
        
        wx, wy, wc = named_kps[wrist_name]
        sx, sy, sc = named_kps[shoulder_name]
        
        if wc < self.min_kp_conf or sc < self.min_kp_conf:
            return {'raised': False, 'confidence': 0.0}
        
        # Compute torso height for normalization
        torso_height = 0
        if hip_name in named_kps:
            hx, hy, hc = named_kps[hip_name]
            if hc >= self.min_kp_conf:
                torso_height = abs(hy - sy)
        
        if torso_height == 0:
            torso_height = 100  # Fallback: use absolute pixels
        
        # Threshold: wrist must be above shoulder by wrist_above_factor * torso_height
        threshold_y = sy - self.wrist_above_factor * torso_height
        
        # In image coords: smaller Y = higher position
        wrist_is_raised = wy < threshold_y
        
        # Bonus: check if elbow is also above shoulder (stronger signal)
        elbow_bonus = 0.0
        if elbow_name in named_kps:
            ex, ey, ec = named_kps[elbow_name]
            if ec >= self.min_kp_conf and ey < sy:
                elbow_bonus = 0.1
        
        if wrist_is_raised:
            # Confidence based on how far above threshold the wrist is
            raise_distance = threshold_y - wy
            normalized_dist = min(raise_distance / torso_height, 1.0)
            confidence = min(1.0, 0.6 + normalized_dist * 0.3 + elbow_bonus)
            confidence = confidence * min(wc, sc)  # Scale by keypoint confidence
            return {'raised': True, 'confidence': float(confidence)}
        
        return {'raised': False, 'confidence': 0.0}
