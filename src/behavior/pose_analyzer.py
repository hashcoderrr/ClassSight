"""Pose estimation using YOLO11n-pose for classroom behavior analysis.

Extracts 17 COCO keypoints per detected person and matches them
to tracked persons from Phase 1 (ByteTrack) via bounding box IoU.
"""

import numpy as np
from typing import Optional

# COCO keypoint indices
KP_NOSE = 0
KP_LEFT_EYE = 1
KP_RIGHT_EYE = 2
KP_LEFT_EAR = 3
KP_RIGHT_EAR = 4
KP_LEFT_SHOULDER = 5
KP_RIGHT_SHOULDER = 6
KP_LEFT_ELBOW = 7
KP_RIGHT_ELBOW = 8
KP_LEFT_WRIST = 9
KP_RIGHT_WRIST = 10
KP_LEFT_HIP = 11
KP_RIGHT_HIP = 12
KP_LEFT_KNEE = 13
KP_RIGHT_KNEE = 14
KP_LEFT_ANKLE = 15
KP_RIGHT_ANKLE = 16

KEYPOINT_NAMES = [
    'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
    'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
]


def compute_iou(box_a: list, box_b: list) -> float:
    """Compute Intersection over Union between two [x1,y1,x2,y2] boxes."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = max(0, box_a[2] - box_a[0]) * max(0, box_a[3] - box_a[1])
    area_b = max(0, box_b[2] - box_b[0]) * max(0, box_b[3] - box_b[1])
    union = area_a + area_b - inter
    
    return inter / union if union > 0 else 0.0


class PoseAnalyzer:
    """Extracts COCO keypoints using YOLO11n-pose model.
    
    Runs pose estimation on the full frame, then matches detected
    pose skeletons to tracked persons from Phase 1 via bounding box IoU.
    """
    
    def __init__(self, model_name: str = 'yolo11n-pose.pt',
                 confidence: float = 0.3,
                 min_keypoint_confidence: float = 0.3):
        """Initialize the pose analyzer.
        
        Args:
            model_name: YOLO pose model name (auto-downloads if needed)
            confidence: Person detection confidence for pose model
            min_keypoint_confidence: Minimum confidence to consider a keypoint valid
        """
        self.model_name = model_name
        self.confidence = confidence
        self.min_kp_conf = min_keypoint_confidence
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load the YOLO pose model."""
        try:
            from ultralytics import YOLO
            self.model = YOLO(self.model_name)
            print(f"[INFO] Pose model loaded: {self.model_name}")
        except Exception as e:
            raise RuntimeError(f"Failed to load pose model '{self.model_name}': {e}")
    
    def analyze(self, frame: np.ndarray,
                person_boxes: list[list],
                track_ids: list[int]) -> dict[int, dict]:
        """Run pose estimation and match keypoints to tracked persons.
        
        Args:
            frame: BGR frame as numpy array
            person_boxes: Phase 1 tracked person boxes [[x1,y1,x2,y2], ...]
            track_ids: Corresponding track IDs
            
        Returns:
            dict mapping track_id -> {
                'keypoints': np.ndarray (17, 3) with x, y, confidence,
                'valid': bool (enough keypoints detected),
                'bbox': [x1, y1, x2, y2] from pose model,
                'named_keypoints': dict mapping name -> (x, y, conf)
            }
        """
        if frame is None or frame.size == 0 or not person_boxes:
            return {}
        
        # Run pose estimation on full frame
        results = self.model(
            frame,
            conf=self.confidence,
            verbose=False
        )
        
        pose_detections = []
        if results and len(results) > 0:
            result = results[0]
            if result.keypoints is not None and result.boxes is not None:
                kps = result.keypoints.data.cpu().numpy()  # (N, 17, 3)
                boxes = result.boxes.xyxy.cpu().numpy()     # (N, 4)
                
                for i in range(len(boxes)):
                    pose_detections.append({
                        'bbox': boxes[i].tolist(),
                        'keypoints': kps[i]  # (17, 3)
                    })
        
        # Match pose detections to tracked persons via IoU
        pose_results = {}
        used_poses = set()
        
        for bbox, tid in zip(person_boxes, track_ids):
            if tid == -1:
                continue
            
            best_iou = 0.0
            best_idx = -1
            
            for i, pose in enumerate(pose_detections):
                if i in used_poses:
                    continue
                iou = compute_iou(bbox, pose['bbox'])
                if iou > best_iou:
                    best_iou = iou
                    best_idx = i
            
            if best_idx >= 0 and best_iou > 0.3:
                used_poses.add(best_idx)
                kps = pose_detections[best_idx]['keypoints']
                
                # Count valid keypoints
                valid_count = np.sum(kps[:, 2] >= self.min_kp_conf)
                
                # Build named keypoints dict
                named = {}
                for j, name in enumerate(KEYPOINT_NAMES):
                    x, y, conf = kps[j]
                    if conf >= self.min_kp_conf:
                        named[name] = (float(x), float(y), float(conf))
                
                pose_results[tid] = {
                    'keypoints': kps,
                    'valid': valid_count >= 5,  # Need at least 5 keypoints
                    'bbox': pose_detections[best_idx]['bbox'],
                    'named_keypoints': named
                }
            else:
                # No matching pose found for this track
                pose_results[tid] = {
                    'keypoints': np.zeros((17, 3)),
                    'valid': False,
                    'bbox': bbox,
                    'named_keypoints': {}
                }
        
        return pose_results
