"""Cell phone detection and association with tracked students.

Uses YOLO's pretrained COCO model (class 67 = 'cell phone') to detect
phones in the frame, then associates each phone with the nearest
tracked student via bounding box proximity or wrist keypoint distance.
"""

import numpy as np
from typing import Optional


def _box_center(box: list) -> tuple[float, float]:
    """Compute center of an [x1, y1, x2, y2] box."""
    return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)


def _distance(p1: tuple, p2: tuple) -> float:
    """Euclidean distance between two 2D points."""
    return ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2) ** 0.5


def _boxes_overlap(box_a: list, box_b: list, margin: int = 0) -> bool:
    """Check if two boxes overlap (with optional margin expansion)."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b[0] - margin, box_b[1] - margin, box_b[2] + margin, box_b[3] + margin
    return not (ax2 < bx1 or ax1 > bx2 or ay2 < by1 or ay1 > by2)


class PhoneDetector:
    """Detects cell phones and associates them with tracked students.
    
    Uses a separate YOLO model instance (same weights as person detector)
    running with classes=[67] to detect cell phones. Each detected phone
    is associated with the nearest tracked person.
    
    Association strategy:
    1. If pose keypoints are available: use wrist distance (most precise)
    2. Otherwise: use bounding box centroid proximity
    3. Only associate if within proximity_pixels threshold
    """
    
    def __init__(self, model_name: str = 'yolo11n.pt',
                 confidence: float = 0.3,
                 phone_class_id: int = 67,
                 proximity_pixels: int = 100):
        """Initialize the phone detector.
        
        Args:
            model_name: YOLO model name (same as person detection model)
            confidence: Minimum confidence for phone detection
            phone_class_id: COCO class ID for cell phone (67)
            proximity_pixels: Maximum distance in pixels to associate phone with person
        """
        self.model_name = model_name
        self.confidence = confidence
        self.phone_class_id = phone_class_id
        self.proximity_pixels = proximity_pixels
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load YOLO model for phone detection."""
        try:
            from ultralytics import YOLO
            self.model = YOLO(self.model_name)
            print(f"[INFO] Phone detector loaded: {self.model_name} (class {self.phone_class_id})")
        except Exception as e:
            raise RuntimeError(f"Failed to load phone detection model: {e}")
    
    def detect(self, frame: np.ndarray,
               person_boxes: list[list],
               track_ids: list[int],
               pose_results: Optional[dict] = None) -> dict[int, dict]:
        """Detect phones and associate with tracked persons.
        
        Args:
            frame: BGR frame
            person_boxes: Phase 1 tracked person boxes [[x1,y1,x2,y2], ...]
            track_ids: Corresponding track IDs
            pose_results: Optional pose keypoints from PoseAnalyzer
                         {track_id: {'named_keypoints': {name: (x,y,conf)}}}
        
        Returns:
            dict mapping track_id -> {
                'detected': bool,
                'confidence': float,
                'phone_bbox': [x1,y1,x2,y2] or None
            }
        """
        results_dict = {}
        for tid in track_ids:
            if tid != -1:
                results_dict[tid] = {
                    'detected': False,
                    'confidence': 0.0,
                    'phone_bbox': None
                }
        
        if frame is None or frame.size == 0:
            return results_dict
        
        # Detect phones
        results = self.model(
            frame,
            conf=self.confidence,
            classes=[self.phone_class_id],
            verbose=False
        )
        
        phone_boxes = []
        phone_confs = []
        
        if results and len(results) > 0:
            result = results[0]
            if result.boxes is not None and len(result.boxes) > 0:
                for box in result.boxes:
                    xyxy = box.xyxy[0].cpu().numpy().tolist()
                    conf = float(box.conf[0].cpu().numpy())
                    phone_boxes.append(xyxy)
                    phone_confs.append(conf)
        
        if not phone_boxes:
            return results_dict
        
        # Associate each phone with nearest person
        for phone_bbox, phone_conf in zip(phone_boxes, phone_confs):
            phone_center = _box_center(phone_bbox)
            best_tid = -1
            best_dist = float('inf')
            
            for bbox, tid in zip(person_boxes, track_ids):
                if tid == -1:
                    continue
                
                # Strategy 1: Check wrist proximity (if keypoints available)
                if pose_results and tid in pose_results:
                    named_kps = pose_results[tid].get('named_keypoints', {})
                    for wrist_name in ['left_wrist', 'right_wrist']:
                        if wrist_name in named_kps:
                            wx, wy, wc = named_kps[wrist_name]
                            dist = _distance(phone_center, (wx, wy))
                            if dist < best_dist:
                                best_dist = dist
                                best_tid = tid
                
                # Strategy 2: Check bounding box proximity/overlap
                if _boxes_overlap(bbox, phone_bbox, margin=self.proximity_pixels // 2):
                    person_center = _box_center(bbox)
                    dist = _distance(phone_center, person_center)
                    if dist < best_dist:
                        best_dist = dist
                        best_tid = tid
            
            # Associate if within threshold
            if best_tid != -1 and best_dist <= self.proximity_pixels:
                # If this person already has a phone detection, keep the higher confidence one
                existing = results_dict.get(best_tid)
                if existing and (not existing['detected'] or phone_conf > existing['confidence']):
                    results_dict[best_tid] = {
                        'detected': True,
                        'confidence': phone_conf,
                        'phone_bbox': phone_bbox
                    }
        
        return results_dict
