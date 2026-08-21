from ultralytics import YOLO
import numpy as np
from typing import Optional

class StudentTracker:
    """ByteTrack-based student tracker using Ultralytics built-in tracking.
    
    Uses YOLO's .track() method with ByteTrack for persistent ID assignment.
    Track IDs represent tracked persons, NOT actual student identities.
    Identity recognition will be added in a future phase.
    """
    
    PERSON_CLASS_ID = 0
    
    def __init__(self, model_name: str = 'yolo11n.pt',
                 confidence: float = 0.4,
                 iou: float = 0.5):
        self.model_name = model_name
        self.confidence = confidence
        self.iou = iou
        self.model = None
        self.unique_track_ids: set = set()
        self._load_model()
    
    def _load_model(self):
        try:
            self.model = YOLO(self.model_name)
            print(f"[INFO] Tracker loaded model: {self.model_name}")
        except Exception as e:
            raise RuntimeError(f"Failed to load YOLO model '{self.model_name}': {e}")
    
    def track(self, frame: np.ndarray) -> dict:
        """Track persons in a frame using ByteTrack.
        
        Args:
            frame: BGR image as numpy array
            
        Returns:
            dict with keys:
                - 'boxes': list of [x1, y1, x2, y2]
                - 'confidences': list of float
                - 'track_ids': list of int track IDs
                - 'count': number of tracked persons in this frame
                - 'unique_ids_total': total unique IDs seen so far
        """
        import time
        start_time = time.time()
        
        results = self.model.track(
            frame,
            conf=self.confidence,
            iou=self.iou,
            classes=[self.PERSON_CLASS_ID],
            persist=True,
            tracker='bytetrack.yaml',
            verbose=False
        )
        
        inference_time = time.time() - start_time
        
        boxes = []
        confidences = []
        track_ids = []
        
        if results and len(results) > 0:
            result = results[0]
            if result.boxes is not None and len(result.boxes) > 0:
                for box in result.boxes:
                    xyxy = box.xyxy[0].cpu().numpy().tolist()
                    conf = float(box.conf[0].cpu().numpy())
                    # track ID may be None if tracking hasn't assigned one yet
                    track_id = int(box.id[0].cpu().numpy()) if box.id is not None else -1
                    
                    boxes.append(xyxy)
                    confidences.append(conf)
                    track_ids.append(track_id)
                    
                    if track_id != -1:
                        self.unique_track_ids.add(track_id)
        
        return {
            'boxes': boxes,
            'confidences': confidences,
            'track_ids': track_ids,
            'count': len(boxes),
            'unique_ids_total': len(self.unique_track_ids),
            'inference_time': inference_time
        }
    
    def reset(self):
        """Reset tracking state."""
        self.unique_track_ids.clear()
        # Re-load model to reset tracker state
        self._load_model()
    
    def get_unique_ids(self) -> set:
        """Return all unique track IDs observed."""
        return self.unique_track_ids.copy()
