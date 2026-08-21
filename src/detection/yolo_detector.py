from ultralytics import YOLO
import numpy as np
import time
from pathlib import Path

class YOLODetector:
    """YOLO-based person detector for classroom environments."""
    
    PERSON_CLASS_ID = 0  # COCO 'person' class
    
    def __init__(self, model_name: str = 'yolo11n.pt', 
                 confidence: float = 0.4, 
                 iou: float = 0.5):
        """Initialize the YOLO detector.
        
        Args:
            model_name: Name of the YOLO model to load (auto-downloads if not present)
            confidence: Minimum confidence threshold for detections
            iou: IoU threshold for NMS
        """
        self.model_name = model_name
        self.confidence = confidence
        self.iou = iou
        self.model = None
        self._load_model()
    
    def _load_model(self):
        try:
            self.model = YOLO(self.model_name)
            print(f"[INFO] Loaded model: {self.model_name}")
        except Exception as e:
            raise RuntimeError(f"Failed to load YOLO model '{self.model_name}': {e}")
    
    def detect(self, frame: np.ndarray) -> dict:
        """Run person detection on a single frame.
        
        Args:
            frame: BGR image as numpy array
            
        Returns:
            dict with keys:
                - 'boxes': list of [x1, y1, x2, y2] bounding boxes
                - 'confidences': list of confidence scores
                - 'count': number of persons detected
                - 'inference_time': time taken in seconds
                - 'results': raw ultralytics Results object
        """
        start_time = time.time()
        
        results = self.model(
            frame,
            conf=self.confidence,
            iou=self.iou,
            classes=[self.PERSON_CLASS_ID],
            verbose=False
        )
        
        inference_time = time.time() - start_time
        
        boxes = []
        confidences = []
        
        if results and len(results) > 0:
            result = results[0]
            if result.boxes is not None and len(result.boxes) > 0:
                for box in result.boxes:
                    xyxy = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0].cpu().numpy())
                    boxes.append(xyxy.tolist())
                    confidences.append(conf)
        
        return {
            'boxes': boxes,
            'confidences': confidences,
            'count': len(boxes),
            'inference_time': inference_time,
            'results': results
        }
    
    def detect_image(self, image_path: str) -> dict:
        """Run detection on an image file."""
        import cv2
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")
        return self.detect(image)
