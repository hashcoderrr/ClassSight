"""Face detection and quality assessment using InsightFace."""

import cv2
import numpy as np
from typing import Optional


class FaceDetector:
    """Detects faces and extracts embeddings using InsightFace.
    
    Uses the InsightFace FaceAnalysis pipeline which includes:
    - SCRFD face detection
    - ArcFace face recognition (embedding extraction)
    """
    
    def __init__(self, model_name: str = 'buffalo_l', 
                 min_face_size: int = 40,
                 det_size: tuple = (640, 640)):
        """Initialize the face detector.
        
        Args:
            model_name: InsightFace model pack name (auto-downloads on first use)
            min_face_size: Minimum face size in pixels to consider valid
            det_size: Detection input size for the face detector
        """
        self.model_name = model_name
        self.min_face_size = min_face_size
        self.det_size = det_size
        self.app = None
        self._load_model()
    
    def _load_model(self):
        """Load the InsightFace model."""
        try:
            from insightface.app import FaceAnalysis
            self.app = FaceAnalysis(
                name=self.model_name,
                providers=['CPUExecutionProvider']
            )
            self.app.prepare(ctx_id=0, det_size=self.det_size)
            print(f"[INFO] InsightFace model loaded: {self.model_name}")
        except Exception as e:
            raise RuntimeError(f"Failed to load InsightFace model '{self.model_name}': {e}")
    
    def detect_faces(self, frame: np.ndarray) -> list[dict]:
        """Detect all faces in a frame.
        
        Args:
            frame: BGR image as numpy array
            
        Returns:
            List of face dicts, each containing:
                - 'bbox': [x1, y1, x2, y2] in frame coordinates
                - 'det_score': detection confidence
                - 'embedding': 512-dim numpy array (L2-normalized)
                - 'landmarks': facial landmarks
                - 'face_size': size of face in pixels (max of width/height)
        """
        if frame is None or frame.size == 0:
            return []
        
        faces = self.app.get(frame)
        results = []
        
        for face in faces:
            bbox = face.bbox.astype(int).tolist()
            x1, y1, x2, y2 = bbox
            face_w = x2 - x1
            face_h = y2 - y1
            face_size = max(face_w, face_h)
            
            results.append({
                'bbox': bbox,
                'det_score': float(face.det_score),
                'embedding': face.embedding,  # 512-dim, L2-normalized
                'landmarks': face.kps if hasattr(face, 'kps') else None,
                'face_size': face_size
            })
        
        return results
    
    def detect_in_region(self, frame: np.ndarray, 
                          person_bbox: list) -> list[dict]:
        """Detect faces within a person bounding box region.
        
        Crops the upper portion of the person bbox (where the face typically is)
        and runs face detection on the crop. Maps coordinates back to full frame.
        
        Args:
            frame: Full BGR frame
            person_bbox: [x1, y1, x2, y2] of the person detection
            
        Returns:
            List of face dicts with coordinates in full-frame space
        """
        h, w = frame.shape[:2]
        px1, py1, px2, py2 = [int(v) for v in person_bbox]
        
        # Clamp to frame bounds
        px1 = max(0, px1)
        py1 = max(0, py1)
        px2 = min(w, px2)
        py2 = min(h, py2)
        
        # Use upper 60% of person bbox (face is typically in the top portion)
        person_h = py2 - py1
        crop_y2 = py1 + int(person_h * 0.6)
        crop_y2 = min(crop_y2, py2)
        
        crop = frame[py1:crop_y2, px1:px2]
        
        if crop.size == 0 or crop.shape[0] < 10 or crop.shape[1] < 10:
            return []
        
        faces = self.detect_faces(crop)
        
        # Map coordinates back to full frame
        for face in faces:
            bx1, by1, bx2, by2 = face['bbox']
            face['bbox'] = [
                bx1 + px1,
                by1 + py1,
                bx2 + px1,
                by2 + py1
            ]
        
        return faces
    
    def check_quality(self, face: dict, 
                       blur_threshold: float = 50.0) -> dict:
        """Check the quality of a detected face.
        
        Args:
            face: Face dict from detect_faces()
            blur_threshold: Minimum Laplacian variance to consider non-blurry
            
        Returns:
            dict with:
                - 'is_valid': bool
                - 'reasons': list of rejection reasons (empty if valid)
                - 'det_score': detection score
                - 'face_size': face size in pixels
        """
        reasons = []
        
        # Check face size
        if face['face_size'] < self.min_face_size:
            reasons.append(f"Face too small: {face['face_size']}px (min: {self.min_face_size}px)")
        
        # Check detection confidence
        if face['det_score'] < 0.5:
            reasons.append(f"Low detection confidence: {face['det_score']:.2f}")
        
        # Check embedding exists
        if face.get('embedding') is None:
            reasons.append("No embedding generated")
        
        return {
            'is_valid': len(reasons) == 0,
            'reasons': reasons,
            'det_score': face['det_score'],
            'face_size': face['face_size']
        }
    
    def check_image_quality(self, image: np.ndarray, 
                             face: dict,
                             blur_threshold: float = 50.0) -> dict:
        """Check quality including blur analysis on the actual face crop.
        
        Args:
            image: The source image
            face: Face dict from detect_faces()
            blur_threshold: Minimum Laplacian variance
            
        Returns:
            Quality dict with is_valid, reasons, blur_score
        """
        quality = self.check_quality(face, blur_threshold)
        
        # Blur check on face crop
        x1, y1, x2, y2 = [int(v) for v in face['bbox']]
        h, w = image.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        if x2 > x1 and y2 > y1:
            face_crop = image[y1:y2, x1:x2]
            if face_crop.size > 0:
                gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
                blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
                quality['blur_score'] = blur_score
                if blur_score < blur_threshold:
                    quality['reasons'].append(
                        f"Face too blurry: variance={blur_score:.1f} (min: {blur_threshold})")
                    quality['is_valid'] = False
        
        return quality
