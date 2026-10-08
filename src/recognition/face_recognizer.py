"""Face recognition orchestrator for classroom environments."""

import numpy as np
from dataclasses import dataclass
from typing import Optional
from pathlib import Path

from .face_detector import FaceDetector
from .embedding_store import EmbeddingStore
from .identity_matcher import IdentityMatcher, MatchResult


@dataclass
class RecognitionResult:
    """Result of face recognition for a single tracked person."""
    track_id: int
    student_id: Optional[str]  # None if unknown or no face
    name: Optional[str]        # None if unknown or no face
    similarity: float          # Best similarity score
    status: str                # 'MATCHED', 'UNKNOWN', 'NO_FACE', 'SMALL_FACE'
    face_bbox: Optional[list]  # Face bounding box if detected
    det_score: float = 0.0     # Face detection confidence


class FaceRecognizer:
    """Orchestrates face detection, embedding extraction, and identity matching.
    
    For each person detected by YOLO+ByteTrack:
    1. Crop the person region from the frame
    2. Detect faces within the person region
    3. Extract face embedding
    4. Compare against enrolled student embeddings
    5. Return match result
    """
    
    def __init__(self, 
                 model_name: str = 'buffalo_l',
                 identities_file: str | Path = 'data/identities/students.json',
                 embeddings_dir: str | Path = 'data/identities/embeddings',
                 threshold: float = 0.4,
                 min_face_size: int = 40):
        """Initialize the face recognizer.
        
        Args:
            model_name: InsightFace model pack name
            identities_file: Path to student identity registry
            embeddings_dir: Path to stored embeddings
            threshold: Recognition similarity threshold
            min_face_size: Minimum face size in pixels
        """
        self.detector = FaceDetector(
            model_name=model_name,
            min_face_size=min_face_size
        )
        self.store = EmbeddingStore(
            identities_file=identities_file,
            embeddings_dir=embeddings_dir
        )
        self.matcher = IdentityMatcher(threshold=threshold)
        self._enrolled = {}  # Cached enrolled embeddings
        self._load_enrolled()
    
    def _load_enrolled(self):
        """Load enrolled student embeddings into memory."""
        self._enrolled = self.store.load_all_embeddings()
        print(f"[INFO] {len(self._enrolled)} enrolled students loaded for recognition")
    
    def reload_enrolled(self):
        """Reload enrolled embeddings (e.g., after new enrollment)."""
        self._load_enrolled()
    
    def recognize_in_frame(self, frame: np.ndarray,
                            person_boxes: list[list],
                            track_ids: list[int]) -> list[RecognitionResult]:
        """Run face recognition on all tracked persons in a frame.
        
        Args:
            frame: Full BGR frame
            person_boxes: List of person bounding boxes [x1,y1,x2,y2]
            track_ids: Corresponding track IDs
            
        Returns:
            List of RecognitionResult for each person
        """
        results = []
        
        for bbox, tid in zip(person_boxes, track_ids):
            if tid == -1:
                results.append(RecognitionResult(
                    track_id=tid,
                    student_id=None,
                    name=None,
                    similarity=0.0,
                    status='NO_FACE',
                    face_bbox=None
                ))
                continue
            
            result = self._recognize_person(frame, bbox, tid)
            results.append(result)
        
        return results
    
    def _recognize_person(self, frame: np.ndarray,
                           person_bbox: list,
                           track_id: int) -> RecognitionResult:
        """Recognize a single person."""
        # Detect face within person region
        faces = self.detector.detect_in_region(frame, person_bbox)
        
        if not faces:
            return RecognitionResult(
                track_id=track_id,
                student_id=None,
                name=None,
                similarity=0.0,
                status='NO_FACE',
                face_bbox=None
            )
        
        # Use the best (highest det_score) face
        best_face = max(faces, key=lambda f: f['det_score'])
        
        # Check face quality
        quality = self.detector.check_quality(best_face)
        if not quality['is_valid']:
            return RecognitionResult(
                track_id=track_id,
                student_id=None,
                name=None,
                similarity=0.0,
                status='SMALL_FACE',
                face_bbox=best_face['bbox'],
                det_score=best_face['det_score']
            )
        
        # Match against enrolled students
        if best_face.get('embedding') is None:
            return RecognitionResult(
                track_id=track_id,
                student_id=None,
                name=None,
                similarity=0.0,
                status='NO_FACE',
                face_bbox=best_face['bbox'],
                det_score=best_face['det_score']
            )
        
        match = self.matcher.match(best_face['embedding'], self._enrolled)
        
        if match.is_known:
            return RecognitionResult(
                track_id=track_id,
                student_id=match.student_id,
                name=match.name,
                similarity=match.similarity,
                status='MATCHED',
                face_bbox=best_face['bbox'],
                det_score=best_face['det_score']
            )
        else:
            return RecognitionResult(
                track_id=track_id,
                student_id=None,
                name=None,
                similarity=match.similarity,
                status='UNKNOWN',
                face_bbox=best_face['bbox'],
                det_score=best_face['det_score']
            )
    
    @property
    def enrolled_count(self) -> int:
        """Number of enrolled students."""
        return len(self._enrolled)
