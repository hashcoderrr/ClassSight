"""Student enrollment and face embedding storage."""

import json
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Optional


class EmbeddingStore:
    """Manages enrolled student face embeddings.
    
    Handles student enrollment, embedding storage/retrieval,
    and the identity registry (students.json).
    
    Directory structure:
        data/identities/
            students.json          # Identity registry
            embeddings/
                STU001.npy         # Embedding for student STU001
                STU002.npy         # etc.
    """
    
    def __init__(self, identities_file: str | Path,
                 embeddings_dir: str | Path):
        """Initialize the embedding store.
        
        Args:
            identities_file: Path to the students.json registry
            embeddings_dir: Path to the embeddings directory
        """
        self.identities_file = Path(identities_file)
        self.embeddings_dir = Path(embeddings_dir)
        self.students: dict = {}  # student_id -> {name, embedding_path, created_at, num_images}
        self._ensure_dirs()
        self._load_registry()
    
    def _ensure_dirs(self):
        """Create necessary directories."""
        self.identities_file.parent.mkdir(parents=True, exist_ok=True)
        self.embeddings_dir.mkdir(parents=True, exist_ok=True)
    
    def _load_registry(self):
        """Load the student registry from JSON."""
        if self.identities_file.exists():
            try:
                with open(self.identities_file, 'r') as f:
                    data = json.load(f)
                self.students = {s['student_id']: s for s in data.get('students', [])}
                print(f"[INFO] Loaded {len(self.students)} enrolled students")
            except (json.JSONDecodeError, KeyError) as e:
                print(f"[WARN] Could not load registry: {e}")
                self.students = {}
        else:
            self.students = {}
    
    def _save_registry(self):
        """Save the student registry to JSON."""
        data = {
            'version': '1.0',
            'updated_at': datetime.now().isoformat(),
            'students': list(self.students.values())
        }
        with open(self.identities_file, 'w') as f:
            json.dump(data, f, indent=4)
    
    def enroll(self, student_id: str, name: str, 
              images: list[Path],
              face_detector) -> dict:
        """Enroll a student with face images.
        
        For each image:
        1. Detect face
        2. Validate quality (single face, good size, not blurry)
        3. Extract embedding
        
        Then average all valid embeddings to create a representative embedding.
        
        Args:
            student_id: Unique student identifier (e.g., 'STU001')
            name: Student display name
            images: List of paths to face images
            face_detector: FaceDetector instance to use for detection
            
        Returns:
            dict with enrollment result:
                - 'success': bool
                - 'student_id': str
                - 'name': str  
                - 'num_images_processed': int
                - 'num_valid_embeddings': int
                - 'rejected_images': list of {path, reason}
                - 'message': str
        """
        import cv2
        
        if not images:
            return {
                'success': False,
                'student_id': student_id,
                'name': name,
                'num_images_processed': 0,
                'num_valid_embeddings': 0,
                'rejected_images': [],
                'message': 'No images provided'
            }
        
        valid_embeddings = []
        rejected = []
        
        for img_path in images:
            img_path = Path(img_path)
            if not img_path.exists():
                rejected.append({'path': str(img_path), 'reason': 'File not found'})
                continue
            
            image = cv2.imread(str(img_path))
            if image is None:
                rejected.append({'path': str(img_path), 'reason': 'Cannot read image'})
                continue
            
            faces = face_detector.detect_faces(image)
            
            if len(faces) == 0:
                rejected.append({'path': str(img_path), 'reason': 'No face detected'})
                continue
            
            if len(faces) > 1:
                rejected.append({'path': str(img_path), 
                               'reason': f'Multiple faces detected ({len(faces)})'})
                continue
            
            face = faces[0]
            quality = face_detector.check_image_quality(image, face)
            
            if not quality['is_valid']:
                rejected.append({
                    'path': str(img_path),
                    'reason': '; '.join(quality['reasons'])
                })
                continue
            
            if face.get('embedding') is not None:
                valid_embeddings.append(face['embedding'])
        
        if not valid_embeddings:
            return {
                'success': False,
                'student_id': student_id,
                'name': name,
                'num_images_processed': len(images),
                'num_valid_embeddings': 0,
                'rejected_images': rejected,
                'message': 'No valid face embeddings could be extracted'
            }
        
        # Average embeddings and normalize
        avg_embedding = np.mean(valid_embeddings, axis=0)
        avg_embedding = avg_embedding / np.linalg.norm(avg_embedding)
        
        # Save embedding
        embedding_path = self.embeddings_dir / f"{student_id}.npy"
        np.save(str(embedding_path), avg_embedding)
        
        # Update registry
        self.students[student_id] = {
            'student_id': student_id,
            'name': name,
            'embedding_path': str(embedding_path),
            'created_at': datetime.now().isoformat(),
            'num_images': len(valid_embeddings)
        }
        self._save_registry()
        
        return {
            'success': True,
            'student_id': student_id,
            'name': name,
            'num_images_processed': len(images),
            'num_valid_embeddings': len(valid_embeddings),
            'rejected_images': rejected,
            'message': f'Successfully enrolled {name} ({student_id}) with {len(valid_embeddings)} face images'
        }
    
    def load_all_embeddings(self) -> dict:
        """Load all enrolled student embeddings.
        
        Returns:
            dict mapping student_id -> {'name': str, 'embedding': np.ndarray}
        """
        enrolled = {}
        for sid, info in self.students.items():
            emb_path = Path(info['embedding_path'])
            if emb_path.exists():
                embedding = np.load(str(emb_path))
                enrolled[sid] = {
                    'name': info['name'],
                    'embedding': embedding
                }
            else:
                print(f"[WARN] Embedding file missing for {sid}: {emb_path}")
        return enrolled
    
    def get_embedding(self, student_id: str) -> Optional[np.ndarray]:
        """Get the stored embedding for a student."""
        if student_id not in self.students:
            return None
        emb_path = Path(self.students[student_id]['embedding_path'])
        if emb_path.exists():
            return np.load(str(emb_path))
        return None
    
    def remove(self, student_id: str) -> bool:
        """Remove an enrolled student."""
        if student_id not in self.students:
            return False
        
        emb_path = Path(self.students[student_id]['embedding_path'])
        if emb_path.exists():
            emb_path.unlink()
        
        del self.students[student_id]
        self._save_registry()
        return True
    
    def list_students(self) -> list[dict]:
        """List all enrolled students."""
        return [
            {
                'student_id': info['student_id'],
                'name': info['name'],
                'num_images': info.get('num_images', 0),
                'created_at': info.get('created_at', '')
            }
            for info in self.students.values()
        ]
    
    @property
    def count(self) -> int:
        """Number of enrolled students."""
        return len(self.students)
