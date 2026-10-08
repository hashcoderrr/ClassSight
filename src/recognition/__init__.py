"""ClassSight AI - Face Recognition & Student Identity Mapping (Phase 2)."""

from .face_detector import FaceDetector
from .embedding_store import EmbeddingStore
from .identity_matcher import IdentityMatcher
from .face_recognizer import FaceRecognizer
from .identity_manager import IdentityManager

__all__ = [
    'FaceDetector',
    'EmbeddingStore',
    'IdentityMatcher',
    'FaceRecognizer',
    'IdentityManager',
]
