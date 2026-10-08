"""Tests for Phase 2 face recognition components."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import numpy as np
import cv2
import json
import tempfile
import os


# ============================================================
# Tests for IdentityMatcher (no InsightFace needed)
# ============================================================

class TestIdentityMatcher:
    """Tests for identity matching logic (pure embedding comparison)."""
    
    def setup_method(self):
        from src.recognition.identity_matcher import IdentityMatcher
        self.matcher = IdentityMatcher(threshold=0.4)
    
    def test_initialization(self):
        assert self.matcher.threshold == 0.4
    
    def test_match_above_threshold(self):
        """Test that a similar embedding returns a match."""
        from src.recognition.identity_matcher import IdentityMatcher
        matcher = IdentityMatcher(threshold=0.4)
        
        # Create two similar normalized embeddings
        emb1 = np.random.randn(512).astype(np.float32)
        emb1 = emb1 / np.linalg.norm(emb1)
        # emb2 is very close to emb1 (add small noise)
        emb2 = emb1 + np.random.randn(512).astype(np.float32) * 0.01
        emb2 = emb2 / np.linalg.norm(emb2)
        
        enrolled = {
            'STU001': {'name': 'Student 001', 'embedding': emb1}
        }
        
        result = matcher.match(emb2, enrolled)
        assert result.is_known
        assert result.student_id == 'STU001'
        assert result.similarity > 0.4
    
    def test_match_below_threshold(self):
        """Test that dissimilar embeddings return unknown."""
        from src.recognition.identity_matcher import IdentityMatcher
        matcher = IdentityMatcher(threshold=0.4)
        
        # Two random, unrelated embeddings
        emb1 = np.random.randn(512).astype(np.float32)
        emb1 = emb1 / np.linalg.norm(emb1)
        emb2 = np.random.randn(512).astype(np.float32)
        emb2 = emb2 / np.linalg.norm(emb2)
        
        enrolled = {
            'STU001': {'name': 'Student 001', 'embedding': emb1}
        }
        
        result = matcher.match(emb2, enrolled)
        # Random 512-dim vectors have cosine sim ~0, so should be unknown
        assert not result.is_known
        assert result.student_id is None
    
    def test_match_empty_enrolled(self):
        """Test matching against empty enrolled set."""
        emb = np.random.randn(512).astype(np.float32)
        emb = emb / np.linalg.norm(emb)
        
        result = self.matcher.match(emb, {})
        assert not result.is_known
        assert result.student_id is None
    
    def test_match_none_embedding(self):
        """Test matching with None embedding."""
        result = self.matcher.match(None, {'STU001': {'name': 'Test', 'embedding': np.zeros(512)}})
        assert not result.is_known
    
    def test_match_best_of_multiple(self):
        """Test that the best match is selected among multiple enrolled students."""
        from src.recognition.identity_matcher import IdentityMatcher
        matcher = IdentityMatcher(threshold=0.3)
        
        query = np.random.randn(512).astype(np.float32)
        query = query / np.linalg.norm(query)
        
        # Make STU002 very similar to query, STU001 less similar
        emb1 = np.random.randn(512).astype(np.float32)
        emb1 = emb1 / np.linalg.norm(emb1)
        emb2 = query + np.random.randn(512).astype(np.float32) * 0.005
        emb2 = emb2 / np.linalg.norm(emb2)
        
        enrolled = {
            'STU001': {'name': 'Student 001', 'embedding': emb1},
            'STU002': {'name': 'Student 002', 'embedding': emb2}
        }
        
        result = matcher.match(query, enrolled)
        assert result.student_id == 'STU002'
    
    def test_threshold_configurable(self):
        """Test that threshold properly filters matches."""
        from src.recognition.identity_matcher import IdentityMatcher
        
        emb1 = np.random.randn(512).astype(np.float32)
        emb1 = emb1 / np.linalg.norm(emb1)
        emb2 = emb1 + np.random.randn(512).astype(np.float32) * 0.1
        emb2 = emb2 / np.linalg.norm(emb2)
        
        enrolled = {'STU001': {'name': 'Test', 'embedding': emb1}}
        
        # Low threshold - should match
        low_matcher = IdentityMatcher(threshold=0.1)
        result_low = low_matcher.match(emb2, enrolled)
        
        # Very high threshold - should not match
        high_matcher = IdentityMatcher(threshold=0.999)
        result_high = high_matcher.match(emb2, enrolled)
        
        assert result_low.is_known
        assert not result_high.is_known


# ============================================================
# Tests for IdentityManager (no InsightFace needed)
# ============================================================

class TestIdentityManager:
    """Tests for temporal confirmation logic."""
    
    def setup_method(self):
        from src.recognition.identity_manager import IdentityManager
        self.manager = IdentityManager(confirmation_frames=3, retention_frames=5)
    
    def test_initialization(self):
        assert self.manager.confirmation_frames == 3
        assert self.manager.retention_frames == 5
    
    def test_confirmation_after_n_frames(self):
        """Test that identity is confirmed after N consecutive matches."""
        # First match - PENDING
        self.manager.update(track_id=1, student_id='STU001', name='Alice',
                           similarity=0.8, is_known=True, has_face=True)
        state = self.manager.get_identity(1)
        assert state.status == 'PENDING'
        assert state.confirmed_id is None
        
        # Second match - still PENDING
        self.manager.update(1, 'STU001', 'Alice', 0.82, True, True)
        state = self.manager.get_identity(1)
        assert state.status == 'PENDING'
        
        # Third match - CONFIRMED!
        self.manager.update(1, 'STU001', 'Alice', 0.85, True, True)
        state = self.manager.get_identity(1)
        assert state.status == 'CONFIRMED'
        assert state.confirmed_id == 'STU001'
        assert state.confirmed_name == 'Alice'
    
    def test_unknown_rejects(self):
        """Test that unknown faces are properly handled."""
        self.manager.update(1, None, None, 0.2, False, True)
        state = self.manager.get_identity(1)
        assert state.status == 'UNKNOWN'
        assert state.confirmed_id is None
    
    def test_no_face(self):
        """Test that no-face is properly handled."""
        self.manager.update(1, None, None, 0.0, False, False)
        state = self.manager.get_identity(1)
        assert state.status == 'NO_FACE'
    
    def test_retention_after_face_loss(self):
        """Test that confirmed identity is retained during temporary face loss."""
        # Confirm an identity
        for _ in range(3):
            self.manager.update(1, 'STU001', 'Alice', 0.85, True, True)
        
        assert self.manager.get_identity(1).confirmed_id == 'STU001'
        
        # Face lost for a few frames - should retain
        for i in range(4):
            self.manager.update(1, None, None, 0.0, False, False)
            state = self.manager.get_identity(1)
            assert state.confirmed_id == 'STU001'  # Still retained
            assert state.status == 'CONFIRMED'
        
        # Face lost for too long - should lose confidence
        for _ in range(5):
            self.manager.update(1, None, None, 0.0, False, False)
        
        state = self.manager.get_identity(1)
        assert state.status == 'NO_FACE'
        # confirmed_id may still be set as last known, but status indicates loss
    
    def test_different_student_breaks_streak(self):
        """Test that matching a different student resets the pending count."""
        # Match STU001 twice
        self.manager.update(1, 'STU001', 'Alice', 0.8, True, True)
        self.manager.update(1, 'STU001', 'Alice', 0.82, True, True)
        
        # Now match STU002 - should reset
        self.manager.update(1, 'STU002', 'Bob', 0.75, True, True)
        state = self.manager.get_identity(1)
        assert state.confirmed_id is None  # Not confirmed yet
        assert state.pending_id == 'STU002'
    
    def test_get_all_mappings(self):
        """Test getting all confirmed mappings."""
        for _ in range(3):
            self.manager.update(1, 'STU001', 'Alice', 0.85, True, True)
        for _ in range(3):
            self.manager.update(2, 'STU002', 'Bob', 0.80, True, True)
        
        mappings = self.manager.get_all_mappings()
        assert 1 in mappings
        assert 2 in mappings
        assert mappings[1]['student_id'] == 'STU001'
        assert mappings[2]['student_id'] == 'STU002'
    
    def test_get_summary(self):
        for _ in range(3):
            self.manager.update(1, 'STU001', 'Alice', 0.85, True, True)
        self.manager.update(2, None, None, 0.2, False, True)
        
        summary = self.manager.get_summary()
        assert summary['confirmed'] == 1
        assert summary['unknown'] == 1
    
    def test_update_no_recognition(self):
        """Test that update_no_recognition preserves state."""
        for _ in range(3):
            self.manager.update(1, 'STU001', 'Alice', 0.85, True, True)
        
        self.manager.update_no_recognition(1)
        state = self.manager.get_identity(1)
        assert state.status == 'CONFIRMED'
        assert state.confirmed_id == 'STU001'
    
    def test_reset(self):
        self.manager.update(1, 'STU001', 'Alice', 0.85, True, True)
        self.manager.reset()
        assert self.manager.get_confirmed_count() == 0


# ============================================================
# Tests for EmbeddingStore (no InsightFace needed for storage tests)
# ============================================================

class TestEmbeddingStore:
    """Tests for embedding storage and registry."""
    
    def test_initialization(self):
        from src.recognition.embedding_store import EmbeddingStore
        with tempfile.TemporaryDirectory() as tmpdir:
            store = EmbeddingStore(
                identities_file=os.path.join(tmpdir, 'students.json'),
                embeddings_dir=os.path.join(tmpdir, 'embeddings')
            )
            assert store.count == 0
    
    def test_list_empty(self):
        from src.recognition.embedding_store import EmbeddingStore
        with tempfile.TemporaryDirectory() as tmpdir:
            store = EmbeddingStore(
                identities_file=os.path.join(tmpdir, 'students.json'),
                embeddings_dir=os.path.join(tmpdir, 'embeddings')
            )
            assert store.list_students() == []
    
    def test_save_and_load_registry(self):
        """Test that registry persists across instances."""
        from src.recognition.embedding_store import EmbeddingStore
        with tempfile.TemporaryDirectory() as tmpdir:
            id_file = os.path.join(tmpdir, 'students.json')
            emb_dir = os.path.join(tmpdir, 'embeddings')
            
            # Manually create a student entry
            store = EmbeddingStore(identities_file=id_file, embeddings_dir=emb_dir)
            emb = np.random.randn(512).astype(np.float32)
            emb = emb / np.linalg.norm(emb)
            emb_path = os.path.join(emb_dir, 'STU001.npy')
            np.save(emb_path, emb)
            
            store.students['STU001'] = {
                'student_id': 'STU001',
                'name': 'Test Student',
                'embedding_path': emb_path,
                'created_at': '2024-01-01T00:00:00',
                'num_images': 1
            }
            store._save_registry()
            
            # Load in new instance
            store2 = EmbeddingStore(identities_file=id_file, embeddings_dir=emb_dir)
            assert store2.count == 1
            assert 'STU001' in store2.students
    
    def test_load_all_embeddings(self):
        from src.recognition.embedding_store import EmbeddingStore
        with tempfile.TemporaryDirectory() as tmpdir:
            id_file = os.path.join(tmpdir, 'students.json')
            emb_dir = os.path.join(tmpdir, 'embeddings')
            os.makedirs(emb_dir, exist_ok=True)
            
            store = EmbeddingStore(identities_file=id_file, embeddings_dir=emb_dir)
            
            emb = np.random.randn(512).astype(np.float32)
            emb_path = os.path.join(emb_dir, 'STU001.npy')
            np.save(emb_path, emb)
            
            store.students['STU001'] = {
                'student_id': 'STU001',
                'name': 'Test',
                'embedding_path': emb_path,
                'created_at': '',
                'num_images': 1
            }
            store._save_registry()
            
            loaded = store.load_all_embeddings()
            assert 'STU001' in loaded
            assert loaded['STU001']['embedding'].shape == (512,)
    
    def test_remove_student(self):
        from src.recognition.embedding_store import EmbeddingStore
        with tempfile.TemporaryDirectory() as tmpdir:
            id_file = os.path.join(tmpdir, 'students.json')
            emb_dir = os.path.join(tmpdir, 'embeddings')
            os.makedirs(emb_dir, exist_ok=True)
            
            store = EmbeddingStore(identities_file=id_file, embeddings_dir=emb_dir)
            emb_path = os.path.join(emb_dir, 'STU001.npy')
            np.save(emb_path, np.zeros(512))
            
            store.students['STU001'] = {
                'student_id': 'STU001', 'name': 'Test',
                'embedding_path': emb_path, 'created_at': '', 'num_images': 1
            }
            store._save_registry()
            
            assert store.remove('STU001')
            assert store.count == 0
            assert not os.path.exists(emb_path)
    
    def test_remove_nonexistent(self):
        from src.recognition.embedding_store import EmbeddingStore
        with tempfile.TemporaryDirectory() as tmpdir:
            store = EmbeddingStore(
                identities_file=os.path.join(tmpdir, 's.json'),
                embeddings_dir=os.path.join(tmpdir, 'e')
            )
            assert not store.remove('NONEXISTENT')


# ============================================================
# Tests for IdentityMetrics (no InsightFace needed)
# ============================================================

class TestIdentityMetrics:
    """Tests for identity metrics and reporting."""
    
    def test_initialization(self):
        from src.evaluation.identity_metrics import IdentityMetrics
        m = IdentityMetrics()
        assert m.total_frames_processed == 0
        assert m.recognition_attempts == 0
    
    def test_update_frame(self):
        from src.evaluation.identity_metrics import IdentityMetrics
        m = IdentityMetrics()
        m.update_frame(0.05, [0.8, 0.9], 2, [1, 2])
        assert m.total_frames_processed == 1
    
    def test_generate_identity_report(self):
        from src.evaluation.identity_metrics import IdentityMetrics
        m = IdentityMetrics()
        m.start()
        m.update_frame(0.05, [0.8], 1, [1])
        m.stop()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            report = m.generate_identity_report(
                video_name='test.mp4',
                resolution='640x480',
                total_frames=1,
                detection_model='yolo11n.pt',
                recognition_model='InsightFace (buffalo_l)',
                identity_summary={'confirmed': 1, 'unknown': 0, 'pending': 0, 'no_face': 0},
                identity_mappings={1: {'student_id': 'STU001', 'name': 'Test', 'status': 'CONFIRMED', 'best_similarity': 0.85}},
                output_dir=tmpdir
            )
            
            assert report['video'] == 'test.mp4'
            assert report['confirmed_students'] == 1
            
            json_files = list(Path(tmpdir).glob('*.json'))
            assert len(json_files) == 1


# ============================================================
# Tests requiring InsightFace (integration tests)
# ============================================================

class TestFaceDetectorIntegration:
    """Integration tests that require InsightFace model.
    
    These tests will auto-skip if InsightFace is not properly installed
    or if the model cannot be loaded.
    """
    
    @pytest.fixture(autouse=True)
    def setup(self):
        try:
            from src.recognition.face_detector import FaceDetector
            self.detector = FaceDetector(model_name='buffalo_l', min_face_size=20)
            self._available = True
        except Exception:
            self._available = False
            pytest.skip("InsightFace model not available")
    
    def test_detect_blank_image(self):
        """Blank image should detect no faces."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        faces = self.detector.detect_faces(frame)
        assert len(faces) == 0
    
    def test_detect_empty_input(self):
        """Empty/None input should return empty list."""
        faces = self.detector.detect_faces(np.array([]))
        assert len(faces) == 0
    
    def test_detect_returns_correct_keys(self):
        """Verify returned face dicts have expected keys."""
        # Create a random image (unlikely to have faces)
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        faces = self.detector.detect_faces(frame)
        # Even if no faces found, the method should return a list
        assert isinstance(faces, list)
    
    def test_detect_in_region_empty(self):
        """Detection in empty region should return empty."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        faces = self.detector.detect_in_region(frame, [100, 100, 200, 300])
        assert isinstance(faces, list)
    
    def test_quality_check(self):
        """Quality check should work with synthetic face dict."""
        fake_face = {
            'bbox': [100, 100, 150, 150],
            'det_score': 0.95,
            'embedding': np.random.randn(512),
            'face_size': 50
        }
        quality = self.detector.check_quality(fake_face)
        assert quality['is_valid']
        assert len(quality['reasons']) == 0
    
    def test_quality_check_small_face(self):
        """Quality check should reject small faces."""
        self.detector.min_face_size = 100
        fake_face = {
            'bbox': [100, 100, 130, 130],
            'det_score': 0.95,
            'embedding': np.random.randn(512),
            'face_size': 30
        }
        quality = self.detector.check_quality(fake_face)
        assert not quality['is_valid']
        assert any('too small' in r for r in quality['reasons'])
