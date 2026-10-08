"""Temporal identity confirmation and Track ID to Student ID mapping."""

from collections import deque
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class IdentityState:
    """Current identity state for a tracked person."""
    track_id: int
    confirmed_id: Optional[str] = None    # Student ID once confirmed
    confirmed_name: Optional[str] = None  # Student name once confirmed
    status: str = 'NO_FACE'               # CONFIRMED, PENDING, UNKNOWN, NO_FACE
    pending_id: Optional[str] = None      # Student ID being evaluated
    pending_name: Optional[str] = None    # Student name being evaluated
    consecutive_matches: int = 0          # Consecutive matches to pending_id
    best_similarity: float = 0.0          # Best similarity score seen
    last_similarity: float = 0.0          # Most recent similarity
    frames_since_face: int = 0            # Frames since face was last detected
    total_recognitions: int = 0           # Total recognition attempts
    total_matches: int = 0                # Total successful matches


class IdentityManager:
    """Manages temporal confirmation of face recognition results.
    
    Prevents false identity assignments by requiring multiple consecutive
    recognitions before confirming a Track ID → Student ID mapping.
    
    Key behaviors:
    - CONFIRMED: Only after N consecutive matches to same student
    - PENDING: Matches detected but not yet confirmed
    - UNKNOWN: Face detected but no match exceeds threshold
    - NO_FACE: No usable face detected in person region
    - Retains confirmed identity for M frames after face loss
    - Resets if different student is consistently matched
    
    Important:
    - Track IDs are NOT student identities
    - Face recognition provides identity EVIDENCE
    - This class manages the evidence accumulation
    """
    
    def __init__(self, confirmation_frames: int = 3,
                 retention_frames: int = 30):
        """Initialize the identity manager.
        
        Args:
            confirmation_frames: Consecutive matches needed to confirm identity
            retention_frames: Frames to retain confirmed identity after face loss
        """
        self.confirmation_frames = max(1, confirmation_frames)
        self.retention_frames = max(1, retention_frames)
        self._states: dict[int, IdentityState] = {}  # track_id -> IdentityState
    
    def _get_state(self, track_id: int) -> IdentityState:
        """Get or create identity state for a track ID."""
        if track_id not in self._states:
            self._states[track_id] = IdentityState(track_id=track_id)
        return self._states[track_id]
    
    def update(self, track_id: int, student_id: Optional[str],
               name: Optional[str], similarity: float,
               is_known: bool, has_face: bool):
        """Update identity state for a track based on recognition result.
        
        Args:
            track_id: ByteTrack track ID
            student_id: Matched student ID (None if unknown/no face)
            name: Matched student name (None if unknown/no face)
            similarity: Cosine similarity score
            is_known: Whether the face matched an enrolled student
            has_face: Whether a face was detected at all
        """
        state = self._get_state(track_id)
        state.total_recognitions += 1
        
        if not has_face:
            # No face detected - increment face loss counter
            state.frames_since_face += 1
            
            if state.confirmed_id is not None:
                # Keep confirmed identity during temporary face loss
                if state.frames_since_face > self.retention_frames:
                    # Too long without face - reset identity
                    state.status = 'NO_FACE'
                    # Don't clear confirmed_id - keep it as last known
                else:
                    state.status = 'CONFIRMED'  # Still confident
            else:
                state.status = 'NO_FACE'
            return
        
        # Face was detected
        state.frames_since_face = 0
        
        if not is_known:
            # Face detected but no match above threshold
            state.status = 'UNKNOWN'
            state.last_similarity = similarity
            
            # If we had a confirmed ID, keep it during temporary unknown matches
            # but don't reset consecutive counter
            if state.confirmed_id is not None:
                # Allow brief lapses - only reset after sustained non-matches
                state.consecutive_matches = max(0, state.consecutive_matches - 1)
                if state.consecutive_matches == 0:
                    # Sustained non-match: keep confirmed but flag uncertain
                    pass  # Let confirmed_id persist, status stays UNKNOWN
            else:
                state.pending_id = None
                state.pending_name = None
                state.consecutive_matches = 0
            return
        
        # Known face matched
        state.total_matches += 1
        state.last_similarity = similarity
        state.best_similarity = max(state.best_similarity, similarity)
        
        if state.confirmed_id is not None:
            if student_id == state.confirmed_id:
                # Reconfirming same identity - strengthen
                state.consecutive_matches = min(
                    state.consecutive_matches + 1,
                    self.confirmation_frames * 2  # Cap to prevent overflow
                )
                state.status = 'CONFIRMED'
            else:
                # Different identity detected - this is unusual
                # Start counting for the new identity
                if student_id == state.pending_id:
                    state.consecutive_matches += 1
                    if state.consecutive_matches >= self.confirmation_frames:
                        # New identity confirmed - update mapping
                        state.confirmed_id = student_id
                        state.confirmed_name = name
                        state.status = 'CONFIRMED'
                else:
                    # Start new pending candidate
                    state.pending_id = student_id
                    state.pending_name = name
                    state.consecutive_matches = 1
                    state.status = 'PENDING'
        else:
            # No confirmed identity yet
            if student_id == state.pending_id:
                state.consecutive_matches += 1
            else:
                state.pending_id = student_id
                state.pending_name = name
                state.consecutive_matches = 1
            
            if state.consecutive_matches >= self.confirmation_frames:
                state.confirmed_id = student_id
                state.confirmed_name = name
                state.status = 'CONFIRMED'
            else:
                state.status = 'PENDING'
    
    def update_no_recognition(self, track_id: int):
        """Update state when no recognition was run (between recognition intervals).
        
        Preserves current state without modifying confirmation counters.
        """
        state = self._get_state(track_id)
        if state.confirmed_id is not None:
            state.status = 'CONFIRMED'
        # Don't change anything else - just maintain current state
    
    def get_identity(self, track_id: int) -> IdentityState:
        """Get the current identity state for a track ID."""
        return self._get_state(track_id)
    
    def get_all_mappings(self) -> dict:
        """Get all confirmed Track ID → Student ID mappings.
        
        Returns:
            dict mapping track_id -> {
                'student_id': str,
                'name': str,
                'status': str,
                'best_similarity': float,
                'total_recognitions': int
            }
        """
        mappings = {}
        for tid, state in self._states.items():
            mappings[tid] = {
                'student_id': state.confirmed_id,
                'name': state.confirmed_name,
                'status': state.status,
                'best_similarity': state.best_similarity,
                'total_recognitions': state.total_recognitions,
                'total_matches': state.total_matches
            }
        return mappings
    
    def get_confirmed_count(self) -> int:
        """Number of track IDs with confirmed student identities."""
        return sum(1 for s in self._states.values() if s.confirmed_id is not None)
    
    def get_unknown_count(self) -> int:
        """Number of track IDs currently marked as UNKNOWN."""
        return sum(1 for s in self._states.values() 
                   if s.status == 'UNKNOWN' and s.confirmed_id is None)
    
    def get_summary(self) -> dict:
        """Get a summary of all identity states."""
        total = len(self._states)
        confirmed = self.get_confirmed_count()
        unknown = self.get_unknown_count()
        pending = sum(1 for s in self._states.values() if s.status == 'PENDING')
        no_face = sum(1 for s in self._states.values() 
                      if s.status == 'NO_FACE' and s.confirmed_id is None)
        
        return {
            'total_tracks': total,
            'confirmed': confirmed,
            'unknown': unknown,
            'pending': pending,
            'no_face': no_face
        }
    
    def reset(self):
        """Reset all identity states."""
        self._states.clear()
