"""Identity matching via face embedding comparison."""

import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class MatchResult:
    """Result of an identity match attempt."""
    student_id: Optional[str]  # None if unknown
    name: Optional[str]        # None if unknown  
    similarity: float          # Cosine similarity (0-1)
    is_known: bool             # True if match exceeds threshold


class IdentityMatcher:
    """Compares face embeddings against enrolled students.
    
    Uses cosine similarity between L2-normalized embeddings.
    InsightFace embeddings are already L2-normalized, so
    cosine similarity = dot product.
    """
    
    def __init__(self, threshold: float = 0.4):
        """Initialize the identity matcher.
        
        Args:
            threshold: Minimum cosine similarity to consider a match.
                       Typical range for InsightFace: 0.3-0.5
        """
        self.threshold = threshold
    
    def match(self, query_embedding: np.ndarray,
              enrolled: dict) -> MatchResult:
        """Match a query embedding against enrolled students.
        
        Args:
            query_embedding: 512-dim face embedding to identify
            enrolled: dict mapping student_id -> {'name': str, 'embedding': np.ndarray}
            
        Returns:
            MatchResult with best match or UNKNOWN
        """
        if len(enrolled) == 0 or query_embedding is None:
            return MatchResult(
                student_id=None,
                name=None,
                similarity=0.0,
                is_known=False
            )
        
        best_id = None
        best_name = None
        best_sim = -1.0
        
        # Ensure query is normalized
        query_norm = np.linalg.norm(query_embedding)
        if query_norm > 0:
            query_embedding = query_embedding / query_norm
        
        for sid, info in enrolled.items():
            emb = info['embedding']
            # Cosine similarity (both should be L2-normalized)
            sim = float(np.dot(query_embedding, emb))
            
            if sim > best_sim:
                best_sim = sim
                best_id = sid
                best_name = info['name']
        
        # Only return a match if it exceeds the threshold
        if best_sim >= self.threshold:
            return MatchResult(
                student_id=best_id,
                name=best_name,
                similarity=best_sim,
                is_known=True
            )
        else:
            return MatchResult(
                student_id=None,
                name=None,
                similarity=best_sim,
                is_known=False
            )
