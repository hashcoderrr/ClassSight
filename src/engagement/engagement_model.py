"""Engagement scoring models.

Provides a pluggable model interface for converting behavioral features
into engagement indicators. The default HeuristicEngagementModel uses
configurable weighted linear combination.

IMPORTANT: Engagement indicators are inferred behavioral heuristics,
NOT validated psychological measurements. All scores are explainable
and all weights are configurable.
"""

from abc import ABC, abstractmethod
from typing import Optional


class BaseEngagementModel(ABC):
    """Abstract base class for engagement models.
    
    Subclasses must implement calculate() which converts behavioral
    features into an engagement indicator.
    
    This interface allows the heuristic model to be replaced with
    an ML-based model in the future without changing the pipeline.
    """
    
    @abstractmethod
    def calculate(self, features: dict) -> dict:
        """Calculate engagement indicator from behavioral features.
        
        Args:
            features: Dict of behavioral features from FeatureExtractor.
        
        Returns:
            Dict with keys:
                'score': float (0-100)
                'level': str ('HIGH', 'MODERATE', 'LOW')
                'factors': list[dict] with contributing factor explanations
        """
        pass
    
    @abstractmethod
    def get_model_info(self) -> dict:
        """Return model metadata for report transparency."""
        pass


class HeuristicEngagementModel(BaseEngagementModel):
    """Weighted linear heuristic engagement model.
    
    Score = base_score + sum(feature_value * weight * 100)
    
    Clamped to [0, 100].
    
    This is a transparent, explainable baseline — NOT a scientifically
    validated engagement measurement instrument.
    """
    
    DEFAULT_WEIGHTS = {
        'forward_ratio': 0.40,
        'raised_hand_rate': 0.20,
        'phone_ratio': -0.20,
        'head_down_ratio': -0.10,
        'sleep_ratio': -0.10,
    }
    
    DEFAULT_THRESHOLDS = {
        'LOW': 0,
        'MODERATE': 40,
        'HIGH': 70,
    }
    
    def __init__(self,
                 weights: Optional[dict] = None,
                 base_score: float = 50.0,
                 level_thresholds: Optional[dict] = None,
                 raised_hand_rate_cap: float = 1.0):
        """Initialize the heuristic engagement model.
        
        Args:
            weights: Dict mapping feature name -> weight.
                Positive weights indicate positive engagement signal.
                Negative weights indicate potential disengagement signal.
            base_score: Starting score before feature contributions.
            level_thresholds: Dict mapping level name -> minimum score.
            raised_hand_rate_cap: Maximum effective raised_hand_rate
                value (per minute) to prevent outlier domination.
        """
        self.weights = dict(weights or self.DEFAULT_WEIGHTS)
        self.base_score = base_score
        self.level_thresholds = dict(
            level_thresholds or self.DEFAULT_THRESHOLDS
        )
        self.raised_hand_rate_cap = max(0.01, raised_hand_rate_cap)
    
    def calculate(self, features: dict) -> dict:
        """Calculate engagement indicator from behavioral features.
        
        Args:
            features: Dict from FeatureExtractor with ratio/rate values.
        
        Returns:
            Dict with 'score', 'level', and 'factors'.
        """
        score = self.base_score
        factors = []
        
        for feature_name, weight in self.weights.items():
            raw_value = features.get(feature_name, 0.0)
            
            # Cap raised_hand_rate to prevent outlier scores
            if feature_name == 'raised_hand_rate':
                effective_value = min(raw_value, self.raised_hand_rate_cap)
            else:
                effective_value = raw_value
            
            contribution = effective_value * weight * 100
            score += contribution
            
            # Generate explainable factor
            if abs(contribution) > 0.5:  # Only include meaningful factors
                factor = self._describe_factor(
                    feature_name, raw_value, weight, contribution
                )
                factors.append(factor)
        
        # Clamp score to [0, 100]
        score = max(0.0, min(100.0, score))
        score = round(score, 1)
        
        # Determine level
        level = self._classify_level(score)
        
        # Sort factors by absolute impact (largest first)
        factors.sort(key=lambda f: abs(f.get('contribution', 0)), reverse=True)
        
        return {
            'score': score,
            'level': level,
            'factors': factors,
        }
    
    def _classify_level(self, score: float) -> str:
        """Classify score into engagement level."""
        # Sort thresholds descending to find highest matching level
        sorted_levels = sorted(
            self.level_thresholds.items(),
            key=lambda x: x[1],
            reverse=True
        )
        for level_name, min_score in sorted_levels:
            if score >= min_score:
                return level_name
        return 'LOW'
    
    def _describe_factor(self, feature_name: str, raw_value: float,
                          weight: float, contribution: float) -> dict:
        """Generate a human-readable factor description."""
        descriptions = {
            'forward_ratio': 'Forward-facing duration',
            'raised_hand_rate': 'Participation events',
            'phone_ratio': 'Phone usage detected',
            'head_down_ratio': 'Head-down posture',
            'sleep_ratio': 'Possible sleeping indicator',
        }
        
        impact = 'high' if abs(contribution) > 10 else (
            'moderate' if abs(contribution) > 3 else 'low'
        )
        direction = '+' if contribution > 0 else '-'
        
        label = descriptions.get(feature_name, feature_name)
        
        # Add context to the description
        if feature_name.endswith('_ratio'):
            pct = round(raw_value * 100, 1)
            detail = f"{label} ({pct}%)"
        elif feature_name == 'raised_hand_rate':
            detail = f"{label} ({raw_value:.1f}/min)"
        else:
            detail = f"{label} ({raw_value:.2f})"
        
        return {
            'factor': detail,
            'direction': direction,
            'impact': impact,
            'contribution': round(contribution, 2),
        }
    
    def get_model_info(self) -> dict:
        """Return model metadata for transparency."""
        return {
            'type': 'HeuristicEngagementModel',
            'weights': dict(self.weights),
            'base_score': self.base_score,
            'score_range': [0, 100],
            'level_thresholds': dict(self.level_thresholds),
            'raised_hand_rate_cap': self.raised_hand_rate_cap,
            'disclaimer': (
                'Engagement indicators are inferred behavioral heuristics '
                'based on observable physical behaviors. They are NOT '
                'validated psychological measurements of attention, '
                'interest, or motivation. All weights are configurable '
                'engineering parameters, not scientifically calibrated values.'
            ),
        }
