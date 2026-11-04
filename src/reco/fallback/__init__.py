"""Cold-start fallbacks: time-decayed popularity and content-based recommendation."""

from src.reco.fallback.content_based import ContentBasedRecommender
from src.reco.fallback.popularity import TimeDecayedPopularity

__all__ = ["TimeDecayedPopularity", "ContentBasedRecommender"]

