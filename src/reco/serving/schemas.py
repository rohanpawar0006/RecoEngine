"""
Pydantic data schemas for FastAPI recommendation service.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class RecommendationRequest(BaseModel):
    user_id: int = Field(..., ge=0, description="Unique user identifier (non-negative integer)")
    k: int = Field(default=10, ge=1, le=100, description="Number of recommendations to return (1-100)")
    variant: Optional[str] = Field(default="v1", max_length=32, description="Model variant or experiment arm")
    seed_genres: Optional[List[str]] = Field(default=None, max_length=10, description="Optional seed genres for cold-start onboarding")


class RecommendationItem(BaseModel):
    item_id: int = Field(..., description="MovieLens unique movie identifier")
    title: str = Field(..., description="Movie title")
    genres: str = Field(..., description="Pipe-delimited genre string")
    score: float = Field(..., description="Model ranking score or similarity score")
    rank: int = Field(..., ge=1, description="1-based rank position in recommendation list")


class RecommendationResponse(BaseModel):
    user_id: int
    variant: str
    model_version: str
    source: str = Field(..., description="Recommendation source: 'cache', 'model', 'fallback_popularity', 'fallback_content'")
    cache_hit: bool
    latency_ms: float
    items: List[RecommendationItem]


class EventRequest(BaseModel):
    user_id: int = Field(..., ge=0)
    item_id: int = Field(..., ge=0)
    variant: str = Field(default="v1", max_length=32)
    event: Literal["impression", "click", "rating"] = Field(..., description="Event type")
    ts: Optional[float] = Field(default_factory=time.time, description="Unix timestamp of event")
    model_version: Optional[str] = Field(default="v1", max_length=32)
    rating_value: Optional[float] = Field(default=None, ge=0.5, le=5.0, description="Optional rating if event is 'rating'")


class EventResponse(BaseModel):
    status: str
    logged: bool


class ABSummaryResponse(BaseModel):
    total_events: int
    variants: Dict[str, Dict[str, Any]]


class HealthResponse(BaseModel):
    status: str
    version: str
    redis_available: bool
    triton_available: bool
    served_by: str
    faiss_loaded: bool
    catalog_size: int

