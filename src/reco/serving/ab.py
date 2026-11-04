"""
A/B Testing Deterministic Bucketing Module.
Routes users deterministically to experimental variants (e.g. Variant A vs Variant B)
using salted cryptographic MD5 hashing.
"""

from __future__ import annotations

import hashlib
import logging
from typing import Dict, Literal

logger = logging.getLogger(__name__)

VariantType = Literal["A", "B"]


def get_user_hash_ratio(user_id: int, salt: str = "recoengine_ab_2026") -> float:
    """
    Compute a deterministic float in [0.0, 1.0) for a given user ID and experiment salt.
    """
    key = f"{salt}:{user_id}".encode("utf-8")
    digest = hashlib.md5(key).hexdigest()
    # Use first 8 hex characters (32 bits)
    val = int(digest[:8], 16)
    return float(val / 0xFFFFFFFF)


def get_experiment_variant(
    user_id: int,
    salt: str = "recoengine_ab_2026",
    split_ratio: float = 0.50,
) -> VariantType:
    """
    Deterministically assign a user to Variant A or Variant B.

    Args:
        user_id: integer user identifier
        salt: experiment namespace salt
        split_ratio: proportion assigned to Variant A (default: 0.50 for 50/50 split)

    Returns:
        "A" (retrieval-only) or "B" (retrieval + DCN-V2 ranker)
    """
    ratio = get_user_hash_ratio(user_id=user_id, salt=salt)
    return "A" if ratio < split_ratio else "B"


def get_variant_description(variant: VariantType) -> Dict[str, str]:
    """Return variant configuration metadata."""
    if variant == "A":
        return {
            "variant": "A",
            "name": "Retrieval-Only Baseline",
            "description": "Two-tower retrieval scoring (dot-product / cosine similarity) without Stage 2 re-ranking.",
        }
    elif variant == "B":
        return {
            "variant": "B",
            "name": "Two-Stage Retrieval + DCN-V2 Ranker",
            "description": "Two-tower retrieval candidate generation + DCN-V2 Deep & Cross Network re-ranking.",
        }
    raise ValueError(f"Unknown variant: {variant}")

